"""
The match itself (main3.py section 9).

`Match` holds what used to be ~40 locals inside `run_game()` and splits the frame
into the same five phases the original ran in order:

    events -> talk to the brain -> AI decides & flies -> physics -> render

Order and arithmetic are unchanged; only the packaging is.
"""

import math
import threading

import pygame

from .config import (BALLISTIC_COLOR, FLARE_COLOR, FPS, GUIDANCE_HOMING, HEIGHT,
                     HUD_HEIGHT, JET_COLOR, MISSILE_COLOR, WIDTH)
from .display import clock, screen
from .sim.decision import AIDecision
from .sim.entities import (BallisticMissile, ExplosionParticle, HomingMissile,
                           Jet, JetMissile, Launcher, Shockwave)
from .sim.inference import (decision_queue, drain_decisions,
                            gnn_inference_worker, state_queue, stop_worker)
from .sim.maneuvers import ManeuverController
from .sim.planner import correct
from .sim.tactics import apply_cluster_repulsion, compute_wall_state
from .ui.effects import draw_radar_bg, draw_scanlines
from .ui.hud import draw_hud
from .ui.screens import run_gameover_screen

# The strategic plan feeds the danger read-out, the maneuver head and the GRU
# memory. None of those need to be fast, and the plan was arriving ~16 frames
# stale anyway while consuming the CPU the per-frame controller needs. Slowing
# it down makes the jet *more* responsive, not less.
GNN_INTERVAL       = 20     # frames between world snapshots sent to the brain
FLARE_LIFETIME_MS  = 2000   # mirrors FlareParticle.is_dead, for the world model
CLUSTER_COOLDOWN   = 360    # 6 s between cluster launches
JET_ATTACK_PERIOD  = 4000   # ms between the jet's counter-attacks
GAMEOVER_HOLD_MS   = 2200   # how long the wreck is shown before the result screen

# Cluster-swarm launch profile: one entry per missile in the volley.
CLUSTER_OFFSETS     = [-20, -10,  0,  10,  20]
CLUSTER_SPEEDS      = [3.5, 3.7, 3.9, 4.1, 4.3]
CLUSTER_TURN_RATES  = [0.06, 0.05, 0.04, 0.03, 0.025]
# Each missile gets a gentle persistent curve to hold formation spread
CLUSTER_LAT_BIASES  = [-0.014, -0.007, 0.0, 0.007, 0.014]


def reset_game():
    """Fresh jet, empty entity lists and two live launchers."""
    jet  = Jet(WIDTH//2, HEIGHT//2)
    missiles, ai_missiles, flares, explosions, shockwaves = [], [], [], [], []
    launcher_homing    = Launcher(100, HEIGHT-HUD_HEIGHT, MISSILE_COLOR)
    launcher_ballistic = Launcher(WIDTH-100, HEIGHT-HUD_HEIGHT, BALLISTIC_COLOR)
    return (jet, missiles, ai_missiles, flares, explosions, shockwaves,
            launcher_homing, launcher_ballistic)


class Match:
    """One round of jet-versus-launchers."""

    def __init__(self):
        (self.jet, self.missiles, self.ai_missiles, self.flares,
         self.explosions, self.shockwaves, self.lh, self.lb) = reset_game()

        self.glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        self.frame_count    = 0
        self.game_over      = False
        self.user_won       = False
        self.game_over_time = 0
        self.radar_angle    = 0.0

        self.show_gnn_overlay = False
        self.cluster_cooldown = 0

        self.controller = ManeuverController()
        self.decision   = AIDecision()
        self.plan_age   = 0      # frames between a snapshot and its advice landing
        # Whatever style of action won this frame. Not a state the jet is locked
        # into -- it is re-chosen every frame and can change every frame.
        self.active_maneuver = "NORMAL"
        self._stop      = threading.Event()   # tells this match's worker to exit

        self.stats = {'fired': 0, 'flares': 0, 'maneuvers': 0}

    # ── 1. input ──────────────────────────────────────────────
    def handle_events(self):
        """Return a routing string to leave the match, or None to keep playing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._shutdown_worker()
                return "quit"
            if self.game_over or event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_h and self.lh.can_fire():
                self.lh.fire()
                self.missiles.append(HomingMissile(self.lh.x, self.lh.y, self.jet,
                                                   self.jet.x, self.jet.y))
                self.stats['fired'] += 1
            elif event.key == pygame.K_b and self.lb.can_fire():
                self.lb.fire()
                self.missiles.append(BallisticMissile(self.lb.x, self.lb.y,
                                                      self.jet.x, self.jet.y))
                self.stats['fired'] += 1
            elif event.key == pygame.K_c and self.cluster_cooldown == 0 and self.lh.alive:
                self._launch_cluster()
            elif event.key == pygame.K_ESCAPE:
                self._shutdown_worker()
                return "menu"
            elif event.key == pygame.K_g:
                self.show_gnn_overlay = not self.show_gnn_overlay
        return None

    def _launch_cluster(self):
        """Five staggered homing missiles fanned out around the jet bearing."""
        self.cluster_cooldown = CLUSTER_COOLDOWN
        base_a = math.atan2(self.jet.y-self.lh.y, self.jet.x-self.lh.x)
        profile = zip(CLUSTER_OFFSETS, CLUSTER_SPEEDS,
                      CLUSTER_TURN_RATES, CLUSTER_LAT_BIASES)
        for i, (off, spd, tr, lat_bias) in enumerate(profile):
            ra  = math.radians(off)
            tx2 = self.lh.x + math.cos(base_a+ra)*120
            ty2 = self.lh.y + math.sin(base_a+ra)*120
            m   = HomingMissile(self.lh.x, self.lh.y, self.jet, tx2, ty2)
            m.speed = spd; m.max_speed = spd
            m.turn_rate = tr; m.spawn_delay = i*3
            m.lateral_bias = lat_bias
            self.missiles.append(m)
        self.stats['fired'] += 5

    # ── 2. brain I/O ──────────────────────────────────────────
    def exchange_with_brain(self):
        """Push a world snapshot every GNN_INTERVAL frames; pick up any answer."""
        if self.frame_count % GNN_INTERVAL == 0 and state_queue.empty():
            now = pygame.time.get_ticks()
            state_queue.put({
                'jet':      {'pos': [self.jet.x, self.jet.y],
                             'vel': [self.jet.vx, self.jet.vy],
                             'heading': self.jet.heading,
                             'speed': self.jet.speed},
                # Full kinematics, not just position and velocity: the world
                # model needs each missile's guidance parameters to predict where
                # it will actually go.
                'missiles': [{'pos': [m.x, m.y], 'vel': [m.vx, m.vy],
                              'heading': m.heading, 'speed': m.speed,
                              'turn_rate': getattr(m, 'turn_rate', 0.0),
                              'lateral_bias': getattr(m, 'lateral_bias', 0.0),
                              'spawn_delay': getattr(m, 'spawn_delay', 0),
                              'radius': m.radius,
                              'mtype': getattr(m, 'mtype', GUIDANCE_HOMING)}
                             for m in self.missiles],
                'flares':   [{'pos': [f.x, f.y], 'vel': [f.vx, f.vy],
                              'life': max(0, (FLARE_LIFETIME_MS - (now - f.spawn_time)) // 17)}
                             for f in self.flares],
                'want_overlay': self.show_gnn_overlay,
                'frame': self.frame_count,
                'plan_age': self.plan_age,
            })

        if not decision_queue.empty():
            self.decision = decision_queue.get()
            self.plan_age = max(0, self.frame_count - self.decision.stamp)

    # ── 3. AI ─────────────────────────────────────────────────
    def update_ai(self):
        """Run the FSM: cooldowns, transition, then fly the chosen maneuver."""
        self.controller.tick_cooldowns()
        self._correct_plan()
        wall = compute_wall_state(self.jet)
        self.controller.select(self.jet, self.missiles, wall,
                               self.decision.maneuver, self.stats, self.decision)
        self.controller.execute(self.jet, self.missiles, self.flares, wall,
                                self.decision, self.stats)
        self._counter_attack()

    def _correct_plan(self):
        """
        Re-check the strategic plan against where things actually are.

        The plan comes off a background thread and is several frames old; a
        missile moves 5-7 px per frame against a 13 px kill radius, so acting on
        it unchecked is how the jet flies into things. This is cheap enough to
        run every frame.
        """
        d = self.decision
        if not self.missiles:
            d.flight_heading, d.flight_throttle = d.heading, d.throttle
            self.active_maneuver = "NORMAL"
            return
        # Run from the first frame, with or without a strategic plan to lean on.
        seed_h = d.heading if d.fresh else self.jet.heading
        seed_t = d.throttle if d.fresh else self.jet.max_speed
        jet = {'pos': [self.jet.x, self.jet.y], 'vel': [self.jet.vx, self.jet.vy],
               'heading': self.jet.heading, 'speed': self.jet.speed}
        ms = [{'pos': [m.x, m.y], 'vel': [m.vx, m.vy], 'heading': m.heading,
               'speed': m.speed, 'turn_rate': getattr(m, 'turn_rate', 0.0),
               'lateral_bias': getattr(m, 'lateral_bias', 0.0),
               'spawn_delay': getattr(m, 'spawn_delay', 0), 'radius': m.radius,
               'mtype': getattr(m, 'mtype', GUIDANCE_HOMING)} for m in self.missiles]
        d.flight_heading, d.flight_throttle, self.active_maneuver = correct(
            jet, ms, seed_h, seed_t, preferred=d.maneuver if d.fresh else None)

    def _counter_attack(self):
        """Every 4 s the jet shoots back at the nearest surviving launcher."""
        if pygame.time.get_ticks()-self.jet.last_attack_time > JET_ATTACK_PERIOD:
            targets = [l for l in (self.lh, self.lb) if l.alive]
            if targets:
                # Prefer the launcher that is closer so the missile is harder to dodge
                tgt = min(targets, key=lambda l: math.hypot(l.x-self.jet.x, l.y-self.jet.y))
                self.ai_missiles.append(JetMissile(self.jet.x, self.jet.y, tgt.x, tgt.y))
            # always reset so the timer stays rhythmic
            self.jet.last_attack_time = pygame.time.get_ticks()

    # ── 4. physics & collisions ───────────────────────────────
    def _end_match(self, user_won):
        self.game_over      = True
        self.user_won       = user_won
        self.game_over_time = pygame.time.get_ticks()
        self._shutdown_worker()

    def _shutdown_worker(self):
        """Signal this match's inference thread to stop, without ever blocking."""
        self._stop.set()
        stop_worker()

    def _blast(self, x, y, color, count, shock_color=None):
        for _ in range(count):
            self.explosions.append(ExplosionParticle(x, y, color))
        self.shockwaves.append(Shockwave(x, y, shock_color or color))

    def update_physics(self):
        self.jet.update()

        # Jet wall collision (lethal)
        if (self.jet.x < self.jet.radius or self.jet.x > WIDTH-self.jet.radius or
                self.jet.y < self.jet.radius or self.jet.y > HEIGHT-HUD_HEIGHT-self.jet.radius):
            self._blast(self.jet.x, self.jet.y, JET_COLOR, 50)
            self._end_match(user_won=True)

        self._update_flares()
        self._update_explosions()
        self._update_ai_missiles()
        self._update_missiles()

        # Homing missiles push each other apart so they never merge/overlap
        apply_cluster_repulsion(self.missiles)

        # AI win: both launchers dead
        if not self.lh.alive and not self.lb.alive:
            self._end_match(user_won=False)

    def _update_flares(self):
        alive = []
        for f in self.flares:
            f.update()
            if not f.is_dead():
                alive.append(f)
        self.flares = alive

    def _update_explosions(self):
        # main3.py stepped these twice per frame -- a leftover filter expression
        # plus the loop -- so explosions faded at double speed. Stepped once now.
        alive = []
        for e in self.explosions:
            e.update()
            if not e.is_dead():
                alive.append(e)
        self.explosions = alive

    def _update_ai_missiles(self):
        alive = []
        for aim in self.ai_missiles:
            aim.update()
            for lnc in (self.lh, self.lb):
                if lnc.alive and math.hypot(lnc.x-aim.x, lnc.y-aim.y) < 15+aim.radius:
                    lnc.alive = False; aim.dead = True
                    self._blast(lnc.x, lnc.y, (255,80,50), 30)
                    break
            if not aim.is_dead():
                alive.append(aim)
        self.ai_missiles = alive

    def _update_missiles(self):
        alive = []
        for m in self.missiles:
            m.update()

            # Jet hit
            if math.hypot(self.jet.x-m.x, self.jet.y-m.y) < self.jet.radius+m.radius:
                self._end_match(user_won=True)
                self._blast(self.jet.x, self.jet.y, JET_COLOR, 50)
                continue

            # Flare hit — the decoy burns out with the missile
            hit_flare = False
            for f in self.flares:
                if math.hypot(f.x-m.x, f.y-m.y) < f.radius+m.radius:
                    f.spawn_time -= 5000; m.dead = True; hit_flare = True
                    self._blast(f.x, f.y, FLARE_COLOR, 5)
                    break
            if hit_flare:
                continue

            # Missile-missile intercept
            hit_aim = False
            for aim in self.ai_missiles:
                if math.hypot(aim.x-m.x, aim.y-m.y) < aim.radius+m.radius:
                    m.dead = True; aim.dead = True; hit_aim = True
                    self._blast(aim.x, aim.y, (255,255,255), 20)
                    break
            if hit_aim:
                continue

            if not m.is_dead():
                alive.append(m)
        self.missiles = alive

    # ── 5. render ─────────────────────────────────────────────
    def render(self):
        self.radar_angle = (self.radar_angle + 0.025) % (math.pi*2)
        draw_radar_bg(screen, self.radar_angle, self.glow_surface)
        screen.blit(self.glow_surface, (0,0))

        self.lh.draw(screen); self.lb.draw(screen)

        if not self.game_over:
            self.jet.draw(self.glow_surface, self.active_maneuver)
        for f in self.flares:        f.draw(self.glow_surface)
        for m in self.missiles:      m.draw(self.glow_surface)
        for aim in self.ai_missiles: aim.draw(self.glow_surface)
        for e in self.explosions:    e.draw(screen)
        for s in self.shockwaves:
            s.update(); s.draw(self.glow_surface)
        self.shockwaves[:] = [s for s in self.shockwaves if not s.is_dead()]
        screen.blit(self.glow_surface, (0,0))

        draw_hud(screen, self.jet, self.active_maneuver, self.decision,
                 self.show_gnn_overlay)

    # ── frame ─────────────────────────────────────────────────
    def run(self):
        """Play until the match ends; returns the next screen to show."""
        gnn_thread = threading.Thread(target=gnn_inference_worker,
                                      args=(self._stop,), daemon=True)
        gnn_thread.start()

        while True:
            self.frame_count += 1
            if self.cluster_cooldown > 0:
                self.cluster_cooldown -= 1

            route = self.handle_events()
            if route is not None:
                return route

            if not self.game_over:
                self.exchange_with_brain()
                self.update_ai()
                self.update_physics()

            self.render()

            if self.game_over:
                ov = pygame.Surface((WIDTH, HEIGHT-HUD_HEIGHT), pygame.SRCALPHA)
                ov.fill((0,0,0,120)); screen.blit(ov,(0,0))
                if pygame.time.get_ticks()-self.game_over_time > GAMEOVER_HOLD_MS:
                    self._shutdown_worker()
                    drain_decisions()   # so the next match starts fresh
                    return run_gameover_screen(self.user_won, self.stats)

            draw_scanlines(screen)
            pygame.display.flip()
            clock.tick(FPS)


def run_game():
    """Compatibility wrapper — same entry point name main3.py used."""
    return Match().run()
