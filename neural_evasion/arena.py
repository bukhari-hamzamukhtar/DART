"""
The arena: one match's world, independent of who is flying the jet.

Frame-based and pilot-agnostic. A pilot (v1, v2 or v3 -- see `pilots`) is asked
for its controls once per frame, then the arena advances the physics. The same
arena runs the windowed game, the headless tests, the training-data generator
and the benchmarks, so all of them see exactly the same world.

One frame:

  begin_frame()   frame counter, launch cooldowns
  <commands>      the player or a scripted commander launches missiles
  advance(pilot)
    1. pilot      the jet's brain sets heading and throttle, may eject flares --
                  paid for from the flare magazine, or they never leave the jet
    2. counter-attack every JET_ATTACK_FRAMES
    3. jet        moves; walls and terrain are lethal
    4. flares     drift and burn out
    5. jet missiles
    6. missiles   seekers sense, datalinks share locks, then each missile moves
                  in its flight style, steering around peaks, and is checked
                  against the jet, flares, jet missiles and terrain
    7. repulsion  classic homing missiles push apart (unchanged from v1/v2)

Two rules make the missile side harder and the jet's flares scarce:

  * every guided missile gets a flight style at launch (sim/doctrine.py), drawn
    from the arena's own random generator -- seed it for a reproducible match;
  * the flare magazine: a life starts with no flares, and each cluster the enemy
    launches gives the jet one volley. A frame in which a pilot ejects flares
    costs one charge, whatever the volley holds; with none left, the flares
    never leave the jet.

`sim.worldmodel_v3` mirrors steps 3, 4, 6 and 7, and `tests/test_worldmodel_v3.py`
holds the two together.
"""

import math
import random

from .config import (CLUSTER_COOLDOWN_FRAMES, FLARE_CHARGES_PER_CLUSTER,
                     FLARE_COLOR, FLARE_LIFE_FRAMES, FLARE_START_CHARGES, HEIGHT,
                     HUD_HEIGHT, JET_ATTACK_FRAMES, JET_COLOR,
                     PINCER_COOLDOWN_FRAMES, WIDTH)
from .levels import LEVEL_1
from .sim.doctrine import (ERRATIC, FLANKER, LINK_RANGE, PINCER_SPEED,
                           PINCER_TURN_RATE, PURSUIT, WEAVER, sample_style)
from .sim.entities import (BallisticMissile, ExplosionParticle, HomingMissile,
                           Jet, JetMissile, Launcher, Obstacle, Shockwave,
                           SmartMissile, apply_style)
from .sim.geometry import inside_any, segment_blocked
from .sim.tactics import apply_cluster_repulsion

# Cluster-swarm launch profile, one entry per missile. The fan is twice v1/v2's
# +-20 degrees and every member flies its own style, so a swarm spreads out
# instead of chasing the jet in one clump.
CLUSTER_OFFSETS    = [-40, -20, 0, 20, 40]
CLUSTER_SPEEDS     = [3.5, 3.7, 3.9, 4.1, 4.3]
CLUSTER_TURN_RATES = [0.06, 0.05, 0.04, 0.03, 0.025]
CLUSTER_LAT_BIASES = [-0.014, -0.007, 0.0, 0.007, 0.014]
CLUSTER_STYLES     = [FLANKER, WEAVER, PURSUIT, ERRATIC, FLANKER]   # shuffled per launch

PINCER_FAN = 0.35          # rad; pincer members leave the rail splayed outward


class Arena(object):
    """Everything that exists in one match, and the rules that move it."""

    def __init__(self, level=LEVEL_1, jet_invulnerable=False,
                 launchers_invulnerable=False, counter_attack=True, visuals=True,
                 seed=None, magazine=True):
        self.level = level
        self.frame = 0
        self.jet = Jet(WIDTH // 2, HEIGHT // 2)
        self.missiles = []
        self.ai_missiles = []
        self.flares = []
        self.explosions = []
        self.shockwaves = []
        self.terrain = [tuple(o) for o in level.obstacles]
        self.obstacles = [Obstacle(*o) for o in self.terrain]
        self.launchers = [Launcher(st.x, st.y, st.color) for st in level.stations]

        self.jet_invulnerable = jet_invulnerable
        self.launchers_invulnerable = launchers_invulnerable
        self.counter_attack = counter_attack
        self.visuals = visuals
        # Flight styles are drawn from here; a seed makes a match reproducible.
        self.rng = random.Random(seed)

        # The flare magazine. magazine=False lifts the limit (training data).
        self.magazine = magazine
        self.flare_charges = FLARE_START_CHARGES
        self._flare_paid = None       # frame whose volley is already paid for

        self.cluster_cooldowns = [0] * len(self.launchers)
        self.pincer_cooldown = 0
        self.last_attack_frame = 0
        self._next_network = 1

        self.game_over = False
        self.user_won = False
        self.cause = None           # "missile" | "wall" | "terrain" | "bases"
        self.killer_uid = None
        self.end_frame = None
        self.stats = {'fired': 0, 'flares': 0, 'maneuvers': 0, 'absorbed': 0,
                      'volleys': 0, 'flares_denied': 0, 'charges_earned': 0}

    # ── launching ─────────────────────────────────────────────
    def _new_network(self):
        n = self._next_network
        self._next_network += 1
        return n

    def _add_missile(self, m):
        m.born_frame = self.frame
        self.missiles.append(m)

    def station_ready(self, i):
        return 0 <= i < len(self.launchers) and self.launchers[i].ready(self.frame)

    def fire(self, i, speed=None, turn_rate=None, lead=None, style=None):
        """
        Fire one missile from station `i`. A guided missile flies `style` (one of
        doctrine.STYLES), or a random style for this level if None. Returns False
        if the station cannot fire.
        """
        if not self.station_ready(i):
            return False
        st = self.level.stations[i]
        self.launchers[i].fire_at(self.frame)
        jet = self.jet
        if st.kind == "ballistic":
            m = BallisticMissile(st.x, st.y, jet.x, jet.y)
            if speed is not None:
                m.speed = speed
                m.vx = math.cos(m.heading) * speed
                m.vy = math.sin(m.heading) * speed
        else:
            if self.level.smart:
                kw = {}
                if speed is not None:
                    kw['speed'] = speed
                if turn_rate is not None:
                    kw['turn_rate'] = turn_rate
                m = SmartMissile(st.x, st.y, jet, jet.x, jet.y,
                                 lead=self.level.lead if lead is None else lead,
                                 network=self._new_network() if self.level.datalink else None,
                                 **kw)
            else:
                m = HomingMissile(st.x, st.y, jet, jet.x, jet.y)
                if speed is not None:
                    m.speed = m.max_speed = speed
                    m.vx = math.cos(m.heading) * speed
                    m.vy = math.sin(m.heading) * speed
                if turn_rate is not None:
                    m.turn_rate = turn_rate
            apply_style(m, sample_style(self.rng, self.level.number, style))
        self._add_missile(m)
        self.stats['fired'] += 1
        return True

    def cluster_ready(self, i=None):
        """Can station `i` launch a cluster now? With i=None: can any station?"""
        if i is None:
            return any(self.cluster_ready(k) for k in range(len(self.launchers)))
        return (0 <= i < len(self.launchers) and self.level.stations[i].kind == "homing"
                and self.launchers[i].alive and self.cluster_cooldowns[i] == 0)

    def launch_cluster(self, i=None, speeds=None, turn_rates=None, lateral_biases=None):
        """
        Five staggered guided missiles from homing station `i` (the first ready
        one if None), fanned out around the jet's bearing, each in its own style.
        Every cluster earns the jet one flare volley.
        """
        if i is None:
            i = next((k for k in range(len(self.launchers)) if self.cluster_ready(k)), None)
        if i is None or not self.cluster_ready(i):
            return False
        self.cluster_cooldowns[i] = CLUSTER_COOLDOWN_FRAMES
        st, jet = self.level.stations[i], self.jet
        base_a = math.atan2(jet.y - st.y, jet.x - st.x)
        speeds = speeds or CLUSTER_SPEEDS
        turn_rates = turn_rates or CLUSTER_TURN_RATES
        biases = lateral_biases or CLUSTER_LAT_BIASES
        net = self._new_network() if self.level.datalink else None
        styles = list(CLUSTER_STYLES)
        self.rng.shuffle(styles)
        flank_sides = [-1.0, 1.0]            # the swarm's two flankers go opposite ways
        self.rng.shuffle(flank_sides)
        for k, off in enumerate(CLUSTER_OFFSETS):
            ra = math.radians(off)
            tx2 = st.x + math.cos(base_a + ra) * 120
            ty2 = st.y + math.sin(base_a + ra) * 120
            if self.level.smart:
                m = SmartMissile(st.x, st.y, jet, tx2, ty2, speed=speeds[k],
                                 turn_rate=turn_rates[k], lead=self.level.lead,
                                 network=net, lateral_bias=biases[k],
                                 spawn_delay=k * 3)
                m.last_known = (jet.x, jet.y)   # the fan is the launch direction, the cue is the jet
            else:
                m = HomingMissile(st.x, st.y, jet, tx2, ty2)
                m.speed = speeds[k]
                m.max_speed = speeds[k]
                m.turn_rate = turn_rates[k]
                m.spawn_delay = k * 3
                m.lateral_bias = biases[k]
            p = sample_style(self.rng, self.level.number, styles[k])
            if p['style'] == FLANKER and flank_sides:
                p['side'] = flank_sides.pop()
            apply_style(m, p)
            self._add_missile(m)
        self.stats['fired'] += 5
        self.flare_charges += FLARE_CHARGES_PER_CLUSTER
        self.stats['charges_earned'] += FLARE_CHARGES_PER_CLUSTER
        return True

    def pincer_ready(self):
        return (self.level.pincer and self.pincer_cooldown == 0 and
                any(l.alive for l in self.launchers))

    def launch_pincer(self, speed=None, turn_rate=None):
        """Two missiles from every live battery on one datalink, split left and right."""
        if not self.pincer_ready():
            return False
        self.pincer_cooldown = PINCER_COOLDOWN_FRAMES
        jet = self.jet
        net = self._new_network()
        n = 0
        for i, st in enumerate(self.level.stations):
            if not self.launchers[i].alive:
                continue
            bearing = math.atan2(jet.y - st.y, jet.x - st.x)
            for j, side in enumerate((-1.0, 1.0)):
                m = SmartMissile(st.x, st.y, jet, jet.x, jet.y,
                                 speed=PINCER_SPEED if speed is None else speed,
                                 turn_rate=PINCER_TURN_RATE if turn_rate is None else turn_rate,
                                 lead=self.level.lead, side=side, network=net,
                                 spawn_delay=j * 4,
                                 launch_heading=bearing - side * PINCER_FAN)
                self._add_missile(m)
                n += 1
        self.stats['fired'] += n
        return True

    # ── the flare magazine ────────────────────────────────────
    def can_flare(self):
        """Is there a volley in the magazine?"""
        return (not self.magazine) or self.flare_charges > 0

    def spend_flare(self):
        """A pilot pays for this frame's volley. False if the magazine is empty."""
        return (not self.magazine) or self._pay_flares()

    def _pay_flares(self):
        if self._flare_paid == self.frame:
            return True
        if self.flare_charges <= 0:
            return False
        self.flare_charges -= 1
        self._flare_paid = self.frame
        self.stats['volleys'] += 1
        return True

    # ── sensing ───────────────────────────────────────────────
    def line_of_sight(self, ax, ay, bx, by):
        return not (self.terrain and segment_blocked(ax, ay, bx, by, self.terrain))

    def radar_picture(self):
        """What the jet's radar can see: (missiles, flares). Terrain blocks it."""
        if not self.terrain:
            return list(self.missiles), list(self.flares)
        jx, jy, t = self.jet.x, self.jet.y, self.terrain
        return ([m for m in self.missiles if not segment_blocked(jx, jy, m.x, m.y, t)],
                [f for f in self.flares if not segment_blocked(jx, jy, f.x, f.y, t)])

    # ── the frame ─────────────────────────────────────────────
    def begin_frame(self):
        self.frame += 1
        cd = self.cluster_cooldowns
        for k in range(len(cd)):
            if cd[k] > 0:
                cd[k] -= 1
        if self.pincer_cooldown > 0:
            self.pincer_cooldown -= 1

    def advance(self, pilot=None):
        if pilot is not None:
            pilot.act(self)
        fresh = [f for f in self.flares if getattr(f, "born_frame", None) is None]
        if fresh:
            if self.magazine and not self._pay_flares():
                # Empty magazine: the flares never leave the jet.
                gone = set(id(f) for f in fresh)
                self.flares = [f for f in self.flares if id(f) not in gone]
                self.stats['flares'] -= len(fresh)
                self.stats['flares_denied'] += len(fresh)
            else:
                for f in fresh:                # flares the pilot just ejected
                    f.born_frame = self.frame
                    f.dead = False
        self._counter_attack()
        self._physics()

    def step(self, pilot=None, commander=None):
        self.begin_frame()
        if commander is not None:
            commander.act(self)
        self.advance(pilot)

    # ── internals ─────────────────────────────────────────────
    def _counter_attack(self):
        if not self.counter_attack or self.game_over:
            return
        if self.frame - self.last_attack_frame > JET_ATTACK_FRAMES:
            targets = [l for l in self.launchers if l.alive]
            if targets:
                jet = self.jet
                clear = [l for l in targets if self.line_of_sight(jet.x, jet.y, l.x, l.y)]
                tgt = min(clear or targets, key=lambda l: math.hypot(l.x - jet.x, l.y - jet.y))
                jm = JetMissile(jet.x, jet.y, tgt.x, tgt.y)
                jm.born_frame = self.frame
                self.ai_missiles.append(jm)
            self.last_attack_frame = self.frame

    def _blast(self, x, y, color, count):
        if not self.visuals:
            return
        for _ in range(count):
            self.explosions.append(ExplosionParticle(x, y, color))
        self.shockwaves.append(Shockwave(x, y, color))

    def _end(self, user_won, cause, killer=None):
        if self.game_over:
            return
        self.game_over = True
        self.user_won = user_won
        self.cause = cause
        self.killer_uid = killer
        self.end_frame = self.frame

    def _physics(self):
        jet = self.jet
        jet.update()
        r = jet.radius
        if (jet.x < r or jet.x > WIDTH - r or jet.y < r or jet.y > HEIGHT - HUD_HEIGHT - r):
            if self.jet_invulnerable:
                self._bounce_jet()
            elif not self.game_over:
                self._blast(jet.x, jet.y, JET_COLOR, 50)
                self._end(True, "wall")
        elif self.terrain and inside_any(jet.x, jet.y, self.terrain, r):
            if self.jet_invulnerable:
                self._push_out_of_terrain()
            elif not self.game_over:
                self._blast(jet.x, jet.y, JET_COLOR, 50)
                self._end(True, "terrain")

        self._update_flares()
        self._update_explosions()
        self._update_ai_missiles()
        self._update_missiles()
        apply_cluster_repulsion(self.missiles)

        if (not self.launchers_invulnerable and self.launchers
                and not any(l.alive for l in self.launchers)):
            self._end(False, "bases")

    def _bounce_jet(self):
        jet, r = self.jet, self.jet.radius
        if jet.x < r or jet.x > WIDTH - r:
            jet.vx = -jet.vx
            jet.heading = math.pi - jet.heading
        if jet.y < r or jet.y > HEIGHT - HUD_HEIGHT - r:
            jet.vy = -jet.vy
            jet.heading = -jet.heading
        jet.x = min(max(jet.x, r + 1.0), WIDTH - r - 1.0)
        jet.y = min(max(jet.y, r + 1.0), HEIGHT - HUD_HEIGHT - r - 1.0)

    def _push_out_of_terrain(self):
        jet = self.jet
        for ox, oy, rad in self.terrain:
            d = math.hypot(jet.x - ox, jet.y - oy)
            lim = rad + jet.radius + 1.0
            if d < lim:
                nx, ny = ((jet.x - ox) / d, (jet.y - oy) / d) if d > 1e-6 else (1.0, 0.0)
                jet.x, jet.y = ox + nx * lim, oy + ny * lim
                dot = jet.vx * nx + jet.vy * ny
                if dot < 0:
                    jet.vx -= 2 * dot * nx
                    jet.vy -= 2 * dot * ny
                jet.heading = math.atan2(jet.vy, jet.vx)

    def _update_flares(self):
        alive = []
        for f in self.flares:
            f.update()
            if getattr(f, "dead", False) or self.frame - f.born_frame > FLARE_LIFE_FRAMES:
                continue
            alive.append(f)
        self.flares = alive

    def _update_explosions(self):
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
            for lnc in self.launchers:
                if lnc.alive and math.hypot(lnc.x - aim.x, lnc.y - aim.y) < 15 + aim.radius:
                    if not self.launchers_invulnerable:
                        lnc.alive = False
                    aim.dead = True
                    self._blast(lnc.x, lnc.y, (255, 80, 50), 30)
                    break
            if not aim.dead and self.terrain and inside_any(aim.x, aim.y, self.terrain, aim.radius):
                aim.dead = True
                self._blast(aim.x, aim.y, (120, 160, 140), 8)
            if not aim.dead and self.frame - aim.born_frame <= aim.life_frames:
                alive.append(aim)
        self.ai_missiles = alive

    def _share_locks(self, smart):
        """Datalink: a missile with no lock of its own guides on a salvo partner's."""
        lockers = [m for m in smart if m.lock and m.network is not None]
        if not lockers:
            return
        for m in smart:
            if m.lock or m.network is None or m.spawn_delay > 0:
                continue
            for o in lockers:
                if (o is not m and o.network == m.network
                        and math.hypot(o.x - m.x, o.y - m.y) < LINK_RANGE):
                    m.linked = True
                    break

    def _update_missiles(self):
        jet = self.jet
        smart = [m for m in self.missiles if isinstance(m, SmartMissile)]
        if smart:
            for m in smart:
                m.sense(jet, self.flares, self.terrain)
            if self.level.datalink:
                self._share_locks(smart)

        alive = []
        for m in self.missiles:
            m.update()

            if math.hypot(jet.x - m.x, jet.y - m.y) < jet.radius + m.radius:
                if self.jet_invulnerable:
                    self.stats['absorbed'] += 1
                else:
                    self._blast(jet.x, jet.y, JET_COLOR, 50)
                    self._end(True, "missile", m.uid)
                continue

            hit_flare = False
            for f in self.flares:
                if math.hypot(f.x - m.x, f.y - m.y) < f.radius + m.radius:
                    f.dead = True
                    hit_flare = True
                    self._blast(f.x, f.y, FLARE_COLOR, 5)
                    break
            if hit_flare:
                continue

            hit_aim = False
            for aim in self.ai_missiles:
                if math.hypot(aim.x - m.x, aim.y - m.y) < aim.radius + m.radius:
                    aim.dead = True
                    hit_aim = True
                    self._blast(aim.x, aim.y, (255, 255, 255), 20)
                    break
            if hit_aim:
                continue

            if self.terrain and inside_any(m.x, m.y, self.terrain, m.radius):
                self._blast(m.x, m.y, (160, 150, 120), 10)
                continue

            if m.dead or self.frame - m.born_frame > m.life_frames:
                continue
            alive.append(m)
        self.missiles = alive
