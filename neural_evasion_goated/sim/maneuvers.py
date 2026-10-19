"""
The evasion finite-state machine (main3.py section 9, FSM + execution blocks).

The GNN proposes a maneuver each tick; this controller decides whether the
geometry actually allows it, then flies it out over a fixed number of frames.

Priority: IMMELMANN > COBRA > FALLING_LEAF > JINKING > BARREL_ROLL.

All thresholds, timers and steering gains are the values from main3.py, just
named instead of inlined.
"""

import math

import pygame

from ..utils import normalize_angle
from .entities import FlareParticle
from .tactics import heading_to_center, predict_intercept_flare, spawn_flares

# ── Entry gates: minimum clearance from the walls, per maneuver ──
CLEARANCE = {
    "COBRA":        140,
    "FALLING_LEAF": 100,
    "JINKING":       80,
    "BARREL_ROLL":  150,
}

# ── Durations (frames) ────────────────────────────────────────
DURATION = {
    "IMMELMANN":     35,
    "COBRA":         40,
    "FALLING_LEAF":  90,
    "JINKING":      120,
    "BARREL_ROLL":   72,
}

COBRA_COOLDOWN          = 360   # 6 s between cobras
COBRA_TRIGGER_DIST      = 100   # GNN-recommended cobra needs a missile this close
COBRA_SAFETY_NET_DIST   =  80   # ...or this close even when the GNN says NORMAL
COBRA_RECOVER_COOLDOWN  =  60
JINK_RECOVER_COOLDOWN   =  45

IMMELMANN_SPEED_MULT = 1.5
IMMELMANN_TURN_GAIN  = 0.12
COBRA_PITCH_GAIN     = 0.22
COBRA_FLEE_GAIN      = 0.25
COBRA_BOOST_MULT     = 1.4
FALLING_LEAF_SPEED   = 0.35
FALLING_LEAF_WOBBLE  = 0.35
JINK_SPEED_MULT      = 0.51
BARREL_ROLL_TURN     = 0.15
NORMAL_STEER_GAIN    = 0.18

FLARE_REARM_MS   = 2000   # min gap between NORMAL-state flare volleys
FLARE_MIN_WALL   = 80     # don't dump flares while hugging a wall

# Short-range reflex. A plan is typically 6-15 frames old by the time it is
# applied, and a missile covers 5-7 px per frame, so inside this range the plan
# can be pointing at where the threat used to be. This runs every frame on live
# positions and covers that gap.
#
# An earlier ablation called it a no-op -- that harness replanned synchronously
# with zero latency, so there was no gap for it to cover. It is not a no-op in
# the game.
REFLEX_RANGE  = 90.0
REFLEX_WEIGHT = 0.45


class ManeuverController:
    """Owns the maneuver state that used to live in `run_game()` locals."""

    def __init__(self):
        self.maneuver         = "NORMAL"
        self.timer            = 0
        self.cobra_cooldown   = 0
        self.general_cooldown = 0
        self.memory           = {}

    # ── per-frame bookkeeping ─────────────────────────────────
    def tick_cooldowns(self):
        if self.cobra_cooldown   > 0: self.cobra_cooldown   -= 1
        if self.general_cooldown > 0: self.general_cooldown -= 1

    def _enter(self, name, timer, stats):
        self.maneuver = name
        self.timer    = timer
        stats['maneuvers'] += 1

    def _enter_cobra(self, jet, stats):
        self._enter("COBRA", DURATION["COBRA"], stats)
        self.cobra_cooldown = COBRA_COOLDOWN
        self.memory['start_speed'] = jet.speed

    # ── transitions ───────────────────────────────────────────
    def select(self, jet, missiles, wall, recommended, stats, decision=None):
        """
        Pick the next maneuver.

        A scripted maneuver hijacks the jet for 35-120 frames, so it now only
        engages when it can beat the planner:

          * no fresh plan yet -- the hand-written logic is all there is;
          * the plan says the jet is cornered, i.e. no heading avoids an impact,
            so the flare-dumping maneuvers are the remaining card to play.

        Letting these fire on ordinary frames measurably costs survival: they
        override a planner that has already simulated every alternative.
        """
        planning = decision is not None and decision.fresh
        cornered = planning and getattr(decision, 'cornered', False)
        wx, wy = wall.wx, wall.wy

        # 1. IMMELMANN — wall escape, only when the planner is not steering.
        #    Letting it fire when the jet is cornered was tried and is much
        #    worse: it seizes the controls for 35 frames and commits to a wall
        #    heading while ignoring missiles, which took brutal-pressure
        #    survival from 1/8 (median 620 frames) to 0/8 (median 250, every
        #    death a missile). A scripted maneuver cannot outperform a planner
        #    that re-decides 60 times a second, even in the case it was written
        #    for.
        if wall.heading_to_wall and self.maneuver != "IMMELMANN" and not planning:
            self._enter("IMMELMANN", DURATION["IMMELMANN"], stats)
            jet.heading = wall.escape_heading
            return

        if self.maneuver != "NORMAL" or self.general_cooldown != 0:
            return
        if planning and not cornered:
            return

        near = [(m, math.hypot(m.x-jet.x, m.y-jet.y)) for m in missiles]
        rec  = recommended   # GNN already computed this

        if rec == "COBRA" and self.cobra_cooldown == 0 and \
                wx > CLEARANCE["COBRA"] and wy > CLEARANCE["COBRA"]:
            # Confirm head-on before committing to COBRA
            if [m for m, d in near if d < COBRA_TRIGGER_DIST]:
                self._enter_cobra(jet, stats)

        elif rec == "FALLING_LEAF" and wx > CLEARANCE["FALLING_LEAF"] \
                and wy > CLEARANCE["FALLING_LEAF"] and len(missiles) >= 1:
            self._enter("FALLING_LEAF", DURATION["FALLING_LEAF"], stats)

        elif rec == "JINKING" and wx > CLEARANCE["JINKING"] \
                and wy > CLEARANCE["JINKING"] and len(missiles) >= 1:
            self._enter("JINKING", DURATION["JINKING"], stats)

        elif rec == "BARREL_ROLL" and wx > CLEARANCE["BARREL_ROLL"] \
                and wy > CLEARANCE["BARREL_ROLL"] and len(missiles) == 1:
            self._enter("BARREL_ROLL", DURATION["BARREL_ROLL"], stats)

        # Geometry safety net: if GNN says NORMAL but a missile is <80px, force COBRA
        elif rec == "NORMAL" and self.cobra_cooldown == 0 and \
                wx > CLEARANCE["COBRA"] and wy > CLEARANCE["COBRA"]:
            if [m for m, d in near if d < COBRA_SAFETY_NET_DIST]:
                self._enter_cobra(jet, stats)

    # ── execution ─────────────────────────────────────────────
    def execute(self, jet, missiles, flares, wall, decision, stats):
        """Fly one frame of the current maneuver."""
        handler = getattr(self, "_fly_" + self.maneuver.lower(), None)
        if handler is not None:
            handler(jet, missiles, flares, wall, decision, stats)

    def _fly_immelmann(self, jet, missiles, flares, wall, decision, stats):
        jet.set_throttle(jet.max_speed * IMMELMANN_SPEED_MULT)
        diff = normalize_angle(wall.escape_heading - jet.heading)
        jet.heading += diff * IMMELMANN_TURN_GAIN
        self.timer -= 1
        if self.timer <= 0:
            jet.heading   = heading_to_center(jet)
            self.maneuver = "NORMAL"

    def _fly_cobra(self, jet, missiles, flares, wall, decision, stats):
        if self.timer > 20:
            # Phase 1: hard decelerate + pitch perpendicular to missile approach
            jet.set_throttle(max(0.3, jet.speed - self.memory['start_speed'] / 20))
            if missiles:
                tm = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                # Two perpendicular options to missile velocity vector
                perp1 = math.atan2(-tm.vx, tm.vy)
                perp2 = math.atan2(tm.vx, -tm.vy)
                # Pick the one that moves the jet farther from the missile
                d1 = math.hypot(jet.x+math.cos(perp1)*8 - tm.x,
                                jet.y+math.sin(perp1)*8 - tm.y)
                d2 = math.hypot(jet.x+math.cos(perp2)*8 - tm.x,
                                jet.y+math.sin(perp2)*8 - tm.y)
                target_h = perp1 if d1 > d2 else perp2
                diff = normalize_angle(target_h - jet.heading)
                jet.heading += diff * COBRA_PITCH_GAIN   # fast pitch to perpendicular
        elif self.timer == 20:
            # Phase 2: peak decel — deploy full ring of intercept flares
            for miss in missiles:
                angle, speed = predict_intercept_flare(jet, miss)
                for spread in [-0.25, 0.0, 0.25]:
                    flares.append(FlareParticle(jet.x, jet.y, angle+spread,
                                                speed * (1.0 - abs(spread)*0.2)))
                    stats['flares'] += 1
            # 360 degree emergency ring too
            for i in range(8):
                flares.append(FlareParticle(jet.x, jet.y,
                                            i*(2*math.pi/8), 5.0))
                stats['flares'] += 1
        else:
            # Phase 3: re-accelerate, flee direction away from missile
            if missiles:
                tm = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                flee_h = math.atan2(jet.y-tm.y, jet.x-tm.x)
                diff   = normalize_angle(flee_h - jet.heading)
                jet.heading += diff * COBRA_FLEE_GAIN
            jet.set_throttle(min(jet.max_speed*COBRA_BOOST_MULT,
                                 jet.speed + jet.max_speed*COBRA_BOOST_MULT/20))
        self.timer -= 1
        if self.timer <= 0:
            self.maneuver = "NORMAL"
            self.general_cooldown = COBRA_RECOVER_COOLDOWN

    def _fly_falling_leaf(self, jet, missiles, flares, wall, decision, stats):
        jet.set_throttle(jet.max_speed * FALLING_LEAF_SPEED)
        fc = DURATION["FALLING_LEAF"] - self.timer
        jet.heading += math.sin(fc*0.35) * FALLING_LEAF_WOBBLE
        # 10 flares every 18 frames in a 180 degree rear arc
        if fc % 18 == 0:
            spawn_flares(flares, jet, 10, jet.heading+math.pi-math.pi/2, 180, 4.0, stats)
        self.timer -= 1
        if self.timer <= 0 or not missiles or decision.danger < 0.2:
            self.maneuver = "NORMAL"

    def _fly_jinking(self, jet, missiles, flares, wall, decision, stats):
        jet.set_throttle(jet.max_speed * JINK_SPEED_MULT)
        if self.timer % 10 == 0 and missiles:
            # Zigzag PERPENDICULAR to nearest missile — not random
            nm = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
            m_angle = math.atan2(nm.vy, nm.vx)
            # Alternate left/right perpendicular each jink
            side = 1 if (self.timer // 10) % 2 == 0 else -1
            prop = m_angle + side * math.pi / 2
            pvx  = math.cos(prop) * jet.speed
            pvy  = math.sin(prop) * jet.speed
            # Wall safety — try other side if this one leads to wall
            if (jet.x+pvx*20 < 80 or jet.x+pvx*20 > 720 or
                    jet.y+pvy*20 < 80 or jet.y+pvy*20 > 470):
                prop = m_angle - side * math.pi / 2
            jet.heading = prop
        # Intercept flares during jinking — aimed at each missile
        if self.timer % 20 == 0:
            for miss in missiles[:3]:   # cap at 3 missiles for perf
                angle, speed = predict_intercept_flare(jet, miss)
                flares.append(FlareParticle(jet.x, jet.y, angle,       speed))
                flares.append(FlareParticle(jet.x, jet.y, angle - 0.2, speed*0.85))
                stats['flares'] += 2
        self.timer -= 1
        if self.timer <= 0:
            self.maneuver = "NORMAL"
            self.general_cooldown = JINK_RECOVER_COOLDOWN

    def _fly_barrel_roll(self, jet, missiles, flares, wall, decision, stats):
        jet.set_throttle(jet.max_speed)
        if wall.wx > 100 and wall.wy > 100:
            jet.heading += BARREL_ROLL_TURN
        else:
            jet.heading   = heading_to_center(jet)
            self.maneuver = "NORMAL"
        # 2 side flares every 24 frames during roll
        if self.timer % 24 == 0:
            spawn_flares(flares, jet, 6, jet.heading+math.pi/2-math.radians(20), 40, 4.0, stats)
        self.timer -= 1
        if self.timer <= 0:
            self.maneuver = "NORMAL"

    def _fly_normal(self, jet, missiles, flares, wall, decision, stats):
        # Speed autonomy: the planner picked this throttle by simulating the
        # physics, so it is the jet's own choice rather than a fixed cruise.
        live = decision.fresh or bool(missiles)   # correction runs on live data
        jet.set_throttle(decision.flight_throttle if live else jet.max_speed)

        if live:
            combined = normalize_angle(decision.flight_heading - jet.heading)

            if missiles and REFLEX_WEIGHT > 0.0:
                # Reflex only inside REFLEX_RANGE, where the plan may be stale.
                nearest_m = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                nearest_d = math.hypot(nearest_m.x-jet.x, nearest_m.y-jet.y)
                if nearest_d < REFLEX_RANGE:
                    flee_h = math.atan2(jet.y - nearest_m.y, jet.x - nearest_m.x)
                    w = REFLEX_WEIGHT * ((REFLEX_RANGE - nearest_d) / REFLEX_RANGE) ** 1.5
                    combined = (combined * (1.0 - w)
                                + normalize_angle(flee_h - jet.heading) * w)

            # Smooth steer — never snap, let physics carry momentum
            jet.heading += combined * NORMAL_STEER_GAIN

        elif missiles:
            # Unreachable while the corrector runs; kept for safety.
            nearest_m = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
            flee_h    = math.atan2(jet.y - nearest_m.y, jet.x - nearest_m.x)
            jet.heading += normalize_angle(flee_h - jet.heading) * NORMAL_STEER_GAIN

        # Intercept-aimed flares — throw toward missile's predicted position
        if (decision.drop_flare and missiles
                and pygame.time.get_ticks()-jet.last_flare_time > FLARE_REARM_MS
                and wall.wx > FLARE_MIN_WALL and wall.wy > FLARE_MIN_WALL):
            jet.last_flare_time = pygame.time.get_ticks()
            for miss in missiles:
                angle, speed = predict_intercept_flare(jet, miss)
                flares.append(FlareParticle(jet.x, jet.y, angle,        speed))
                flares.append(FlareParticle(jet.x, jet.y, angle - 0.20, speed*0.85))
                flares.append(FlareParticle(jet.x, jet.y, angle + 0.20, speed*0.85))
                stats['flares'] += 3
            decision.drop_flare = False
