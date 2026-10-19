"""
Scripted players, for data generation and benchmarks.

ScriptedCommander plays the missile side. At `pressure >= 1` it fires every
battery the moment it is ready and launches every battery's clusters and the
pincers the moment they come off cooldown -- that is the benchmark setting.
Below 1, `pressure` is the per-frame chance of firing once ready. With
`randomize=True` every launch varies speed, turn rate and lead around the game's
nominal profiles, so the network has to read a missile's behaviour rather than
recognise exact numbers. Flight styles are drawn by the arena itself, from its
own random generator.

Autopilot flies the jet while generating training data. It is not meant to be
good; it is meant to produce varied geometry -- long straight runs, hard breaks
near missiles, flying behind terrain and back out, flares now and then.
"""

import math
import random

from ..config import HEIGHT, HUD_HEIGHT, WIDTH
from ..sim.entities import FlareParticle
from ..sim.geometry import inside_any
from ..sim.tactics import predict_intercept_flare
from ..utils import normalize_angle

CLUSTER_BASE_BIASES = [-0.014, -0.007, 0.0, 0.007, 0.014]


class ScriptedCommander(object):
    """
    `jitter` > 0 makes a maximum-pressure opponent differ from seed to seed: each
    battery opens fire after a random delay and hesitates up to `jitter` frames
    each time it comes ready. Without it, maximum pressure is fully determined by
    the cooldowns, and every seed replays the same match.
    """

    def __init__(self, rng=None, pressure=1.0, randomize=False, jitter=0):
        self.rng = rng or random.Random()
        self.pressure = pressure
        self.randomize = randomize
        self.jitter = jitter
        self.hold = {}          # launch key -> frame before which it will not fire

    def _go(self, key, arena, scale=1.0):
        if self.jitter:
            if key not in self.hold:
                self.hold[key] = arena.frame + self.rng.randint(0, 3 * self.jitter)
            if arena.frame < self.hold[key]:
                return False
        ok = self.pressure >= 1.0 or self.rng.random() < self.pressure * scale
        if ok and self.jitter:
            self.hold[key] = arena.frame + self.rng.randint(0, self.jitter) + 1
        return ok

    def act(self, arena):
        rng, level = self.rng, arena.level
        for i, st in enumerate(level.stations):
            if arena.station_ready(i) and self._go(("fire", i), arena):
                kw = {}
                if self.randomize:
                    if st.kind == "ballistic":
                        kw['speed'] = rng.uniform(6.0, 8.0)
                    else:
                        kw['speed'] = rng.uniform(3.8, 6.2)
                        kw['turn_rate'] = rng.uniform(0.05, 0.16)
                        if level.smart:
                            if level.datalink:
                                kw['lead'] = rng.uniform(0.2, 0.8)
                            else:
                                kw['lead'] = 0.0 if rng.random() < 0.5 else rng.uniform(0.2, 0.6)
                arena.fire(i, **kw)
        for i, st in enumerate(level.stations):
            if arena.cluster_ready(i) and self._go(("cluster", i), arena, 0.25):
                if self.randomize:
                    arena.launch_cluster(
                        i,
                        speeds=sorted(rng.uniform(3.2, 4.8) for _ in range(5)),
                        turn_rates=[rng.uniform(0.02, 0.07) for _ in range(5)],
                        lateral_biases=[b * rng.uniform(0.5, 1.5) for b in CLUSTER_BASE_BIASES])
                else:
                    arena.launch_cluster(i)
        if level.pincer and arena.pincer_ready() and self._go("pincer", arena, 0.3):
            if self.randomize:
                arena.launch_pincer(speed=rng.uniform(4.6, 5.8), turn_rate=rng.uniform(0.07, 0.13))
            else:
                arena.launch_pincer()


class Autopilot(object):
    """A deliberately mediocre jet, for varied training geometry."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.waypoint = None
        self.timer = 0
        self.throttle = 5.0
        self.break_side = 1.0
        self.last_flare = -10 ** 6

    def _new_waypoint(self, arena):
        rng = self.rng
        for _ in range(30):
            wx = rng.uniform(80, WIDTH - 80)
            wy = rng.uniform(80, HEIGHT - HUD_HEIGHT - 80)
            if not inside_any(wx, wy, arena.terrain, 50.0):
                break
        self.waypoint = (wx, wy)
        self.timer = rng.randint(60, 220)
        self.throttle = rng.uniform(2.0, 7.5)
        self.break_side = rng.choice([-1.0, 1.0])

    def act(self, arena):
        jet, rng = arena.jet, self.rng
        self.timer -= 1
        if (self.waypoint is None or self.timer <= 0 or
                math.hypot(self.waypoint[0] - jet.x, self.waypoint[1] - jet.y) < 40):
            self._new_waypoint(arena)
        desired = math.atan2(self.waypoint[1] - jet.y, self.waypoint[0] - jet.x)

        nearest, nd = None, 1e9
        for m in arena.missiles:
            d = math.hypot(m.x - jet.x, m.y - jet.y)
            if d < nd:
                nearest, nd = m, d
        if nearest is not None and nd < 150:
            away = math.atan2(jet.y - nearest.y, jet.x - nearest.x)
            desired = away + self.break_side * 1.2

        vx, vy = math.cos(desired), math.sin(desired)
        margin = 90.0
        if jet.x < margin:                    vx += (margin - jet.x) / 30.0
        if jet.x > WIDTH - margin:            vx -= (jet.x - (WIDTH - margin)) / 30.0
        if jet.y < margin:                    vy += (margin - jet.y) / 30.0
        if jet.y > HEIGHT - HUD_HEIGHT - margin:
            vy -= (jet.y - (HEIGHT - HUD_HEIGHT - margin)) / 30.0
        for ox, oy, r in arena.terrain:
            d = math.hypot(jet.x - ox, jet.y - oy)
            gap = d - r
            if gap < 80.0 and d > 1e-6:
                push = (80.0 - gap) / 25.0
                vx += (jet.x - ox) / d * push
                vy += (jet.y - oy) / d * push
        desired = math.atan2(vy, vx)
        jet.heading += normalize_angle(desired - jet.heading) * 0.18
        jet.set_throttle(self.throttle)

        if (nearest is not None and nd < 120 and arena.frame - self.last_flare > 90
                and rng.random() < 0.3):
            for m in arena.missiles[:4]:
                angle, speed = predict_intercept_flare(jet, m)
                arena.flares.append(FlareParticle(jet.x, jet.y, angle, speed))
                arena.stats['flares'] += 1
            self.last_flare = arena.frame
