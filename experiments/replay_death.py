"""Replay a benchmark seed exactly (same arena seed as benchmark.py) and dump the frames before death."""
import collections
import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from neural_evasion.training import benchmark as B  # noqa: E402
from neural_evasion.levels import LEVELS  # noqa: E402
from neural_evasion.arena import Arena  # noqa: E402
from neural_evasion.training.commander import ScriptedCommander  # noqa: E402
from neural_evasion.sim.geometry import inside_any  # noqa: E402
import neural_evasion.pilots.graph_pilot as GP  # noqa: E402

pygame.time.get_ticks = B._frame_ticks
import torch  # noqa: E402
torch.set_num_threads(1)

_real_plan = GP.plan_v3


def plan_keep(jet, threats, level, preferred=None, keep_future=False, **kw):
    return _real_plan(jet, threats, level, preferred=preferred, keep_future=True, **kw)


GP.plan_v3 = plan_keep


def run(seed, level_no, spec, frames=1800, tail=150, every=5):
    level = LEVELS[level_no - 1]
    rng = random.Random(seed * 7919 + level.number)
    arena = Arena(level, launchers_invulnerable=True, visuals=False, seed=seed * 7919 + level.number)
    while True:
        arena.jet.x = 400.0 + rng.uniform(-120.0, 120.0)
        arena.jet.y = 275.0 + rng.uniform(-80.0, 80.0)
        if not inside_any(arena.jet.x, arena.jet.y, arena.terrain, 40.0):
            break
    arena.jet.heading = rng.uniform(-3.14159, 3.14159)
    pilot = B.make(spec, level)
    pilot.start(arena)
    cmd = ScriptedCommander(rng, pressure=1.0, jitter=B.JITTER)
    log = collections.deque(maxlen=tail)
    mag = []
    for _ in range(frames):
        B._FRAME['n'] = arena.frame + 1
        f0, jx, jy, jh, js = arena.frame, arena.jet.x, arena.jet.y, arena.jet.heading, arena.jet.speed
        ch0, vol0 = arena.flare_charges, arena.stats['volleys']
        arena.step(pilot, cmd)
        if arena.flare_charges > ch0 or arena.stats['volleys'] > vol0:
            mag.append("f%d charges %d->%d volleys %d" % (f0, ch0, arena.flare_charges, arena.stats['volleys']))
        p = pilot.plan
        res, b = p.result, p.best
        hit_ok = res.hit_frame < 0
        wall_ok = res.wall_frame < 0
        terr_ok = res.terrain_frame < 0
        safe = hit_ok & wall_ok & terr_ok
        near = min([math.hypot(m.x - jx, m.y - jy) for m in arena.missiles] or [999.0])
        log.append("f%4d jet(%5.1f,%5.1f) h%+5.2f v%4.2f | %-12s h%+5.2f t%4.2f | best hit%3d wall%3d terr%3d "
                   "gap%6.1f | safe%2d/%d nohit%2d nowall%2d | thr%2d near%5.0f | chg %d fire %s gain %+.2f"
                   % (f0, jx, jy, jh, js, p.label, p.heading, p.throttle,
                      res.hit_frame[b], res.wall_frame[b], res.terrain_frame[b], p.min_gap,
                      int(safe.sum()), len(safe), int(hit_ok.sum()), int(wall_ok.sum()),
                      len(pilot.plan_threats), near, arena.flare_charges, p.flare, p.flare_gain))
        if arena.game_over:
            break
    print("=== L%d %s seed %d: %s at frame %d, jet (%.1f, %.1f), volleys %d, charges earned %d, denied %d"
          % (level_no, spec, seed, arena.cause or "SURVIVED", arena.frame, arena.jet.x, arena.jet.y,
             arena.stats['volleys'], arena.stats['charges_earned'], arena.stats['flares_denied']))
    print("magazine:", "; ".join(mag[-30:]))
    rows = list(log)
    for i, row in enumerate(rows):
        if i % every == 0 or i >= len(rows) - 15:
            print(row)


if __name__ == "__main__":
    spec = sys.argv[1]
    level_no = int(sys.argv[2])
    for s in sys.argv[3:]:
        run(int(s), level_no, spec)
