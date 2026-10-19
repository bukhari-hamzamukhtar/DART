"""Try a terminal cost in the V3 planner on benchmark seeds (monkeypatched; the repo is untouched)."""
import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from neural_evasion.training import benchmark as B  # noqa: E402
from neural_evasion.levels import LEVELS  # noqa: E402
import neural_evasion.sim.planner_v3 as P  # noqa: E402
from neural_evasion.sim.planner import HIT_RADIUS, WALL_MARGIN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="3-oracle")
ap.add_argument("--level", type=int, default=3)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--tw", type=float, default=0.0, help="terminal threat weight")
ap.add_argument("--tm", type=float, default=150.0, help="terminal threat margin, px above HIT_RADIUS")
ap.add_argument("--tf", type=float, default=2.0, help="falloff exponent")
ap.add_argument("--ww", type=float, default=0.0, help="terminal wall weight")
args = ap.parse_args()

_base = P.cost_v3


def cost_terminal(res, throttles, horizon=P.HORIZON):
    cost = _base(res, throttles, horizon)
    ok = res.hit_frame < 0
    if args.tw > 0.0:
        gap = np.maximum(res.dist_profile[:, -1] - HIT_RADIUS, 0.0)
        cost = cost + np.where(ok, args.tw * np.clip(1.0 - gap / args.tm, 0.0, 1.0) ** args.tf, 0.0)
    if args.ww > 0.0:
        room = np.maximum(res.wall_profile[:, -1], 0.0)
        cost = cost + args.ww * np.clip(1.0 - room / WALL_MARGIN, 0.0, 1.0) ** args.tf
    return cost


P.cost_v3 = cost_terminal
pygame.time.get_ticks = B._frame_ticks
import torch  # noqa: E402
torch.set_num_threads(1)

level = LEVELS[args.level - 1]
alive = 0
causes = {}
for s in args.seeds:
    r = B.run_one(args.spec, level, s, 1800)
    alive += r['survived']
    if not r['survived']:
        causes[r['cause']] = causes.get(r['cause'], 0) + 1
    print("L%d %s seed %d: %s at frame %d" % (args.level, args.spec, s,
                                             "SURVIVED" if r['survived'] else "died (%s)" % r['cause'], r['frames']))
    sys.stdout.flush()
print("VARIANT tw %.1f tm %.0f tf %.1f ww %.1f  L%d %s: %d/%d  %s"
      % (args.tw, args.tm, args.tf, args.ww, args.level, args.spec, alive, len(args.seeds), causes))
