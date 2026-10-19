"""
Closed-loop benchmark: every pilot on every level, against the same opponent.

Pure evasion. Launchers cannot be destroyed, the scripted commander fires at
maximum rate -- every battery the moment it is ready, clusters and pincers the
moment they are off cooldown -- and a run ends when the jet dies or after
--frames frames. Wall-clock timers inside the frozen v1/v2 code (flare rearm) are
driven from the frame counter, so every pilot gets identical timing however fast
or slow its brain runs.

Pilots:
    1, 2          the frozen builds, inference run inline (deterministic)
    3             v3 with the trained temporal graph brain
    3-oracle      v3 given every missile's true guidance values (the ceiling)
    3-prior       v3 with no GNN: tracker priors only (what the GNN is worth)
    3-no_edges    v3 with the no-message-passing ablation checkpoint
    3-no_memory   v3 with the no-memory ablation checkpoint
    <any>-cal     that v3 pilot with the conformal turn-rate correction
                  (training/calibrate.py)

    python -m neural_evasion.training.benchmark --levels 2 --pilots 1 2 3 3-oracle 3-prior --seeds 8
"""

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from ..arena import Arena                           # noqa: E402
from ..config import PROJECT_ROOT                   # noqa: E402
from ..levels import LEVELS                         # noqa: E402
from .commander import ScriptedCommander            # noqa: E402

_FRAME = {'n': 0}


def _frame_ticks():
    return int(_FRAME['n'] * 1000 / 60)


def make(spec, level):
    if spec in ("1", "2"):
        from ..pilots.frozen import FrozenPilot
        return FrozenPilot(int(spec), threaded=False)
    from ..pilots.graph_pilot import GraphPilot
    calibrated = False
    if spec.endswith("-cal"):                # the same pilot, conformal turn-rate correction on
        calibrated, spec = True, spec[:-len("-cal")]
    if spec == "3":
        p = GraphPilot(level, calibrated=calibrated)
    elif spec == "3-oracle":
        p = GraphPilot(level, oracle=True, use_brain=False)   # decisions never read the brain
    elif spec == "3-prior":
        p = GraphPilot(level, use_brain=False)
    elif spec.startswith("3-"):
        from ..model.temporal_gnn import load_graph_brain
        model = load_graph_brain("jet_brain_tgn_%s.pt" % spec[2:], verbose=False)
        if model is None:
            raise SystemExit("missing checkpoint for %s" % spec)
        p = GraphPilot(level, model=model, calibrated=calibrated)
    else:
        raise SystemExit("unknown pilot %s" % spec)
    p.show_overlay = False
    return p


JITTER = 25     # frames; see ScriptedCommander


def run_one(spec, level, seed, frames):
    rng = random.Random(seed * 7919 + level.number)
    # The arena's seed fixes every missile's flight style for this match.
    arena = Arena(level, launchers_invulnerable=True, visuals=False, seed=seed * 7919 + level.number)
    # Each seed is a different match: the jet starts somewhere else, pointing
    # somewhere else, and the batteries keep a different rhythm.
    from ..sim.geometry import inside_any
    while True:
        arena.jet.x = 400.0 + rng.uniform(-120.0, 120.0)
        arena.jet.y = 275.0 + rng.uniform(-80.0, 80.0)
        if not inside_any(arena.jet.x, arena.jet.y, arena.terrain, 40.0):
            break
    arena.jet.heading = rng.uniform(-3.14159, 3.14159)
    pilot = make(spec, level)
    pilot.start(arena)
    cmd = ScriptedCommander(rng, pressure=1.0, jitter=JITTER)
    t0 = time.perf_counter()
    for _ in range(frames):
        _FRAME['n'] = arena.frame + 1
        arena.step(pilot, cmd)
        if arena.game_over:
            break
    wall = time.perf_counter() - t0
    pilot.stop()
    return {'pilot': spec, 'level': level.number, 'seed': seed,
            'survived': not arena.game_over, 'frames': arena.frame,
            'cause': arena.cause, 'fired': arena.stats['fired'],
            'flares': arena.stats['flares'], 'ms_per_frame': wall / max(1, arena.frame) * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pilots", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=0,
                    help="run seeds first-seed .. first-seed+seeds-1, to extend an earlier run")
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pygame.time.get_ticks = _frame_ticks
    import torch
    torch.set_num_threads(1)

    results = []
    for ln in args.levels:
        level = LEVELS[ln - 1]
        for spec in args.pilots:
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                r = run_one(spec, level, seed, args.frames)
                results.append(r)
                print("L%d pilot %-11s seed %d: %s at frame %4d  fired %3d  flares %3d  %.1f ms/frame"
                      % (ln, spec, seed, "SURVIVED" if r['survived'] else "died (%s)" % r['cause'],
                         r['frames'], r['fired'], r['flares'], r['ms_per_frame']))
                sys.stdout.flush()

    print()
    print("%-6s %-12s %9s %12s %10s  %s" % ("level", "pilot", "survived", "mean alive", "ms/frame", "deaths"))
    for ln in args.levels:
        for spec in args.pilots:
            rs = [r for r in results if r['level'] == ln and r['pilot'] == spec]
            if not rs:
                continue
            alive = sum(r['survived'] for r in rs)
            causes = {}
            for r in rs:
                if not r['survived']:
                    causes[r['cause']] = causes.get(r['cause'], 0) + 1
            print("L%-5d %-12s %5d/%-3d %10.1f s %9.1f   %s"
                  % (ln, spec, alive, len(rs), sum(r['frames'] for r in rs) / len(rs) / 60.0,
                     sum(r['ms_per_frame'] for r in rs) / len(rs),
                     ", ".join("%s %d" % kv for kv in sorted(causes.items())) or "-"))

    out = args.out or os.path.join(PROJECT_ROOT, "training_data_v3",
                                   "bench_L%s_%s.json" % ("".join(map(str, args.levels)),
                                                          "_".join(args.pilots)))
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
