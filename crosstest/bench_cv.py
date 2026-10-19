"""
crosstest/bench_cv.py  --  closed loop: the V3 pilot flying on CausalVis's world model
====================================================================================

neural_evasion/training/benchmark.py's harness, called unmodified: the same 16 seeds
per level, jet start, scripted opponent at maximum rate, frame-locked timing, the same
record format (so training/report.py merges these with the V1/V2/V3 results). The
pilot is the one the "3-oracle" row uses -- GraphPilot fed every missile's true state,
no GNN -- and the only thing that differs between the rows below is the world model
the planner imagines missiles with:

  v3-oracle  V3's own world model (must reproduce bench_L*_oracle_prior.json exactly)
  cv-zero    CausalVis's world model, zero-shot (crosstest/cv_bridge.py)
  cv-refit   the same model fine-tuned on game trajectories (crosstest/train_cv_refit.py)
  straight   straight lines at constant velocity, through the same bridge

Inputs are the true state for every row, so each gets the most favourable start.

    python crosstest/bench_cv.py --levels 1 --pilots cv-zero straight --seeds 16
"""

import argparse
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame   # noqa: E402

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
sys.path.insert(0, DNN)
sys.path.insert(0, os.path.join(DNN, "crosstest"))

import neural_evasion.sim.planner_v3 as PV             # noqa: E402
import neural_evasion.training.benchmark as B          # noqa: E402
from neural_evasion.levels import LEVELS               # noqa: E402

REFIT = os.path.join(DNN, "crosstest", "cv_refit.pt")
V3_ROLLOUT = PV.rollout
_WM = {}


def world_model(spec):
    if spec == "v3-oracle":
        return V3_ROLLOUT
    if spec not in _WM:
        from cv_bridge import CVWorldModel
        if spec == "straight":
            _WM[spec] = CVWorldModel(free=True)
        elif spec == "cv-zero":
            _WM[spec] = CVWorldModel(mapping="zero")
        elif spec == "cv-refit":
            from world import load_spatial
            _WM[spec] = CVWorldModel(model=load_spatial(REFIT, yaw_known=False), mapping="refit")
        else:
            raise SystemExit("unknown pilot %s" % spec)
    return _WM[spec].rollout


def make(spec, level):
    from neural_evasion.pilots.graph_pilot import GraphPilot
    PV.rollout = world_model(spec)            # the planner's imagination
    p = GraphPilot(level, oracle=True, use_brain=False)
    p.show_overlay = False
    return p


B.make = make                                 # run_one builds its pilot through this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pilots", nargs="+", default=["cv-zero", "straight"])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pygame.time.get_ticks = B._frame_ticks
    import torch
    torch.set_num_threads(1)
    results = []
    for ln in a.levels:
        level = LEVELS[ln - 1]
        for spec in a.pilots:
            for seed in range(a.first_seed, a.first_seed + a.seeds):
                r = B.run_one(spec, level, seed, a.frames)
                results.append(r)
                print("L%d pilot %-10s seed %2d: %s at frame %4d  fired %3d  flares %3d  %.1f ms/frame"
                      % (ln, spec, seed, "SURVIVED" if r['survived'] else "died (%s)" % r['cause'],
                         r['frames'], r['fired'], r['flares'], r['ms_per_frame']), flush=True)
    print()
    for ln in a.levels:
        for spec in a.pilots:
            rs = [r for r in results if r['level'] == ln and r['pilot'] == spec]
            causes = {}
            for r in rs:
                if not r['survived']:
                    causes[r['cause']] = causes.get(r['cause'], 0) + 1
            print("L%d %-10s %2d/%d survived  mean alive %.1f s  %s"
                  % (ln, spec, sum(r['survived'] for r in rs), len(rs),
                     sum(r['frames'] for r in rs) / len(rs) / 60.0,
                     ", ".join("%s %d" % kv for kv in sorted(causes.items())) or "-"))
    out = a.out or os.path.join(DNN, "crosstest", "results", "bench_L%s_%s.json"
                                % ("".join(map(str, a.levels)), "_".join(a.pilots)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
