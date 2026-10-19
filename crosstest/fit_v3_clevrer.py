"""
crosstest/fit_v3_clevrer.py  --  re-fit the V3 world model's free constants on CLEVRER TRAIN
===========================================================================================

The mirror of fine-tuning CausalVis's model on game trajectories: V3 keeps its rules
and only the numbers inside them are fitted to the new world. TRAIN clips only; the
extraction of free-motion segments and isolated collisions is CausalVis's own
(probe/physics_laws.py), so the data a law is fitted on is exactly the data v8's
laws were fitted on.

  drag     V3 slows exactly one thing: flares, v <- v * DRAG before each move. Per
           CLEVRER object kind (shape/material) the per-frame loss c is fitted on the
           same grid physics_laws uses (0 .. 0.04); c = 0 is V3's missiles, which hold
           their speed.
  push     V3's only object-object interaction is the swarm push between classic
           homing missiles: headings turned apart inside REPULSE_DIST by up to
           REPULSE_FORCE. Its two numbers are fitted on isolated TRAIN collisions with
           physics_laws' head-to-head protocol: start both objects 8 frames before the
           recorded collision, run 16 frames, score the outgoing direction.

    C:/Python314/python crosstest/fit_v3_clevrer.py      -> crosstest/v3_clevrer_fit.json
"""

import json
import math
import os
import sys
import time

import numpy as np

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
CV = r"G:/CausalVis"
sys.path.insert(0, os.path.join(DNN, "crosstest"))
import clevrer_v3 as C3                                   # noqa: E402  (the kernel as a CLEVRER step)
sys.path.insert(0, os.path.join(CV, "probe"))
import physics_laws as PL                                 # noqa: E402  (CausalVis's extraction)

COLLISION_CLIPS = 500          # TRAIN clips for the push fit (of physics_laws' 1500)
DIST_GRID = (26.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0)   # px; 26 is the game's value
FORCE_GRID = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)          # 0.20 is the game's value


def fit_drag():
    P0, V0, F, G, _ = PL.free_segments(PL.TRAIN)
    grid = np.linspace(0, 0.04, 41)
    out, report = {}, {}
    for g in sorted(set(G)):
        m = G == g
        errs = np.array([PL.err(PL.pred_exp(P0[m], V0[m], c), F[m], 20).mean() for c in grid])
        best = int(np.argmin(errs))
        out[g] = float(grid[best])
        report[g] = {"c": float(grid[best]), "train_err20": float(errs[best]),
                     "train_err20_constant_speed": float(errs[0]), "n": int(m.sum())}
    return out, report, int(len(P0))


def head_to_head(r, drag, dist_px, force, repel=True):
    """physics_laws part C for the V3 kernel: two objects, 16 frames, outgoing direction."""
    P = np.stack([r["pi_s"], r["pj_s"]]).astype(np.float64)
    V = np.stack([r["vi_s"], r["vj_s"]]).astype(np.float64)
    keep = np.array([[1.0 - drag.get(r["gi"], 0.0)], [1.0 - drag.get(r["gj"], 0.0)]])
    traj, hit = [], False
    for _ in range(16):
        V = V * keep
        x1, v1 = C3.kernel_frame(P * C3.PX + C3.OFFSET, V * C3.PX, repel, dist_px, force)
        P = (x1 - C3.OFFSET) / C3.PX
        V = v1 / C3.PX
        if not hit and np.linalg.norm(P[0] - P[1]) - (r["Ri"] + r["Rj"]) <= 0.01:
            hit = True
        traj.append(P.copy())
    TR = np.array(traj)
    errs = [PL.angle(PL.slope(TR[:, k], 10, 15), obs)
            for k, obs in ((0, r["VI1"]), (1, r["VJ1"])) if np.linalg.norm(obs) > 0.001]
    return errs, hit


def score(rows, drag, dist_px, force, repel=True):
    errs, hits = [], 0
    for r in rows:
        e, h = head_to_head(r, drag, dist_px, force, repel)
        errs += e
        hits += h
    errs = np.array(errs)
    return float(np.median(errs)), float(np.percentile(errs, 75)), hits / max(len(rows), 1)


def main():
    t0 = time.time()
    drag, drag_report, n_seg = fit_drag()
    print("drag fitted on %d TRAIN free-motion segments (%.0fs)" % (n_seg, time.time() - t0))
    for g, d in drag_report.items():
        print("   %-16s c=%.3f/frame  train err@20 %.5f  (constant speed %.5f)  n=%d"
              % (g, d["c"], d["train_err20"], d["train_err20_constant_speed"], d["n"]))

    t0 = time.time()
    rows = PL.collisions(PL.TRAIN[:COLLISION_CLIPS])
    print("\n%d isolated TRAIN collisions from %d clips (%.0fs)" % (len(rows), COLLISION_CLIPS, time.time() - t0))
    t0 = time.time()
    base = score(rows, drag, C3.REPULSE_DIST_GAME, 0.0, repel=False)
    print("   no push (straight lines + fitted drag): median %.1f deg  p75 %.1f  contact %.1f%%"
          % (base[0], base[1], 100 * base[2]))
    grid = []
    for dist in DIST_GRID:
        for force in FORCE_GRID:
            med, p75, hit = score(rows, drag, dist, force)
            grid.append((med, p75, dist, force, hit))
            print("   push dist %5.1f px  force %.2f: median %.1f deg  p75 %.1f  contact %.1f%%"
                  % (dist, force, med, p75, 100 * hit), flush=True)
    med, p75, dist, force, hit = min(grid)
    print("\nchosen on TRAIN: dist %.1f px (%.4f table units), force %.2f -> median %.1f deg, p75 %.1f   (%.0fs)"
          % (dist, dist / C3.PX, force, med, p75, time.time() - t0))
    out = {"drag": drag, "repulse_dist_px": dist, "repulse_force": force,
           "train": {"drag": drag_report, "push_grid": [{"dist_px": d, "force": f, "median_deg": m, "p75_deg": p,
                                                          "contact": h} for m, p, d, f, h in grid],
                     "no_push": {"median_deg": base[0], "p75_deg": base[1], "contact": base[2]},
                     "chosen": {"median_deg": med, "p75_deg": p75, "contact": hit}},
           "fitted_on": "TRAIN only: %d free-motion segments; %d isolated collisions from the first %d "
                        "of physics_laws' TRAIN clips" % (n_seg, len(rows), COLLISION_CLIPS)}
    path = os.path.join(DNN, "crosstest", "v3_clevrer_fit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
