"""
crosstest/wm_error_cv.py  --  missile prediction error in the game: CausalVis's world model
                              against V3's, on identical sampled moments
==========================================================================================

The protocol and the sampled moments are neural_evasion/training/evaluate.py's
world_model_error, reproduced statement for statement (same seeds, same scripted
opponent and autopilot, a moment every 30 frames, each missile's future predicted
along the jet's ACTUAL next 72 frames and compared with where it really went). So
the V3 rows here must equal training_data_v3/evaluation.json -- a check that this
harness is the same one.

Rows
  const_vel      evaluate.py's: keep flying straight from the tracker's estimate
  v3_gnn         V3 world model fed the trained brain's estimates, as the pilot plans
  v3_oracle      V3 world model fed every missile's true state, guidance values and flares
  straight_true  straight lines from the TRUE state, through the bridge (the floor any
                 physics-without-guidance should beat)
  cv_zero        CausalVis's world model (v6_voxel.pt), zero-shot, fed the true state
  cv_refit       the same model fine-tuned on game trajectories (crosstest/train_cv_refit.py)

    python crosstest/wm_error_cv.py            (from the DNN folder, PYTHONPATH=.)
"""

import argparse
import json
import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np   # noqa: E402
import torch         # noqa: E402

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
sys.path.insert(0, DNN)
sys.path.insert(0, os.path.join(DNN, "crosstest"))

from neural_evasion.arena import Arena                                     # noqa: E402
from neural_evasion.config import GRAPH_TICK                               # noqa: E402
from neural_evasion.levels import LEVELS                                   # noqa: E402
from neural_evasion.model.temporal_gnn import GraphBrain, load_graph_brain  # noqa: E402
from neural_evasion.sim.scene_graph import SceneGraphBuilder               # noqa: E402
from neural_evasion.sim.tracker import Tracker                             # noqa: E402
from neural_evasion.sim.worldmodel_v3 import flare_state, oracle_threat, rollout  # noqa: E402
from neural_evasion.training.commander import Autopilot, ScriptedCommander  # noqa: E402
from cv_bridge import CVWorldModel                                         # noqa: E402

HORIZONS = (24, 48, 72)
REFIT = os.path.join(DNN, "crosstest", "cv_refit.pt")


def load_models(names):
    from world import load_spatial
    models = {}
    for name in names:
        if name == "straight_true":
            models[name] = CVWorldModel(free=True)
        elif name == "cv_zero":
            models[name] = CVWorldModel(mapping="zero")
        elif name == "cv_refit":
            models[name] = CVWorldModel(model=load_spatial(REFIT, yaw_known=False), mapping="refit")
        else:
            raise SystemExit("unknown model %s" % name)
    return models


def world_model_error(level, brain_model, cv_models, episodes=4, frames=1500, sample_every=30, seed=500):
    names = ["const_vel", "v3_gnn", "v3_oracle"] + list(cv_models)
    errs = {k: {h: [] for h in HORIZONS} for k in names}
    for e in range(episodes):
        rng = random.Random(seed + 97 * e + level.number)
        arena = Arena(level, jet_invulnerable=True, launchers_invulnerable=True, visuals=False,
                      seed=seed + 97 * e + level.number, magazine=False)
        cmd = ScriptedCommander(rng, pressure=rng.uniform(0.015, 0.04), randomize=True)
        pilot = Autopilot(rng)
        tracker = Tracker(level)
        builder = SceneGraphBuilder(level)
        brain = GraphBrain(brain_model)
        snaps, jet_hist, truth_hist = [], {}, {}
        for _ in range(frames):
            arena.step(pilot, cmd)
            f = arena.frame
            jet = arena.jet
            jet_hist[f] = (complex(jet.x, jet.y), complex(jet.vx, jet.vy))
            for m in arena.missiles:
                truth_hist.setdefault(m.uid, {})[f] = (m.x, m.y)
            vis, vis_f = arena.radar_picture()
            tracker.update(f, jet, vis)
            if f % GRAPH_TICK == 0:
                g = builder.build(f, jet, tracker, vis_f, arena.launchers)
                rows, links, _ = brain.step(g)
                tracker.apply_estimates(g.track_tids, rows, links)
            if f % sample_every == 0 and tracker.tracks and f + 72 <= frames:
                f0 = f + 1
                gnn = tracker.threat_dicts(f0)
                truth = {m.uid: m for m in arena.missiles}
                tids = [d['tid'] for d in gnn]
                oracle = [oracle_threat(truth[tid], level.datalink, f0) if tid in truth else None for tid in tids]
                seen_flares = [flare_state(fl, f0) for fl in vis_f if not getattr(fl, 'dead', False)]
                all_flares = [flare_state(fl, f0) for fl in arena.flares if not getattr(fl, 'dead', False)]
                snaps.append((f, tids, gnn, oracle, seen_flares, all_flares))
        for f, tids, gnn, oracle, seen_flares, all_flares in snaps:
            path = np.array([jet_hist[f + k][0] for k in range(1, 73)])
            vel = np.array([jet_hist[f + k][1] for k in range(1, 73)])
            keep = [i for i, tid in enumerate(tids) if oracle[i] is not None]
            if not keep:
                continue
            kw = dict(terrain=level.obstacles, datalink=level.datalink, jet_path=(path, vel))
            preds = {}
            res = rollout(None, [gnn[i] for i in keep], None, None, 72, flares=seen_flares, **kw)
            preds["v3_gnn"] = (res.threat_x[0], res.threat_y[0])
            res = rollout(None, [oracle[i] for i in keep], None, None, 72, flares=all_flares, **kw)
            preds["v3_oracle"] = (res.threat_x[0], res.threat_y[0])
            start = (jet_hist[f][0].real, jet_hist[f][0].imag)
            for name, wm in cv_models.items():
                res = wm.rollout(None, [oracle[i] for i in keep], None, None, 72, jet_start=start, **kw)
                preds[name] = (res.threat_x[0], res.threat_y[0])
            for j, i in enumerate(keep):
                hist = truth_hist.get(tids[i], {})
                for h in HORIZONS:
                    real = hist.get(f + h)
                    if real is None:
                        continue
                    d0 = gnn[i]
                    errs["const_vel"][h].append(math.hypot(d0['pos'][0] + d0['vel'][0] * h - real[0],
                                                           d0['pos'][1] + d0['vel'][1] * h - real[1]))
                    for name, (X, Y) in preds.items():
                        errs[name][h].append(math.hypot(X[j, h - 1] - real[0], Y[j, h - 1] - real[1]))
    return {k: {h: (float(np.median(v)) if v else None, float(np.mean(v)) if v else None, len(v))
                for h, v in hs.items()} for k, hs in errs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["straight_true", "cv_zero"],
                    help="bridge rows to add: straight_true, cv_zero, cv_refit")
    ap.add_argument("--out", default=os.path.join(DNN, "crosstest", "results", "wm_error_game.json"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    brain_model = load_graph_brain("jet_brain_tgn.pt", verbose=False)
    cv_models = load_models(a.models)
    report = {}
    for ln in a.levels:
        lv = LEVELS[ln - 1]
        wm = world_model_error(lv, brain_model, cv_models)
        report["L%d" % ln] = wm
        print("\nMISSILE PREDICTION ERROR, level %d  (median px / mean px, n)" % ln)
        for name, rows in wm.items():
            print("  %-14s " % name + "   ".join("+%d: %6.1f / %6.1f (n=%d)" % (h, v[0], v[1], v[2])
                                                 if v[0] is not None else "+%d: -" % h
                                                 for h, v in sorted(rows.items())))
        sys.stdout.flush()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
