"""
Offline evaluation of the v3 brain.

Does it actually infer hidden missile state, and do the graph (message passing)
and the memory each earn their place? Two tables.

1. Inference accuracy on the held-out episodes, per estimate, for
     rule       a hand-written estimator with its own per-track memory: turn rate
                is the largest turn observed so far (0.10 until one is seen),
                ballistic if it has never turned and flies faster than 6.3,
                lock = doctrine geometry right now, threat = constant-velocity
                closest approach, datalink = born within 16 frames of each
                other, lead 0, side 0, no flank angle, no weave
     no_edges   the temporal GNN without message passing
     no_memory  message passing, but a fresh network every tick
     full       the temporal graph network
   reported for all tracks and separately for tracks coasting behind terrain.

2. World-model error. From moments sampled in fresh arena runs, the v3 world
   model is rolled along the jet's *actual* subsequent flight and each missile's
   predicted position is compared with where it really went, +24/+48/+72 frames:
     const_vel  no model: keep flying straight
     prior      the world model fed tracker priors and the flares the radar sees
     gnn_mean   the world model fed the trained brain's estimates, turn rate at
                its mean, and the flares the radar sees: how accurate inference is
     gnn        the same, turn rate at the network's 90% quantile, which is what
                the pilot actually plans against
     oracle     the world model fed true values and every flare; the only
                unmodelled events left are the jet's counter-attack missiles

   A weaving or erratic missile's swing cannot be predicted dart by dart without
   its true numbers, so the GNN rows carry that error too.

    python -m neural_evasion.training.evaluate
    python -m neural_evasion.training.evaluate --only-wm   # refresh table 2 only
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

from ..arena import Arena                                           # noqa: E402
from ..config import GRAPH_TICK, PROJECT_ROOT                       # noqa: E402
from ..levels import LEVELS                                         # noqa: E402
from ..model.temporal_gnn import GraphBrain, decode_tracks, load_graph_brain  # noqa: E402
from ..sim.scene_graph import SEEKER, THREAT, SceneGraphBuilder     # noqa: E402
from ..sim.tracker import PRIOR, Tracker                            # noqa: E402
from ..sim.worldmodel_v3 import flare_state, oracle_threat, rollout  # noqa: E402
from .commander import Autopilot, ScriptedCommander                 # noqa: E402
from .generate import DATA_DIR                                      # noqa: E402
from .train import forward_tick, load_split                         # noqa: E402

TRACK_KEYS = ["pursuit", "turn_rate", "lead", "side", "guided", "threat", "turn_hi", "bias",
              "flank", "wobble"]
N_EST = len(TRACK_KEYS)


# ── estimators over stored ticks ─────────────────────────────
class RuleEstimator(object):
    def __init__(self):
        self.max_turn = {}

    def reset(self):
        self.max_turn = {}

    def __call__(self, t):
        x = t['x'].astype(np.float32)
        rows = t['track_rows']
        est = np.zeros((len(rows), N_EST), dtype=np.float32)
        seeker_src = set(int(s) for s, e in zip(t['src'], t['etype']) if e == SEEKER)
        threat_attr = {int(s): t['eattr'][i].astype(np.float32)
                       for i, (s, e) in enumerate(zip(t['src'], t['etype'])) if e == THREAT}
        for i, (r, tid) in enumerate(zip(rows, t['track_tids'])):
            turn = abs(float(x[r, 14]) * 0.15)
            m = max(self.max_turn.get(int(tid), 0.0), turn)
            self.max_turn[int(tid)] = m
            speed = float(x[r, 19]) * 8.0
            est[i, 0] = 0.0 if (m < 0.004 and speed > 6.3) else 1.0
            est[i, 1] = m if m > 0.02 else 0.10
            est[i, 6] = max(m, 0.16)          # no way to know better: assume the most agile
            est[i, 4] = 1.0 if int(r) in seeker_src else 0.0
            a = threat_attr.get(int(r))
            if a is not None:
                rx, ry, rvx, rvy = a[0] * 400, a[1] * 400, a[2] * 10, a[3] * 10
                v2 = rvx * rvx + rvy * rvy
                tca = -(rx * rvx + ry * rvy) / v2 if v2 > 1e-6 else 0.0
                tca = min(max(tca, 0.0), 72.0)
                miss = math.hypot(rx + rvx * tca, ry + rvy * tca)
                est[i, 5] = 1.0 if miss < 40.0 else 0.0
        links = np.zeros(len(t['link_rows']), dtype=np.float32)
        for j, er in enumerate(t['link_rows']):
            a, b = int(t['src'][er]), int(t['dst'][er])
            links[j] = 1.0 if abs(float(x[a, 12]) - float(x[b, 12])) * 300.0 <= 16.0 else 0.0
        return est, links


class ModelEstimator(object):
    def __init__(self, model):
        self.model = model.eval()
        self.zero = torch.zeros(model.hidden)
        self.banks = [{}]

    def reset(self):
        self.banks = [{}]

    @torch.no_grad()
    def __call__(self, t):
        b, raw, link_logit = forward_tick(self.model, [t], self.banks, self.zero)
        est = np.zeros((len(t['track_rows']), N_EST), dtype=np.float32)
        if raw is not None:
            d = decode_tracks(raw)
            for k, key in enumerate(TRACK_KEYS):
                est[:, k] = d[key].numpy()
        links = torch.sigmoid(link_logit).numpy() if link_logit is not None else np.zeros(0)
        return est, links


def score(estimator, episodes):
    acc = {}

    def add(key, num, den):
        a = acc.setdefault(key, [0.0, 0.0])
        a[0] += num
        a[1] += den

    for ep in episodes:
        estimator.reset()
        for t in ep:
            est, links = estimator(t)
            y, m, vis = t['y'], t['ymask'], t['visible']
            for tag, sel in (("all", m), ("coast", m & ~vis)):
                if not sel.any():
                    continue
                n = float(sel.sum())
                add(tag + "/pursuit_acc", float(((est[sel, 0] > 0.5) == (y[sel, 0] > 0.5)).sum()), n)
                ps = sel & (y[:, 0] > 0.5)
                if ps.any():
                    k = float(ps.sum())
                    add(tag + "/turn_mae", float(np.abs(est[ps, 1] - y[ps, 1]).sum()), k)
                    add(tag + "/turn_hi_cover", float((est[ps, 6] + 0.005 >= y[ps, 1]).sum()), k)
                    add(tag + "/turn_hi_excess", float((est[ps, 6] - y[ps, 1]).sum()), k)
                    add(tag + "/bias_mae", float(np.abs(est[ps, 7] - y[ps, 7]).sum()), k)
                    add(tag + "/lead_mae", float(np.abs(est[ps, 2] - y[ps, 2]).sum()), k)
                    add(tag + "/side_acc", float((np.round(est[ps, 3]) == y[ps, 3]).sum()), k)
                    add(tag + "/wobble_mae", float(np.abs(est[ps, 9] - y[ps, 9]).sum()), k)
                    fl = ps & (np.abs(y[:, 3]) > 0.5)
                    if fl.any():
                        add(tag + "/flank_mae", float(np.abs(est[fl, 8] - y[fl, 8]).sum()), float(fl.sum()))
                sm = sel & (y[:, 6] > 0.5)
                if sm.any():
                    add(tag + "/guided_acc", float(((est[sm, 4] > 0.5) == (y[sm, 4] > 0.5)).sum()),
                        float(sm.sum()))
                pos = sel & (y[:, 5] > 0.5)
                if pos.any():
                    add(tag + "/threat_recall", float((est[pos, 5] > 0.5).sum()), float(pos.sum()))
                predpos = sel & (est[:, 5] > 0.5)
                if predpos.any():
                    add(tag + "/threat_precision", float((y[predpos, 5] > 0.5).sum()), float(predpos.sum()))
            lm = t['link_mask']
            if lm.any():
                pred = links[lm] > 0.5
                true = t['link_y'][lm] > 0.5
                add("all/link_acc", float((pred == true).sum()), float(lm.sum()))
                add("all/link_tp", float((pred & true).sum()), 0.0)
                add("all/link_pp", float(pred.sum()), 0.0)
                add("all/link_p", float(true.sum()), 0.0)
    out = {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items() if not k.startswith("all/link_") or k == "all/link_acc"}
    tp, pp, p = (acc.get("all/link_" + s, [0, 0])[0] for s in ("tp", "pp", "p"))
    if p:
        prec, rec = (tp / pp if pp else 0.0), tp / p
        out["all/link_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


# ── world-model error in fresh runs ──────────────────────────
def world_model_error(level, model, episodes=4, frames=1500, sample_every=30, seed=500):
    horizons = (24, 48, 72)
    errs = {k: {h: [] for h in horizons} for k in ("const_vel", "prior", "gnn_mean", "gnn", "oracle")}
    for e in range(episodes):
        rng = random.Random(seed + 97 * e + level.number)
        arena = Arena(level, jet_invulnerable=True, launchers_invulnerable=True, visuals=False,
                      seed=seed + 97 * e + level.number, magazine=False)
        cmd = ScriptedCommander(rng, pressure=rng.uniform(0.015, 0.04), randomize=True)
        pilot = Autopilot(rng)
        tracker = Tracker(level)
        builder = SceneGraphBuilder(level)
        brain = GraphBrain(model)
        snaps = []
        jet_hist = {}
        truth_hist = {}
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
                f0 = f + 1              # the recorded flight it is scored on starts next frame
                gnn = tracker.threat_dicts(f0)
                gnn_mean = [dict(d, turn_rate=tracker.tracks[d['tid']].est['turn_rate']) for d in gnn]
                saved = {tid: t.est for tid, t in tracker.tracks.items()}
                groups = {tid: t.group for tid, t in tracker.tracks.items()}
                margin = tracker.turn_margin
                tracker.turn_margin = 0.0
                for t in tracker.tracks.values():
                    t.est = dict(PRIOR)
                    t.group = -1
                prior = tracker.threat_dicts(f0)
                tracker.turn_margin = margin
                for tid, t in tracker.tracks.items():
                    t.est = saved[tid]
                    t.group = groups[tid]
                truth = {m.uid: m for m in arena.missiles}
                tids = [d['tid'] for d in gnn]
                oracle = [oracle_threat(truth[tid], level.datalink, f0) if tid in truth else None for tid in tids]
                seen_flares = [flare_state(fl, f0) for fl in vis_f if not getattr(fl, 'dead', False)]
                all_flares = [flare_state(fl, f0) for fl in arena.flares if not getattr(fl, 'dead', False)]
                snaps.append((f, tids, gnn, prior, oracle,
                              {tid: complex(truth[tid].x, truth[tid].y) for tid in tids if tid in truth},
                              seen_flares, all_flares, gnn_mean))
        for f, tids, gnn, prior, oracle, now, seen_flares, all_flares, gnn_mean in snaps:
            path = np.array([jet_hist[f + k][0] for k in range(1, 73)])
            vel = np.array([jet_hist[f + k][1] for k in range(1, 73)])
            keep = [i for i, tid in enumerate(tids) if oracle[i] is not None]
            if not keep:
                continue
            preds = {}
            for name, threats, fl in (("prior", [prior[i] for i in keep], seen_flares),
                                      ("gnn_mean", [gnn_mean[i] for i in keep], seen_flares),
                                      ("gnn", [gnn[i] for i in keep], seen_flares),
                                      ("oracle", [oracle[i] for i in keep], all_flares)):
                res = rollout(None, threats, None, None, 72, terrain=level.obstacles,
                              datalink=level.datalink, jet_path=(path, vel), flares=fl)
                preds[name] = (res.threat_x[0], res.threat_y[0], res.alive_path[0])
            for j, i in enumerate(keep):
                tid = tids[i]
                hist = truth_hist.get(tid, {})
                for h in horizons:
                    real = hist.get(f + h)
                    if real is None:
                        continue
                    d0 = gnn[i]
                    cvx = d0['pos'][0] + d0['vel'][0] * h
                    cvy = d0['pos'][1] + d0['vel'][1] * h
                    errs["const_vel"][h].append(math.hypot(cvx - real[0], cvy - real[1]))
                    for name in ("prior", "gnn_mean", "gnn", "oracle"):
                        X, Y, A = preds[name]
                        errs[name][h].append(math.hypot(X[j, h - 1] - real[0], Y[j, h - 1] - real[1]))
    return {k: {h: (float(np.median(v)) if v else None, float(np.mean(v)) if v else None, len(v))
                for h, v in hs.items()} for k, hs in errs.items()}


def run_wm(report):
    model = load_graph_brain("jet_brain_tgn.pt", verbose=False)
    for lv in LEVELS:
        wm = world_model_error(lv, model)
        report['world_model']["L%d" % lv.number] = wm
        print("\nWORLD MODEL ERROR, level %d  (median px / mean px, n)" % lv.number)
        for name in ("const_vel", "prior", "gnn_mean", "gnn", "oracle"):
            print("  %-10s " % name + "   ".join(
                "+%d: %6.1f / %6.1f (n=%d)" % (h, v[0], v[1], v[2]) if v[0] is not None else "+%d: -" % h
                for h, v in sorted(wm[name].items())))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wm", action="store_true")
    ap.add_argument("--only-wm", action="store_true",
                    help="recompute the world-model table only, keeping the stored inference table")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = os.path.join(DATA_DIR, "evaluation.json")

    if args.only_wm:
        with open(out) as fh:
            report = json.load(fh)
        report['world_model'] = {}
        run_wm(report)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        print("\nwrote %s" % out)
        return

    val = load_split("val")
    by_level = {}
    for ep in val:
        smart = any(t['y'][t['ymask'], 6].sum() > 0 for t in ep[:200])
        links = any(len(t['link_rows']) for t in ep[:200])
        by_level.setdefault(3 if links else (2 if smart else 1), []).append(ep)

    estimators = [("rule", RuleEstimator())]
    for name, ck in (("no_edges", "jet_brain_tgn_no_edges.pt"),
                     ("no_memory", "jet_brain_tgn_no_memory.pt"),
                     ("full", "jet_brain_tgn.pt")):
        model = load_graph_brain(ck, verbose=False)
        if model is not None:
            estimators.append((name, ModelEstimator(model)))

    report = {'inference': {}, 'world_model': {}}
    cols = ["all/pursuit_acc", "all/turn_mae", "all/turn_hi_cover", "all/turn_hi_excess",
            "all/bias_mae", "all/lead_mae", "all/side_acc", "all/flank_mae", "all/wobble_mae",
            "all/guided_acc", "coast/guided_acc", "all/threat_recall", "all/threat_precision",
            "all/link_f1"]
    for lv in (1, 2, 3):
        eps = by_level.get(lv, [])
        print("\nLEVEL %d  (%d held-out episodes)" % (lv, len(eps)))
        print("%-10s " % "estimator" + " ".join("%13s" % c.replace("all/", "").replace("coast/", "hidden_") for c in cols))
        for name, est in estimators:
            s = score(est, eps)
            report['inference']["L%d/%s" % (lv, name)] = s
            print("%-10s " % name + " ".join("%13s" % ("-" if s.get(c) is None else "%.3f" % s[c]) for c in cols))
            sys.stdout.flush()

    if not args.skip_wm:
        run_wm(report)

    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
