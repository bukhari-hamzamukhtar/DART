"""
crosstest/clevrer_traj.py  --  the V3 kernel on CLEVRER motion itself, with CausalVis's
                               own trajectory-level protocols (VAL clips only)
======================================================================================

CausalVis measured that its counterfactual benchmark barely moves when the physics
gets better. So the benchmark alone cannot say whether a simulator understands
CLEVRER. These three checks can, and each is run exactly as CausalVis runs it:

  A  FREE MOTION (probe/physics_laws.py part A): from a true state, 20 and 40 frames
     ahead, on segments with no collision. CausalVis's rows (constant velocity, fitted
     damping, fitted friction, the per-kind law choice, the learned v6_voxel) are
     produced by their code in this run; the V3 rows by the V3 kernel.
  C  HEAD-TO-HEAD AT REAL COLLISIONS (physics_laws part C): both objects start 8
     frames before an isolated annotated collision and each engine runs 16 frames
     alone; outgoing direction error against the recording, and whether contact is
     made. CausalVis's law engine and learned model come from their part_bc in this run.
  R  SHADOW REPLAY (probe/shadow_replay.py): every annotated collision in the first 200
     VAL clips, replayed from 12 and 30 frames before with nothing following the
     recording; reproduced = the pair meets the benchmark's check within +-5 frames.
     V3 engines run through shadow_replay.simulate; v7 and v8 are re-scored from
     CausalVis's stored records (probe/shadow_v7.jsonl, shadow_v8.jsonl).

    C:/Python314/python crosstest/clevrer_traj.py
"""

import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
CV = r"G:/CausalVis"
sys.path.insert(0, os.path.join(DNN, "crosstest"))
import clevrer_v3 as C3                                   # noqa: E402
import fit_v3_clevrer as FIT                              # noqa: E402
sys.path.insert(0, os.path.join(CV, "probe"))
import physics_laws as PL                                 # noqa: E402
import shadow_replay as SR                                # noqa: E402
from world import load_spatial                            # noqa: E402

OUT = os.path.join(DNN, "crosstest", "results", "clevrer_traj.json")
REFIT = json.load(open(os.path.join(DNN, "crosstest", "v3_clevrer_fit.json")))
ENGINES = {
    "v3_kinematic": {"repel": False, "dist": C3.REPULSE_DIST_GAME, "force": C3.REPULSE_FORCE_GAME, "drag": {}},
    "v3_swarm": {"repel": True, "dist": C3.REPULSE_DIST_GAME, "force": C3.REPULSE_FORCE_GAME, "drag": {}},
    "v3_refit": {"repel": True, "dist": REFIT["repulse_dist_px"], "force": REFIT["repulse_force"],
                 "drag": REFIT["drag"]},
}


def v3_free_motion(P0, V0, G, eng, H=40):
    """Free segments rolled by the kernel, one object at a time (no neighbours: free motion)."""
    pred = np.empty((len(P0), H, 2))
    for s in range(len(P0)):
        P = P0[s:s + 1].astype(np.float64)
        V = V0[s:s + 1].astype(np.float64)
        keep = 1.0 - eng["drag"].get(G[s], 0.0)
        for t in range(H):
            V = V * keep
            x1, v1 = C3.kernel_frame(P * C3.PX + C3.OFFSET, V * C3.PX, eng["repel"], eng["dist"], eng["force"])
            P = (x1 - C3.OFFSET) / C3.PX
            V = v1 / C3.PX
            pred[s, t] = P[0]
    return pred


def part_a(model, report):
    print("\n" + "=" * 90 + "\nA. FREE MOTION, VAL (CausalVis's protocol)\n" + "=" * 90, flush=True)
    laws = PL.part_a(model)                               # prints CausalVis's rows
    P0, V0, F, G, _ = PL.free_segments(PL.VAL)
    rows = {}
    for name, eng in ENGINES.items():
        t0 = time.time()
        pred = v3_free_motion(P0, V0, G, eng)
        e20, e40 = PL.err(pred, F, 20).mean(), PL.err(pred, F, 40).mean()
        rows[name] = {"err20": float(e20), "err40": float(e40)}
        print("      %-30s  %.5f      %.5f   (V3 kernel, %.0fs)" % (name, e20, e40, time.time() - t0), flush=True)
    report["A_free_motion"] = {"segments": int(len(P0)), "v3": rows}
    return laws


def part_c(model, laws, report):
    print("\n" + "=" * 90 + "\nB/C. COLLISIONS, VAL (CausalVis's protocol)\n" + "=" * 90, flush=True)
    PL.part_bc(model, laws)                               # prints CausalVis's law engine and learned rows
    rows = PL.collisions(PL.VAL)
    out = {}
    print("   %-46s contact made   direction error median   p75" % "")
    for name, eng in ENGINES.items():
        errs, hits = [], 0
        for r in rows:
            e, h = FIT.head_to_head(r, eng["drag"], eng["dist"], eng["force"], repel=eng["repel"])
            errs += e
            hits += h
        errs = np.array(errs)
        out[name] = {"contact": hits / len(rows), "median_deg": float(np.median(errs)),
                     "p75_deg": float(np.percentile(errs, 75)), "n_collisions": len(rows)}
        print("   %-46s %5.1f%%        %6.1f°               %6.1f°"
              % (name + " (V3 kernel)", 100 * hits / len(rows), np.median(errs), np.percentile(errs, 75)))
    report["C_head_to_head"] = out


def summarise(recs, L):
    sub = [r for r in recs if r["L"] == L]
    rep = sum(r["reproduced"] for r in sub)
    causes = Counter(r["cause"] for r in sub if not r["reproduced"])
    return {"n": len(sub), "reproduced": rep / max(len(sub), 1),
            "seen_on_recording": sum(r["real_seen"] for r in sub) / max(len(sub), 1),
            "miss_causes": dict(causes.most_common())}


def replay(model, eng_name, eng, clips=200, lookbacks=(12, 30)):
    """shadow_replay.main's loop, with the V3 kernel as the engine, results kept here."""
    SR.V8.make_law_context = C3.make_v3_context
    SR.V8.law_step = C3.v3_step
    laws = {"variant": "refit" if eng_name == "v3_refit" else ("swarm" if eng["repel"] else "kinematic"),
            "repulse_dist_px": eng["dist"], "repulse_force": eng["force"], "drag": eng["drag"]}
    files = {SR._vid_num(f): f for f in glob.glob(os.path.join(CV, "data", "trajectories_3d_yaw", "*.npz"))}
    val = sorted(k for k in SR.load_split_set(os.path.join(CV, "split3d_yaw.json"), "val") if k in files)[:clips]
    recs = []
    for v in val:
        z = np.load(files[v], allow_pickle=True)
        pos = z["positions"].astype(np.float32); pres = z["presence"]; attrs = z["attrs"]
        T = pos.shape[0]; N = min(8, pos.shape[1]); rad = attrs[:N, 15]
        real = np.where((pres[:, :N] > 0)[..., None], pos[:, :N], np.nan)
        cols = sorted((int(f), int(i), int(j)) for f, i, j in z["collisions"] if int(i) < N and int(j) < N)
        enter = {k: int(np.argmax(pres[:, k] > 0)) for k in range(N) if (pres[:, k] > 0).any()}
        for f, i, j in cols:
            for L in lookbacks:
                s, end = f - L, f + 5
                if s < 0 or end >= T:
                    continue
                sim = SR.simulate("v8", model, laws, z, s, end)
                ok = SR.touched(sim, i, j, f - 5, f + 5, rad)
                rec = {"video": v, "f": f, "i": i, "j": j, "L": L, "reproduced": bool(ok),
                       "real_seen": bool(SR.touched(real, i, j, f - 5, f + 5, rad))}
                if not ok:
                    involved = {i, j} | {x for g, x, y in cols if s < g <= f and y in (i, j)} \
                                      | {y for g, x, y in cols if s < g <= f and x in (i, j)}
                    first = None
                    for t in range(s, end + 1):
                        for k in involved:
                            if np.isfinite(sim[t, k]).all() and np.isfinite(real[t, k]).all() \
                                    and np.linalg.norm(sim[t, k] - real[t, k]) > SR.DEV:
                                first = (t, k)
                                break
                        if first:
                            break
                    if first is None:
                        cause = "never diverged"
                    else:
                        t_, k_ = first
                        near = [(g, x, y) for g, x, y in cols if t_ - 8 <= g <= t_ and k_ in (x, y)]
                        if near:
                            g, x, y = near[-1]
                            cause = "wrong bounce" if SR.touched(sim, x, y, g - 5, g + 5, rad) else "missed collision"
                        elif enter.get(k_, -99) >= t_ - 10:
                            cause = "bad entry"
                        elif any(SR.touched(sim, k_, m, t_ - 8, t_, rad) and not SR.touched(real, k_, m, t_ - 8, t_, rad)
                                 for m in range(N) if m != k_):
                            cause = "invented contact"
                        else:
                            cause = "slow drift"
                    rec["cause"] = cause
                recs.append(rec)
    return recs


def part_r(model, report):
    print("\n" + "=" * 90 + "\nR. SHADOW REPLAY, first 200 VAL clips (CausalVis's protocol)\n" + "=" * 90, flush=True)
    out = {}
    for name in ("v7", "v8"):
        path = os.path.join(CV, "probe", "shadow_%s.jsonl" % name)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        out[name + " (CausalVis record)"] = {L: summarise(recs, L) for L in (12, 30)}
    for name, eng in ENGINES.items():
        t0 = time.time()
        out[name] = {L: summarise(replay(model, name, eng), L) for L in (12, 30)}
        print("   %s replayed (%.0fs)" % (name, time.time() - t0), flush=True)
    print("\n   %-26s %22s %22s   top miss causes at L=30" % ("engine", "reproduced from 12", "reproduced from 30"))
    for name, s in out.items():
        top = ", ".join("%s %d" % kv for kv in list(s[30]["miss_causes"].items())[:3])
        print("   %-26s %9.1f%% (n=%4d)   %9.1f%% (n=%4d)   %s"
              % (name, 100 * s[12]["reproduced"], s[12]["n"], 100 * s[30]["reproduced"], s[30]["n"], top))
    report["R_shadow_replay"] = out


def main():
    model = load_spatial(os.path.join(CV, "v6_voxel.pt"), yaw_known=False)
    report = {"refit_constants": {k: REFIT[k] for k in ("drag", "repulse_dist_px", "repulse_force")}}
    laws = part_a(model, report)
    part_c(model, laws, report)
    part_r(model, report)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
