"""
crosstest/clevrer_compare.py  --  CLEVRER counterfactual TEST: V3 simulators against
                                  CausalVis's own systems, paired
===================================================================================

Every system is scored from its TEST records with v7/report.py's scoring, at the
detector and extension its VAL run chose (CausalVis's rule; the V3 settings are
recomputed from the VAL records here, not typed in). TEST was run once per system.

Paired comparisons use McNemar's z on the options both systems scored.

    C:/Python314/python crosstest/clevrer_compare.py
"""

import glob
import json
import math
import os
import sys

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
CV = r"G:/CausalVis"
sys.path.insert(0, os.path.join(CV, "v7"))
import report as R                                        # noqa: E402  (CausalVis's scoring)

RUNS = os.path.join(DNN, "crosstest", "runs", "clevrer")
OUT = os.path.join(DNN, "crosstest", "results", "clevrer_test.json")


def load(d):
    """Every record, as v7/report.py scores them (a handful of options appear twice in
    the question file and are scored twice there too), and the same keyed for pairing."""
    recs = [json.loads(line) for f in sorted(glob.glob(os.path.join(d, "shard*.jsonl")))
            for line in open(f, encoding="utf-8")]
    return recs, {(r["video"], str(r["qid"]), r["choice"]): r for r in recs}


def val_choice(d):
    recs = load(d)[0]
    best = None
    for det, tol in R.DETECTORS:
        for e in R.EXTENDS:
            ok, n, qok, qn = R.score(recs, det, tol, e)
            if best is None or ok > best[0]:
                best = (ok, n, det, tol, e)
    return best


def right(r, det, tol, ext):
    return ("correct" if R.happens(r, det, tol, ext) != r["negate"] else "wrong") == r["truth"]


def simulated(r):
    return not r["participant_removed"] and any(x is not None and x[1] != "video_ended"
                                                for x in (r["taint_i"], r["taint_j"]))


def main():
    systems = [
        ("no simulation: delete the object, keep the recording", os.path.join(CV, "v7", "runs", "test_nosim"),
         ("cal", None, 0)),
        ("CausalVis v7: learned world model (v6_voxel)", os.path.join(CV, "v7", "runs", "test_A"), ("cal", None, 30)),
        ("CausalVis v8: textbook laws fitted on TRAIN", os.path.join(CV, "v8", "runs", "test"), ("cal", None, 30)),
    ]
    choices = {}
    for v in ("kinematic", "swarm", "refit"):
        vd, td = os.path.join(RUNS, v + "_val"), os.path.join(RUNS, v + "_test")
        if not glob.glob(os.path.join(td, "shard*.jsonl")):
            continue
        ok, n, det, tol, e = val_choice(vd)
        choices[v] = {"det": det, "ext": e, "val_ok": ok, "val_n": n}
        systems.append(("V3 %s (setting chosen on VAL: %s +%d, VAL %d/%d)" % (v, det, e, ok, n), td, (det, tol, e)))

    loaded = [(name,) + load(d) + (s,) for name, d, s in systems]
    report = {"val_choices": choices, "systems": {}, "paired": {}}
    print("%-72s %16s %16s %16s %7s %8s" % ("CLEVRER counterfactual TEST", "per option", "per question",
                                           "simulated pairs", "missed", "invented"))
    for name, rs, _, (det, tol, e) in loaded:
        ok, n, qok, qn = R.score(rs, det, tol, e)
        sim = [r for r in rs if simulated(r)]
        sok = sum(right(r, det, tol, e) for r in sim)
        missed = sum(1 for r in rs if r["truth_collides"] and not R.happens(r, det, tol, e))
        invented = sum(1 for r in rs if not r["truth_collides"] and R.happens(r, det, tol, e))
        print("%-72s %6.1f%% (%4d)  %6.1f%% (%3d)  %6.1f%% (%4d) %7d %8d"
              % (name, 100 * ok / n, ok, 100 * qok / qn, qok, 100 * sok / max(len(sim), 1), len(sim), missed, invented))
        report["systems"][name] = {"options_ok": ok, "options": n, "questions_ok": qok, "questions": qn,
                                   "simulated_ok": sok, "simulated": len(sim), "missed": missed, "invented": invented,
                                   "setting": [det, e]}
    print("\npaired, McNemar on matched options (A right & B wrong vs B right & A wrong)")
    for na, _, ra, sa in loaded:
        if not na.startswith("V3"):
            continue
        for nb, _, rb, sb in loaded:
            if nb == na:
                continue
            keys = [k for k in ra if k in rb]
            a_only = sum(right(ra[k], *sa) and not right(rb[k], *sb) for k in keys)
            b_only = sum(right(rb[k], *sb) and not right(ra[k], *sa) for k in keys)
            z = (a_only - b_only) / math.sqrt(a_only + b_only) if a_only + b_only else 0.0
            print("   %-40s vs %-52s matched %4d   %3d vs %3d   z = %+.2f" % (na[:40], nb[:52], len(keys), a_only, b_only, z))
            report["paired"]["%s | %s" % (na, nb)] = {"matched": len(keys), "a_only": a_only, "b_only": b_only, "z": z}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
