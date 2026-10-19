"""
Merge closed-loop benchmark result files into one table with 95% intervals.

Later files override earlier ones for the same (level, pilot, seed), so an
extension run can be listed after the run it extends.

    python -m neural_evasion.training.report training_data_v3/bench_L123_1_2.json training_data_v3/bench_final_*.json
"""

import argparse
import glob
import json
import math

ORDER = ["1", "2", "3", "3-no_edges", "3-no_memory", "3-prior", "3-oracle",
         "3-cal", "3-no_edges-cal", "3-no_memory-cal"]
NAMES = {
    "1": "V1 (goated)",
    "2": "V2 (soft maneuvers)",
    "3": "V3 (dynamic graph)",
    "3-no_edges": "V3, GNN without message passing",
    "3-no_memory": "V3, GNN without memory",
    "3-prior": "V3, no GNN (tracker priors)",
    "3-oracle": "V3, true missile values (ceiling)",
    "3-cal": "V3, conformally calibrated turn bound",
    "3-no_edges-cal": "V3 no message passing, calibrated bound",
    "3-no_memory-cal": "V3 no memory, calibrated bound",
}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load(patterns):
    runs = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                for r in json.load(fh):
                    runs[(r['level'], r['pilot'], r['seed'])] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    runs = load(args.files)
    pilots = ORDER + sorted({p for _, p, _ in runs} - set(ORDER))

    print("| level | pilot | survived | 95% interval | mean time alive | deaths | flares per run | ms/frame |")
    print("|---|---|---|---|---|---|---|---|")
    for ln in sorted({l for l, _, _ in runs}):
        for spec in pilots:
            rs = [r for (l, p, _), r in sorted(runs.items()) if l == ln and p == spec]
            if not rs:
                continue
            n = len(rs)
            k = sum(1 for r in rs if r['survived'])
            lo, hi = wilson(k, n)
            causes = {}
            for r in rs:
                if not r['survived']:
                    causes[r['cause']] = causes.get(r['cause'], 0) + 1
            print("| L%d | %s | %d/%d | %.0f-%.0f%% | %.1f s | %s | %.0f | %.1f |"
                  % (ln, NAMES.get(spec, spec), k, n, 100 * lo, 100 * hi,
                     sum(r['frames'] for r in rs) / n / 60.0,
                     ", ".join("%s %d" % kv for kv in sorted(causes.items())) or "-",
                     sum(r['flares'] for r in rs) / float(n),
                     sum(r['ms_per_frame'] for r in rs) / n))


if __name__ == "__main__":
    main()
