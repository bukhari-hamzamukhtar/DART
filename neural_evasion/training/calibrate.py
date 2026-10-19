"""
Conformal calibration of the turn-rate upper quantile.

The planner flies against the network's `turn_hi` head, trained with a pinball
loss at tau = 0.9. On held-out episodes it covers the true turn rate 95-97% of
the time, and the misses are the dangerous case: an agile missile that has not
yet had to turn hard, so nothing it has done reveals its limit. In closed loop
those misses are how the trained v3 dies on level 1 while the oracle does not.

Conformalized quantile regression (Romano, Patterson and Candès, NeurIPS 2019)
fixes the coverage without retraining. On held-out data, take the residual
(true turn rate - turn_hi) for every guided track at every tick, and add the
residuals' (1 - alpha) conformal quantile to the head's output. Done per level
and per checkpoint, and written beside the checkpoint as
`<checkpoint>.calib.json`, which `load_graph_brain` picks up.

Stated plainly, two caveats. Consecutive ticks of one track are strongly
correlated, so the data are not exchangeable and the target coverage is
approximate, not a finite-sample guarantee. And these held-out episodes also
selected the checkpoint during training.

    python -m neural_evasion.training.calibrate --alpha 0.01
"""

import argparse
import json
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np   # noqa: E402
import torch         # noqa: E402

from ..model.temporal_gnn import checkpoint_path, load_graph_brain  # noqa: E402
from .evaluate import ModelEstimator                                 # noqa: E402
from .train import load_split                                        # noqa: E402

CHECKPOINTS = ("jet_brain_tgn.pt", "jet_brain_tgn_no_edges.pt", "jet_brain_tgn_no_memory.pt")


def split_by_level(episodes):
    """Same rule as evaluate.main: datalinks mean level 3, seekers level 2, else level 1."""
    by_level = {}
    for ep in episodes:
        smart = any(t['y'][t['ymask'], 6].sum() > 0 for t in ep[:200])
        links = any(len(t['link_rows']) for t in ep[:200])
        by_level.setdefault(3 if links else (2 if smart else 1), []).append(ep)
    return by_level


def turn_residuals(estimator, episodes):
    """true turn rate - turn_hi, for every guided track at every held-out tick."""
    out = []
    for ep in episodes:
        estimator.reset()
        for t in ep:
            est, _ = estimator(t)
            y, m = t['y'], t['ymask']
            guided = m & (y[:, 0] > 0.5)
            if guided.any():
                out.append(y[guided, 1] - est[guided, 6])
    return np.concatenate(out) if out else np.zeros(0)


def conformal_quantile(residuals, alpha):
    """The ceil((n+1)(1-alpha))-th smallest residual (split-conformal rule)."""
    n = residuals.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    s = np.sort(residuals)
    return float(s[min(k, n) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    torch.set_num_threads(2)

    by_level = split_by_level(load_split("val"))
    for ck in CHECKPOINTS:
        path = checkpoint_path(ck)
        model = load_graph_brain(ck, verbose=False)
        if model is None:
            print("skip %s: no checkpoint" % ck)
            continue
        est = ModelEstimator(model)
        report = {'alpha': args.alpha, 'method': 'conformalized quantile regression on turn_hi',
                  'levels': {}}
        for lv in sorted(by_level):
            r = turn_residuals(est, by_level[lv])
            margin = max(0.0, conformal_quantile(r, args.alpha))
            report['levels'][str(lv)] = {
                'margin': margin, 'n': int(r.size), 'episodes': len(by_level[lv]),
                'cover_raw': float((r <= 0.0).mean()), 'cover_calibrated': float((r <= margin).mean()),
                'residual_p50': float(np.median(r)), 'residual_max': float(r.max())}
            print("%-28s L%d  n=%6d  coverage %.3f -> %.3f  margin +%.4f rad/frame  (max miss %.4f)"
                  % (ck, lv, r.size, (r <= 0.0).mean(), (r <= margin).mean(), margin, r.max()))
        out = os.path.splitext(path)[0] + ".calib.json"
        with open(out, "w") as fh:
            json.dump(report, fh, indent=1)
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
