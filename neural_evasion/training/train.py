"""
Train the v3 temporal graph brain.

Truncated backpropagation through time: sample windows of consecutive ticks from
different episodes, run a no-grad burn-in so the memory is warm before any loss
is taken (as in R2D2), then train on the rest of the window. Windows are merged
into one disjoint graph per tick, so a batch is still a single forward pass.

    python -m neural_evasion.training.train --variant full
    python -m neural_evasion.training.train --variant no_edges   --out jet_brain_tgn_no_edges.pt
    python -m neural_evasion.training.train --variant no_memory  --out jet_brain_tgn_no_memory.pt
"""

import argparse
import glob
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PROJECT_ROOT
from ..model.temporal_gnn import (MAX_BIAS, MAX_FLANK, MAX_TURN, MAX_WOBBLE,
                                  TURN_QUANTILE, TemporalGraphNet, decode_tracks)
from .generate import DATA_DIR

VARIANTS = {
    'full':      dict(use_edges=True, use_memory=True),
    'no_edges':  dict(use_edges=False, use_memory=True),
    'no_memory': dict(use_edges=True, use_memory=False),
}

LOSS_WEIGHTS = {'pursuit': 1.0, 'turn': 4.0, 'turn_hi': 4.0, 'lead': 2.0,
                'side': 1.0, 'side_sign': 1.0, 'guided': 1.0, 'threat': 1.0, 'link': 1.0, 'bias': 2.0,
                'flank': 1.0, 'wobble': 2.0}


def pinball(pred, target, tau):
    """Quantile loss: minimised when `pred` is the tau-quantile of `target`."""
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def load_split(split):
    episodes = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "level*_%s.pkl" % split))):
        with open(path, "rb") as fh:
            episodes.extend(pickle.load(fh))
    return [ep for ep in episodes if len(ep) > 4]


def merge(ticks, banks, zero):
    """Disjoint union of several ticks' graphs, with each node's memory from its window's bank."""
    xs, srcs, dsts, ets, eas, mems = [], [], [], [], [], []
    track_idx, ys, ymasks, vis = [], [], [], []
    link_idx, link_ys, link_masks = [], [], []
    owners = []
    off = 0
    for w, t in enumerate(ticks):
        n = t['x'].shape[0]
        xs.append(t['x'])
        srcs.append(t['src'].astype(np.int64) + off)
        dsts.append(t['dst'].astype(np.int64) + off)
        ets.append(t['etype'])
        eas.append(t['eattr'])
        bank = banks[w]
        mems.extend(bank.get(int(k), zero) for k in t['keys'])
        owners.append((w, t['keys'], off, n))
        track_idx.append(t['track_rows'].astype(np.int64) + off)
        ys.append(t['y'])
        ymasks.append(t['ymask'])
        vis.append(t['visible'])
        e_off = sum(s.shape[0] for s in srcs[:-1])
        link_idx.append(t['link_rows'].astype(np.int64) + e_off)
        link_ys.append(t['link_y'])
        link_masks.append(t['link_mask'])
        off += n
    batch = {
        'x': torch.from_numpy(np.concatenate(xs).astype(np.float32)),
        'src': torch.from_numpy(np.concatenate(srcs)),
        'dst': torch.from_numpy(np.concatenate(dsts)),
        'etype': torch.from_numpy(np.concatenate(ets).astype(np.int64)),
        'eattr': torch.from_numpy(np.concatenate(eas).astype(np.float32)),
        'mem': torch.stack(mems) if mems else torch.zeros(0, zero.shape[0]),
        'track': torch.from_numpy(np.concatenate(track_idx)),
        'y': torch.from_numpy(np.concatenate(ys)),
        'ymask': torch.from_numpy(np.concatenate(ymasks)),
        'visible': torch.from_numpy(np.concatenate(vis)),
        'link': torch.from_numpy(np.concatenate(link_idx)),
        'link_y': torch.from_numpy(np.concatenate(link_ys)),
        'link_mask': torch.from_numpy(np.concatenate(link_masks)),
    }
    return batch, owners


def forward_tick(model, ticks, banks, zero):
    b, owners = merge(ticks, banks, zero)
    z, new_mem, e, _ = model(b['x'], b['src'], b['dst'], b['etype'], b['eattr'], b['mem'])
    for w, keys, off, n in owners:
        banks[w] = {int(k): new_mem[off + i] for i, k in enumerate(keys)}
    raw = model.track_head(z[b['track']]) if b['track'].numel() else None
    link_logit = (model.link_logits(z, e, b['src'], b['dst'], b['link'])
                  if b['link'].numel() else None)
    return b, raw, link_logit


def losses(b, raw, link_logit):
    out = {}
    if raw is not None and b['ymask'].any():
        m = b['ymask'].float()
        y = b['y']
        pursuit_t, turn_t, lead_t, side_t, guided_t, threat_t = (y[:, i] for i in range(6))
        denom = m.sum().clamp(min=1.0)
        out['pursuit'] = (F.binary_cross_entropy_with_logits(raw[:, 0], pursuit_t, reduction='none') * m).sum() / denom
        pm = m * pursuit_t
        pden = pm.sum().clamp(min=1.0)
        turn_p = torch.sigmoid(raw[:, 1])
        out['turn'] = (F.smooth_l1_loss(turn_p, turn_t / MAX_TURN, reduction='none', beta=0.05) * pm).sum() / pden
        # Upper quantile over every track, ballistic included: uncertainty about
        # whether a missile steers at all has to widen it too.
        out['turn_hi'] = (pinball(torch.sigmoid(raw[:, 6]), turn_t / MAX_TURN, TURN_QUANTILE) * m).sum() / denom
        out['bias'] = (F.smooth_l1_loss(torch.tanh(raw[:, 7]), y[:, 7] / MAX_BIAS, reduction='none',
                                        beta=0.1) * m).sum() / denom
        out['lead'] = (F.smooth_l1_loss(torch.sigmoid(raw[:, 2]), lead_t, reduction='none', beta=0.1) * pm).sum() / pden
        # Side is two questions: does the missile fly off to a flank at all, and
        # if so which one. Most do not, so one regression onto {-1, 0, +1}
        # settles on the median, 0, and never names a side.
        fm = pm * (side_t.abs() > 0.5).float()
        out['side'] = (F.binary_cross_entropy_with_logits(raw[:, 10], (side_t.abs() > 0.5).float(),
                                                          reduction='none') * pm).sum() / pden
        out['side_sign'] = (F.binary_cross_entropy_with_logits(raw[:, 3], (side_t > 0).float(), reduction='none')
                            * fm).sum() / fm.sum().clamp(min=1.0)
        out['guided'] = (F.binary_cross_entropy_with_logits(raw[:, 4], guided_t, reduction='none') * m).sum() / denom
        out['threat'] = (F.binary_cross_entropy_with_logits(
            raw[:, 5], threat_t, reduction='none', pos_weight=torch.tensor(3.0)) * m).sum() / denom
        # A flank angle only means something for a missile that has a flank side (fm).
        out['flank'] = (F.smooth_l1_loss(torch.sigmoid(raw[:, 8]), y[:, 8] / MAX_FLANK, reduction='none',
                                         beta=0.05) * fm).sum() / fm.sum().clamp(min=1.0)
        out['wobble'] = (F.smooth_l1_loss(torch.sigmoid(raw[:, 9]), y[:, 9] / MAX_WOBBLE, reduction='none',
                                          beta=0.05) * pm).sum() / pden
    if link_logit is not None and b['link_mask'].any():
        lm = b['link_mask'].float()
        out['link'] = (F.binary_cross_entropy_with_logits(link_logit, b['link_y'], reduction='none') * lm).sum() / lm.sum()
    return out


class Metrics(object):
    def __init__(self):
        self.s = {}

    def add(self, key, value, n=1):
        a = self.s.setdefault(key, [0.0, 0])
        a[0] += float(value)
        a[1] += n

    def mean(self, key):
        a = self.s.get(key)
        return a[0] / a[1] if a and a[1] else float('nan')


@torch.no_grad()
def evaluate(model, episodes, max_ticks=None):
    """Run each episode in order with memory; accumulate accuracy metrics."""
    model.eval()
    zero = torch.zeros(model.hidden)
    met = Metrics()
    for ep in episodes:
        banks = [{}]
        for t in ep[:max_ticks]:
            b, raw, link_logit = forward_tick(model, [t], banks, zero)
            if raw is not None and b['ymask'].any():
                est = decode_tracks(raw)
                y, m = b['y'], b['ymask']
                for tag, sel in (("all", m), ("coast", m & ~b['visible'])):
                    if not sel.any():
                        continue
                    n = int(sel.sum())
                    pursuit_t = y[sel, 0]
                    met.add(tag + "/pursuit_acc", ((est['pursuit'][sel] > 0.5).float() == pursuit_t).float().sum(), n)
                    ps = sel & (y[:, 0] > 0.5)
                    if ps.any():
                        k = int(ps.sum())
                        met.add(tag + "/turn_mae", (est['turn_rate'][ps] - y[ps, 1]).abs().sum(), k)
                        met.add(tag + "/turn_hi_cover", (est['turn_hi'][ps] + 0.005 >= y[ps, 1]).float().sum(), k)
                        met.add(tag + "/bias_mae", (est['bias'][ps] - y[ps, 7]).abs().sum(), k)
                        met.add(tag + "/lead_mae", (est['lead'][ps] - y[ps, 2]).abs().sum(), k)
                        met.add(tag + "/side_acc", (torch.round(est['side'][ps]) == y[ps, 3]).float().sum(), k)
                        met.add(tag + "/wobble_mae", (est['wobble'][ps] - y[ps, 9]).abs().sum(), k)
                        fs = ps & (y[:, 3].abs() > 0.5)
                        if fs.any():
                            met.add(tag + "/flank_mae", (est['flank'][fs] - y[fs, 8]).abs().sum(), int(fs.sum()))
                    sm = sel & (y[:, 6] > 0.5)
                    if sm.any():
                        met.add(tag + "/guided_acc", ((est['guided'][sm] > 0.5).float() == y[sm, 4]).float().sum(), int(sm.sum()))
                    met.add(tag + "/threat_acc", ((est['threat'][sel] > 0.5).float() == y[sel, 5]).float().sum(), n)
                    pos = sel & (y[:, 5] > 0.5)
                    if pos.any():
                        met.add(tag + "/threat_recall", (est['threat'][pos] > 0.5).float().sum(), int(pos.sum()))
            if link_logit is not None and b['link_mask'].any():
                lm = b['link_mask']
                pred = (torch.sigmoid(link_logit[lm]) > 0.5).float()
                met.add("all/link_acc", (pred == b['link_y'][lm]).float().sum(), int(lm.sum()))
                tp = ((pred > 0.5) & (b['link_y'][lm] > 0.5)).float().sum()
                met.add("all/link_tp", tp, 0)
                met.add("all/link_pos", (b['link_y'][lm] > 0.5).float().sum(), 0)
                met.add("all/link_predpos", (pred > 0.5).float().sum(), 0)
    model.train()
    return met


def summarize(met):
    keys = ["pursuit_acc", "turn_mae", "turn_hi_cover", "bias_mae", "lead_mae",
            "side_acc", "flank_mae", "wobble_mae", "guided_acc", "threat_acc", "threat_recall"]
    parts = []
    for k in keys:
        parts.append("%s %.3f" % (k, met.mean("all/" + k)))
    tp = met.s.get("all/link_tp", [0, 0])[0]
    pos = met.s.get("all/link_pos", [0, 0])[0]
    pp = met.s.get("all/link_predpos", [0, 0])[0]
    if pos:
        prec = tp / pp if pp else 0.0
        rec = tp / pos
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        parts.append("link_acc %.3f link_f1 %.3f" % (met.mean("all/link_acc"), f1))
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--burn", type=int, default=6)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="jet_brain_tgn.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_eps = load_split("train")
    val_eps = load_split("val")
    print("train episodes %d (%d ticks), val episodes %d"
          % (len(train_eps), sum(len(e) for e in train_eps), len(val_eps)))

    model = TemporalGraphNet(**VARIANTS[args.variant])
    print("variant %s, %d parameters" % (args.variant, sum(p.numel() for p in model.parameters())))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    zero = torch.zeros(model.hidden)
    span = args.burn + args.window
    out_path = os.path.join(PROJECT_ROOT, args.out)

    best, t0 = None, time.time()
    running = {}
    for step in range(1, args.steps + 1):
        windows = []
        for _ in range(args.batch):
            ep = random.choice(train_eps)
            start = random.randint(0, max(0, len(ep) - span))
            windows.append(ep[start:start + span])
        banks = [{} for _ in windows]

        with torch.no_grad():
            for t in range(args.burn):
                forward_tick(model, [w[t] for w in windows if t < len(w)], banks, zero)
        for i in range(len(banks)):
            banks[i] = {k: v.detach() for k, v in banks[i].items()}

        total = 0.0
        for t in range(args.burn, span):
            active = [i for i, w in enumerate(windows) if t < len(w)]
            if not active:
                break
            sub_banks = [banks[i] for i in active]
            b, raw, link_logit = forward_tick(model, [windows[i][t] for i in active], sub_banks, zero)
            for j, i in enumerate(active):
                banks[i] = sub_banks[j]
            parts = losses(b, raw, link_logit)
            for k, v in parts.items():
                total = total + LOSS_WEIGHTS[k] * v
                fv = float(v.detach())
                running[k] = 0.98 * running.get(k, fv) + 0.02 * fv
        total = total / args.window
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print("step %d  %.0fs  %s" % (step, time.time() - t0,
                                         "  ".join("%s %.3f" % kv for kv in sorted(running.items()))))
            sys.stdout.flush()
        if step % 500 == 0 or step == args.steps:
            met = evaluate(model, val_eps, max_ticks=300)
            wobble = met.mean("all/wobble_mae")
            score = (met.mean("all/turn_mae") * 20 + (1 - met.mean("all/guided_acc"))
                     + (1 - met.mean("all/pursuit_acc")) + met.mean("all/lead_mae")
                     + (1 - met.mean("all/side_acc"))
                     + abs(met.mean("all/turn_hi_cover") - 0.9)
                     + (0.0 if math.isnan(wobble) else wobble / 0.2))
            print("VAL step %d: %s" % (step, summarize(met)))
            if best is None or score < best:
                best = score
                torch.save({'config': model.config(), 'state_dict': model.state_dict(),
                            'variant': args.variant, 'step': step,
                            'val': summarize(met)}, out_path)
                print("saved %s (score %.4f)" % (out_path, score))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
