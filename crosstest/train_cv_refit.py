"""
crosstest/train_cv_refit.py  --  fine-tune CausalVis's world model on game trajectories
=====================================================================================

Zero-shot, a potential learned on tabletop collisions has no reason to know that a
missile turns toward a jet. This gives the model the same chance V3's temporal GNN
had: training on the game's own data (crosstest/record_game.py).

Kept from CausalVis, unchanged: the model class and its initial weights (v6_voxel.pt),
radius frozen at the measured value, v3_6's legacy objective (position MSE + 0.1 x
velocity MSE over padded slots), the 3 -> 20 frame horizon curriculum, AdamW with
gradient clipping at 1.0, lr 3e-4 (their v6 fine-tune), best checkpoint by
validation rollout error at 20 frames.

How the scene is posed is the same as in the planner (crosstest/cv_bridge.py): the
jet is an external body moved along its recorded path every step, terrain peaks are
pinned, and only missiles are free and only missiles are in the loss. The attribute
encoder sees each missile kind as its own CLEVRER kind (cv_bridge.MAPPINGS["refit"]).

Cost. Every force is an autograd derivative of V over every pair of slots, and
training differentiates through it again, so the cost grows with slots squared.
Each batch therefore comes from one level (no empty peak slots on open sky) and
holds at most --max-missiles missiles per window (a random subset beyond).

    python crosstest/train_cv_refit.py --epochs 22 --batches 60 --out crosstest/cv_refit.pt
"""

import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
sys.path.insert(0, DNN)
sys.path.insert(0, os.path.join(DNN, "crosstest"))

from cv_bridge import CHECKPOINT, CX, CY, KINDS, MAPPINGS, MISSILE_RADIUS, UNIT, attrs_row  # noqa: E402
from neural_evasion.sim.worldmodel import JET_RADIUS                      # noqa: E402
from world import load_spatial, strip_colour                              # noqa: E402

DATA = os.path.join(DNN, "crosstest", "data")
MAP = MAPPINGS["refit"]
MMAX = 8                      # set from --max-missiles


def to_u(xy):
    xy = np.asarray(xy, np.float32)
    return np.stack([(xy[..., 0] - CX) * UNIT, (xy[..., 1] - CY) * UNIT], axis=-1).astype(np.float32)


def load(split):
    """{level: [episodes]}"""
    out = {}
    for ln in (1, 2, 3):
        path = os.path.join(DATA, "game_L%d_%s.pkl" % (ln, split))
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                out[ln] = pickle.load(fh)
    return out


def sample_window(eps, H, rng):
    while True:
        ep = rng.choice(eps)
        t0 = rng.randint(1, ep["frames"] - H - 2)
        live = [tr for tr in ep["tracks"] if tr["t0"] <= t0 and tr["t0"] + len(tr["pos"]) - 1 >= t0 + H]
        if live:
            break
    if len(live) > MMAX:
        live = rng.sample(live, MMAX)
    K = len(ep["peaks"])
    N = MMAX + 1 + K
    attrs = np.zeros((N, 16), np.float32)
    act = np.zeros(N, np.float32)
    ext = np.zeros(N, np.float32)
    q0 = np.zeros((N, 2), np.float32)
    v0 = np.zeros((N, 2), np.float32)
    qf = np.zeros((H, MMAX, 2), np.float32)
    vf = np.zeros((H, MMAX, 2), np.float32)
    mask = np.zeros(MMAX, np.float32)
    for i, tr in enumerate(live):
        k = t0 - tr["t0"]
        attrs[i] = attrs_row(*MAP[KINDS[tr["kind"]]], MISSILE_RADIUS)
        act[i] = mask[i] = 1.0
        q0[i] = to_u(tr["pos"][k])
        v0[i] = tr["vel"][k] * UNIT
        qf[:, i] = to_u(tr["pos"][k + 1:k + 1 + H])
        vf[:, i] = tr["vel"][k + 1:k + 1 + H] * UNIT
    attrs[MMAX] = attrs_row(*MAP["jet"], JET_RADIUS)
    act[MMAX] = ext[MMAX] = 1.0
    q0[MMAX] = to_u(ep["jet"][t0, :2])
    v0[MMAX] = ep["jet"][t0, 2:] * UNIT
    jp = to_u(ep["jet"][t0 + 1:t0 + 1 + H, :2])
    for p, (ox, oy, r) in enumerate(ep["peaks"]):
        s = MMAX + 1 + p
        attrs[s] = attrs_row(*MAP["peak"], r)
        act[s] = ext[s] = 1.0
        q0[s] = to_u((ox, oy))
    return attrs, act, ext, q0, v0, jp, qf, vf, mask


def batch(by_level, H, B, rng, level=None):
    ln = level if level is not None else rng.choice(sorted(by_level))
    cols = list(zip(*[sample_window(by_level[ln], H, rng) for _ in range(B)]))
    return [torch.from_numpy(np.stack(c)) for c in cols]


def rollout(model, attrs, act, ext, q0, v0, jp, H, create_graph):
    """SpatialWorld's step with the jet prescribed and peaks pinned (as in cv_bridge)."""
    phys = strip_colour(attrs)
    mass, radius, e = model.properties(phys)
    c = model.drag_coeff(e, act)
    dec = torch.exp(-0.5 * c).unsqueeze(-1)
    m_ = mass.unsqueeze(-1)
    ext3 = ext.unsqueeze(-1)
    q = q0.clone()
    p = m_ * v0 * act.unsqueeze(-1)
    qs, vs = [], []
    for k in range(H):
        p = p * dec
        F, _ = model.forces_torques(q, None, mass, radius, e, act, phys, create_graph)
        p = p + 0.5 * F
        q = q + p / m_
        target = q0.detach().clone()
        target[:, MMAX] = jp[:, k]
        q = q * (1 - ext3) + target * ext3
        F, _ = model.forces_torques(q, None, mass, radius, e, act, phys, create_graph)
        p = (p + 0.5 * F) * dec
        p = p * (1 - ext3)
        qs.append(q[:, :MMAX])
        vs.append(p[:, :MMAX] / m_[:, :MMAX])
    return torch.stack(qs, 1), torch.stack(vs, 1)


def val_error(model, val_batches, H=20, free=False):
    """Mean distance at frame H over every missile in the fixed validation windows, px."""
    model.eval()
    tot, n = 0.0, 0
    for attrs, act, ext, q0, v0, jp, qf, vf, mask in val_batches:
        if free:
            qs = q0[:, None, :MMAX] + v0[:, None, :MMAX] * torch.arange(1, H + 1).view(1, H, 1, 1)
        else:
            qs, _ = rollout(model, attrs, act, ext, q0, v0, jp, H, create_graph=False)
        d = (qs[:, H - 1] - qf[:, H - 1]).norm(dim=-1) / UNIT
        tot += float((d * mask).sum().detach())
        n += int(mask.sum())
    model.train()
    return tot / max(n, 1)


def main():
    global MMAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batches", type=int, default=60, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-missiles", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val-batches", type=int, default=8, help="per level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DNN, "crosstest", "cv_refit.pt"))
    a = ap.parse_args()
    MMAX = a.max_missiles
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train_eps, val_eps = load("train"), load("val")
    print("train episodes %s, val episodes %s, up to %d missiles per window, batch %d"
          % ({k: len(v) for k, v in train_eps.items()}, {k: len(v) for k, v in val_eps.items()}, MMAX, a.batch))
    vrng = random.Random(12345)
    val_batches = [batch(val_eps, 20, a.batch, vrng, level=ln) for ln in sorted(val_eps) for _ in range(a.val_batches)]

    model = load_spatial(CHECKPOINT, yaw_known=False)
    model.freeze_radius = True
    for p_ in model.radius_corr.parameters():
        torch.nn.init.zeros_(p_)
        p_.requires_grad_(False)
    frozen = {id(p_) for p_ in model.radius_corr.parameters()}
    trainable = [p_ for p_ in model.parameters() if id(p_) not in frozen]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    straight = val_error(model, val_batches, free=True)
    best = val_error(model, val_batches)
    print("before fine-tuning (v6_voxel.pt, refit attributes): val err@20 %.2f px   straight lines %.2f px"
          % (best, straight), flush=True)
    torch.save(model.state_dict(), a.out)
    t_start = time.time()
    for ep in range(a.epochs):
        H = min(3 + ep, 20)
        tot, n, skipped, t0 = 0.0, 0, 0, time.time()
        for _ in range(a.batches):
            attrs, act, ext, q0, v0, jp, qf, vf, mask = batch(train_eps, H, a.batch, rng)
            qs, vs = rollout(model, attrs, act, ext, q0, v0, jp, H, create_graph=True)
            m = mask[:, None, :, None]
            loss = (((qs - qf) * m) ** 2).mean() + 0.1 * (((vs - vf) * m) ** 2).mean()
            if not torch.isfinite(loss):
                opt.zero_grad()
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        ve = val_error(model, val_batches)
        tag = ""
        if ve < best:
            best = ve
            torch.save(model.state_dict(), a.out)
            tag = "  <- best, saved"
        print("epoch %2d  horizon %2d  loss %.3e  val err@20 %.2f px  (%.0fs, %.2f s/batch)%s%s"
              % (ep, H, tot / max(n, 1), ve, time.time() - t0, (time.time() - t0) / a.batches, tag,
                 "  [%d non-finite skipped]" % skipped if skipped else ""), flush=True)
    print("\nbest val err@20 %.2f px (straight lines %.2f px) -> %s   total %.0f min, %d batches of %d windows"
          % (best, straight, a.out, (time.time() - t_start) / 60, a.epochs * a.batches, a.batch))


if __name__ == "__main__":
    main()
