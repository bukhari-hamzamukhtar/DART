"""
crosstest/clevrer_v3.py  --  the Neural Evasion V3 world model as the simulator of
                             CausalVis's CLEVRER counterfactual benchmark
===================================================================================

WHAT IS PLUGGED IN, AND WHAT IS NOT TOUCHED
CausalVis v8 (G:/CausalVis/v8/cf_world.py) builds ONE "what if" world per question
and advances every simulated object with a pluggable per-frame step (law_step).
This file replaces that step with the V3 world model's own rollout kernel,
neural_evasion/sim/rollout_kernel.missile_rollout, called unmodified. Everything else
is CausalVis's, loaded from their files: which objects follow the recording and
which are simulated, entries, the extension past the end of the video, the event
detectors, contact sizes for detection, the question parsing, the record format,
and the VAL -> TEST selection rule (v7/report.py). No CausalVis file is edited.

HOW A CLEVRER OBJECT BECOMES A V3 TRACK
The kernel works in game pixels, so the 1.9-unit table is mapped onto the 800 px
arena (the inverse of the mapping crosstest/cv_bridge.py uses the other way round).
Each object is a track with the V3 state: position, heading, speed. CLEVRER objects
steer at nothing, so they go in the way V3 represents anything that does not steer
-- turn rate 0, no lead, no seeker -- and the jet the kernel expects is placed out of
reach. Objects following the recording take part in interactions but keep their
recorded motion, as in v8.

THE VARIANTS (fixed before any result was seen)
  kinematic  V3 as built: every simulated object is an unguided track. V3 has no
             friction for missiles and no bounce, so this is straight-line motion at
             constant speed -- which IS what the V3 world model predicts for anything
             that does not steer.
  swarm      the one object-object rule V3 has: classic homing missiles push each
             other's headings apart inside REPULSE_DIST (tactics.apply_cluster_repulsion),
             at the game's constants, 26 px and 0.20.
  refit      V3's rules with their constants re-fitted on CLEVRER TRAIN only
             (crosstest/fit_v3_clevrer.py): per object kind, constant speed or the
             drag law V3 uses for flares (v <- v * (1 - c) before the move), and the
             swarm push's distance and strength fitted on isolated TRAIN collisions.
             The mirror of fine-tuning CausalVis's model on game trajectories.

WHAT THE KERNEL CANNOT DO HERE, STATED UP FRONT
It has no momentum exchange: a resting object that is struck stays at rest (its
speed floor is 1e-6 px/frame), a head-on pair passes through each other, and the
swarm push only turns headings. Those are the rules of a missile game, not of
rigid bodies.

Runs in CausalVis's interpreter (C:/Python314/python: scipy for the voxel near-lists).
numba is not installed there, so the kernel runs as plain Python -- the same code.

    python crosstest/clevrer_v3.py --selftest
    python crosstest/clevrer_v3.py --variant kinematic --which val --shard 0 --nshards 4 \
        --out crosstest/runs/clevrer/kinematic_val
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np
import torch

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
CV = r"G:/CausalVis"
sys.path.insert(0, DNN)
for _d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(CV, _d))

from neural_evasion.sim import rollout_kernel as K                         # noqa: E402  (V3, unmodified)
from benchmark_eval import (load_questions, normalise_choices,             # noqa: E402  (CausalVis)
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set, _vid_num)
from world import load_spatial                                             # noqa: E402

_spec = importlib.util.spec_from_file_location("cf_world_v3sim", os.path.join(CV, "v8", "cf_world.py"))
CFW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CFW)

PX = 800.0 / 1.9          # game px per CLEVRER table unit
OFFSET = 400.0            # table centre -> arena centre
ARENA = 800.0
NEVER = 1e9               # worldmodel_v3.NEVER: remaining life of something that does not burn out
SPEED_FLOOR = 1e-6        # worldmodel_v3.ThreatPack floors speed here


def _game_constant(name):
    """REPULSE_DIST / REPULSE_FORCE read from the game's source. tactics.py pulls in
    pygame, which this interpreter does not have, so the numbers are read, not imported."""
    src = open(os.path.join(DNN, "neural_evasion", "sim", "tactics.py"), encoding="utf-8").read()
    return float(re.search(r"^%s\s*=\s*([0-9.]+)" % name, src, re.M).group(1))


REPULSE_DIST_GAME = _game_constant("REPULSE_DIST")
REPULSE_FORCE_GAME = _game_constant("REPULSE_FORCE")


def kernel_frame(Ppx, Vpx, repel, rep_dist, rep_force):
    """One frame of the V3 kernel for n unguided tracks.

    Returns positions after the frame and the velocity each track carries into the
    next one. The kernel moves a track, then applies the swarm push to its heading;
    the push shows up in the NEXT step's motion, so the kernel is run for two steps
    and the second step's displacement is that velocity."""
    n = Ppx.shape[0]
    C, T = 1, 2
    speed = np.maximum(np.hypot(Vpx[:, 0], Vpx[:, 1]), SPEED_FLOOR)
    heading = np.arctan2(Vpx[:, 1], Vpx[:, 0])
    zf = np.zeros(n)
    zi = np.zeros(n, dtype=np.int64)
    zb = np.zeros(n, dtype=np.bool_)
    far = np.full((C, T), 1e9)                     # the jet, out of reach
    TX = np.empty((C, n, T)); TY = np.empty((C, n, T))
    AL = np.empty((C, n, T), dtype=np.bool_)
    K.missile_rollout(far, far.copy(), np.zeros((C, T)), np.zeros((C, T)),
                      Ppx[:, 0].astype(np.float64), Ppx[:, 1].astype(np.float64), heading, speed,
                      zf, zf, zf, np.full(n, 2.0), zf, zf, np.full(n, 0.40),
                      zb, np.full(n, -1, dtype=np.int64), zb, zf, zf, np.full(n, NEVER),
                      np.full(n, bool(repel), dtype=np.bool_),
                      zf, zf, zf, zf, np.ones(n, dtype=np.int64), zi, zi, np.zeros((n, 12)), zf,
                      np.zeros((T, 0)), np.zeros((T, 0)), np.zeros((T, 0), dtype=np.bool_), 6.0,
                      float(rep_dist), float(rep_force),
                      np.zeros(0), np.zeros(0), np.zeros(0), False, ARENA, ARENA, 11.0,
                      TX, TY, AL, np.empty((1, 1, 1), dtype=np.bool_), False,
                      np.empty((C, T)), np.empty((C, T), dtype=np.bool_), np.full((C, 0), -1, dtype=np.int32))
    x1 = np.stack([TX[0, :, 0], TY[0, :, 0]], axis=1)
    moved = np.stack([TX[0, :, 1] - TX[0, :, 0], TY[0, :, 1] - TY[0, :, 0]], axis=1)
    v_next = np.where(AL[0, :, 1][:, None], moved, Vpx)
    return x1, v_next


def make_v3_context(z, shapes, N, laws):
    """Per-object facts the V3 step needs. Signature of cf_world.make_law_context."""
    keys = [str(k) for k in z["obj_keys"]][:N]
    kinds = ["%s/%s" % (shapes[k], keys[k].split("_")[1]) for k in range(N)]
    drag = laws.get("drag", {})
    return {"variant": laws["variant"], "repel": laws["variant"] in ("swarm", "refit"),
            "dist": float(laws["repulse_dist_px"]), "force": float(laws["repulse_force"]),
            "drag": np.array([float(drag.get(kd, 0.0)) for kd in kinds])}


def v3_step(t, mode, M, pos, vel, q_s, p_s, mass, ctx, N):
    """One frame for every simulated object, by the V3 world model's kernel.
    Signature of cf_world.law_step."""
    idx = [k for k in range(N) if mode[k] == "sim" or M[t, k] == 1]
    if not any(mode[k] == "sim" for k in idx):
        return
    n = len(idx)
    P = np.empty((n, 2)); V = np.empty((n, 2))
    for a, k in enumerate(idx):
        if mode[k] == "sim":
            P[a] = q_s[k].numpy().astype(np.float64)
            V[a] = (p_s[k] / mass[0, k]).detach().numpy().astype(np.float64) * (1.0 - ctx["drag"][k])
        else:
            P[a] = pos[t, k].astype(np.float64)
            V[a] = vel[t, k].astype(np.float64)
    x1, v1 = kernel_frame(P * PX + OFFSET, V * PX, ctx["repel"], ctx["dist"], ctx["force"])
    for a, k in enumerate(idx):
        if mode[k] != "sim":
            continue
        q_s[k] = torch.from_numpy(((x1[a] - OFFSET) / PX).astype(np.float32))
        p_s[k] = mass[0, k].detach() * torch.from_numpy((v1[a] / PX).astype(np.float32))


# every simulated object goes through the V3 kernel from here on
CFW.make_law_context = make_v3_context
CFW.law_step = v3_step


def variant_laws(variant, refit_path=None):
    if variant == "kinematic":
        return {"variant": "kinematic", "repulse_dist_px": REPULSE_DIST_GAME, "repulse_force": REPULSE_FORCE_GAME}
    if variant == "swarm":
        return {"variant": "swarm", "repulse_dist_px": REPULSE_DIST_GAME, "repulse_force": REPULSE_FORCE_GAME}
    fit = json.load(open(refit_path))
    return {"variant": "refit", "repulse_dist_px": fit["repulse_dist_px"],
            "repulse_force": fit["repulse_force"], "drag": fit["drag"]}


def selftest():
    """The kernel, called as above, must reproduce V3's rules exactly."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(1, 8))
        P = rng.uniform(100, 700, (n, 2)); V = rng.normal(0, 3, (n, 2))
        x1, v1 = kernel_frame(P, V, False, REPULSE_DIST_GAME, REPULSE_FORCE_GAME)
        worst = max(worst, float(np.abs(x1 - (P + V)).max()), float(np.abs(v1 - V).max()))
    print("kinematic: kernel vs position += velocity, 200 random scenes: max diff %.2e px" % worst)
    ok = worst < 1e-9
    # swarm: two tracks inside REPULSE_DIST turn apart and keep their speeds
    P = np.array([[400.0, 400.0], [410.0, 400.0]]); V = np.array([[1.0, 0.0], [-1.0, 0.0]])
    x1, v1 = kernel_frame(P, V, True, REPULSE_DIST_GAME, REPULSE_FORCE_GAME)
    d = x1[1] - x1[0]
    dd = math.hypot(*d)
    s = (REPULSE_DIST_GAME - dd) / REPULSE_DIST_GAME * REPULSE_FORCE_GAME * math.pi * (1 - dd / REPULSE_DIST_GAME)
    want0 = np.array([math.cos(-s), math.sin(-s)])
    want1 = np.array([math.cos(math.pi + s), math.sin(math.pi + s)])
    err = max(float(np.abs(v1[0] - want0).max()), float(np.abs(v1[1] - want1).max()))
    print("swarm: two tracks %.0f px apart after moving, heading push matches tactics.apply_cluster_repulsion "
          "to %.2e, speeds %.6f %.6f" % (dd, err, np.hypot(*v1[0]), np.hypot(*v1[1])))
    ok = ok and err < 1e-9
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("kinematic", "swarm", "refit"), default="kinematic")
    ap.add_argument("--refit", default=os.path.join(DNN, "crosstest", "v3_clevrer_fit.json"))
    ap.add_argument("--model", default=os.path.join(CV, "v6_voxel.pt"),
                    help="only for masses/inertia in the world builder, exactly as v8 uses it")
    ap.add_argument("--data", default=os.path.join(CV, "data", "trajectories_3d_yaw"))
    ap.add_argument("--split", default=os.path.join(CV, "split3d_yaw.json"))
    ap.add_argument("--which", default="val")
    ap.add_argument("--questions", default=os.path.join(CV, "zechennlp", "counterfactual",
                                                         "validation-00000-of-00001.json"))
    ap.add_argument("--extend", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    torch.set_num_threads(1)
    LAWS = variant_laws(a.variant, a.refit)
    os.makedirs(a.out, exist_ok=True)
    fout = open(os.path.join(a.out, "shard%d.jsonl" % a.shard), "w", encoding="utf-8")

    model = load_spatial(a.model, yaw_known=False)
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    allowed = load_split_set(a.split, a.which)
    rows = load_questions(a.questions)

    # ---- from here to the end: v8/evaluate.py's loop, unchanged except the laws --------
    worlds = {}
    qn, done = -1, 0
    for row in rows:
        v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if v is None or v not in by_num or v not in allowed:
            continue
        qn += 1
        if qn % a.nshards != a.shard:
            continue
        if a.limit and done >= a.limit:
            break
        z = np.load(by_num[v], allow_pickle=True)
        keys = [str(k) for k in z["obj_keys"]]; N = min(8, z["positions"].shape[1])
        qd = parse_descriptors(row.get("program", []))
        if not qd:
            continue
        removed = set(k for k in resolve_all(qd[0], keys) if k < N)
        if not removed:
            continue
        negate = program_has(row.get("program", []), "negate")
        wk = (v, frozenset(removed))
        if wk not in worlds:
            worlds.clear()
            worlds[wk] = CFW.build_world(model, z, removed, extend=a.extend, laws=LAWS)
        w = worlds[wk]
        cols = [(int(f), int(x), int(y)) for f, x, y in z["collisions"]]
        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)
        for ci, c in enumerate(choices):
            truth = c["answer"]
            if truth not in ("correct", "wrong"):
                continue
            cd = parse_descriptors(c["program"])
            if len(cd) < 2:
                continue
            h1 = [k for k in resolve_all(cd[0], keys) if k < N]
            h2 = [k for k in resolve_all(cd[1], keys) if k < N]
            if len(h1) != 1 or len(h2) != 1 or h1[0] == h2[0]:
                continue
            i, j = sorted((h1[0], h2[0]))
            rec = {"video": v, "qid": str(row.get("question_id")), "question": row.get("question", ""),
                   "choice": c["choice"], "truth": truth, "negate": bool(negate),
                   "truth_collides": (truth == "correct") != bool(negate),
                   "i": i, "j": j, "removed": sorted(removed), "T": w["T"],
                   "participant_removed": bool(i in removed or j in removed),
                   "taint_i": w["taint"].get(i), "taint_j": w["taint"].get(j),
                   "rule": w["events"]["rule"].get((i, j), []),
                   "cal": w["events"]["cal"].get((i, j), []),
                   "kick2": w["events"]["kick2"].get((i, j), []),
                   "kick4": w["events"]["kick4"].get((i, j), []),
                   "near": w["near"].get((i, j), []),
                   "first_both": w["first_both"].get((i, j)),
                   "diag_obs_frames": [f for f, x, y in cols if {x, y} == {i, j}]}
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        done += 1
        print("shard %d  q %d  video %d  sim objects %d" % (a.shard, done, v, len(w["taint"])), flush=True)
    fout.close()


if __name__ == "__main__":
    main()
