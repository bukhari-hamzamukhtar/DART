"""
The model path must stay identical to main3.py.

The *planner* around it is deliberately different now — main3.py scores frozen
candidates, the package rolls the world forward — so decisions no longer match
and should not. What must still match is everything below the planner: the
weights, the graph construction, and the network's output for a given scene.

A: every checkpoint tensor loads, and both builds hold the same weights
B: the graph builder produces identical tensors
C: the brain produces identical danger / attention / maneuver for a given scene
D: batched scene evaluation equals calling the model once per scene
"""
import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

import main3                                              # noqa: E402
from neural_evasion.model import device, get_brain        # noqa: E402
from neural_evasion.sim import graph_builder as gb_new    # noqa: E402
from neural_evasion.sim.evaluate import evaluate_scenes   # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        failures.append(name)


# ── A. weights ────────────────────────────────────────────────
new_model = get_brain()
old_sd, new_sd = main3.model.state_dict(), new_model.state_dict()
check("A1 same parameter names", set(old_sd) == set(new_sd), "%d tensors" % len(old_sd))
loaded = [k for k in old_sd if torch.equal(old_sd[k], new_sd[k])]
check("A2 every weight loads from the checkpoint", len(loaded) == len(old_sd),
      "%d/%d tensors identical (no layer falls back to random init)"
      % (len(loaded), len(old_sd)))


def make_scene(rng):
    js = {'pos': [rng.uniform(120, 680), rng.uniform(120, 430)],
          'vel': [rng.uniform(-8, 8), rng.uniform(-8, 8)], 'type': 0}
    ms = [{'pos': [rng.uniform(0, 800), rng.uniform(0, 550)],
           'vel': [rng.uniform(-8, 8), rng.uniform(-8, 8)],
           'type': 1, 'mtype': rng.randint(0, 1)}
          for _ in range(rng.randint(1, 5))]
    fs = [{'pos': [rng.uniform(0, 800), rng.uniform(0, 550)],
           'vel': [rng.uniform(-4, 4), rng.uniform(-4, 4)], 'type': 2}
          for _ in range(rng.randint(0, 7))]
    return js, ms, fs


# ── B. graph builder ──────────────────────────────────────────
rng = random.Random(1234)
b_ok = True
for _ in range(25):
    js, ms, fs = make_scene(rng)
    a = main3.build_candidate_sub_graphs(js, ms, fs)
    b = gb_new.build_candidate_sub_graphs(js, ms, fs)
    if len(a) != len(b):
        b_ok = False
        break
    for ga, gb in zip(a, b):
        if not (torch.equal(ga.x, gb.x) and torch.equal(ga.edge_index, gb.edge_index)
                and torch.equal(ga.edge_attr, gb.edge_attr)
                and torch.equal(ga.context, gb.context)):
            b_ok = False
            break
    if not b_ok:
        break
check("B  graph builder tensors identical (25 scenes)", b_ok)


# ── C. brain outputs ──────────────────────────────────────────
rng = random.Random(99)
worst = {"danger": 0.0, "attn": 0.0, "logit": 0.0}
mismatched_maneuver = 0
N = 40
for _ in range(N):
    js, ms, fs = make_scene(rng)
    sgs_old = main3.build_candidate_sub_graphs(js, ms, fs)
    sgs_new = gb_new.build_candidate_sub_graphs(js, ms, fs)
    ctx_old = torch.stack([g.context for g in sgs_old])
    ctx_new = torch.stack([g.context for g in sgs_new]).to(device)
    with torch.no_grad():
        d_o, a_o, _, l_o = main3.model(sgs_old, ctx_old, None)
        d_n, a_n, _, l_n = new_model(sgs_new, ctx_new, None)
    worst["danger"] = max(worst["danger"], abs(d_o.item() - d_n.item()))
    worst["attn"] = max(worst["attn"], (a_o - a_n).abs().max().item())
    worst["logit"] = max(worst["logit"], (l_o - l_n).abs().max().item())
    if l_o.argmax().item() != l_n.argmax().item():
        mismatched_maneuver += 1
check("C  brain outputs identical over %d scenes" % N,
      worst["danger"] == 0.0 and worst["attn"] == 0.0 and worst["logit"] == 0.0
      and mismatched_maneuver == 0,
      "max diffs danger %.1e attn %.1e logits %.1e" %
      (worst["danger"], worst["attn"], worst["logit"]))


# ── D. batched evaluation ─────────────────────────────────────
rng = random.Random(7)
worst_b = {"danger": 0.0, "attn": 0.0, "logit": 0.0}
for _ in range(20):
    scenes = []
    for _ in range(rng.randint(2, 5)):
        js, ms, fs = make_scene(rng)
        scenes.append(gb_new.build_candidate_sub_graphs(js, ms, fs))
    batched = evaluate_scenes(new_model, scenes, None)
    for sgs, res in zip(scenes, batched):
        ctx = torch.stack([g.context for g in sgs]).to(device)
        with torch.no_grad():
            d, aw, _, lg = new_model(sgs, ctx, None)
        worst_b["danger"] = max(worst_b["danger"], abs(d.item() - res.danger))
        worst_b["attn"] = max(worst_b["attn"],
                              max(abs(x - y) for x, y in zip(aw.tolist(), res.attention)))
        worst_b["logit"] = max(worst_b["logit"],
                               (lg - res.maneuver_logits).abs().max().item())
TOL = 1e-5
check("D  batched evaluation matches per-scene calls",
      all(v < TOL for v in worst_b.values()),
      "max diffs danger %.1e attn %.1e logits %.1e (float32 tolerance %.0e)" %
      (worst_b["danger"], worst_b["attn"], worst_b["logit"], TOL))

print()
print("RESULT:", "MODEL PATH UNCHANGED" if not failures else "MISMATCHES: %s" % failures)
sys.exit(1 if failures else 0)
