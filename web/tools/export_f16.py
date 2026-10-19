"""Export the trained checkpoint as the float16 weights the browser build loads.

    python web/tools/export_f16.py            # from the repo root

Writes web/assets/brain16.json (shapes and offsets) and web/assets/weights.b64.txt
(the float16 blob as base64 text: some static hosts will not serve a .bin).
float32 is 458 KB; float16 halves it and costs about 2e-4 relative on weights
whose values are O(1). web/tests/net_parity.mjs measures what that does to the
network's estimates.
"""
import base64
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "web", "assets")

blob = torch.load(os.path.join(ROOT, "jet_brain_tgn.pt"), map_location="cpu")
sd = blob["state_dict"]

man = {"config": blob["config"], "dtype": "float16", "tensors": {}}
flat, offset = [], 0
for k, v in sd.items():
    t = v.detach().contiguous().float().reshape(-1).numpy().astype(np.float16)
    man["tensors"][k] = {"shape": list(v.shape), "offset": offset, "size": int(t.size)}
    flat.append(t)
    offset += int(t.size)
raw = np.concatenate(flat).astype(np.float16)
man["floats"] = int(raw.size)
man["b64"] = "weights.b64.txt"

with open(os.path.join(OUT, "weights.b64.txt"), "w") as fh:
    fh.write(base64.b64encode(raw.tobytes()).decode())
with open(os.path.join(OUT, "brain16.json"), "w") as fh:
    json.dump(man, fh)

f32 = np.concatenate([v.detach().contiguous().float().reshape(-1).numpy() for v in sd.values()])
rel = np.abs(f32 - raw.astype(np.float32)) / np.maximum(np.abs(f32), 1e-8)
print("%d floats, %d bytes; weight error mean rel %.2e" % (raw.size, raw.nbytes, rel.mean()))
