"""Dump a trained temporal-GNN checkpoint to a flat float32 blob + JSON manifest for the web build."""
import json, os, struct, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "web", "assets")

def dump(ckpt, stem):
    blob = torch.load(os.path.join(ROOT, ckpt), map_location="cpu")
    sd = blob["state_dict"]
    man = {"config": blob["config"], "tensors": {}}
    buf = bytearray()
    for k, v in sd.items():
        t = v.detach().contiguous().float()
        man["tensors"][k] = {"shape": list(t.shape), "offset": len(buf) // 4, "size": t.numel()}
        buf += struct.pack("<%df" % t.numel(), *t.reshape(-1).tolist())
    with open(os.path.join(OUT, stem + ".bin"), "wb") as fh:
        fh.write(buf)
    with open(os.path.join(OUT, stem + ".json"), "w") as fh:
        json.dump(man, fh)
    print("%s: %d tensors, %d floats (%.0f KB)" % (stem, len(sd), len(buf) // 4, len(buf) / 1024.0))
    for k, m in man["tensors"].items():
        print("   %-28s %s" % (k, m["shape"]))

dump("jet_brain_tgn.pt", "brain_full")
