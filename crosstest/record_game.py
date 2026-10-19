"""
crosstest/record_game.py  --  game trajectories for fine-tuning CausalVis's world model
=====================================================================================

The same opponent and autopilot as neural_evasion/training/generate.py -- randomised
launch profiles and flight styles, an invulnerable jet, an unlimited flare magazine --
but instead of scene graphs it stores what a CLEVRER-style world model learns from:
every object's position and velocity at every frame. The jet is stored whole; each
missile as contiguous segments of flight (a missile still sitting out its launch
delay is not flying and is not stored).

    python crosstest/record_game.py --level 2 --episodes 24 --val 4
"""

import argparse
import os
import pickle
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np   # noqa: E402

DNN = r"C:/Users/Hamza Bukhari/Desktop/DNN"
sys.path.insert(0, DNN)

from neural_evasion.arena import Arena                                   # noqa: E402
from neural_evasion.levels import LEVELS                                 # noqa: E402
from neural_evasion.training.commander import Autopilot, ScriptedCommander  # noqa: E402

OUT = os.path.join(DNN, "crosstest", "data")
KIND = {"HomingMissile": 0, "BallisticMissile": 1, "SmartMissile": 2}   # cv_bridge.KINDS order


def run_episode(level, seed, frames):
    rng = random.Random(seed)
    arena = Arena(level, jet_invulnerable=True, launchers_invulnerable=True, visuals=False,
                  seed=seed, magazine=False)
    commander = ScriptedCommander(rng, pressure=rng.uniform(0.012, 0.05), randomize=True)
    pilot = Autopilot(rng)
    jet = np.zeros((frames, 4), np.float32)
    open_, done = {}, []
    for t in range(frames):
        arena.step(pilot, commander)
        j = arena.jet
        jet[t] = (j.x, j.y, j.vx, j.vy)
        seen = set()
        for m in arena.missiles:
            kind = KIND.get(type(m).__name__)
            if kind is None or getattr(m, 'spawn_delay', 0) > 0:
                continue
            seen.add(m.uid)
            tr = open_.get(m.uid)
            if tr is None:
                tr = open_[m.uid] = {"kind": kind, "t0": t, "pos": [], "vel": []}
            tr["pos"].append((m.x, m.y))
            tr["vel"].append((m.vx, m.vy))
        for uid in [u for u in open_ if u not in seen]:        # gone, or grounded: the segment ends
            done.append(open_.pop(uid))
    done += list(open_.values())
    tracks = [{"kind": tr["kind"], "t0": tr["t0"], "pos": np.asarray(tr["pos"], np.float32),
               "vel": np.asarray(tr["vel"], np.float32)} for tr in done if len(tr["pos"]) >= 2]
    return {"level": level.number, "seed": seed, "frames": frames, "jet": jet,
            "peaks": [tuple(float(v) for v in o) for o in level.obstacles], "tracks": tracks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--val", type=int, default=4)
    ap.add_argument("--frames", type=int, default=2400)
    a = ap.parse_args()
    level = LEVELS[a.level - 1]
    os.makedirs(OUT, exist_ok=True)
    for split, n, base in (("train", a.episodes, 20000), ("val", a.val, 95000)):
        eps, t0 = [], time.time()
        for e in range(n):
            ep = run_episode(level, base + 1000 * a.level + e, a.frames)
            eps.append(ep)
            print("level %d %s episode %d/%d: %d missile segments, %.0fs"
                  % (a.level, split, e + 1, n, len(ep["tracks"]), time.time() - t0), flush=True)
        path = os.path.join(OUT, "game_L%d_%s.pkl" % (a.level, split))
        with open(path, "wb") as fh:
            pickle.dump(eps, fh, protocol=4)
        print("wrote", path)


if __name__ == "__main__":
    main()
