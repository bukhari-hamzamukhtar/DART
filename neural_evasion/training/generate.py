"""
Generate training data for the v3 temporal graph brain.

Each episode runs the real arena with a scripted commander firing randomised
launch profiles (every guided missile in a random flight style) and the
autopilot flying the jet (invulnerable, with an unlimited flare magazine, so
every episode runs its full length and flares are common). Every GRAPH_TICK
frames it records the scene graph the jet would see -- built by the same Tracker
and SceneGraphBuilder the v3 pilot uses at runtime -- together with the truth
about every track:

    pursuit   does it steer at all (0 = ballistic)
    turn      its turn rate, rad/frame
    lead      lead-pursuit gain
    side      flank side, -1 / 0 / +1 (flankers and pincer members)
    guided    does it hold the jet right now (own lock or datalink)
    threat    will it pass within THREAT_DIST of the jet in the next 72 frames
    smart     seeker missile (levels 2-3) or classic
    bias      persistent lateral curve, rad/frame
    flank     how far off to its flank side it aims, tan of the angle (0 with no side)
    wobble    how far it weaves or darts, tan (weave amplitude + dart amplitude)

and, for every LINK edge, whether the two tracks really share a datalink.

    python -m neural_evasion.training.generate --level 2 --episodes 40 --val 6
"""

import argparse
import math
import os
import pickle
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from ..arena import Arena                                       # noqa: E402
from ..config import GRAPH_TICK, GUIDANCE_SMART, PROJECT_ROOT   # noqa: E402
from ..levels import LEVELS                                     # noqa: E402
from ..sim.scene_graph import SceneGraphBuilder                 # noqa: E402
from ..sim.tracker import PRIOR, Tracker                        # noqa: E402
from .commander import Autopilot, ScriptedCommander             # noqa: E402

THREAT_WINDOW = 72
THREAT_DIST   = 40.0
N_LABELS      = 10
DATA_DIR      = os.path.join(PROJECT_ROOT, "training_data_v3")


def _truth(m):
    smart = getattr(m, 'mtype', 0) == GUIDANCE_SMART
    tr = float(getattr(m, 'turn_rate', 0.0))
    if smart:
        guided = 1.0 if m.mode in ("LOCK", "LINK") else 0.0
    else:
        guided = 1.0 if tr > 0.0 else 0.0
    side = float(getattr(m, 'side', 0.0))
    flank = float(getattr(m, 'flank', 0.0)) if side != 0.0 else 0.0
    wobble = float(getattr(m, 'weave_amp', 0.0)) + float(getattr(m, 'jink_amp', 0.0))
    return [1.0 if tr > 0.0 else 0.0, tr, float(getattr(m, 'lead', 0.0)),
            side, guided, 0.0, 1.0 if smart else 0.0,
            float(getattr(m, 'lateral_bias', 0.0)), flank, wobble]


def run_episode(level, seed, frames):
    rng = random.Random(seed)
    arena = Arena(level, jet_invulnerable=True, launchers_invulnerable=True, visuals=False,
                  seed=seed, magazine=False)
    commander = ScriptedCommander(rng, pressure=rng.uniform(0.012, 0.05), randomize=True)
    pilot = Autopilot(rng)
    tracker = Tracker(level)
    builder = SceneGraphBuilder(level)

    ticks = []
    track_pos = {}        # uid -> {frame: (x, y)}
    jet_pos = {}
    for _ in range(frames):
        arena.step(pilot, commander)
        f = arena.frame
        jet_pos[f] = (arena.jet.x, arena.jet.y)
        for m in arena.missiles:
            track_pos.setdefault(m.uid, {})[f] = (m.x, m.y)
        # Hidden tracks are dead-reckoned from the estimates. At runtime those
        # come from the trained network; here they come from the truth, so the
        # coasting positions in the data look like a well-trained network's.
        truth_now = {m.uid: m for m in arena.missiles}
        for tid, t in tracker.tracks.items():
            m = truth_now.get(tid)
            if m is not None:
                tr = float(getattr(m, 'turn_rate', 0.0))
                t.est = dict(PRIOR, pursuit=1.0 if tr > 0 else 0.0, turn_rate=tr, turn_hi=tr,
                             bias=float(getattr(m, 'lateral_bias', 0.0)))
        vis_m, vis_f = arena.radar_picture()
        tracker.update(f, arena.jet, vis_m)
        if f % GRAPH_TICK:
            continue

        g = builder.build(f, arena.jet, tracker, vis_f, arena.launchers)
        truth = {m.uid: m for m in arena.missiles}
        y = np.zeros((len(g.track_rows), N_LABELS), dtype=np.float32)
        ymask = np.zeros(len(g.track_rows), dtype=bool)
        for i, tid in enumerate(g.track_tids):
            m = truth.get(tid)
            if m is not None:
                y[i] = _truth(m)
                ymask[i] = True
        link_y = np.zeros(len(g.link_rows), dtype=np.float32)
        link_mask = np.zeros(len(g.link_rows), dtype=bool)
        tid_of = dict(zip(g.track_rows, g.track_tids))
        for j, r in enumerate(g.link_rows):
            a, b = truth.get(tid_of[int(g.src[r])]), truth.get(tid_of[int(g.dst[r])])
            if a is None or b is None:
                continue
            link_mask[j] = True
            na, nb = getattr(a, 'network', None), getattr(b, 'network', None)
            link_y[j] = 1.0 if (na is not None and na == nb) else 0.0

        ticks.append({
            'frame': f,
            'keys': np.asarray(g.keys, dtype=np.int64),
            'x': g.x.astype(np.float16),
            'src': g.src.astype(np.int32), 'dst': g.dst.astype(np.int32),
            'etype': g.etype.astype(np.int8), 'eattr': g.eattr.astype(np.float16),
            'track_rows': np.asarray(g.track_rows, dtype=np.int32),
            'track_tids': np.asarray(g.track_tids, dtype=np.int64),
            'visible': np.asarray([tracker.tracks[t].visible for t in g.track_tids], dtype=bool),
            'y': y, 'ymask': ymask,
            'link_rows': np.asarray(g.link_rows, dtype=np.int32),
            'link_y': link_y, 'link_mask': link_mask,
        })

    # Threat labels need the future: does the missile come within THREAT_DIST
    # of the jet in the next THREAT_WINDOW frames?
    for t in ticks:
        f0 = t['frame']
        for i, tid in enumerate(t['track_tids'].tolist()):
            if not t['ymask'][i]:
                continue
            path = track_pos.get(tid, {})
            hit = 0.0
            for f in range(f0, f0 + THREAT_WINDOW + 1):
                p = path.get(f)
                if p is None:
                    if f > f0:
                        break
                    continue
                j = jet_pos.get(f)
                if j is not None and math.hypot(p[0] - j[0], p[1] - j[1]) < THREAT_DIST:
                    hit = 1.0
                    break
            t['y'][i, 5] = hit
    return ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--frames", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    level = LEVELS[args.level - 1]
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, n, base in (("train", args.episodes, 10000), ("val", args.val, 90000)):
        episodes, t0, n_ticks = [], time.time(), 0
        for e in range(n):
            ep = run_episode(level, base + 1000 * args.level + e + args.seed, args.frames)
            episodes.append(ep)
            n_ticks += len(ep)
            print("level %d %s episode %d/%d: %d ticks, %.0fs elapsed"
                  % (args.level, split, e + 1, n, len(ep), time.time() - t0))
            sys.stdout.flush()
        path = os.path.join(DATA_DIR, "level%d_%s.pkl" % (args.level, split))
        with open(path, "wb") as fh:
            pickle.dump(episodes, fh, protocol=4)
        print("wrote %s (%d episodes, %d ticks)" % (path, n, n_ticks))


if __name__ == "__main__":
    main()
