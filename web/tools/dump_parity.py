"""Dump real scene-graph ticks plus the torch model's outputs, for the JS port to match."""
import json
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from neural_evasion.arena import Arena  # noqa: E402
from neural_evasion.config import GRAPH_TICK  # noqa: E402
from neural_evasion.levels import LEVELS  # noqa: E402
from neural_evasion.model.temporal_gnn import GraphBrain, load_graph_brain  # noqa: E402
from neural_evasion.sim.scene_graph import SceneGraphBuilder  # noqa: E402
from neural_evasion.sim.tracker import Tracker  # noqa: E402
from neural_evasion.training.commander import Autopilot, ScriptedCommander  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "web", "tests", "parity.json")

level = LEVELS[2]            # level 3: datalink, so LINK rows are exercised
rng = random.Random(7)
arena = Arena(level, jet_invulnerable=True, launchers_invulnerable=True, visuals=False, seed=11)
cmd = ScriptedCommander(rng, pressure=1.0, jitter=20)
pilot = Autopilot(rng)
tracker = Tracker(level)
builder = SceneGraphBuilder(level)
model = load_graph_brain("jet_brain_tgn.pt", verbose=True)
brain = GraphBrain(model)

ticks = []
for _ in range(400):
    arena.step(pilot, cmd)
    f = arena.frame
    vis, vis_f = arena.radar_picture()
    tracker.update(f, arena.jet, vis)
    if f % GRAPH_TICK:
        continue
    g = builder.build(f, arena.jet, tracker, vis_f, arena.launchers)
    rows, links, _ = brain.step(g)
    tracker.apply_estimates(g.track_tids, rows, links)
    if True:
        ticks.append({
            "frame": f,
            "keys": [int(k) for k in g.keys],
            "x": [[float(v) for v in row] for row in g.x],
            "src": [int(v) for v in g.src],
            "dst": [int(v) for v in g.dst],
            "etype": [int(v) for v in g.etype],
            "eattr": [[float(v) for v in row] for row in g.eattr],
            "track_rows": [int(v) for v in g.track_rows],
            "track_tids": [int(v) for v in g.track_tids],
            "link_rows": [int(v) for v in g.link_rows],
            "rows": [{k: float(v) for k, v in r.items()} for r in rows],
            "links": [[int(a), int(b), float(p)] for a, b, p in links],
        })
    if len(ticks) >= 10:
        break

with open(OUT, "w") as fh:
    json.dump({"ticks": ticks}, fh)
print("wrote %s: %d ticks, nodes %s, edges %s, links %s"
      % (OUT, len(ticks), [len(t["keys"]) for t in ticks],
         [len(t["src"]) for t in ticks], [len(t["link_rows"]) for t in ticks]))
