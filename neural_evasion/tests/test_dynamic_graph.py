"""
The scene graph must be a dynamic graph, not a string of unrelated snapshots.

  A  a missile keeps one node key for its whole life
  B  flying behind a peak turns its track into a coasting node -- still in the
     graph, same key -- with an OCCLUDE edge from that peak, and it is
     reacquired under the same key when it comes out
  C  tracker events and edge insertions/deletions report all of that
  D  the GNN's memory is keyed on node keys: a new node starts from zero, a
     surviving node carries its memory forward, a deleted node's memory is gone
  E  datalink edges appear between tracks in link range on level 3
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

from neural_evasion.arena import Arena                                   # noqa: E402
from neural_evasion.levels import LEVEL_2, LEVEL_3                       # noqa: E402
from neural_evasion.model.temporal_gnn import GraphBrain, TemporalGraphNet  # noqa: E402
from neural_evasion.sim.entities import BallisticMissile, SmartMissile   # noqa: E402
from neural_evasion.sim.scene_graph import (LINK, OCCLUDE, SceneGraphBuilder,  # noqa: E402
                                            diff, key_obstacle, key_track)
from neural_evasion.sim.tracker import Tracker                           # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        failures.append(name)


class Hover(object):
    """Keep the jet nearly still at the centre."""
    def act(self, arena):
        arena.jet.set_throttle(0.3)
        arena.jet.vx *= 0.5
        arena.jet.vy *= 0.5


torch.manual_seed(0)
a = Arena(LEVEL_2, counter_attack=False, visuals=False, launchers_invulnerable=True)
a.jet.x, a.jet.y = 400.0, 300.0
# Flies left along y=100, passing behind the peak at (400, 200) as seen from the jet.
PEAK_INDEX = 2
assert LEVEL_2.obstacles[PEAK_INDEX][:2] == (400, 200), "level 2 terrain changed; update this scenario"
m = BallisticMissile(480.0, 100.0, 0.0, 100.0)
a._add_missile(m)
tracker, builder = Tracker(LEVEL_2), SceneGraphBuilder(LEVEL_2)
brain = GraphBrain(TemporalGraphNet())
key = key_track(m.uid)
peak = key_obstacle(PEAK_INDEX)

keys_seen, vis_seq, occlude_while_hidden, prev = set(), [], [], None
inserted_occlude = False
mem_first, mem_later = None, None
for f in range(30):
    a.begin_frame()
    a.advance(Hover())
    vis, vf = a.radar_picture()
    tracker.update(a.frame, a.jet, vis)
    t = tracker.tracks.get(m.uid)
    if t is None:
        break
    g = builder.build(a.frame, a.jet, tracker, vf, a.launchers)
    had_memory = key in brain.memory
    brain.step(g)
    if f == 0:
        mem_first = (had_memory, float(brain.memory[key].abs().sum()))
    if f == 5:
        mem_later = had_memory
    keys_seen.add(g.keys[g.track_rows[0]])
    vis_seq.append(t.visible)
    if not t.visible:
        occl = [(g.keys[int(s)], g.keys[int(d)]) for s, d, e in zip(g.src, g.dst, g.etype) if e == OCCLUDE]
        occlude_while_hidden.append((peak, key) in occl)
    d = diff(prev, g)
    if OCCLUDE in d and (peak, key) in d[OCCLUDE][0]:
        inserted_occlude = True
    prev = g

check("A  one missile, one node key for its whole life", keys_seen == {key}, "keys %s" % sorted(keys_seen))
hidden_then_back = (True in vis_seq and False in vis_seq and
                    vis_seq.index(False) < len(vis_seq) - 1 - vis_seq[::-1].index(True))
check("B  behind the peak it coasts, then is reacquired", hidden_then_back,
      "visibility %s" % "".join("V" if v else "-" for v in vis_seq))
check("B' while hidden, the peak that hides it has an OCCLUDE edge to it",
      bool(occlude_while_hidden) and all(occlude_while_hidden),
      "%d hidden frames" % len(occlude_while_hidden))
kinds = [e[1] for e in tracker.events]
check("C  tracker reports born, occluded, reacquired", {"born", "occluded", "reacquired"} <= set(kinds),
      "events %s" % kinds)
check("C' the edge insertion shows up in the graph diff", inserted_occlude)
check("D  a new node has no memory; after one tick it has some",
      mem_first is not None and mem_first[0] is False and mem_first[1] > 0.0,
      "had memory before first tick: %s, |memory| after: %.3f" % mem_first if mem_first else "")
check("D' a surviving node carries memory forward", mem_later is True)
for m2 in list(a.missiles):
    a.missiles.remove(m2)
vis, vf = a.radar_picture()
tracker.update(a.frame + 1, a.jet, vis)
g = builder.build(a.frame + 1, a.jet, tracker, vf, a.launchers)
brain.step(g)
check("D'' a deleted node's memory is gone", key not in brain.memory)

a3 = Arena(LEVEL_3, counter_attack=False, visuals=False)
s1 = SmartMissile(300, 400, a3.jet, 400, 300, network=7)
s2 = SmartMissile(360, 420, a3.jet, 400, 300, network=7)
a3._add_missile(s1)
a3._add_missile(s2)
tr3, b3 = Tracker(LEVEL_3), SceneGraphBuilder(LEVEL_3)
tr3.update(1, a3.jet, a3.missiles)
g3 = b3.build(4, a3.jet, tr3, [], a3.launchers)
check("E  level 3 builds LINK edges between tracks in datalink range",
      len(g3.link_rows) == 1 and sum(1 for e in g3.etype if e == LINK) == 2,
      "link rows %d" % len(g3.link_rows))

print()
print("RESULT:", "DYNAMIC GRAPH BEHAVES AS ONE EVOLVING GRAPH" if not failures else "FAILED: %s" % failures)
sys.exit(1 if failures else 0)
