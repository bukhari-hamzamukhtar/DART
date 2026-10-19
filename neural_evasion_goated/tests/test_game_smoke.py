"""Headless run of the modular match loop: exercise every FSM branch and collision path."""
import os
import random
import sys
import threading
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pygame  # noqa: E402

from neural_evasion.game import Match  # noqa: E402
from neural_evasion.model import get_brain  # noqa: E402
from neural_evasion.sim.entities import BallisticMissile, HomingMissile  # noqa: E402
from neural_evasion.sim.inference import gnn_inference_worker  # noqa: E402

get_brain()

random.seed(7)
seen_maneuvers = set()
max_missiles = 0
frames = 2000
errors = []

m = Match()
threading.Thread(target=gnn_inference_worker, args=(m._stop,), daemon=True).start()

try:
    for frame in range(frames):
        m.frame_count += 1
        if m.cluster_cooldown > 0:
            m.cluster_cooldown -= 1

        # Keep the pressure on: spawn threats directly (launcher cooldowns are
        # wall-clock based and this loop runs far faster than real time).
        if frame % 45 == 0 and len(m.missiles) < 8:
            m.missiles.append(HomingMissile(random.choice([80, 720]),
                                            random.choice([80, 500]),
                                            m.jet, m.jet.x, m.jet.y))
        if frame % 120 == 0:
            m.missiles.append(BallisticMissile(400, 560, m.jet.x, m.jet.y))
        if frame % 300 == 0:
            m.cluster_cooldown = 0
            m._launch_cluster()
        if frame % 200 == 0:
            m.show_gnn_overlay = not m.show_gnn_overlay   # exercise the [G] overlay path

        route = m.handle_events()
        if route is not None:
            errors.append("unexpected route %s at frame %d" % (route, frame))
            break

        if not m.game_over:
            m.exchange_with_brain()
            m.update_ai()
            m.update_physics()
        m.render()

        seen_maneuvers.add(m.controller.maneuver)
        max_missiles = max(max_missiles, len(m.missiles))

        if m.game_over:
            # restart so the run keeps covering ground
            m._stop.set()
            m = Match()
            threading.Thread(target=gnn_inference_worker,
                             args=(m._stop,), daemon=True).start()
except Exception:
    errors.append(traceback.format_exc())

m._stop.set()

print("frames run          :", frames)
print("maneuvers exercised :", sorted(seen_maneuvers))
print("peak missiles       :", max_missiles)
print("stats (last match)  :", m.stats)
print("jet pos             : (%.1f, %.1f)  heading %.2f" % (m.jet.x, m.jet.y, m.jet.heading))
print()
if errors:
    print("FAILED")
    for e in errors:
        print(e)
    sys.exit(1)
print("PASS - no exceptions across %d frames" % frames)
