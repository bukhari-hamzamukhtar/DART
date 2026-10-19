"""
Headless run of every level x every pilot version through the real match loop.

Scripted opponent at maximum fire rate, rendering on (dummy video driver) with
the graph overlay toggled, so the arena, all three pilots, both HUDs, terrain
shadows and every launch type are exercised. Fails on any exception.
"""
import os
import random
import sys
import time
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from neural_evasion.game import Match                          # noqa: E402
from neural_evasion.levels import LEVELS                       # noqa: E402
from neural_evasion.pilots import make_pilot                   # noqa: E402
from neural_evasion.training.commander import ScriptedCommander  # noqa: E402

FRAMES = 500
errors = []

for level in LEVELS:
    for version in (1, 2, 3):
        name = "L%d V%d" % (level.number, version)
        try:
            m = Match(level, version)
            m.pilot = make_pilot(version, level, threaded=False)
            m.pilot.start(m.arena)
            cmd = ScriptedCommander(random.Random(3), pressure=1.0)
            peak, t0, frames = 0, time.perf_counter(), 0
            for f in range(FRAMES):
                a = m.arena
                if a.game_over:
                    break
                a.begin_frame()
                cmd.act(a)
                a.advance(m.pilot)
                m.show_overlay = (f // 120) % 2 == 0
                m.render()
                peak = max(peak, len(a.missiles))
                frames += 1
            m.pilot.stop()
            dt = (time.perf_counter() - t0) / max(1, frames) * 1e3
            print("PASS  %s  %3d frames  %5.1f ms/frame (incl. render)  fired %3d  peak missiles %2d  "
                  "maneuver %-12s outcome %s"
                  % (name, frames, dt, a.stats['fired'], peak, m.pilot.maneuver_label,
                     a.cause or "alive"))
        except Exception:
            errors.append(name)
            print("FAIL  " + name)
            traceback.print_exc()

print()
if errors:
    print("FAILED: %s" % errors)
    sys.exit(1)
print("PASS - every level x version ran without exceptions")
