"""Force every maneuver branch to execute end-to-end."""
import os, sys, traceback
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from neural_evasion.game import Match
from neural_evasion.sim.entities import HomingMissile
from neural_evasion.sim.maneuvers import DURATION

errors = []
for name, dur in list(DURATION.items()) + [("NORMAL", 5)]:
    try:
        m = Match()
        m.jet.x, m.jet.y = 400.0, 275.0
        for k in range(3):
            m.missiles.append(HomingMissile(400 + 60*k, 200, m.jet, m.jet.x, m.jet.y))
        m.controller.maneuver = name
        m.controller.timer = dur
        m.controller.memory['start_speed'] = m.jet.speed
        m.decision.heading = 1.0
        m.decision.drop_flare = True
        m.decision.danger = 0.9
        for f in range(dur + 20):
            m.update_ai()
            m.update_physics()
            m.render()
        print("PASS  %-14s ran %3d frames -> ended in %-12s flares=%d maneuvers=%d"
              % (name, dur + 20, m.controller.maneuver, m.stats['flares'], m.stats['maneuvers']))
    except Exception:
        errors.append(name); print("FAIL  " + name); traceback.print_exc()

sys.exit(1 if errors else 0)
