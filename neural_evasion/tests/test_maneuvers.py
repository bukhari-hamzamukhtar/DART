"""Force every maneuver branch to execute end-to-end inside the arena."""
import os, sys, traceback
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from neural_evasion.arena import Arena
from neural_evasion.levels import LEVEL_1
from neural_evasion.sim.decision import AIDecision
from neural_evasion.sim.entities import HomingMissile
from neural_evasion.sim.maneuvers import DURATION, ManeuverController
from neural_evasion.sim.tactics import compute_wall_state


class ForcedManeuver(object):
    """Pilot that flies whatever maneuver the controller is in."""

    def __init__(self, controller, decision):
        self.c, self.d = controller, decision

    def act(self, arena):
        wall = compute_wall_state(arena.jet)
        self.c.tick_cooldowns()
        self.c.select(arena.jet, arena.missiles, wall, "NORMAL", arena.stats, self.d)
        self.c.execute(arena.jet, arena.missiles, arena.flares, wall, self.d, arena.stats)


errors = []
for name, dur in list(DURATION.items()) + [("NORMAL", 5)]:
    try:
        a = Arena(LEVEL_1, counter_attack=False)
        a.jet.x, a.jet.y = 400.0, 275.0
        for k in range(3):
            a._add_missile(HomingMissile(400 + 60*k, 200, a.jet, a.jet.x, a.jet.y))
        c = ManeuverController()
        c.maneuver, c.timer = name, dur
        c.memory['start_speed'] = a.jet.speed
        d = AIDecision(heading=1.0, drop_flare=True, danger=0.9)
        pilot = ForcedManeuver(c, d)
        for f in range(dur + 20):
            a.begin_frame()
            a.advance(pilot)
        print("PASS  %-14s ran %3d frames -> ended in %-12s flares=%d maneuvers=%d"
              % (name, dur + 20, c.maneuver, a.stats['flares'], a.stats['maneuvers']))
    except Exception:
        errors.append(name); print("FAIL  " + name); traceback.print_exc()

sys.exit(1 if errors else 0)
