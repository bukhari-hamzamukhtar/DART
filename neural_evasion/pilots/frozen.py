"""
V1 and V2, flown unmodified inside the shared arena.

The frozen packages are imported as they are. This adapter does exactly what
their `Match` did around the brain -- push a world snapshot to the inference
worker every GNN_INTERVAL frames, run the per-frame `correct()` on live
positions, then let the ManeuverController select and fly -- in the same order.
Nothing in `neural_evasion_goated/` or `neural_evasion_v2/` is edited.

Two things are necessarily new, and apply to every version equally:

  * they see what the arena's radar sees. On open levels that is everything, as
    before; behind terrain it is less.
  * level 2/3 seeker missiles are reported to them as homing (guidance type 0):
    their network was never trained on type 2. The guidance values themselves
    (turn rate, bias) are read straight off the missile, as v1/v2 always did.

`threaded=True` runs their inference worker on its own thread, as in the real
game. `threaded=False` runs the same worker body inline, which is deterministic
and is what the headless benchmarks use.
"""

import importlib
import threading

from ..config import FLARE_LIFE_FRAMES, GUIDANCE_HOMING, GUIDANCE_SMART
from ..ui.hud import draw_hud

GNN_INTERVAL = 20       # as in both frozen game.py files


def _missile_dict(m):
    mtype = getattr(m, 'mtype', GUIDANCE_HOMING)
    if mtype == GUIDANCE_SMART:
        mtype = GUIDANCE_HOMING
    return {'pos': [m.x, m.y], 'vel': [m.vx, m.vy], 'heading': m.heading,
            'speed': m.speed, 'turn_rate': getattr(m, 'turn_rate', 0.0),
            'lateral_bias': getattr(m, 'lateral_bias', 0.0),
            'spawn_delay': getattr(m, 'spawn_delay', 0), 'radius': m.radius,
            'mtype': mtype}


class FrozenPilot(object):
    def __init__(self, version, threaded=True):
        pkg = "neural_evasion_goated" if version == 1 else "neural_evasion_v2"
        self.version = version
        self.threaded = threaded
        self.planner = importlib.import_module(pkg + ".sim.planner")
        self.inference = importlib.import_module(pkg + ".sim.inference")
        self.maneuvers = importlib.import_module(pkg + ".sim.maneuvers")
        self.tactics = importlib.import_module(pkg + ".sim.tactics")
        self.decision_mod = importlib.import_module(pkg + ".sim.decision")
        self.model_mod = importlib.import_module(pkg + ".model")
        self.show_overlay = False

    # ── lifecycle ─────────────────────────────────────────────
    def start(self, arena):
        self.controller = self.maneuvers.ManeuverController()
        self.decision = self.decision_mod.AIDecision()
        self.plan_age = 0
        self.active_maneuver = "NORMAL"
        self.model = self.model_mod.get_brain()
        self._stop = None
        if self.threaded:
            self.inference.drain_queues()
            self._stop = threading.Event()
            threading.Thread(target=self.inference.gnn_inference_worker,
                             args=(self._stop,), daemon=True).start()
        else:
            self._hidden = None
            self._prev_count = 0
            self._lead = self.inference.DEFAULT_LEAD

    def stop(self):
        if self._stop is not None:
            self._stop.set()
            self.inference.stop_worker()
            self.inference.drain_decisions()
            self._stop = None

    @property
    def maneuver_label(self):
        return self.controller.maneuver if self.version == 1 else self.active_maneuver

    # ── one frame ─────────────────────────────────────────────
    def act(self, arena):
        jet = arena.jet
        visible, vis_flares = arena.radar_picture()
        self._exchange(arena, visible, vis_flares)
        self.controller.tick_cooldowns()
        self._correct(jet, visible)
        wall = self.tactics.compute_wall_state(jet)
        self.controller.select(jet, visible, wall, self.decision.maneuver,
                               arena.stats, self.decision)
        self.controller.execute(jet, visible, arena.flares, wall, self.decision, arena.stats)

    def _snapshot(self, arena, visible, vis_flares):
        jet = arena.jet
        return {
            'jet': {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy],
                    'heading': jet.heading, 'speed': jet.speed},
            'missiles': [_missile_dict(m) for m in visible],
            'flares': [{'pos': [f.x, f.y], 'vel': [f.vx, f.vy],
                        'life': max(0, FLARE_LIFE_FRAMES - (arena.frame - (f.born_frame or arena.frame)))}
                       for f in vis_flares],
            'want_overlay': self.show_overlay,
            'frame': arena.frame,
            'plan_age': self.plan_age,
        }

    def _exchange(self, arena, visible, vis_flares):
        inf = self.inference
        if self.threaded:
            if arena.frame % GNN_INTERVAL == 0 and inf.state_queue.empty():
                inf.state_queue.put(self._snapshot(arena, visible, vis_flares))
            if not inf.decision_queue.empty():
                self.decision = inf.decision_queue.get()
                self.plan_age = max(0, arena.frame - self.decision.stamp)
            return

        if arena.frame % GNN_INTERVAL:
            return
        # The worker's loop body, run inline (see sim/inference.py in the frozen package).
        state = self._snapshot(arena, visible, vis_flares)
        count = len(state['missiles'])
        if count != self._prev_count:
            self._hidden = None
        self._prev_count = count
        self._lead = int(min(inf.MAX_LEAD, max(2, 0.5 * self._lead + 0.5 * (self.plan_age + 3))))
        result = self.planner.plan(self.model, state['jet'], state['missiles'],
                                   state['flares'], self._hidden, lead=self._lead)
        self._hidden = result.hidden
        self.decision = self.decision_mod.AIDecision(
            heading=result.heading, throttle=result.throttle,
            drop_flare=count > 0 and (result.cornered or result.danger > inf.FLARE_DANGER_THRESHOLD),
            attention=result.attention, danger=result.danger, edges=[],
            candidate_scores=result.candidate_scores, maneuver=result.maneuver,
            fresh=True, stamp=arena.frame, cornered=result.cornered)
        self.plan_age = 0

    def _correct(self, jet, visible):
        """Match._correct_plan from the frozen game.py of this version."""
        d = self.decision
        if not visible:
            d.flight_heading, d.flight_throttle = d.heading, d.throttle
            if self.version == 2:
                self.active_maneuver = "NORMAL"
            return
        seed_h = d.heading if d.fresh else jet.heading
        seed_t = d.throttle if d.fresh else jet.max_speed
        jd = {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy],
              'heading': jet.heading, 'speed': jet.speed}
        ms = [_missile_dict(m) for m in visible]
        if self.version == 1:
            d.flight_heading, d.flight_throttle = self.planner.correct(jd, ms, seed_h, seed_t)
        else:
            d.flight_heading, d.flight_throttle, self.active_maneuver = self.planner.correct(
                jd, ms, seed_h, seed_t, preferred=d.maneuver if d.fresh else None)

    # ── HUD ───────────────────────────────────────────────────
    def draw_hud(self, surface, arena, show_overlay):
        self.show_overlay = show_overlay
        draw_hud(surface, arena.jet, self.maneuver_label, self.decision, show_overlay,
                 controls=arena.level.controls)
