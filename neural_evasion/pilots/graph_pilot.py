"""
V3: the dynamic-graph pilot.

One frame, in order:

  radar      what the jet can see (terrain blocks it)
  tracker    persistent tracks: identity, coasting behind terrain, and each
             missile's estimated last-known point from lock-edge events
  graph      every GRAPH_TICK frames, a snapshot of the dynamic scene graph
  brain      the temporal GNN reads the snapshot with its per-node memory and
             re-estimates every track's hidden guidance state and flight style
  planner    every frame, every candidate action rolled 72 frames through the
             graph world model with those estimates and the flares in the air;
             when a volley is in the magazine, every action is imagined with it
             as well, and the cheapest future decides both the flight and the
             flares
  controls   the chosen heading and throttle, applied exactly as the world
             model assumed. Maneuvers stay soft: each one is just a candidate
             action the planner can take up, drop and resume frame by frame

Flares come out of the arena's magazine -- one volley per enemy cluster launch --
so a volley is only imagined while a charge is left, and firing it has to buy
something close to a missed hit.

`oracle=True` swaps the radar and the GNN for the truth: every missile and flare
in the arena, hidden or not, with its true guidance values and flight style. It
is not a mode to play; it measures the ceiling, so the gap between it and v3 is
the cost of not knowing.
"""

import math
import time
from collections import deque

from ..config import FLARE_LIFE_FRAMES, GRAPH_TICK
from ..model.temporal_gnn import GraphBrain, load_graph_brain
from ..sim.decision import AIDecision
from ..sim.entities import FlareParticle
from ..sim.planner_v3 import plan_v3
from ..sim.scene_graph import (DECOY, EDGE_TYPES, LINK, OCCLUDE, SEEKER,
                               SceneGraphBuilder, diff, key_track)
from ..sim.tactics import predict_intercept_flare
from ..sim.tracker import Tracker
from ..sim.worldmodel import STEER_GAIN
from ..sim.worldmodel_v3 import flare_state, oracle_threat
from ..utils import normalize_angle

FLARE_REARM_FRAMES = 30     # frames between volleys, so two charges are never dumped at once
FLARE_SPREAD       = (0.0, -0.20, 0.20)
VOLLEY_TARGETS     = 6      # a volley answers the nearest few missiles, not the whole sky

_MODEL_CACHE = {}


def shared_model():
    if 'model' not in _MODEL_CACHE:
        _MODEL_CACHE['model'] = load_graph_brain()
    return _MODEL_CACHE['model']


class GraphPilot(object):
    version = 3

    def __init__(self, level, oracle=False, model=None, use_brain=True, calibrated=False):
        self.level = level
        self.oracle = oracle
        # calibrated=True adds the conformal correction (training/calibrate.py)
        # to the network's 90% turn-rate quantile. Off by default: lifting
        # coverage to 99% did not buy closed-loop survival (ARCHITECTURE_V3.md).
        self.calibrated = calibrated
        # use_brain=False flies on the tracker's prior estimates alone: the
        # ablation that shows what the GNN is worth in closed loop.
        self.model = (model if model is not None else shared_model()) if use_brain else None
        self.show_overlay = True

    def start(self, arena):
        self.tracker = Tracker(self.level)
        if self.calibrated and self.model is not None:
            self.tracker.turn_margin = getattr(self.model, 'turn_margin', {}).get(self.level.number, 0.0)
        self.builder = SceneGraphBuilder(self.level)
        self.brain = GraphBrain(self.model) if self.model is not None else None
        self.hold = None            # last frame's choice, offered again this frame
        self.decision = AIDecision()
        self.graph = None
        self.gates = None
        self.links = []
        self.plan = None
        self.maneuver_label = "NORMAL"
        self.ticker = deque(maxlen=7)
        self.edge_delta = (0, 0)
        self.timing = {'graph': 0.0, 'gnn': 0.0, 'plan': 0.0}
        self._seen_events = 0
        self.last_flare_frame = -10 ** 6
        self._charges = arena.flare_charges

    def stop(self):
        pass

    # ── one frame ─────────────────────────────────────────────
    def act(self, arena):
        jet = arena.jet
        if arena.flare_charges > self._charges:
            self.ticker.append("f%-5d MAGAZINE   +%d volley (cluster launched), %d held"
                               % (arena.frame, arena.flare_charges - self._charges, arena.flare_charges))
        self._charges = arena.flare_charges
        visible, vis_flares = arena.radar_picture()
        self.tracker.update(arena.frame, jet, visible)

        if arena.frame % GRAPH_TICK == 0:
            t0 = time.perf_counter()
            g = self.builder.build(arena.frame, jet, self.tracker, vis_flares, arena.launchers)
            t1 = time.perf_counter()
            if self.brain is not None:
                rows, self.links, self.gates = self.brain.step(g)
                self.tracker.apply_estimates(g.track_tids, rows, self.links)
            t2 = time.perf_counter()
            self._log(arena.frame, self.graph, g)
            self.graph = g
            self.timing['graph'] = (t1 - t0) * 1e3
            self.timing['gnn'] = (t2 - t1) * 1e3

        if self.oracle:
            # Omniscient: every missile and flare in the arena with its true
            # state, hidden or not. The ceiling any inference could reach.
            threats = [oracle_threat(m, self.level.datalink, arena.frame) for m in arena.missiles]
            in_air = arena.flares
        else:
            threats = self.tracker.threat_dicts(arena.frame)
            in_air = vis_flares
        flares = [flare_state(f, arena.frame) for f in in_air if not getattr(f, 'dead', False)]

        # The volley the jet could fire this frame, if the magazine holds one.
        shots = volley = None
        if (visible and arena.can_flare()
                and arena.frame - self.last_flare_frame >= FLARE_REARM_FRAMES):
            shots = self._volley(jet, visible)
            volley = [{'pos': (jet.x, jet.y), 'vel': (math.cos(a) * s, math.sin(a) * s),
                       'ttl': FLARE_LIFE_FRAMES} for a, s in shots]

        jd = {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy],
              'heading': jet.heading, 'speed': jet.speed}
        t3 = time.perf_counter()
        p = plan_v3(jd, threats, self.level, keep_future=self.show_overlay,
                    flares=flares, volley=volley, hold=self.hold)
        self.timing['plan'] = (time.perf_counter() - t3) * 1e3
        self.plan = p
        self.plan_threats = threats

        d = self.decision
        d.heading = d.flight_heading = p.heading
        d.throttle = d.flight_throttle = p.throttle
        d.fresh = True
        d.cornered = p.cornered
        d.danger = p.danger
        d.candidate_scores = p.candidate_scores
        d.maneuver = p.label
        if p.label != self.maneuver_label and p.label != "NORMAL":
            arena.stats['maneuvers'] += 1
        self.maneuver_label = p.label
        self.hold = (p.heading, p.throttle, p.label)

        # Countermeasures are the world model's call: plan_v3 imagined every
        # action with and without this volley, and fires it only if the best
        # future with it is cheaper by more than a volley is worth. The same
        # FlareParticle arithmetic the rollout assumed, so the flares fly
        # exactly as imagined; the arena takes the charge.
        if p.flare and arena.spend_flare():
            self.last_flare_frame = arena.frame
            for a, s in shots:
                arena.flares.append(FlareParticle(jet.x, jet.y, a, s))
            arena.stats['flares'] += len(shots)
            self._charges = arena.flare_charges
            self.ticker.append("f%-5d VOLLEY     %d flares, future %.2f cheaper, %d left"
                               % (arena.frame, len(shots), p.flare_gain, arena.flare_charges))
        d.drop_flare = False

        # Fly exactly the control the world model imagined: steer toward the
        # chosen heading at the jet's steering gain, at the chosen throttle.
        # v2's controller blended in a flee reflex near missiles and could hand
        # the jet to a scripted maneuver when cornered -- both were there to
        # patch plans that arrived frames late. v3 plans every frame on an exact
        # model, and any other control here flies a future nobody imagined.
        jet.heading += normalize_angle(p.heading - jet.heading) * STEER_GAIN
        jet.set_throttle(p.throttle)

    @staticmethod
    def _volley(jet, visible):
        """(angle, speed) of each flare in a volley thrown across the nearest missiles' paths."""
        near = sorted(visible, key=lambda m: (m.x - jet.x) ** 2 + (m.y - jet.y) ** 2)[:VOLLEY_TARGETS]
        shots = []
        for m in near:
            angle, speed = predict_intercept_flare(jet, m)
            for k, spread in enumerate(FLARE_SPREAD):
                shots.append((angle + spread, speed * (1.0 if k == 0 else 0.85)))
        return shots

    # ── the event stream, for the HUD ─────────────────────────
    def _log(self, frame, prev, cur):
        fresh = min(self.tracker.event_count - self._seen_events, len(self.tracker.events))
        if fresh > 0:
            for f, kind, tid, detail in list(self.tracker.events)[-fresh:]:
                self.ticker.append("f%-5d %-10s T%d %s" % (f, kind.upper(), tid, detail))
        self._seen_events = self.tracker.event_count
        delta = diff(prev, cur)
        added = sum(len(v[0]) for v in delta.values())
        removed = sum(len(v[1]) for v in delta.values())
        self.edge_delta = (added, removed)
        names = {k: n for k, n in zip(cur.keys, cur.names)}
        if prev is not None:
            names.update({k: n for k, n in zip(prev.keys, prev.names)})
        for typ, verb in ((SEEKER, "+"), (OCCLUDE, "+"), (DECOY, "+")):
            for s, dd in delta.get(typ, ([], []))[0]:
                self.ticker.append("f%-5d +%-9s %s->%s" % (frame, EDGE_TYPES[typ], names.get(s, "?"), names.get(dd, "?")))
        for s, dd in delta.get(SEEKER, ([], []))[1]:
            self.ticker.append("f%-5d -%-9s %s->%s" % (frame, "SEEKER", names.get(s, "?"), names.get(dd, "?")))

    # ── HUD ───────────────────────────────────────────────────
    def draw_hud(self, surface, arena, show_overlay):
        from ..ui.hud_v3 import draw_hud_v3
        self.show_overlay = show_overlay
        draw_hud_v3(surface, arena, self, show_overlay)
