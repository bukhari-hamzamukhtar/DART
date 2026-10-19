"""
The packet the inference thread hands back to the render/game thread.

main3.py pushed a bare 7-tuple through `decision_queue` and unpacked it into
seven loose locals.  Same data, same defaults — just named, so neither side can
silently drift out of order.
"""

from ..config import BASE_SPEED


class AIDecision:
    """One frame of advice from the GNN brain."""

    __slots__ = ("heading", "throttle", "drop_flare", "attention", "danger",
                 "edges", "candidate_scores", "maneuver", "fresh", "stamp",
                 "flight_heading", "flight_throttle", "cornered")

    def __init__(self, heading=0.0, throttle=None, drop_flare=False, attention=None,
                 danger=0.0, edges=None, candidate_scores=None, maneuver="NORMAL",
                 fresh=False, stamp=0, cornered=False):
        self.heading          = heading            # best evaluated heading, radians
        self.throttle         = BASE_SPEED if throttle is None else throttle
        self.drop_flare       = drop_flare         # brain wants countermeasures now
        self.attention        = attention if attention is not None else []
        self.danger           = danger             # scalar danger of the real situation
        self.edges            = edges if edges is not None else []
        self.candidate_scores = candidate_scores if candidate_scores is not None else []
        self.maneuver         = maneuver           # argmax of the maneuver head
        # False on the placeholder the match starts with, True once the planner
        # has actually run. main3.py used `heading != 0.0` as this sentinel.
        self.fresh            = fresh
        # Frame the snapshot behind this advice was taken on. The game runs at
        # 60 fps while a plan takes tens of ms, so advice always arrives stale;
        # this is how stale.
        self.stamp            = stamp
        # What the jet actually flies this frame: the strategic heading above,
        # adjusted every frame against live positions by planner.correct().
        self.flight_heading   = heading
        self.flight_throttle  = self.throttle
        # True when no action avoids an impact. Gates the scripted maneuvers,
        # which are a last resort now that the planner flies the jet.
        self.cornered         = cornered
