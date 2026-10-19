"""
Who flies the jet.

A pilot is the brain half of a match: each frame the arena calls `act(arena)`,
and the pilot sets the jet's heading and throttle (and may eject flares). Every
version implements the same interface, so every version can fly every level:

    V1  the "goated" build        frozen in neural_evasion_goated/
    V2  soft maneuvers            frozen in neural_evasion_v2/
    V3  dynamic graph             this package

V1 and V2 run their own frozen planner, inference and maneuver code, unmodified;
`frozen.FrozenPilot` only does what their `Match` class used to do around it.
"""


class VersionInfo(object):
    __slots__ = ("number", "name", "summary", "knows")

    def __init__(self, number, name, summary, knows):
        self.number = number
        self.name = name
        self.summary = summary
        self.knows = knows


VERSIONS = [
    VersionInfo(1, "V1  GOATED",
                "every-frame world-model MPC",
                "reads each missile's guidance values directly"),
    VersionInfo(2, "V2  SOFT MANEUVERS",
                "maneuvers as per-frame options",
                "reads each missile's guidance values directly"),
    VersionInfo(3, "V3  DYNAMIC GRAPH",
                "temporal GNN + graph world model",
                "radar tracks only; infers the rest"),
]


def make_pilot(version, level, threaded=True, oracle=False):
    if version in (1, 2):
        from .frozen import FrozenPilot
        return FrozenPilot(version, threaded=threaded)
    from .graph_pilot import GraphPilot
    return GraphPilot(level, oracle=oracle)
