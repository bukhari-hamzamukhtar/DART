"""
The three theatres, and what each one asks of the jet.

  Level 1  OPEN SKY         the arena v1 and v2 were built for
  Level 2  TERRAIN MASKING  terrain blocks line of sight -- for the jet's radar,
                            for missile seekers, and for airframes
  Level 3  PINCER NETWORK   two homing batteries firing datalinked salvos that
                            lead the jet and converge from both flanks

A level only describes the world. Which brain flies the jet is chosen separately
(see `pilots`), so every version can be flown on every level and compared on
equal terms.
"""

from .config import BALLISTIC_COLOR, HEIGHT, HUD_HEIGHT, MISSILE_COLOR, WIDTH
from .sim.doctrine import L3_LEAD

GROUND_Y = HEIGHT - HUD_HEIGHT


class Station(object):
    """A ground battery: where it is, what it fires, which keys fire it."""

    __slots__ = ("kind", "x", "y", "key", "label", "color", "cluster_key")

    def __init__(self, kind, x, y, key, label, color, cluster_key=None):
        self.kind = kind          # "homing" or "ballistic"
        self.x = x
        self.y = y
        self.key = key            # fires one missile
        self.label = label
        self.color = color
        self.cluster_key = cluster_key    # launches a cluster (homing batteries)


class Level(object):
    __slots__ = ("number", "name", "tagline", "blurb", "obstacles", "stations",
                 "smart", "datalink", "lead", "pincer", "controls", "legend")

    def __init__(self, number, name, tagline, blurb, obstacles, stations,
                 smart, datalink, lead, pincer, controls, legend):
        self.number = number
        self.name = name
        self.tagline = tagline
        self.blurb = blurb
        self.obstacles = obstacles    # [(x, y, r)] -- terrain, lethal and opaque
        self.stations = stations
        self.smart = smart            # homing batteries fire seeker missiles
        self.datalink = datalink      # members of one salvo share a lock
        self.lead = lead              # lead-pursuit gain of single shots
        self.pincer = pincer          # the S key fires a two-battery pincer
        self.controls = controls      # HUD one-liner
        self.legend = legend          # [(key, description)] for the menus

    @property
    def has_terrain(self):
        return bool(self.obstacles)


LEVEL_1 = Level(
    1, "OPEN SKY", "the original arena",
    ["Homing battery west, ballistic east.",
     "Missiles flank, weave and dart."],
    obstacles=[],
    stations=[Station("homing", 100, GROUND_Y, "H", "HOMING", MISSILE_COLOR, cluster_key="C"),
              Station("ballistic", WIDTH - 100, GROUND_Y, "B", "BALLISTIC", BALLISTIC_COLOR)],
    smart=False, datalink=False, lead=0.0, pincer=False,
    controls="H homing  C cluster  B ballistic  G graph  ESC",
    legend=[("H", "homing missile"), ("C", "cluster of five"), ("B", "ballistic shot")],
)

LEVEL_2 = Level(
    2, "TERRAIN MASKING", "terrain blocks line of sight",
    ["Peaks block radar; seekers fly around.",
     "Lose the lock: it flies to your last spot."],
    # Laid out so both batteries can see the centre: terrain offers places to
    # break a lock, not a permanent safe zone. (The first layout put a ridge
    # between the batteries and the start point; every missile flew into it and
    # no pilot ever had to evade.)
    obstacles=[(230, 140, 40), (570, 140, 40), (400, 200, 32),
               (230, 360, 34), (560, 360, 38)],
    stations=[Station("homing", 100, GROUND_Y, "H", "SEEKER", MISSILE_COLOR, cluster_key="C"),
              Station("ballistic", WIDTH - 100, GROUND_Y, "B", "BALLISTIC", BALLISTIC_COLOR)],
    smart=True, datalink=False, lead=0.0, pincer=False,
    controls="H seeker  C cluster  B ballistic  G graph  ESC",
    legend=[("H", "seeker missile"), ("C", "seeker cluster"), ("B", "ballistic shot")],
)

LEVEL_3 = Level(
    3, "PINCER NETWORK", "two homing batteries, datalinked salvos",
    ["One lock guides every missile in its salvo.",
     "Lead pursuit and pincers cut off escape."],
    obstacles=[],
    stations=[Station("homing", 100, GROUND_Y, "H", "WEST", MISSILE_COLOR, cluster_key="C"),
              Station("homing", WIDTH - 100, GROUND_Y, "K", "EAST", MISSILE_COLOR, cluster_key="M")],
    smart=True, datalink=True, lead=L3_LEAD, pincer=True,
    controls="H/C west  K/M east  S pincer  G graph  ESC",
    legend=[("H", "west missile"), ("C", "west cluster"), ("K", "east missile"),
            ("M", "east cluster"), ("S", "pincer salvo")],
)

LEVELS = [LEVEL_1, LEVEL_2, LEVEL_3]
