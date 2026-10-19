"""Small math helpers shared by the simulation and the AI."""

import math


def normalize_angle(a):
    """Wrap an angle into (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
