"""
Line of sight and terrain.

Terrain is a list of circles `(x, y, r)`. The scalar functions serve the arena
and the tracker; `blocked_array` serves the world model. Both use the same
formula in the same order, so the arena and the jet's imagination agree to
floating-point rounding about who can see whom.
"""

import math

import numpy as np


def segment_blocked(ax, ay, bx, by, obstacles):
    """True if the segment a->b passes through any obstacle."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    for ox, oy, r in obstacles:
        ex = ox - ax
        ey = oy - ay
        if l2 > 1e-12:
            t = (ex * dx + ey * dy) / l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        else:
            t = 0.0
        cx = t * dx - ex
        cy = t * dy - ey
        if math.hypot(cx, cy) < r:
            return True
    return False


def inside_any(x, y, obstacles, pad=0.0):
    """True if (x, y) is within `pad` of any obstacle's surface."""
    for ox, oy, r in obstacles:
        if math.hypot(x - ox, y - oy) < r + pad:
            return True
    return False


def clearance(x, y, obstacles):
    """Distance from (x, y) to the nearest obstacle surface (inf with no terrain)."""
    best = float("inf")
    for ox, oy, r in obstacles:
        best = min(best, math.hypot(x - ox, y - oy) - r)
    return best


def blocking_obstacle(ax, ay, bx, by, obstacles):
    """Index of the first obstacle blocking a->b, or -1."""
    for k, obs in enumerate(obstacles):
        if segment_blocked(ax, ay, bx, by, (obs,)):
            return k
    return -1


def obstacle_arrays(obstacles):
    """Pack terrain into (ox, oy, r) float arrays for the vectorised code."""
    if not obstacles:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    arr = np.asarray(obstacles, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def blocked_array(px, py, qx, qy, ox, oy, r):
    """
    Vectorised `segment_blocked` for many segments at once.

    px, py, qx, qy broadcast to a common shape S (e.g. candidates x missiles);
    ox, oy, r are the K obstacles. Returns a bool array of shape S.
    """
    dx = qx - px
    dy = qy - py
    l2 = dx * dx + dy * dy
    ex = ox - px[..., None]
    ey = oy - py[..., None]
    dxk = dx[..., None]
    dyk = dy[..., None]
    l2k = l2[..., None]
    t = np.where(l2k > 1e-12, (ex * dxk + ey * dyk) / np.where(l2k > 1e-12, l2k, 1.0), 0.0)
    t = np.minimum(np.maximum(t, 0.0), 1.0)
    cx = t * dxk - ex
    cy = t * dyk - ey
    return (np.hypot(cx, cy) < r).any(axis=-1)
