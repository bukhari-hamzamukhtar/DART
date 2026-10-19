"""
Hand-written tactical geometry that sits alongside the network.

These are the helpers that used to be nested inside `run_game()` in main3.py
(plus the wall-proximity and cluster-repulsion blocks lifted out of the loop
body).  Nothing here learns anything — it is the deterministic layer the FSM
leans on.
"""

import math

from ..config import HEIGHT, HUD_HEIGHT, WIDTH
from ..utils import normalize_angle
from .entities import FlareParticle, HomingMissile

# ── Wall handling ─────────────────────────────────────────────
WALL_PANIC_DIST  = 110    # closer than this + flying at it => IMMELMANN
WALL_PANIC_SPEED = 1.0

# ── Cluster repulsion ─────────────────────────────────────────
REPULSE_DIST  = 26
REPULSE_FORCE = 0.20

# ── Intercept flare ───────────────────────────────────────────
MIN_INTERCEPT_FRAMES = 5.0
MAX_INTERCEPT_FRAMES = 50.0
FLARE_LEAD_BOOST     = 1.4      # flare must beat the missile to the meeting point
MIN_FLARE_SPEED      = 5.0
MAX_FLARE_SPEED      = 14.0


class WallState(object):
    """Distances to each wall plus the derived escape heading."""

    __slots__ = ("d_left", "d_right", "d_top", "d_bot", "wx", "wy",
                 "escape_heading", "heading_to_wall")

    def __init__(self, d_left, d_right, d_top, d_bot, wx, wy,
                 escape_heading, heading_to_wall):
        self.d_left  = d_left
        self.d_right = d_right
        self.d_top   = d_top
        self.d_bot   = d_bot
        self.wx      = wx                 # margin on the horizontal axis
        self.wy      = wy                 # margin on the vertical axis
        self.escape_heading  = escape_heading
        self.heading_to_wall = heading_to_wall


def compute_wall_state(jet):
    """Where the walls are relative to the jet, and whether it is flying at one."""
    d_left  = jet.x
    d_right = WIDTH - jet.x
    d_top   = jet.y
    d_bot   = (HEIGHT - HUD_HEIGHT) - jet.y
    wx = min(d_left, d_right)
    wy = min(d_top,  d_bot)

    if   d_left  < d_right and d_left  < d_top and d_left  < d_bot: weh = 0.0
    elif d_right < d_left  and d_right < d_top and d_right < d_bot: weh = math.pi
    elif d_top   < d_bot:                                           weh = math.pi / 2
    else:                                                           weh = -math.pi / 2

    htw = False
    if d_left  < WALL_PANIC_DIST and jet.vx < -WALL_PANIC_SPEED: htw = True
    if d_right < WALL_PANIC_DIST and jet.vx >  WALL_PANIC_SPEED: htw = True
    if d_top   < WALL_PANIC_DIST and jet.vy < -WALL_PANIC_SPEED: htw = True
    if d_bot   < WALL_PANIC_DIST and jet.vy >  WALL_PANIC_SPEED: htw = True

    return WallState(d_left, d_right, d_top, d_bot, wx, wy, weh, htw)


def heading_to_center(jet):
    """Heading that points the jet back at the middle of the playfield."""
    return math.atan2(300 - jet.y, 400 - jet.x)


def predict_intercept_flare(jet, m):
    """
    Computes angle+speed to throw a flare into a missile's predicted path.
    Uses time-of-closest-approach math — flare arrives at the same point
    the missile will be at, making the missile lock onto the flare instead.
    """
    dx  = m.x  - jet.x
    dy  = m.y  - jet.y
    rvx = m.vx - jet.vx   # missile vel relative to jet
    rvy = m.vy - jet.vy
    rel_spd_sq = rvx*rvx + rvy*rvy + 1e-5
    # Time of closest approach (when missile is nearest to jet's current pos)
    t = -(dx*rvx + dy*rvy) / rel_spd_sq
    t = max(MIN_INTERCEPT_FRAMES, min(MAX_INTERCEPT_FRAMES, t))   # clamp 5-50 frames
    # Predicted missile world position at time t
    target_x = m.x + m.vx * t
    target_y = m.y + m.vy * t
    # Angle and speed from jet to that prediction
    angle = math.atan2(target_y - jet.y, target_x - jet.x)
    dist  = math.hypot(target_x - jet.x, target_y - jet.y)
    # 1.4x boost so flare beats the missile to the intercept point
    speed = min(max(dist / t * FLARE_LEAD_BOOST, MIN_FLARE_SPEED), MAX_FLARE_SPEED)
    return angle, speed


def spawn_flares(flares, jet, n, arc_start, arc_deg, speed_boost=2.0, stats=None):
    """Eject `n` flares evenly across an arc, and count them in `stats`."""
    angle_step = math.radians(arc_deg) / max(n - 1, 1)
    for i in range(n):
        angle = arc_start + i * angle_step
        flares.append(FlareParticle(jet.x, jet.y, angle, jet.speed + speed_boost))
    if stats is not None:
        stats['flares'] += n
    return n


def apply_cluster_repulsion(missiles):
    """
    Homing missiles push each other apart so they never merge/overlap.

    Pure heading nudges — positions are untouched, so a swarm keeps its formation
    spread instead of collapsing into a single dot.
    """
    homing = [m for m in missiles if isinstance(m, HomingMissile) and not m.dead]
    for i in range(len(homing)):
        for j in range(i + 1, len(homing)):
            m1, m2 = homing[i], homing[j]
            dx = m2.x - m1.x; dy = m2.y - m1.y
            d  = math.hypot(dx, dy)
            if 0 < d < REPULSE_DIST:
                strength = (REPULSE_DIST - d) / REPULSE_DIST * REPULSE_FORCE
                # m1 turns away, m2 turns away in opposite direction
                m1.heading = normalize_angle(m1.heading - strength * math.pi * (1 - d/REPULSE_DIST))
                m2.heading = normalize_angle(m2.heading + strength * math.pi * (1 - d/REPULSE_DIST))
