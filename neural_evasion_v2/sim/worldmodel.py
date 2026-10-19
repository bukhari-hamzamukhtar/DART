"""
The jet's world model: a forward simulator of the arena.

Given the current scene and a set of candidate actions (target heading +
throttle), this rolls the *whole world* forward — the jet under its real drag
physics and steering gain, homing missiles re-aiming at the jet's predicted
position at their real turn rate, ballistic missiles flying straight — and
reports what happens along the way.

That is what turns spatial awareness into spatial reasoning: instead of asking
"is this missile near me right now", the jet asks "if I fly like this and it
flies like that, where do we end up, and do we collide".

Every candidate is simulated at once, and positions/velocities are held as
complex numbers so each 2D operation is one numpy call instead of two. The cost
here is dominated by call overhead on small arrays, and this decision has to be
affordable *every frame*: deciding often beats deciding precisely by a wide
margin — replanning every frame survives 7/8 test scenarios, every third frame
survives 0/8.

Physics mirrors `entities.Jet.update`, `entities.HomingMissile.update` and
`entities.BallisticMissile.update`; `tests/test_worldmodel.py` holds it to zero
divergence from them at step=1.
"""

import math

import numpy as np

from ..config import HEIGHT, HUD_HEIGHT, WIDTH

JET_DRAG        = 0.92
JET_THRUST_GAIN = 0.1
JET_RADIUS      = 11
STEER_GAIN      = 0.18   # matches the NORMAL-state steering in maneuvers.py

TWO_PI = 2.0 * math.pi


def _wrap(a):
    """Vectorised normalize_angle."""
    return (a + math.pi) % TWO_PI - math.pi


class Threats(object):
    """Missile kinematics packed into flat arrays, ready to roll forward."""

    __slots__ = ("pos", "heading", "speed", "turn_rate", "lateral_bias",
                 "spawn_delay", "radius", "n")

    def __init__(self, missiles):
        self.n = len(missiles)
        if not self.n:
            empty = np.zeros(0, dtype=np.float64)
            self.pos = np.zeros(0, dtype=np.complex128)
            self.heading = self.speed = self.turn_rate = empty
            self.lateral_bias = self.spawn_delay = self.radius = empty
            return

        f = lambda key, default: np.array(
            [float(m.get(key, default)) for m in missiles], dtype=np.float64)
        self.pos = np.array([complex(m['pos'][0], m['pos'][1]) for m in missiles],
                            dtype=np.complex128)
        vel = np.array([complex(m['vel'][0], m['vel'][1]) for m in missiles],
                       dtype=np.complex128)
        self.heading = np.array(
            [m.get('heading', math.atan2(m['vel'][1], m['vel'][0])) for m in missiles],
            dtype=np.float64)
        declared = f('speed', 0.0)
        self.speed = np.where(declared > 0.0, declared, np.abs(vel))
        # A ballistic missile has turn_rate 0, so one update rule covers both.
        self.turn_rate    = f('turn_rate', 0.0)
        self.lateral_bias = f('lateral_bias', 0.0)
        self.spawn_delay  = f('spawn_delay', 0.0)
        self.radius       = f('radius', 2.0)


class RolloutResult(object):
    """Per-candidate outcome of a rollout."""

    __slots__ = ("min_dist", "t_min", "hit_frame", "wall_frame", "min_wall",
                 "dist_profile", "wall_profile", "times",
                 "end_x", "end_y", "end_vx", "end_vy",
                 "end_mx", "end_my", "end_mvx", "end_mvy", "end_alive")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _jet_path(jp0, jv0, jh0, th, tr, n_steps, step):
    """
    The jet's trajectory in closed form, for every candidate at once.

    Steering is jh <- jh + wrap(th - jh) * g, so the heading error decays
    geometrically: e_k = e_0 * (1 - g)^k, with no iteration needed. Velocity is
    then a linear recurrence in the heading, which is a cumulative sum once the
    drag is factored out. That replaces 72 Python-level iterations with a
    handful of array operations, and it is exact rather than approximate.

    Returns (positions, velocities, headings), each (C, n_steps).
    """
    g = STEER_GAIN * step
    drag_s = JET_DRAG ** step
    k = np.arange(1, n_steps + 1)

    err0 = _wrap(th - jh0)[:, None]
    heads = th[:, None] - err0 * (1.0 - g) ** k          # (C, n_steps)

    thrust = np.exp(1j * heads) * (tr[:, None] * (JET_THRUST_GAIN * step))
    inv = drag_s ** (-k)
    vel = (drag_s ** k) * (jv0[:, None] + np.cumsum(thrust * inv, axis=1))
    pos = jp0[:, None] + np.cumsum(vel * step, axis=1)
    return pos, vel, heads


def rollout(jet, missiles, target_headings, throttles, horizon,
            lead=0, hold_heading=None, hold_throttle=None, step=1):
    """
    Simulate every (target_heading, throttle) pair forward `horizon` frames.

    `lead` is latency compensation: for the first `lead` frames the jet is
    simulated still flying `hold_heading`/`hold_throttle`, because a plan does
    not take effect the instant it is made. Outcomes inside that window are
    identical for every candidate, so they are excluded from the scores.

    `step` coarsens the simulation — step=2 advances two frames per iteration.
    step=1 is exact against the game, and measurement says it has to be: at
    step=2 the predicted closest approach is off by hundreds of pixels against a
    13 px kill radius.

    The jet's path is closed-form; only the missiles, which chase it, need
    iterating. Returns a RolloutResult carrying the summary (`min_dist`,
    `hit_frame`, `wall_frame`, `min_wall`) and the per-step profiles the cost
    function needs to judge a whole path rather than its single worst instant.
    Frame indices count from the end of the lead window, or -1 for "never".
    """
    th = np.asarray(target_headings, dtype=np.float64)
    tr = np.asarray(throttles, dtype=np.float64)
    C = th.shape[0]
    step = max(1, int(step))
    lead = max(0, int(lead))

    jp0 = np.full(C, complex(jet['pos'][0], jet['pos'][1]), dtype=np.complex128)
    jv0 = np.full(C, complex(jet['vel'][0], jet['vel'][1]), dtype=np.complex128)
    jh0 = np.full(C, float(jet['heading']))

    T = Threats(missiles)
    M = T.n

    # ── the committed window: every candidate flies the same thing ──
    lead_steps = len(range(0, lead, step))
    lead_pos = lead_vel = None
    if lead_steps:
        hold_h = np.full(C, float(jet['heading'] if hold_heading is None else hold_heading))
        hold_t = np.full(C, float(tr[0] if hold_throttle is None else hold_throttle))
        lead_pos, lead_vel, lead_heads = _jet_path(
            jp0, jv0, jh0, hold_h, hold_t, lead_steps, step)
        jp0, jv0, jh0 = lead_pos[:, -1], lead_vel[:, -1], lead_heads[:, -1]

    n_steps = len(range(0, horizon, step))
    path, vel, _ = _jet_path(jp0, jv0, jh0, th, tr, n_steps, step)

    # ── missiles: they chase the jet, so these do need stepping ──
    mp = np.tile(T.pos, (C, 1))
    mh = np.tile(T.heading, (C, 1))
    mv = np.zeros((C, M), dtype=np.complex128)
    threat_path = np.empty((C, M, n_steps), dtype=np.complex128)

    if M:
        delay = np.tile(T.spawn_delay, (C, 1)) - lead
        speed_s = T.speed[None, :] * step
        turn_s = T.turn_rate[None, :] * step
        bias_s = T.lateral_bias[None, :] * step
        any_delay = bool((delay > 0).any())

        if lead_steps:
            # advance the threats through the committed window too
            for k in range(lead_steps):
                aim = np.clip(_wrap(np.angle(lead_pos[:, k][:, None] - mp) - mh),
                              -turn_s, turn_s)
                mh = mh + aim + bias_s
                mp = mp + np.exp(1j * mh) * speed_s

        for k in range(n_steps):
            aim = np.clip(_wrap(np.angle(path[:, k][:, None] - mp) - mh),
                          -turn_s, turn_s)
            if any_delay:
                moving = delay <= 0
                delay = delay - step
                mh = mh + np.where(moving, aim + bias_s, 0.0)
                mv = np.exp(1j * mh) * speed_s
                mp = mp + np.where(moving, mv, 0.0)
            else:
                mh = mh + aim + bias_s
                mv = np.exp(1j * mh) * speed_s
                mp = mp + mv
            threat_path[:, :, k] = mp

    # ── score the paths, all steps at once ────────────────────
    x_lo, x_hi = JET_RADIUS, WIDTH - JET_RADIUS
    y_lo, y_hi = JET_RADIUS, HEIGHT - HUD_HEIGHT - JET_RADIUS
    px, py = path.real, path.imag
    wall_profile = np.minimum(np.minimum(px - x_lo, x_hi - px),
                              np.minimum(py - y_lo, y_hi - py))
    min_wall = wall_profile.min(axis=1)
    breach = wall_profile < 0.0
    wall_frame = np.where(breach.any(axis=1),
                          breach.argmax(axis=1) * step, -1).astype(np.int32)

    far = float(max(WIDTH, HEIGHT))
    if M:
        onscreen = ((threat_path.real >= 0.0) & (threat_path.real <= WIDTH) &
                    (threat_path.imag >= 0.0) & (threat_path.imag <= HEIGHT))
        alive = np.logical_and.accumulate(onscreen, axis=2)
        d = np.where(alive, np.abs(path[:, None, :] - threat_path), np.inf)
        dist_profile = d.min(axis=1)
        min_dist = dist_profile.min(axis=1)
        t_min = (dist_profile.argmin(axis=1) * step).astype(np.int32)
        struck = dist_profile < float((T.radius + JET_RADIUS).min())
        hit_frame = np.where(struck.any(axis=1),
                             struck.argmax(axis=1) * step, -1).astype(np.int32)
        end_alive = alive[:, :, -1]
    else:
        dist_profile = np.full((C, max(1, n_steps)), far)
        min_dist = np.full(C, far)
        t_min = np.zeros(C, dtype=np.int32)
        hit_frame = np.full(C, -1, dtype=np.int32)
        end_alive = np.ones((C, 0), dtype=bool)

    return RolloutResult(
        min_dist=min_dist, t_min=t_min, hit_frame=hit_frame,
        wall_frame=wall_frame, min_wall=min_wall,
        dist_profile=dist_profile, wall_profile=wall_profile,
        times=np.arange(n_steps, dtype=np.float64) * step,
        end_x=px[:, -1], end_y=py[:, -1],
        end_vx=vel[:, -1].real, end_vy=vel[:, -1].imag,
        end_mx=(threat_path[:, :, -1].real if M else np.zeros((C, 0))),
        end_my=(threat_path[:, :, -1].imag if M else np.zeros((C, 0))),
        end_mvx=mv.real, end_mvy=mv.imag, end_alive=end_alive)


def drift_flares(flares, horizon):
    """
    Where the flares will be at the end of the horizon.

    Flares only coast on drag (0.85/frame), so after ~30 frames they are
    effectively parked; ones that expire before the horizon are dropped.
    """
    out = []
    for f in flares:
        if f.get('life', 10 ** 6) <= horizon:
            continue
        x, y = f['pos'][0], f['pos'][1]
        vx, vy = f['vel'][0], f['vel'][1]
        for _ in range(horizon):
            vx *= 0.85
            vy *= 0.85
            x += vx
            y += vy
        out.append({'pos': [x, y], 'vel': [vx, vy], 'type': 2})
    return out
