"""
The v3 world model: jet, missiles, flares, terrain and datalinks rolled forward together.

It extends v2's rollout (`sim/worldmodel.py`, which v1 and v2 still use):

  * terrain   -- the jet's path is scored against peaks as well as walls; a
                 missile that flies into one is gone, and a seeker steers around
                 peaks as the arena makes it;
  * seekers   -- at every imagined step each missile checks whether it can see
                 the jet (range, field of view, line of sight), so the planner
                 can foresee that ducking behind a peak breaks a lock, and that
                 a missile which lost the jet flies to where it last saw it;
  * datalinks -- a missile whose salvo partner holds a lock keeps guiding;
  * styles    -- a flanker steers off to one side, a weaver swings its aim, an
                 erratic missile darts; given the numbers, all of it is exact;
  * flares    -- every flare in the air drifts and burns out; a seeker that sees
                 one nearer than the jet chases it, and a missile that touches
                 one is destroyed with it. A volley the jet has not fired yet
                 goes in the same way, which is how the planner decides whether
                 flaring now is worth a charge from the magazine;
  * endurance -- a missile burns out after its class's flight time;
  * swarms    -- classic homing missiles push apart when they bunch up.

So the imagined future is itself a sequence of dynamic graphs: lock, link and
decoy edges appear and vanish inside the rollout just as they would in the
arena. Each candidate action spawns its own branch of that future.

What goes in per missile comes from two places. Kinematics come from the radar.
Guidance parameters -- turn rate, lead, flank side and angle, datalink group,
last known point, how much it weaves -- come from whoever calls this: the
temporal GNN's estimates for the v3 pilot, or the true values (`oracle_threat`)
for tests and upper bounds. With true values the rollout matches the arena to
rounding error. The GNN cannot know the phase of a weave or the next dart of an
erratic missile, so for those it hands in the missile's estimated swing and the
rollout adds it as a `margin`: the missile counts as that much closer.

Two backends run the same code (`sim/rollout_kernel.py`): "numba", compiled with
candidates in parallel -- a few milliseconds per decision -- and "python", the
same functions run by CPython, exact but far too slow to play on.
"""

import math
import os

import numpy as np

from ..config import (FLARE_LIFE_FRAMES, GUIDANCE_HOMING, GUIDANCE_SMART,
                      HEIGHT, HUD_HEIGHT, MAX_THROTTLE, MIN_THROTTLE, WIDTH)
from . import rollout_kernel as _K
from .doctrine import JINK_TABLE, PINCER_FLANK, jink_value
from .entities import FLARE_DRAG, FLARE_RADIUS
from .tactics import REPULSE_DIST, REPULSE_FORCE
from .worldmodel import JET_DRAG, JET_RADIUS, JET_THRUST_GAIN, STEER_GAIN

HAVE_KERNEL = _K.HAVE_NUMBA and not os.environ.get("NEURAL_EVASION_NO_NUMBA")
if HAVE_KERNEL:
    import numba
    numba.set_num_threads(max(1, min(4, numba.config.NUMBA_NUM_THREADS)))

FAR = 1e18
NEVER = 1e9       # remaining life of something that does not burn out


class ThreatPack(object):
    """Missile state and guidance parameters packed into flat arrays."""

    __slots__ = ("n", "x", "y", "heading", "speed", "turn_rate", "lateral_bias",
                 "spawn_delay", "radius", "lead", "side", "flank", "seeker", "group",
                 "has_lk", "lkx", "lky", "ttl", "repel", "weave_amp", "weave_freq",
                 "weave_phase", "jink_amp", "jink_period", "age", "jink_base",
                 "jink_vals", "margin")

    def __init__(self, threats):
        self.n = n = len(threats)

        def col(key, default):
            out = np.empty(n, dtype=np.float64)
            for i, t in enumerate(threats):
                v = t.get(key)
                out[i] = default if v is None else float(v)
            return out

        self.x = np.array([t['pos'][0] for t in threats], dtype=np.float64)
        self.y = np.array([t['pos'][1] for t in threats], dtype=np.float64)
        vmag = np.array([math.hypot(t['vel'][0], t['vel'][1]) for t in threats], dtype=np.float64)
        self.heading = np.array(
            [t['heading'] if t.get('heading') is not None
             else math.atan2(t['vel'][1], t['vel'][0]) for t in threats],
            dtype=np.float64)
        declared = col('speed', 0.0)
        self.speed = np.maximum(np.where(declared > 0.0, declared, vmag), 1e-6)
        self.turn_rate = col('turn_rate', 0.0)
        self.lateral_bias = col('lateral_bias', 0.0)
        self.spawn_delay = col('spawn_delay', 0.0)
        self.radius = col('radius', 2.0)
        self.lead = col('lead', 0.0)
        self.side = col('side', 0.0)
        self.flank = col('flank', PINCER_FLANK)
        self.seeker = np.array([bool(t.get('seeker', False)) for t in threats], dtype=np.bool_)
        self.group = np.array([-1 if t.get('group') is None else int(t['group'])
                               for t in threats], dtype=np.int64)
        lks = [t.get('last_known') for t in threats]
        self.has_lk = np.array([lk is not None for lk in lks], dtype=np.bool_)
        self.lkx = np.array([lk[0] if lk is not None else 0.0 for lk in lks], dtype=np.float64)
        self.lky = np.array([lk[1] if lk is not None else 0.0 for lk in lks], dtype=np.float64)
        # Frames of flight left after the rollout's first step: the missile
        # takes part in steps 0 .. ttl+1 and is removed at the end of step ttl+1.
        self.ttl = col('ttl', NEVER)
        self.repel = np.array([bool(t.get('repel', False)) for t in threats], dtype=np.bool_)
        self.weave_amp = col('weave_amp', 0.0)
        self.weave_freq = col('weave_freq', 0.0)
        self.weave_phase = col('weave_phase', 0.0)
        self.jink_amp = col('jink_amp', 0.0)
        self.jink_period = np.array([max(1, int(t.get('jink_period') or 1)) for t in threats],
                                    dtype=np.int64)
        self.age = np.array([int(t.get('age') or 0) for t in threats], dtype=np.int64)
        self.jink_base = self.age // self.jink_period
        self.jink_vals = np.zeros((n, JINK_TABLE), dtype=np.float64)
        for i, t in enumerate(threats):
            v = t.get('jink_vals')
            if v:
                k = min(len(v), JINK_TABLE)
                self.jink_vals[i, :k] = v[:k]
        self.margin = col('margin', 0.0)


class RolloutV3(object):
    """Per-candidate outcome of a v3 rollout."""

    __slots__ = ("min_dist", "t_min", "hit_frame", "wall_frame", "min_wall",
                 "dist_profile", "wall_profile", "terrain_profile",
                 "terrain_frame", "min_terrain", "times", "end_x", "end_y",
                 "end_vx", "end_vy", "threat_x", "threat_y", "alive_path",
                 "guided_path", "jet_x", "jet_y", "flare_hit")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def oracle_threat(m, datalink=False, frame0=None):
    """
    A missile's true state, guidance parameters and flight style, as the world
    model takes them. `frame0` is the arena frame the rollout's first step will
    simulate; with it the missile's remaining flight time is included.
    """
    d = {'pos': [m.x, m.y], 'vel': [m.vx, m.vy], 'heading': m.heading,
         'speed': m.speed, 'turn_rate': getattr(m, 'turn_rate', 0.0),
         'lateral_bias': getattr(m, 'lateral_bias', 0.0),
         'spawn_delay': getattr(m, 'spawn_delay', 0), 'radius': m.radius}
    if frame0 is not None:
        d['ttl'] = m.life_frames - (frame0 - m.born_frame)
    mtype = getattr(m, 'mtype', None)
    if mtype == GUIDANCE_HOMING:
        d['repel'] = True
    elif mtype == GUIDANCE_SMART:
        d['seeker'] = True
        d['lead'] = m.lead
        d['group'] = m.network if (datalink and m.network is not None) else -1
        d['last_known'] = m.last_known
    if hasattr(m, 'style'):
        d['side'] = m.side
        d['flank'] = m.flank
        d['weave_amp'] = m.weave_amp
        d['weave_freq'] = m.weave_freq
        d['weave_phase'] = m.weave_phase
        d['jink_amp'] = m.jink_amp
        d['jink_period'] = m.jink_period
        d['age'] = m.age
        if m.jink_amp:
            base = m.age // m.jink_period
            d['jink_vals'] = [jink_value(m.jink_seed, base + j) for j in range(JINK_TABLE)]
    return d


def flare_state(f, frame0):
    """
    A flare in the air, as the world model takes it. `frame0` is the arena frame
    the rollout's first step will simulate. The jet ejected every flare itself,
    so their age is known, not inferred.
    """
    return {'pos': (f.x, f.y), 'vel': (f.vx, f.vy),
            'ttl': FLARE_LIFE_FRAMES - (frame0 - f.born_frame)}


def flare_tracks(flares, T):
    """
    Every flare's position after each step's drift, and whether it still burns.

    A flare ignores everything, so its path is the same in every imagined
    future; only whether a missile uses it up differs, and the rollout tracks
    that. Same arithmetic as FlareParticle.update and Arena._update_flares.
    """
    F = len(flares)
    FX = np.empty((T, F), dtype=np.float64)
    FY = np.empty((T, F), dtype=np.float64)
    FPRES = np.empty((T, F), dtype=np.bool_)
    if F:
        x = np.array([f['pos'][0] for f in flares], dtype=np.float64)
        y = np.array([f['pos'][1] for f in flares], dtype=np.float64)
        vx = np.array([f['vel'][0] for f in flares], dtype=np.float64)
        vy = np.array([f['vel'][1] for f in flares], dtype=np.float64)
        ttl = np.array([f.get('ttl', NEVER) for f in flares], dtype=np.float64)
        for k in range(T):
            vx = vx * FLARE_DRAG
            vy = vy * FLARE_DRAG
            x = x + vx
            y = y + vy
            FX[k] = x
            FY[k] = y
        FPRES[:] = np.arange(T)[:, None] <= ttl[None, :]
    return FX, FY, FPRES


def _first(flags):
    """Index of the first True per row, or -1."""
    return np.where(flags.any(axis=1), flags.argmax(axis=1), -1).astype(np.int32)


def _impl(fn, backend):
    """The compiled kernel, or the same function for CPython."""
    return fn if backend == "numba" else getattr(fn, "py_func", fn)


def rollout(jet, threats, target_headings, throttles, horizon, terrain=(),
            datalink=False, jet_path=None, keep_guided=False, backend=None, flares=()):
    """
    Simulate every (target heading, throttle) candidate `horizon` frames ahead.

    `terrain` is a list of (x, y, r). `datalink` enables lock sharing between
    missiles with the same `group`. `flares` are `flare_state` dicts, in the
    arena's list order (it decides ties). `jet_path=(positions, velocities)`
    replaces the candidate actions with one explicit trajectory (complex arrays,
    one entry per frame) -- used to score the world model against recorded
    flights. `backend` is "numba", "python" or None for the fastest available.
    """
    if backend is None:
        backend = "numba" if HAVE_KERNEL else "python"
    elif backend == "numpy":
        backend = "python"
    if backend == "numba" and not HAVE_KERNEL:
        raise RuntimeError("numba backend requested but numba is not available")

    if terrain:
        arr = np.asarray(terrain, dtype=np.float64)
        ox = np.ascontiguousarray(arr[:, 0])
        oy = np.ascontiguousarray(arr[:, 1])
        orad = np.ascontiguousarray(arr[:, 2])
    else:
        ox = oy = orad = np.zeros(0, dtype=np.float64)
    P = ThreatPack(threats)
    T = horizon if jet_path is None else int(np.asarray(jet_path[0]).size)
    FX, FY, FPRES = flare_tracks(list(flares), T)

    if jet_path is None:
        th = np.ascontiguousarray(np.atleast_1d(np.asarray(target_headings, dtype=np.float64)))
        tr = np.ascontiguousarray(np.atleast_1d(np.asarray(throttles, dtype=np.float64)))
        C = th.shape[0]
        JX = np.empty((C, T))
        JY = np.empty((C, T))
        VX = np.empty((C, T))
        VY = np.empty((C, T))
        _impl(_K.jet_paths, backend)(
            float(jet['pos'][0]), float(jet['pos'][1]), float(jet['vel'][0]),
            float(jet['vel'][1]), float(jet['heading']), th, tr, STEER_GAIN,
            JET_DRAG, JET_THRUST_GAIN, MIN_THROTTLE, MAX_THROTTLE, JX, JY, VX, VY)
    else:
        path = np.asarray(jet_path[0], dtype=np.complex128).reshape(1, -1)
        vel = np.asarray(jet_path[1], dtype=np.complex128).reshape(1, -1)
        C = 1
        JX, JY = np.ascontiguousarray(path.real), np.ascontiguousarray(path.imag)
        VX, VY = np.ascontiguousarray(vel.real), np.ascontiguousarray(vel.imag)

    K = orad.shape[0]
    WALL = np.empty((C, T))
    TERR = np.empty((C, T)) if K else np.empty((1, 1))
    _impl(_K.jet_profiles, backend)(JX, JY, ox, oy, orad, float(WIDTH), float(HEIGHT - HUD_HEIGHT),
                                    float(JET_RADIUS), WALL, TERR, K > 0)

    M = P.n
    F = FX.shape[1]
    TX = np.empty((C, M, T))
    TY = np.empty((C, M, T))
    ALIVE = np.empty((C, M, T), dtype=np.bool_)
    GUIDED = np.empty((C, M, T), dtype=np.bool_) if keep_guided else np.empty((1, 1, 1), dtype=np.bool_)
    FHIT = np.full((C, F), -1, dtype=np.int32)
    if M:
        DIST = np.empty((C, T))
        STRUCK = np.empty((C, T), dtype=np.bool_)
        _impl(_K.missile_rollout, backend)(
            JX, JY, VX, VY, P.x, P.y, P.heading, P.speed, P.turn_rate,
            P.lateral_bias, P.spawn_delay, P.radius, P.lead, P.side, P.flank,
            P.seeker, P.group, P.has_lk, P.lkx, P.lky, P.ttl, P.repel,
            P.weave_amp, P.weave_freq, P.weave_phase, P.jink_amp, P.jink_period,
            P.age, P.jink_base, P.jink_vals, P.margin,
            FX, FY, FPRES, float(FLARE_RADIUS), float(REPULSE_DIST), float(REPULSE_FORCE),
            ox, oy, orad, bool(datalink), float(WIDTH), float(HEIGHT), float(JET_RADIUS),
            TX, TY, ALIVE, GUIDED, bool(keep_guided), DIST, STRUCK, FHIT)
        hit_frame = _first(STRUCK)
    else:
        DIST = np.full((C, T), FAR)
        hit_frame = np.full(C, -1, dtype=np.int32)

    return RolloutV3(
        min_dist=DIST.min(axis=1), t_min=DIST.argmin(axis=1).astype(np.int32),
        hit_frame=hit_frame, wall_frame=_first(WALL < 0.0), min_wall=WALL.min(axis=1),
        dist_profile=DIST, wall_profile=WALL,
        terrain_profile=TERR if K else None,
        terrain_frame=_first(TERR < 0.0) if K else np.full(C, -1, dtype=np.int32),
        min_terrain=TERR.min(axis=1) if K else np.full(C, np.inf),
        times=np.arange(T, dtype=np.float64),
        end_x=JX[:, -1], end_y=JY[:, -1], end_vx=VX[:, -1], end_vy=VY[:, -1],
        threat_x=TX, threat_y=TY, alive_path=ALIVE,
        guided_path=GUIDED if keep_guided else None, jet_x=JX, jet_y=JY, flare_hit=FHIT)
