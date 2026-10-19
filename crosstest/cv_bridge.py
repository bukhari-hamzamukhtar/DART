"""
crosstest/cv_bridge.py  --  CausalVis's learned world model inside Neural Evasion's V3 planner
=============================================================================================

WHAT IS PLUGGED IN
The V3 pilot imagines 57-58 candidate flights 72 frames ahead every frame. From its
world model it needs one thing it cannot compute itself: where every missile will be
at every imagined step. `CVWorldModel.rollout` answers that with CausalVis's world
model (G:/CausalVis, class SpatialWorld, checkpoint v6_voxel.pt): a learned pairwise
potential V(gap), forces -dV/dq, a kick-drift-kick step with exact per-object drag
decay. It has the same signature and return type as worldmodel_v3.rollout, so the
planner, its cost and the pilot are untouched. What the jet knows about ITSELF stays
exact, as in V3 -- its own flight under each candidate control, walls and terrain --
so every difference comes from missile prediction.

HOW THE GAME BECOMES A CLEVRER TABLE (fixed before any result was seen)
- Units: the 800 px arena is mapped onto CausalVis's 1.9-unit table (extent +-0.95),
  centred; 1 game frame = 1 CLEVRER frame.
- Objects: every missile. The jet, whose path is PRESCRIBED by the candidate control
  -- an external body, the role the table plays in CausalVis's momentum ledger. Every
  terrain peak, PINNED, the way CausalVis's app pins an object down.
- Attributes, in CausalVis's layout (mass prior, restitution, shape, colour, material,
  radius), mapping "zero": every missile and the jet are rubber spheres -- the CLEVRER
  kind whose learned drag is lowest (0.72% of speed per frame), i.e. the closest to a
  missile that holds its speed -- and peaks are rubber cylinders. Mapping "refit"
  gives each missile kind its own attributes so a fine-tuned attribute encoder can
  tell them apart.
- Rules the game has and CausalVis's physics does not are left out rather than
  bolted on: seekers chasing flares, missiles destroyed by flares or terrain,
  burn-out. With no flares in its imagination the planner never sees a volley pay
  off, so this pilot does not flare.
- CausalVis v7 also applies a voxel contact impulse every frame. It is left out: at
  15 ms a call it would cost about a minute per planning frame here, and a 2 px
  missile is about 3 voxels across on its grid. The learned potential -- which
  v6_voxel was trained with, alongside the voxels -- acts at contact.
- What counts as a hit and as leaving the arena is the game's rule (the kernel's),
  applied to the predicted positions.

EXACTNESS OF THE PORT
V for a pair depends only on their surface gap and their two attribute embeddings,
and no object changes type during a rollout. So each pair of types has a 1-D force
law f'(gap). It is tabulated from CausalVis's own forces_torques on two-object scenes
(1e-5 grid over [-0.6, 0.6], outside which every feature of V is flat) and the same
step is run in numba over all candidates. One of V's input features is
clamp(gap, -0.5, 0.5), so f' JUMPS at gap = +-0.5; linear interpolation across that
cell would smear the jump, so the table also stores f' just outside each kink and the
lookup uses it in the cell beyond. `selftest()` compares the port with a rollout that
calls SpatialWorld.forces_torques directly every step, in float32 (how CausalVis runs)
and float64.
"""

import math
import os
import sys

import numpy as np
import torch
from numba import njit, prange

CV = r"G:/CausalVis"
for _d in ("v5", "src2"):
    _p = os.path.join(CV, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from world import load_spatial, strip_colour                                  # noqa: E402  (CausalVis)

from neural_evasion.config import HEIGHT, HUD_HEIGHT, MAX_THROTTLE, MIN_THROTTLE, WIDTH  # noqa: E402
from neural_evasion.sim import rollout_kernel as _K                           # noqa: E402
from neural_evasion.sim.worldmodel import (JET_DRAG, JET_RADIUS,               # noqa: E402
                                           JET_THRUST_GAIN, STEER_GAIN)
from neural_evasion.sim.worldmodel_v3 import FAR, RolloutV3, _first            # noqa: E402

CHECKPOINT = os.path.join(CV, "v6_voxel.pt")
UNIT = 1.9 / WIDTH                          # table units per px
CX, CY = WIDTH / 2.0, (HEIGHT - HUD_HEIGHT) / 2.0
MISSILE_RADIUS = 2.0
GAP_LO, GAP_HI, GAP_STEP = -0.6, 0.6, 1e-5
KINK = 0.5                                  # V's input clamp(gap, -KINK, KINK)
KNOT_LO = int(round((-KINK - GAP_LO) / GAP_STEP))   # grid node at gap = -0.5
KNOT_HI = int(round((KINK - GAP_LO) / GAP_STEP))    # grid node at gap = +0.5
KINDS = ("homing", "ballistic", "smart")    # type indices 0..2
JET_T = 3                                   # the jet; peaks are JET_T + 1 + p
SHAPES = ("cube", "sphere", "cylinder")
MATERIALS = ("metal", "rubber")

MAPPINGS = {
    "zero": {"homing": ("sphere", "rubber"), "ballistic": ("sphere", "rubber"),
             "smart": ("sphere", "rubber"), "jet": ("sphere", "rubber"),
             "peak": ("cylinder", "rubber")},
    "refit": {"homing": ("sphere", "metal"), "ballistic": ("cube", "metal"),
              "smart": ("sphere", "rubber"), "jet": ("cylinder", "metal"),
              "peak": ("cylinder", "rubber")},
}


def attrs_row(shape, material, radius_px):
    """One object's 16 attributes in CausalVis's layout (colour left zero: stripped anyway)."""
    a = np.zeros(16, np.float32)
    a[0] = 1.5 if material == "metal" else 1.0       # CausalVis's mass prior
    a[1] = 0.8 if material == "metal" else 0.5       # and restitution
    a[2 + SHAPES.index(shape)] = 1.0
    a[13 + MATERIALS.index(material)] = 1.0
    a[15] = radius_px * UNIT
    return a


def missile_kind(threat):
    """oracle_threat marks seekers with 'seeker' and classic homing missiles with 'repel'."""
    if threat.get('seeker'):
        return 2
    if threat.get('repel'):
        return 0
    return 1


def scene_attrs(mapping, terrain):
    rows = [attrs_row(*mapping[k], MISSILE_RADIUS) for k in KINDS]
    rows.append(attrs_row(*mapping["jet"], JET_RADIUS))
    rows += [attrs_row(*mapping["peak"], float(r)) for (_, _, r) in terrain]
    return np.stack(rows)


def force_law(model, phys, mass, radius, e, a, b, gaps):
    """f'(gap) for types a and b, from CausalVis's own force computation: a at the
    origin, b on the +x axis at centre distance gap + contact. The x-force on a is
    -dV/dq_a = f'(gap) exactly."""
    out = np.empty(gaps.shape[0])
    contact = float(radius[0, a] + radius[0, b])
    idx = torch.tensor([a, b])
    for s in range(0, gaps.shape[0], 20000):
        g = torch.from_numpy(gaps[s:s + 20000]).float()
        B = g.shape[0]
        q = torch.zeros(B, 2, 2)
        q[:, 1, 0] = g + contact
        F, _ = model.forces_torques(q, None, mass[:, idx].expand(B, -1), radius[:, idx].expand(B, -1),
                                    e[:, idx].expand(B, -1, -1), torch.ones(B, 2),
                                    phys[:, idx].expand(B, -1, -1), create_graph=False)
        out[s:s + B] = F[:, 0, 0].double().numpy()
    return out


def build_physics(model, mapping, terrain):
    A = scene_attrs(mapping, terrain)
    n_types = A.shape[0]
    phys = strip_colour(torch.from_numpy(A).unsqueeze(0))
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
        c = model.drag_coeff(e, torch.ones_like(mass))
    gaps = np.arange(GAP_LO, GAP_HI + 0.5 * GAP_STEP, GAP_STEP)
    TAB = np.zeros((len(KINDS), n_types, gaps.shape[0]))
    KNOT = np.zeros((len(KINDS), n_types, 2))      # f' just outside -0.5 and just outside +0.5
    contact = np.zeros((len(KINDS), n_types))
    # both one-sided values at each kink, 1e-6 either side (float32 resolves ~6e-8 there)
    sides = np.array([-KINK + 1e-6, KINK - 1e-6, -KINK - 1e-6, KINK + 1e-6])
    for a in range(len(KINDS)):
        for b in range(n_types):
            contact[a, b] = float(radius[0, a] + radius[0, b])
            if b < a:
                TAB[a, b] = TAB[b, a]              # V(i, j) == V(j, i)
                KNOT[a, b] = KNOT[b, a]
                continue
            TAB[a, b] = force_law(model, phys, mass, radius, e, a, b, gaps)
            inside_lo, inside_hi, outside_lo, outside_hi = force_law(model, phys, mass, radius, e, a, b, sides)
            TAB[a, b, KNOT_LO] = inside_lo         # the grid nodes on the kinks take the inside value
            TAB[a, b, KNOT_HI] = inside_hi
            KNOT[a, b] = (outside_lo, outside_hi)
    return {"attrs": A, "mass": mass[0].double().numpy(), "radius": radius[0].double().numpy(),
            "drag": c[0].double().numpy() if c is not None else np.zeros(n_types),
            "contact": contact, "TAB": TAB, "KNOT": KNOT}


def free_physics(terrain):
    n_types = len(KINDS) + 1 + len(terrain)
    return {"attrs": None, "mass": np.ones(n_types), "radius": np.zeros(n_types), "drag": np.zeros(n_types),
            "contact": np.zeros((len(KINDS), n_types)), "TAB": np.zeros((len(KINDS), n_types, 2)),
            "KNOT": np.zeros((len(KINDS), n_types, 2))}


# ── the step, compiled ─────────────────────────────────────────
@njit(cache=True)
def _fprime(TAB, KNOT, a, b, gap, G):
    u = (gap - GAP_LO) / GAP_STEP
    if u <= 0.0 or u >= G - 1.0:
        return 0.0
    i = int(u)
    w = u - i
    lo = TAB[a, b, i]
    hi = TAB[a, b, i + 1]
    if i == KNOT_HI:                 # (0.5, 0.5 + step): the clamp is flat here
        lo = KNOT[a, b, 1]
    elif i + 1 == KNOT_LO:           # (-0.5 - step, -0.5): likewise
        hi = KNOT[a, b, 0]
    return lo + (hi - lo) * w


@njit(cache=True)
def _forces(qx, qy, alive, kind, jx, jy, pkx, pky, contact, TAB, KNOT, fx, fy):
    """Force on every live missile from the jet, every peak and every other live
    missile. Missile pairs are equal and opposite, as in CausalVis."""
    M = qx.shape[0]
    K = pkx.shape[0]
    G = TAB.shape[2]
    for i in range(M):
        fx[i] = 0.0
        fy[i] = 0.0
    for i in range(M):
        if not alive[i]:
            continue
        a = kind[i]
        rx = qx[i] - jx
        ry = qy[i] - jy
        d = math.sqrt(rx * rx + ry * ry + 1e-12)
        fp = _fprime(TAB, KNOT, a, JET_T, d - contact[a, JET_T], G)
        fx[i] -= fp * rx / d
        fy[i] -= fp * ry / d
        for p in range(K):
            rx = qx[i] - pkx[p]
            ry = qy[i] - pky[p]
            d = math.sqrt(rx * rx + ry * ry + 1e-12)
            fp = _fprime(TAB, KNOT, a, JET_T + 1 + p, d - contact[a, JET_T + 1 + p], G)
            fx[i] -= fp * rx / d
            fy[i] -= fp * ry / d
        for j in range(i + 1, M):
            if not alive[j]:
                continue
            rx = qx[i] - qx[j]
            ry = qy[i] - qy[j]
            d = math.sqrt(rx * rx + ry * ry + 1e-12)
            fp = _fprime(TAB, KNOT, a, kind[j], d - contact[a, kind[j]], G)
            fx[i] -= fp * rx / d
            fy[i] -= fp * ry / d
            fx[j] += fp * rx / d
            fy[j] += fp * ry / d


@njit(cache=True, parallel=True)
def cv_missile_rollout(JX, JY, jx0, jy0, px0, py0, vx0, vy0, kind, mass_t, drag_t, contact, TAB, KNOT,
                       pk_x, pk_y, free, width, height, jet_radius, TX, TY, ALIVE, DIST, STRUCK):
    """Every missile along every candidate jet path, CausalVis's step.

    Per frame: half drag, kick with the jet where it is at the start of the frame,
    drift, kick with the jet where the candidate puts it at the end, half drag.
    `free` skips forces and drag: straight lines at constant velocity."""
    C, T = JX.shape
    M = px0.shape[0]
    K = pk_x.shape[0]
    pkx = np.empty(K)
    pky = np.empty(K)
    for p in range(K):
        pkx[p] = (pk_x[p] - CX) * UNIT
        pky[p] = (pk_y[p] - CY) * UNIT
    for c in prange(C):
        qx = np.empty(M)
        qy = np.empty(M)
        mx = np.empty(M)
        my = np.empty(M)
        fx = np.empty(M)
        fy = np.empty(M)
        alive = np.empty(M, dtype=np.bool_)
        for i in range(M):
            m = mass_t[kind[i]]
            qx[i] = (px0[i] - CX) * UNIT
            qy[i] = (py0[i] - CY) * UNIT
            mx[i] = m * vx0[i] * UNIT
            my[i] = m * vy0[i] * UNIT
            alive[i] = True
        for k in range(T):
            if free:
                for i in range(M):
                    if alive[i]:
                        m = mass_t[kind[i]]
                        qx[i] += mx[i] / m
                        qy[i] += my[i] / m
            else:
                if k == 0:
                    jxa = (jx0 - CX) * UNIT
                    jya = (jy0 - CY) * UNIT
                else:
                    jxa = (JX[c, k - 1] - CX) * UNIT
                    jya = (JY[c, k - 1] - CY) * UNIT
                jxb = (JX[c, k] - CX) * UNIT
                jyb = (JY[c, k] - CY) * UNIT
                for i in range(M):
                    if alive[i]:
                        dec = math.exp(-0.5 * drag_t[kind[i]])
                        mx[i] *= dec
                        my[i] *= dec
                _forces(qx, qy, alive, kind, jxa, jya, pkx, pky, contact, TAB, KNOT, fx, fy)
                for i in range(M):
                    if alive[i]:
                        m = mass_t[kind[i]]
                        mx[i] += 0.5 * fx[i]
                        my[i] += 0.5 * fy[i]
                        qx[i] += mx[i] / m
                        qy[i] += my[i] / m
                _forces(qx, qy, alive, kind, jxb, jyb, pkx, pky, contact, TAB, KNOT, fx, fy)
                for i in range(M):
                    if alive[i]:
                        dec = math.exp(-0.5 * drag_t[kind[i]])
                        mx[i] = (mx[i] + 0.5 * fx[i]) * dec
                        my[i] = (my[i] + 0.5 * fy[i]) * dec
            dmin = 1e18
            struck = False
            for i in range(M):
                x = qx[i] / UNIT + CX
                y = qy[i] / UNIT + CY
                TX[c, i, k] = x
                TY[c, i, k] = y
                if not alive[i]:
                    ALIVE[c, i, k] = False
                    continue
                ALIVE[c, i, k] = True
                dx = JX[c, k] - x
                dy = JY[c, k] - y
                d = math.sqrt(dx * dx + dy * dy)
                if d < dmin:
                    dmin = d
                if d < jet_radius + MISSILE_RADIUS:
                    struck = True
                    alive[i] = False
                    continue
                if x < 0.0 or x > width or y < 0.0 or y > height:
                    alive[i] = False
            DIST[c, k] = dmin
            STRUCK[c, k] = struck


class CVWorldModel(object):
    """A drop-in for worldmodel_v3.rollout with CausalVis's physics for the missiles.

    free=True gives the straight-line baseline in the same harness."""

    def __init__(self, model=None, mapping="zero", checkpoint=CHECKPOINT, free=False):
        self.free = free
        self.mapping = MAPPINGS[mapping] if isinstance(mapping, str) else mapping
        if free:
            self.model = None
        else:
            self.model = model if model is not None else load_spatial(checkpoint, yaw_known=False)
        self._cache = {}

    def physics(self, terrain):
        key = tuple(tuple(float(v) for v in o) for o in terrain)
        if key not in self._cache:
            self._cache[key] = (free_physics(terrain) if self.free
                                else build_physics(self.model, self.mapping, terrain))
        return self._cache[key]

    def rollout(self, jet, threats, target_headings, throttles, horizon, terrain=(),
                datalink=False, jet_path=None, keep_guided=False, backend=None, flares=(),
                jet_start=None):
        if terrain:
            arr = np.asarray(terrain, dtype=np.float64)
            ox, oy, orad = (np.ascontiguousarray(arr[:, i]) for i in range(3))
        else:
            ox = oy = orad = np.zeros(0, dtype=np.float64)
        T = horizon if jet_path is None else int(np.asarray(jet_path[0]).size)

        # the jet: exactly as worldmodel_v3.rollout
        if jet_path is None:
            th = np.ascontiguousarray(np.atleast_1d(np.asarray(target_headings, dtype=np.float64)))
            tr = np.ascontiguousarray(np.atleast_1d(np.asarray(throttles, dtype=np.float64)))
            C = th.shape[0]
            JX = np.empty((C, T)); JY = np.empty((C, T)); VX = np.empty((C, T)); VY = np.empty((C, T))
            _K.jet_paths(float(jet['pos'][0]), float(jet['pos'][1]), float(jet['vel'][0]),
                         float(jet['vel'][1]), float(jet['heading']), th, tr, STEER_GAIN,
                         JET_DRAG, JET_THRUST_GAIN, MIN_THROTTLE, MAX_THROTTLE, JX, JY, VX, VY)
            jx0, jy0 = float(jet['pos'][0]), float(jet['pos'][1])
        else:
            path = np.asarray(jet_path[0], dtype=np.complex128).reshape(1, -1)
            vel = np.asarray(jet_path[1], dtype=np.complex128).reshape(1, -1)
            C = 1
            JX, JY = np.ascontiguousarray(path.real), np.ascontiguousarray(path.imag)
            VX, VY = np.ascontiguousarray(vel.real), np.ascontiguousarray(vel.imag)
            jx0, jy0 = (float(jet_start[0]), float(jet_start[1])) if jet_start is not None \
                else (float(JX[0, 0] - VX[0, 0]), float(JY[0, 0] - VY[0, 0]))
        K = orad.shape[0]
        WALL = np.empty((C, T))
        TERR = np.empty((C, T)) if K else np.empty((1, 1))
        _K.jet_profiles(JX, JY, ox, oy, orad, float(WIDTH), float(HEIGHT - HUD_HEIGHT),
                        float(JET_RADIUS), WALL, TERR, K > 0)

        # the missiles: CausalVis's physics
        M = len(threats)
        F = len(flares)
        TX = np.empty((C, M, T)); TY = np.empty((C, M, T))
        ALIVE = np.empty((C, M, T), dtype=np.bool_)
        if M:
            ph = self.physics(terrain)
            px0 = np.array([t['pos'][0] for t in threats], dtype=np.float64)
            py0 = np.array([t['pos'][1] for t in threats], dtype=np.float64)
            vx0 = np.array([t['vel'][0] for t in threats], dtype=np.float64)
            vy0 = np.array([t['vel'][1] for t in threats], dtype=np.float64)
            kind = np.array([missile_kind(t) for t in threats], dtype=np.int64)
            DIST = np.empty((C, T))
            STRUCK = np.empty((C, T), dtype=np.bool_)
            cv_missile_rollout(JX, JY, jx0, jy0, px0, py0, vx0, vy0, kind, ph["mass"], ph["drag"],
                               ph["contact"], ph["TAB"], ph["KNOT"], ox, oy, bool(self.free),
                               float(WIDTH), float(HEIGHT), float(JET_RADIUS), TX, TY, ALIVE, DIST, STRUCK)
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
            guided_path=None, jet_x=JX, jet_y=JY, flare_hit=np.full((C, F), -1, dtype=np.int32))


# ── the reference the port is checked against ─────────────────
def torch_reference(model, attrs_types, jet_start, jet_xy, threats, terrain, dtype=torch.float32):
    """The same step written with SpatialWorld.forces_torques called on the whole scene
    every half-step (and its shared force cap), with the game's removals. One candidate.
    `model` must already be in `dtype`: float32 is how CausalVis runs it; float64
    separates the port's arithmetic from float32 rounding."""
    kinds = [missile_kind(t) for t in threats]
    M, Kp = len(threats), len(terrain)
    types = kinds + [JET_T] + [JET_T + 1 + p for p in range(Kp)]
    phys = strip_colour(torch.from_numpy(attrs_types[types]).unsqueeze(0).to(dtype))
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
        c = model.drag_coeff(e, torch.ones_like(mass))
    N = M + 1 + Kp
    to_u = lambda x, y: [(x - CX) * UNIT, (y - CY) * UNIT]
    q = torch.tensor([to_u(*t['pos']) for t in threats] + [to_u(*jet_start)]
                     + [to_u(ox, oy) for ox, oy, _ in terrain], dtype=dtype).unsqueeze(0)
    v = torch.zeros(1, N, 2, dtype=dtype)
    v[0, :M] = torch.tensor([[t['vel'][0] * UNIT, t['vel'][1] * UNIT] for t in threats], dtype=dtype)
    p = mass.unsqueeze(-1) * v
    act = torch.ones(1, N, dtype=dtype)
    ext = torch.zeros(1, N, 1, dtype=dtype)
    ext[:, M:] = 1.0
    fixed = q.clone()
    dec = torch.exp(-0.5 * c).unsqueeze(-1)
    out = np.empty((M, len(jet_xy), 2))
    maxf = 0.0
    for k, (jx, jy) in enumerate(jet_xy):
        p = p * dec
        F, _ = model.forces_torques(q.detach(), None, mass, radius, e, act, phys, create_graph=False)
        maxf = max(maxf, float(F.abs().max()))
        p = p + 0.5 * F.detach()
        q = q + p / mass.unsqueeze(-1)
        target = fixed.clone()
        target[0, M] = torch.tensor(to_u(jx, jy), dtype=dtype)
        q = q * (1 - ext) + target * ext
        F, _ = model.forces_torques(q.detach(), None, mass, radius, e, act, phys, create_graph=False)
        maxf = max(maxf, float(F.abs().max()))
        p = (p + 0.5 * F.detach()) * dec
        p = p * (1 - ext)
        out[:, k] = q[0, :M].detach().double().numpy() / UNIT + np.array([CX, CY])
        # the game's removals, as in the port: a missile that reaches the jet or leaves
        # the arena is gone -- absent from V from the next frame, and frozen where it was
        for i in range(M):
            if act[0, i] == 0:
                continue
            x, y = out[i, k]
            if math.hypot(jx - x, jy - y) < JET_RADIUS + MISSILE_RADIUS or not (0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT):
                act[0, i] = 0.0
                p[0, i] = 0.0
    return out, maxf


def selftest(n_scenes=12, steps=72, seed=0):
    import copy
    from neural_evasion.levels import LEVELS
    rng = np.random.default_rng(seed)
    model = load_spatial(CHECKPOINT, yaw_known=False)
    model64 = copy.deepcopy(model).double()
    worst_all = 0.0
    ok = True
    for mapping in ("zero", "refit"):
        wm = CVWorldModel(model=model, mapping=mapping)
        for level in LEVELS:
            terrain = [tuple(o) for o in level.obstacles]
            ph = wm.physics(terrain)
            worst, worst64, maxf = 0.0, 0.0, 0.0
            for s in range(n_scenes):
                M = int(rng.integers(2, 9))
                jet = np.array([rng.uniform(250, 550), rng.uniform(180, 370)])
                jv = rng.normal(0, 3, 2)
                jet_xy = [(jet[0] + jv[0] * (k + 1) + 8 * math.sin(0.05 * k), jet[1] + jv[1] * (k + 1))
                          for k in range(steps)]
                threats = []
                for i in range(M):
                    if s % 3 == 0 and i < 2:            # a close pass: jet within the potential's reach
                        ang = rng.uniform(0, 2 * math.pi)
                        pos = (jet[0] + 60 * math.cos(ang), jet[1] + 60 * math.sin(ang))
                        vel = (-5 * math.cos(ang) + jv[0], -5 * math.sin(ang) + jv[1])
                    else:
                        pos = (rng.uniform(80, 720), rng.uniform(60, 490))
                        ang = rng.uniform(0, 2 * math.pi)
                        sp = rng.uniform(3, 7)
                        vel = (sp * math.cos(ang), sp * math.sin(ang))
                    threats.append({'pos': pos, 'vel': vel, 'seeker': i % 3 == 2, 'repel': i % 3 == 0})
                ref, mf = torch_reference(model, ph["attrs"], tuple(jet), jet_xy, threats, terrain)
                ref64, _ = torch_reference(model64, ph["attrs"], tuple(jet), jet_xy, threats, terrain,
                                           dtype=torch.float64)
                maxf = max(maxf, mf)
                path = np.array([complex(x, y) for x, y in jet_xy])
                res = wm.rollout(None, threats, None, None, steps, terrain=terrain,
                                 jet_path=(path, np.zeros_like(path)), jet_start=tuple(jet))
                X = np.stack([res.threat_x[0], res.threat_y[0]], axis=-1)      # [M, T, 2], removals included
                worst = max(worst, float(np.linalg.norm(X - ref, axis=-1).max()))
                worst64 = max(worst64, float(np.linalg.norm(X - ref64, axis=-1).max()))
            print("%-6s L%d  numba port vs SpatialWorld.forces_torques, %d scenes x %d frames: "
                  "max position diff %.2e px vs float64, %.2e px vs float32 (CausalVis's own precision)   "
                  "(largest force %.2e, cap 50)" % (mapping, level.number, n_scenes, steps, worst64, worst, maxf))
            worst_all = max(worst_all, worst64)
            ok = ok and worst64 < 1e-2 and maxf < 50.0
    print("SELFTEST", "PASS" if ok else "FAIL", "(worst %.2e px)" % worst_all)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
