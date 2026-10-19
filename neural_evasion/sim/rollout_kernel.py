"""
Compiled inner loops of the v3 world model.

The same logic as scalar loops compiled with numba, with candidates spread over
the CPU cores, costs a few milliseconds per decision.

Written to mirror `sim.entities.Jet`, `SmartMissile`, `HomingMissile`,
`BallisticMissile`, `FlareParticle`, `Arena._update_flares`,
`Arena._update_missiles`, `tactics.apply_cluster_repulsion` and the flight-style
and terrain-avoidance rules in `sim.doctrine`, statement for statement -- same
update order, same guidance arithmetic -- so the arena and the jet's imagination
agree to rounding error. Distance tests that the arena does with `hypot` are done
here on squared distances; the two can only disagree on a pair sitting exactly on
a boundary.

Without numba the decorators below do nothing and CPython runs this same code:
exact, but far too slow to plan with. `tests/test_worldmodel_v3.py` checks both
against the arena.
"""

import math

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

from .doctrine import (AVOID_FAR, AVOID_LOOK_MAX, AVOID_LOOK_MIN, AVOID_MARGIN,
                       AVOID_RADII, DECOY_RANGE, FLANK_NEAR, FLANK_SPAN,
                       LINK_RANGE, MAX_TGO, MEMORY_REACHED, SEEKER_COS_FOV,
                       SEEKER_RANGE, WOBBLE_NEAR, WOBBLE_SPAN)

_SR2 = SEEKER_RANGE * SEEKER_RANGE
_DR2 = DECOY_RANGE * DECOY_RANGE
_LR2 = LINK_RANGE * LINK_RANGE
_MR2 = MEMORY_REACHED * MEMORY_REACHED


@njit(cache=True)
def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def _blocked(ax, ay, bx, by, ox, oy, orad2):
    """geometry.segment_blocked on squared distances."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    for j in range(ox.shape[0]):
        ex = ox[j] - ax
        ey = oy[j] - ay
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
        if cx * cx + cy * cy < orad2[j]:
            return True
    return False


@njit(cache=True)
def _aim(mx, my, speed, jx, jy, jvx, jvy, lead, side, flank, wobble):
    """doctrine.guidance_aim."""
    if lead == 0.0 and side == 0.0 and wobble == 0.0:
        return jx, jy
    rx = jx - mx
    ry = jy - my
    d = math.sqrt(rx * rx + ry * ry)
    tgo = d / speed if speed > 1e-6 else MAX_TGO
    if tgo > MAX_TGO:
        tgo = MAX_TGO
    ax = jx + lead * jvx * tgo
    ay = jy + lead * jvy * tgo
    if d > 1e-9 and (side != 0.0 or wobble != 0.0):
        k = 0.0
        if side != 0.0:
            fade = (d - FLANK_NEAR) / FLANK_SPAN
            if fade < 0.0:
                fade = 0.0
            elif fade > 1.0:
                fade = 1.0
            k = side * flank * fade
        if wobble != 0.0:
            wf = (d - WOBBLE_NEAR) / WOBBLE_SPAN
            if wf < 0.0:
                wf = 0.0
            elif wf > 1.0:
                wf = 1.0
            k = k + wobble * wf
        ax = ax - ry * k
        ay = ay + rx * k
    return ax, ay


@njit(cache=True)
def _avoid(mx, my, ch, sh, speed, turn_rate, ox, oy, orad, aim_d):
    """doctrine.avoid_turn."""
    if turn_rate <= 0.0:
        return 0.0
    look = speed / turn_rate * AVOID_RADII + AVOID_LOOK_MIN
    if look > AVOID_LOOK_MAX:
        look = AVOID_LOOK_MAX
    best = AVOID_FAR
    turn = 0.0
    for j in range(ox.shape[0]):
        r = orad[j]
        ex = ox[j] - mx
        ey = oy[j] - my
        along = ex * ch + ey * sh
        if along <= 0.0 or along > look + r or along - r > aim_d:
            continue
        lateral = ey * ch - ex * sh
        reach = r + AVOID_MARGIN
        if lateral >= reach or lateral <= -reach:
            continue
        if along < best:
            best = along
            turn = -1.0 if lateral >= 0.0 else 1.0
    return turn


@njit(cache=True, parallel=True)
def jet_paths(x0, y0, vx0, vy0, h0, targets, throttles, steer_gain, drag,
              thrust_gain, min_t, max_t, JX, JY, VX, VY):
    """The jet under each candidate (target heading, throttle): Jet.update, steered."""
    C, T = JX.shape
    for c in prange(C):
        x, y, vx, vy, h = x0, y0, vx0, vy0, h0
        spd = min(max(throttles[c], min_t), max_t)
        tgt = targets[c]
        for k in range(T):
            h += _wrap(tgt - h) * steer_gain
            vx = vx * drag + math.cos(h) * spd * thrust_gain
            vy = vy * drag + math.sin(h) * spd * thrust_gain
            x += vx
            y += vy
            JX[c, k] = x
            JY[c, k] = y
            VX[c, k] = vx
            VY[c, k] = vy


@njit(cache=True, parallel=True)
def jet_profiles(JX, JY, ox, oy, orad, width, field_h, jet_radius, WALL, TERR, with_terrain):
    """Clearance from the walls and from terrain along every jet path."""
    C, T = JX.shape
    K = ox.shape[0]
    for c in prange(C):
        for k in range(T):
            x = JX[c, k]
            y = JY[c, k]
            w = min(x - jet_radius, width - jet_radius - x)
            w = min(w, y - jet_radius)
            WALL[c, k] = min(w, field_h - jet_radius - y)
            if with_terrain:
                best = 1e18
                for j in range(K):
                    dx = x - ox[j]
                    dy = y - oy[j]
                    v = math.sqrt(dx * dx + dy * dy) - (orad[j] + jet_radius)
                    if v < best:
                        best = v
                TERR[c, k] = best


@njit(cache=True, parallel=True)
def missile_rollout(JX, JY, VX, VY, px0, py0, h0, speed, turn, bias, delay0,
                    radius, lead, side, flank, seeker, group, has_lk0, lkx0, lky0,
                    ttl, repel, weave_amp, weave_freq, weave_phase, jink_amp,
                    jink_period, age0, jink_base, JV, margin,
                    FX, FY, FPRES, flare_radius, repulse_dist, repulse_force,
                    ox, oy, orad, datalink, width, height, jet_radius,
                    TX, TY, ALIVE, GUIDED, keep_guided, DIST, STRUCK, FHIT):
    """
    Roll every missile forward along every candidate jet path.

    Flares do not depend on the jet's action, so their paths come in precomputed:
    FX, FY (T, F) positions after each step's drift and FPRES (T, F) still
    burning. Which flares get used up by a missile does depend on it.

    A weaving or erratic missile's aim swing is computed from its flight age; an
    erratic one's darts come in as JV (M, JINK_TABLE), dart numbers counted from
    jink_base. `margin` (M,) px is subtracted from a missile's distance to the
    jet in DIST and STRUCK only -- an uncertainty tube for missiles whose path
    can only be predicted on average -- and never changes the physics.

    Writes, all per candidate:
      TX, TY, ALIVE (, GUIDED)   (C, M, T)   positions; alive at step k's check
      DIST, STRUCK               (C, T)      nearest live missile; any inside the kill radius
      FHIT                       (C, F)      step a missile destroyed the flare, or -1
    """
    C, T = JX.shape
    M = px0.shape[0]
    K = ox.shape[0]
    F = FX.shape[1]
    orad2 = orad * orad
    # A missile that never turns (a ballistic shot) needs no steering maths:
    # adding a zero turn and zero bias leaves its heading bit-for-bit unchanged.
    # Swarm repulsion can still turn it, so those are never static.
    static = np.empty(M, dtype=np.bool_)
    wobbly = np.empty(M, dtype=np.bool_)
    any_repel = False
    for i in range(M):
        static[i] = turn[i] == 0.0 and bias[i] == 0.0 and not repel[i]
        wobbly[i] = weave_amp[i] != 0.0 or jink_amp[i] != 0.0
        if repel[i]:
            any_repel = True

    for c in prange(C):
        px = np.empty(M)
        py = np.empty(M)
        h = np.empty(M)
        ch = np.empty(M)
        sh = np.empty(M)
        delay = np.empty(M)
        lkx = np.empty(M)
        lky = np.empty(M)
        has_lk = np.empty(M, dtype=np.bool_)
        alive = np.empty(M, dtype=np.bool_)
        own = np.empty(M, dtype=np.bool_)
        guided = np.empty(M, dtype=np.bool_)
        moving = np.empty(M, dtype=np.bool_)
        decoy = np.empty(M, dtype=np.int64)
        age = np.empty(M, dtype=np.int64)
        fdead = np.zeros(F, dtype=np.bool_)
        fkill = np.zeros(F, dtype=np.bool_)
        for i in range(M):
            px[i] = px0[i]
            py[i] = py0[i]
            h[i] = h0[i]
            ch[i] = math.cos(h0[i])
            sh[i] = math.sin(h0[i])
            delay[i] = delay0[i]
            has_lk[i] = has_lk0[i]
            lkx[i] = lkx0[i]
            lky[i] = lky0[i]
            alive[i] = True
            age[i] = age0[i]
        for f in range(F):
            FHIT[c, f] = -1

        for k in range(T):
            jx = JX[c, k]
            jy = JY[c, k]

            # 1. seekers look, before anything moves (SmartMissile.sense): the
            #    jet, and the nearest flare in view -- which wins if the jet is
            #    out of view or farther away
            for i in range(M):
                moving[i] = delay[i] <= 0.0
                o = False
                dec = -1
                if seeker[i] and moving[i] and alive[i]:
                    dx = jx - px[i]
                    dy = jy - py[i]
                    d2 = dx * dx + dy * dy
                    if d2 <= _SR2:
                        if dx * ch[i] + dy * sh[i] >= SEEKER_COS_FOV * math.sqrt(d2):
                            if K == 0 or not _blocked(px[i], py[i], jx, jy, ox, oy, orad2):
                                o = True
                    best = 0.0
                    for f in range(F):
                        if fdead[f] or not FPRES[k, f]:
                            continue
                        ex = FX[k, f] - px[i]
                        ey = FY[k, f] - py[i]
                        fd2 = ex * ex + ey * ey
                        if o and fd2 >= d2:
                            continue
                        if dec >= 0 and fd2 >= best:
                            continue
                        if fd2 <= _DR2 and ex * ch[i] + ey * sh[i] >= SEEKER_COS_FOV * math.sqrt(fd2):
                            if K == 0 or not _blocked(px[i], py[i], FX[k, f], FY[k, f], ox, oy, orad2):
                                dec = f
                                best = fd2
                own[i] = o
                decoy[i] = dec

            # 2. datalinks share locks (Arena._share_locks)
            for i in range(M):
                g = own[i] or not seeker[i]
                if (not g) and datalink and group[i] >= 0 and moving[i] and alive[i]:
                    for j in range(M):
                        if j != i and own[j] and group[j] == group[i]:
                            ex = px[j] - px[i]
                            ey = py[j] - py[i]
                            if ex * ex + ey * ey < _LR2:
                                g = True
                                break
                guided[i] = g

            # 3. steer, move, collide
            dmin = 1e18
            struck = False
            for i in range(M):
                if not alive[i]:
                    # gone from this future: nothing reads its path any more
                    TX[c, i, k] = px[i]
                    TY[c, i, k] = py[i]
                    ALIVE[c, i, k] = False
                    if keep_guided:
                        GUIDED[c, i, k] = False
                    continue
                if not moving[i]:
                    delay[i] -= 1.0
                elif static[i]:
                    px[i] += ch[i] * speed[i]
                    py[i] += sh[i] * speed[i]
                    age[i] += 1
                else:
                    steer = False
                    ax = 0.0
                    ay = 0.0
                    if decoy[i] >= 0:
                        ax = FX[k, decoy[i]]
                        ay = FY[k, decoy[i]]
                        steer = True
                    elif guided[i]:
                        wob = 0.0
                        if wobbly[i]:
                            a = age[i]
                            jv = 0.0
                            if jink_amp[i] != 0.0:
                                jv = JV[i, a // jink_period[i] - jink_base[i]]
                            wob = weave_amp[i] * math.sin(weave_freq[i] * a + weave_phase[i]) + jink_amp[i] * jv
                        ax, ay = _aim(px[i], py[i], speed[i], jx, jy, VX[c, k], VY[c, k],
                                      lead[i], side[i], flank[i], wob)
                        steer = True
                        if seeker[i]:
                            lkx[i] = jx
                            lky[i] = jy
                            has_lk[i] = True
                    elif has_lk[i]:
                        ex = lkx[i] - px[i]
                        ey = lky[i] - py[i]
                        if ex * ex + ey * ey < _MR2:
                            has_lk[i] = False
                        else:
                            ax = lkx[i]
                            ay = lky[i]
                            steer = True
                    tr = turn[i]
                    tn = 0.0
                    aim_d = AVOID_FAR
                    if steer:
                        ex = ax - px[i]
                        ey = ay - py[i]
                        aim_d = math.sqrt(ex * ex + ey * ey)
                        diff = _wrap(math.atan2(ay - py[i], ax - px[i]) - h[i])
                        tn = max(-tr, min(tr, diff))
                    if seeker[i] and K > 0:
                        av = _avoid(px[i], py[i], ch[i], sh[i], speed[i], tr, ox, oy, orad, aim_d)
                        if av != 0.0:
                            tn = av * tr
                    h[i] += tn
                    h[i] += bias[i]
                    ch[i] = math.cos(h[i])
                    sh[i] = math.sin(h[i])
                    px[i] += ch[i] * speed[i]
                    py[i] += sh[i] * speed[i]
                    age[i] += 1

                TX[c, i, k] = px[i]
                TY[c, i, k] = py[i]
                ALIVE[c, i, k] = True
                if keep_guided:
                    GUIDED[c, i, k] = guided[i] and decoy[i] < 0

                # The arena's order: the jet, then flares, then terrain, then
                # the map edge and burn-out. Only a missile that got past the
                # jet can spend itself on a flare.
                dx = jx - px[i]
                dy = jy - py[i]
                d = math.sqrt(dx * dx + dy * dy)
                hr = jet_radius + radius[i]
                de = d - margin[i]
                if de < dmin:
                    dmin = de
                if de < hr:
                    struck = True
                if d < hr:
                    alive[i] = False
                    continue
                gone = False
                rr = flare_radius + radius[i]
                rr2 = rr * rr
                for f in range(F):
                    # a flare used up earlier this step is still in the list
                    if fdead[f] or not FPRES[k, f]:
                        continue
                    ex = FX[k, f] - px[i]
                    ey = FY[k, f] - py[i]
                    if ex * ex + ey * ey < rr2:
                        fkill[f] = True
                        gone = True
                        break
                if not gone:
                    for j in range(K):
                        ex = px[i] - ox[j]
                        ey = py[i] - oy[j]
                        tr2 = orad[j] + radius[i]
                        if ex * ex + ey * ey < tr2 * tr2:
                            gone = True
                            break
                if not gone and moving[i] and (px[i] < 0.0 or px[i] > width
                                               or py[i] < 0.0 or py[i] > height):
                    gone = True
                if not gone and k > ttl[i]:
                    gone = True
                if gone:
                    alive[i] = False
            DIST[c, k] = dmin
            STRUCK[c, k] = struck

            # 4. classic homing missiles push apart (tactics.apply_cluster_repulsion)
            if any_repel:
                for i in range(M):
                    if not (repel[i] and alive[i]):
                        continue
                    for j in range(i + 1, M):
                        if not (repel[j] and alive[j]):
                            continue
                        dd = math.hypot(px[j] - px[i], py[j] - py[i])
                        if 0.0 < dd < repulse_dist:
                            strength = (repulse_dist - dd) / repulse_dist * repulse_force
                            h[i] = _wrap(h[i] - strength * math.pi * (1 - dd / repulse_dist))
                            h[j] = _wrap(h[j] + strength * math.pi * (1 - dd / repulse_dist))
                            ch[i] = math.cos(h[i])
                            sh[i] = math.sin(h[i])
                            ch[j] = math.cos(h[j])
                            sh[j] = math.sin(h[j])

            # 5. flares used up this step are gone from the next one
            for f in range(F):
                if fkill[f]:
                    fkill[f] = False
                    fdead[f] = True
                    FHIT[c, f] = k
