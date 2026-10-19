"""
The v3 world model must predict the arena, not an approximation of it.

Given every missile's true guidance parameters and flight style (the oracle),
rolling the world model forward and stepping the real arena must agree to
rounding error -- in the open, among terrain where seekers lose lock, steer
around peaks and crash, in a datalinked pincer, among flares that seduce seekers
and destroy missiles, in a classic swarm that pushes itself apart and burns out,
and for flankers, weavers and erratic missiles. Whatever the v3 pilot then gets
wrong is estimation error in its inputs, which is measured separately.

Runs for both backends (the compiled numba kernel, and the same code run by
CPython), and checks the scenarios actually exercise lock losses, memory
flights, datalink guidance, decoys, flare kills, repulsion, burn-out, terrain
kills, every flight style and terrain avoidance, so agreement is not just
agreement on the easy case.
"""
import math
import os
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from neural_evasion.arena import Arena                                  # noqa: E402
from neural_evasion.levels import LEVEL_1, LEVEL_2, LEVEL_3             # noqa: E402
from neural_evasion.sim import worldmodel_v3                            # noqa: E402
from neural_evasion.sim.doctrine import (ERRATIC, FLANKER, STYLES,      # noqa: E402
                                         WEAVER, sample_style)
from neural_evasion.sim.entities import (BallisticMissile,              # noqa: E402
                                         FlareParticle, HomingMissile,
                                         SmartMissile, apply_style)
from neural_evasion.sim.geometry import inside_any                      # noqa: E402
from neural_evasion.sim.tactics import REPULSE_DIST                     # noqa: E402
from neural_evasion.sim.worldmodel import STEER_GAIN                    # noqa: E402
from neural_evasion.sim.worldmodel_v3 import flare_state, oracle_threat  # noqa: E402
from neural_evasion.utils import normalize_angle                        # noqa: E402

HORIZON = 72
TOL = 1e-6
failures = []
BACKENDS = ["numba", "python"] if worldmodel_v3.HAVE_KERNEL else ["python"]


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        failures.append(name)


class FixedAction(object):
    """A pilot that flies one (target heading, throttle) for the whole run."""

    def __init__(self, heading, throttle):
        self.heading, self.throttle = heading, throttle

    def act(self, arena):
        jet = arena.jet
        jet.heading += normalize_angle(self.heading - jet.heading) * STEER_GAIN
        jet.set_throttle(self.throttle)


class Outcome(object):
    pass


def compare(arena, target, throttle, backend, horizon=HORIZON):
    """Roll the world model, step the arena; return divergence and what happened."""
    jet = arena.jet
    frame0 = arena.frame + 1          # the rollout's first step is the arena's next frame
    state = {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy], 'heading': jet.heading}
    threats = [oracle_threat(m, arena.level.datalink, frame0) for m in arena.missiles]
    objs = {m.uid: m for m in arena.missiles}
    uids = [m.uid for m in arena.missiles]
    live_flares = [f for f in arena.flares if not f.dead]
    fuids = [f.uid for f in live_flares]
    res = worldmodel_v3.rollout(state, threats, [target], [throttle], horizon,
                                terrain=arena.terrain, datalink=arena.level.datalink,
                                keep_guided=True, backend=backend,
                                flares=[flare_state(f, frame0) for f in live_flares])

    pilot = FixedAction(target, throttle)
    out = Outcome()
    out.modes, out.lock_losses, out.guided_mismatch = set(), 0, 0
    out.expired, out.repulsion_frames, out.avoid_frames = 0, 0, 0
    out.styles = set(getattr(m, 'style', None) for m in arena.missiles) - {None}
    prev, burnt, present_before = {}, {}, set(uids)
    for k in range(horizon):
        arena.begin_frame()
        arena.advance(pilot)
        present = {m.uid: m for m in arena.missiles}
        for f in arena.flares:
            if f.dead and f.uid not in burnt:
                burnt[f.uid] = k
        for uid in present_before - set(present):
            m = objs.get(uid)
            if m is not None and arena.frame - m.born_frame > m.life_frames:
                out.expired += 1
        present_before = set(present)
        out.avoid_frames += sum(1 for m in arena.missiles if getattr(m, 'avoiding', False))
        homing = [m for m in arena.missiles if isinstance(m, HomingMissile)]
        if any(0 < math.hypot(a.x - b.x, a.y - b.y) < REPULSE_DIST
               for n, a in enumerate(homing) for b in homing[n + 1:]):
            out.repulsion_frames += 1
        for i, uid in enumerate(uids):
            m = present.get(uid)
            if m is None or not isinstance(m, SmartMissile):
                continue
            out.modes.add(m.mode)
            g = m.mode in ("LOCK", "LINK")
            if prev.get(uid) and not g:
                out.lock_losses += 1
            prev[uid] = g
            if m.spawn_delay == 0 and bool(res.guided_path[0, i, k]) != g:
                out.guided_mismatch += 1

    out.worst, out.alive_mismatch = 0.0, 0
    present = {m.uid: m for m in arena.missiles}
    for i, uid in enumerate(uids):
        m = present.get(uid)
        ex, ey = res.threat_x[0, i, -1], res.threat_y[0, i, -1]
        predicted_alive = (bool(res.alive_path[0, i, -1]) and 0 <= ex <= 800 and 0 <= ey <= 600
                           and not inside_any(ex, ey, arena.terrain, 2.0))
        if m is None:
            # a missile that hit the jet is gone from the arena but may still be
            # in the rollout's future; only an unexplained disappearance counts
            if predicted_alive and res.hit_frame[0] < 0:
                out.alive_mismatch += 1
            continue
        if not predicted_alive:
            out.alive_mismatch += 1
            continue
        out.worst = max(out.worst, math.hypot(m.x - ex, m.y - ey))

    # every flare a missile used up, on the frame it happened
    out.flare_mismatch = sum(int(res.flare_hit[0, j]) != burnt.get(uid, -1)
                             for j, uid in enumerate(fuids))
    out.burnt = len(burnt)
    out.res = res
    return out


def add_flare(arena, rng, x, y, max_age=115):
    f = FlareParticle(x, y, rng.uniform(-3.14, 3.14), rng.uniform(0.0, 6.0))
    f.born_frame = arena.frame - rng.randint(0, max_age)
    f.dead = False
    arena.flares.append(f)


def place_jet(a, rng, x_lo=150, x_hi=650, y_lo=100, y_hi=450, clear=45):
    while True:
        a.jet.x, a.jet.y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
        if not inside_any(a.jet.x, a.jet.y, a.terrain, clear):
            break
    a.jet.heading = rng.uniform(-3, 3)
    a.jet.vx, a.jet.vy = rng.uniform(-4, 4), rng.uniform(-4, 4)


def suite(backend):
    rng = random.Random(11)
    tag = "[%s] " % backend

    # A. open sky, classic missiles
    worst, mism = 0.0, 0
    for _ in range(25):
        a = Arena(LEVEL_1, counter_attack=False, visuals=False)
        a.jet.x, a.jet.y = rng.uniform(200, 600), rng.uniform(150, 400)
        a.jet.heading = rng.uniform(-3, 3)
        a.jet.vx, a.jet.vy = rng.uniform(-4, 4), rng.uniform(-4, 4)
        for _ in range(rng.randint(1, 4)):
            mx, my = rng.uniform(40, 760), rng.uniform(40, 520)
            if rng.random() < 0.6:
                m = HomingMissile(mx, my, a.jet, a.jet.x, a.jet.y)
                m.speed = m.max_speed = rng.uniform(3.5, 6.5)
                m.turn_rate = rng.uniform(0.03, 0.15)
                m.lateral_bias = rng.choice([0.0, 0.007, -0.014])
            else:
                m = BallisticMissile(mx, my, a.jet.x, a.jet.y)
            a._add_missile(m)
        o = compare(a, a.jet.heading + rng.uniform(-1.5, 1.5),
                    rng.choice([1.75, 3.5, 5.0, 7.5]), backend, 40)
        worst, mism = max(worst, o.worst), mism + o.alive_mismatch
    check(tag + "A  open sky: classic missiles match the arena", worst < TOL and mism == 0,
          "max divergence %.2e px, alive mismatches %d" % (worst, mism))

    # B. terrain: seekers lose lock, fly to memory, steer around peaks, crash
    worst, mism, gm, losses, modes_all, removed = 0.0, 0, 0, 0, set(), 0
    for _ in range(40):
        a = Arena(LEVEL_2, counter_attack=False, visuals=False)
        while True:
            a.jet.x, a.jet.y = rng.uniform(80, 720), rng.uniform(80, 470)
            if not inside_any(a.jet.x, a.jet.y, a.terrain, 40):
                break
        a.jet.heading = rng.uniform(-3, 3)
        a.jet.vx, a.jet.vy = rng.uniform(-4, 4), rng.uniform(-4, 4)
        n = rng.randint(2, 5)
        for _ in range(n):
            while True:
                mx, my = rng.uniform(30, 770), rng.uniform(30, 540)
                if not inside_any(mx, my, a.terrain, 10):
                    break
            m = SmartMissile(mx, my, a.jet, a.jet.x, a.jet.y,
                             speed=rng.uniform(3.8, 6.2), turn_rate=rng.uniform(0.04, 0.15),
                             launch_heading=math.atan2(a.jet.y - my, a.jet.x - mx) + rng.uniform(-1.2, 1.2),
                             spawn_delay=rng.choice([0, 0, 0, 3]))
            m.last_known = (rng.uniform(50, 750), rng.uniform(50, 500)) if rng.random() < 0.7 else None
            a._add_missile(m)
        o = compare(a, a.jet.heading + rng.uniform(-2, 2), rng.choice([2.0, 3.5, 5.0, 7.5]), backend)
        removed += n - len(a.missiles)
        worst, mism, gm, losses = max(worst, o.worst), mism + o.alive_mismatch, gm + o.guided_mismatch, losses + o.lock_losses
        modes_all |= o.modes
    check(tag + "B  terrain: seeker missiles match the arena", worst < TOL and mism == 0 and gm == 0,
          "max divergence %.2e px, alive mismatches %d, lock-state mismatches %d" % (worst, mism, gm))
    check(tag + "B' terrain scenarios exercise lock loss, memory flight and removals",
          losses > 0 and "MEMORY" in modes_all and removed > 0,
          "lock losses %d, modes seen %s, missiles removed %d" % (losses, sorted(modes_all), removed))

    # C. datalinked pincer
    worst, mism, gm, modes_all = 0.0, 0, 0, set()
    for _ in range(30):
        a = Arena(LEVEL_3, counter_attack=False, visuals=False)
        a.jet.x, a.jet.y = rng.uniform(250, 550), rng.uniform(150, 350)
        a.jet.heading = rng.uniform(-3, 3)
        a.jet.vx, a.jet.vy = rng.uniform(-4, 4), rng.uniform(-4, 4)
        for net in range(rng.randint(1, 3)):
            for _ in range(rng.randint(2, 3)):
                mx, my = rng.uniform(30, 770), rng.uniform(30, 540)
                a._add_missile(SmartMissile(
                    mx, my, a.jet, a.jet.x, a.jet.y,
                    speed=rng.uniform(4.0, 6.0), turn_rate=rng.uniform(0.05, 0.14),
                    lead=rng.uniform(0.0, 0.8), side=rng.choice([-1.0, 0.0, 1.0]),
                    network=100 + net, launch_heading=rng.uniform(-3.14, 3.14)))
        o = compare(a, a.jet.heading + rng.uniform(-2, 2), rng.choice([2.0, 5.0, 7.5]), backend)
        worst, mism, gm = max(worst, o.worst), mism + o.alive_mismatch, gm + o.guided_mismatch
        modes_all |= o.modes
    check(tag + "C  datalink pincer: networked missiles match the arena",
          worst < TOL and mism == 0 and gm == 0,
          "max divergence %.2e px, alive mismatches %d, lock-state mismatches %d" % (worst, mism, gm))
    check(tag + "C' pincer scenarios exercise datalink guidance", "LINK" in modes_all,
          "modes seen %s" % sorted(modes_all))

    # D. terrain is lethal to the jet, and the rollout knows when
    a = Arena(LEVEL_2, counter_attack=False, visuals=False)
    ox, oy, r = LEVEL_2.obstacles[0]
    a.jet.x, a.jet.y = ox - r - 90.0, oy
    a.jet.heading, a.jet.vx, a.jet.vy = 0.0, 5.0, 0.0
    state = {'pos': [a.jet.x, a.jet.y], 'vel': [a.jet.vx, a.jet.vy], 'heading': 0.0}
    # candidate 0 flies straight at the peak; candidate 1 turns back the way it came
    res = worldmodel_v3.rollout(state, [], [0.0, math.pi], [5.0, 5.0], HORIZON,
                                terrain=a.terrain, backend=backend)
    pilot = FixedAction(0.0, 5.0)
    crash = None
    for k in range(HORIZON):
        a.begin_frame()
        a.advance(pilot)
        if a.game_over:
            crash = k
            break
    check(tag + "D  flying into a peak is predicted on the frame it happens",
          res.terrain_frame[0] >= 0 and crash == res.terrain_frame[0],
          "predicted frame %d, arena frame %s" % (res.terrain_frame[0], crash))
    check(tag + "D' turning away from the peak is predicted to survive", res.terrain_frame[1] < 0,
          "closest terrain clearance %.1f px" % res.min_terrain[1])

    # E. flares among seekers: seduction, flare kills, burn-out, under terrain and datalinks
    worst, mism, gm, fm, modes_all, burnt = 0.0, 0, 0, 0, set(), 0
    for case in range(40):
        level = LEVEL_2 if case % 2 == 0 else LEVEL_3
        a = Arena(level, counter_attack=False, visuals=False)
        place_jet(a, rng, 150, 650, 120, 430, 40)
        for _ in range(rng.randint(2, 5)):
            ang, dist = rng.uniform(-3.14, 3.14), rng.uniform(110, 320)
            mx = min(max(a.jet.x + math.cos(ang) * dist, 20.0), 780.0)
            my = min(max(a.jet.y + math.sin(ang) * dist, 20.0), 530.0)
            if inside_any(mx, my, a.terrain, 10):
                continue
            m = SmartMissile(mx, my, a.jet, a.jet.x, a.jet.y,
                             speed=rng.uniform(3.8, 6.2), turn_rate=rng.uniform(0.05, 0.15),
                             lead=rng.uniform(0.0, 0.8) if level.lead else 0.0,
                             side=rng.choice([-1.0, 0.0, 1.0]) if level.pincer else 0.0,
                             network=(100 + rng.randint(0, 1)) if level.datalink else None,
                             launch_heading=math.atan2(a.jet.y - my, a.jet.x - mx) + rng.uniform(-0.6, 0.6))
            a._add_missile(m)
            for _ in range(rng.randint(1, 3)):      # flares ahead of it, some about to burn out
                s = rng.uniform(15.0, dist * 0.8)
                add_flare(a, rng, mx + math.cos(m.heading) * s + rng.uniform(-25, 25),
                          my + math.sin(m.heading) * s + rng.uniform(-25, 25))
        o = compare(a, a.jet.heading + rng.uniform(-2, 2), rng.choice([2.0, 3.5, 5.0, 7.5]), backend)
        worst, mism, gm, fm = (max(worst, o.worst), mism + o.alive_mismatch,
                               gm + o.guided_mismatch, fm + o.flare_mismatch)
        burnt += o.burnt
        modes_all |= o.modes
    check(tag + "E  flares: seekers, decoys and flare kills match the arena",
          worst < TOL and mism == 0 and gm == 0 and fm == 0,
          "max divergence %.2e px, alive mismatches %d, lock-state mismatches %d, flare mismatches %d"
          % (worst, mism, gm, fm))
    check(tag + "E' flare scenarios exercise decoys and flare kills", "DECOY" in modes_all and burnt > 0,
          "modes seen %s, flares used up %d" % (sorted(modes_all), burnt))

    # F. classic swarm: repulsion, burn-out, flares destroying unguided and homing missiles
    worst, mism, fm, burnt, expired, repelled = 0.0, 0, 0, 0, 0, 0
    for _ in range(25):
        a = Arena(LEVEL_1, counter_attack=False, visuals=False, seed=rng.randint(0, 10 ** 6))
        a.jet.x, a.jet.y = rng.uniform(250, 550), rng.uniform(150, 350)
        a.jet.heading = rng.uniform(-3, 3)
        a.launch_cluster()
        warm = FixedAction(a.jet.heading, 3.5)
        for _ in range(rng.randint(0, 14)):
            a.begin_frame()
            a.advance(warm)
        for _ in range(rng.randint(1, 3)):          # old missiles near the end of their flight
            mx, my = rng.uniform(40, 760), rng.uniform(40, 520)
            if rng.random() < 0.6:
                m = HomingMissile(mx, my, a.jet, a.jet.x, a.jet.y)
                m.turn_rate = rng.uniform(0.03, 0.15)
                a._add_missile(m)
                m.born_frame = a.frame - rng.randint(450, 505)
            else:
                m = BallisticMissile(mx, my, a.jet.x, a.jet.y)
                a._add_missile(m)
                m.born_frame = a.frame - rng.randint(120, 175)
        for m in list(a.missiles):
            if rng.random() < 0.5:                   # a flare across its path
                s = rng.uniform(20.0, 90.0)
                add_flare(a, rng, m.x + math.cos(m.heading) * s, m.y + math.sin(m.heading) * s)
        o = compare(a, a.jet.heading + rng.uniform(-1.5, 1.5), rng.choice([2.0, 3.5, 5.0]), backend)
        worst, mism, fm = max(worst, o.worst), mism + o.alive_mismatch, fm + o.flare_mismatch
        burnt, expired, repelled = burnt + o.burnt, expired + o.expired, repelled + o.repulsion_frames
    check(tag + "F  classic swarm: repulsion, burn-out and flare kills match the arena",
          worst < TOL and mism == 0 and fm == 0,
          "max divergence %.2e px, alive mismatches %d, flare mismatches %d" % (worst, mism, fm))
    check(tag + "F' swarm scenarios exercise repulsion, burn-out and flare kills",
          repelled > 0 and expired > 0 and burnt > 0,
          "frames with repulsion %d, missiles burnt out %d, flares used up %d" % (repelled, expired, burnt))

    # G. flight styles, and seekers steering around peaks
    worst, mism, gm, styles_seen, avoid_frames = 0.0, 0, 0, set(), 0
    for case in range(45):
        level = (LEVEL_1, LEVEL_2, LEVEL_3)[case % 3]
        a = Arena(level, counter_attack=False, visuals=False)
        place_jet(a, rng)
        for _ in range(rng.randint(2, 5)):
            mx = my = None
            if a.terrain and rng.random() < 0.6:
                # behind a peak as seen from the jet, so its way in runs over the peak
                ox, oy, r = rng.choice(a.terrain)
                ux, uy = ox - a.jet.x, oy - a.jet.y
                n = math.hypot(ux, uy) or 1.0
                dist = r + rng.uniform(40.0, 110.0)
                mx = ox + ux / n * dist + rng.uniform(-25.0, 25.0)
                my = oy + uy / n * dist + rng.uniform(-25.0, 25.0)
            if mx is None or not (20 < mx < 780 and 20 < my < 530) or inside_any(mx, my, a.terrain, 10):
                while True:
                    mx, my = rng.uniform(30, 770), rng.uniform(30, 540)
                    if not inside_any(mx, my, a.terrain, 10):
                        break
            aim = math.atan2(a.jet.y - my, a.jet.x - mx)
            if level.smart:
                m = SmartMissile(mx, my, a.jet, a.jet.x, a.jet.y,
                                 speed=rng.uniform(3.8, 6.2), turn_rate=rng.uniform(0.05, 0.15),
                                 lead=level.lead,
                                 network=(100 + rng.randint(0, 1)) if level.datalink else None,
                                 launch_heading=aim + rng.uniform(-0.8, 0.8))
            else:
                m = HomingMissile(mx, my, a.jet, a.jet.x, a.jet.y)
                m.speed = m.max_speed = rng.uniform(3.5, 6.5)
                m.turn_rate = rng.uniform(0.04, 0.15)
            apply_style(m, sample_style(rng, level.number, rng.choice(STYLES)))
            m.age = rng.randint(0, 200)
            a._add_missile(m)
        o = compare(a, a.jet.heading + rng.uniform(-2, 2), rng.choice([2.0, 3.5, 5.0, 7.5]), backend)
        worst, mism, gm = max(worst, o.worst), mism + o.alive_mismatch, gm + o.guided_mismatch
        styles_seen |= o.styles
        avoid_frames += o.avoid_frames
    check(tag + "G  flight styles and terrain avoidance match the arena",
          worst < TOL and mism == 0 and gm == 0,
          "max divergence %.2e px, alive mismatches %d, lock-state mismatches %d" % (worst, mism, gm))
    check(tag + "G' style scenarios exercise flankers, weavers, erratic darts and avoidance",
          {FLANKER, WEAVER, ERRATIC} <= styles_seen and avoid_frames > 0,
          "styles %s, missile-frames spent steering around a peak %d" % (sorted(styles_seen), avoid_frames))

    # H. an uncertainty margin makes a missile count as closer, and changes nothing else
    state = {'pos': [400.0, 300.0], 'vel': [0.0, 0.0], 'heading': 0.0}
    th = {'pos': [220.0, 300.0], 'vel': [6.0, 0.0], 'heading': 0.0, 'speed': 6.0, 'radius': 2}
    r0 = worldmodel_v3.rollout(state, [th], [math.pi], [0.3], HORIZON, backend=backend)
    r1 = worldmodel_v3.rollout(state, [dict(th, margin=20.0)], [math.pi], [0.3], HORIZON, backend=backend)
    check(tag + "H  a margin brings a predicted hit forward and nothing else",
          (r0.hit_frame[0] >= 0 and 0 <= r1.hit_frame[0] < r0.hit_frame[0]
           and abs((r0.min_dist[0] - 20.0) - r1.min_dist[0]) < 1e-9
           and np.array_equal(r0.threat_x, r1.threat_x)),
          "hit frame %d -> %d" % (r0.hit_frame[0], r1.hit_frame[0]))


def timing(level, n_missiles, backend, n_flares=0, reps=15):
    a = Arena(level, counter_attack=False, visuals=False)
    rng = random.Random(3)
    for i in range(n_missiles):
        mx = 60 + 680 * i / max(1, n_missiles - 1)
        if level.smart:
            m = SmartMissile(mx, 520, a.jet, a.jet.x, a.jet.y, lead=level.lead,
                             side=(-1.0, 1.0)[i % 2] if level.pincer else 0.0,
                             network=(1 if level.datalink else None))
        else:
            m = HomingMissile(mx, 520, a.jet, a.jet.x, a.jet.y)
        apply_style(m, sample_style(rng, level.number, STYLES[i % len(STYLES)]))
        a._add_missile(m)
    for _ in range(n_flares):
        add_flare(a, rng, a.jet.x + rng.uniform(-150, 150), a.jet.y + rng.uniform(-150, 150), 60)
    state = {'pos': [a.jet.x, a.jet.y], 'vel': [0.0, 0.0], 'heading': 0.0}
    threats = [oracle_threat(m, level.datalink, a.frame) for m in a.missiles]
    flares = [flare_state(f, a.frame) for f in a.flares]
    hs = np.linspace(-3.1, 3.1, 57)
    ts = np.full(57, 5.0)
    kw = dict(terrain=a.terrain, datalink=level.datalink, backend=backend, flares=flares)
    worldmodel_v3.rollout(state, threats, hs, ts, HORIZON, **kw)
    t0 = time.perf_counter()
    for _ in range(reps):
        worldmodel_v3.rollout(state, threats, hs, ts, HORIZON, **kw)
    return (time.perf_counter() - t0) / reps * 1e3


for b in BACKENDS:
    t0 = time.perf_counter()
    suite(b)
    print("INFO  [%s] suite took %.1fs (includes compiling on first use)" % (b, time.perf_counter() - t0))

if worldmodel_v3.HAVE_KERNEL:
    print("INFO  [numba] rollout cost, 57 actions x 72 frames x 6 styled missiles:  L1 %.1f ms   L2 %.1f ms"
          "   L3 %.1f ms   L3 + 24 flares %.1f ms"
          % (timing(LEVEL_1, 6, "numba"), timing(LEVEL_2, 6, "numba"),
             timing(LEVEL_3, 6, "numba"), timing(LEVEL_3, 6, "numba", 24)))

print()
print("RESULT:", "V3 WORLD MODEL MATCHES THE ARENA (%s)" % ", ".join(BACKENDS)
      if not failures else "FAILED: %s" % failures)
sys.exit(1 if failures else 0)
