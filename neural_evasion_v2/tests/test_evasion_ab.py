"""
Closed-loop A/B: old frozen-candidate search vs the world-model planner.

Both policies fly the same jet through the same physics with the same maneuver
FSM on identical scenarios. The only difference is where the heading (and, for
the new policy, the throttle) comes from:

  OLD  main3.py's worker  - scores 12 headings with the jet frozen in place and
                            every missile frozen too; throttle fixed at 5.0
  NEW  planner.plan       - rolls jet and missiles forward for 48 frames per
                            (heading, throttle) candidate

Reports survival rate, frames survived and closest approach, split out by
scenario type so the ballistic case is visible on its own.

Run: python neural_evasion/tests/test_evasion_ab.py [n_scenarios]
"""
import math
import os
import random
import sys
import threading

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import main3                                                    # noqa: E402
from neural_evasion.config import (BASE_SPEED, GUIDANCE_HOMING, HEIGHT,   # noqa: E402
                                   HUD_HEIGHT, WIDTH)
from neural_evasion.model import get_brain                      # noqa: E402
from neural_evasion.sim.decision import AIDecision              # noqa: E402
from neural_evasion.sim.entities import (BallisticMissile,      # noqa: E402
                                         HomingMissile, Jet)
from neural_evasion.sim.maneuvers import ManeuverController     # noqa: E402
from neural_evasion.sim.planner import plan                     # noqa: E402
from neural_evasion.sim.tactics import compute_wall_state       # noqa: E402

FRAMES     = 240
REPLAN     = 6
SCENARIOS  = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def build_scenario(seed, kind):
    """Deterministic scenario: a jet in the middle and a set of incoming missiles."""
    rng = random.Random(seed)
    jet = Jet(rng.uniform(300, 500), rng.uniform(200, 350))
    jet.heading = rng.uniform(0, 2 * math.pi)
    missiles = []

    def edge_spawn():
        side = rng.randint(0, 3)
        if side == 0:   return rng.uniform(0, 60), rng.uniform(0, HEIGHT - HUD_HEIGHT)
        if side == 1:   return rng.uniform(WIDTH - 60, WIDTH), rng.uniform(0, HEIGHT - HUD_HEIGHT)
        if side == 2:   return rng.uniform(0, WIDTH), rng.uniform(0, 60)
        return rng.uniform(0, WIDTH), rng.uniform(HEIGHT - HUD_HEIGHT - 60, HEIGHT - HUD_HEIGHT)

    n_h, n_b = kind
    for _ in range(n_h):
        mx, my = edge_spawn()
        m = HomingMissile(mx, my, jet, jet.x, jet.y)
        m.speed = m.max_speed = rng.uniform(4.0, 6.5)
        m.turn_rate = rng.uniform(0.06, 0.15)
        missiles.append(m)
    for _ in range(n_b):
        mx, my = edge_spawn()
        # Aimed at the jet: a straight shot the jet must not fly into.
        missiles.append(BallisticMissile(mx, my, jet.x, jet.y))
    return jet, missiles


def _old_decision(jet, missiles, flares):
    """Ask main3.py's worker for a heading (its own queues, run synchronously)."""
    main3.state_queue.put({
        'jet': {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy], 'heading': jet.heading},
        'missiles': [{'pos': [m.x, m.y], 'vel': [m.vx, m.vy],
                      'mtype': getattr(m, 'mtype', 0)} for m in missiles],
        'flares': [{'pos': [f.x, f.y], 'vel': [f.vx, f.vy]} for f in flares],
    })
    (heading, drop_flare, attn, danger, edges, cands, manu) = main3.decision_queue.get(timeout=60)
    return AIDecision(heading=heading, throttle=BASE_SPEED, drop_flare=drop_flare,
                      attention=attn, danger=danger, candidate_scores=cands,
                      maneuver=manu, fresh=True)


def _new_decision(brain, jet, missiles, flares, hidden):
    state_jet = {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy],
                 'heading': jet.heading, 'speed': jet.speed}
    ms = [{'pos': [m.x, m.y], 'vel': [m.vx, m.vy], 'heading': m.heading,
           'speed': m.speed, 'turn_rate': getattr(m, 'turn_rate', 0.0),
           'lateral_bias': getattr(m, 'lateral_bias', 0.0),
           'spawn_delay': getattr(m, 'spawn_delay', 0), 'radius': m.radius,
           'mtype': getattr(m, 'mtype', GUIDANCE_HOMING)} for m in missiles]
    fs = [{'pos': [f.x, f.y], 'vel': [f.vx, f.vy], 'life': 120} for f in flares]
    p = plan(brain, state_jet, ms, fs, hidden)
    return AIDecision(heading=p.heading, throttle=p.throttle,
                      drop_flare=bool(ms) and (p.cornered or p.danger > 0.55),
                      attention=p.attention, danger=p.danger,
                      candidate_scores=p.candidate_scores, maneuver=p.maneuver,
                      fresh=True), p.hidden


def run_episode(policy, seed, kind, brain):
    """One episode. Returns (survived, frames, closest_approach, killed_by)."""
    jet, missiles = build_scenario(seed, kind)
    flares = []
    controller = ManeuverController()
    decision = AIDecision()
    hidden = None
    stats = {'fired': 0, 'flares': 0, 'maneuvers': 0}
    closest = 1e9

    for frame in range(FRAMES):
        if frame % REPLAN == 0 and missiles:
            if policy == "old":
                decision = _old_decision(jet, missiles, flares)
            else:
                decision, hidden = _new_decision(brain, jet, missiles, flares, hidden)

        controller.tick_cooldowns()
        wall = compute_wall_state(jet)
        controller.select(jet, missiles, wall, decision.maneuver, stats, decision)
        controller.execute(jet, missiles, flares, wall, decision, stats)
        jet.update()

        if (jet.x < jet.radius or jet.x > WIDTH - jet.radius or
                jet.y < jet.radius or jet.y > HEIGHT - HUD_HEIGHT - jet.radius):
            return False, frame, closest, "wall"

        # Flares age by frame here (no real clock in a headless loop); the game
        # gives them 2000 ms == 120 frames.
        alive_f = []
        for f in flares:
            f.update()
            f.age = getattr(f, 'age', 0) + 1
            if f.age < 120:
                alive_f.append(f)
        flares = alive_f[:120]

        alive = []
        for m in missiles:
            m.update()
            d = math.hypot(jet.x - m.x, jet.y - m.y)
            closest = min(closest, d)
            if d < jet.radius + m.radius:
                return False, frame, closest, ("ballistic" if getattr(m, 'mtype', 0) == 1
                                               else "homing")
            # As in the game, a flare kills any missile it touches.
            hit_flare = False
            for f in flares:
                if math.hypot(f.x - m.x, f.y - m.y) < f.radius + m.radius:
                    f.age = 999
                    hit_flare = True
                    break
            if not hit_flare and not (m.x < 0 or m.x > WIDTH or m.y < 0 or m.y > HEIGHT):
                alive.append(m)
        missiles = alive
        if not missiles:
            return True, frame, closest, None

    return True, FRAMES, closest, None


KINDS = [
    ("homing only",     (3, 0)),
    ("ballistic only",  (0, 3)),
    ("mixed",           (2, 2)),
    ("swarm",           (5, 2)),
]

if __name__ == "__main__":
    brain = get_brain()
    threading.Thread(target=main3.gnn_inference_worker, daemon=True).start()

    print("%d scenarios per type, %d frames each\n" % (SCENARIOS, FRAMES))
    print("%-16s %-6s %10s %10s %14s %s"
          % ("scenario", "policy", "survived", "avg frames", "closest (px)", "deaths"))
    print("-" * 82)

    totals = {"old": [0, 0], "new": [0, 0]}
    for name, kind in KINDS:
        for policy in ("old", "new"):
            wins = frames = 0
            closest_all = []
            deaths = {}
            for i in range(SCENARIOS):
                ok, f, c, by = run_episode(policy, 1000 + i, kind, brain)
                wins += ok
                frames += f
                closest_all.append(c)
                if by:
                    deaths[by] = deaths.get(by, 0) + 1
            totals[policy][0] += wins
            totals[policy][1] += SCENARIOS
            print("%-16s %-6s %8.0f%% %10.1f %14.1f %s"
                  % (name if policy == "old" else "", policy,
                     100.0 * wins / SCENARIOS, frames / float(SCENARIOS),
                     sum(closest_all) / len(closest_all),
                     ", ".join("%s %d" % kv for kv in sorted(deaths.items())) or "-"))
        print()

    print("-" * 82)
    for policy in ("old", "new"):
        w, n = totals[policy]
        print("OVERALL %-4s survival %.0f%%  (%d/%d)" % (policy, 100.0 * w / n, w, n))
    main3.state_queue.put(None)
