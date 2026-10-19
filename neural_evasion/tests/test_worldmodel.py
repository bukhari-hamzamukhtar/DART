"""
The world model must predict the game, not an approximation of it.

Rolls the vectorised simulator forward and compares against stepping the real
entity classes frame by frame. If these drift, the planner is reasoning about a
world the player does not live in.
"""
import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from neural_evasion.config import GUIDANCE_BALLISTIC, GUIDANCE_HOMING  # noqa: E402
from neural_evasion.sim.entities import (BallisticMissile,             # noqa: E402
                                         HomingMissile, Jet)
from neural_evasion.sim.worldmodel import STEER_GAIN, rollout          # noqa: E402
from neural_evasion.utils import normalize_angle                       # noqa: E402

HORIZON = 40
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        failures.append(name)


def ground_truth(jet, missiles, target_heading, throttle, horizon):
    """Step the real entity classes exactly as the game loop does."""
    for _ in range(horizon):
        jet.heading += normalize_angle(target_heading - jet.heading) * STEER_GAIN
        jet.set_throttle(throttle)
        jet.update()
        for m in missiles:
            m.update()
    return jet, missiles


worst_jet = 0.0
worst_missile = 0.0
rng = random.Random(5)

for trial in range(20):
    jx, jy = rng.uniform(150, 650), rng.uniform(150, 400)
    heading = rng.uniform(-3, 3)
    target = heading + rng.uniform(-1.5, 1.5)
    throttle = rng.choice([1.75, 3.5, 5.0, 7.5])

    jet = Jet(jx, jy)
    jet.heading = heading
    jet.vx, jet.vy = rng.uniform(-5, 5), rng.uniform(-5, 5)

    missiles = []
    for _ in range(rng.randint(1, 4)):
        mx, my = rng.uniform(50, 750), rng.uniform(50, 500)
        if rng.random() < 0.5:
            m = HomingMissile(mx, my, jet, jx, jy)
            m.speed = m.max_speed = rng.uniform(4.0, 6.5)
            m.turn_rate = rng.uniform(0.05, 0.15)
            m.lateral_bias = rng.choice([0.0, 0.007, -0.014])
        else:
            m = BallisticMissile(mx, my, jx, jy)
        missiles.append(m)

    packed = [{'pos': [m.x, m.y], 'vel': [m.vx, m.vy], 'heading': m.heading,
               'speed': m.speed, 'turn_rate': getattr(m, 'turn_rate', 0.0),
               'lateral_bias': getattr(m, 'lateral_bias', 0.0),
               'spawn_delay': getattr(m, 'spawn_delay', 0), 'radius': m.radius,
               'mtype': m.mtype} for m in missiles]
    state = {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy], 'heading': jet.heading}

    res = rollout(state, packed, [target], [throttle], HORIZON)
    ground_truth(jet, missiles, target, throttle, HORIZON)

    worst_jet = max(worst_jet, math.hypot(res.end_x[0] - jet.x, res.end_y[0] - jet.y))
    for k, m in enumerate(missiles):
        if 0 <= m.x <= 800 and 0 <= m.y <= 600:
            worst_missile = max(
                worst_missile,
                math.hypot(res.end_mx[0, k] - m.x, res.end_my[0, k] - m.y))

check("A  jet path matches the real Jet over %d frames" % HORIZON,
      worst_jet < 1e-6, "max divergence %.2e px" % worst_jet)
check("B  missile paths match the real missiles", worst_missile < 1e-6,
      "max divergence %.2e px" % worst_missile)

# ── C. a head-on ballistic missile must be predicted as a hit ──
jet = Jet(400, 275)
jet.heading = 0.0
jet.vx, jet.vy = 5.0, 0.0
inbound = [{'pos': [640.0, 275.0], 'vel': [-7.0, 0.0], 'heading': math.pi,
            'speed': 7.0, 'turn_rate': 0.0, 'lateral_bias': 0.0,
            'spawn_delay': 0, 'radius': 2, 'mtype': GUIDANCE_BALLISTIC}]
state = {'pos': [400.0, 275.0], 'vel': [5.0, 0.0], 'heading': 0.0}
res = rollout(state, inbound, [0.0, math.pi / 2], [5.0, 5.0], HORIZON)
check("C  head-on collision predicted for the closing heading",
      res.hit_frame[0] >= 0, "impact at frame %d, closest %.1f px"
      % (res.hit_frame[0], res.min_dist[0]))
check("D  turning away is predicted to miss",
      res.hit_frame[1] < 0 and res.min_dist[1] > res.min_dist[0],
      "closest %.1f px vs %.1f px head-on" % (res.min_dist[1], res.min_dist[0]))

print()
print("RESULT:", "WORLD MODEL MATCHES THE GAME" if not failures else "FAILED: %s" % failures)
sys.exit(1 if failures else 0)
