"""
Game rules that are not physics, and the new missile behaviour:

  A  the flare magazine: a life starts empty, each cluster launch earns one
     volley, a volley costs one charge whatever it holds, no charge means the
     flares never leave the jet
  B  launch keys: level 3 has H/C on the west battery and K/M on the east; each
     battery launches its own clusters, and each cluster earns the jet a volley
  C  flight styles: all four are drawn, a cluster flies mixed styles and spreads
     out wider than an all-pursuit cluster, erratic darts are deterministic
  D  a seeker flying at a peak steers around it; the same seeker blind to
     terrain flies into it
"""
import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pygame  # noqa: E402

from neural_evasion.arena import Arena                                   # noqa: E402
from neural_evasion.game import launch_keymap                            # noqa: E402
from neural_evasion.levels import LEVEL_1, LEVEL_2, LEVEL_3              # noqa: E402
from neural_evasion.sim.doctrine import (PURSUIT_STYLE, STYLES,          # noqa: E402
                                         jink_value, sample_style)
from neural_evasion.sim.entities import (FlareParticle, SmartMissile,    # noqa: E402
                                         apply_style)

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        failures.append(name)


class Ejector(object):
    """Ejects three flares every frame, without looking at the magazine."""

    def act(self, arena):
        for k in range(3):
            arena.flares.append(FlareParticle(arena.jet.x, arena.jet.y, float(k), 4.0))
            arena.stats['flares'] += 1


class Payer(object):
    """Pays for a volley itself, then ejects it in the same frame."""

    def __init__(self):
        self.paid = None

    def act(self, arena):
        self.paid = arena.spend_flare()
        if self.paid:
            for k in range(4):
                arena.flares.append(FlareParticle(arena.jet.x, arena.jet.y, float(k), 4.0))


class Hover(object):
    """Keep the jet nearly still."""

    def act(self, arena):
        arena.jet.set_throttle(0.3)
        arena.jet.vx *= 0.5
        arena.jet.vy *= 0.5


# ── A. the flare magazine ─────────────────────────────────────
a = Arena(LEVEL_1, counter_attack=False, visuals=False, seed=1)
check("A  a life starts with an empty magazine", a.flare_charges == 0 and not a.can_flare())
a.begin_frame()
a.advance(Ejector())
check("A' with no charge, ejected flares never leave the jet",
      len(a.flares) == 0 and a.stats['flares_denied'] == 3 and a.stats['flares'] == 0,
      "flares in the air %d, denied %d" % (len(a.flares), a.stats['flares_denied']))
ok = a.launch_cluster()
check("A'' a cluster launch earns one volley", ok and a.flare_charges == 1 and a.can_flare())
a.begin_frame()
a.advance(Ejector())
check("A''' a volley costs one charge, however many flares it holds",
      len(a.flares) == 3 and a.flare_charges == 0 and a.stats['volleys'] == 1)
a.begin_frame()
a.advance(Ejector())
check("A'''' the next frame's flares are refused again",
      len(a.flares) == 3 and a.stats['flares_denied'] == 6)
a.cluster_cooldowns[0] = 0
a.launch_cluster(0)
payer = Payer()
a.begin_frame()
a.advance(payer)
check("A''''' a pilot that pays for its own volley is charged once, not twice",
      payer.paid and a.flare_charges == 0 and len(a.flares) == 7 and a.stats['volleys'] == 2,
      "charges %d, flares %d, volleys %d" % (a.flare_charges, len(a.flares), a.stats['volleys']))

# ── B. launch keys and per-battery clusters ───────────────────
km3 = launch_keymap(LEVEL_3)
check("B  level 3: H/C fire the west battery, K/M the east, S the pincer",
      km3.get(pygame.K_h) == ("fire", 0) and km3.get(pygame.K_c) == ("cluster", 0)
      and km3.get(pygame.K_k) == ("fire", 1) and km3.get(pygame.K_m) == ("cluster", 1)
      and km3.get(pygame.K_s, (None,))[0] == "pincer")
km1 = launch_keymap(LEVEL_1)
check("B' levels 1-2: H/C the homing battery, B the ballistic one, no pincer key",
      km1.get(pygame.K_h) == ("fire", 0) and km1.get(pygame.K_c) == ("cluster", 0)
      and km1.get(pygame.K_b) == ("fire", 1) and pygame.K_s not in km1
      and launch_keymap(LEVEL_2) == km1)
a3 = Arena(LEVEL_3, counter_attack=False, visuals=False, seed=2)
both = a3.launch_cluster(0) and a3.launch_cluster(1)
check("B'' each level-3 battery launches its own cluster, and each earns a volley",
      both and len(a3.missiles) == 10 and a3.flare_charges == 2
      and not a3.cluster_ready(0) and not a3.cluster_ready(1)
      and set(m.x for m in a3.missiles) == {100.0, 700.0},
      "missiles %d, charges %d" % (len(a3.missiles), a3.flare_charges))

# ── C. flight styles ──────────────────────────────────────────
rng = random.Random(5)
counts = dict((s, 0) for s in STYLES)
for _ in range(2000):
    counts[sample_style(rng, 2)['style']] += 1
check("C  all four flight styles are drawn", all(v > 200 for v in counts.values()), str(counts))


def cluster_spread(styled):
    arena = Arena(LEVEL_1, counter_attack=False, visuals=False, jet_invulnerable=True, seed=7)
    arena.jet.x, arena.jet.y = 400.0, 120.0
    arena.launch_cluster(0)
    styles = sorted(m.style for m in arena.missiles)
    if not styled:
        for m in arena.missiles:
            apply_style(m, PURSUIT_STYLE)
    spread = []
    for _ in range(110):
        arena.begin_frame()
        arena.advance(Hover())
        ms = [m for m in arena.missiles if m.spawn_delay == 0]
        if len(ms) >= 2:
            pairs = [math.hypot(p.x - q.x, p.y - q.y) for i, p in enumerate(ms) for q in ms[i + 1:]]
            spread.append(sum(pairs) / len(pairs))
    return styles, (sum(spread) / len(spread) if spread else 0.0)


styles, spread_styled = cluster_spread(True)
_, spread_pursuit = cluster_spread(False)
check("C' a cluster flies mixed styles", len(set(styles)) >= 3, str(styles))
check("C'' ...and spreads out wider than the same cluster flying pure pursuit",
      spread_styled > 1.2 * spread_pursuit,
      "mean distance between members %.0f px vs %.0f px" % (spread_styled, spread_pursuit))
vals = [jink_value(12345, n) for n in range(2000)]
check("C''' erratic darts are deterministic and spread over [-1, 1]",
      vals == [jink_value(12345, n) for n in range(2000)] and min(vals) < -0.9
      and max(vals) > 0.9 and abs(sum(vals) / len(vals)) < 0.05)


# ── D. terrain avoidance ──────────────────────────────────────
class BlindSeeker(SmartMissile):
    """The same seeker, ignoring terrain when it steers."""

    def update(self):
        self.obstacles = ()
        SmartMissile.update(self)


def seeker_vs_peak(cls):
    arena = Arena(LEVEL_2, counter_attack=False, visuals=False, jet_invulnerable=True, seed=3)
    ox, oy, r = LEVEL_2.obstacles[2]          # the peak at (400, 200)
    arena.jet.x, arena.jet.y = float(ox), oy - r - 110.0
    m = cls(float(ox), oy + r + 100.0, arena.jet, arena.jet.x, arena.jet.y,
            speed=5.0, turn_rate=0.11, launch_heading=-math.pi / 2)
    arena._add_missile(m)
    avoided = 0
    for f in range(120):
        absorbed = arena.stats['absorbed']
        arena.begin_frame()
        arena.advance(Hover())
        avoided += 1 if getattr(m, 'avoiding', False) else 0
        if m not in arena.missiles:
            return ("reached the jet" if arena.stats['absorbed'] > absorbed else "destroyed"), f, avoided
    return "still flying", 120, avoided


seen, f_seen, avoided = seeker_vs_peak(SmartMissile)
blind, f_blind, _ = seeker_vs_peak(BlindSeeker)
check("D  a seeker flying at a peak steers around it", seen != "destroyed" and avoided > 0,
      "%s after %d frames, %d frames spent steering around" % (seen, f_seen, avoided))
check("D' the same seeker blind to terrain flies into the peak", blind == "destroyed" and f_blind < 40,
      "%s after %d frames" % (blind, f_blind))

print()
print("RESULT:", "GAME RULES HOLD" if not failures else "FAILED: %s" % failures)
sys.exit(1 if failures else 0)
