"""
Threat library for the missiles.

This is doctrine, not intelligence about a particular missile. The jet is
allowed to know how this *class* of weapon works -- seekers have a range and a
field of view, terrain blocks them, salvo members share a datalink, a seeker
steers around peaks, a missile flies in one of a handful of styles -- because a
real aircrew is briefed on exactly that. What it is never handed is the state of
an individual missile in the air: its turn rate, how much it leads, which style
it flies and with what numbers, which datalink net it is on, or whether it holds
a lock right now. Those are what the temporal GNN has to infer from radar tracks.

`sim.entities` (what the missiles do) and `sim.worldmodel_v3` /
`sim.rollout_kernel` (what the jet imagines they do) all read these numbers and
mirror these functions, so the two cannot silently drift apart.
"""

import math

SEEKER_RANGE    = 340.0                   # px; beyond this the seeker sees nothing
SEEKER_HALF_FOV = math.radians(70.0)      # gimbal limit either side of the nose
SEEKER_COS_FOV  = math.cos(SEEKER_HALF_FOV)
DECOY_RANGE     = 0.7 * SEEKER_RANGE      # a flare is dimmer than a jet engine
LINK_RANGE      = 380.0                   # px; datalink between members of one salvo
MEMORY_REACHED  = 24.0                    # px; reaching the last known point ends the chase
MAX_TGO         = 60.0                    # frames; cap on the lead-pursuit horizon

# Nominal launch profiles for the game. The data generator randomises around
# these so the network cannot memorise a lookup table of exact values.
SMART_SPEED      = 5.0
SMART_TURN_RATE  = 0.11
L3_LEAD          = 0.55
PINCER_SPEED     = 5.2
PINCER_TURN_RATE = 0.10

# ── Flight styles ─────────────────────────────────────────────
# Every guided missile leaves the rail with a style, so a salvo does not fly one
# tail-chase line behind the jet:
#
#   PURSUIT  straight at the jet (leading it, where the level has lead)
#   FLANKER  steers well off to one side of the jet and closes from the flank
#   WEAVER   swings its aim left and right, like a fly
#   ERRATIC  darts to a new random side every few frames
#
# All of it is a deviation of the aim from the line of sight, as the tangent of
# an angle: the missile steers for a point beside the jet at that fraction of
# the range, so the arcs are wide at any distance. A flank deviation fades as
# the range closes -- as a pincer member's does, and a pincer member is a
# flanker -- so the missile still arrives; weaving and darting fade only at
# knife-fight range.
PURSUIT, FLANKER, WEAVER, ERRATIC = "PURSUIT", "FLANKER", "WEAVER", "ERRATIC"
STYLES = (PURSUIT, FLANKER, WEAVER, ERRATIC)
STYLE_MIX = {1: (0.35, 0.25, 0.20, 0.20),      # chance of each style, by level
             2: (0.25, 0.30, 0.20, 0.25),
             3: (0.30, 0.25, 0.25, 0.20)}
FLANK_ANGLE  = (0.45, 0.90)     # tan of a flanker's deviation, about 24-42 degrees
PINCER_FLANK = 0.40             # tan of a pincer member's, about 22 degrees
FLANK_NEAR   = 60.0             # px; a flank deviation is gone inside this range...
FLANK_SPAN   = 180.0            # ...and fades in over this much more
WEAVE_AMP    = (0.25, 0.50)     # tan of a weaver's swing
WEAVE_FREQ   = (0.07, 0.16)     # rad per frame of flight
JINK_AMP     = (0.30, 0.60)     # tan of an erratic missile's darts
JINK_PERIOD  = (10, 26)         # frames per dart
WOBBLE_NEAR  = 25.0             # px; weaving and darting are gone inside this range...
WOBBLE_SPAN  = 90.0             # ...and fade in over this much more
JINK_TABLE   = 12               # darts handed to the world model ahead; >= 72/10 + 2

PURSUIT_STYLE = {'style': PURSUIT, 'side': 0.0, 'flank': 0.0, 'weave_amp': 0.0,
                 'weave_freq': 0.0, 'weave_phase': 0.0, 'jink_amp': 0.0,
                 'jink_period': 1, 'jink_seed': 0}

# ── Terrain avoidance ─────────────────────────────────────────
# A seeker looks ahead along its nose, a turning circle or so, and turns hard
# away from the nearest peak it would clip -- unless what it is chasing is
# nearer than that peak.
AVOID_MARGIN   = 14.0      # px of clearance kept from a peak's surface
AVOID_LOOK_MIN = 50.0      # px always looked ahead...
AVOID_RADII    = 1.2       # ...plus this many turning radii (speed / turn rate)
AVOID_LOOK_MAX = 260.0
AVOID_FAR      = 1e18

_MASK64 = 0xFFFFFFFFFFFFFFFF


def guidance_aim(mx, my, speed, jx, jy, jvx, jvy, lead, side, flank=PINCER_FLANK, wobble=0.0):
    """
    Where a guided missile steers this frame.

    Lead pursuit aims ahead of the jet by `lead` of its velocity times the time
    to go. The aim point is then pushed across the line of sight by a fraction
    of the range: `side` * `flank` for a flanker or pincer member, faded out as
    the range closes so the flanks converge for the kill; plus `wobble` for a
    weaving or darting missile, faded out only at knife-fight range.
    """
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
        ax = ax - ry * k       # perpendicular to the line of sight: (-ry, rx), length d
        ay = ay + rx * k
    return ax, ay


def sample_style(rng, level_number, style=None):
    """A flight style and its numbers for one guided missile; random unless `style` is given."""
    if style is None:
        style = rng.choices(STYLES, weights=STYLE_MIX.get(level_number, STYLE_MIX[1]))[0]
    p = dict(PURSUIT_STYLE)
    p['style'] = style
    if style == FLANKER:
        p['side'] = rng.choice((-1.0, 1.0))
        p['flank'] = rng.uniform(*FLANK_ANGLE)
    elif style == WEAVER:
        p['weave_amp'] = rng.uniform(*WEAVE_AMP)
        p['weave_freq'] = rng.uniform(*WEAVE_FREQ)
        p['weave_phase'] = rng.uniform(0.0, 2.0 * math.pi)
    elif style == ERRATIC:
        p['jink_amp'] = rng.uniform(*JINK_AMP)
        p['jink_period'] = rng.randint(*JINK_PERIOD)
        p['jink_seed'] = rng.getrandbits(31)
    return p


def jink_value(seed, n):
    """Dart number `n` of an erratic missile, in [-1, 1). Deterministic (splitmix64 mixing)."""
    z = (seed * 0x9E3779B97F4A7C15 + (n + 1) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z ^= z >> 31
    return (z >> 11) * (2.0 / 9007199254740992.0) - 1.0


def wobble_at(weave_amp, weave_freq, weave_phase, jink_amp, jink, age):
    """A missile's aim swing (tangent of the angle) at `age` frames of flight."""
    return weave_amp * math.sin(weave_freq * age + weave_phase) + jink_amp * jink


def avoid_turn(mx, my, ch, sh, speed, turn_rate, obstacles, aim_d=AVOID_FAR):
    """
    -1, 0 or +1: which way a seeker at (mx, my), nose along (ch, sh), must turn
    to clear the nearest peak ahead of it. `aim_d` is how far away the thing it
    is chasing is; a peak beyond that is not in the way.
    """
    if turn_rate <= 0.0:
        return 0.0
    look = speed / turn_rate * AVOID_RADII + AVOID_LOOK_MIN
    if look > AVOID_LOOK_MAX:
        look = AVOID_LOOK_MAX
    best = AVOID_FAR
    turn = 0.0
    for ox, oy, r in obstacles:
        ex = ox - mx
        ey = oy - my
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
