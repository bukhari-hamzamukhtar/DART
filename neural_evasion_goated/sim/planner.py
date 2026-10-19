"""
Action selection: score a grid of (heading, throttle) actions with the world
model, then let the network re-rank the survivors.

The old scheme fed the network the jet at its *current* position with a
hypothetical velocity and froze every missile, so it could not tell the
difference between turning away from a threat and turning into its path — which
is how the jet used to fly straight into ballistic missiles.

Now every candidate is genuinely simulated:

  1. `worldmodel.rollout` advances the jet and all missiles for the horizon and
     reports closest approach, predicted impact frame and wall contact.
  2. Those give an analytic cost — a predicted collision outranks everything.
  3. The best ROLLOUT_TOP_K candidates are then evaluated by the GNN *on the
     rolled-out end state*, so the learned danger score is asked about the future
     the jet is actually steering into.

The analytic pass is what makes throttle meaningful: it simulates real physics,
so it knows that braking tightens a turn and that sprinting outruns a slow
missile — something the danger head cannot express, having been trained on a
constant-throttle jet.
"""

import math

import numpy as np

from ..config import (BASE_SPEED, HEIGHT, HUD_HEIGHT, MAX_THROTTLE,
                      MIN_THROTTLE, ROLLOUT_HORIZON, ROLLOUT_TOP_K,
                      THROTTLE_CHOICES, WIDTH)
from .evaluate import evaluate_scenes
from .graph_builder import build_candidate_sub_graphs
from .worldmodel import drift_flares, rollout

# Heading offsets (radians) relative to the jet's current heading.
HEADING_DELTAS = [0.0, 0.20, -0.20, 0.40, -0.40, 0.65, -0.65,
                  1.0, -1.0, 1.4, -1.4, math.pi]

# ── Cost weights ──────────────────────────────────────────────
# A predicted impact has to outrank everything else, including a confident GNN
# danger score (which saturates at 1.0).
HIT_COST      = 8.0     # a predicted impact, scaled up by how soon it lands
WALL_COST     = 8.0     # a predicted wall breach: equally lethal, equally priced

# Proximity weights are deliberately LOWER than the lethal costs above.
# Escaping a corner means flying *along* a wall, so punishing nearness as hard
# as punishing a breach made the jet prefer to fly out through the wall instead
# of hugging it -- a certain late breach cost ~9.3 against ~7.2 for a safe 5 px
# skim, and missile proximity then cast the deciding vote for death.
#
# Two retunes were tried here and both traded missile safety for wall safety:
# a flat lethal cost of 40 (0/8, six missile deaths where there had been none)
# and WALL_PROX_COST 3.0 (fewer wall deaths, four missile deaths, median life
# down from 620 to 384). Neither is a win, so the weights stay as they are and
# the corner case is handled by acting on it instead -- see `cornered` below.
HIT_PROX_COST  = 8.0    # scale for *surviving* proximity to a missile
WALL_PROX_COST = 8.0    # scale for *surviving* proximity to a wall

# Danger has to be CONTINUOUS across the collision boundary. Scoring a 3 px miss
# as "no impact, small penalty" and a 3 px touch as "impact, cost 8" makes the
# planner thread needles: it will happily pick a path that clears a missile by
# less than its own radius, and any staleness in the plan turns that into a hit.
# These curves reach the full kill cost as the gap closes to zero.
HIT_RADIUS    = 13.0    # jet radius (11) + missile radius (2)
THREAT_MARGIN = 95.0    # gap above HIT_RADIUS at which a missile stops mattering
WALL_MARGIN   = 150.0   # room at which a wall stops mattering. Swept: 110 -> 1/6
                        # survival, 150 -> 3/6, 190 -> 1/6. Too little and the jet
                        # gets herded into walls; too much and it surrenders the
                        # outer ring of the arena and gets cornered in open ground.
PROX_FALLOFF  = 3.0     # exponent: how sharply danger falls off with clearance
DISCOUNT      = 0.985   # per-frame discount when integrating danger over a path

CENTRE_COST   = 0.10    # mild pull away from the arena edges
GNN_WEIGHT    = 1.0     # weight on the learned danger score

# Prefer the throttle that keeps options open when nothing separates candidates.
THROTTLE_TIEBREAK = 0.02

CENTRE_X = WIDTH / 2.0
CENTRE_Y = (HEIGHT - HUD_HEIGHT) / 2.0


def action_grid(current_heading):
    """Every (target heading, throttle) pair the jet may pick this tick."""
    headings, throttles = [], []
    for d in HEADING_DELTAS:
        for mult in THROTTLE_CHOICES:
            headings.append(current_heading + d)
            throttles.append(
                max(MIN_THROTTLE, min(MAX_THROTTLE, BASE_SPEED * mult)))
    return np.array(headings), np.array(throttles)


def analytic_cost(result, throttles, horizon=ROLLOUT_HORIZON):
    """Cost of each candidate from the rollout alone — no network involved."""
    horizon = float(horizon)

    # ── missiles ──────────────────────────────────────────────
    # A predicted impact costs the most, and the sooner it lands the worse.
    hit = result.hit_frame >= 0
    hit_urgency = np.where(hit, 1.0 - result.hit_frame / horizon, 0.0)

    # Integrate danger along the whole path rather than taking its worst
    # instant, so a path that escapes scores better than one that stays pinned
    # even when both start equally close.
    w = DISCOUNT ** result.times
    w = w / max(w.sum(), 1e-9)

    gap = np.maximum(result.dist_profile - HIT_RADIUS, 0.0)
    threat = np.clip(1.0 - gap / THREAT_MARGIN, 0.0, 1.0) ** PROX_FALLOFF
    near = (threat * w).sum(axis=1)
    cost = np.where(hit, HIT_COST * (1.0 + hit_urgency), HIT_PROX_COST * near)

    # ── walls ─────────────────────────────────────────────────
    # Same shape, same reasoning: hitting one is fatal, and being squeezed
    # against one is most of the way to fatal because it removes escape routes.
    wall = result.wall_frame >= 0
    wall_urgency = np.where(wall, 1.0 - result.wall_frame / horizon, 0.0)

    room = np.maximum(result.wall_profile, 0.0)
    squeeze = np.clip(1.0 - room / WALL_MARGIN, 0.0, 1.0) ** PROX_FALLOFF
    pinned = (squeeze * w).sum(axis=1)
    cost = cost + np.where(wall, WALL_COST * (1.0 + wall_urgency),
                           WALL_PROX_COST * pinned)

    # Stay out of the corners: they remove escape routes.
    off_centre = np.hypot(result.end_x - CENTRE_X, result.end_y - CENTRE_Y) / 800.0
    cost = cost + CENTRE_COST * off_centre

    cost = cost + THROTTLE_TIEBREAK * np.abs(throttles - BASE_SPEED) / BASE_SPEED
    return cost


def _end_state_scene(result, idx, missiles, flares_future):
    """Rebuild the rolled-out world at candidate `idx` as graph-builder input."""
    js = {'pos': [float(result.end_x[idx]), float(result.end_y[idx])],
          'vel': [float(result.end_vx[idx]), float(result.end_vy[idx])],
          'type': 0}
    ms = []
    for k, m in enumerate(missiles):
        if not result.end_alive[idx, k]:
            continue
        ms.append({'pos': [float(result.end_mx[idx, k]), float(result.end_my[idx, k])],
                   'vel': [float(result.end_mvx[idx, k]), float(result.end_mvy[idx, k])],
                   'type': 1,                      # node role: this node is a missile
                   'mtype': m.get('mtype', 0)})    # guidance: homing or ballistic
    return js, ms, flares_future


def _current_scene(jet, missiles, flares):
    """The scene as it is right now — what the GRU memory is updated from."""
    js = {'pos': jet['pos'], 'vel': jet['vel'], 'type': 0}
    ms = [{'pos': m['pos'], 'vel': m['vel'], 'type': 1, 'mtype': m.get('mtype', 0)}
          for m in missiles]
    fs = [{'pos': f['pos'], 'vel': f['vel'], 'type': 2} for f in flares]
    return js, ms, fs


class Plan(object):
    """What the planner decided this tick, plus the state the HUD wants."""

    __slots__ = ("heading", "throttle", "candidate_scores", "scene",
                 "attention", "danger", "hidden", "maneuver", "cornered")

    def __init__(self, heading, throttle, candidate_scores, scene,
                 attention, danger, hidden, maneuver, cornered=False):
        self.heading = heading
        self.throttle = throttle
        self.candidate_scores = candidate_scores
        self.scene = scene              # winning rolled-out scene, for the [G] overlay
        self.attention = attention
        self.danger = danger            # danger of the CURRENT state, for the HUD
        self.hidden = hidden            # advanced GRU memory
        self.maneuver = maneuver
        # True when even the best action still ends in a predicted impact --
        # flying alone will not save the jet, so spend countermeasures.
        self.cornered = cornered


def plan(model, jet, missiles, flares, gru_hidden, horizon=ROLLOUT_HORIZON,
         top_k=ROLLOUT_TOP_K, lead=0):
    """
    Choose a heading and a throttle, and advance the GRU memory.

    One batched encoder pass covers the current scene plus every shortlisted
    rollout future, so the network is consulted about the world it is steering
    into rather than only the world it is standing in.
    """
    cur_js, cur_ms, cur_fs = _current_scene(jet, missiles, flares)
    cur_sgs = build_candidate_sub_graphs(cur_js, cur_ms, cur_fs) if cur_ms else []

    # The rollout runs even with no missiles: walls are lethal too, and the
    # planner is the only thing steering now that the FSM is a fallback.
    headings, throttles = action_grid(jet['heading'])
    # Plan for where the world will be when the jet can actually act, not for
    # the snapshot that is already several frames old by now.
    result = rollout(jet, missiles, headings, throttles, horizon,
                     lead=lead, hold_heading=jet['heading'],
                     hold_throttle=jet.get('speed', BASE_SPEED))
    cost = analytic_cost(result, throttles, horizon)

    flares_future = drift_flares(flares, lead + horizon)
    order = [int(i) for i in np.argsort(cost)[:max(1, top_k)]]
    analytic_best = order[0]

    # top_k = 0 means the network does not re-rank: the action is chosen from
    # the rollout alone. The current scene is still evaluated, because the danger
    # score, the maneuver head and the GRU memory all depend on it. Re-ranking
    # changes the chosen action in ~16% of frames and costs about half the plan
    # budget, and plan freshness turned out to matter far more than that 16%.
    scenes, futures = [cur_sgs], []
    for idx in (order if top_k > 0 else []):
        js, ms, fs = _end_state_scene(result, idx, missiles, flares_future)
        sgs = build_candidate_sub_graphs(js, ms, fs) if ms else []
        scenes.append(sgs)
        futures.append((sgs, js, ms, fs))

    results = evaluate_scenes(model, scenes, gru_hidden)
    current, shortlist = results[0], results[1:]

    # No escape by flying? Then it is time for flares. This is what the old
    # `danger > 0.55` heuristic was standing in for.
    # Strictly: no heading at all avoids an impact. This gates both the flare
    # dump and the scripted maneuvers, and it has to stay strict -- widening it
    # to "most options are lost, walls included" let the FSM fire often, and
    # every scripted maneuver that seizes the controls costs more than it saves
    # (brutal-pressure survival 1/8 -> 0/8, median 620 -> 258 frames).
    cornered = bool(result.hit_frame.min() >= 0)

    best_idx, best_total, best_scene, best_attn = analytic_best, None, None, []
    for idx, res, fut in zip(order, shortlist, futures):
        total = float(cost[idx]) + GNN_WEIGHT * res.danger
        cost[idx] = total          # so the HUD shows the blended score
        if best_total is None or total < best_total:
            best_total, best_idx = total, idx
            best_attn = res.attention
            best_scene = fut if fut[0] else None

    # Collapse the action grid to one risk value per heading for the polar HUD.
    n_throttle = len(THROTTLE_CHOICES)
    candidate_scores = [
        (float(headings[h * n_throttle]), float(cost[h * n_throttle:(h + 1) * n_throttle].min()))
        for h in range(len(HEADING_DELTAS))
    ]

    return Plan(float(headings[best_idx]), float(throttles[best_idx]),
                candidate_scores, best_scene,
                best_attn or current.attention,
                current.danger,
                current.hidden if cur_sgs else gru_hidden,
                current.maneuver if cur_sgs else "NORMAL",
                cornered)


# ══════════════════════════════════════════════════════════════
#  Per-frame correction
# ══════════════════════════════════════════════════════════════
# The strategic plan above costs tens of milliseconds, so it arrives 6-15 frames
# old. Measured latency tolerance of this policy is brutal: replanning every
# frame with 1 frame of staleness survives 7/8 test scenarios, every 3 frames
# with 3 frames of staleness survives 0/8. A missile covers 5-7 px per frame and
# the kill radius is 13 px, so a plan a few frames old is simply wrong.
#
# This correction is cheap enough to run on the game thread every single frame,
# on live positions. It only searches near the strategic heading, so it follows
# the plan when the plan is safe and overrides it when the plan would collide.

# Measured requirements, not guesses: a 72-frame horizon at step 1 survives 8/8
# test scenarios; horizon 48 survives 0/8 and step 2 misjudges closest approach
# by hundreds of pixels. Both have to stay where they are.
CORRECT_HORIZON = 72          # 1.2 s of lookahead
CORRECT_STEP    = 1           # exact; coarsening breaks collision judgement
DRIFT_COST      = 0.0         # do NOT pull toward the strategic heading.
                              # It arrives ~12 frames old, and biasing a fresh
                              # every-frame decision toward stale advice costs
                              # more than the continuity is worth.


def correct(jet, missiles, target_heading, target_throttle,
            horizon=None, step=None):
    """
    Choose what to fly *this frame*, from live positions.

    This searches the same action grid as `plan`, but coarsely and cheaply
    enough to run on the game thread every frame. The strategic heading from
    `plan` — which is the one the network had a say in — is preferred via
    DRIFT_COST when nothing separates the options on safety, so the plan still
    steers while this keeps the jet out of things that moved since.

    Returns (heading, throttle).
    """
    if not missiles:
        return target_heading, target_throttle
    if horizon is None: horizon = CORRECT_HORIZON
    if step is None:    step = CORRECT_STEP

    headings, throttles = action_grid(jet['heading'])
    res = rollout(jet, missiles, headings, throttles, horizon, step=step)
    cost = analytic_cost(res, throttles, horizon)

    # prefer staying near the strategic heading, all else equal
    drift = np.abs((headings - target_heading + math.pi) % (2 * math.pi) - math.pi)
    cost = cost + DRIFT_COST * (drift / math.pi)

    best = int(np.argmin(cost))
    return float(headings[best]), float(throttles[best])
