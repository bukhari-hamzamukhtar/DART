"""
v3 action selection.

The same receding-horizon search that made v2 hard to kill -- every frame, every
candidate action rolled 72 frames forward, pick the cheapest future -- but run on
the v3 world model and fed by the temporal graph brain:

  * the rollout knows terrain, seekers and datalinks (`sim/worldmodel_v3`), so
    "duck behind that peak" and "that missile's partner can still see me" are
    things the planner can actually foresee;
  * each missile's guidance parameters come from the GNN's estimates on its radar
    track, not from the missile object;
  * terrain is lethal but priced with its own, much tighter proximity margin.
    Skimming a peak is how a lock gets broken; pricing it like a wall would
    forbid the one manoeuvre terrain exists for.

Maneuvers stay soft, as in v2: each is a candidate action judged by the same
rollout, every frame, and can be entered, abandoned and resumed at will.
"""

import numpy as np

from ..config import HEIGHT, HUD_HEIGHT, THROTTLE_CHOICES, WIDTH
from .planner import (CENTRE_COST, CENTRE_X, CENTRE_Y, DISCOUNT, HEADING_DELTAS,
                      HIT_COST, HIT_PROX_COST, HIT_RADIUS, PROX_FALLOFF,
                      THREAT_MARGIN, THROTTLE_TIEBREAK, WALL_COST, WALL_MARGIN,
                      WALL_PROX_COST, action_grid, maneuver_actions)
from .worldmodel import JET_DRAG, JET_RADIUS
from .worldmodel_v3 import rollout

HORIZON           = 72
TERRAIN_COST      = 8.0     # flying into a peak: as lethal as a wall or a missile
TERRAIN_PROX_COST = 3.0     # being close to one: much cheaper than being close to a wall
TERRAIN_MARGIN    = 40.0    # px of clearance at which a peak stops mattering
BASE_SPEED_REF    = 5.0
FLARE_COST        = 4.0     # what a volley has to buy. The magazine holds one volley per
                            # enemy cluster, so a volley is for averting something close to
                            # a hit (a predicted hit costs 8-16), not for shaving margin
COAST             = 1.0 / (1.0 - JET_DRAG)   # px a jet still drifts per px/frame of speed


def cost_v3(res, throttles, horizon=HORIZON):
    """Cost of each candidate future. Same shape as v2's analytic_cost, plus terrain."""
    horizon = float(horizon)
    w = DISCOUNT ** res.times
    w = w / max(w.sum(), 1e-9)

    hit = res.hit_frame >= 0
    urgency = np.where(hit, 1.0 - res.hit_frame / horizon, 0.0)
    gap = np.maximum(res.dist_profile - HIT_RADIUS, 0.0)
    threat = np.clip(1.0 - gap / THREAT_MARGIN, 0.0, 1.0) ** PROX_FALLOFF
    cost = np.where(hit, HIT_COST * (1.0 + urgency), HIT_PROX_COST * (threat * w).sum(axis=1))

    wall = res.wall_frame >= 0
    urgency = np.where(wall, 1.0 - res.wall_frame / horizon, 0.0)
    room = np.maximum(res.wall_profile, 0.0)
    squeeze = np.clip(1.0 - room / WALL_MARGIN, 0.0, 1.0) ** PROX_FALLOFF
    cost = cost + np.where(wall, WALL_COST * (1.0 + urgency), WALL_PROX_COST * (squeeze * w).sum(axis=1))

    # A future that ends fast and pointed at a wall is a crash the horizon only
    # hides: the jet cannot shed that momentum in time. Where the jet would
    # coast to from its final state has to be inside the arena, or the future
    # costs what a breach on its last frame would.
    cx = res.end_x + res.end_vx * COAST
    cy = res.end_y + res.end_vy * COAST
    coast_room = np.minimum(np.minimum(cx - JET_RADIUS, WIDTH - JET_RADIUS - cx),
                            np.minimum(cy - JET_RADIUS, HEIGHT - HUD_HEIGHT - JET_RADIUS - cy))
    cost = cost + np.where(~wall & (coast_room < 0.0), WALL_COST, 0.0)

    if res.terrain_profile is not None:
        crash = res.terrain_frame >= 0
        urgency = np.where(crash, 1.0 - res.terrain_frame / horizon, 0.0)
        room = np.maximum(res.terrain_profile, 0.0)
        squeeze = np.clip(1.0 - room / TERRAIN_MARGIN, 0.0, 1.0) ** PROX_FALLOFF
        cost = cost + np.where(crash, TERRAIN_COST * (1.0 + urgency),
                               TERRAIN_PROX_COST * (squeeze * w).sum(axis=1))

    off_centre = np.hypot(res.end_x - CENTRE_X, res.end_y - CENTRE_Y) / 800.0
    cost = cost + CENTRE_COST * off_centre
    return cost + THROTTLE_TIEBREAK * np.abs(throttles - BASE_SPEED_REF) / BASE_SPEED_REF


class PlanV3(object):
    __slots__ = ("heading", "throttle", "label", "cornered", "candidate_scores",
                 "danger", "min_gap", "best", "result", "labels", "lethal",
                 "flare", "flare_gain")


def plan_v3(jet, threats, level, preferred=None, keep_future=False, flares=(), volley=None,
            hold=None):
    """
    Choose what to fly this frame, and whether to flare.

    `jet` is {'pos', 'vel', 'heading'}; `threats` are world-model inputs (from
    the tracker, or `oracle_threat` for the white-box ceiling); `flares` are the
    flares already in the air (`worldmodel_v3.flare_state`).

    `hold` is last frame's choice, (heading, throttle, label). The action grid
    is laid out around the current heading, so without it the exact action that
    was safe a frame ago may not be on offer now -- and a safe plan would have to
    be abandoned for a neighbour that is not.

    `volley`, when the jet could fire one now, is that volley as flare states.
    Every candidate action is then imagined twice -- without the volley and with
    it -- and the volley is fired only if the best future with it is cheaper,
    after paying FLARE_COST. Countermeasures become one more action the world
    model scores, not a trigger bolted on beside it.
    """
    grid_h, grid_t = action_grid(jet['heading'])
    labels = ["NORMAL"] * len(grid_h)
    extras = maneuver_actions(jet, threats)
    if extras:
        eh, et, el = zip(*extras)
        headings = np.concatenate([grid_h, np.asarray(eh, dtype=np.float64)])
        throttles = np.concatenate([grid_t, np.asarray(et, dtype=np.float64)])
        labels = labels + list(el)
    else:
        headings, throttles = grid_h, grid_t
    if hold is not None:
        headings = np.append(headings, float(hold[0]))
        throttles = np.append(throttles, float(hold[1]))
        labels = labels + [hold[2]]

    kw = dict(terrain=level.obstacles, datalink=level.datalink, keep_guided=keep_future)
    bonus = None
    if preferred is not None:
        bonus = 0.4 * np.array([1.0 if lb == preferred else 0.0 for lb in labels])

    res = rollout(jet, threats, headings, throttles, HORIZON, flares=flares, **kw)
    cost = cost_v3(res, throttles)
    if bonus is not None:
        cost = cost - bonus

    fire, gain = False, 0.0
    # Every cost term is non-negative, so a future with the volley costs at
    # least FLARE_COST: when the best future without it is already cheaper
    # than that, the volley cannot win and imagining it is skipped. Exact.
    if volley and threats and (bonus is not None or cost.min() > FLARE_COST):
        res_v = rollout(jet, threats, headings, throttles, HORIZON,
                        flares=list(flares) + list(volley), **kw)
        cost_v = cost_v3(res_v, throttles) + FLARE_COST
        if bonus is not None:
            cost_v = cost_v - bonus
        gain = float(cost.min() - cost_v.min())
        if gain > 0.0:
            res, cost, fire = res_v, cost_v, True
    best = int(np.argmin(cost))

    n_t = len(THROTTLE_CHOICES)
    p = PlanV3()
    p.heading = float(headings[best])
    p.throttle = float(throttles[best])
    p.label = labels[best]
    p.labels = labels
    p.cornered = bool(threats) and bool((res.hit_frame >= 0).all())
    p.candidate_scores = [(float(grid_h[h * n_t]), float(cost[h * n_t:(h + 1) * n_t].min()))
                          for h in range(len(HEADING_DELTAS))]
    p.danger = float(np.clip(cost[best] / HIT_COST, 0.0, 1.0))
    p.min_gap = float(res.min_dist[best] - HIT_RADIUS)
    # Even the best imagined future ends badly -- by missile, wall or terrain.
    # v2's "cornered" only counted missile hits, so a jet being herded into a
    # wall by missiles it could still dodge never spent a flare.
    p.lethal = bool(res.hit_frame[best] >= 0 or res.wall_frame[best] >= 0
                    or res.terrain_frame[best] >= 0)
    p.flare = fire
    p.flare_gain = gain
    p.best = best
    p.result = res if keep_future else None
    return p
