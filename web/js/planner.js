// sim/planner_v3.py (+ the cost weights and action grid of sim/planner.py).
//
// Every frame, every candidate action is rolled 72 frames through the world
// model and the cheapest imagined future is flown. When a volley is in the
// magazine, every action is imagined twice -- without it and with it -- and the
// volley is fired only if the best future with it is cheaper.
import {
  BASE_SPEED, MIN_THROTTLE, MAX_THROTTLE, THROTTLE_CHOICES, WIDTH, FIELD_H,
  JET_DRAG, JET_RADIUS,
} from "./config.js";
import { Rollout } from "./worldmodel.js";

export const HEADING_DELTAS = [0.0, 0.20, -0.20, 0.40, -0.40, 0.65, -0.65,
                               1.0, -1.0, 1.4, -1.4, Math.PI];

export const HIT_COST = 8.0, WALL_COST = 8.0;
export const HIT_PROX_COST = 8.0, WALL_PROX_COST = 8.0;
export const HIT_RADIUS = 13.0;
export const THREAT_MARGIN = 95.0, WALL_MARGIN = 150.0;
export const PROX_FALLOFF = 3.0, DISCOUNT = 0.985;
export const CENTRE_COST = 0.10, THROTTLE_TIEBREAK = 0.02;
export const CENTRE_X = WIDTH / 2.0, CENTRE_Y = FIELD_H / 2.0;

export const HORIZON = 72;
export const TERRAIN_COST = 8.0, TERRAIN_PROX_COST = 3.0, TERRAIN_MARGIN = 40.0;
export const BASE_SPEED_REF = 5.0;
// What a volley has to buy. The magazine holds one volley per enemy cluster, so
// a volley is for averting something close to a hit (a predicted hit costs 8-16).
export const FLARE_COST = 4.0;
export const COAST = 1.0 / (1.0 - JET_DRAG);

// Every (target heading, throttle) pair the jet may pick this tick. `choices`
// is the throttle set: the full four of the desktop build, or a trimmed three
// on slower devices (see the quality setting).
export function actionGrid(currentHeading, choices) {
  const throttleSet = choices || THROTTLE_CHOICES;
  const headings = [], throttles = [];
  for (const d of HEADING_DELTAS) {
    for (const mult of throttleSet) {
      headings.push(currentHeading + d);
      throttles.push(Math.max(MIN_THROTTLE, Math.min(MAX_THROTTLE, BASE_SPEED * mult)));
    }
  }
  return [headings, throttles];
}

// Both perpendiculars to a missile's approach, nearer one first.
function perpendicularOptions(jet, missile) {
  const mvx = missile.vel[0], mvy = missile.vel[1];
  const perp1 = Math.atan2(-mvx, mvy);
  const perp2 = Math.atan2(mvx, -mvy);
  const jx = jet.pos[0], jy = jet.pos[1];
  const mx = missile.pos[0], my = missile.pos[1];
  const d1 = Math.hypot(jx + Math.cos(perp1) * 8 - mx, jy + Math.sin(perp1) * 8 - my);
  const d2 = Math.hypot(jx + Math.cos(perp2) * 8 - mx, jy + Math.sin(perp2) * 8 - my);
  return d1 > d2 ? [perp1, perp2] : [perp2, perp1];
}

// Candidate (heading, throttle, label) triples, one family per maneuver. These
// are suggestions shaped by the geometry; the rollout decides.
export function maneuverActions(jet, missiles) {
  const out = [];
  const jx = jet.pos[0], jy = jet.pos[1];
  const heading = jet.heading;
  const lo = Math.max(MIN_THROTTLE, BASE_SPEED * THROTTLE_CHOICES[0]);
  const hi = Math.min(MAX_THROTTLE, BASE_SPEED * THROTTLE_CHOICES[THROTTLE_CHOICES.length - 1]);

  // IMMELMANN -- hard turn away from the nearest wall, full power
  const left = jx, right = WIDTH - jx, top = jy, bottom = FIELD_H - jy;
  const nearest = Math.min(left, right, top, bottom);
  let escape;
  if (nearest === left) escape = 0.0;
  else if (nearest === right) escape = Math.PI;
  else if (nearest === top) escape = Math.PI / 2;
  else escape = -Math.PI / 2;
  out.push([escape, hi, "IMMELMANN"]);

  if (missiles.length) {
    let nearestM = missiles[0], bestD = Infinity;
    for (const m of missiles) {
      const d = Math.hypot(m.pos[0] - jx, m.pos[1] - jy);
      if (d < bestD) { bestD = d; nearestM = m; }
    }
    const [perpA, perpB] = perpendicularOptions(jet, nearestM);
    const flee = Math.atan2(jy - nearestM.pos[1], jx - nearestM.pos[0]);
    out.push([perpA, lo, "COBRA"]);
    out.push([perpB, lo, "COBRA"]);
    out.push([perpA, BASE_SPEED * 0.7, "JINKING"]);
    out.push([perpB, BASE_SPEED * 0.7, "JINKING"]);
    out.push([flee + 0.5, lo, "FALLING_LEAF"]);
    out.push([flee - 0.5, lo, "FALLING_LEAF"]);
    out.push([heading + 0.6, hi, "BARREL_ROLL"]);
    out.push([heading - 0.6, hi, "BARREL_ROLL"]);
  }
  return out;
}

// Cost of each candidate future (planner_v3.cost_v3).
export function costV3(res, throttles, horizon) {
  horizon = horizon === undefined ? HORIZON : horizon;
  const C = res.C, T = res.T;
  // discount weights, normalised
  const w = new Float64Array(T);
  let wsum = 0.0;
  for (let k = 0; k < T; k++) { w[k] = Math.pow(DISCOUNT, k); wsum += w[k]; }
  const inv = 1.0 / Math.max(wsum, 1e-9);
  for (let k = 0; k < T; k++) w[k] *= inv;

  const cost = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    const base = c * T;
    let v = 0.0;

    // missiles
    if (res.hit_frame[c] >= 0) {
      v += HIT_COST * (1.0 + (1.0 - res.hit_frame[c] / horizon));
    } else {
      let near = 0.0;
      for (let k = 0; k < T; k++) {
        const gap = Math.max(res.dist_profile[base + k] - HIT_RADIUS, 0.0);
        let th = 1.0 - gap / THREAT_MARGIN;
        if (th < 0.0) th = 0.0; else if (th > 1.0) th = 1.0;
        near += th * th * th * w[k];
      }
      v += HIT_PROX_COST * near;
    }

    // walls
    const wall = res.wall_frame[c] >= 0;
    if (wall) {
      v += WALL_COST * (1.0 + (1.0 - res.wall_frame[c] / horizon));
    } else {
      let pinned = 0.0;
      for (let k = 0; k < T; k++) {
        const room = Math.max(res.wall_profile[base + k], 0.0);
        let sq = 1.0 - room / WALL_MARGIN;
        if (sq < 0.0) sq = 0.0; else if (sq > 1.0) sq = 1.0;
        pinned += sq * sq * sq * w[k];
      }
      v += WALL_PROX_COST * pinned;
    }

    // A future that ends fast and pointed at a wall is a crash the horizon only
    // hides: where the jet would coast to has to be inside the arena.
    const cx = res.end_x[c] + res.end_vx[c] * COAST;
    const cy = res.end_y[c] + res.end_vy[c] * COAST;
    const coastRoom = Math.min(
      Math.min(cx - JET_RADIUS, WIDTH - JET_RADIUS - cx),
      Math.min(cy - JET_RADIUS, FIELD_H - JET_RADIUS - cy));
    if (!wall && coastRoom < 0.0) v += WALL_COST;

    // terrain: lethal, but priced with its own much tighter proximity margin
    if (res.terrain_profile) {
      if (res.terrain_frame[c] >= 0) {
        v += TERRAIN_COST * (1.0 + (1.0 - res.terrain_frame[c] / horizon));
      } else {
        let squeeze = 0.0;
        for (let k = 0; k < T; k++) {
          const room = Math.max(res.terrain_profile[base + k], 0.0);
          let sq = 1.0 - room / TERRAIN_MARGIN;
          if (sq < 0.0) sq = 0.0; else if (sq > 1.0) sq = 1.0;
          squeeze += sq * sq * sq * w[k];
        }
        v += TERRAIN_PROX_COST * squeeze;
      }
    }

    const offCentre = Math.hypot(res.end_x[c] - CENTRE_X, res.end_y[c] - CENTRE_Y) / 800.0;
    v += CENTRE_COST * offCentre;
    v += THROTTLE_TIEBREAK * Math.abs(throttles[c] - BASE_SPEED_REF) / BASE_SPEED_REF;
    cost[c] = v;
  }
  return cost;
}

export const THROTTLES_FULL = THROTTLE_CHOICES;          // 12 headings x 4 = 48 + 9 + hold
export const THROTTLES_FAST = [0.35, 1.00, 1.50];        // 12 headings x 3 = 36 + 9 + hold

export class PlannerV3 {
  constructor(opts) {
    this.roll = new Rollout();
    this.rollVolley = new Rollout();
    this.throttleChoices = (opts && opts.throttles) || THROTTLES_FULL;
  }

  /**
   * Choose what to fly this frame, and whether to flare.
   *
   * `hold` is last frame's choice, (heading, throttle, label): the action grid
   * is laid out around the current heading, so without it the exact action that
   * was safe a frame ago may not be on offer now.
   */
  plan(jet, threats, level, opt) {
    opt = opt || {};
    const flares = opt.flares || [];
    const volley = opt.volley || null;
    const hold = opt.hold || null;
    const keepFuture = !!opt.keepFuture;

    const [gridH, gridT] = actionGrid(jet.heading, this.throttleChoices);
    const labels = gridH.map(() => "NORMAL");
    const headings = gridH.slice(), throttles = gridT.slice();
    for (const [h, t, l] of maneuverActions(jet, threats)) {
      headings.push(h); throttles.push(t); labels.push(l);
    }
    if (hold) {
      headings.push(hold[0]); throttles.push(hold[1]); labels.push(hold[2]);
    }

    const terrain = level.obstacles;
    let res = this.roll.run(jet, threats, headings, throttles, HORIZON,
                            terrain, level.datalink, flares, keepFuture);
    let cost = costV3(res, throttles);
    let costMin = Infinity;
    for (let c = 0; c < cost.length; c++) if (cost[c] < costMin) costMin = cost[c];

    let fire = false, gain = 0.0;
    // Every cost term is non-negative, so a future with the volley costs at
    // least FLARE_COST: when the best future without it is already cheaper than
    // that, the volley cannot win and imagining it is skipped. Exact.
    if (volley && threats.length && costMin > FLARE_COST) {
      const withVolley = flares.concat(volley);
      const resV = this.rollVolley.run(jet, threats, headings, throttles, HORIZON,
                                       terrain, level.datalink, withVolley, false);
      const costV = costV3(resV, throttles);
      let vMin = Infinity;
      for (let c = 0; c < costV.length; c++) {
        costV[c] += FLARE_COST;
        if (costV[c] < vMin) vMin = costV[c];
      }
      gain = costMin - vMin;
      if (gain > 0.0) {
        res = resV; cost = costV; fire = true; costMin = vMin;
      }
    }

    let best = 0;
    for (let c = 1; c < cost.length; c++) if (cost[c] < cost[best]) best = c;

    const nT = this.throttleChoices.length;
    const candidateScores = [];
    for (let hIdx = 0; hIdx < HEADING_DELTAS.length; hIdx++) {
      let m = Infinity;
      for (let t = 0; t < nT; t++) m = Math.min(m, cost[hIdx * nT + t]);
      candidateScores.push([gridH[hIdx * nT], m]);
    }

    let cornered = threats.length > 0;
    if (cornered) {
      for (let c = 0; c < res.C; c++) if (res.hit_frame[c] < 0) { cornered = false; break; }
    }

    return {
      heading: headings[best], throttle: throttles[best], label: labels[best],
      labels: labels, cornered: cornered,
      candidate_scores: candidateScores,
      danger: Math.max(0.0, Math.min(1.0, cost[best] / HIT_COST)),
      min_gap: res.min_dist[best] - HIT_RADIUS,
      // Even the best imagined future ends badly -- by missile, wall or terrain.
      lethal: res.hit_frame[best] >= 0 || res.wall_frame[best] >= 0 || res.terrain_frame[best] >= 0,
      flare: fire, flare_gain: gain, best: best,
      result: keepFuture ? res : null,
      cost: cost,
    };
  }
}
