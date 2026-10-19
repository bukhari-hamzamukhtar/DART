// sim/doctrine.py -- the threat library: seeker limits, flight styles, terrain avoidance.
import { clamp } from "./config.js";

export const SEEKER_RANGE = 340.0;
export const SEEKER_HALF_FOV = 70.0 * Math.PI / 180.0;
export const SEEKER_COS_FOV = Math.cos(SEEKER_HALF_FOV);
export const DECOY_RANGE = 0.7 * SEEKER_RANGE;
export const LINK_RANGE = 380.0;
export const MEMORY_REACHED = 24.0;
export const MAX_TGO = 60.0;

export const SMART_SPEED = 5.0, SMART_TURN_RATE = 0.11, L3_LEAD = 0.55;
export const PINCER_SPEED = 5.2, PINCER_TURN_RATE = 0.10;

export const PURSUIT = "PURSUIT", FLANKER = "FLANKER", WEAVER = "WEAVER", ERRATIC = "ERRATIC";
export const STYLES = [PURSUIT, FLANKER, WEAVER, ERRATIC];
export const STYLE_MIX = { 1: [0.35, 0.25, 0.20, 0.20], 2: [0.25, 0.30, 0.20, 0.25], 3: [0.30, 0.25, 0.25, 0.20] };
export const FLANK_ANGLE = [0.45, 0.90], PINCER_FLANK = 0.40;
export const FLANK_NEAR = 60.0, FLANK_SPAN = 180.0;
export const WEAVE_AMP = [0.25, 0.50], WEAVE_FREQ = [0.07, 0.16];
export const JINK_AMP = [0.30, 0.60], JINK_PERIOD = [10, 26];
export const WOBBLE_NEAR = 25.0, WOBBLE_SPAN = 90.0, JINK_TABLE = 12;

export const PURSUIT_STYLE = {
  style: PURSUIT, side: 0.0, flank: 0.0, weave_amp: 0.0, weave_freq: 0.0,
  weave_phase: 0.0, jink_amp: 0.0, jink_period: 1, jink_seed: 0,
};

export const AVOID_MARGIN = 14.0, AVOID_LOOK_MIN = 50.0, AVOID_RADII = 1.2;
export const AVOID_LOOK_MAX = 260.0, AVOID_FAR = 1e18;

// Where a guided missile steers this frame (lead pursuit + style deviation).
export function guidanceAim(mx, my, speed, jx, jy, jvx, jvy, lead, side, flank = PINCER_FLANK, wobble = 0.0) {
  if (lead === 0.0 && side === 0.0 && wobble === 0.0) return [jx, jy];
  const rx = jx - mx, ry = jy - my;
  const d = Math.sqrt(rx * rx + ry * ry);
  let tgo = speed > 1e-6 ? d / speed : MAX_TGO;
  if (tgo > MAX_TGO) tgo = MAX_TGO;
  let ax = jx + lead * jvx * tgo, ay = jy + lead * jvy * tgo;
  if (d > 1e-9 && (side !== 0.0 || wobble !== 0.0)) {
    let k = 0.0;
    if (side !== 0.0) k = side * flank * clamp((d - FLANK_NEAR) / FLANK_SPAN, 0.0, 1.0);
    if (wobble !== 0.0) k += wobble * clamp((d - WOBBLE_NEAR) / WOBBLE_SPAN, 0.0, 1.0);
    ax -= ry * k;
    ay += rx * k;
  }
  return [ax, ay];
}

export function sampleStyle(rng, levelNumber, style = null) {
  if (style === null) style = rng.choices(STYLES, STYLE_MIX[levelNumber] || STYLE_MIX[1]);
  const p = Object.assign({}, PURSUIT_STYLE);
  p.style = style;
  if (style === FLANKER) {
    p.side = rng.choice([-1.0, 1.0]);
    p.flank = rng.uniform(FLANK_ANGLE[0], FLANK_ANGLE[1]);
  } else if (style === WEAVER) {
    p.weave_amp = rng.uniform(WEAVE_AMP[0], WEAVE_AMP[1]);
    p.weave_freq = rng.uniform(WEAVE_FREQ[0], WEAVE_FREQ[1]);
    p.weave_phase = rng.uniform(0.0, 2.0 * Math.PI);
  } else if (style === ERRATIC) {
    p.jink_amp = rng.uniform(JINK_AMP[0], JINK_AMP[1]);
    p.jink_period = rng.randint(JINK_PERIOD[0], JINK_PERIOD[1]);
    p.jink_seed = rng.bits31();
  }
  return p;
}

// Dart number `n` of an erratic missile, in [-1, 1). splitmix64 mixing, as in Python.
const MASK64 = (1n << 64n) - 1n;
const K1 = 0x9E3779B97F4A7C15n, K2 = 0xBF58476D1CE4E5B9n, K3 = 0x94D049BB133111EBn;
export function jinkValue(seed, n) {
  let z = (BigInt(seed) * K1 + BigInt(n + 1) * K2) & MASK64;
  z = (((z ^ (z >> 30n)) & MASK64) * K2) & MASK64;
  z = (((z ^ (z >> 27n)) & MASK64) * K3) & MASK64;
  z = (z ^ (z >> 31n)) & MASK64;
  return Number(z >> 11n) * (2.0 / 9007199254740992.0) - 1.0;
}

export function wobbleAt(weaveAmp, weaveFreq, weavePhase, jinkAmp, jink, age) {
  return weaveAmp * Math.sin(weaveFreq * age + weavePhase) + jinkAmp * jink;
}

// -1, 0 or +1: which way a seeker must turn to clear the nearest peak ahead.
export function avoidTurn(mx, my, ch, sh, speed, turnRate, obstacles, aimD = AVOID_FAR) {
  if (turnRate <= 0.0) return 0.0;
  let look = speed / turnRate * AVOID_RADII + AVOID_LOOK_MIN;
  if (look > AVOID_LOOK_MAX) look = AVOID_LOOK_MAX;
  let best = AVOID_FAR, turn = 0.0;
  for (let i = 0; i < obstacles.length; i++) {
    const ox = obstacles[i][0], oy = obstacles[i][1], r = obstacles[i][2];
    const ex = ox - mx, ey = oy - my;
    const along = ex * ch + ey * sh;
    if (along <= 0.0 || along > look + r || along - r > aimD) continue;
    const lateral = ey * ch - ex * sh;
    const reach = r + AVOID_MARGIN;
    if (lateral >= reach || lateral <= -reach) continue;
    if (along < best) { best = along; turn = lateral >= 0.0 ? -1.0 : 1.0; }
  }
  return turn;
}
