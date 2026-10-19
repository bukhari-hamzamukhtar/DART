// Constants, ported verbatim from neural_evasion/config.py and the sim modules.
export const WIDTH = 800, HEIGHT = 600, HUD_HEIGHT = 50;
export const FIELD_H = HEIGHT - HUD_HEIGHT;
export const FPS = 60;

export const GUIDANCE_HOMING = 0, GUIDANCE_BALLISTIC = 1, GUIDANCE_SMART = 2;

export const FLARE_LIFE_FRAMES = 120;
export const HOMING_LIFE_FRAMES = 510;
export const BALLISTIC_LIFE_FRAMES = 180;
export const JET_MISSILE_LIFE_FRAMES = 240;
export const LAUNCHER_COOLDOWN_FRAMES = 120;
export const CLUSTER_COOLDOWN_FRAMES = 360;
export const PINCER_COOLDOWN_FRAMES = 420;
export const JET_ATTACK_FRAMES = 240;

export const FLARE_START_CHARGES = 0;
export const FLARE_CHARGES_PER_CLUSTER = 1;

export const GRAPH_TICK = 4;

export const BASE_SPEED = 5.0, MIN_THROTTLE = 0.3, MAX_THROTTLE = 7.5;
export const THROTTLE_CHOICES = [0.35, 0.70, 1.00, 1.50];
export const ROLLOUT_HORIZON = 72;

// sim/worldmodel.py
export const JET_DRAG = 0.92, JET_THRUST_GAIN = 0.1, JET_RADIUS = 11, STEER_GAIN = 0.18;
// sim/entities.py
export const FLARE_DRAG = 0.85, FLARE_RADIUS = 6;
// sim/tactics.py
export const REPULSE_DIST = 26, REPULSE_FORCE = 0.20;
export const MIN_INTERCEPT_FRAMES = 5.0, MAX_INTERCEPT_FRAMES = 50.0;
export const FLARE_LEAD_BOOST = 1.4, MIN_FLARE_SPEED = 5.0, MAX_FLARE_SPEED = 14.0;

// Palette
export const BG_COLOR = "#050a14";
export const JET_COLOR = "#00ffc8";
export const MISSILE_COLOR = "#ff4444";
export const FLARE_COLOR = "#ffb020";
export const BALLISTIC_COLOR = "#ff64ff";
export const TERRAIN_FILL = "#0e1a16";
export const TERRAIN_CONTOUR = "#2e6046";

export const TWO_PI = Math.PI * 2;
export function normalizeAngle(a) { return ((a + Math.PI) % TWO_PI + TWO_PI) % TWO_PI - Math.PI; }
export function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
export function hypot(x, y) { return Math.sqrt(x * x + y * y); }

// Deterministic PRNG (mulberry32) so a seeded match replays the same way.
export function makeRng(seed) {
  let a = (seed >>> 0) || 0x9e3779b9;
  const rnd = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random: rnd,
    uniform: (lo, hi) => lo + (hi - lo) * rnd(),
    randint: (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1)),   // inclusive, like Python
    bits31: () => Math.floor(rnd() * 2147483648),
    choice: (arr) => arr[Math.floor(rnd() * arr.length)],
    shuffle: (arr) => {                                            // Fisher-Yates
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
      return arr;
    },
    choices: (items, weights) => {
      let total = 0;
      for (const w of weights) total += w;
      let r = rnd() * total;
      for (let i = 0; i < items.length; i++) { r -= weights[i]; if (r <= 0) return items[i]; }
      return items[items.length - 1];
    },
  };
}
