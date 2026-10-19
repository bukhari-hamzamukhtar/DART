// sim/entities.py -- each object owns its physics. Lifetimes are counted in frames.
import {
  BASE_SPEED, MIN_THROTTLE, MAX_THROTTLE, WIDTH, HEIGHT, FLARE_DRAG, FLARE_RADIUS,
  HOMING_LIFE_FRAMES, BALLISTIC_LIFE_FRAMES, JET_MISSILE_LIFE_FRAMES,
  LAUNCHER_COOLDOWN_FRAMES, GUIDANCE_HOMING, GUIDANCE_BALLISTIC, GUIDANCE_SMART,
  normalizeAngle, clamp,
} from "./config.js";
import {
  AVOID_FAR, DECOY_RANGE, FLANKER, MEMORY_REACHED, PINCER_FLANK, PURSUIT_STYLE,
  SEEKER_COS_FOV, SEEKER_RANGE, SMART_SPEED, SMART_TURN_RATE, avoidTurn, guidanceAim,
  jinkValue, wobbleAt,
} from "./doctrine.js";
import { segmentBlocked } from "./geometry.js";

let _uid = 0;
export function newUid() { return ++_uid; }
export function resetUids() { _uid = 0; }

export const STYLE_FIELDS = ["style", "side", "flank", "weave_amp", "weave_freq",
                             "weave_phase", "jink_amp", "jink_period", "jink_seed"];

export function applyStyle(m, style) {
  for (const k of STYLE_FIELDS) m[k] = style[k];
}

// A missile's aim swing (tangent of the angle) at flight age `age`.
function styleSwing(m, age) {
  const jink = m.jink_amp ? jinkValue(m.jink_seed, Math.floor(age / m.jink_period)) : 0.0;
  return wobbleAt(m.weave_amp, m.weave_freq, m.weave_phase, m.jink_amp, jink, age);
}

const TRAIL = 12;
function pushTrail(t, x, y, max) {
  t.push(x, y);
  while (t.length > max * 2) { t.shift(); t.shift(); }
}

export class Jet {
  constructor(x, y) {
    this.x = x; this.y = y;
    this.vx = 0.0; this.vy = 0.0;
    this.heading = 0.0;
    this.speed = BASE_SPEED;       // commanded throttle; the jet controls it
    this.drag = 0.92;
    this.radius = 11;
    this.trail = [];
  }
  setThrottle(speed) { this.speed = clamp(speed, MIN_THROTTLE, MAX_THROTTLE); }
  update() {
    this.vx = this.vx * this.drag + Math.cos(this.heading) * this.speed * 0.1;
    this.vy = this.vy * this.drag + Math.sin(this.heading) * this.speed * 0.1;
    this.x += this.vx;
    this.y += this.vy;
    pushTrail(this.trail, this.x, this.y, 20);
  }
}

export class JetMissile {
  constructor(x, y, tx, ty) {
    this.kind = "jet";
    this.x = x; this.y = y;
    this.heading = Math.atan2(ty - y, tx - x);
    this.speed = 4.0;
    this.radius = 4;
    this.born_frame = 0;
    this.life_frames = JET_MISSILE_LIFE_FRAMES;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.dead = false;
    this.trail = [];
  }
  update() {
    this.x += this.vx; this.y += this.vy;
    pushTrail(this.trail, this.x, this.y, TRAIL);
    if (this.x < 0 || this.x > WIDTH || this.y < 0 || this.y > HEIGHT) this.dead = true;
  }
}

export class FlareParticle {
  constructor(x, y, angle, speed) {
    this.uid = newUid();
    this.x = x; this.y = y;
    this.vx = Math.cos(angle) * speed;
    this.vy = Math.sin(angle) * speed;
    this.drag = FLARE_DRAG;
    this.radius = FLARE_RADIUS;
    this.born_frame = null;        // stamped by the arena the frame it appears
    this.dead = false;
  }
  update() {
    this.vx *= this.drag; this.vy *= this.drag;
    this.x += this.vx; this.y += this.vy;
  }
}

export class HomingMissile {
  constructor(x, y, target, tx, ty) {
    this.kind = "homing";
    this.mtype = GUIDANCE_HOMING;
    this.life_frames = HOMING_LIFE_FRAMES;
    this.mode = "PURSUIT";
    this.uid = newUid();
    this.x = x; this.y = y;
    this.target = target;
    this.heading = Math.atan2(ty - y, tx - x);
    this.speed = 5.0; this.max_speed = 5.0;
    this.turn_rate = 0.15;
    this.radius = 2;
    this.born_frame = 0;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.dead = false;
    this.trail = [];
    this.spawn_delay = 0;
    this.lateral_bias = 0.0;       // persistent curve for cluster spread
    this.age = 0;
    applyStyle(this, PURSUIT_STYLE);
  }
  update() {
    if (this.spawn_delay > 0) { this.spawn_delay -= 1; return; }
    const a = this.age;
    const t = this.target;
    // Plain pursuit aims at the jet itself; a flanker, weaver or erratic
    // missile aims off the line of sight.
    const aim = guidanceAim(this.x, this.y, this.speed, t.x, t.y, t.vx, t.vy,
                            0.0, this.side, this.flank, styleSwing(this, a));
    const dh = Math.atan2(aim[1] - this.y, aim[0] - this.x);
    const diff = normalizeAngle(dh - this.heading);
    this.heading += clamp(diff, -this.turn_rate, this.turn_rate);
    this.heading += this.lateral_bias;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.x += this.vx; this.y += this.vy;
    this.age = a + 1;
    pushTrail(this.trail, this.x, this.y, TRAIL);
    if (this.x < 0 || this.x > WIDTH || this.y < 0 || this.y > HEIGHT) this.dead = true;
  }
}

export class BallisticMissile {
  constructor(x, y, tx, ty) {
    this.kind = "ballistic";
    this.mtype = GUIDANCE_BALLISTIC;
    this.life_frames = BALLISTIC_LIFE_FRAMES;
    this.mode = "BALLISTIC";
    this.turn_rate = 0.0;          // never steers
    this.lateral_bias = 0.0;
    this.spawn_delay = 0;
    this.uid = newUid();
    this.x = x; this.y = y;
    this.heading = Math.atan2(ty - y, tx - x);
    this.speed = 7.0;
    this.radius = 2;
    this.born_frame = 0;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.dead = false;
    this.trail = [];
  }
  update() {
    this.x += this.vx; this.y += this.vy;
    pushTrail(this.trail, this.x, this.y, TRAIL);
    if (this.x < 0 || this.x > WIDTH || this.y < 0 || this.y > HEIGHT) this.dead = true;
  }
}

export const SMART_MODE_COLORS = {
  LOCK: "#ff4444",      // own seeker has the jet
  LINK: "#5ac8ff",      // guided by a salvo partner's lock
  MEMORY: "#ffd246",    // flying to where it last knew the jet was
  DECOY: "#ff9628",     // seduced by a flare
  SEARCH: "#9696a0",    // nothing to guide on
  CUE: "#ff7878",       // just launched, on the battery's cue
};

export class SmartMissile {
  constructor(x, y, target, cueX, cueY, opt) {
    opt = opt || {};
    this.kind = "smart";
    this.mtype = GUIDANCE_SMART;
    this.life_frames = HOMING_LIFE_FRAMES;
    this.uid = newUid();
    this.x = x; this.y = y;
    this.target = target;
    this.heading = (opt.launch_heading === undefined || opt.launch_heading === null)
      ? Math.atan2(cueY - y, cueX - x) : opt.launch_heading;
    this.speed = this.max_speed = opt.speed === undefined ? SMART_SPEED : opt.speed;
    this.turn_rate = opt.turn_rate === undefined ? SMART_TURN_RATE : opt.turn_rate;
    this.lead = opt.lead === undefined ? 0.0 : opt.lead;
    applyStyle(this, PURSUIT_STYLE);
    // A pincer member is a flanker at the pincer's deviation.
    this.side = opt.side === undefined ? 0.0 : opt.side;
    this.flank = opt.flank === undefined ? PINCER_FLANK : opt.flank;
    if (this.side !== 0.0) this.style = FLANKER;
    this.network = opt.network === undefined ? null : opt.network;
    this.lateral_bias = opt.lateral_bias === undefined ? 0.0 : opt.lateral_bias;
    this.spawn_delay = opt.spawn_delay === undefined ? 0 : opt.spawn_delay;
    this.radius = 2;
    this.born_frame = 0;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.dead = false;
    this.trail = [];
    this.last_known = [cueX, cueY];     // the battery's cue
    this.lock = false;
    this.linked = false;
    this.decoy = null;
    this.mode = "CUE";
    this.age = 0;
    this.obstacles = [];
    this.avoiding = false;
  }

  // Could this seeker see (px, py)? Range, field of view, line of sight.
  sees(px, py, obstacles, maxRange) {
    if (maxRange === undefined) maxRange = SEEKER_RANGE;
    const dx = px - this.x, dy = py - this.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > maxRange) return false;
    if ((dx * Math.cos(this.heading) + dy * Math.sin(this.heading)) < SEEKER_COS_FOV * d) return false;
    return !(obstacles.length && segmentBlocked(this.x, this.y, px, py, obstacles));
  }

  // Pass 1: what the seeker can see, before anything moves this frame.
  sense(jet, flares, obstacles) {
    this.obstacles = obstacles;
    this.lock = false;
    this.linked = false;
    this.decoy = null;
    if (this.spawn_delay > 0) return;
    this.lock = this.sees(jet.x, jet.y, obstacles);
    const jdx = jet.x - this.x, jdy = jet.y - this.y;
    const jetD2 = jdx * jdx + jdy * jdy;
    let bestD2 = 0.0, best = null;
    for (let i = 0; i < flares.length; i++) {
      const f = flares[i];
      if (f.dead) continue;
      const ex = f.x - this.x, ey = f.y - this.y;
      const fd2 = ex * ex + ey * ey;
      // With the jet in view a flare only wins if it is closer; with the jet
      // out of view the seeker grabs whatever it can see.
      if (this.lock && fd2 >= jetD2) continue;
      if (best !== null && fd2 >= bestD2) continue;
      if (this.sees(f.x, f.y, obstacles, DECOY_RANGE)) { best = f; bestD2 = fd2; }
    }
    if (best !== null) this.decoy = best;
  }

  // Pass 3: steer and move.
  update() {
    if (this.spawn_delay > 0) { this.spawn_delay -= 1; return; }
    const jet = this.target;
    const a = this.age;
    let aim = null;
    if (this.decoy !== null) {
      aim = [this.decoy.x, this.decoy.y];
      this.mode = "DECOY";
    } else if (this.lock || this.linked) {
      aim = guidanceAim(this.x, this.y, this.speed, jet.x, jet.y, jet.vx, jet.vy,
                        this.lead, this.side, this.flank, styleSwing(this, a));
      this.last_known = [jet.x, jet.y];
      this.mode = this.lock ? "LOCK" : "LINK";
    } else if (this.last_known !== null) {
      const lx = this.last_known[0], ly = this.last_known[1];
      if (Math.sqrt((lx - this.x) * (lx - this.x) + (ly - this.y) * (ly - this.y)) < MEMORY_REACHED) {
        this.last_known = null;
        this.mode = "SEARCH";
      } else {
        aim = this.last_known;
        this.mode = "MEMORY";
      }
    } else {
      this.mode = "SEARCH";
    }

    let turn = 0.0, aimD = AVOID_FAR;
    if (aim !== null) {
      const ex = aim[0] - this.x, ey = aim[1] - this.y;
      aimD = Math.sqrt(ex * ex + ey * ey);
      const diff = normalizeAngle(Math.atan2(ey, ex) - this.heading);
      turn = clamp(diff, -this.turn_rate, this.turn_rate);
    }
    let av = 0.0;
    if (this.obstacles.length) {
      av = avoidTurn(this.x, this.y, Math.cos(this.heading), Math.sin(this.heading),
                     this.speed, this.turn_rate, this.obstacles, aimD);
    }
    this.avoiding = av !== 0.0;
    if (av !== 0.0) turn = av * this.turn_rate;
    this.heading += turn;
    this.heading += this.lateral_bias;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.x += this.vx; this.y += this.vy;
    this.age = a + 1;
    pushTrail(this.trail, this.x, this.y, TRAIL);
    if (this.x < 0 || this.x > WIDTH || this.y < 0 || this.y > HEIGHT) this.dead = true;
  }
}

export class Launcher {
  constructor(x, y, color) {
    this.x = x; this.y = y; this.color = color;
    this.alive = true;
    this.last_fire_frame = -1e6;
  }
  ready(frame) { return this.alive && frame - this.last_fire_frame >= LAUNCHER_COOLDOWN_FRAMES; }
  fireAt(frame) { this.last_fire_frame = frame; }
}

// sim/tactics.py: angle and speed to throw a flare into a missile's predicted path.
export function predictInterceptFlare(jet, m, MIN_F, MAX_F, BOOST, MINS, MAXS) {
  const dx = m.x - jet.x, dy = m.y - jet.y;
  const rvx = m.vx - jet.vx, rvy = m.vy - jet.vy;
  const relSq = rvx * rvx + rvy * rvy + 1e-5;
  let t = -(dx * rvx + dy * rvy) / relSq;
  t = clamp(t, MIN_F, MAX_F);
  const tx = m.x + m.vx * t, ty = m.y + m.vy * t;
  const angle = Math.atan2(ty - jet.y, tx - jet.x);
  const dist = Math.sqrt((tx - jet.x) * (tx - jet.x) + (ty - jet.y) * (ty - jet.y));
  const speed = clamp(dist / t * BOOST, MINS, MAXS);
  return [angle, speed];
}

// sim/tactics.py: classic homing missiles push each other apart (heading only).
export function applyClusterRepulsion(missiles, REPULSE_DIST, REPULSE_FORCE) {
  const homing = [];
  for (let i = 0; i < missiles.length; i++) {
    if (missiles[i].kind === "homing" && !missiles[i].dead) homing.push(missiles[i]);
  }
  for (let i = 0; i < homing.length; i++) {
    for (let j = i + 1; j < homing.length; j++) {
      const m1 = homing[i], m2 = homing[j];
      const dx = m2.x - m1.x, dy = m2.y - m1.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > 0 && d < REPULSE_DIST) {
        const strength = (REPULSE_DIST - d) / REPULSE_DIST * REPULSE_FORCE;
        m1.heading = normalizeAngle(m1.heading - strength * Math.PI * (1 - d / REPULSE_DIST));
        m2.heading = normalizeAngle(m2.heading + strength * Math.PI * (1 - d / REPULSE_DIST));
      }
    }
  }
}
