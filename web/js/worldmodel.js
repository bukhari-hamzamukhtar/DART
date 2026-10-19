// sim/worldmodel_v3.py + sim/rollout_kernel.py in the browser.
//
// The jet, every missile, every flare, terrain and datalinks rolled forward
// together, for each candidate action. Mirrors entities.Jet, SmartMissile,
// HomingMissile, BallisticMissile, FlareParticle, Arena._update_flares,
// Arena._update_missiles, tactics.apply_cluster_repulsion and the flight-style
// and terrain-avoidance rules in doctrine -- same update order, same arithmetic.
import {
  WIDTH, HEIGHT, HUD_HEIGHT, FIELD_H, JET_DRAG, JET_THRUST_GAIN, JET_RADIUS,
  STEER_GAIN, MIN_THROTTLE, MAX_THROTTLE, FLARE_DRAG, FLARE_RADIUS,
  REPULSE_DIST, REPULSE_FORCE, TWO_PI,
} from "./config.js";
import {
  AVOID_FAR, AVOID_LOOK_MAX, AVOID_LOOK_MIN, AVOID_MARGIN, AVOID_RADII,
  DECOY_RANGE, FLANK_NEAR, FLANK_SPAN, JINK_TABLE, LINK_RANGE, MAX_TGO,
  MEMORY_REACHED, PINCER_FLANK, SEEKER_COS_FOV, SEEKER_RANGE, WOBBLE_NEAR,
  WOBBLE_SPAN, jinkValue,
} from "./doctrine.js";

const SR2 = SEEKER_RANGE * SEEKER_RANGE;
const DR2 = DECOY_RANGE * DECOY_RANGE;
const LR2 = LINK_RANGE * LINK_RANGE;
const MR2 = MEMORY_REACHED * MEMORY_REACHED;
export const FAR = 1e18;
export const NEVER = 1e9;

function wrap(a) { return ((a + Math.PI) % TWO_PI + TWO_PI) % TWO_PI - Math.PI; }

// geometry.segment_blocked on squared distances.
function blocked(ax, ay, bx, by, ox, oy, orad2, K) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  for (let j = 0; j < K; j++) {
    const ex = ox[j] - ax, ey = oy[j] - ay;
    let t = 0.0;
    if (l2 > 1e-12) {
      t = (ex * dx + ey * dy) / l2;
      if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    }
    const cx = t * dx - ex, cy = t * dy - ey;
    if (cx * cx + cy * cy < orad2[j]) return true;
  }
  return false;
}

// doctrine.avoid_turn
function avoid(mx, my, ch, sh, speed, turnRate, ox, oy, orad, K, aimD) {
  if (turnRate <= 0.0) return 0.0;
  let look = speed / turnRate * AVOID_RADII + AVOID_LOOK_MIN;
  if (look > AVOID_LOOK_MAX) look = AVOID_LOOK_MAX;
  let best = AVOID_FAR, turn = 0.0;
  for (let j = 0; j < K; j++) {
    const r = orad[j];
    const ex = ox[j] - mx, ey = oy[j] - my;
    const along = ex * ch + ey * sh;
    if (along <= 0.0 || along > look + r || along - r > aimD) continue;
    const lateral = ey * ch - ex * sh;
    const reach = r + AVOID_MARGIN;
    if (lateral >= reach || lateral <= -reach) continue;
    if (along < best) { best = along; turn = lateral >= 0.0 ? -1.0 : 1.0; }
  }
  return turn;
}

// Missile state and guidance parameters packed into flat arrays (ThreatPack).
// `pool` is a Rollout, so a plan allocates nothing: the arrays are reused.
export class ThreatPack {
  constructor(threats, pool) {
    const n = this.n = threats.length;
    const alloc = pool
      ? (name, size) => pool._f64("P_" + name, size).fill(0.0, 0, size)
      : (name, size) => new Float64Array(size);
    const allocI = pool
      ? (name, size) => pool._i32("P_" + name, size).fill(0, 0, size)
      : (name, size) => new Int32Array(size);
    const allocU = pool
      ? (name, size) => pool._u8("P_" + name, size).fill(0, 0, size)
      : (name, size) => new Uint8Array(size);
    const f = (key, dflt) => {
      const out = alloc(key, Math.max(1, n));
      for (let i = 0; i < n; i++) {
        const v = threats[i][key];
        out[i] = (v === undefined || v === null) ? dflt : v;
      }
      return out;
    };
    this.x = alloc("x", Math.max(1, n));
    this.y = alloc("y", Math.max(1, n));
    this.heading = alloc("heading", Math.max(1, n));
    this.speed = alloc("speed", Math.max(1, n));
    for (let i = 0; i < n; i++) {
      const t = threats[i];
      this.x[i] = t.pos[0];
      this.y[i] = t.pos[1];
      const vmag = Math.sqrt(t.vel[0] * t.vel[0] + t.vel[1] * t.vel[1]);
      this.heading[i] = (t.heading === undefined || t.heading === null)
        ? Math.atan2(t.vel[1], t.vel[0]) : t.heading;
      const declared = (t.speed === undefined || t.speed === null) ? 0.0 : t.speed;
      this.speed[i] = Math.max(declared > 0.0 ? declared : vmag, 1e-6);
    }
    this.turn_rate = f("turn_rate", 0.0);
    this.lateral_bias = f("lateral_bias", 0.0);
    this.spawn_delay = f("spawn_delay", 0.0);
    this.radius = f("radius", 2.0);
    this.lead = f("lead", 0.0);
    this.side = f("side", 0.0);
    this.flank = f("flank", PINCER_FLANK);
    this.ttl = f("ttl", NEVER);
    this.margin = f("margin", 0.0);
    this.weave_amp = f("weave_amp", 0.0);
    this.weave_freq = f("weave_freq", 0.0);
    this.weave_phase = f("weave_phase", 0.0);
    this.jink_amp = f("jink_amp", 0.0);
    this.seeker = allocU("seeker", Math.max(1, n));
    this.repel = allocU("repel", Math.max(1, n));
    this.group = allocI("group", Math.max(1, n));
    this.has_lk = allocU("has_lk", Math.max(1, n));
    this.lkx = alloc("lkx", Math.max(1, n));
    this.lky = alloc("lky", Math.max(1, n));
    this.jink_period = allocI("jink_period", Math.max(1, n));
    this.age = allocI("age", Math.max(1, n));
    this.jink_base = allocI("jink_base", Math.max(1, n));
    this.jink_vals = alloc("jink_vals", Math.max(1, n * JINK_TABLE));
    for (let i = 0; i < n; i++) {
      const t = threats[i];
      this.seeker[i] = t.seeker ? 1 : 0;
      this.repel[i] = t.repel ? 1 : 0;
      this.group[i] = (t.group === undefined || t.group === null) ? -1 : t.group;
      const lk = t.last_known;
      this.has_lk[i] = (lk === undefined || lk === null) ? 0 : 1;
      this.lkx[i] = lk ? lk[0] : 0.0;
      this.lky[i] = lk ? lk[1] : 0.0;
      this.jink_period[i] = Math.max(1, t.jink_period || 1);
      this.age[i] = t.age || 0;
      this.jink_base[i] = Math.floor(this.age[i] / this.jink_period[i]);
      if (t.jink_vals) {
        const k = Math.min(t.jink_vals.length, JINK_TABLE);
        for (let j = 0; j < k; j++) this.jink_vals[i * JINK_TABLE + j] = t.jink_vals[j];
      } else if (this.jink_amp[i] !== 0.0 && t.jink_seed !== undefined) {
        for (let j = 0; j < JINK_TABLE; j++) {
          this.jink_vals[i * JINK_TABLE + j] = jinkValue(t.jink_seed, this.jink_base[i] + j);
        }
      }
    }
  }
}

// A missile's true state, guidance parameters and flight style, as the world
// model takes them (worldmodel_v3.oracle_threat).
export function oracleThreat(m, datalink, frame0) {
  const d = {
    pos: [m.x, m.y], vel: [m.vx, m.vy], heading: m.heading, speed: m.speed,
    turn_rate: m.turn_rate || 0.0, lateral_bias: m.lateral_bias || 0.0,
    spawn_delay: m.spawn_delay || 0, radius: m.radius, tid: m.uid,
  };
  if (frame0 !== undefined && frame0 !== null) d.ttl = m.life_frames - (frame0 - m.born_frame);
  if (m.kind === "homing") {
    d.repel = true;
  } else if (m.kind === "smart") {
    d.seeker = true;
    d.lead = m.lead;
    d.group = (datalink && m.network !== null) ? m.network : -1;
    d.last_known = m.last_known;
  }
  if (m.style !== undefined) {
    d.side = m.side;
    d.flank = m.flank;
    d.weave_amp = m.weave_amp;
    d.weave_freq = m.weave_freq;
    d.weave_phase = m.weave_phase;
    d.jink_amp = m.jink_amp;
    d.jink_period = m.jink_period;
    d.age = m.age;
    if (m.jink_amp) {
      const base = Math.floor(m.age / m.jink_period);
      const vals = [];
      for (let j = 0; j < JINK_TABLE; j++) vals.push(jinkValue(m.jink_seed, base + j));
      d.jink_vals = vals;
    }
  }
  return d;
}

// A flare in the air, as the world model takes it.
export function flareState(f, frame0, FLARE_LIFE_FRAMES) {
  return { pos: [f.x, f.y], vel: [f.vx, f.vy], ttl: FLARE_LIFE_FRAMES - (frame0 - f.born_frame) };
}

// Reusable buffers: a rollout allocates nothing in the hot path.
export class Rollout {
  constructor() { this.b = {}; }

  _f64(name, size) {
    let v = this.b[name];
    if (v === undefined || v.length < size) { v = new Float64Array(size); this.b[name] = v; }
    return v;
  }
  _u8(name, size) {
    let v = this.b[name];
    if (v === undefined || v.length < size) { v = new Uint8Array(size); this.b[name] = v; }
    return v;
  }
  _i32(name, size) {
    let v = this.b[name];
    if (v === undefined || v.length < size) { v = new Int32Array(size); this.b[name] = v; }
    return v;
  }

  // Every flare's position after each step's drift, and whether it still burns.
  _flareTracks(flares, T) {
    const F = flares.length;
    const FX = this._f64("FX", Math.max(1, T * F));
    const FY = this._f64("FY", Math.max(1, T * F));
    const FPRES = this._u8("FPRES", Math.max(1, T * F));
    if (!F) return F;
    const x = this._f64("fx0", F), y = this._f64("fy0", F);
    const vx = this._f64("fvx0", F), vy = this._f64("fvy0", F);
    const ttl = this._f64("fttl", F);
    for (let i = 0; i < F; i++) {
      x[i] = flares[i].pos[0]; y[i] = flares[i].pos[1];
      vx[i] = flares[i].vel[0]; vy[i] = flares[i].vel[1];
      ttl[i] = flares[i].ttl === undefined ? NEVER : flares[i].ttl;
    }
    for (let k = 0; k < T; k++) {
      for (let i = 0; i < F; i++) {
        vx[i] *= FLARE_DRAG; vy[i] *= FLARE_DRAG;
        x[i] += vx[i]; y[i] += vy[i];
        FX[k * F + i] = x[i];
        FY[k * F + i] = y[i];
        FPRES[k * F + i] = k <= ttl[i] ? 1 : 0;
      }
    }
    return F;
  }

  // The jet under each candidate (target heading, throttle): Jet.update, steered.
  _jetPaths(jet, headings, throttles, C, T) {
    const JX = this._f64("JX", C * T), JY = this._f64("JY", C * T);
    const VX = this._f64("VX", C * T), VY = this._f64("VY", C * T);
    for (let c = 0; c < C; c++) {
      let x = jet.pos[0], y = jet.pos[1], vx = jet.vel[0], vy = jet.vel[1], h = jet.heading;
      const spd = Math.min(Math.max(throttles[c], MIN_THROTTLE), MAX_THROTTLE);
      const tgt = headings[c];
      const base = c * T;
      for (let k = 0; k < T; k++) {
        h += wrap(tgt - h) * STEER_GAIN;
        vx = vx * JET_DRAG + Math.cos(h) * spd * JET_THRUST_GAIN;
        vy = vy * JET_DRAG + Math.sin(h) * spd * JET_THRUST_GAIN;
        x += vx; y += vy;
        JX[base + k] = x; JY[base + k] = y;
        VX[base + k] = vx; VY[base + k] = vy;
      }
    }
  }

  // Clearance from the walls and from terrain along every jet path.
  _jetProfiles(C, T, ox, oy, orad, K) {
    const JX = this.b.JX, JY = this.b.JY;
    const WALL = this._f64("WALL", C * T);
    const TERR = this._f64("TERR", Math.max(1, K ? C * T : 1));
    for (let c = 0; c < C; c++) {
      const base = c * T;
      for (let k = 0; k < T; k++) {
        const x = JX[base + k], y = JY[base + k];
        let w = Math.min(x - JET_RADIUS, WIDTH - JET_RADIUS - x);
        w = Math.min(w, y - JET_RADIUS);
        WALL[base + k] = Math.min(w, FIELD_H - JET_RADIUS - y);
        if (K) {
          let best = 1e18;
          for (let j = 0; j < K; j++) {
            const dx = x - ox[j], dy = y - oy[j];
            const v = Math.sqrt(dx * dx + dy * dy) - (orad[j] + JET_RADIUS);
            if (v < best) best = v;
          }
          TERR[base + k] = best;
        }
      }
    }
  }

  /**
   * Roll every candidate forward `T` frames.
   *
   * jet: {pos, vel, heading}; threats: world-model inputs; flares: flareState
   * dicts. Returns per-candidate outcomes the planner cost reads.
   */
  run(jet, threats, headings, throttles, T, terrain, datalink, flares, keepPaths) {
    terrain = terrain || [];
    flares = flares || [];
    const K = terrain.length;
    const ox = this._f64("ox", Math.max(1, K)), oy = this._f64("oy", Math.max(1, K));
    const orad = this._f64("orad", Math.max(1, K)), orad2 = this._f64("orad2", Math.max(1, K));
    for (let j = 0; j < K; j++) {
      ox[j] = terrain[j][0]; oy[j] = terrain[j][1];
      orad[j] = terrain[j][2]; orad2[j] = terrain[j][2] * terrain[j][2];
    }

    const C = headings.length;
    const P = new ThreatPack(threats, this);
    const Px = P.x, Py = P.y, Phead = P.heading, Pspeed = P.speed, Pturn = P.turn_rate,
          Pbias = P.lateral_bias, Pdelay = P.spawn_delay, Prad = P.radius, Plead = P.lead,
          Pside = P.side, Pflank = P.flank, Pttl = P.ttl, Pmargin = P.margin,
          PwA = P.weave_amp, PwF = P.weave_freq, PwP = P.weave_phase, PjA = P.jink_amp,
          Pseek = P.seeker, Prepel = P.repel, Pgroup = P.group, PhasLk = P.has_lk,
          Plkx = P.lkx, Plky = P.lky, PjP = P.jink_period, Page = P.age,
          PjB = P.jink_base, PjV = P.jink_vals;
    const M = P.n;
    const F = this._flareTracks(flares, T);
    this._jetPaths(jet, headings, throttles, C, T);
    this._jetProfiles(C, T, ox, oy, orad, K);

    const JX = this.b.JX, JY = this.b.JY, VX = this.b.VX, VY = this.b.VY;
    const WALL = this.b.WALL, TERR = this.b.TERR;
    const FX = this.b.FX, FY = this.b.FY, FPRES = this.b.FPRES;
    // Per step: where the flares are, as one disc. A missile farther from that
    // disc than the decoy range cannot see any of them, and farther than its
    // own kill radius cannot touch one, so both scans are skipped outright.
    const FCX = this._f64("FCX", Math.max(1, T));
    const FCY = this._f64("FCY", Math.max(1, T));
    const FRAD = this._f64("FRAD", Math.max(1, T));
    if (F) {
      for (let k = 0; k < T; k++) {
        let sx = 0.0, sy = 0.0;
        for (let i = 0; i < F; i++) { sx += FX[k * F + i]; sy += FY[k * F + i]; }
        const cx = sx / F, cy = sy / F;
        let rad = 0.0;
        for (let i = 0; i < F; i++) {
          const dx = FX[k * F + i] - cx, dy = FY[k * F + i] - cy;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d > rad) rad = d;
        }
        FCX[k] = cx; FCY[k] = cy; FRAD[k] = rad;
      }
    }

    const DIST = this._f64("DIST", C * T);
    const STRUCK = this._u8("STRUCK", C * T);
    const FHIT = this._i32("FHIT", Math.max(1, C * Math.max(1, F)));
    const TX = keepPaths ? this._f64("TX", Math.max(1, C * M * T)) : null;
    const TY = keepPaths ? this._f64("TY", Math.max(1, C * M * T)) : null;
    const ALIVEP = keepPaths ? this._u8("ALIVEP", Math.max(1, C * M * T)) : null;

    // per-missile scratch, reused across candidates
    const px = this._f64("px", Math.max(1, M)), py = this._f64("py", Math.max(1, M));
    const h = this._f64("h", Math.max(1, M));
    const ch = this._f64("ch", Math.max(1, M)), sh = this._f64("sh", Math.max(1, M));
    const delay = this._f64("delay", Math.max(1, M));
    const lkx = this._f64("lkx", Math.max(1, M)), lky = this._f64("lky", Math.max(1, M));
    const hasLk = this._u8("hasLk", Math.max(1, M));
    const alive = this._u8("alive", Math.max(1, M));
    const own = this._u8("own", Math.max(1, M));
    const guided = this._u8("guided", Math.max(1, M));
    const moving = this._u8("moving", Math.max(1, M));
    const decoy = this._i32("decoy", Math.max(1, M));
    const age = this._i32("age", Math.max(1, M));
    const statics = this._u8("statics", Math.max(1, M));
    const wobbly = this._u8("wobbly", Math.max(1, M));
    const fdead = this._u8("fdead", Math.max(1, F));
    const fkill = this._u8("fkill", Math.max(1, F));

    let anyRepel = false;
    for (let i = 0; i < M; i++) {
      // A missile that never turns needs no steering maths; swarm repulsion can
      // still turn one, so those are never static.
      statics[i] = (Pturn[i] === 0.0 && Pbias[i] === 0.0 && !Prepel[i]) ? 1 : 0;
      wobbly[i] = (PwA[i] !== 0.0 || PjA[i] !== 0.0) ? 1 : 0;
      if (Prepel[i]) anyRepel = true;
    }

    const minDist = new Float64Array(C);
    const hitFrame = new Int32Array(C);
    const wallFrame = new Int32Array(C);
    const terrFrame = new Int32Array(C);
    const minWall = new Float64Array(C);
    const endX = new Float64Array(C), endY = new Float64Array(C);
    const endVX = new Float64Array(C), endVY = new Float64Array(C);

    for (let c = 0; c < C; c++) {
      for (let i = 0; i < M; i++) {
        px[i] = Px[i]; py[i] = Py[i];
        h[i] = Phead[i];
        ch[i] = Math.cos(Phead[i]); sh[i] = Math.sin(Phead[i]);
        delay[i] = Pdelay[i];
        hasLk[i] = PhasLk[i];
        lkx[i] = Plkx[i]; lky[i] = Plky[i];
        alive[i] = 1;
        age[i] = Page[i];
      }
      for (let f = 0; f < F; f++) { fdead[f] = 0; fkill[f] = 0; FHIT[c * F + f] = -1; }

      const jbase = c * T;
      for (let k = 0; k < T; k++) {
        const jx = JX[jbase + k], jy = JY[jbase + k];
        const fbase = k * F;

        // 1. seekers look, before anything moves (SmartMissile.sense)
        for (let i = 0; i < M; i++) {
          moving[i] = delay[i] <= 0.0 ? 1 : 0;
          let o = 0, dec = -1, best = 0.0, d2 = 0.0;
          if (Pseek[i] && moving[i] && alive[i]) {
            const dx = jx - px[i], dy = jy - py[i];
            d2 = dx * dx + dy * dy;
            if (d2 <= SR2 && dx * ch[i] + dy * sh[i] >= SEEKER_COS_FOV * Math.sqrt(d2)) {
              if (K === 0 || !blocked(px[i], py[i], jx, jy, ox, oy, orad2, K)) o = 1;
            }
            const scx = FCX[k] - px[i], scy = FCY[k] - py[i];
            const sNear = F ? Math.sqrt(scx * scx + scy * scy) - FRAD[k] : 1e18;
            for (let f = 0; sNear <= DECOY_RANGE && f < F; f++) {
              if (fdead[f] || !FPRES[fbase + f]) continue;
              const ex = FX[fbase + f] - px[i], ey = FY[fbase + f] - py[i];
              const fd2 = ex * ex + ey * ey;
              if (o && fd2 >= d2) continue;
              if (dec >= 0 && fd2 >= best) continue;
              if (fd2 <= DR2 && ex * ch[i] + ey * sh[i] >= SEEKER_COS_FOV * Math.sqrt(fd2)) {
                if (K === 0 || !blocked(px[i], py[i], FX[fbase + f], FY[fbase + f], ox, oy, orad2, K)) {
                  dec = f; best = fd2;
                }
              }
            }
          }
          own[i] = o;
          decoy[i] = dec;
        }

        // 2. datalinks share locks (Arena._share_locks)
        for (let i = 0; i < M; i++) {
          let g = (own[i] || !Pseek[i]) ? 1 : 0;
          if (!g && datalink && Pgroup[i] >= 0 && moving[i] && alive[i]) {
            for (let j = 0; j < M; j++) {
              if (j !== i && own[j] && Pgroup[j] === Pgroup[i]) {
                const ex = px[j] - px[i], ey = py[j] - py[i];
                if (ex * ex + ey * ey < LR2) { g = 1; break; }
              }
            }
          }
          guided[i] = g;
        }

        // 3. steer, move, collide
        let dmin = 1e18, struck = 0;
        for (let i = 0; i < M; i++) {
          if (!alive[i]) {
            if (keepPaths) {
              const pi = (c * M + i) * T + k;
              TX[pi] = px[i]; TY[pi] = py[i]; ALIVEP[pi] = 0;
            }
            continue;
          }
          if (!moving[i]) {
            delay[i] -= 1.0;
          } else if (statics[i]) {
            px[i] += ch[i] * Pspeed[i];
            py[i] += sh[i] * Pspeed[i];
            age[i] += 1;
          } else {
            let steer = false, ax = 0.0, ay = 0.0;
            if (decoy[i] >= 0) {
              ax = FX[fbase + decoy[i]]; ay = FY[fbase + decoy[i]];
              steer = true;
            } else if (guided[i]) {
              let wob = 0.0;
              if (wobbly[i]) {
                const a = age[i];
                let jv = 0.0;
                if (PjA[i] !== 0.0) {
                  const idx = Math.floor(a / PjP[i]) - PjB[i];
                  jv = (idx >= 0 && idx < JINK_TABLE) ? PjV[i * JINK_TABLE + idx] : 0.0;
                }
                wob = PwA[i] * Math.sin(PwF[i] * a + PwP[i]) + PjA[i] * jv;
              }
              // doctrine.guidance_aim, inlined
              const lead = Plead[i], side = Pside[i];
              if (lead === 0.0 && side === 0.0 && wob === 0.0) {
                ax = jx; ay = jy;
              } else {
                const rx = jx - px[i], ry = jy - py[i];
                const d = Math.sqrt(rx * rx + ry * ry);
                let tgo = Pspeed[i] > 1e-6 ? d / Pspeed[i] : MAX_TGO;
                if (tgo > MAX_TGO) tgo = MAX_TGO;
                ax = jx + lead * VX[jbase + k] * tgo;
                ay = jy + lead * VY[jbase + k] * tgo;
                if (d > 1e-9 && (side !== 0.0 || wob !== 0.0)) {
                  let kk = 0.0;
                  if (side !== 0.0) {
                    let fade = (d - FLANK_NEAR) / FLANK_SPAN;
                    if (fade < 0.0) fade = 0.0; else if (fade > 1.0) fade = 1.0;
                    kk = side * Pflank[i] * fade;
                  }
                  if (wob !== 0.0) {
                    let wf = (d - WOBBLE_NEAR) / WOBBLE_SPAN;
                    if (wf < 0.0) wf = 0.0; else if (wf > 1.0) wf = 1.0;
                    kk += wob * wf;
                  }
                  ax -= ry * kk;
                  ay += rx * kk;
                }
              }
              steer = true;
              if (Pseek[i]) { lkx[i] = jx; lky[i] = jy; hasLk[i] = 1; }
            } else if (hasLk[i]) {
              const ex = lkx[i] - px[i], ey = lky[i] - py[i];
              if (ex * ex + ey * ey < MR2) hasLk[i] = 0;
              else { ax = lkx[i]; ay = lky[i]; steer = true; }
            }
            const tr = Pturn[i];
            let tn = 0.0, aimD = AVOID_FAR;
            if (steer) {
              const ex = ax - px[i], ey = ay - py[i];
              aimD = Math.sqrt(ex * ex + ey * ey);
              const diff = wrap(Math.atan2(ey, ex) - h[i]);
              tn = diff < -tr ? -tr : (diff > tr ? tr : diff);
            }
            if (Pseek[i] && K > 0) {
              const av = avoid(px[i], py[i], ch[i], sh[i], Pspeed[i], tr, ox, oy, orad, K, aimD);
              if (av !== 0.0) tn = av * tr;
            }
            h[i] += tn;
            h[i] += Pbias[i];
            ch[i] = Math.cos(h[i]); sh[i] = Math.sin(h[i]);
            px[i] += ch[i] * Pspeed[i];
            py[i] += sh[i] * Pspeed[i];
            age[i] += 1;
          }

          if (keepPaths) {
            const pi = (c * M + i) * T + k;
            TX[pi] = px[i]; TY[pi] = py[i]; ALIVEP[pi] = 1;
          }

          // The arena order: the jet, then flares, then terrain, then the map
          // edge and burn-out. Only a missile that got past the jet can spend
          // itself on a flare.
          const dx = jx - px[i], dy = jy - py[i];
          const d = Math.sqrt(dx * dx + dy * dy);
          const hr = JET_RADIUS + Prad[i];
          const de = d - Pmargin[i];
          if (de < dmin) dmin = de;
          if (de < hr) struck = 1;
          if (d < hr) { alive[i] = 0; continue; }

          let gone = false;
          const rr = FLARE_RADIUS + Prad[i], rr2 = rr * rr;
          const ccx = FCX[k] - px[i], ccy = FCY[k] - py[i];
          const cNear = F ? Math.sqrt(ccx * ccx + ccy * ccy) - FRAD[k] : 1e18;
          for (let f = 0; cNear <= rr && f < F; f++) {
            if (fdead[f] || !FPRES[fbase + f]) continue;
            const ex = FX[fbase + f] - px[i], ey = FY[fbase + f] - py[i];
            if (ex * ex + ey * ey < rr2) { fkill[f] = 1; gone = true; break; }
          }
          if (!gone) {
            for (let j = 0; j < K; j++) {
              const ex = px[i] - ox[j], ey = py[i] - oy[j];
              const tr2 = orad[j] + Prad[i];
              if (ex * ex + ey * ey < tr2 * tr2) { gone = true; break; }
            }
          }
          if (!gone && moving[i] && (px[i] < 0.0 || px[i] > WIDTH || py[i] < 0.0 || py[i] > HEIGHT)) gone = true;
          if (!gone && k > Pttl[i]) gone = true;
          if (gone) alive[i] = 0;
        }
        DIST[jbase + k] = M ? dmin : FAR;
        STRUCK[jbase + k] = struck;

        // 4. classic homing missiles push apart (tactics.apply_cluster_repulsion)
        if (anyRepel) {
          for (let i = 0; i < M; i++) {
            if (!(Prepel[i] && alive[i])) continue;
            for (let j = i + 1; j < M; j++) {
              if (!(Prepel[j] && alive[j])) continue;
              const dd = Math.sqrt((px[j] - px[i]) * (px[j] - px[i]) + (py[j] - py[i]) * (py[j] - py[i]));
              if (dd > 0.0 && dd < REPULSE_DIST) {
                const strength = (REPULSE_DIST - dd) / REPULSE_DIST * REPULSE_FORCE;
                h[i] = wrap(h[i] - strength * Math.PI * (1 - dd / REPULSE_DIST));
                h[j] = wrap(h[j] + strength * Math.PI * (1 - dd / REPULSE_DIST));
                ch[i] = Math.cos(h[i]); sh[i] = Math.sin(h[i]);
                ch[j] = Math.cos(h[j]); sh[j] = Math.sin(h[j]);
              }
            }
          }
        }

        // 5. flares used up this step are gone from the next one
        for (let f = 0; f < F; f++) {
          if (fkill[f]) { fkill[f] = 0; fdead[f] = 1; FHIT[c * F + f] = k; }
        }
      }

      // per-candidate summaries
      let dm = Infinity, hf = -1, wf = -1, tf = -1, mw = Infinity;
      for (let k = 0; k < T; k++) {
        const dv = DIST[jbase + k];
        if (dv < dm) dm = dv;
        if (hf < 0 && STRUCK[jbase + k]) hf = k;
        const wv = WALL[jbase + k];
        if (wv < mw) mw = wv;
        if (wf < 0 && wv < 0.0) wf = k;
        if (K && tf < 0 && TERR[jbase + k] < 0.0) tf = k;
      }
      minDist[c] = dm;
      hitFrame[c] = hf;
      wallFrame[c] = wf;
      terrFrame[c] = tf;
      minWall[c] = mw;
      endX[c] = JX[jbase + T - 1];
      endY[c] = JY[jbase + T - 1];
      endVX[c] = VX[jbase + T - 1];
      endVY[c] = VY[jbase + T - 1];
    }

    return {
      C: C, M: M, T: T, F: F,
      min_dist: minDist, hit_frame: hitFrame, wall_frame: wallFrame,
      terrain_frame: terrFrame, min_wall: minWall,
      dist_profile: DIST, wall_profile: WALL,
      terrain_profile: K ? TERR : null,
      end_x: endX, end_y: endY, end_vx: endVX, end_vy: endVY,
      jet_x: JX, jet_y: JY, threat_x: TX, threat_y: TY, alive_path: ALIVEP,
      flare_hit: FHIT,
    };
  }
}
