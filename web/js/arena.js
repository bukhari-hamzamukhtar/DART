// arena.py -- one match: the world and the rules that move it, pilot-agnostic.
import {
  CLUSTER_COOLDOWN_FRAMES, FLARE_CHARGES_PER_CLUSTER, FLARE_LIFE_FRAMES,
  FLARE_START_CHARGES, HEIGHT, HUD_HEIGHT, JET_ATTACK_FRAMES, PINCER_COOLDOWN_FRAMES,
  WIDTH, REPULSE_DIST, REPULSE_FORCE, makeRng,
} from "./config.js";
import { ERRATIC, FLANKER, LINK_RANGE, PINCER_SPEED, PINCER_TURN_RATE, PURSUIT, WEAVER, sampleStyle } from "./doctrine.js";
import {
  BallisticMissile, HomingMissile, Jet, JetMissile, Launcher, SmartMissile,
  applyStyle, applyClusterRepulsion,
} from "./entities.js";
import { insideAny, segmentBlocked } from "./geometry.js";

// Cluster-swarm launch profile, one entry per missile.
export const CLUSTER_OFFSETS = [-40, -20, 0, 20, 40];
export const CLUSTER_SPEEDS = [3.5, 3.7, 3.9, 4.1, 4.3];
export const CLUSTER_TURN_RATES = [0.06, 0.05, 0.04, 0.03, 0.025];
export const CLUSTER_LAT_BIASES = [-0.014, -0.007, 0.0, 0.007, 0.014];
export const CLUSTER_STYLES = [FLANKER, WEAVER, PURSUIT, ERRATIC, FLANKER];
export const PINCER_FAN = 0.35;

export class Arena {
  constructor(level, opt) {
    opt = opt || {};
    this.level = level;
    this.frame = 0;
    this.jet = new Jet(WIDTH / 2, HEIGHT / 2);
    this.missiles = [];
    this.ai_missiles = [];
    this.flares = [];
    this.blasts = [];                 // render-only: (x, y, color, age, count)
    this.terrain = level.obstacles.map((o) => [o[0], o[1], o[2]]);
    this.launchers = level.stations.map((st) => new Launcher(st.x, st.y, st.color));

    this.jet_invulnerable = !!opt.jet_invulnerable;
    this.launchers_invulnerable = !!opt.launchers_invulnerable;
    this.counter_attack = opt.counter_attack === undefined ? true : !!opt.counter_attack;
    this.rng = makeRng(opt.seed === undefined || opt.seed === null
      ? (Math.random() * 2 ** 31) | 0 : opt.seed);

    // The flare magazine. magazine=false lifts the limit.
    this.magazine = opt.magazine === undefined ? true : !!opt.magazine;
    this.flare_charges = FLARE_START_CHARGES;
    this._flare_paid = null;

    this.cluster_cooldowns = this.launchers.map(() => 0);
    this.pincer_cooldown = 0;
    this.last_attack_frame = 0;
    this._next_network = 1;

    this.game_over = false;
    this.user_won = false;
    this.cause = null;                // "missile" | "wall" | "terrain" | "bases"
    this.killer_uid = null;
    this.end_frame = null;
    this.stats = {
      fired: 0, flares: 0, maneuvers: 0, absorbed: 0,
      volleys: 0, flares_denied: 0, charges_earned: 0,
    };
  }

  // -- launching ----------------------------------------------
  _newNetwork() { return this._next_network++; }

  _addMissile(m) {
    m.born_frame = this.frame;
    this.missiles.push(m);
  }

  stationReady(i) {
    return i >= 0 && i < this.launchers.length && this.launchers[i].ready(this.frame);
  }

  // Fire one missile from station `i`, in a random style for this level.
  fire(i, opt) {
    opt = opt || {};
    if (!this.stationReady(i)) return false;
    const st = this.level.stations[i];
    this.launchers[i].fireAt(this.frame);
    const jet = this.jet;
    let m;
    if (st.kind === "ballistic") {
      m = new BallisticMissile(st.x, st.y, jet.x, jet.y);
      if (opt.speed !== undefined) {
        m.speed = opt.speed;
        m.vx = Math.cos(m.heading) * opt.speed;
        m.vy = Math.sin(m.heading) * opt.speed;
      }
    } else {
      if (this.level.smart) {
        m = new SmartMissile(st.x, st.y, jet, jet.x, jet.y, {
          speed: opt.speed, turn_rate: opt.turn_rate,
          lead: opt.lead === undefined ? this.level.lead : opt.lead,
          network: this.level.datalink ? this._newNetwork() : null,
        });
      } else {
        m = new HomingMissile(st.x, st.y, jet, jet.x, jet.y);
        if (opt.speed !== undefined) {
          m.speed = m.max_speed = opt.speed;
          m.vx = Math.cos(m.heading) * opt.speed;
          m.vy = Math.sin(m.heading) * opt.speed;
        }
        if (opt.turn_rate !== undefined) m.turn_rate = opt.turn_rate;
      }
      applyStyle(m, sampleStyle(this.rng, this.level.number, opt.style || null));
    }
    this._addMissile(m);
    this.stats.fired += 1;
    return true;
  }

  clusterReady(i) {
    if (i === undefined || i === null) {
      for (let k = 0; k < this.launchers.length; k++) if (this.clusterReady(k)) return true;
      return false;
    }
    return i >= 0 && i < this.launchers.length && this.level.stations[i].kind === "homing"
      && this.launchers[i].alive && this.cluster_cooldowns[i] === 0;
  }

  // Five staggered guided missiles, fanned out, each in its own style. Every
  // cluster earns the jet one flare volley.
  launchCluster(i) {
    if (i === undefined || i === null) {
      i = null;
      for (let k = 0; k < this.launchers.length; k++) {
        if (this.clusterReady(k)) { i = k; break; }
      }
    }
    if (i === null || !this.clusterReady(i)) return false;
    this.cluster_cooldowns[i] = CLUSTER_COOLDOWN_FRAMES;
    const st = this.level.stations[i], jet = this.jet;
    const baseA = Math.atan2(jet.y - st.y, jet.x - st.x);
    const net = this.level.datalink ? this._newNetwork() : null;
    const styles = this.rng.shuffle(CLUSTER_STYLES.slice());
    const flankSides = this.rng.shuffle([-1.0, 1.0]);
    for (let k = 0; k < CLUSTER_OFFSETS.length; k++) {
      const ra = CLUSTER_OFFSETS[k] * Math.PI / 180.0;
      const tx2 = st.x + Math.cos(baseA + ra) * 120;
      const ty2 = st.y + Math.sin(baseA + ra) * 120;
      let m;
      if (this.level.smart) {
        m = new SmartMissile(st.x, st.y, jet, tx2, ty2, {
          speed: CLUSTER_SPEEDS[k], turn_rate: CLUSTER_TURN_RATES[k],
          lead: this.level.lead, network: net,
          lateral_bias: CLUSTER_LAT_BIASES[k], spawn_delay: k * 3,
        });
        m.last_known = [jet.x, jet.y];   // the fan is the launch direction, the cue is the jet
      } else {
        m = new HomingMissile(st.x, st.y, jet, tx2, ty2);
        m.speed = CLUSTER_SPEEDS[k];
        m.max_speed = CLUSTER_SPEEDS[k];
        m.turn_rate = CLUSTER_TURN_RATES[k];
        m.spawn_delay = k * 3;
        m.lateral_bias = CLUSTER_LAT_BIASES[k];
      }
      const p = sampleStyle(this.rng, this.level.number, styles[k]);
      if (p.style === FLANKER && flankSides.length) p.side = flankSides.pop();
      applyStyle(m, p);
      this._addMissile(m);
    }
    this.stats.fired += 5;
    this.flare_charges += FLARE_CHARGES_PER_CLUSTER;
    this.stats.charges_earned += FLARE_CHARGES_PER_CLUSTER;
    return true;
  }

  pincerReady() {
    return this.level.pincer && this.pincer_cooldown === 0
      && this.launchers.some((l) => l.alive);
  }

  // Two missiles from every live battery on one datalink, split left and right.
  launchPincer(opt) {
    opt = opt || {};
    if (!this.pincerReady()) return false;
    this.pincer_cooldown = PINCER_COOLDOWN_FRAMES;
    const jet = this.jet;
    const net = this._newNetwork();
    let n = 0;
    for (let i = 0; i < this.level.stations.length; i++) {
      if (!this.launchers[i].alive) continue;
      const st = this.level.stations[i];
      const bearing = Math.atan2(jet.y - st.y, jet.x - st.x);
      const sides = [-1.0, 1.0];
      for (let j = 0; j < sides.length; j++) {
        const side = sides[j];
        const m = new SmartMissile(st.x, st.y, jet, jet.x, jet.y, {
          speed: opt.speed === undefined ? PINCER_SPEED : opt.speed,
          turn_rate: opt.turn_rate === undefined ? PINCER_TURN_RATE : opt.turn_rate,
          lead: this.level.lead, side: side, network: net,
          spawn_delay: j * 4, launch_heading: bearing - side * PINCER_FAN,
        });
        this._addMissile(m);
        n += 1;
      }
    }
    this.stats.fired += n;
    return true;
  }

  // -- the flare magazine -------------------------------------
  canFlare() { return !this.magazine || this.flare_charges > 0; }

  spendFlare() { return !this.magazine || this._payFlares(); }

  _payFlares() {
    if (this._flare_paid === this.frame) return true;
    if (this.flare_charges <= 0) return false;
    this.flare_charges -= 1;
    this._flare_paid = this.frame;
    this.stats.volleys += 1;
    return true;
  }

  // -- sensing ------------------------------------------------
  lineOfSight(ax, ay, bx, by) {
    return !(this.terrain.length && segmentBlocked(ax, ay, bx, by, this.terrain));
  }

  // What the radar can see: [missiles, flares]. Terrain blocks it.
  radarPicture() {
    if (!this.terrain.length) return [this.missiles.slice(), this.flares.slice()];
    const jx = this.jet.x, jy = this.jet.y, t = this.terrain;
    return [
      this.missiles.filter((m) => !segmentBlocked(jx, jy, m.x, m.y, t)),
      this.flares.filter((f) => !segmentBlocked(jx, jy, f.x, f.y, t)),
    ];
  }

  // -- the frame ----------------------------------------------
  beginFrame() {
    this.frame += 1;
    for (let k = 0; k < this.cluster_cooldowns.length; k++) {
      if (this.cluster_cooldowns[k] > 0) this.cluster_cooldowns[k] -= 1;
    }
    if (this.pincer_cooldown > 0) this.pincer_cooldown -= 1;
  }

  advance(pilot) {
    if (pilot) pilot.act(this);
    const fresh = this.flares.filter((f) => f.born_frame === null);
    if (fresh.length) {
      if (this.magazine && !this._payFlares()) {
        // Empty magazine: the flares never leave the jet.
        const gone = new Set(fresh);
        this.flares = this.flares.filter((f) => !gone.has(f));
        this.stats.flares -= fresh.length;
        this.stats.flares_denied += fresh.length;
      } else {
        for (const f of fresh) { f.born_frame = this.frame; f.dead = false; }
      }
    }
    this._counterAttack();
    this._physics();
  }

  step(pilot, commands) {
    this.beginFrame();
    if (commands) commands(this);
    this.advance(pilot);
  }

  // -- internals ----------------------------------------------
  _counterAttack() {
    if (!this.counter_attack || this.game_over) return;
    if (this.frame - this.last_attack_frame > JET_ATTACK_FRAMES) {
      const targets = this.launchers.filter((l) => l.alive);
      if (targets.length) {
        const jet = this.jet;
        const clear = targets.filter((l) => this.lineOfSight(jet.x, jet.y, l.x, l.y));
        const pool = clear.length ? clear : targets;
        let tgt = pool[0];
        let best = Infinity;
        for (const l of pool) {
          const d = Math.sqrt((l.x - jet.x) * (l.x - jet.x) + (l.y - jet.y) * (l.y - jet.y));
          if (d < best) { best = d; tgt = l; }
        }
        const jm = new JetMissile(jet.x, jet.y, tgt.x, tgt.y);
        jm.born_frame = this.frame;
        this.ai_missiles.push(jm);
      }
      this.last_attack_frame = this.frame;
    }
  }

  _blast(x, y, color, count) {
    this.blasts.push({ x: x, y: y, color: color, born: this.frame, count: count });
    if (this.blasts.length > 24) this.blasts.shift();
  }

  _end(userWon, cause, killer) {
    if (this.game_over) return;
    this.game_over = true;
    this.user_won = userWon;
    this.cause = cause;
    this.killer_uid = killer === undefined ? null : killer;
    this.end_frame = this.frame;
  }

  _physics() {
    const jet = this.jet;
    jet.update();
    const r = jet.radius;
    if (jet.x < r || jet.x > WIDTH - r || jet.y < r || jet.y > HEIGHT - HUD_HEIGHT - r) {
      if (this.jet_invulnerable) this._bounceJet();
      else if (!this.game_over) {
        this._blast(jet.x, jet.y, "#00ffc8", 50);
        this._end(true, "wall");
      }
    } else if (this.terrain.length && insideAny(jet.x, jet.y, this.terrain, r)) {
      if (this.jet_invulnerable) this._pushOutOfTerrain();
      else if (!this.game_over) {
        this._blast(jet.x, jet.y, "#00ffc8", 50);
        this._end(true, "terrain");
      }
    }

    this._updateFlares();
    this._updateAiMissiles();
    this._updateMissiles();
    applyClusterRepulsion(this.missiles, REPULSE_DIST, REPULSE_FORCE);

    if (!this.launchers_invulnerable && this.launchers.length
        && !this.launchers.some((l) => l.alive)) {
      this._end(false, "bases");
    }
  }

  _bounceJet() {
    const jet = this.jet, r = this.jet.radius;
    if (jet.x < r || jet.x > WIDTH - r) { jet.vx = -jet.vx; jet.heading = Math.PI - jet.heading; }
    if (jet.y < r || jet.y > HEIGHT - HUD_HEIGHT - r) { jet.vy = -jet.vy; jet.heading = -jet.heading; }
    jet.x = Math.min(Math.max(jet.x, r + 1.0), WIDTH - r - 1.0);
    jet.y = Math.min(Math.max(jet.y, r + 1.0), HEIGHT - HUD_HEIGHT - r - 1.0);
  }

  _pushOutOfTerrain() {
    const jet = this.jet;
    for (const o of this.terrain) {
      const d = Math.sqrt((jet.x - o[0]) * (jet.x - o[0]) + (jet.y - o[1]) * (jet.y - o[1]));
      const lim = o[2] + jet.radius + 1.0;
      if (d < lim) {
        const nx = d > 1e-6 ? (jet.x - o[0]) / d : 1.0;
        const ny = d > 1e-6 ? (jet.y - o[1]) / d : 0.0;
        jet.x = o[0] + nx * lim; jet.y = o[1] + ny * lim;
        const dot = jet.vx * nx + jet.vy * ny;
        if (dot < 0) { jet.vx -= 2 * dot * nx; jet.vy -= 2 * dot * ny; }
        jet.heading = Math.atan2(jet.vy, jet.vx);
      }
    }
  }

  _updateFlares() {
    const alive = [];
    for (const f of this.flares) {
      f.update();
      if (f.dead || this.frame - f.born_frame > FLARE_LIFE_FRAMES) continue;
      alive.push(f);
    }
    this.flares = alive;
  }

  _updateAiMissiles() {
    const alive = [];
    for (const aim of this.ai_missiles) {
      aim.update();
      for (const lnc of this.launchers) {
        if (lnc.alive && Math.sqrt((lnc.x - aim.x) * (lnc.x - aim.x) + (lnc.y - aim.y) * (lnc.y - aim.y)) < 15 + aim.radius) {
          if (!this.launchers_invulnerable) lnc.alive = false;
          aim.dead = true;
          this._blast(lnc.x, lnc.y, "#ff5032", 30);
          break;
        }
      }
      if (!aim.dead && this.terrain.length && insideAny(aim.x, aim.y, this.terrain, aim.radius)) {
        aim.dead = true;
        this._blast(aim.x, aim.y, "#78a08c", 8);
      }
      if (!aim.dead && this.frame - aim.born_frame <= aim.life_frames) alive.push(aim);
    }
    this.ai_missiles = alive;
  }

  // Datalink: a missile with no lock of its own guides on a salvo partner's.
  _shareLocks(smart) {
    const lockers = smart.filter((m) => m.lock && m.network !== null);
    if (!lockers.length) return;
    for (const m of smart) {
      if (m.lock || m.network === null || m.spawn_delay > 0) continue;
      for (const o of lockers) {
        if (o !== m && o.network === m.network
            && Math.sqrt((o.x - m.x) * (o.x - m.x) + (o.y - m.y) * (o.y - m.y)) < LINK_RANGE) {
          m.linked = true;
          break;
        }
      }
    }
  }

  _updateMissiles() {
    const jet = this.jet;
    const smart = this.missiles.filter((m) => m.kind === "smart");
    if (smart.length) {
      for (const m of smart) m.sense(jet, this.flares, this.terrain);
      if (this.level.datalink) this._shareLocks(smart);
    }

    const alive = [];
    for (const m of this.missiles) {
      m.update();

      if (Math.sqrt((jet.x - m.x) * (jet.x - m.x) + (jet.y - m.y) * (jet.y - m.y)) < jet.radius + m.radius) {
        if (this.jet_invulnerable) this.stats.absorbed += 1;
        else {
          this._blast(jet.x, jet.y, "#00ffc8", 50);
          this._end(true, "missile", m.uid);
        }
        continue;
      }

      let hitFlare = false;
      for (const f of this.flares) {
        if (Math.sqrt((f.x - m.x) * (f.x - m.x) + (f.y - m.y) * (f.y - m.y)) < f.radius + m.radius) {
          f.dead = true;
          hitFlare = true;
          this._blast(f.x, f.y, "#ffb020", 5);
          break;
        }
      }
      if (hitFlare) continue;

      let hitAim = false;
      for (const aim of this.ai_missiles) {
        if (Math.sqrt((aim.x - m.x) * (aim.x - m.x) + (aim.y - m.y) * (aim.y - m.y)) < aim.radius + m.radius) {
          aim.dead = true;
          hitAim = true;
          this._blast(aim.x, aim.y, "#ffffff", 20);
          break;
        }
      }
      if (hitAim) continue;

      if (this.terrain.length && insideAny(m.x, m.y, this.terrain, m.radius)) {
        this._blast(m.x, m.y, "#a09678", 10);
        continue;
      }

      if (m.dead || this.frame - m.born_frame > m.life_frames) continue;
      alive.push(m);
    }
    this.missiles = alive;
  }
}
