// sim/tracker.py -- the jet's persistent picture of every missile, from radar returns alone.
import {
  BALLISTIC_LIFE_FRAMES, HEIGHT, HOMING_LIFE_FRAMES, WIDTH, normalizeAngle, clamp,
} from "./config.js";
import {
  AVOID_FAR, MEMORY_REACHED, PINCER_FLANK, SEEKER_COS_FOV, SEEKER_RANGE, avoidTurn,
} from "./doctrine.js";
import { insideAny, segmentBlocked } from "./geometry.js";

export const MAX_COAST_FRAMES = 120;
export const HISTORY = 9;
export const PURSUIT_MIN = 0.35;
export const LEAD_MIN = 0.08;
// A weaving or darting missile can only be predicted along its average line.
// The world model counts it as closer than that line by this many px per unit
// of estimated swing (a swing of 0.4 -- about 22 degrees -- is a 16 px margin).
export const WOBBLE_MARGIN_PX = 40.0;

// Before the network has seen a track: turn rate is unknown, so the planner
// assumes the most agile missile in the threat library (turn_hi).
export const PRIOR = {
  pursuit: 0.5, turn_rate: 0.10, turn_hi: 0.16, lead: 0.0,
  side: 0.0, guided: 0.5, threat: 0.0, bias: 0.0, flank: 0.0, wobble: 0.0,
};

// Could a seeker at (mx, my) pointing along mh see (px, py)? Doctrine only.
export function doctrineSees(mx, my, mh, px, py, terrain) {
  const dx = px - mx, dy = py - my;
  const d = Math.sqrt(dx * dx + dy * dy);
  if (d > SEEKER_RANGE) return false;
  if (dx * Math.cos(mh) + dy * Math.sin(mh) < SEEKER_COS_FOV * d) return false;
  return !(terrain.length && segmentBlocked(mx, my, px, py, terrain));
}

export class Track {
  constructor(tid, frame, x, y, jet) {
    this.tid = tid;
    this.x = this.birth_x = x;
    this.y = this.birth_y = y;
    this.vx = 0.0; this.vy = 0.0;
    this.heading = 0.0;
    this.speed = 0.0;
    this.visible = true;
    this.born = frame;
    this.last_seen = frame;
    this.coast = 0;
    this.heads = [];
    // A launch is cued with the jet's position, so that is the first belief.
    this.last_known = [jet.x, jet.y];
    this.seeker_edge = false;
    this.est = Object.assign({}, PRIOR);
    this.group = -1;
  }

  observe(frame, x, y, vx, vy) {
    this.x = x; this.y = y; this.vx = vx; this.vy = vy;
    this.speed = Math.sqrt(vx * vx + vy * vy);
    this.heading = Math.atan2(vy, vx);
    this.heads.push(this.heading);
    if (this.heads.length > HISTORY) this.heads.shift();
    this.visible = true;
    this.last_seen = frame;
    this.coast = 0;
  }

  // Mean heading change per frame over the last `span` frames.
  turnPerFrame(span = 4) {
    if (this.heads.length < 2) return 0.0;
    const n = Math.min(span, this.heads.length - 1);
    return normalizeAngle(this.heads[this.heads.length - 1] - this.heads[this.heads.length - 1 - n]) / n;
  }

  // Advance an unseen track one frame the way doctrine says it flies blind.
  deadReckon(terrain) {
    terrain = terrain || [];
    this.coast += 1;
    const guided = this.est.pursuit > 0.5;
    const tr = this.est.turn_rate;
    let turn = 0.0, aimD = AVOID_FAR;
    if (guided && this.last_known !== null) {
      const lx = this.last_known[0], ly = this.last_known[1];
      if (Math.sqrt((lx - this.x) * (lx - this.x) + (ly - this.y) * (ly - this.y)) < MEMORY_REACHED) {
        this.last_known = null;
      } else {
        const ex = lx - this.x, ey = ly - this.y;
        aimD = Math.sqrt(ex * ex + ey * ey);
        const diff = normalizeAngle(Math.atan2(ey, ex) - this.heading);
        turn = clamp(diff, -tr, tr);
      }
    }
    if (guided && terrain.length) {
      const av = avoidTurn(this.x, this.y, Math.cos(this.heading), Math.sin(this.heading),
                           this.speed, tr, terrain, aimD);
      if (av !== 0.0) turn = av * tr;
    }
    this.heading += turn;
    if (guided) this.heading += this.est.bias || 0.0;
    this.vx = Math.cos(this.heading) * this.speed;
    this.vy = Math.sin(this.heading) * this.speed;
    this.x += this.vx;
    this.y += this.vy;
    this.heads.push(this.heading);
    if (this.heads.length > HISTORY) this.heads.shift();
  }

  get age() { return this.last_seen - this.born + this.coast; }
}

export class Tracker {
  constructor(level) {
    this.level = level;
    this.terrain = level.obstacles.map((o) => [o[0], o[1], o[2]]);
    this.tracks = new Map();
    this.events = [];
    this.event_count = 0;
    this.dropped = new Map();
    // Conformal correction added to the turn-rate quantile; 0 flies on the raw one.
    this.turn_margin = 0.0;
  }

  _event(frame, kind, tid, detail) {
    this.events.push([frame, kind, tid, detail || ""]);
    if (this.events.length > 64) this.events.shift();
    this.event_count += 1;
  }

  update(frame, jet, visibleMissiles) {
    const seen = new Set();
    for (const m of visibleMissiles) {
      let t = this.tracks.get(m.uid);
      if (t === undefined) {
        t = new Track(m.uid, frame, m.x, m.y, jet);
        this.tracks.set(m.uid, t);
        if (this.dropped.has(m.uid)) {
          this.dropped.delete(m.uid);
          this._event(frame, "reacquired", m.uid, "after drop");
        } else {
          this._event(frame, "born", m.uid);
        }
      } else if (!t.visible) {
        this._event(frame, "reacquired", m.uid, "after " + t.coast + " f");
      }
      t.observe(frame, m.x, m.y, m.vx, m.vy);
      seen.add(m.uid);
    }

    const blindTerrain = this.level.smart ? this.terrain : [];
    for (const tid of Array.from(this.tracks.keys())) {
      const t = this.tracks.get(tid);
      if (seen.has(tid)) {
        this._updateBelief(t, jet);
        continue;
      }
      if (t.visible) {
        // Gone from the radar. Behind terrain it is merely hidden; in the open,
        // a missile that vanishes has been destroyed.
        let hidden = false;
        if (this.terrain.length) {
          for (const s of [0.0, 1.0, 2.0]) {
            if (segmentBlocked(jet.x, jet.y, t.x + s * t.vx, t.y + s * t.vy, this.terrain)) {
              hidden = true; break;
            }
          }
        }
        if (hidden) {
          t.visible = false;
          t.seeker_edge = false;
          this._event(frame, "occluded", tid);
        } else {
          this.tracks.delete(tid);
          this._event(frame, "destroyed", tid);
          continue;
        }
      }
      t.deadReckon(blindTerrain);
      if (t.coast > MAX_COAST_FRAMES || !(t.x >= 0 && t.x <= WIDTH && t.y >= 0 && t.y <= HEIGHT)
          || insideAny(t.x, t.y, this.terrain, 2.0)) {
        this.tracks.delete(tid);
        this.dropped.set(tid, frame);
        this._event(frame, "dropped", tid, "coasted " + t.coast + " f");
      }
    }
    if (this.dropped.size > 200) {
      for (const [k, f] of Array.from(this.dropped.entries())) {
        if (frame - f >= 600) this.dropped.delete(k);
      }
    }
  }

  // Keep the track's estimated last-known point in step with doctrine.
  _updateBelief(t, jet) {
    if (!this.level.smart) {
      t.last_known = [jet.x, jet.y];
      t.seeker_edge = true;
      return;
    }
    const sees = doctrineSees(t.x, t.y, t.heading, jet.x, jet.y, this.terrain);
    if (sees) {
      t.last_known = [jet.x, jet.y];
    } else if (t.last_known !== null
               && Math.sqrt((t.last_known[0] - t.x) * (t.last_known[0] - t.x)
                            + (t.last_known[1] - t.y) * (t.last_known[1] - t.y)) < MEMORY_REACHED) {
      t.last_known = null;
    }
    t.seeker_edge = sees;
  }

  // Estimates in: datalink groups are the connected components of live links.
  applyEstimates(tids, rows, linkPairs) {
    for (let i = 0; i < tids.length; i++) {
      const t = this.tracks.get(tids[i]);
      if (t !== undefined) t.est = rows[i];
    }
    const parent = new Map();
    for (const tid of this.tracks.keys()) parent.set(tid, tid);
    const find = (a) => {
      while (parent.get(a) !== a) {
        parent.set(a, parent.get(parent.get(a)));
        a = parent.get(a);
      }
      return a;
    };
    for (const [a, b, p] of (linkPairs || [])) {
      if (p >= 0.5 && parent.has(a) && parent.has(b)) {
        const ra = find(a), rb = find(b);
        if (ra !== rb) parent.set(rb, ra);
      }
    }
    const sizes = new Map();
    for (const tid of this.tracks.keys()) {
      const r = find(tid);
      sizes.set(r, (sizes.get(r) || 0) + 1);
    }
    for (const [tid, t] of this.tracks) {
      const r = find(tid);
      t.group = sizes.get(r) > 1 ? r : -1;
    }
  }

  // Every track as a world-model input, guidance parameters from the estimates.
  // Turn rate uses the network's upper (90%) quantile, not its mean.
  threatDicts(frame) {
    const out = [];
    const smart = this.level.smart;
    for (const t of this.tracks.values()) {
      const e = t.est;
      const guided = e.pursuit >= PURSUIT_MIN;
      const d = {
        pos: [t.x, t.y], vel: [t.vx, t.vy], heading: t.heading,
        speed: t.speed, radius: 2, tid: t.tid,
        turn_rate: (e.turn_hi === undefined ? e.turn_rate : e.turn_hi) + this.turn_margin,
      };
      if (frame !== undefined && frame !== null) {
        const life = guided ? HOMING_LIFE_FRAMES : BALLISTIC_LIFE_FRAMES;
        d.ttl = life - (frame - t.born);
      }
      if (guided) {
        d.lateral_bias = e.bias || 0.0;
        d.lead = e.lead >= LEAD_MIN ? e.lead : 0.0;
        d.side = Math.round(e.side);
        d.flank = e.flank === undefined ? PINCER_FLANK : e.flank;
        d.margin = WOBBLE_MARGIN_PX * (e.wobble || 0.0);
        if (smart) {
          d.seeker = true;
          d.group = this.level.datalink ? t.group : -1;
          d.last_known = t.last_known;
        } else {
          d.repel = true;    // classic homing missiles push apart
        }
      }
      out.push(d);
    }
    return out;
  }
}
