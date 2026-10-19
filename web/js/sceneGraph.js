// sim/scene_graph.py -- one snapshot of the dynamic graph the network looks at.
import { WIDTH, FIELD_H } from "./config.js";
import { DECOY_RANGE, LINK_RANGE, SEEKER_COS_FOV } from "./doctrine.js";
import { segmentBlocked } from "./geometry.js";

export const JET = 0, TRACK = 1, FLARE = 2, OBSTACLE = 3, STATION = 4;
export const NODE_TYPES = ["JET", "TRACK", "FLARE", "OBSTACLE", "STATION"];

export const THREAT = 0, WATCH = 1, SEEKER = 2, LINK = 3, NEAR = 4,
             DECOY = 5, TERRAIN = 6, OCCLUDE = 7, LAUNCH = 8;
export const EDGE_TYPES = ["THREAT", "WATCH", "SEEKER", "LINK", "NEAR", "DECOY",
                           "TERRAIN", "OCCLUDE", "LAUNCH"];

export const NODE_DIM = 23;
export const EDGE_DIM = 9;

export const THREAT_RADIUS = 520.0;
export const NEAR_RADIUS = 120.0;
export const TERRAIN_RADIUS = 110.0;
export const LAUNCH_AGE = 90;
export const LAUNCH_SNAP = 30.0;
export const MAX_FLARE_NODES = 16;

export const KEY_JET = 0;
export function keyTrack(tid) { return 1000000 + tid; }
export function keyFlare(uid) { return 2000000 + uid; }
export function keyObstacle(k) { return 3000000 + k; }
export function keyStation(k) { return 4000000 + k; }

function nodeRow(out, base, ntype, x, y, vx, vy, heading, visible, age, coast, turn, size, lk, speed, jet) {
  for (let i = 0; i < NODE_DIM; i++) out[base + i] = 0.0;
  if (jet) {
    // Where a track points relative to its line of sight to the jet: a pursuer
    // points at it, a flanker holds off to one side, a weaver swings across it.
    const off = heading - Math.atan2(jet[1] - y, jet[0] - x);
    out[base + 20] = Math.sin(off);
    out[base + 21] = Math.cos(off);
    out[base + 22] = Math.min(Math.sqrt((jet[0] - x) * (jet[0] - x) + (jet[1] - y) * (jet[1] - y)) / 400.0, 2.0);
  }
  out[base + ntype] = 1.0;
  out[base + 5] = x / WIDTH;
  out[base + 6] = y / FIELD_H;
  out[base + 7] = vx / 8.0;
  out[base + 8] = vy / 8.0;
  out[base + 9] = Math.cos(heading);
  out[base + 10] = Math.sin(heading);
  out[base + 11] = visible;
  out[base + 12] = Math.min(age / 300.0, 1.0);
  out[base + 13] = Math.min(coast / 60.0, 1.0);
  out[base + 14] = Math.max(-2.0, Math.min(2.0, turn / 0.15));
  out[base + 15] = size / 60.0;
  if (lk) {
    out[base + 16] = 1.0;
    out[base + 17] = (lk[0] - x) / 400.0;
    out[base + 18] = (lk[1] - y) / 400.0;
  }
  out[base + 19] = (speed === undefined || speed === null ? Math.sqrt(vx * vx + vy * vy) : speed) / 8.0;
}

export class SceneGraphBuilder {
  constructor(level) {
    this.level = level;
    this.terrain = level.obstacles.map((o) => [o[0], o[1], o[2]]);
  }

  build(frame, jet, tracker, flares, launchers) {
    const keys = [], ntype = [], names = [], kin = [], pos = [];
    const rowsData = [];

    const add = (key, t, rowFn, x, y, vx, vy, h, name) => {
      keys.push(key); ntype.push(t); names.push(name);
      pos.push(x, y); kin.push(x, y, vx, vy, h);
      rowsData.push(rowFn);
      return keys.length - 1;
    };

    const jetI = add(KEY_JET, JET,
      (out, base) => nodeRow(out, base, JET, jet.x, jet.y, jet.vx, jet.vy, jet.heading, 1.0, frame, 0, 0.0, jet.radius, null, null, null),
      jet.x, jet.y, jet.vx, jet.vy, jet.heading, "JET");

    const tracks = Array.from(tracker.tracks.values()).sort((a, b) => a.tid - b.tid);
    const trackRows = [], trackTids = [];
    for (const t of tracks) {
      const i = add(keyTrack(t.tid), TRACK,
        (out, base) => nodeRow(out, base, TRACK, t.x, t.y, t.vx, t.vy, t.heading,
                               t.visible ? 1.0 : 0.0, t.age, t.coast, t.turnPerFrame(), 2.0,
                               t.last_known, t.speed, [jet.x, jet.y]),
        t.x, t.y, t.vx, t.vy, t.heading, "T" + t.tid);
      trackRows.push(i);
      trackTids.push(t.tid);
    }

    // Only flares near some track can matter; keep the closest ones.
    let useFlares = [];
    if (flares.length && tracks.length) {
      const scored = [];
      for (const f of flares) {
        let dmin = Infinity;
        for (const t of tracks) {
          const d = Math.sqrt((f.x - t.x) * (f.x - t.x) + (f.y - t.y) * (f.y - t.y));
          if (d < dmin) dmin = d;
        }
        if (dmin < DECOY_RANGE) scored.push([dmin, f]);
      }
      scored.sort((a, b) => a[0] - b[0]);
      useFlares = scored.slice(0, MAX_FLARE_NODES).map((p) => p[1]);
    }
    const flareRows = [];
    for (const f of useFlares) {
      const age = frame - (f.born_frame === null || f.born_frame === undefined ? frame : f.born_frame);
      flareRows.push(add(keyFlare(f.uid), FLARE,
        (out, base) => nodeRow(out, base, FLARE, f.x, f.y, f.vx, f.vy, Math.atan2(f.vy, f.vx), 1.0, age, 0, 0.0, f.radius, null, null, null),
        f.x, f.y, f.vx, f.vy, 0.0, "F"));
    }

    const obsRows = [];
    for (let k = 0; k < this.terrain.length; k++) {
      const o = this.terrain[k];
      obsRows.push(add(keyObstacle(k), OBSTACLE,
        (out, base) => nodeRow(out, base, OBSTACLE, o[0], o[1], 0.0, 0.0, 0.0, 1.0, 300, 0, 0.0, o[2], null, null, null),
        o[0], o[1], 0.0, 0.0, 0.0, "P" + (k + 1)));
    }

    const stRows = [];
    for (let k = 0; k < launchers.length; k++) {
      const l = launchers[k];
      stRows.push(add(keyStation(k), STATION,
        (out, base) => nodeRow(out, base, STATION, l.x, l.y, 0.0, 0.0, -Math.PI / 2, l.alive ? 1.0 : 0.0, 300, 0, 0.0, 15.0, null, null, null),
        l.x, l.y, 0.0, 0.0, -Math.PI / 2, "S" + (k + 1)));
    }

    // -- edges --------------------------------------------------
    const src = [], dst = [], et = [];
    const edge = (a, b, typ) => { src.push(a); dst.push(b); et.push(typ); };
    const terrain = this.terrain;

    for (let ti = 0; ti < tracks.length; ti++) {
      const t = tracks[ti], row = trackRows[ti];
      const d = Math.sqrt((t.x - jet.x) * (t.x - jet.x) + (t.y - jet.y) * (t.y - jet.y));
      if (d < THREAT_RADIUS) {
        edge(row, jetI, THREAT);
        edge(jetI, row, WATCH);
      }
      if (t.visible && t.seeker_edge) edge(row, jetI, SEEKER);
      if (terrain.length) {
        for (let k = 0; k < terrain.length; k++) {
          const o = terrain[k];
          if (Math.sqrt((t.x - o[0]) * (t.x - o[0]) + (t.y - o[1]) * (t.y - o[1])) - o[2] < TERRAIN_RADIUS) {
            edge(obsRows[k], row, TERRAIN);
          }
          if (segmentBlocked(t.x, t.y, jet.x, jet.y, [o])) edge(obsRows[k], row, OCCLUDE);
        }
      }
      if (t.age < LAUNCH_AGE) {
        for (let k = 0; k < launchers.length; k++) {
          const l = launchers[k];
          if (Math.sqrt((t.birth_x - l.x) * (t.birth_x - l.x) + (t.birth_y - l.y) * (t.birth_y - l.y)) < LAUNCH_SNAP) {
            edge(stRows[k], row, LAUNCH);
          }
        }
      }
    }

    const linkRows = [];
    for (let a = 0; a < tracks.length; a++) {
      const ta = tracks[a];
      for (let b = a + 1; b < tracks.length; b++) {
        const tb = tracks[b];
        const d = Math.sqrt((ta.x - tb.x) * (ta.x - tb.x) + (ta.y - tb.y) * (ta.y - tb.y));
        if (this.level.datalink && d < LINK_RANGE) {
          linkRows.push(src.length);
          edge(trackRows[a], trackRows[b], LINK);
          edge(trackRows[b], trackRows[a], LINK);
        }
        if (d < NEAR_RADIUS) {
          edge(trackRows[a], trackRows[b], NEAR);
          edge(trackRows[b], trackRows[a], NEAR);
        }
      }
    }

    for (let fi = 0; fi < useFlares.length; fi++) {
      const f = useFlares[fi];
      for (let ti = 0; ti < tracks.length; ti++) {
        const t = tracks[ti];
        const dx = f.x - t.x, dy = f.y - t.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < DECOY_RANGE && (dx * Math.cos(t.heading) + dy * Math.sin(t.heading) >= SEEKER_COS_FOV * d)) {
          if (!(terrain.length && segmentBlocked(t.x, t.y, f.x, f.y, terrain))) {
            edge(flareRows[fi], trackRows[ti], DECOY);
          }
        }
      }
    }

    if (terrain.length) {
      for (let k = 0; k < terrain.length; k++) {
        const o = terrain[k];
        if (Math.sqrt((jet.x - o[0]) * (jet.x - o[0]) + (jet.y - o[1]) * (jet.y - o[1])) - o[2] < TERRAIN_RADIUS) {
          edge(obsRows[k], jetI, TERRAIN);
        }
      }
    }

    // -- pack ---------------------------------------------------
    const N = keys.length, E = src.length;
    const x = new Float32Array(N * NODE_DIM);
    for (let i = 0; i < N; i++) rowsData[i](x, i * NODE_DIM);

    const eattr = new Float32Array(E * EDGE_DIM);
    for (let e = 0; e < E; e++) {
      const s = src[e] * 5, dd = dst[e] * 5;
      const sx = kin[s], sy = kin[s + 1], svx = kin[s + 2], svy = kin[s + 3];
      const dx_ = kin[dd], dy_ = kin[dd + 1], dvx = kin[dd + 2], dvy = kin[dd + 3], dh = kin[dd + 4];
      const rx = sx - dx_, ry = sy - dy_;
      const d = Math.sqrt(rx * rx + ry * ry);
      const rvx = svx - dvx, rvy = svy - dvy;
      const bearing = Math.atan2(ry, rx) - dh;
      let clear = 1.0;
      const ty = et[e];
      if (terrain.length && (ty === THREAT || ty === WATCH || ty === LINK || ty === NEAR)) {
        clear = segmentBlocked(sx, sy, dx_, dy_, terrain) ? 0.0 : 1.0;
      }
      const b = e * EDGE_DIM;
      eattr[b] = rx / 400.0;
      eattr[b + 1] = ry / 400.0;
      eattr[b + 2] = rvx / 10.0;
      eattr[b + 3] = rvy / 10.0;
      eattr[b + 4] = d / 400.0;
      eattr[b + 5] = -(rx * rvx + ry * rvy) / (d + 1e-6) / 10.0;
      eattr[b + 6] = Math.cos(bearing);
      eattr[b + 7] = Math.sin(bearing);
      eattr[b + 8] = clear;
    }

    return {
      frame: frame, keys: keys, ntype: ntype, x: x, pos: pos,
      src: Int32Array.from(src), dst: Int32Array.from(dst), etype: Int32Array.from(et),
      eattr: eattr, track_rows: trackRows, track_tids: trackTids,
      link_rows: linkRows, names: names, n: N, e: E,
    };
  }
}

// Edge insertions and deletions between two snapshots, for the HUD ticker.
export function diff(prev, cur) {
  if (!prev) return {};
  const setOf = (g) => {
    const s = new Set();
    for (let e = 0; e < g.e; e++) s.add(g.etype[e] + ":" + g.keys[g.src[e]] + ":" + g.keys[g.dst[e]]);
    return s;
  };
  const a = setOf(prev), b = setOf(cur);
  const out = {};
  const bump = (key, slot) => {
    const parts = key.split(":");
    const typ = parseInt(parts[0], 10);
    if (!out[typ]) out[typ] = [[], []];
    out[typ][slot].push([parseInt(parts[1], 10), parseInt(parts[2], 10)]);
  };
  for (const k of b) if (!a.has(k)) bump(k, 0);
  for (const k of a) if (!b.has(k)) bump(k, 1);
  return out;
}
