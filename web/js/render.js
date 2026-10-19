// Canvas rendering: the arena, the HUD and the dynamic-graph overlay.
import {
  WIDTH, HEIGHT, HUD_HEIGHT, FIELD_H, BG_COLOR, JET_COLOR, MISSILE_COLOR,
  FLARE_COLOR, BALLISTIC_COLOR, TERRAIN_FILL, TERRAIN_CONTOUR,
} from "./config.js";
import { SMART_MODE_COLORS } from "./entities.js";
import { SEEKER_RANGE } from "./doctrine.js";
import { EDGE_TYPES, THREAT, WATCH, SEEKER, LINK, NEAR, DECOY, TERRAIN, OCCLUDE, LAUNCH } from "./sceneGraph.js";

const EDGE_STYLE = {
  [THREAT]: ["rgba(255,80,80,0.55)", 1.4],
  [WATCH]: ["rgba(0,255,200,0.18)", 1],
  [SEEKER]: ["rgba(255,200,60,0.75)", 1.6],
  [LINK]: ["rgba(90,200,255,0.55)", 1.4],
  [NEAR]: ["rgba(160,160,200,0.22)", 1],
  [DECOY]: ["rgba(255,150,40,0.7)", 1.4],
  [TERRAIN]: ["rgba(90,200,140,0.25)", 1],
  [OCCLUDE]: ["rgba(140,120,200,0.45)", 1.2],
  [LAUNCH]: ["rgba(255,120,255,0.4)", 1.2],
};

function trailStroke(ctx, trail, color, width) {
  const n = trail.length / 2;
  if (n < 2) return;
  for (let i = 1; i < n; i++) {
    const a = (i / n) * 0.75;
    ctx.strokeStyle = color.replace("ALPHA", a.toFixed(3));
    ctx.lineWidth = Math.max(0.6, (i / n) * width);
    ctx.beginPath();
    ctx.moveTo(trail[(i - 1) * 2], trail[(i - 1) * 2 + 1]);
    ctx.lineTo(trail[i * 2], trail[i * 2 + 1]);
    ctx.stroke();
  }
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.showOverlay = false;
  }

  draw(arena, pilot, hud) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    this._grid(ctx);
    this._terrainShadows(ctx, arena);
    this._terrain(ctx, arena);
    this._launchers(ctx, arena);
    if (this.showOverlay) this._graph(ctx, pilot);
    this._flares(ctx, arena);
    this._jetMissiles(ctx, arena);
    this._missiles(ctx, arena);
    this._jet(ctx, arena);
    this._blasts(ctx, arena);
    this._hud(ctx, arena, pilot, hud);
    ctx.restore();
  }

  _grid(ctx) {
    ctx.strokeStyle = "rgba(0,255,200,0.045)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= WIDTH; x += 50) { ctx.moveTo(x, 0); ctx.lineTo(x, FIELD_H); }
    for (let y = 0; y <= FIELD_H; y += 50) { ctx.moveTo(0, y); ctx.lineTo(WIDTH, y); }
    ctx.stroke();
  }

  // Terrain blocks the jet radar: shade what it cannot see.
  _terrainShadows(ctx, arena) {
    if (!arena.terrain.length) return;
    const jx = arena.jet.x, jy = arena.jet.y;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, WIDTH, FIELD_H);
    ctx.clip();
    ctx.fillStyle = "rgba(0,0,0,0.42)";
    for (const [ox, oy, r] of arena.terrain) {
      const dx = ox - jx, dy = oy - jy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 1e-6) continue;
      const a = Math.asin(Math.min(1, r / Math.max(d, r + 0.001)));
      const base = Math.atan2(dy, dx);
      const far = 1400;
      const a1 = base - a, a2 = base + a;
      ctx.beginPath();
      ctx.moveTo(jx + Math.cos(a1) * d, jy + Math.sin(a1) * d);
      ctx.lineTo(jx + Math.cos(a1) * far, jy + Math.sin(a1) * far);
      ctx.lineTo(jx + Math.cos(a2) * far, jy + Math.sin(a2) * far);
      ctx.lineTo(jx + Math.cos(a2) * d, jy + Math.sin(a2) * d);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  _terrain(ctx, arena) {
    for (const [ox, oy, r] of arena.terrain) {
      ctx.fillStyle = TERRAIN_FILL;
      ctx.beginPath();
      ctx.arc(ox, oy, r, 0, Math.PI * 2);
      ctx.fill();
      let k = 0;
      for (const frac of [1.0, 0.74, 0.5, 0.28]) {
        ctx.strokeStyle = `rgba(70,150,110,${0.45 + 0.17 * k})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(ox, oy, Math.max(2, r * frac), 0, Math.PI * 2);
        ctx.stroke();
        k += 1;
      }
      ctx.fillStyle = "#78be96";
      ctx.beginPath();
      ctx.arc(ox, oy, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _launchers(ctx, arena) {
    for (let i = 0; i < arena.launchers.length; i++) {
      const l = arena.launchers[i];
      const st = arena.level.stations[i];
      if (!l.alive) {
        ctx.strokeStyle = "#282828";
        ctx.lineWidth = 2;
        ctx.strokeRect(l.x - 15, l.y - 10, 30, 20);
        ctx.strokeStyle = "#962828";
        ctx.beginPath();
        ctx.moveTo(l.x - 10, l.y - 15); ctx.lineTo(l.x + 10, l.y + 5);
        ctx.moveTo(l.x - 10, l.y + 5); ctx.lineTo(l.x + 10, l.y - 15);
        ctx.stroke();
        continue;
      }
      const ready = arena.stationReady(i);
      ctx.strokeStyle = l.color;
      ctx.lineWidth = 2;
      ctx.strokeRect(l.x - 15, l.y - 10, 30, 20);
      ctx.fillStyle = ready ? l.color : "rgba(120,120,120,0.6)";
      ctx.beginPath();
      ctx.arc(l.x, l.y, 5, 0, Math.PI * 2);
      ctx.fill();
      // key caps under each battery
      ctx.font = "600 11px ui-monospace, monospace";
      ctx.textAlign = "center";
      const keys = st.cluster_key ? st.key + " / " + st.cluster_key : st.key;
      ctx.fillStyle = ready ? "rgba(230,240,255,0.95)" : "rgba(150,160,180,0.7)";
      ctx.fillText(keys, l.x, l.y + 24);
      ctx.fillStyle = "rgba(150,160,180,0.65)";
      ctx.fillText(st.label, l.x, l.y - 16);
      if (arena.clusterReady(i)) {
        ctx.strokeStyle = "rgba(255,120,120,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(l.x, l.y, 11, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  _graph(ctx, pilot) {
    const g = pilot && pilot.graph;
    if (!g) return;
    for (let e = 0; e < g.e; e++) {
      const st = EDGE_STYLE[g.etype[e]];
      if (!st) continue;
      const s = g.src[e] * 2, d = g.dst[e] * 2;
      ctx.strokeStyle = st[0];
      ctx.lineWidth = st[1];
      ctx.beginPath();
      ctx.moveTo(g.pos[s], g.pos[s + 1]);
      ctx.lineTo(g.pos[d], g.pos[d + 1]);
      ctx.stroke();
    }
    for (let i = 0; i < g.n; i++) {
      ctx.strokeStyle = "rgba(200,230,255,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(g.pos[i * 2], g.pos[i * 2 + 1], 7, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  _flares(ctx, arena) {
    for (const f of arena.flares) {
      const age = f.born_frame === null ? 0 : arena.frame - f.born_frame;
      const fade = Math.max(0.15, 1 - age / 120);
      ctx.fillStyle = `rgba(255,176,32,${fade.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(255,230,160,${(fade * 0.25).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.radius + 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _jetMissiles(ctx, arena) {
    for (const m of arena.ai_missiles) {
      trailStroke(ctx, m.trail, "rgba(0,255,200,ALPHA)", 2.5);
      ctx.fillStyle = JET_COLOR;
      ctx.beginPath();
      ctx.arc(m.x, m.y, m.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _missiles(ctx, arena) {
    for (const m of arena.missiles) {
      let col = MISSILE_COLOR;
      if (m.kind === "ballistic") col = BALLISTIC_COLOR;
      else if (m.kind === "smart") col = SMART_MODE_COLORS[m.mode] || MISSILE_COLOR;
      const rgb = col.startsWith("#")
        ? [parseInt(col.slice(1, 3), 16), parseInt(col.slice(3, 5), 16), parseInt(col.slice(5, 7), 16)]
        : [255, 68, 68];
      trailStroke(ctx, m.trail, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},ALPHA)`, 2.2);
      if (m.spawn_delay > 0) continue;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(m.x, m.y, m.radius + (m.kind === "smart" ? 1 : 0), 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.18)`;
      ctx.beginPath();
      ctx.arc(m.x, m.y, 7, 0, Math.PI * 2);
      ctx.fill();
      if (m.kind === "smart" && (m.mode === "LOCK" || m.mode === "LINK")) {
        // a hint of the seeker cone, so a lock is visible
        ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.13)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.x, m.y);
        ctx.arc(m.x, m.y, 26, m.heading - 1.22, m.heading + 1.22);
        ctx.closePath();
        ctx.stroke();
      }
    }
  }

  _jet(ctx, arena) {
    const jet = arena.jet;
    trailStroke(ctx, jet.trail, "rgba(0,255,200,ALPHA)", 3.2);
    const R = jet.radius;
    const ch = Math.cos(jet.heading), sh = Math.sin(jet.heading);
    const rot = (px, py) => [jet.x + px * ch - py * sh, jet.y + px * sh + py * ch];
    const pts = [rot(R * 2.2, 0), rot(-R, R * 1.2), rot(-R * 0.2, 0), rot(-R, -R * 1.2)];
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fillStyle = "rgba(0,255,200,0.35)";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.stroke();
    const tail = rot(-R * 0.9, 0);
    ctx.fillStyle = "rgba(255,255,200,0.5)";
    ctx.beginPath();
    ctx.arc(tail[0], tail[1], 5, 0, Math.PI * 2);
    ctx.fill();
  }

  _blasts(ctx, arena) {
    for (const b of arena.blasts) {
      const age = arena.frame - b.born;
      if (age > 18) continue;
      const t = age / 18;
      ctx.strokeStyle = b.color;
      ctx.globalAlpha = Math.max(0, 1 - t);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(b.x, b.y, 8 + t * 46, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  _hud(ctx, arena, pilot, hud) {
    const y0 = FIELD_H;
    ctx.fillStyle = "rgba(6,12,22,0.95)";
    ctx.fillRect(0, y0, WIDTH, HUD_HEIGHT);
    ctx.strokeStyle = "rgba(0,255,200,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y0 + 0.5); ctx.lineTo(WIDTH, y0 + 0.5);
    ctx.stroke();

    ctx.font = "600 12px ui-monospace, monospace";
    ctx.textAlign = "left";
    const secs = (arena.frame / 60).toFixed(1);
    ctx.fillStyle = "rgba(200,220,240,0.9)";
    ctx.fillText("TIME " + secs + "s", 12, y0 + 20);
    ctx.fillText("MISSILES " + arena.missiles.length, 12, y0 + 38);

    ctx.fillStyle = "rgba(255,176,32,0.95)";
    ctx.fillText("MAGAZINE", 140, y0 + 20);
    for (let i = 0; i < 8; i++) {
      const on = i < arena.flare_charges;
      ctx.fillStyle = on ? FLARE_COLOR : "rgba(120,110,90,0.35)";
      ctx.fillRect(140 + i * 14, y0 + 28, 10, 10);
    }
    ctx.fillStyle = "rgba(150,160,180,0.8)";
    ctx.fillText("+1 per cluster", 258, y0 + 38);

    // what the pilot is doing
    const p = pilot && pilot.plan;
    ctx.fillStyle = "rgba(200,220,240,0.9)";
    ctx.fillText("PILOT " + (hud.pilotLabel || "V3"), 258, y0 + 20);
    if (p) {
      const danger = Math.min(1, Math.max(0, p.danger));
      ctx.fillStyle = "rgba(150,160,180,0.8)";
      ctx.fillText("PLAN " + p.label, 400, y0 + 20);
      ctx.fillText("GAP " + p.min_gap.toFixed(0) + "px", 400, y0 + 38);
      ctx.fillStyle = "rgba(120,130,150,0.6)";
      ctx.fillRect(500, y0 + 12, 120, 8);
      ctx.fillStyle = danger > 0.66 ? "#ff5a5a" : (danger > 0.33 ? "#ffb020" : "#00ffc8");
      ctx.fillRect(500, y0 + 12, 120 * danger, 8);
      ctx.fillStyle = "rgba(150,160,180,0.8)";
      ctx.fillText("DANGER", 500, y0 + 38);
    }
    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(150,160,180,0.75)";
    ctx.fillText(hud.fps.toFixed(0) + " fps   " + hud.frameMs.toFixed(1) + " ms", WIDTH - 12, y0 + 20);
    ctx.fillText(hud.hint || "", WIDTH - 12, y0 + 38);
    ctx.textAlign = "left";
  }
}
