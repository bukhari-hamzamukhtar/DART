// pilots/graph_pilot.py -- the dynamic-graph pilot, one frame at a time:
//
//   radar      what the jet can see (terrain blocks it)
//   tracker    persistent tracks, coasting behind terrain, last-known points
//   graph      every GRAPH_TICK frames, a scene-graph snapshot
//   brain      the temporal GNN re-reads every track's hidden guidance state
//   planner    every frame, every candidate action rolled 72 frames forward
//   controls   the chosen heading and throttle, applied exactly as imagined
import {
  GRAPH_TICK, FLARE_LIFE_FRAMES, STEER_GAIN, normalizeAngle,
  MIN_INTERCEPT_FRAMES, MAX_INTERCEPT_FRAMES, FLARE_LEAD_BOOST,
  MIN_FLARE_SPEED, MAX_FLARE_SPEED,
} from "./config.js";
import { FlareParticle, predictInterceptFlare } from "./entities.js";
import { Tracker, PRIOR } from "./tracker.js";
import { SceneGraphBuilder, diff, EDGE_TYPES, SEEKER, OCCLUDE, DECOY } from "./sceneGraph.js";
import { GraphBrain } from "./net.js";
import { PlannerV3, THROTTLES_FULL, THROTTLES_FAST } from "./planner.js";
import { oracleThreat, flareState } from "./worldmodel.js";

export const FLARE_REARM_FRAMES = 30;    // so two charges are never dumped at once
export const FLARE_SPREAD = [0.0, -0.20, 0.20];
export const VOLLEY_TARGETS = 6;         // a volley answers the nearest few missiles

export class GraphPilot {
  // mode: "trained" (the GNN), "prior" (no GNN, tracker priors only),
  //       "oracle" (every missile's true values -- the ceiling, not a mode to play)
  // quality: "full" plans over 58 candidate manoeuvres every frame, as the
  // desktop build does; "fast" trims the throttle set to three; "lite" also
  // plans every second frame and flies the held plan in between.
  constructor(level, model, mode, quality) {
    this.level = level;
    this.mode = mode || "trained";
    this.model = this.mode === "trained" ? model : null;
    this.quality = quality || "full";
    this.showOverlay = false;
  }

  start(arena) {
    this.tracker = new Tracker(this.level);
    this.builder = new SceneGraphBuilder(this.level);
    this.brain = this.model ? new GraphBrain(this.model) : null;
    this.planner = new PlannerV3({
      throttles: this.quality === "full" ? THROTTLES_FULL : THROTTLES_FAST,
    });
    this.planEvery = this.quality === "lite" ? 2 : 1;
    this.hold = null;
    this.graph = null;
    this.gates = null;
    this.links = [];
    this.plan = null;
    this.planThreats = [];
    this.maneuverLabel = "NORMAL";
    this.ticker = [];
    this.edgeDelta = [0, 0];
    this.timing = { graph: 0.0, gnn: 0.0, plan: 0.0 };
    this._seenEvents = 0;
    this.lastFlareFrame = -1e6;
    this._charges = arena.flare_charges;
  }

  stop() {}

  _log(line) {
    this.ticker.push(line);
    if (this.ticker.length > 7) this.ticker.shift();
  }

  act(arena) {
    const jet = arena.jet;
    if (arena.flare_charges > this._charges) {
      this._log("f" + arena.frame + "  MAGAZINE   +" + (arena.flare_charges - this._charges)
                + " volley (cluster launched), " + arena.flare_charges + " held");
    }
    this._charges = arena.flare_charges;

    const [visible, visFlares] = arena.radarPicture();
    this.tracker.update(arena.frame, jet, visible);

    if (arena.frame % GRAPH_TICK === 0) {
      const t0 = performance.now();
      const g = this.builder.build(arena.frame, jet, this.tracker, visFlares, arena.launchers);
      const t1 = performance.now();
      if (this.brain) {
        const out = this.brain.step(g);
        this.links = out.links;
        this.gates = out.gates;
        this.tracker.applyEstimates(g.track_tids, out.rows, out.links);
      }
      const t2 = performance.now();
      this._logEvents(arena.frame, this.graph, g);
      this.graph = g;
      this.timing.graph = t1 - t0;
      this.timing.gnn = t2 - t1;
    }

    let threats, inAir;
    if (this.mode === "oracle") {
      // Omniscient: every missile and flare in the arena with its true state.
      threats = arena.missiles.map((m) => oracleThreat(m, this.level.datalink, arena.frame));
      inAir = arena.flares;
    } else {
      threats = this.tracker.threatDicts(arena.frame);
      inAir = visFlares;
    }
    const flares = inAir.filter((f) => !f.dead).map((f) => flareState(f, arena.frame, FLARE_LIFE_FRAMES));

    // The volley the jet could fire this frame, if the magazine holds one.
    let shots = null, volley = null;
    if (visible.length && arena.canFlare()
        && arena.frame - this.lastFlareFrame >= FLARE_REARM_FRAMES) {
      shots = this._volley(jet, visible);
      volley = shots.map(([a, s]) => ({
        pos: [jet.x, jet.y], vel: [Math.cos(a) * s, Math.sin(a) * s], ttl: FLARE_LIFE_FRAMES,
      }));
    }

    // On "lite" the jet re-plans every second frame and flies the plan it is
    // holding in between -- the same candidate the planner would offer itself.
    const skip = this.planEvery > 1 && this.plan && (arena.frame % this.planEvery) !== 0;
    let p = this.plan;
    if (!skip) {
      const jd = { pos: [jet.x, jet.y], vel: [jet.vx, jet.vy], heading: jet.heading, speed: jet.speed };
      const t3 = performance.now();
      p = this.planner.plan(jd, threats, this.level, {
        flares: flares, volley: volley, hold: this.hold, keepFuture: this.showOverlay,
      });
      this.timing.plan = performance.now() - t3;
      this.plan = p;
      this.planThreats = threats;
    }

    if (p.label !== this.maneuverLabel && p.label !== "NORMAL") arena.stats.maneuvers += 1;
    this.maneuverLabel = p.label;
    this.hold = [p.heading, p.throttle, p.label];

    // Countermeasures are the world model's call: plan_v3 imagined every action
    // with and without this volley. The arena takes the charge.
    if (!skip && p.flare && arena.spendFlare()) {
      this.lastFlareFrame = arena.frame;
      for (const [a, s] of shots) arena.flares.push(new FlareParticle(jet.x, jet.y, a, s));
      arena.stats.flares += shots.length;
      this._charges = arena.flare_charges;
      this._log("f" + arena.frame + "  VOLLEY     " + shots.length + " flares, future "
                + p.flare_gain.toFixed(2) + " cheaper, " + arena.flare_charges + " left");
    }

    // Fly exactly the control the world model imagined.
    jet.heading += normalizeAngle(p.heading - jet.heading) * STEER_GAIN;
    jet.setThrottle(p.throttle);
  }

  // (angle, speed) of each flare in a volley thrown across the nearest missiles.
  _volley(jet, visible) {
    const near = visible.slice().sort((a, b) =>
      ((a.x - jet.x) ** 2 + (a.y - jet.y) ** 2) - ((b.x - jet.x) ** 2 + (b.y - jet.y) ** 2)
    ).slice(0, VOLLEY_TARGETS);
    const shots = [];
    for (const m of near) {
      const [angle, speed] = predictInterceptFlare(jet, m, MIN_INTERCEPT_FRAMES,
        MAX_INTERCEPT_FRAMES, FLARE_LEAD_BOOST, MIN_FLARE_SPEED, MAX_FLARE_SPEED);
      for (let k = 0; k < FLARE_SPREAD.length; k++) {
        shots.push([angle + FLARE_SPREAD[k], speed * (k === 0 ? 1.0 : 0.85)]);
      }
    }
    return shots;
  }

  // The event stream, for the HUD ticker.
  _logEvents(frame, prev, cur) {
    const fresh = Math.min(this.tracker.event_count - this._seenEvents, this.tracker.events.length);
    if (fresh > 0) {
      for (const [f, kind, tid, detail] of this.tracker.events.slice(-fresh)) {
        this._log("f" + f + "  " + kind.toUpperCase() + " T" + tid + " " + detail);
      }
    }
    this._seenEvents = this.tracker.event_count;
    const delta = diff(prev, cur);
    let added = 0, removed = 0;
    for (const v of Object.values(delta)) { added += v[0].length; removed += v[1].length; }
    this.edgeDelta = [added, removed];
    const names = new Map();
    if (prev) for (let i = 0; i < prev.keys.length; i++) names.set(prev.keys[i], prev.names[i]);
    for (let i = 0; i < cur.keys.length; i++) names.set(cur.keys[i], cur.names[i]);
    for (const typ of [SEEKER, OCCLUDE, DECOY]) {
      const d = delta[typ];
      if (!d) continue;
      for (const [s, dd] of d[0]) {
        this._log("f" + frame + "  +" + EDGE_TYPES[typ] + " " + (names.get(s) || "?") + "->" + (names.get(dd) || "?"));
      }
    }
    const sk = delta[SEEKER];
    if (sk) {
      for (const [s, dd] of sk[1]) {
        this._log("f" + frame + "  -SEEKER " + (names.get(s) || "?") + "->" + (names.get(dd) || "?"));
      }
    }
  }
}
