// Screens, input and the frame-locked loop.
import { LEVELS } from "./levels.js";
import { Arena } from "./arena.js";
import { resetUids } from "./entities.js";
import { GraphPilot } from "./pilot.js";
import { Renderer } from "./render.js";
import { loadBrain } from "./net.js";

const PILOTS = [
  { id: "trained", label: "V3 trained",
    note: "The temporal graph network infers every missile's hidden guidance from radar tracks alone, then the world model plans against its estimates. This is the real build." },
  { id: "oracle", label: "Oracle (cheats)",
    note: "The jet is handed every missile's true turn rate, lead, flight style and datalink. No inference, just planning: the ceiling any brain could reach." },
  { id: "prior", label: "No network",
    note: "The world model runs on fixed prior guesses instead of the network. Shows what the graph network is actually worth: it dies early and often." },
];

// How much of the planner this device can afford at 60 fps. Every rule stays
// the same; only the number of candidate manoeuvres (and on "lite" how often
// the jet re-plans) changes.
const QUALITIES = [
  { id: "full", label: "Full", note: "58 candidate manoeuvres every frame, exactly as the desktop build plans." },
  { id: "fast", label: "Fast", note: "46 candidate manoeuvres every frame: three throttle settings instead of four." },
  { id: "lite", label: "Lite", note: "46 candidates, re-planned every second frame. For phones and slow machines." },
];

const el = (id) => document.getElementById(id);
const canvas = el("game");
const renderer = new Renderer(canvas);

const state = {
  levelIdx: 0,
  pilotId: "trained",
  quality: "full",
  model: null,
  arena: null,
  pilot: null,
  running: false,
  paused: false,
  screen: "menu",
  acc: 0,
  last: 0,
  fps: 60,
  frameMs: 0,
  queue: [],
};

// ---- screens -------------------------------------------------
function show(name) {
  state.screen = name;
  for (const id of ["menu", "briefing", "over"]) el(id).classList.toggle("show", id === name);
  el("paused").classList.remove("show");
  state.running = name === "match";
  state.paused = false;
  if (name === "match") state.last = performance.now();
  syncTouch();
}

function buildMenu() {
  const lo = el("levelOpts");
  lo.innerHTML = "";
  LEVELS.forEach((lv, i) => {
    const b = document.createElement("button");
    b.textContent = lv.number + " · " + lv.name;
    b.className = i === state.levelIdx ? "on" : "";
    b.onclick = () => { state.levelIdx = i; buildMenu(); };
    lo.appendChild(b);
  });
  const po = el("pilotOpts");
  po.innerHTML = "";
  for (const p of PILOTS) {
    const b = document.createElement("button");
    b.textContent = p.label;
    b.className = p.id === state.pilotId ? "on" : "";
    b.onclick = () => { state.pilotId = p.id; buildMenu(); };
    po.appendChild(b);
  }
  const qo = el("qualityOpts");
  qo.innerHTML = "";
  for (const q of QUALITIES) {
    const b = document.createElement("button");
    b.textContent = q.label;
    b.className = q.id === state.quality ? "on" : "";
    b.onclick = () => { state.quality = q.id; buildMenu(); };
    qo.appendChild(b);
  }
  const pilotNote = PILOTS.find((p) => p.id === state.pilotId).note;
  const qNote = QUALITIES.find((q) => q.id === state.quality).note;
  el("brainNote").innerHTML = pilotNote + "<br><span style='opacity:.75'>" + qNote + "</span>";
}

function buildBriefing() {
  const lv = LEVELS[state.levelIdx];
  el("brName").textContent = "LEVEL " + lv.number + " — " + lv.name;
  el("brTag").textContent = lv.tagline;
  const keys = el("brKeys");
  keys.innerHTML = "";
  for (const [k, what] of lv.legend) {
    const li = document.createElement("li");
    li.innerHTML = "<kbd>" + k + "</kbd> " + what;
    keys.appendChild(li);
  }
  const blurb = el("brBlurb");
  blurb.innerHTML = "";
  for (const line of lv.blurb) {
    const li = document.createElement("li");
    li.textContent = line;
    blurb.appendChild(li);
  }
  const extra = [];
  if (lv.obstacles.length) extra.push("Peaks are lethal to missiles and block radar");
  if (lv.datalink) extra.push("Salvo members share one lock over a datalink");
  if (lv.lead) extra.push("Missiles lead the jet instead of chasing it");
  for (const line of extra) {
    const li = document.createElement("li");
    li.textContent = line;
    blurb.appendChild(li);
  }
}

// ---- match ---------------------------------------------------
function startMatch() {
  const lv = LEVELS[state.levelIdx];
  resetUids();
  state.arena = new Arena(lv, { seed: (Math.random() * 2 ** 31) | 0 });
  state.pilot = new GraphPilot(lv, state.model, state.pilotId, state.quality);
  state.pilot.showOverlay = renderer.showOverlay;
  state.pilot.start(state.arena);
  state.queue = [];
  state.acc = 0;
  buildTouch();
  show("match");
}

// Every launch key for this level: what it fires, and whether it is ready.
function launchKeys() {
  const lv = LEVELS[state.levelIdx];
  const out = [];
  lv.stations.forEach((st, i) => {
    out.push({ key: st.key, label: st.label, kind: "fire", station: i });
    if (st.cluster_key) {
      out.push({ key: st.cluster_key, label: "CLUSTER", kind: "cluster", station: i });
    }
  });
  if (lv.pincer) out.push({ key: "S", label: "PINCER", kind: "pincer", station: -1 });
  return out;
}

function fireAction(a) {
  const arena = state.arena;
  if (!arena || arena.game_over) return;
  if (a.kind === "fire") arena.fire(a.station);
  else if (a.kind === "cluster") arena.launchCluster(a.station);
  else if (a.kind === "pincer") arena.launchPincer();
}

function applyQueue(arena) {
  while (state.queue.length) fireAction(state.queue.shift());
}

function endMatch() {
  const arena = state.arena;
  const won = arena.user_won;          // the jet died => the player won
  el("ovTitle").textContent = won ? "JET DOWN" : "BATTERIES LOST";
  const causes = {
    missile: "One of your missiles hit it.",
    wall: "You herded it into a wall.",
    terrain: "You herded it into a peak.",
    bases: "Its counter-attacks destroyed every battery you had.",
  };
  el("ovSub").textContent = causes[arena.cause] || "";
  const s = arena.stats;
  const stats = [
    ["TIME ALIVE", (arena.frame / 60).toFixed(1) + "s"],
    ["MISSILES FIRED", s.fired],
    ["FLARES SPENT", s.flares],
    ["VOLLEYS", s.volleys],
    ["CHARGES EARNED", s.charges_earned],
    ["FLARES DENIED", s.flares_denied],
  ];
  const grid = el("ovStats");
  grid.innerHTML = "";
  for (const [k, v] of stats) {
    const d = document.createElement("div");
    d.className = "stat";
    d.innerHTML = "<div class='v'>" + v + "</div><div class='k'>" + k + "</div>";
    grid.appendChild(d);
  }
  show("over");
}

// ---- loop ----------------------------------------------------
const STEP_MS = 1000 / 60;

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(now - state.last, 250);
  state.last = now;
  state.fps = state.fps * 0.9 + (1000 / Math.max(dt, 1)) * 0.1;

  if (state.running && !state.paused && state.arena) {
    const t0 = performance.now();
    state.acc += dt;
    // Frame-locked physics: a slow device runs the match slower, never coarser.
    let steps = 0;
    while (state.acc >= STEP_MS && steps < 3) {
      state.acc -= STEP_MS;
      steps += 1;
      state.arena.step(state.pilot, applyQueue);
      if (state.arena.game_over) break;
    }
    state.frameMs = state.frameMs * 0.8 + (performance.now() - t0) * 0.2;
    if (state.arena.game_over) {
      drawMatch();
      endMatch();
      return;
    }
    syncTouch();
  }
  if (state.arena) drawMatch();
}

function drawMatch() {
  const pilot = state.pilot;
  renderer.draw(state.arena, pilot, {
    fps: state.fps,
    frameMs: state.frameMs,
    pilotLabel: PILOTS.find((p) => p.id === state.pilotId).label.toUpperCase(),
    hint: (renderer.showOverlay ? "GRAPH ON  " : "") + (state.quality === "full" ? "" : state.quality.toUpperCase()),
  });
  if (pilot && pilot.ticker.length) {
    el("ticker").textContent = pilot.ticker[pilot.ticker.length - 1];
  }
}

// ---- input ---------------------------------------------------
function keyDown(ev) {
  const k = (ev.key || "").toUpperCase();
  if (state.screen === "match") {
    if (k === "P") { state.paused = !state.paused; el("paused").classList.toggle("show", state.paused); return; }
    if (k === "ESCAPE") { show("menu"); return; }
    if (k === "G") {
      renderer.showOverlay = !renderer.showOverlay;
      if (state.pilot) state.pilot.showOverlay = renderer.showOverlay;
      return;
    }
    for (const a of launchKeys()) {
      if (a.key === k) { state.queue.push(a); ev.preventDefault(); return; }
    }
  } else if (k === "ENTER") {
    if (state.screen === "menu") { buildBriefing(); show("briefing"); }
    else if (state.screen === "briefing") startMatch();
    else if (state.screen === "over") startMatch();
  }
}

function buildTouch() {
  const wrap = el("touch");
  wrap.innerHTML = "";
  for (const a of launchKeys()) {
    const b = document.createElement("button");
    b.innerHTML = "<span>" + a.key + "</span>" + a.label;
    b.dataset.kind = a.kind;
    b.dataset.station = a.station;
    const press = (ev) => { ev.preventDefault(); state.queue.push(a); };
    b.addEventListener("touchstart", press, { passive: false });
    b.addEventListener("mousedown", press);
    wrap.appendChild(b);
  }
  el("keyHints").textContent = launchKeys().map((a) => a.key).join(" ");
}

// Dim a touch button while its battery is reloading.
function syncTouch() {
  const arena = state.arena;
  if (!arena) return;
  for (const b of el("touch").children) {
    const i = parseInt(b.dataset.station, 10);
    let ready = true;
    if (b.dataset.kind === "fire") ready = arena.stationReady(i);
    else if (b.dataset.kind === "cluster") ready = arena.clusterReady(i);
    else if (b.dataset.kind === "pincer") ready = arena.pincerReady();
    b.classList.toggle("cool", !ready);
  }
}

// ---- boot ----------------------------------------------------
el("startBtn").onclick = () => { buildBriefing(); show("briefing"); };
el("brBack").onclick = () => show("menu");
el("brGo").onclick = () => startMatch();
el("ovRetry").onclick = () => startMatch();
el("ovNext").onclick = () => {
  state.levelIdx = (state.levelIdx + 1) % LEVELS.length;
  buildMenu();
  buildBriefing();
  show("briefing");
};
el("ovMenu").onclick = () => { buildMenu(); show("menu"); };
window.addEventListener("keydown", keyDown);

// One rollout of a busy scene says what this device can afford.
async function pickQuality() {
  const { Rollout } = await import("./worldmodel.js");
  const { actionGrid, maneuverActions } = await import("./planner.js");
  const jd = { pos: [400, 300], vel: [3, 1], heading: 0.4, speed: 5 };
  const [gh, gt] = actionGrid(0.4);
  const headings = gh.slice(), throttles = gt.slice();
  for (const [h, t] of maneuverActions(jd, [])) { headings.push(h); throttles.push(t); }
  const threats = [];
  for (let i = 0; i < 8; i++) {
    threats.push({ pos: [100 + i * 30, 500], vel: [2, -4], heading: -1.1, speed: 5,
                   turn_rate: 0.12, radius: 2, side: 0, flank: 0.4, ttl: 400, seeker: true });
  }
  const roll = new Rollout();
  const lv = LEVELS[1];
  roll.run(jd, threats, headings, throttles, 72, lv.obstacles, false, [], false);
  const t0 = performance.now();
  for (let k = 0; k < 8; k++) roll.run(jd, threats, headings, throttles, 72, lv.obstacles, false, [], false);
  const ms = (performance.now() - t0) / 8;
  const touch = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  state.quality = ms < 7 && !touch ? "full" : (ms < 16 ? "fast" : "lite");
  console.log("planner benchmark " + ms.toFixed(1) + " ms -> quality " + state.quality);
  buildMenu();
}

buildMenu();
show("menu");
requestAnimationFrame(frame);
pickQuality();

loadBrain((window.__ASSET_BASE || "assets/") + "brain16").then((model) => {
  state.model = model;
  const btn = el("startBtn");
  btn.disabled = false;
  btn.textContent = "BRIEFING";
  console.log("brain online:", model.params, "parameters");
}).catch((err) => {
  console.error(err);
  const btn = el("startBtn");
  btn.disabled = false;
  btn.textContent = "BRIEFING (no brain file)";
  el("brainNote").textContent = "The trained weights failed to load, so the jet falls back to prior guesses.";
});
