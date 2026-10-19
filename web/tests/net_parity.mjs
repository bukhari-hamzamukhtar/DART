// Does the JS network reproduce the torch model on real scene graphs?
//
//   node tests/net_parity.mjs [parity.json] [f16|f32]
//
// parity.json defaults to tests/parity.json, written by tools/dump_parity.py.
//
// The dump holds a sequence of real level-3 scene-graph ticks and, for each,
// what the trained torch model estimated for every track and datalink. The JS
// network replays the same sequence with its own per-node memory.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { TemporalGraphNet, GraphBrain, halfToFloat } from "../js/net.js";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dumpPath = process.argv[2] || path.join(WEB, "tests", "parity.json");

// Which weight file: the float32 reference, or the float16 chunks the web build ships.
const which = process.argv[3] || "f16";
let manifest, data;
if (which === "f32") {
  manifest = JSON.parse(fs.readFileSync(path.join(WEB, "assets/brain_full.json"), "utf8"));
  const raw = fs.readFileSync(path.join(WEB, "assets/brain_full.bin"));
  data = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
} else {
  manifest = JSON.parse(fs.readFileSync(path.join(WEB, "assets/brain16.json"), "utf8"));
  const bytes = Buffer.from(fs.readFileSync(path.join(WEB, "assets", manifest.b64), "utf8").trim(), "base64");
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  data = halfToFloat(u16);
}
console.log("weights:", which);
const model = new TemporalGraphNet(manifest, data);
console.log("model loaded:", model.params, "parameters, config", JSON.stringify(model.config));

const dump = JSON.parse(fs.readFileSync(dumpPath, "utf8"));
const brain = new GraphBrain(model);

const KEYS = ["pursuit", "turn_rate", "lead", "side", "guided", "threat", "turn_hi", "bias", "flank", "wobble"];
const worst = {};
for (const k of KEYS) worst[k] = 0;
let worstLink = 0;

for (const t of dump.ticks) {
  const N = t.keys.length, E = t.src.length;
  const x = new Float32Array(N * 23);
  for (let i = 0; i < N; i++) for (let j = 0; j < 23; j++) x[i * 23 + j] = t.x[i][j];
  const eattr = new Float32Array(E * 9);
  for (let i = 0; i < E; i++) for (let j = 0; j < 9; j++) eattr[i * 9 + j] = t.eattr[i][j];
  const g = {
    frame: t.frame, keys: t.keys, n: N, e: E, x: x, eattr: eattr,
    src: Int32Array.from(t.src), dst: Int32Array.from(t.dst), etype: Int32Array.from(t.etype),
    track_rows: t.track_rows, track_tids: t.track_tids, link_rows: t.link_rows,
    names: t.keys.map(String),
  };
  const out = brain.step(g);
  let tickWorst = 0, tickKey = "";
  for (let i = 0; i < out.rows.length; i++) {
    for (const k of KEYS) {
      const d = Math.abs(out.rows[i][k] - t.rows[i][k]);
      if (d > worst[k]) worst[k] = d;
      if (d > tickWorst) { tickWorst = d; tickKey = k; }
    }
  }
  console.log("   worst this tick: " + tickWorst.toExponential(2) + " (" + tickKey + ")");
  for (let i = 0; i < out.links.length; i++) {
    const d = Math.abs(out.links[i][2] - t.links[i][2]);
    if (d > worstLink) worstLink = d;
    if (out.links[i][0] !== t.links[i][0] || out.links[i][1] !== t.links[i][1]) {
      console.log("FAIL link pair mismatch", out.links[i], t.links[i]);
      process.exit(1);
    }
  }
  console.log("tick f" + t.frame + ": " + N + " nodes, " + E + " edges, "
              + out.rows.length + " tracks, " + out.links.length + " links");
}

console.log("\nmax |JS - torch| per estimate:");
for (const k of KEYS) console.log("  " + k.padEnd(10) + " " + worst[k].toExponential(2));
console.log("  " + "link_prob".padEnd(10) + " " + worstLink.toExponential(2));

// float32 should be exact to float noise; float16 weights cost ~1e-3 on a probability.
const tol = which === "f32" ? 2e-4 : 2e-3;
const bad = KEYS.filter((k) => worst[k] > tol);
if (worstLink > tol) bad.push("link_prob");
console.log(bad.length
  ? "\nFAIL: " + bad.join(", ") + " exceed " + tol
  : "\nPASS: the JS network matches the torch model (float32 tolerance " + tol + ")");
process.exit(bad.length ? 1 : 0);
