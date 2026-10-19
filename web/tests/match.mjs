// Full matches in Node: the browser pilot against a maximum-rate opponent.
//
//   node tests/match.mjs [seeds] [frames]
//
// Mirrors training/benchmark.py: launchers cannot be destroyed, every battery
// fires the moment it is ready, clusters and pincers the moment they are off
// cooldown, and a run ends when the jet dies or after `frames` frames. The
// numbers are not seed-for-seed comparable with the Python benchmark (different
// random generator), but survival and frame cost should be in the same range.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Arena } from "../js/arena.js";
import { LEVELS } from "../js/levels.js";
import { resetUids } from "../js/entities.js";
import { TemporalGraphNet } from "../js/net.js";
import { GraphPilot } from "../js/pilot.js";
import { makeRng } from "../js/config.js";
import { insideAny } from "../js/geometry.js";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SEEDS = parseInt(process.argv[2] || "4", 10);
const FRAMES = parseInt(process.argv[3] || "1800", 10);

const manifest = JSON.parse(fs.readFileSync(path.join(WEB, "assets/brain_full.json"), "utf8"));
const raw = fs.readFileSync(path.join(WEB, "assets/brain_full.bin"));
const model = new TemporalGraphNet(manifest, new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4));

// Everything the enemy can fire, as fast as it is allowed to.
function maxRateCommander(rng, level) {
  const jitter = [];
  for (let i = 0; i < level.stations.length; i++) jitter.push(rng.randint(0, 25));
  return (arena) => {
    for (let i = 0; i < arena.launchers.length; i++) {
      if (arena.frame % 1 === 0 && arena.stationReady(i) && arena.frame > jitter[i]) arena.fire(i);
      if (arena.clusterReady(i)) arena.launchCluster(i);
    }
    if (arena.pincerReady()) arena.launchPincer();
  };
}

const modes = process.argv[4] ? [process.argv[4]] : ["trained", "prior", "oracle"];
const QUALITY = process.argv[5] || "full";
const table = [];

for (const mode of modes) {
  for (const level of LEVELS) {
    let alive = 0, frames = 0, ms = 0, causes = {}, flares = 0, volleys = 0;
    for (let seed = 0; seed < SEEDS; seed++) {
      resetUids();
      const rng = makeRng(seed * 7919 + level.number);
      const arena = new Arena(level, { launchers_invulnerable: true, seed: seed * 7919 + level.number });
      // Each seed is a different match: the jet starts somewhere else, pointing
      // somewhere else, and the batteries keep a different rhythm.
      for (let guard = 0; guard < 50; guard++) {
        arena.jet.x = 400.0 + rng.uniform(-120.0, 120.0);
        arena.jet.y = 275.0 + rng.uniform(-80.0, 80.0);
        if (!insideAny(arena.jet.x, arena.jet.y, arena.terrain, 40.0)) break;
      }
      arena.jet.heading = rng.uniform(-3.14159, 3.14159);
      const pilot = new GraphPilot(level, model, mode, QUALITY);
      pilot.start(arena);
      const cmd = maxRateCommander(rng, level);
      const t0 = performance.now();
      for (let k = 0; k < FRAMES; k++) {
        arena.step(pilot, cmd);
        if (arena.game_over) break;
      }
      const wall = performance.now() - t0;
      pilot.stop();
      alive += arena.game_over ? 0 : 1;
      frames += arena.frame;
      ms += wall / Math.max(1, arena.frame);
      flares += arena.stats.flares;
      volleys += arena.stats.volleys;
      if (arena.game_over) causes[arena.cause] = (causes[arena.cause] || 0) + 1;
    }
    const row = {
      mode: mode, level: level.number, alive: alive, seeds: SEEDS,
      sec: (frames / SEEDS / 60).toFixed(1),
      ms: (ms / SEEDS).toFixed(1),
      flares: (flares / SEEDS).toFixed(0),
      volleys: (volleys / SEEDS).toFixed(1),
      causes: Object.entries(causes).map(([k, v]) => k + " " + v).join(", ") || "-",
    };
    table.push(row);
    console.log("L" + row.level + " " + mode.padEnd(8) + " " + QUALITY.padEnd(5) + " " + row.alive + "/" + row.seeds
      + "  alive " + row.sec + " s  " + row.ms + " ms/frame  flares " + row.flares
      + "  volleys " + row.volleys + "  " + row.causes);
  }
}

console.log("\n| level | pilot | survived | mean alive | ms/frame | deaths |");
console.log("|---|---|---|---|---|---|");
for (const r of table) {
  console.log("| L" + r.level + " | " + r.mode + " | " + r.alive + "/" + r.seeds
    + " | " + r.sec + " s | " + r.ms + " | " + r.causes + " |");
}
