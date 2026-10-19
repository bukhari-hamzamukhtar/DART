// Is the world model the arena? (the browser twin of tests/test_worldmodel_v3.py)
//
//   node tests/exactness.mjs
//
// Given every missile's true values, rolling one candidate action forward must
// reproduce what the arena does when the jet flies exactly that action: same
// positions, same frame each missile dies, same flare destroyed on the same
// step. Divergence should be floating-point noise.
import { Arena } from "../js/arena.js";
import { LEVELS } from "../js/levels.js";
import { FlareParticle, resetUids } from "../js/entities.js";
import { Rollout, oracleThreat, flareState } from "../js/worldmodel.js";
import { FLARE_LIFE_FRAMES, STEER_GAIN, normalizeAngle } from "../js/config.js";

const T = 72;

// A pilot that flies one fixed action, exactly as the rollout assumes.
function fixedPilot(target, throttle) {
  return {
    start() {},
    stop() {},
    act(arena) {
      const jet = arena.jet;
      jet.heading += normalizeAngle(target - jet.heading) * STEER_GAIN;
      jet.setThrottle(throttle);
    },
  };
}

function scenario(name, levelIdx, setup, target, throttle, seed) {
  resetUids();
  const level = LEVELS[levelIdx];
  const arena = new Arena(level, {
    jet_invulnerable: true, launchers_invulnerable: true,
    counter_attack: false, magazine: false, seed: seed,
  });
  setup(arena);

  // Snapshot: what the jet would imagine from here.
  const frame0 = arena.frame + 1;
  const threats = arena.missiles.map((m) => oracleThreat(m, level.datalink, frame0));
  const flares = arena.flares.filter((f) => !f.dead).map((f) => flareState(f, frame0, FLARE_LIFE_FRAMES));
  const uids = arena.missiles.map((m) => m.uid);
  const flareUids = arena.flares.filter((f) => !f.dead).map((f) => f.uid);

  const roll = new Rollout();
  const res = roll.run(
    { pos: [arena.jet.x, arena.jet.y], vel: [arena.jet.vx, arena.jet.vy], heading: arena.jet.heading },
    threats, [target], [throttle], T, level.obstacles, level.datalink, flares, true);

  // Now fly it for real.
  const pilot = fixedPilot(target, throttle);
  const truth = new Map();          // uid -> {step: [x, y]} after each step
  const goneAt = new Map();         // uid -> the step it was removed in
  const jetPath = [];
  const flareGone = new Map();      // uid -> frame it was destroyed
  const modes = new Set();
  let liveFlares = new Set(flareUids);
  let prevPresent = new Set(uids);
  for (let k = 0; k < T; k++) {
    arena.step(pilot, null);
    jetPath.push([arena.jet.x, arena.jet.y]);
    const present = new Set();
    for (const m of arena.missiles) {
      if (!truth.has(m.uid)) truth.set(m.uid, new Map());
      truth.get(m.uid).set(k, [m.x, m.y]);
      present.add(m.uid);
      if (m.mode) modes.add(m.mode);
    }
    // A missile removed during step k was still alive for that step's check,
    // which is what the kernel's ALIVE[k] reports.
    for (const u of prevPresent) if (!present.has(u)) goneAt.set(u, k);
    prevPresent = present;
    const now = new Set(arena.flares.filter((f) => !f.dead).map((f) => f.uid));
    for (const u of liveFlares) if (!now.has(u)) flareGone.set(u, k);
    liveFlares = now;
  }

  // Compare up to the frame the rollout says the jet leaves the arena: past
  // that the real match is over, while this test arena bounces the jet instead.
  let kMax = T;
  for (let k = 0; k < T; k++) {
    if (res.wall_profile[k] < 0.0 || (res.terrain_profile && res.terrain_profile[k] < 0.0)) {
      kMax = k;
      break;
    }
  }
  let maxJet = 0, maxMissile = 0, aliveMismatch = 0, flareMismatch = 0, removed = 0;
  for (let k = 0; k < kMax; k++) {
    maxJet = Math.max(maxJet,
      Math.abs(res.jet_x[k] - jetPath[k][0]), Math.abs(res.jet_y[k] - jetPath[k][1]));
  }
  for (let i = 0; i < uids.length; i++) {
    const hist = truth.get(uids[i]) || new Map();
    for (let k = 0; k < kMax; k++) {
      const pi = i * T + k;                   // one candidate, so c = 0
      const predAlive = res.alive_path[pi] === 1;
      const real = hist.get(k);
      const realAlive = real !== undefined || goneAt.get(uids[i]) === k;
      if (real && predAlive) {
        maxMissile = Math.max(maxMissile,
          Math.abs(res.threat_x[pi] - real[0]), Math.abs(res.threat_y[pi] - real[1]));
      }
      if (realAlive !== predAlive) aliveMismatch += 1;
      if (!predAlive && (k === 0 || res.alive_path[i * T + k - 1] === 1)) removed += 1;
    }
  }
  for (let f = 0; f < flareUids.length && kMax === T; f++) {
    const predicted = res.flare_hit[f];       // step a missile destroyed it, or -1
    const real = flareGone.has(flareUids[f]) ? flareGone.get(flareUids[f]) : -1;
    // The arena also retires a flare when it burns out; only count kills it saw.
    if (predicted >= 0 && predicted !== real) flareMismatch += 1;
  }

  const ok = maxJet < 1e-9 && maxMissile < 1e-9 && aliveMismatch === 0 && flareMismatch === 0;
  console.log((ok ? "PASS  " : "FAIL  ") + name
    + "\n        jet " + maxJet.toExponential(2) + " px, missiles " + maxMissile.toExponential(2)
    + " px, alive mismatches " + aliveMismatch + ", flare mismatches " + flareMismatch
    + "\n        " + uids.length + " missiles, " + flareUids.length + " flares, "
    + removed + " removed, " + kMax + " frames compared, modes " + Array.from(modes).sort().join("/"));
  return ok;
}

let allOk = true;

// A: level 1 -- a classic cluster, with swarm repulsion and flight styles.
allOk = scenario("A  level 1: cluster of five, repulsion and styles", 0, (arena) => {
  arena.jet.x = 420; arena.jet.y = 180; arena.jet.heading = 0.4;
  arena.launchCluster(0);
  for (let i = 0; i < 25; i++) arena.step(fixedPilot(0.4, 5.0), null);
}, 1.1, 5.0, 3) && allOk;

// B: level 2 -- seekers, terrain, lock loss and flares to be seduced by.
allOk = scenario("B  level 2: seekers, terrain avoidance, flares", 1, (arena) => {
  arena.jet.x = 400; arena.jet.y = 300; arena.jet.heading = -1.2;
  arena.fire(0);
  arena.launchCluster(0);
  for (let i = 0; i < 20; i++) arena.step(fixedPilot(-1.2, 5.0), null);
  for (let k = 0; k < 6; k++) {
    const f = new FlareParticle(arena.jet.x, arena.jet.y, -1.2 + k * 0.5, 7.0);
    f.born_frame = arena.frame;
    arena.flares.push(f);
  }
  for (let i = 0; i < 5; i++) arena.step(fixedPilot(-1.2, 5.0), null);
}, -0.3, 7.5, 5) && allOk;

// C: level 3 -- datalinked pincer plus a cluster, lead pursuit, flares.
allOk = scenario("C  level 3: datalink pincer, cluster, lead pursuit", 2, (arena) => {
  arena.jet.x = 380; arena.jet.y = 260; arena.jet.heading = 2.0;
  arena.launchPincer();
  arena.launchCluster(1);
  for (let i = 0; i < 30; i++) arena.step(fixedPilot(2.0, 5.0), null);
  for (let k = 0; k < 4; k++) {
    const f = new FlareParticle(arena.jet.x, arena.jet.y, 2.0 + k * 0.4, 6.0);
    f.born_frame = arena.frame;
    arena.flares.push(f);
  }
}, 2.6, 3.5, 9) && allOk;

console.log(allOk
  ? "\nRESULT: THE BROWSER WORLD MODEL MATCHES THE BROWSER ARENA"
  : "\nRESULT: MISMATCH");
process.exit(allOk ? 0 : 1);
