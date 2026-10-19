# Cross-test: each world model on the other project's problem

Each world model was swapped into the other project's evaluation, with every other part
of that evaluation left as it was:

- **CausalVis** (`G:/CausalVis`) is a *learned* world model of CLEVRER tabletop scenes.
  - It learns a pairwise potential V(gap); forces are −dV/dq, advanced by a
    kick-drift-kick step with per-object drag.
  - Checkpoint `v6_voxel.pt`, class `SpatialWorld`: the model of v7/v7.1.
- **Neural Evasion V3** (this repo) is a *hand-written* world model of the game's rules:
  guidance, seekers, datalinks, flight styles, flares, terrain, burn-out
  (`neural_evasion/sim/rollout_kernel.py`).
  - Only each missile's hidden parameters are learned, by the temporal GNN.

No CausalVis file was edited. All cross-test code, logs and results live in `crosstest/`.

## At a glance

| world model | in the game: matches survived (16 per level, 48 total) | on CLEVRER counterfactuals: TEST options right |
|---|---|---|
| V3, fed true missile values | **42/48** (16 / 16 / 10) | — |
| V3 with its trained GNN | **27/48** (10 / 15 / 2) | — |
| V3 as built, run on CLEVRER objects | — | 86.8% (it is exactly straight lines there) |
| V3 with 2 constants + drag per kind fitted on CLEVRER TRAIN | — | 88.0% |
| CausalVis learned model, as trained (zero-shot) | **0/48** (dies in 2–3 s) | **88.1%** |
| CausalVis learned model, fine-tuned on game trajectories | **0/48** (dies in 2–3 s) | — |
| straight lines at constant velocity | 0/48 (dies in 2–3 s) | 86.8% |
| CausalVis textbook-law engine (v8) | — | 89.4% |
| no simulation at all | — | 81.2% |

Read plainly:
- **CausalVis's model in the game.** It cannot fly the jet, zero-shot or fine-tuned. In
  closed loop it is no better than imagining straight lines.
- **V3 on CLEVRER.** It answers the counterfactual benchmark within 1.3 points of
  CausalVis's model as built. With three fitted numbers it ties that model (z = −0.44).
  That says as much about the benchmark as about either model: on CausalVis's own
  trajectory probes, V3's bounce predictions are the worst of every engine tested
  (section 2).

---

## 1. CausalVis's world model flying the V3 jet

### How it was plugged in (`crosstest/cv_bridge.py`)
The V3 planner imagines 57–58 candidate flights 72 frames ahead every frame. From its
world model it needs where every missile will be at every imagined step.

**What was swapped.** `CVWorldModel.rollout` answers that with CausalVis's physics, using
the same signature and output as `worldmodel_v3.rollout`.

**What stayed exact, as in V3:**
- the planner and its cost;
- the pilot;
- everything the jet knows about itself (its own flight under each control, walls and
  terrain).

**Scene mapping, fixed before any result was seen:**
- **Units:** the 800 px arena maps onto CausalVis's 1.9-unit table, and one game frame
  is one CLEVRER frame.
- **The jet:** an external body moved along each candidate path.
- **Terrain peaks:** pinned cylinders.
- **Zero-shot attributes:** every missile and the jet are rubber spheres, the CLEVRER kind
  with the lowest learned drag (0.72% of speed per frame).
- **Fine-tuned attributes:** each missile kind gets its own CLEVRER kind.

**Game rules CausalVis's physics does not have, left out:**
- seekers chasing flares, and missiles destroyed by flares or terrain;
- burn-out.

Because there are no flares in its imagination, the planner never sees a volley pay off,
so these pilots never flare.

**v7's voxel contact impulse, also left out.** At 15 ms a call it would cost about a
minute per planning frame. A 2 px missile would also be about 3 voxels across on its grid.

**Speed-up.** Their force law is a function of the gap for each pair of object types. It
is tabulated from CausalVis's own `forces_torques` and run in numba.

### Checks that the plumbing is exact
| check | result |
|---|---|
| port vs `SpatialWorld.forces_torques` called every step (float64), 72 frames, 3 levels × 2 mappings | max 3.6e-4 px (vs their float32: 8.3e-4 px) |
| bridge in straight-line mode vs the V3 kernel fed the same missiles as unguided tracks, 80 arena moments × 48 candidates | 5e-12 px, 0 hit-frame mismatches |
| closed-loop harness with V3's own world model vs the stored `3-oracle` benchmark records | identical frames, missiles fired and flares |
| prediction-error harness: its V3 and constant-velocity rows vs `training_data_v3/evaluation.json` | identical |

**A bug found and fixed.** The first check found one. One of V's inputs is
`clamp(gap, −0.5, 0.5)`, so the force jumps at gap = ±0.5. Linear interpolation smeared
that jump across one table cell. In one level-2 scene a missile sat 0.000003 units past
the kink, and the error grew to 0.14 px. The table now stores the force on both sides of
each kink, and the check passes.

### Fine-tuning on game data (`crosstest/train_cv_refit.py`)
A potential learned on tabletop collisions has no reason to know that a missile turns
toward a jet. So the model got the chance V3's GNN got: training on the game's own
trajectories.

**Data:** 72 train and 12 validation episodes (24 + 4 per level), recorded from the same
opponent and autopilot as the GNN's data (`record_game.py`).

**Kept from CausalVis:**
- initial weights `v6_voxel.pt`;
- frozen radius;
- their legacy objective and the 3 → 20 frame curriculum;
- AdamW with clip 1.0 and lr 3e-4;
- the best checkpoint by validation error at 20 frames.

**Budget:** 1,320 batches of 8 windows, 25 min on the laptop CPU.

| validation error of missile position after 20 frames | px |
|---|---|
| CausalVis model before fine-tuning | 21.65 |
| CausalVis model after fine-tuning (best checkpoint) | 18.34 |
| straight lines at the current velocity | 18.15 |

### Missile prediction error in the game (`crosstest/wm_error_cv.py`)
The moments are `training/evaluate.py`'s: a moment every 30 frames of 4 fresh runs per
level. Each missile's future is predicted along the jet's *actual* next 72 frames and
compared with where it went. Figures are median px at +24 / +48 / +72 frames. Every row
except the V3 + GNN row starts from each missile's **true** state; that row starts from
the radar tracker.

| world model | L1 open sky | L2 terrain masking | L3 pincer network |
|---|---|---|---|
| V3, true guidance values | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| V3 + trained GNN (from the radar) | 6.4 / 15.2 / 29.5 | 2.7 / 7.6 / 17.2 | 8.0 / 25.1 / 49.6 |
| straight lines | 28.6 / 98.5 / 202.4 | 8.6 / 30.1 / 43.2 | 17.9 / 65.3 / 120.6 |
| CausalVis, zero-shot | 27.1 / 88.3 / 174.2 | 15.3 / 47.8 / 86.1 | 21.6 / 65.2 / 117.7 |
| CausalVis, fine-tuned | 25.3 / 79.1 / 149.4 | 11.2 / 32.9 / 64.0 | 17.1 / 57.2 / 98.8 |

### Closed loop (`crosstest/bench_cv.py`)
- **Harness:** `training/benchmark.py`'s, with the same 16 seeds per level.
- **Pilot:** the one behind the "V3, true values" row, fed every missile's true state.
- **The only swap:** the world model the planner imagines missiles with.

| planner's world model | L1 | L2 | L3 | mean time alive (of 30 s) | deaths |
|---|---|---|---|---|---|
| V3 (true values) | 16/16 | 16/16 | 10/16 | 30.0 / 30.0 / 24.0 | 6 wall |
| straight lines | 0/16 | 0/16 | 0/16 | 2.6 / 3.0 / 1.9 | 48 missile |
| CausalVis, zero-shot | 0/16 | 0/16 | 0/16 | 2.7 / 2.7 / 2.1 | 47 missile, 1 terrain |
| CausalVis, fine-tuned | 0/16 | 0/16 | 0/16 | 2.5 / 2.7 / 2.1 | 48 missile |

### Reading
- **Pursuit.** A potential between two objects that depends only on their distance gives
  a force along the line between them. A guided missile holds its speed and turns at a
  limited rate toward where the jet is going. A central force plus drag can only
  approximate that.
  - **Measured:** fine-tuning cut the error at +72 frames by 14–27% against zero-shot.
  - **Where it stands:** it stays 3–6× worse than V3 with its GNN, and worse than straight
    lines on level 2.
  - **Short range:** at 20 frames it never beat straight lines on validation windows.
- **Closed loop.** Every one of the 96 matches ends in 2–3 seconds, the same as imagining
  straight lines. The planner dodges paths the missiles do not fly. V3's own model, given
  the same true missile states, survives 42 of 48.
- **Caveats, stated rather than hidden:**
  - The attribute mapping is a choice.
  - Fine-tuning had a 25-minute CPU budget.
  - The voxel impulse was left out.
  - These pilots never flare.

  None of these touches the central mismatch. Even with flares, a planner that cannot
  predict pursuit is dodging the wrong paths.

---

## 2. The V3 world model answering CausalVis's CLEVRER counterfactuals

### How it was plugged in (`crosstest/clevrer_v3.py`)
CausalVis v8 builds one "what if" world per question and advances every simulated object
with a pluggable step. That step was replaced by the V3 kernel,
`rollout_kernel.missile_rollout`, called unmodified.

**Kept exactly as CausalVis's own, loaded from their files:**
- which objects follow the recording, and entries;
- the extension past the video;
- the event detectors and their contact sizes;
- question parsing;
- choosing the setting on VAL, then running TEST once.

Each CLEVRER object is a track that steers at nothing. Three variants were fixed before
any result:

| variant | what it is |
|---|---|
| **as built** (kinematic) | V3's rule for anything unguided: constant speed in a straight line. V3 has no friction for missiles and no bounce. |
| **swarm** | adds V3's only object-object rule, the heading push between classic homing missiles, at the game's constants (26 px, 0.20) |
| **refit** | V3's own rules with their free constants fitted on CLEVRER **TRAIN** only (`fit_v3_clevrer.py`): per object kind, the drag law V3 uses for flares, v ← v·(1−c); and the swarm push's distance and strength. The mirror of fine-tuning CausalVis on game data. |

**Fitted on TRAIN:**
- **Drag per frame:** cubes 0.012, cylinders 0.011, metal spheres 0.003, rubber spheres
  0.002.
- **The push:** it did not help. On 964 TRAIN collisions the best push gave a 41.7°
  median outgoing-direction error, against 42.0° with no push.

**Checks:**
- The kernel called this way equals position += velocity to 1e-13 px.
- The push matches `tactics.apply_cluster_repulsion` to 3e-14.
- CausalVis's own scoring, run from this repo, reproduces their published TEST counts
  exactly: 2704, 2936 and 2980 options.

### The benchmark: CLEVRER counterfactual TEST, 930 questions / 3,332 options
Each variant's setting was chosen on VAL by CausalVis's rule (the most VAL options right).
TEST was then run once per variant.

| system | VAL options right | chosen setting |
|---|---|---|
| V3 as built | 2966/3391 | cal +30 |
| V3 swarm | 2979 | cal +30 |
| V3 refit | 3011 | cal +45 |
| CausalVis v7 (reference) | 3017 | — |
| CausalVis v8 (reference) | 3055 | — |

| system | per option | per question | pairs the simulator decided | missed | invented |
|---|---|---|---|---|---|
| no simulation: delete the object, keep the recording | 81.2% | 48.7% | — | 465 | 163 |
| **V3 as built** (straight lines) | 86.8% | 64.4% | 80.9% | 237 | 204 |
| **V3 swarm** | 87.1% | 64.3% | 81.6% | 237 | 193 |
| **V3 refit** (2 fitted constants + drag per kind) | **88.0%** | **66.6%** | 83.2% | 209 | 191 |
| CausalVis v7: learned world model | 88.1% | 66.6% | 83.6% | 207 | 189 |
| CausalVis v8: textbook laws fitted on TRAIN | 89.4% | 69.8% | 86.1% | 199 | 153 |

**Paired comparisons** (McNemar z on 3,325 matched options):

| V3 variant | vs no simulation | vs v7 | vs v8 |
|---|---|---|---|
| as built | +7.98 | **−2.79** | −6.14 |
| refit | +9.69 | **−0.44** (no difference) | −3.79 |

Refit beats as built by z = +3.07.

### The physics itself: CausalVis's trajectory probes, VAL
CausalVis found that the benchmark barely moves when the physics improves. These probes
are what can tell physics apart. Each uses CausalVis's own protocol, and their engines were
run in the same script.

**A. Free motion**, 7,417 segments with no collision. Errors are in table units; a contact
distance is about 0.117.

| engine | @20 frames | @40 frames |
|---|---|---|
| constant velocity | 0.02837 | 0.07272 |
| **V3 as built / swarm** | 0.02837 | 0.07272 (identical: V3 *is* straight lines here) |
| CausalVis learned (v6_voxel) | 0.02496 | 0.05656 |
| **V3 refit** (flare drag law, fitted) | **0.02456** | **0.05399** |
| CausalVis law engine (best law per kind) | 0.02392 | 0.05018 |

**C. Head-to-head at real collisions.** On 1,345 isolated collisions, both objects start
8 frames before contact and each engine runs 16 frames alone.

| engine | contact made | outgoing direction error, median / p75 |
|---|---|---|
| no collision at all (keep going) | — | 36.6° / 86.6° |
| CausalVis law engine | 95.5% | 12.1° / 23.8° |
| CausalVis learned (voxel bounces) | 93.6% | 17.6° / 34.7° |
| V3 as built / swarm | 96.2% | 44.6° / 97.9°, 44.1° / 98.2° |
| V3 refit | 95.7% | 42.4° / 92.9° |

**R. Shadow replay.** Every annotated collision in 200 VAL clips is replayed from L frames
before it; "reproduced" means the pair still collides.

| engine | from 12 frames | from 30 frames | misses at 30 caused by a wrong bounce |
|---|---|---|---|
| CausalVis v7 | 96.7% | 55.0% | 24 |
| CausalVis v8 | 96.0% | 57.4% | 22 |
| V3 as built | 95.4% | 47.1% | 75 |
| V3 swarm | 95.6% | 49.5% | 67 |
| V3 refit | 95.6% | 47.4% | 62 |

### Reading
- **Our world model, applied to CLEVRER, is a straight-line predictor.**
  - **Where its exactness comes from:** the game's rules, which it encodes.
  - **What it lacks:** any rule by which a struck body moves, so objects pass through each
    other. Its bounce directions are as bad as predicting no collision at all.
- **The benchmark still gives it 86.8%, 1.3 points below CausalVis's learned model.**
  - **Refit:** with two constants and a drag per object kind fitted on TRAIN, it ties that
    model (88.0% vs 88.1%, z = −0.44).
  - **Free motion:** the fitted drag even predicts it better than the learned model.
- **This reproduces CausalVis's own finding, from the other side.** Most of the score is
  the evaluation logic: 81.2% without any simulation. A binary "did they touch" question
  cannot see bounce quality, so a simulator 25–30° worse at every bounce loses about 1
  point.
- **The law engine stays clearly ahead** (89.4%, z = −3.8 against refit). That fits it
  being the only engine here with the collision law right.

---

## Files
| file | what it is |
|---|---|
| `cv_bridge.py` | CausalVis's model as a V3 rollout; `python crosstest/cv_bridge.py` runs the port check |
| `record_game.py`, `train_cv_refit.py` | game trajectories; fine-tuning CausalVis's model on them (`cv_refit.pt`) |
| `wm_error_cv.py` | missile prediction error, same moments as `training/evaluate.py` |
| `bench_cv.py` | closed-loop benchmark, `training/benchmark.py`'s harness |
| `clevrer_v3.py` | the V3 kernel as CausalVis v8's simulator; `--selftest` |
| `fit_v3_clevrer.py` | V3's constants fitted on CLEVRER TRAIN → `v3_clevrer_fit.json` |
| `clevrer_compare.py`, `clevrer_traj.py` | TEST scoring with paired tests; the trajectory probes |
| `results/`, `runs/clevrer/` | logs, JSON results, per-option records |
