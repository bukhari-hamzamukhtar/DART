# neural_evasion — main3.py, split up

`main3.py` (1459 lines, one file) is unchanged and still runs. This package is
the same program broken into parts. Same architecture, same `jet_brain_v3.pt`,
same gameplay.

```
python play.py             # or:  python -m neural_evasion
python main3.py            # the original, untouched
```

## File map

| File | Lines | Was main3.py | Holds |
|---|---|---|---|
| `config.py` | 29 | §4 | arena size, palette, `MANEUVER_NAMES`, checkpoint path |
| `utils.py` | 8 | §4 | `normalize_angle` |
| `display.py` | 17 | §4 | `pygame.init()`, `screen`, `clock` |
| `model/layers.py` | 63 | §1 | `EdgeModel`, `NodeModel`, `ThreatGNN` |
| `model/attention.py` | 41 | §1 | `TemporalThreatAttention` (GRU + attention + danger head) |
| `model/brain.py` | 46 | §1 | `NeuralEvasionBrain` |
| `model/loader.py` | 71 | §2 | shape-filtered checkpoint load, `get_brain()` singleton |
| `sim/graph_builder.py` | 66 | §3 | scene → per-missile sub-graphs + context vectors |
| `sim/worldmodel.py` | 196 | new | vectorised forward simulator of jet + missiles |
| `sim/planner.py` | 202 | new | rolls out an action grid, network re-ranks the shortlist |
| `sim/evaluate.py` | 74 | new | one batched encoder pass for many scenes |
| `sim/decision.py` | 24 | §3 | `AIDecision` — the packet the worker returns |
| `sim/inference.py` | 257 | §3 | the two queues + `gnn_inference_worker` |
| `sim/entities.py` | 240 | §5 | jet, missiles, flares, explosions, launchers |
| `sim/tactics.py` | 131 | §9 | wall geometry, intercept-flare prediction, cluster repulsion |
| `sim/maneuvers.py` | 276 | §9 | `ManeuverController` — the 6-state evasion FSM |
| `ui/fonts.py` | 22 | §6 | the seven fonts |
| `ui/effects.py` | 105 | §6 | radar backdrop, particle field, glow text, scanlines |
| `ui/widgets.py` | 63 | §6 | `NeonButton` |
| `ui/hud.py` | 161 | §9 | threat bar, attention histogram, polar chart, `[G]` overlay |
| `ui/screens.py` | 196 | §7, §8 | start screen, game-over screen |
| `game.py` | 329 | §9 | `Match` — one round, five phases per frame |
| `app.py` | 21 | §10 | menu ↔ match routing |

## Dependency direction

```
config, utils
    ^          ^
  model/     display  ->  ui/{fonts,effects,widgets,hud,screens}
    ^                          ^
  sim/{graph_builder, entities, tactics, maneuvers, inference}
                        ^
                      game.py  ->  app.py  ->  play.py
```

No cycles. `model/` and `sim/graph_builder.py` never import pygame, so they can
be reused headless (benchmarks, latency runs, param counting).

## One frame

```
Match.run()
  handle_events()        keys: H / B / C fire, G toggles overlay, ESC exits
  exchange_with_brain()  every 6th frame: push world snapshot; collect AIDecision
  update_ai()            cooldowns -> ManeuverController.select() -> .execute()
                         -> jet counter-attack
  update_physics()       jet, flares, explosions, both missile lists,
                         collisions, cluster repulsion, win checks
  render()               radar, entities, glow pass, HUD
```

The brain runs on its own thread. Both queues are `maxsize=1`, so a slow forward
pass drops a snapshot instead of stalling the render loop.

## Checks

```
python neural_evasion/tests/run_all.py
```

* `test_equivalence.py` — the model path must still match main3.py exactly:
  same weights, same graph tensors, same danger / attention / maneuver output.
  The *planner* around it is deliberately different now, so decisions no longer
  match and are no longer compared. Also checks that batched scene evaluation
  equals calling the model once per scene.
* `test_worldmodel.py` — steps the vectorised simulator alongside the real
  entity classes and asserts they do not diverge (currently 0.00e+00 px over 40
  frames), plus the head-on ballistic case.
* `test_maneuvers.py` — forces all six FSM branches to execute.
* `test_game_smoke.py` — 2000 headless frames under attack, no exceptions.
* `test_screens.py` — renders both menu screens.

Not in `run_all.py` because it takes tens of minutes:

* `test_evasion_ab.py [n]` — closed-loop A/B of the old frozen-candidate search
  against the world-model planner, same physics and FSM for both, reported per
  scenario type with death causes.

## Deliberate differences from main3.py

Everything below is behaviour-preserving except the two marked **fix**.

1. Dead code dropped: the `close_80/close_150/close_200/head_on` block (computed
   every frame, never read), a `for ... : pass` loop in the start screen, three
   unused locals, and a `current_situation_danger` assignment that was
   overwritten before use.
2. `spawn_flares` / `predict_intercept_flare` / `heading_to_center` were closures
   inside `run_game()`; they now take `jet` and `flares` as arguments.
3. The decision queue carries an `AIDecision` object instead of a bare 7-tuple.
4. **fix** — shutting the worker down no longer uses a blocking `put(None)` on a
   `maxsize=1` queue. Sending that sentinel more than once after the worker had
   exited could freeze the game; `stop_worker()` is non-blocking and the worker
   also watches a per-match `threading.Event`.
5. **fix** — the checkpoint is looked up next to the repo root as a fallback, so
   the game starts from any working directory instead of silently falling back
   to random weights.

## Maneuver head — fixed

`maneuver_head` used to be declared with three Linear layers while
`jet_brain_v3.pt` stores a **two-layer** head. The loader's shape filter silently
skipped the mismatched tensors, so 630 parameters were re-randomised on every
launch and the recommended maneuver was untrained. The declaration now matches
the checkpoint:

```python
self.maneuver_head = nn.Sequential(
    nn.Linear(65, 32), nn.ReLU(), nn.LayerNorm(32),
    nn.Linear(32, len(MANEUVER_NAMES))
)
```

Applied to `main3.py`, `benchmark_v2.py`, `headless_latency.py`, `param_count.py`
and this package. All 104 checkpoint tensors now load; nothing falls back to
random init. Real parameter count is 198,440.

## Missile `type` — fixed

Training (`architecture.ipynb` cell 4) sets the missile **node** feature to the
constant 1 (it is the role: jet 0 / missile 1 / flare 2) and the **context**
`mtype` to the real guidance type (0 homing, 1 ballistic). The game used to
hard-code both to 1, so every homing missile was presented to the network as if
it were ballistic; the benchmarks had the opposite half wrong.

`HomingMissile.mtype = 0` and `BallisticMissile.mtype = 1` now travel with the
state snapshot, and all four inference paths follow the training convention.

## Spatial reasoning and the world model

The old candidate search fed the network the jet at its **current** position
with a hypothetical velocity and froze every missile, so the projected position
it computed never reached the model. It could not distinguish turning away from
a threat from turning into one — which is how the jet flew into ballistic
missiles.

`sim/worldmodel.py` now simulates the real thing: the jet under its actual drag
physics and steering gain, homing missiles re-aiming at the jet's predicted
position at their real turn rate, ballistic missiles flying straight. Every
candidate action is rolled forward `ROLLOUT_HORIZON` frames at once with numpy,
yielding closest approach, predicted impact frame and wall contact.

`sim/planner.py` turns that into an action:

1. simulate the whole `HEADING_DELTAS x THROTTLE_CHOICES` grid;
2. score analytically — a predicted impact outranks everything;
3. hand the best `ROLLOUT_TOP_K` futures to the network, which scores the
   **rolled-out end state** rather than the present one;
4. pick the lowest combined cost.

`sim/evaluate.py` runs the current scene and every shortlisted future through
**one** encoder pass, which is what pays for the extra reasoning: results are
identical to per-scene calls (verified to float32 tolerance) at a fraction of
the cost. The `[G]` overlay's second encoder pass now only runs when the overlay
is actually on — it used to run on every improving candidate.

## Speed autonomy

The jet chooses its own throttle. `THROTTLE_CHOICES` are simulated alongside the
headings, so the choice comes from physics the rollout actually models — braking
tightens a turn, sprinting outruns a slow missile. The danger head cannot
express any of that, having been trained on a jet permanently at 5.0.

`Jet.set_throttle` clamps to `MIN_THROTTLE`/`MAX_THROTTLE`; the scripted
maneuvers keep their own speed profiles (the COBRA's hard brake to 0.3 is the
point of the maneuver), and NORMAL flight uses the planner's choice.

## Countermeasures

`Plan.cornered` is true when *every* candidate action still ends in a predicted
impact — flying alone will not save the jet, so it deploys flares. That replaces
`danger > 0.55` as the primary trigger; the danger threshold remains as a
secondary one.

## Where the hand-written layers went

Two layers used to sit on top of the search and override it.

**The flee blend.** "Point away from the nearest missile" applied out to 200 px
and routinely outvoted the plan. Ablating it changed survival by exactly nothing
(52.1% either way), and it can point the jet at a wall the plan was steering
around — so it no longer blends into a live plan (`REFLEX_WEIGHT = 0`). The code
stays, because before the first plan arrives it is the only thing steering.

**The scripted maneuvers.** A COBRA brakes to 0.3 and pitches perpendicular for
40 frames; a JINKING zigzags for 120. Whatever the planner had decided, those
frames belong to the script. With the maneuver head recommending NORMAL almost
always, most entries came from the geometry safety net firing COBRA whenever a
missile was inside 80 px — exactly the moment the planner most needs the
controls. Turning the FSM off entirely raised survival from 52% to 88% in the
A/B harness.

So the FSM is now a **fallback**, not an override. `ManeuverController.select`
engages a maneuver only when

* there is no fresh plan yet (the hand-written logic is all there is), or
* the plan reports `cornered` — no heading avoids an impact, so the
  flare-dumping maneuvers are the remaining card to play.

The maneuvers all still exist and still fly exactly as before; they just no
longer take the controls away from a planner that has already simulated the
alternatives. Expect the HUD to read NORMAL far more often, with a COBRA when
the jet is genuinely trapped.

## Measured effect

**Read the earlier version of this section with suspicion — it has been
rewritten.** The numbers it carried (52% -> 87.5% from gating the FSM, and the
old vs new A/B) came from a harness that replanned *synchronously with zero
latency*, which the real game does not. Two harness bugs also corrupted the
first round of real-game diagnosis: leaked worker threads between trials (so
some trials never received a plan at all), and — worse — the match ends when the
jet destroys both launchers, which the harness was scoring as the jet *dying*.

### What actually decides this: how often the jet re-decides

Same policy, same physics, varying only how often it re-judges the trajectory:

| re-decide every | survives |
|---|---|
| 1 frame | **8/8** |
| 2 frames | 1/8 |
| 3 frames | 0/8 |
| 6 frames (the original design) | 0/8 |

Nothing else in this project comes close to that effect size. A missile covers
5-7 px per frame against a 13 px kill radius, so a decision even three frames
old is answering a question about a world that no longer exists.

### What the horizon has to be

At 1-frame cadence, with and without the network re-ranking candidates:

| horizon | with GNN re-rank | without |
|---|---|---|
| 48 | 0/8 | 0/8 |
| 72 | **8/8** | 6/8 |

Horizon 72 is the requirement; the network is worth about 2/8 on top of it. It
costs ~30 ms though, which would put the control decision three frames behind —
and by the table above that loses far more than it gains. So the network no
longer re-ranks candidates (`ROLLOUT_TOP_K = 0`); it still supplies the danger
score, the maneuver head and the GRU memory, on the worker thread.

Coarsening the simulation is not available either: at `step=2` the predicted
closest approach is off by ~314 px against a 13 px kill radius, and it picks a
different action in 7/40 cases.

### Making it affordable

Full fidelity every frame is the requirement, so the rollout had to get faster.
The jet's own path turns out to be closed-form — its heading error decays
geometrically toward the target and its velocity is a linear recurrence in that
heading, so all 72 steps collapse into a cumulative sum. Only the missiles, which
chase the jet, still need iterating.

| | horizon 72, step 1, 48 actions, 5 missiles |
|---|---|
| original per-candidate loop | ~26 ms |
| closed-form jet path | **12 ms** |

Still exact: `test_worldmodel.py` holds it to ~1e-13 px against the real entity
classes. Two other attempts are recorded here because they failed — a complex
number rewrite (exact, no speedup: the ops got dearer as the call count fell)
and predicting threats once against a nominal path (2.4x faster, 69 px of error,
unusable and removed).

### Where it stands

In the real game loop at 60 fps, with the player firing steadily:

| | result |
|---|---|
| full match | 5/5 — four jet wins, one ran out the clock |
| pure evasion, invulnerable launchers, steady fire | 3/6 survive 15 s, **0 missile hits** |
| pure evasion, brutal fire (0.8 s intervals + clusters) | median 10.3 s, **0 missile hits**, 7/8 deaths are walls |

Missile evasion is effectively solved — across those trials nothing was hit by a
missile. What kills the jet now is being herded into a wall in a 800x550 arena,
which is a genuinely hard squeeze and the open problem.

`WALL_MARGIN` was swept against the corrected cost: 110 -> 1/6, 150 -> 3/6,
190 -> 1/6. Too little and the jet gets herded; too much and it gives up the
outer ring of the arena and gets cornered in open ground instead.
