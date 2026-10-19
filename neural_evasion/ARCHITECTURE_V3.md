# v3 — dynamic scene graph, temporal graph network, graph world model

```
python play.py        # start screen -> pick a theatre -> pick a pilot (V1 / V2 / V3)
python play_goated.py # V1 exactly as it was
python play_v2.py     # V2 exactly as it was
python neural_evasion/tests/run_all.py
```

In a browser, with no install: **https://neural-evasion.vercel.app**

That build is the whole v3 stack ported to JavaScript — arena, dynamic scene
graph, trained network and graph world model all run on the visitor's own CPU,
since torch and numba cannot ship to a browser. Two tests hold the port to the
Python one: the network reproduces the torch model on real scene-graph ticks
(5e-7 with float32 weights, 9e-4 with the float16 weights the page ships), and
the world model still reproduces the arena exactly (0.00e+0 px). See
`web/README.md`.

## 1. What "dynamic graph" means in the literature

Two formal definitions are standard (Kazemi et al., *Representation Learning for
Dynamic Graphs: A Survey*, JMLR 2020):

* **Discrete-time dynamic graph (DTDG)** — a sequence of snapshots
  G¹, G², …, Gᵀ sampled at regular times.
* **Continuous-time dynamic graph (CTDG)** — an initial graph plus a stream of
  timestamped events: node added, node deleted, edge added, edge deleted,
  attributes updated. The snapshot at any time t is the initial graph with every
  event up to t applied.

The model family built on this is **Temporal Graph Networks** (Rossi et al.,
2020). Each node keeps a **memory** vector, keyed by the node's identity.
A node seen for the first time starts from zero. Its memory is updated by a
recurrent cell (GRU/LSTM) whenever the node takes part in an event. An
embedding module aggregates neighbours' memories so a node that has been quiet
does not go stale. Current work is still in this family: at NeurIPS 2025, TAMI
and SALoM (Structure-Aware TGN with a Long-Short Memory Updater); at ICLR 2026,
the TGM library and Graph Retention Networks. Graph-structured world models are
their own line — Graph World Model (ICML 2025) and a 2026 survey, *Graph World
Models: Concepts, Taxonomy, and Future Directions*. For simulation, **GNS**
(Sanchez-Gonzalez et al., ICML 2020) recomputes graph edges at every timestep
from a connectivity radius, so topology changes as the objects move.

### How that maps onto the idea this project started from

The starting description was: a scene becomes a rich graph; when something new
appears, a child graph is made of the new scene; and so on until the problem is
solved.

* **Right:** the scene is a graph, it is rebuilt as the scene changes, and a
  snapshot sequence is exactly a DTDG.
* **Not what the literature means:** the snapshots are not independent child
  graphs. The *same* nodes persist from one snapshot to the next with their
  identity and memory. New things are node insertions and vanished things are
  node deletions, in one graph that evolves; nothing stops "when the problem is
  solved".
* **Where branching child graphs really exist:** in the world model. From the
  current graph, every candidate action spawns an imagined future — a sequence
  of predicted graphs in which lock, datalink and decoy edges appear and
  disappear. That is a tree of imagined graphs, rebuilt every frame. v3 has
  both: one real evolving graph (TGN-style), and a fresh fan of imagined futures
  per decision.

## 2. What v3 is

| component | file | learned or hand-written | job |
|---|---|---|---|
| arena + levels | `arena.py`, `levels.py` | rules | one frame-based world shared by every pilot: guided missiles in four flight styles, seekers that steer around peaks, a flare magazine |
| radar + tracker | `sim/tracker.py` | hand-written | persistent tracks; dead-reckons tracks hidden by terrain, including a seeker's own terrain avoidance; keeps each missile's estimated *last known point* from seeker-edge deletion events |
| dynamic scene graph | `sim/scene_graph.py` | hand-written construction | 5 node types, 9 edge types, snapshot every 4 frames, persistent node keys, `diff` = the event stream |
| temporal graph network | `model/temporal_gnn.py` | **learned** (117k parameters) | memory per node key (GRU) + 2 rounds of typed, gated message passing; estimates each track's hidden guidance state |
| graph world model | `sim/worldmodel_v3.py`, `sim/rollout_kernel.py` | hand-written dynamics, **learned inputs** | 57–58 candidate actions (a heading × throttle grid, maneuver options, last frame's choice) × 72 frames; seekers and their terrain avoidance, terrain, datalinks, flight styles, flares (decoys and flare kills), missile burn-out and swarm repulsion re-evaluated at every imagined step |
| planner | `sim/planner_v3.py` | hand-written cost | every frame, flies the cheapest imagined future; while the magazine holds a charge, imagines every action with and without a volley and fires only if the volley makes the best future cheaper; maneuvers stay soft, as in v2 |

**Nodes:** JET, TRACK (visible or coasting), FLARE, OBSTACLE, STATION.
**Edges:** THREAT / WATCH (track↔jet), SEEKER (by doctrine, this track's seeker
can see the jet now), LINK (tracks in datalink range), NEAR, DECOY (a flare in a
track's seeker cone), TERRAIN, OCCLUDE (this peak blocks this track's view of the
jet), LAUNCH (fired from this battery, recently).

**What the network estimates, per track:** does it steer at all; turn rate
(mean *and* 90% upper quantile); lateral bias; lead-pursuit gain; flank side and
flank angle; how widely it weaves or darts; does it hold the jet right now; will
it threaten the jet in the next 72 frames. Side is learned as two classifiers:
does the missile fly off to a flank at all, and if so which side. Each track node
carries its heading relative to its line of sight to the jet, and its range.
Earlier builds regressed side onto −1 / 0 / +1. Most missiles have no side, so
that regression settled on the median, 0, and never named one: its validation
accuracy stayed exactly at the share of side-0 tracks.
**Per LINK edge:** is this a live datalink.

### Learned versus analytic — said plainly

The world model's *dynamics* are hand-written. They reproduce the arena exactly:
≤ 3e-12 px divergence with either backend, including lock losses, memory
flights, datalink guidance, terrain kills, seekers seduced by flares, missiles
and flares destroying each other, burn-out, swarm repulsion, all four flight
styles and seekers steering around peaks (`tests/test_worldmodel_v3.py`). What is learned is the **hidden state those
dynamics need**, which the radar cannot see. v1 and v2 read that state straight
off each missile object. v3 only sees positions and velocities, and the temporal
graph network has to infer the rest.

This split is deliberate. The physics here is known, so learning it could only
add error: an earlier approximation that was 69 px off against a 13 px kill
radius was unusable. The unknown part is each individual missile's internals,
and inferring those from behaviour over time and from relations to terrain,
flares and other missiles is a temporal graph problem. Real trackers face the
same split; the classical answer is an Interacting-Multiple-Model filter bank,
which handles neither relations nor datalinks.

### Flight styles, terrain avoidance and the flare magazine

Three rules keep the missiles from being easy to dodge. All three are inside the
world model.

**Flight styles** (`sim/doctrine.py`). Every guided missile leaves the rail in
one of four styles, drawn at random per missile. Each style is a deviation of
the aim from the line of sight, given as the tangent of an angle, so the arcs
stay wide at any range:

| style | what it does | numbers |
|---|---|---|
| pursuit | straight at the jet, leading it where the level has lead | — |
| flanker | steers off to one side and closes from the flank; the deviation fades out between 240 and 60 px of range so the missile still arrives | tan 0.45–0.90 (24–42°) |
| weaver | swings its aim left and right in a sine | tan 0.25–0.50, 0.07–0.16 rad per frame |
| erratic | darts to a new random side every 10–26 frames | tan 0.30–0.60 |

- **Mix:** pursuit is 25–35% of single launches, depending on the level.
- **Fade:** weaving and darting fade out only inside 115 px.
- **Pincers:** pincer members are flankers at tan 0.40 (22°).
- **Clusters:** a cluster of five always holds two flankers on opposite sides,
  a weaver, a pursuer and an erratic missile, in shuffled slots. In
  `tests/test_rules.py` such a cluster spreads to 146 px mean spacing, against
  93 px for the same cluster in pure pursuit.
- **Darts:** the direction of each dart is a deterministic hash of the
  missile's seed and the dart number. The world model therefore reproduces
  them exactly, *given* the seed.

**What the jet can know about a style:**
- **Estimated by the network:** flank side, flank angle and swing amplitude
  (weave plus dart).
- **Not observable in advance:** the phase of a weave and the direction of the
  next dart.
- **How the planner copes:** it imagines a weaving or erratic track along its
  average line. In the distance and hit costs only, that track counts as closer
  by 40 px per unit of estimated swing, so a swing of 0.4 is a 16 px margin.
- **The oracle** gets the true phase and dart table, and no margin.

**Terrain avoidance.** A seeker (levels 2–3) looks ahead along its nose:
1.2 turning radii (speed / turn rate) plus 50 px, capped at 260 px. It turns at
its full rate away from the nearest peak it would pass within 14 px of, unless
what it is chasing is nearer than that peak. In `tests/test_rules.py` a seeker
aimed through a peak reaches the jet after 55 frames; the same seeker without
avoidance is destroyed on the peak after 19. The tracker's dead reckoning of
hidden tracks and the world model apply the same rule.

**Flare magazine.**
- **Start:** each life starts with an empty magazine.
- **Earning:** every cluster launch adds one charge; on level 3, either
  battery's cluster counts.
- **Spending:** a frame's volley costs one charge, however many flares it
  holds. Flares ejected with the magazine empty never leave the jet.
- **Same rule for every pilot:** the arena applies it to all of them. V1 and V2
  keep their own flare triggers and lose the flares they cannot pay for.
- **V3:** considers a volley only while a charge is available, at most once per
  30 frames.

The HUD shows the magazine.

### Why turn rate is planned against an upper quantile

A missile flying straight at the jet shows no turning, so its turn rate cannot
be observed exactly when it matters. An early v3 estimated 0.051 for a missile
whose true rate was 0.150, flew a path that only a sluggish missile would miss,
and was hit. Given true values it survived the same scenario. So the network
also predicts the 90% quantile (trained with a pinball loss), and the planner
flies against that. The quantile is wide until the missile shows its hand, and
tight after. Section 6.2 shows this input also predicts missile paths better
than the mean estimate on levels 1 and 2 at +24 and +48 frames. On level 3,
where flankers and weavers dominate, the mean predicts better.

The quantile still misses some missiles. On held-out episodes it covers the true
turn rate 88–93% of the time. Replays of the trained v3's level-1 deaths, where
the oracle survived, found exactly such misses. In one, a missile's true turn
rate was 0.150, the head read 0.136–0.142 for most of the approach, and the
missile closed the 23 px gap the plan had left. So we tried to fix the coverage
with **conformalized quantile regression** (Romano, Patterson and Candès,
NeurIPS 2019), with no retraining (`training/calibrate.py`):
1. On the held-out episodes, compute *true turn rate − turn_hi* for every guided
   track at every tick.
2. Add that residual's split-conformal 99% quantile to the head's output, per
   level and per checkpoint. The margin is stored beside the checkpoint
   (`*.calib.json`).

| checkpoint | L1 coverage → margin | L2 coverage → margin | L3 coverage → margin |
|---|---|---|---|
| full temporal GNN | 0.934 → +0.011 | 0.883 → +0.013 | 0.900 → +0.014 |
| no message passing | 0.907 → +0.027 | 0.907 → +0.051 | 0.893 → +0.028 |
| no memory | 0.899 → +0.022 | 0.867 → +0.056 | 0.855 → +0.053 |

(Coverage of the raw quantile, then the margin in rad/frame that lifts it to
0.990.) At equal 99% coverage the full network's bound is the tightest on every
level:
- **Without message passing**, the margin is 1.9–3.8× larger.
- **Without memory**, it is 2.0–4.2× larger.

So both the graph and the memory buy a *less conservative* bound, not only lower
error. Two caveats: consecutive ticks of a track are
correlated, so the 99% is approximate rather than a finite-sample guarantee; and
the same held-out episodes also selected the checkpoint.

**In closed loop it did not help, so it is off by default.** With the
calibrated bound, the full v3 survived as often as without it (16 seeds per
level, measured on the earlier build with pure-pursuit missiles and unlimited
flares; not re-run for this build). The two ablations, whose margins are larger, died more
often. An inflated turn rate is not a conservative input for pursuit dynamics.
A missile predicted to turn harder than it can is predicted on the wrong path,
and the jet can dodge into the real one. The planner needs the right value. The
remaining level-1 deaths are estimation error that a wider bound does not
remove. Calibration stays available as a documented negative result:
`GraphPilot(calibrated=True)`, or benchmark pilots `3-cal`, `3-no_edges-cal`
and `3-no_memory-cal`.

### Why flares are inside the world model

An earlier v3 left flares out of the imagined future and threw them on a
trigger: when the best future was lethal or a near miss. With true missile
values (the oracle) it still died on level 3, always against a wall. Replaying
those runs showed the same pattern each time. Missiles chased the jet into a
corner, and for about 90 frames before the crash *every* one of the 57 imagined
futures ended in a wall or a missile hit. Turning back toward the missiles
always looked fatal, because the model could not see the ~20 flares already in
the air, or that some missiles were about to burn out.

Now the rollout carries every flare the radar sees (the jet ejected them itself,
so their age is known). A seeker that sees a flare nearer than the jet chases
it, and a missile that touches one is destroyed with it — the same rules as the
arena, in the same order. Missiles burn out after their class's flight time
(doctrine, counted from the frame the radar first saw the track, so it can only
err long). Classic homing missiles push apart as the arena makes them. While the
magazine holds a charge, the planner imagines every action twice, without the
volley and with it. It fires only if the best future with the volley is cheaper by more
than `FLARE_COST`. Countermeasures become one more action the world model
scores, rather than a trigger beside it. The HUD ticker logs each volley and how
much cheaper it made the future.

### Flying the future it imagined

With an exact world model, the jet should fly the control it imagined. Replays
of the remaining deaths turned up three places where it did not.

1. **The controls.** V3 used to steer through v2's maneuver controller. Near a
   missile, that controller blended in a flee reflex. When every future looked
   lethal, it could hand the jet to a scripted maneuver. Both patched v2's
   plans, which arrived frames late. For V3 they meant flying a path nobody
   had imagined. In one oracle run the jet had ten safe futures two frames
   before a missile hit it. V3 now applies the chosen heading and throttle
   with the rollout's own steering law.
2. **Plan persistence.** The candidate headings are laid out around the current
   heading. The exact action that was safe last frame could therefore be
   missing from this frame's options. Last frame's choice is now always offered
   again, the usual warm start of receding-horizon control. A safe plan can
   always be kept unless something new appears, such as a fresh launch.
3. **What lies past the horizon.** A path can stay clear of the walls for all 72
   frames yet end fast and pointed at a corner, with no way to stop. The final
   state is now coasted forward by its momentum. Drag 0.92 means 12.5 px of
   drift per px/frame of speed. If that point is outside the arena, the future
   costs what a wall breach on its last frame would. This is a terminal
   constraint in the usual MPC sense.
4. **What is still chasing it at the end — tried, not adopted.** Under
   flanking salvos and a scarce flare magazine, the oracle is herded into walls
   on level 3 (section 6.1). Replays showed the same sequence each time:
   - the jet loitered while its safe futures dwindled from about 20 of 58 to
     none;
   - with every future lethal, it flew for about 70 frames toward the wall
     breach that was furthest away, into a corner.

   The path cost weights the last frames of the horizon least, so we tried a
   **cost-to-go**: each surviving future paid for its missile clearance at the
   last frame (weight 8, fading out at 150 px).
   - **The six seeds the oracle lost:** it rescued two and delayed the other
     four deaths (for example, frame 402 → 1013).
   - **The first nine seeds overall:** it survived 2 against 5 without it. It
     lost four runs the plain planner survived, still into walls, only later.
   - **Levels 1 and 2:** no change on the seeds run.
   - **A variant that also charged for wall room at the end:** every missile
     death came sooner.

   Our reading is that one missile's clearance is not what the jet lacks; what
   it lacks is room to manoeuvre against a whole salvo and the walls at once.
   The cost-to-go is not in the planner.

## 3. Why each level exists

| level | what changes | what it makes necessary |
|---|---|---|
| 1 OPEN SKY | the original arena | v3 must infer what v1/v2 are handed |
| 2 TERRAIN MASKING | peaks block the jet's radar, missile seekers and airframes; a missile that loses the jet flies to where it last saw it | node insertion/deletion as tracks go in and out of radar shadow; memory to carry a hidden track; lock-edge deletion events to know where each missile is now heading; a world model that foresees lock breaks. v1/v2 have no representation of terrain at all |
| 3 PINCER NETWORK | two homing batteries, west and east; salvo members share a datalink; lead pursuit | missile–missile edges finally carry information (one lock guides the salvo); link membership is not visible and has to be inferred; pure-pursuit prediction is wrong |

On every level, guided missiles fly the four styles of section 2, and every
life starts with an empty flare magazine.

**Launch keys:**
- **Levels 1–2:** H missile, C cluster, B ballistic shot.
- **Level 3:** H / C fire the west battery, K / M the east battery, S a
  pincer salvo.
- **Always:** G toggles the graph overlay.

Before each round, a briefing screen shows the map with each battery's keys, the
missile styles and the flare rules.

Weapon doctrine lives in `sim/doctrine.py`, with flight times in `config.py`:
- seeker range 340 px and a ±70° field of view;
- flares seen out to 238 px;
- datalink range 380 px;
- flight time 510 frames guided, 180 unguided;
- flight styles and terrain avoidance as in section 2.

The missiles and the jet's world model both read these numbers. The jet is
allowed to know how this *class* of weapon works, never the state of an
individual missile.

## 4. Pilots

Every pilot implements `start(arena)`, `act(arena)`, `stop()`, `draw_hud(...)`.

* **V1 / V2** — `pilots/frozen.py` imports the frozen packages' planner,
  inference and maneuver code unmodified and replays what their `Match` did
  around it. Three things are necessarily new and apply to every pilot: they see
  what the radar sees (less, behind terrain); seeker missiles are reported to
  them as homing, since their network was never trained on a third type; and
  their flares come out of the same magazine as everyone's.
* **V3** — `pilots/graph_pilot.py`: radar → tracker → graph → temporal GNN →
  graph world model → planner → the chosen heading and throttle, applied with
  exactly the steering the world model assumed. V3 no longer passes through
  v2's maneuver controller (see "Flying the future it imagined").

## 5. Checks

| test | holds |
|---|---|
| `test_worldmodel_v3.py` | world model = arena with true parameters, both backends, including which flare each missile destroys and on which frame; scenarios must exercise lock loss, memory flight, datalink guidance, terrain kills, decoys, flare kills, burn-out and swarm repulsion |
| `test_rules.py` | an empty magazine at the start of a life; one charge per cluster launch, from either level-3 battery; one charge per volley; unpaid flares never leave the jet; the level-3 keys H/C west, K/M east, S pincer; all four styles are drawn and a cluster spreads wider than pure pursuit; darts are deterministic; a seeker steers around a peak that a blind copy of it hits |
| `test_dynamic_graph.py` | one missile keeps one node key; behind a peak it coasts with an OCCLUDE edge and is reacquired under the same key; events and edge diffs report it; memory is created, carried and deleted by key; LINK edges on level 3 |
| `test_game_smoke.py` | every level × every pilot through the real match loop with rendering |
| `test_equivalence.py`, `test_worldmodel.py`, `test_maneuvers.py`, `test_screens.py` | v1/v2 model path, v2 rollout, maneuver branches, menus |

Training and evaluation, not in `run_all.py` because they take a while:

```
python -m neural_evasion.training.generate --level 2 --episodes 40 --val 6
python -m neural_evasion.training.train --variant full        # also no_edges, no_memory
python -m neural_evasion.training.calibrate --alpha 0.01       # conformal turn-rate margins
python -m neural_evasion.training.evaluate
python -m neural_evasion.training.benchmark --levels 1 2 3 --pilots 1 2 3 3-oracle 3-prior --seeds 16
python -m neural_evasion.training.report training_data_v3/bench_*.json   # merged table, 95% intervals
```

## 6. Results

**Setup:**
- **Matches:** 16 per level per pilot. Every pilot flies the same 16 seeds, and
  each seed randomises the jet's start, its heading and the batteries' rhythm.
- **Opponent:** a scripted commander firing at the maximum rate for 30 s (1800
  frames), against launchers that cannot be destroyed, so this is pure evasion.
- **Intervals:** 95% Wilson intervals.
- **Commands:** `training/benchmark.py` runs the matches; `training/report.py`
  merges them.

### 6.1 Survival

All runs use the rules of section 2: flight styles, seekers steering around
peaks, and a flare magazine that starts empty. The earlier build's results
(pure-pursuit missiles, unlimited flares) are archived in
`training_data_v3/archive_pre_styles/`.

| level | V1 | V2 | V3 | V3, no message passing | V3, no memory | V3, no GNN | V3, oracle |
|---|---|---|---|---|---|---|---|
| L1 open sky | 0/16 (0–19%) | 10/16 (39–82%) | 10/16 (39–82%) | 12/16 | 8/16 | 8/16 | 16/16 |
| L2 terrain masking | 0/16 (0–19%) | 0/16 (0–19%) | 15/16 (72–99%) | 15/16 | 16/16 | 14/16 | 16/16 |
| L3 pincer network | 0/16 (0–19%) | 0/16 (0–19%) | 2/16 (3–36%) | 3/16 | 1/16 | 0/16 | 10/16 (39–82%) |

Mean time alive, in seconds of 30:

| level | V1 | V2 | V3 | no message passing | no memory | no GNN | oracle |
|---|---|---|---|---|---|---|---|
| L1 | 12.9 | 26.6 | 26.5 | 25.9 | 23.2 | 21.9 | 30.0 |
| L2 | 2.0 | 2.0 | 30.0 | 28.8 | 30.0 | 28.8 | 30.0 |
| L3 | 2.4 | 3.6 | 19.0 | 19.5 | 17.5 | 8.1 | 24.0 |

Causes of death:
- **V1:** L1 wall 16; L2 terrain 15, wall 1; L3 missile 8, wall 8.
- **V2:** L1 missile 2, wall 4; L2 terrain 16; L3 missile 14, wall 2.
- **V3:** L1 missile 6; L2 missile 1; L3 missile 1, wall 13.
- **V3, no GNN:** L1 missile 8; L2 missile 1, wall 1; L3 missile 16.
- **V3, oracle:** L3 wall 6.

Read plainly:
- **The missiles are harder to dodge.**
  - **V3 and its oracle:** under the earlier build's missiles, on the same 16
    seeds per level, V3 survived 43 of 48 matches and the oracle 48 of 48. Now
    it is 27 and 42.
  - **V1 and V2:** they lose ground too; on level 1, V1 drops from 6 to 0 and
    V2 from 14 to 10.
  - **Caveat:** V3 was retrained for the new missiles, so this compares rule
    sets, not pilots.
- **Levels 2 and 3 still need the new representation.** V1 and V2 survive none
  of 32 matches and live 2–4 s. V3 survives 17 of 32 and lives 19–30 s.
- **On level 1, V3 ties V2:** 10/16 each, 26.5 s against 26.6 s alive. The
  oracle's 16/16 says V3's six deaths, all by missile, are inference errors.
- **Level 3 outruns the planner.**
  - **The oracle:** survives only 10/16, and all six deaths are walls (section
    6.5).
  - **With a trained GNN:** the jet lives 17.5–19.5 s against 8.1 s without
    one, and dies against walls instead of missiles.
  - **Survival** (0–3 of 16) does not separate the GNN variants.
- **Survival does not separate the inference variants at 16 seeds.** Across all
  three levels, every interval overlaps:
  - full: 27/48
  - no message passing: 30/48
  - no memory: 25/48
  - no GNN: 22/48

The closed-loop test of the calibrated turn bound (section 2) was run on the
earlier build only; its results are archived with that build.

### 6.2 World-model prediction error

Moments are sampled from fresh runs. Each missile's future is predicted along
the jet's actual subsequent flight and compared with where it really went.
Figures are median px at +24 / +48 / +72 frames; means are in
`training_data_v3/evaluation.json`.

| level | constant velocity | world model, no GNN | world model + GNN (mean turn) | world model + GNN, as planned (90% turn quantile) | world model, true values |
|---|---|---|---|---|---|
| L1 | 28.6 / 97.7 / 204.4 | 27.9 / 49.7 / 63.9 | 10.1 / 18.6 / **29.0** | **6.4 / 15.2** / 29.5 | 0.0 / 0.0 / 0.0 |
| L2 | 9.6 / 30.1 / 40.9 | 9.1 / 19.2 / 32.5 | 4.4 / 11.2 / 19.8 | **2.7 / 7.6 / 17.2** | 0.0 / 0.0 / 0.0 |
| L3 | 17.9 / 65.3 / 120.6 | 17.8 / 34.8 / 53.0 | **5.0 / 16.9 / 35.8** | 8.0 / 25.1 / 49.6 | 0.0 / 0.0 / 0.0 |

- **True values.** The median error is zero outside the unit tests too. The mean
  at +72 is 0.6 px on level 1, 5.7 px on level 2 and 19.7 px on level 3. The one
  event the model is known to leave out is the jet's own counter-attack missiles
  shooting missiles down. That this explains the residual is our reading, not
  separately verified.
- **With the GNN versus without.** Error is lower at every horizon on every
  level. At +48 frames it is about half or less:

  | level | +48 frames, with GNN | +48 frames, without |
  |---|---|---|
  | L1 | 15.2–18.6 px | 49.7 px |
  | L2 | 7.6–11.2 px | 19.2 px |
  | L3 | 16.9–25.1 px | 34.8 px |

- **Mean versus quantile.**
  - **Levels 1–2:** the as-planned 90% turn quantile predicts better than the
    network's mean turn rate at +24 and +48. On level 1 the two tie at +72.
  - **Level 3:** it is the other way round, and the quantile's paths are
    40–60% further off.
  - **Our reading:** a pure pursuer spends much of its flight at its turn
    limit, so an upper value describes it well. Flankers, weavers and pincer
    members, which dominate level 3, turn at the limit less often, so the same
    upper value overstates them.
- **Flight styles.** The GNN rows include the error from weaves and darts. No
  estimate has their phase, and the swing head learned nothing (section 6.3).

### 6.3 Inference accuracy on held-out episodes

| estimator | L1 turn-rate MAE | L3 lead MAE | L3 lock-state accuracy | L3 datalink F1 | flank side accuracy L1 / L2 / L3 | threat recall L1 / L2 / L3 |
|---|---|---|---|---|---|---|
| hand-written rules | 0.151 | 0.541 | 0.752 | 0.843 | 0.63 / 0.58 / 0.55 | 0.39 / 0.41 / 0.41 |
| GNN, no message passing | 0.017 | 0.134 | 0.773 | 0.946 | 0.75 / 0.59 / 0.70 | 0.79 / 0.60 / 0.76 |
| GNN, no memory | 0.016 | 0.058 | 0.916 | 0.927 | 0.71 / 0.58 / 0.64 | 0.84 / 0.55 / 0.73 |
| temporal GNN (full) | 0.016 | 0.054 | 0.922 | 0.968 | 0.82 / 0.65 / 0.75 | 0.85 / 0.58 / 0.74 |

The rules always answer "no flank side", so their side accuracy is the share of
tracks that have none.

What each part buys:
- **Message passing** carries lead (0.134 → 0.054) and lock state
  (0.773 → 0.922) on level 3. Both depend on the missile's relation to the jet
  and to its salvo partners.
- **Memory** carries datalink membership (F1 0.927 → 0.968) and flank side
  (0.64 → 0.75 on level 3, 0.71 → 0.82 on level 1). Both show only in how a
  track behaves over time.
- **Both** tighten the calibrated turn-rate bound on every level (section 2).

Limits:
- **Weaving and darting: nothing learned.** The swing estimate is no better
  than always answering 0, so the planner's per-track weave margin is
  effectively off for the trained pilots:

  | level | swing error, GNN | always 0 |
  |---|---|---|
  | L1 | 0.166 | 0.166 |
  | L2 | 0.162 | 0.149 |
  | L3 | 0.151 | 0.147 |

- **Flank angle is coarse.** The error on level 3 is 0.082 (as a tangent),
  against 0.136 for the best constant.
- **Flank side is weak on level 2.** It barely beats always answering "none"
  (0.65 against 0.58).
- **Flank side cannot simply be read off the heading.** A hand rule on the
  averaged heading offset from the line of sight scores below always answering
  "none" on every level: 0.60 / 0.52 / 0.48.
- **The threat head warns more often than it needs to.** Precision is
  0.42–0.48.

### 6.4 Frame cost

How this was measured:
- the real match loop with rendering and the graph overlay on (dummy video driver);
- numba on 4 threads, torch on 1, as in the game;
- jet and launchers invulnerable, maximum fire rate, 840 measured frames per level.

An unrelated four-process job was running on the same laptop at the time, so
treat these as upper bounds.

| level | world-model plan | graph + GNN (every 4th frame) | render | whole frame, median / p95 |
|---|---|---|---|---|
| L1 | 4.7 ms | 4.1 ms | 6.8 ms | 13.5 / 21.0 ms |
| L2 | 6.0 ms | 3.9 ms | 10.0 ms | 18.3 / 31.3 ms |
| L3 | 4.3 ms | 5.0 ms | 7.3 ms | 13.7 / 23.0 ms |

The 60 fps budget is 16.7 ms per frame. The game is frame-locked, so a slow
frame slows the game down but never makes the physics coarser. Level 2 is over
budget at the median, mostly from rendering: the terrain radar shadows add a
full-screen alpha pass. Since this measurement, the planner skips imagining a
volley when it provably cannot win, which only removes work.

These timings were taken on the earlier build. The rollout now also carries
flight styles and seeker terrain avoidance:
- **Unit test:** 2–5 ms for 57 actions × 72 frames × 6 styled missiles (numba,
  on a machine running other jobs), in line with the plan column above.
- **Benchmarks (no rendering):** 8–11 ms per frame for V3 and 11–20 ms for the
  oracle, with up to eight benchmark and training processes sharing the CPU.

### 6.5 Known limitations

- **Level 1.** V3 survives 10/16 and only ties V2 (10/16); the oracle survives
  16/16.
  - **All six deaths are missile hits,** so they are inference errors: the
    swing head learned nothing, and flank side is right about four times in
    five (section 6.3).
  - **Conformal calibration** of the turn-rate bound did not help on the
    earlier build.
- **No foresight of launches.** The planner imagines only missiles already in the
  air, so a salvo fired at a jet already boxed in can still end a run.
- **The opponent is scripted.** It fires at the maximum rate with randomised
  profiles, but it does not adapt. That changes with the planned duel mode.
- **Training distribution.** The network is trained on flights by a simple
  autopilot, not on V3's own flying. Its accuracy in the states V3 actually
  reaches is only measured indirectly, through survival.
- **The exactness is a premise.** The world model matches the arena because the
  simulator's rules are known. What is learned is each missile's hidden state.
- **Resolution.** With 16 matches per cell, differences of less than roughly 30
  percentage points cannot be resolved; see the intervals.
- **Scope.** Seeker doctrine, datalinks, lead pursuit and pincers are this
  simulator's rules. Nothing here claims anything about real missiles.
  Timings come from one laptop CPU (i5-8350U).
- **Level 3 outruns the planner.** Given every missile's true values, V3
  survives 10 of 16 matches on level 3, and all six deaths are walls.
  - **How it dies:** flanking salvos herd the jet while its flare magazine is
    empty.
  - **What was tried:** a cost-to-go on terminal missile clearance, which did
    not fix it (section 2, "Flying the future it imagined").
  - **Likely cause:** the planner flies one constant action per imagined
    future. It cannot represent "run along the wall, then turn back through
    the gap", which is our reading of what these runs needed.
- **Flight styles are only partly observable.**
  - **Flank side:** not simply readable from a track's heading. A hand rule on
    the averaged heading offset from the line of sight scores below always
    guessing "no side" on every level (section 6.3).
  - **Weave and darts:** the phase of a weave and the direction of the next
    dart cannot be seen at all.
  - **Margin:** the planner's 40 px per unit of estimated swing is hand-set,
    not tuned.
- **Training data uses an unlimited flare magazine.** Flares are therefore more
  common in the data than in play.
