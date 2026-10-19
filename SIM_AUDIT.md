# Does the headless sim actually train the scenario we think it does?

Audit of `headless_sim3.py` (data generation), `architecture.ipynb` cell 4
(training graph builder) and the inference paths in `main3.py`,
`benchmark_v2.py` and `headless_latency.py`.

Everything below is measured, not inferred from reading. Scripts that produced
the numbers are named per section.

> **Status.** This audit describes the state *before* the planner rewrite.
> Fixed since, in `neural_evasion/`: the maneuver-head/checkpoint mismatch, the
> missile `type` convention (all four inference paths), and — the big ones — the
> counterfactual search (§3c) and the missing speed autonomy (§4), which are now
> a real forward-simulating world model in `sim/worldmodel.py` plus an action
> planner over (heading x throttle) in `sim/planner.py`. See
> `neural_evasion/ARCHITECTURE.md`.
>
> Still open, because they need regenerated data or retraining: the naive-policy
> rollout behind the danger label (§3a), the hand-written maneuver oracle (§3b),
> harmless ballistic missiles (§1), and the stage-1 checkpoint (finding A).

---

## Verdict on the intended scenario

| Intended property | Status | Short version |
|---|---|---|
| Jet vs **homing** missiles | yes | 71.5% of missiles, cause 91–99% of hits |
| Jet vs **ballistic** missiles | nominal only | 28.5% of missiles, cause 1.4–8.6% of hits; 88–95% fly off the map |
| **Complete spatial awareness** (radar-like) | yes for state, no for reasoning | perfect information is in the snapshot, but missiles are never connected to each other in the graph |
| **Future rollout / world model** | half | the *danger label* is a 1-second lookahead; the *maneuver label* is a hand-written rule, and the in-game "counterfactual search" does not roll the world forward |
| **Speed autonomy** | no | throttle is the constant 5.0 in every frame of every training sim |

---

## 1. Homing vs ballistic

`headless_sim3.py:81` spawns 75% homing / 25% ballistic. Re-running the
generator with instrumentation (400 sims per stage):

| stage | spawned H/B | hits H/B | ballistic share of hits | ballistic left the arena |
|---|---|---|---|---|
| slow_singles | 455 / 150 | 74 / 7 | 8.6% | 53% |
| fast_homing | 642 / 189 | 408 / 6 | 1.4% | 95% |
| multi_threat | 898 / 307 | 705 / 10 | 1.4% | 95% |
| cluster_swarm | 1355 / 461 | 1238 / 28 | 2.2% | 93% |
| adversarial | 1824 / 590 | 1794 / 70 | 3.8% | 88% |

Why: a ballistic missile takes its heading once, at spawn, aimed at where the
jet *is at that instant* (`headless_sim3.py:103`). It never re-aims and there is
no lead prediction. By the time it arrives the jet has moved, so it flies past
and off the edge. Median closest approach is 208 px in stage 1.

So ballistic threats are in the data but contribute almost nothing to the danger
label the network is trained on. The network has effectively learned "dodge
homing missiles".

If you want ballistic threats to matter, give them a lead-prediction launch
(aim at where the jet *will* be) rather than at its current position — that is
what makes a ballistic shot dangerous against a moving target.

## 2. Spatial awareness

**The state is complete.** Every snapshot carries every live missile and every
flare with exact position and velocity — no occlusion, no sensor noise, no
field-of-view cone, no detection range. Omniscient radar, as intended.

**The graph is not.** Both the training builder (`architecture.ipynb` cell 4) and
the inference builder (`neural_evasion/sim/graph_builder.py`) construct **one
sub-graph per missile**, each containing `[jet, that missile, ≤6 flares within
200 px of that missile]`. `Batch.from_data_list` keeps those disjoint, so **no
edge ever connects one missile to another**. The model literally cannot see that
five missiles form a pincer; it sees five independent jet-vs-missile pairs whose
embeddings are then pooled by attention.

That is what "cross-threat attention" means here — attention over per-threat
embeddings, not message passing between threats. Worth stating precisely in the
paper, because "hierarchical GNN over the scene" implies something stronger.

Two smaller notes: node features are raw absolute pixel coordinates (0–800,
0–600) fed straight into a `Linear` with no normalisation; and arena walls are
not an explicit feature — the model can only infer them from those absolute
coordinates, which it evidently does, since the maneuver label is dominated by a
wall rule (below).

## 3. The "future rollout"

There are three separate things here and only one of them is a rollout.

**a. The danger label — this one is genuine.**
`headless_sim3.py:238-241`: for frame `t`, the label is 1 if a collision occurs
anywhere in `t … t+60` (1 second). So the danger head learns
`P(hit within the next second)` — a learned value function, which is the useful
world-model-flavoured part of this project.

The caveat: that future is rolled out under a **fixed naive policy** — 50% random
heading jitter, 50% steer-away-from-nearest capped at 0.18 rad/frame
(`headless_sim3.py:119-126`). So the label answers *"if the jet keeps flying like
this, will it be hit?"*, not *"if the jet performs maneuver X, will it be hit?"*.
There is no per-maneuver counterfactual anywhere in the dataset.

**b. The maneuver label — not a rollout at all.**
`oracle_maneuver()` (`headless_sim3.py:33`, mirrored in the notebook) is a
hand-written decision tree over distance thresholds and a wall margin. The
maneuver head is therefore imitating a rule, not learning which maneuver
survives. Measured over all 330,134 records:

| maneuver | count | share |
|---|---|---|
| IMMELMANN | 175,721 | 53.2% |
| NORMAL | 80,223 | 24.3% |
| FALLING_LEAF | 29,737 | 9.0% |
| COBRA | 28,029 | 8.5% |
| JINKING | 12,776 | 3.9% |
| BARREL_ROLL | 3,648 | 1.1% |

Over half the maneuver supervision is "you are within 45 px of a wall, do an
Immelmann" — and the game already has a hard-coded wall check that overrides the
network anyway. The notebook comment says lowering the margin from 110 to 45 cut
IMMELMANN from 68%; it is still 53%.

**c. The in-game candidate search — not a rollout either.**
`neural_evasion/sim/inference.py:_score_candidate` (identical to `main3.py:216`)
projects the jet 30 frames ahead on each of 12 candidate headings, but look at
what actually reaches the network:

```python
px, py = jx + tvx*30, jy + tvy*30          # the projection
js = {'pos': [jx, jy], 'vel': [tvx, tvy]}  # position is CURRENT, not projected
```

The projected `px, py` is used only by the two hand-written arithmetic terms
(escape bonus and centre pull). The GNN is fed the jet at its **present**
position with a hypothetical velocity, and the missiles are **never advanced**.
It answers "what if my velocity were different, right now", not "what will the
world look like in half a second".

Measured over 295 random states (`audit_rollout.py`), spread across the 12
candidates within one frame:

| term | median spread | p90 | max |
|---|---|---|---|
| GNN danger | 0.111 | 0.576 | 0.800 |
| hand-written escape + centre | 0.138 | 0.234 | 0.330 |

and the chosen heading matches the argmin of the hand-written term 66.4% of the
time versus 42.0% for the GNN term. The network does contribute — it dominates
in high-threat frames, where its spread is 2.5× larger — but a good share of the
steering is the arithmetic, not the model.

A real rollout would advance both the jet and the missiles by the lookahead and
score *that* state. That is a contained change to `_score_candidate`.

## 4. Speed autonomy — absent

`headless_sim3.py:71` sets `"speed": 5.0` and nothing ever writes to it again.
Instrumented re-run confirms: the set of throttle values used across every frame
of every sim, every stage, is exactly `{5.0}`.

The velocity magnitude in the data does vary (0.494–6.253) but only as a side
effect of drag during turns — never because the jet chose to accelerate or brake.

The game, meanwhile, changes throttle constantly: COBRA brakes to 0.3 and boosts
to 7.0, IMMELMANN runs at 7.5, FALLING_LEAF 1.75, JINKING 2.55. Measured over
3000 live frames:

| maneuver | \|v\| min | median | max | frames outside the training range |
|---|---|---|---|---|
| NORMAL | 0.50 | 5.55 | 8.40 | 8.8% |
| COBRA | 0.17 | 3.38 | 8.23 | 15.5% |
| IMMELMANN | 0.55 | 5.33 | 8.57 | 39.1% |
| all | 0.17 | 5.40 | 8.57 | **19.1%** |

So roughly one frame in five puts the jet at a speed the network never saw.

Worse, the network barely responds to speed at all. Same head-on scene, sweeping
only the jet's speed:

| \|v\| | 0.30 | 1.00 | 3.00 | 5.00 | 6.25 | 8.75 | 12.00 |
|---|---|---|---|---|---|---|---|
| danger | 0.819 | 0.847 | 0.889 | 0.893 | 0.895 | 0.896 | 0.898 |

A 40× change in speed moves the danger score by 0.08. Braking hard — the entire
premise of the COBRA — barely registers as safer. That is the direct consequence
of never varying throttle in training: the network has no examples of "same
geometry, different jet speed", so it cannot have learned the relationship.

**This is the single biggest gap between the intended scenario and the data.**
The fix is in the generator: give the sim jet a throttle policy (random walk over
0.3–7.5, or a small set of scripted maneuver profiles that mirror the game's FSM)
so the same geometry appears at many speeds.

---

## Findings that affect the paper

**A. The shipped checkpoint is the stage-1 model, not the end-of-curriculum
model.** `paper_data/v2_gru/val_auc_v2.txt` contains exactly one line:
`0.9384 - stage 1, epoch 7`, and `label_distribution_v2.txt` says that value was
what got saved to `jet_brain_v3.pt`. Because stage 1 (slow single missiles, 95.8%
safe) yields the highest AUC of any stage, a save-on-global-best rule never
overwrote it — stage 5's 0.9381 missed by 0.0003. Scoring the checkpoint against
each stage's own data confirms it:

| stage | reported peak AUC | this checkpoint scores |
|---|---|---|
| slow_singles | 0.9384 | **0.9351** |
| fast_homing | 0.8758 | 0.8169 |
| multi_threat | 0.8688 | 0.7428 |
| cluster_swarm | 0.8780 | 0.7190 |
| adversarial | 0.9381 | 0.8386 |

It matches its own stage and falls well short on the other four — on their
*training* data, where it should if anything score high. The deployed brain is
the one trained on slow single missiles. Saving per-stage checkpoints, or simply
saving the final weights, would give a materially better model.

**B. The maneuver distribution in `label_distribution_v2.txt` does not match the
data.** Record counts match exactly (94194 + 73040 + 65312 + 54751 + 42837 =
330,134) and so does the label split (4.2 / 22.2 / 40.2 / 60.6 / 72.9%, aggregate
33.6%). But the reported maneuver mix — NORMAL 49.4%, COBRA 17.0%, IMMELMANN
13.2% — is not what the files contain (24.3% / 8.5% / 53.2%). Only BARREL_ROLL
(3,648) agrees. Both sets sum to 330,134, so it looks like a table carried over
from an earlier run rather than a computation over these files.

**C. Every published number was measured with the random maneuver head.**
`survival_benchmark_v2.txt` shows JINKING, FALLING_LEAF and COBRA at exactly 0
decisions and BARREL_ROLL at 34.1% — despite BARREL_ROLL being 1.1% of the
training labels. That is the signature of the untrained head (now fixed). Any
result that touched maneuver selection needs re-running.

**D. The game feeds the network a context feature the training never used.**
Training (`architecture.ipynb` cell 4) sets the missile node type to the constant
1 and the context `mtype` to the **real guidance type** (`m.get("type", 0)` — 0
homing, 1 ballistic). The game hard-codes *both* to 1:

```python
ms = [{'pos': m['pos'], 'vel': m['vel'], 'type': 1} for m in state['missiles']]
```

so every homing missile is presented as if it were ballistic. `benchmark_v2.py`
and `headless_latency.py` get `mtype` right but pass the real type as the *node*
feature too, which training did not — that half turns out not to matter (identical
AUC to four decimals), but the `mtype` half does:

| stage | AUC, training convention | AUC, game convention |
|---|---|---|
| slow_singles | 0.9351 | 0.9322 |
| fast_homing | 0.8169 | 0.7975 |
| multi_threat | 0.7428 | 0.7375 |
| cluster_swarm | 0.7190 | 0.7035 |
| adversarial | 0.8386 | 0.8263 |

The game is giving away 0.3–1.9 AUC points for free. The fix is one line, and it
needs a `mtype` attribute on the missile classes (0 for `HomingMissile`, 1 for
`BallisticMissile`) so the state snapshot carries it.

**E. The survival benchmark never executes a maneuver.** In
`benchmark_v2.py:304` the recommended maneuver is only counted:

```python
maneuver_counts[rec_manu] = maneuver_counts.get(rec_manu, 0) + 1
```

Nothing else reads it. The benchmark jet snaps its heading straight to
`best_heading`, drops an 8-flare ring when `drop_flare` fires, flies at a constant
speed 5.0, and is `clamp`ed inside the walls so it cannot die by hitting one. So
the published 84% survival rate measures the danger head, the candidate search
and the flare rule — and nothing else. It does not test the FSM, the speed
changes, the intercept-flare prediction or the wall-death rule that the player
actually faces. Re-running it after the head fix confirms this: survival stayed
at 84.0%, while the recommendation mix swung from BARREL_ROLL 34.1% / NORMAL
61.2% (random head) to NORMAL 99.5% / COBRA 0.5% (trained head). Identical
survival, completely different recommendations — because the recommendations were
never used.

That also means the trained head is heavily NORMAL-biased in this regime, which
follows from its supervision: over half the maneuver labels are the wall rule,
and the benchmark clamps the jet away from walls.

**F. None of the scripts seed their RNG.** `benchmark_v2.py`,
`headless_latency.py` and `headless_sim3.py` all call `random.*` without
`random.seed(...)`, so no run is reproducible and two runs of the benchmark
differ by a few percent on their own. Any before/after comparison needs a seed
first.

---

## Recommended order of work

1. **One-line fix, no retraining:** carry the real guidance type into the game's
   context feature (finding D). Free accuracy.
2. **No retraining:** seed the RNGs (finding F), then re-run `param_count.py`,
   `benchmark_v2.py` and `headless_latency.py` now that the maneuver head loads
   (finding C), and correct the maneuver table in `label_distribution_v2.txt`
   (finding B). Make the benchmark actually execute the maneuvers it recommends
   (finding E), or rename it so it is clear it benchmarks the danger head only.
3. **Retrain, no new data:** save the final / per-stage checkpoint instead of
   global-best-AUC (finding A). Likely the largest single win available.
4. **Regenerate data:** add jet throttle variation (§4). Without it the model
   cannot reason about the speed changes its own maneuvers make.
5. **Regenerate data:** give ballistic missiles lead prediction (§1) so the
   second threat type is actually a threat.
6. **Architecture, if you want the paper's claim to hold literally:** connect
   missiles to each other in the graph (§2), and advance the world state in the
   candidate search so it is a real rollout (§3c).
