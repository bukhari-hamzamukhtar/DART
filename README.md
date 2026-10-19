# DART: a jet that dodges missiles in real time

DART is a 2D game. You fire missiles at a jet, and an AI flies the jet. Everything runs live on
one laptop CPU (Intel i5-8350U). No GPU is needed to play.

The repo holds two generations of the AI.

| | when | how the jet decides | where |
|---|---|---|---|
| **Spring build** | Apr to May 2026 | a graph neural network with GRU attention scores 12 candidate headings | `main3.py`, `simulation/`, `colab/`, `paper/report.pdf` |
| **Current build (V3)** | Sep 2026 | a temporal graph network infers each missile's hidden settings, then a world model imagines 57 to 58 moves 1.2 s ahead before every frame | `neural_evasion/` |

The spring code exactly as it was is kept at the tag
[`spring-2026`](https://github.com/bukhari-hamzamukhtar/DART/tree/spring-2026).

## Play

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python play.py
```

Pick a level, then a pilot. V1 and V2 are frozen earlier builds flown inside the same arena, so
all three can be compared on the same missiles.

| level | what it adds |
|---|---|
| 1 Open sky | the original arena |
| 2 Terrain masking | peaks hide missiles from the jet's radar and block seekers; a missile that loses the jet flies to where it last saw it |
| 3 Pincer network | two batteries, west and east; salvo members share a datalink and lead the jet |

**Keys.** Levels 1 and 2: `H` missile, `C` cluster of five, `B` ballistic shot. Level 3: `H` / `C`
west battery, `K` / `M` east battery, `S` pincer salvo. `G` shows the scene graph. A briefing screen
shows the keys before each round.

## How V3 flies

The jet's radar only sees where each missile is and how fast it moves. How each missile is
guided (how hard it can turn, whether it flanks, weaves or shares a datalink) stays hidden.

1. **Scene graph.** Every 4 frames the scene becomes a graph. The jet, missile tracks, flares,
   peaks and launch stations are nodes. Relations such as "this seeker can see the jet" or "this
   peak blocks that missile's view" are edges. A missile keeps the same node from frame to frame,
   also while it hides behind a peak.
2. **Temporal graph network** (learned, 117k parameters). Every node keeps a GRU memory, and two
   rounds of message passing let missiles inform each other. It estimates each missile's hidden
   settings.
3. **World model.** With those estimates it imagines 57 to 58 candidate actions (heading,
   throttle, flare volley) 72 frames ahead. Its physics is hand-written and matches the arena
   exactly (`neural_evasion/tests/test_worldmodel_v3.py`). Learning supplies what the radar cannot
   see.
4. **Planner.** The jet flies the cheapest imagined future, and fires flares only when a volley
   makes that future cheaper.

The full design, and the papers each part follows, is in
[`neural_evasion/ARCHITECTURE_V3.md`](neural_evasion/ARCHITECTURE_V3.md).

## Results

16 matches per level per pilot, the same 16 seeds for every pilot, against a scripted opponent
firing at the maximum rate for 30 seconds.

| level | V1 | V2 | V3 | V3 given true missile settings |
|---|---|---|---|---|
| 1 Open sky | 0/16 | 10/16 | 10/16 | 16/16 |
| 2 Terrain masking | 0/16 | 0/16 | **15/16** | 16/16 |
| 3 Pincer network | 0/16 | 0/16 | 2/16 | 10/16 |

- On levels 2 and 3, V1 and V2 survive none of 32 matches and last 2 to 4 seconds. V3 survives 17
  and lasts 19 to 30 seconds.
- On level 1, V3 ties V2. Given true missile settings it survives every match, so its six level-1
  deaths are inference errors.
- Level 3 is still too hard for the planner. Even with true settings, flanking salvos herd the
  jet into walls.
- The GNN cuts the world model's prediction error 48 frames ahead on every level (level 1: 15 to
  19 px, against 50 px without it).

Ablations (no message passing, no memory, no GNN), prediction error and inference accuracy are in
section 6 of the architecture doc. The raw benchmark files are in `training_data_v3/`.

## The spring build, and two bugs found later

The spring build (`main3.py`) scores 12 candidate headings with a 198K-parameter GNN and GRU
attention, trained on 330K simulated frames over a 5-stage curriculum. In its 50-trial benchmark
it survived 84% of runs, against 34% for the same network without memory
([`paper/report.pdf`](paper/report.pdf)).

An audit in September ([`SIM_AUDIT.md`](SIM_AUDIT.md)) found two bugs in that code. Both are fixed
in `main3.py` here:

- The maneuver head was declared with three layers while the checkpoint stores two, so 630 weights
  were re-randomised on every launch and the recommended maneuver was untrained.
- Every homing missile was presented to the network as a ballistic one.

The spring report's numbers were measured before these fixes.

## Repo map

| path | what it is |
|---|---|
| `neural_evasion/` | V3, the live build |
| `neural_evasion_goated/`, `neural_evasion_v2/` | V1 and V2, frozen; nothing imports them, so work on V3 cannot change them |
| `play.py`, `play_goated.py`, `play_v2.py` | launchers |
| `jet_brain_v3.pt` | the spring network, used by V1 and V2 |
| `jet_brain_tgn*.pt`, `*.calib.json` | V3's temporal GNN and its two ablations, with their conformal margins |
| `main3.py`, `simulation/`, `colab/` | spring build: game, data generator, training notebook |
| `SIM_AUDIT.md`, `benchmark_v2.py`, `headless_latency.py`, `param_count.py` | the September audit of the spring build |
| `training_data_v3/` | benchmark and evaluation results (the training episodes are too big to commit) |
| `paper/` | spring report (IEEE format) and architecture figure |

## Test, train, benchmark

```bash
python neural_evasion/tests/run_all.py
python -m neural_evasion.training.generate --level 2 --episodes 40 --val 6
python -m neural_evasion.training.train --variant full
python -m neural_evasion.training.benchmark --levels 1 2 3 --pilots 1 2 3 3-oracle --seeds 16
```

## Limits

- The opponent is scripted. It fires at the maximum rate, but it does not adapt.
- The world model is exact because the simulator's rules are known. The learned part is each
  missile's hidden state.
- The network learned nothing about weaving: its swing estimate is no better than always
  answering 0.
- With 16 matches per cell, differences smaller than about 30 percentage points cannot be
  resolved.
- Seeker doctrine, datalinks and pincers are this simulator's rules. Nothing here claims anything
  about real missiles.
