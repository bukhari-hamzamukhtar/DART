# Neural Evasion — browser build

Live: **https://neural-evasion.vercel.app**

The whole v3 stack runs in the browser: the arena, the dynamic scene graph, the
trained temporal graph network and the graph world model. Nothing is simulated
on a server — the page you load plans 58 futures a frame on your own CPU.

Torch and numba cannot ship to a browser, so both were ported to plain
JavaScript against the Python sources:

| browser file | ported from |
|---|---|
| `js/config.js`, `js/geometry.js` | `config.py`, `sim/geometry.py` |
| `js/doctrine.js` | `sim/doctrine.py` (flight styles, seeker limits, terrain avoidance) |
| `js/entities.js`, `js/arena.js`, `js/levels.js` | `sim/entities.py`, `arena.py`, `levels.py` |
| `js/tracker.js`, `js/sceneGraph.js` | `sim/tracker.py`, `sim/scene_graph.py` |
| `js/net.js` | `model/temporal_gnn.py` (message passing, GRU memory, both heads) |
| `js/worldmodel.js` | `sim/worldmodel_v3.py` + `sim/rollout_kernel.py` |
| `js/planner.js`, `js/pilot.js` | `sim/planner_v3.py`, `pilots/graph_pilot.py` |
| `js/render.js`, `js/main.js` | `ui/`, `game.py` (canvas instead of pygame) |

## Is the port faithful?

Three checks, all runnable with Node:

```bash
node tests/net_parity.mjs                     # network vs the torch model (f16 weights)
node tests/exactness.mjs                      # world model vs the arena
node tests/match.mjs 3 1800 trained full      # full matches, survival + ms/frame
```

- **`net_parity.mjs`** replays real level-3 scene-graph ticks (dumped from Python
  with `tools/dump_parity.py` into `tests/parity.json`) through the JS network and compares every
  estimate with what torch produced, memory and datalink head included.
  float32 weights agree to 5e-7; the float16 weights this build ships agree to
  9e-4 on a probability.
- **`exactness.mjs`** is the browser twin of `tests/test_worldmodel_v3.py`: given
  true missile values, one rolled candidate reproduces the arena exactly —
  0.00e+0 px on the jet and on every missile, with the same frame each missile
  dies and the same flare destroyed on the same step.
- **`match.mjs`** plays full matches against a maximum-rate opponent. It lands
  where the Python benchmark does: level 2 survived, level 1 lost to a missile,
  level 3 lost on walls at ~20 s.

## What differs from the desktop build

- **Weights are float16** (234 KB instead of 458 KB), measured above.
- **Planner quality tiers.** Full plans 58 candidate manoeuvres every frame, as
  the desktop build does. Fast trims the throttle set to three (46 candidates).
  Lite also re-plans every second frame and flies the held plan in between. The
  page benchmarks one rollout at load and picks a default; the menu overrides it.
- **V1 and V2 are not here.** Their networks are a different architecture in the
  frozen packages; the browser build is the v3 demo.

## How it is deployed

`index.html` and the modules are static files. They are stored in Vercel Blob
and served from the project's own origin through the rewrites in `vercel.json`,
so the page has no cross-origin loads:

```bash
# re-upload a changed file (BLOB_READ_WRITE_TOKEN comes from the Vercel project)
curl -X PUT "https://blob.vercel-storage.com/js/main.js" \
  -H "authorization: Bearer $BLOB_READ_WRITE_TOKEN" \
  -H "x-api-version: 7" -H "x-content-type: text/javascript" \
  -H "x-add-random-suffix: 0" --data-binary @js/main.js
```

`assets/weights.b64.txt` is the float16 blob as base64 text, written by
`tools/export_f16.py` from `jet_brain_tgn.pt`; `js/net.js` decodes it at load.
`tools/export_weights.py` writes the float32 copy (`assets/brain_full.*`) the
parity test compares against.

## Running it locally

```bash
python -m http.server 8765        # from this directory
# then open http://127.0.0.1:8765/index.html
```
