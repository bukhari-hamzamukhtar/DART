# Frozen snapshot — "goated" build

A byte copy of `neural_evasion/` at the point where it survived 13 of 15 matches
in real play, losing twice, both times to a wall rather than a missile.

Run it with `python play_goated.py` from the repo root. It is self-contained:
nothing else imports it, so changes to the live `neural_evasion/` package cannot
reach it.

What defines this build:

| setting | value |
|---|---|
| decision cadence | every frame, on the game thread |
| rollout horizon / step | 72 frames (1.2 s) / 1 (exact) |
| action grid | 12 headings x 4 throttles |
| `ROLLOUT_TOP_K` | 0 — the network does not re-rank candidates |
| `WALL_MARGIN` | 150 |
| scripted maneuvers | effectively never fire (`cornered` is strict) |

Known weakness: missiles can herd it into a corner. Zero missile hits observed;
every loss was a wall.
