# v2 — interruptible maneuvers

Frozen copy of `neural_evasion/` after maneuvers became per-frame candidate
actions instead of 35-120 frame scripts, and the network's maneuver head was
given a say in action selection (proposes; the rollout can veto).

Measured by playing 5 real matches at maximum fire rate: **0 kills against it**,
one match where the jet destroyed both launchers, ~58 fps. The v1 build lost one
of the same 5.

Run with `python play_v2.py`.

## The baseline this sets for v3

| | v1 (goated) | v2 |
|---|---|---|
| kills against it (5 matches, max pressure) | 1 | **0** |
| avg match length | 47.0 s | 58.5 s |
| threat model | per-missile star subgraphs, no missile-missile edges |
| recurrence | GRU per list index, resets when count changes |
| world model | analytic, hand-written dynamics |
| planner | 48-action MPC, 72-frame horizon, every frame |
