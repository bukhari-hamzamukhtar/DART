"""
Shared constants for the Neural Evasion simulator.

Every value here is a verbatim copy of the literal that lived at the top of
main3.py (section 4) so the modular build renders and behaves identically.
"""

import os

# ── Arena ─────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 600
HUD_HEIGHT    = 50          # bottom bar; playfield is HEIGHT - HUD_HEIGHT tall
FPS           = 60

# ── Palette ───────────────────────────────────────────────────
BG_COLOR        = (5, 10, 20)
JET_COLOR       = (0, 255, 200)
MISSILE_COLOR   = (255, 68, 68)
FLARE_COLOR     = (255, 176, 32)
BALLISTIC_COLOR = (255, 100, 255)

# ── Model ─────────────────────────────────────────────────────
MANEUVER_NAMES = ["NORMAL", "BARREL_ROLL", "JINKING", "FALLING_LEAF", "COBRA", "IMMELMANN"]

CHECKPOINT_NAME = "jet_brain_v3.pt"
# Repo root = parent of the neural_evasion package.
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW_CAPTION = "Neural Evasion — Tactical AI 1v1"

# ── Guidance types (must match the training data's missile "type" field) ──
GUIDANCE_HOMING    = 0
GUIDANCE_BALLISTIC = 1
GUIDANCE_SMART     = 2      # seeker + datalink missile, levels 2 and 3 only.
                            # v1/v2's network was never trained on this value,
                            # so their adapters present it to them as homing.

# ── Frame-based lifetimes ─────────────────────────────────────
# The arena ages everything in frames, so a headless run and a 60 fps window
# see identical timings. Each is the old wall-clock value at 60 fps.
FLARE_LIFE_FRAMES        = 120    # 2000 ms
HOMING_LIFE_FRAMES       = 510    # 8500 ms
BALLISTIC_LIFE_FRAMES    = 180    # 3000 ms
JET_MISSILE_LIFE_FRAMES  = 240    # 4000 ms
LAUNCHER_COOLDOWN_FRAMES = 120    # 2000 ms
CLUSTER_COOLDOWN_FRAMES  = 360    # 6 s
PINCER_COOLDOWN_FRAMES   = 420    # 7 s
JET_ATTACK_FRAMES        = 240    # 4000 ms between the jet's counter-attacks

# ── Flare magazine ────────────────────────────────────────────
# Flares are not free. Every life starts with an empty magazine, and the jet
# earns one volley each time the enemy launches a cluster.
FLARE_START_CHARGES       = 0
FLARE_CHARGES_PER_CLUSTER = 1

# ── v3 temporal graph brain ───────────────────────────────────
GRAPH_TICK = 4      # frames between scene-graph snapshots (15 Hz). The planner
                    # still re-decides every frame; this is how often the
                    # network refreshes its read on each missile's hidden state.

TERRAIN_FILL    = (14, 26, 22)
TERRAIN_CONTOUR = (46, 96, 70)

# ── Jet throttle envelope ─────────────────────────────────────
# The jet commands its own speed. BASE_SPEED is the old fixed 5.0.
BASE_SPEED   = 5.0
MIN_THROTTLE = 0.3    # the COBRA brakes this hard
MAX_THROTTLE = 7.5

# Throttle settings the planner may choose from, as multiples of BASE_SPEED.
THROTTLE_CHOICES = [0.35, 0.70, 1.00, 1.50]

# ── World-model rollout ───────────────────────────────────────
ROLLOUT_HORIZON = 72    # frames simulated per candidate (1.2 s at 60 fps)
ROLLOUT_TOP_K   = 0     # 0 = the network does not re-rank candidates.
                        # It is worth ~2/8 survival but costs ~30 ms, and the
                        # resulting staleness costs much more than that.
