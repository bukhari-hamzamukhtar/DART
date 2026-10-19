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
