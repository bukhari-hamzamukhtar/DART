"""
Launcher for the live build (v3).

    python play.py

Start screen -> mission setup: pick a theatre (level 1 open sky, level 2 terrain
masking, level 3 pincer network), then pick which pilot flies the jet (V1, V2 or
V3). V1 and V2 are the frozen builds, flown unmodified inside the same arena;
`play_goated.py` and `play_v2.py` still launch them exactly as they were.

See neural_evasion/ARCHITECTURE_V3.md.
"""

from neural_evasion.app import main

if __name__ == "__main__":
    main()
