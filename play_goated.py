"""
Launcher for the preserved build — the one that took only 2 losses in 15 matches,
both to walls.

    python play_goated.py     # the frozen snapshot
    python play.py            # the live version, which we keep changing

Nothing in this project imports neural_evasion_goated, so the snapshot cannot be
affected by work on the live package.
"""

from neural_evasion_goated.app import main

if __name__ == "__main__":
    main()
