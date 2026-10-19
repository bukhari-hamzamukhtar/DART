"""Top-level routing between the menu, a match and the result screen."""

from .model import get_brain
from .sim.inference import drain_queues
from .ui.screens import run_start_screen


def main():
    get_brain()                 # load weights up front, not on the first frame

    from .game import run_game  # imported late so the window exists first

    result = run_start_screen()
    while result == "play":
        drain_queues()          # fresh queues for each game session
        result = run_game()
    return result


if __name__ == "__main__":
    main()
