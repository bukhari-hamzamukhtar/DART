"""Top-level routing: start screen -> mission setup -> briefing -> match -> result."""

import pygame

from .display import clock, screen
from .levels import LEVELS
from .ui.fonts import font_sub, font_tiny


def _loading(text):
    screen.fill((4, 8, 16))
    s = font_sub.render(text, True, (0, 200, 160))
    screen.blit(s, (screen.get_width() // 2 - s.get_width() // 2, screen.get_height() // 2 - 10))
    pygame.display.flip()
    pygame.event.pump()


def warm_up():
    """Load the v3 brain and compile (or load from cache) the world-model kernels."""
    import torch
    torch.set_num_threads(1)
    from .arena import Arena
    from .pilots.graph_pilot import shared_model
    from .sim.planner_v3 import plan_v3
    shared_model()
    for level in LEVELS:
        a = Arena(level, visuals=False)
        a.fire(0)
        from .sim.worldmodel_v3 import oracle_threat
        jet = {'pos': [a.jet.x, a.jet.y], 'vel': [0.0, 0.0], 'heading': 0.0}
        plan_v3(jet, [oracle_threat(m, level.datalink) for m in a.missiles], level, keep_future=True)


def main():
    _loading("COMPILING WORLD MODEL ...")
    warm_up()
    from .game import Match
    from .ui.screens import run_briefing_screen, run_setup_screen, run_start_screen

    choice = {'level': 0, 'version': 3}
    route = run_start_screen()
    while True:
        if route == "quit":
            return route
        if route in ("play", "setup"):
            route = run_setup_screen(choice)
            if route == "back":
                route = run_start_screen()
            elif route == "start":
                # The launch keys and the rules, just before the round starts.
                route = run_briefing_screen(LEVELS[choice['level']], choice['version'])
                if route == "back":
                    route = "setup"
            continue
        if route in ("start", "again"):
            route = Match(LEVELS[choice['level']], choice['version']).run()
            continue
        route = run_setup_screen(choice)


if __name__ == "__main__":
    main()
