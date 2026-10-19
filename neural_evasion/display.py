"""
Pygame display bootstrap.

Importing this module initialises pygame and opens the window, exactly like the
module-level `pygame.init()` block in main3.py section 4.  Everything that draws
imports `screen` / `clock` from here, and `ui.fonts` imports it first so that
`pygame.font` is ready before any SysFont call.
"""

import pygame

from .config import WIDTH, HEIGHT, WINDOW_CAPTION

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(WINDOW_CAPTION)
clock = pygame.time.Clock()
