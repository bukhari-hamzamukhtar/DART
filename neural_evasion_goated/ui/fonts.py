"""
Fonts, loaded once (main3.py section 6).

Importing `..display` first guarantees `pygame.init()` has run before any
SysFont call, which is what made the original module-level block work.
"""

import pygame

from .. import display  # noqa: F401  (imported for its pygame.init() side effect)

try:
    font_title  = pygame.font.SysFont("impact", 72)
    font_sub    = pygame.font.SysFont("courier", 18, bold=True)
    font_hud    = pygame.font.SysFont("courier", 13, bold=True)
    font_sm     = pygame.font.SysFont("courier", 11)
    font_tiny   = pygame.font.SysFont("courier", 10)
    font_btn    = pygame.font.SysFont("courier", 16, bold=True)
    font_result = pygame.font.SysFont("impact", 56)
except Exception:
    font_title = font_sub = font_hud = font_sm = font_tiny = font_btn = font_result = \
        pygame.font.SysFont(None, 32)
