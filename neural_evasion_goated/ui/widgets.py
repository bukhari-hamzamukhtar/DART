"""Menu widgets (main3.py section 6)."""

import math

import pygame

from .effects import draw_glow_text
from .fonts import font_btn


class NeonButton:
    """Hover-responsive neon button for menus."""

    def __init__(self, cx, cy, w, h, text, color_primary, color_glow):
        self.rect   = pygame.Rect(cx-w//2, cy-h//2, w, h)
        self.text   = text
        self.cp     = color_primary
        self.cg     = color_glow
        self.hovered = False
        self.pulse   = 0.0

    def update(self, mx, my):
        self.hovered = self.rect.collidepoint(mx, my)
        self.pulse   = (self.pulse + 0.08) % (math.pi*2)

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and
                event.button == 1 and self.rect.collidepoint(event.pos))

    def draw(self, surface):
        osc = 0.5 + 0.5*math.sin(self.pulse)

        # outer glow box (multiple layers)
        for expand in ([6,4,2] if self.hovered else [3,1]):
            r = self.rect.inflate(expand*2, expand*2)
            a = int(40*osc) if self.hovered else 20
            glow_s = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            gc = (*self.cg, a)
            pygame.draw.rect(glow_s, gc, (0,0,r.width,r.height), border_radius=8)
            surface.blit(glow_s, (r.x, r.y))

        # fill
        fill_col = tuple(min(255,int(c*0.25)) for c in self.cp) if not self.hovered else \
                   tuple(min(255,int(c*0.45)) for c in self.cp)
        pygame.draw.rect(surface, fill_col, self.rect, border_radius=6)

        # border
        b_alpha = int(180+75*osc) if self.hovered else 140
        border_s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(border_s, (*self.cp, b_alpha),
                         (0,0,self.rect.width,self.rect.height), 2, border_radius=6)
        surface.blit(border_s, self.rect.topleft)

        # text
        tc = (255,255,255) if self.hovered else self.cp
        draw_glow_text(surface, self.text, font_btn,
                       self.rect.centerx, self.rect.centery, tc, center=True, layers=2)

        # hover scanline overlay
        if self.hovered:
            hl = pygame.Surface((self.rect.width, self.rect.height//2), pygame.SRCALPHA)
            hl.fill((*self.cp, 15))
            surface.blit(hl, self.rect.topleft)
