"""
Reusable visual effects (main3.py section 6).

The radar backdrop, drifting particle field, glow text and CRT scanlines are
shared by the menu, the game-over screen and the match itself.
"""

import math
import random

import pygame

from ..config import HEIGHT, HUD_HEIGHT, WIDTH

BG_PARTICLE_COUNT = 120
RADAR_RINGS       = [60, 120, 180, 240, 300]
RADAR_REACH       = 340


class BgParticle:
    """One drifting speck in the shared background field."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.speed = random.uniform(0.2, 0.8)
        self.angle = random.uniform(0, math.pi*2)
        self.alpha = random.randint(30, 120)
        self.size  = random.choice([1,1,1,2])
        self.color = random.choice([(0,255,200),(0,180,140),(255,68,68),(255,176,32)])

    def update(self):
        self.x += math.cos(self.angle)*self.speed
        self.y += math.sin(self.angle)*self.speed
        if self.x<-5 or self.x>WIDTH+5 or self.y<-5 or self.y>HEIGHT+5:
            self.reset()


BG_PARTICLES = [BgParticle() for _ in range(BG_PARTICLE_COUNT)]


def update_bg_particles():
    for p in BG_PARTICLES:
        p.update()


def draw_bg_particles(surface):
    for p in BG_PARTICLES:
        ps = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
        pygame.draw.circle(ps, (*p.color, p.alpha), (p.size, p.size), p.size)
        surface.blit(ps, (int(p.x), int(p.y)))


def draw_radar_bg(surface, radar_angle, glow_surf):
    """Fill `surface` with the radar grid and paint the sweep into `glow_surf`."""
    surface.fill((4, 8, 16))
    glow_surf.fill((0,0,0,0))
    cx, cy = WIDTH//2, (HEIGHT-HUD_HEIGHT)//2
    # Rings
    for r in RADAR_RINGS:
        pygame.draw.circle(surface,(0,40,60),(cx,cy),r,1)
    # Radials
    for i in range(12):
        a = i*math.pi/6
        ex,ey = cx+math.cos(a)*RADAR_REACH, cy+math.sin(a)*RADAR_REACH
        pygame.draw.line(surface,(0,30,50),(cx,cy),(int(ex),int(ey)),1)
    # Sweep
    for off in range(0,40,4):
        aa = radar_angle - math.radians(off)
        sa = (0, max(0,200-off*5), max(0,160-off*4), max(0,180-off*5))
        ex,ey = cx+math.cos(aa)*RADAR_REACH, cy+math.sin(aa)*RADAR_REACH
        pygame.draw.line(glow_surf, sa, (cx,cy),(int(ex),int(ey)),2)


def draw_glow_text(surface, text, font, x, y, color, center=True, layers=3):
    """Draw text with a multi-layer glow effect."""
    for i in range(layers, 0, -1):
        alpha = int(60 / i)
        # Fake glow: render at same pos but with dim color
        glow_c = tuple(min(255, int(c * 0.6)) for c in color)
        surf = font.render(text, True, glow_c)
        if center:
            rx, ry = x - surf.get_width()//2, y - surf.get_height()//2
        else:
            rx, ry = x, y
        # Offset copies for bloom
        for dx, dy in [(-i,0),(i,0),(0,-i),(0,i)]:
            surf2 = font.render(text, True, glow_c)
            surf2.set_alpha(alpha)
            surface.blit(surf2, (rx+dx, ry+dy))
    # Sharp center
    sharp = font.render(text, True, color)
    if center:
        surface.blit(sharp, (x-sharp.get_width()//2, y-sharp.get_height()//2))
    else:
        surface.blit(sharp, (x, y))


_SCANLINE_CACHE = {}


def draw_scanlines(surface, alpha=18):
    """
    Overlay subtle scanlines for CRT feel.

    The overlay never changes, so it is rendered once and blitted after that.
    Drawing the 200 lines every frame cost 3.6 ms of the 16.7 ms frame budget
    and — because it is 200 Python-level calls — held the GIL against the
    inference thread the whole time.
    """
    surf = _SCANLINE_CACHE.get(alpha)
    if surf is None:
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 3):
            pygame.draw.line(surf, (0, 0, 0, alpha), (0, y), (WIDTH, y), 1)
        _SCANLINE_CACHE[alpha] = surf
    surface.blit(surf, (0, 0))
