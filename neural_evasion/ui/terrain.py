"""
Terrain and what it hides.

The radar shadow behind each peak is the region the jet's radar cannot see from
where it is right now. Missiles inside it are not on the jet's scope; the v3
tracker has to dead-reckon them, and v1/v2 simply do not know they are there.
"""

import math

import pygame

from ..config import HEIGHT, WIDTH

_SHADOW = {}
SHADOW_REACH = 1400.0


def _surface():
    s = _SHADOW.get('s')
    if s is None:
        s = _SHADOW['s'] = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return s


def draw_radar_shadows(surface, jet, terrain):
    """Darken everything the jet's radar cannot see past each peak."""
    if not terrain:
        return
    s = _surface()
    s.fill((0, 0, 0, 0))
    for ox, oy, r in terrain:
        dx, dy = ox - jet.x, oy - jet.y
        d = math.hypot(dx, dy)
        if d <= r + 1.0:
            continue
        base = math.atan2(dy, dx)
        half = math.asin(min(1.0, r / d))
        tan_len = math.sqrt(max(d * d - r * r, 0.0))
        pts = []
        for sgn in (-1.0, 1.0):
            a = base + sgn * half
            pts.append((jet.x + math.cos(a) * tan_len, jet.y + math.sin(a) * tan_len))
        far = [(jet.x + math.cos(base + sgn * half) * SHADOW_REACH,
                jet.y + math.sin(base + sgn * half) * SHADOW_REACH) for sgn in (1.0, -1.0)]
        pygame.draw.polygon(s, (0, 0, 0, 95), pts + far)
        pygame.draw.polygon(s, (60, 110, 90, 40), pts + far, 1)
    surface.blit(s, (0, 0))
