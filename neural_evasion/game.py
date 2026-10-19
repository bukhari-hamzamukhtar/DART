"""
One match: a theatre, a pilot version, and a human on the launch keys.

    begin_frame -> keys launch missiles -> arena.advance(pilot) -> render

The arena owns the world and its rules (`arena.py`), the pilot owns the jet's
brain (`pilots/`), and this file owns the window: input, drawing and the result
screen. Every version flies through exactly the same code path here.

Launch keys come from the level: each battery has a key for one missile, and a
homing battery a second key for a cluster (level 3: H/C west, K/M east).
"""

import math

import pygame

from .arena import Arena
from .config import FLARE_COLOR, FPS, HEIGHT, HUD_HEIGHT, WIDTH
from .display import clock, screen
from .pilots import VERSIONS, make_pilot
from .ui.effects import draw_radar_bg, draw_scanlines
from .ui.fonts import font_tiny
from .ui.screens import run_gameover_screen
from .ui.terrain import draw_radar_shadows

GAMEOVER_HOLD_MS = 2200
MAGAZINE_POS = (14, 134)       # under the heading-risk chart, top left


def key_code(name):
    """The pygame key constant for a single-letter key name such as "H"."""
    return getattr(pygame, "K_" + name.lower())


def launch_keymap(level):
    """{pygame key: (action, station index)} for this level's batteries."""
    keys = {}
    for i, st in enumerate(level.stations):
        keys[key_code(st.key)] = ("fire", i)
        if st.cluster_key:
            keys[key_code(st.cluster_key)] = ("cluster", i)
    if level.pincer:
        keys[pygame.K_s] = ("pincer", None)
    return keys


def draw_magazine(surface, arena):
    """The jet's flare magazine: one pip per volley it holds."""
    x, y = MAGAZINE_POS
    n = arena.flare_charges
    surface.blit(font_tiny.render("FLARES", True, (0, 160, 120)), (x, y))
    px = x + 48
    if n == 0:
        surface.blit(font_tiny.render("empty  (+1 per cluster)", True, (120, 100, 70)), (px, y))
        return
    for k in range(min(n, 8)):
        pygame.draw.rect(surface, FLARE_COLOR, (px + k * 9, y + 2, 6, 9), 0, 2)
    if n > 8:
        surface.blit(font_tiny.render("+%d" % (n - 8), True, FLARE_COLOR), (px + 8 * 9 + 2, y))


class Match(object):
    def __init__(self, level, version):
        self.level = level
        self.version = version
        self.arena = Arena(level)
        self.pilot = make_pilot(version, level, threaded=True)
        self.keymap = launch_keymap(level)
        self.glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.dim = pygame.Surface((WIDTH, HEIGHT - HUD_HEIGHT), pygame.SRCALPHA)
        self.dim.fill((0, 0, 0, 120))
        self.radar_angle = 0.0
        self.show_overlay = version == 3
        self.game_over_time = None
        self.tag = font_tiny.render("L%d %s  ·  %s" % (level.number, level.name,
                                                       VERSIONS[version - 1].name),
                                    True, (60, 110, 100))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return "setup"
            if event.key == pygame.K_g:
                self.show_overlay = not self.show_overlay
                continue
            a = self.arena
            if a.game_over:
                continue
            action = self.keymap.get(event.key)
            if action is None:
                continue
            kind, i = action
            if kind == "fire":
                a.fire(i)
            elif kind == "cluster":
                a.launch_cluster(i)
            else:
                a.launch_pincer()
        return None

    def render(self):
        a = self.arena
        g = self.glow_surface
        self.radar_angle = (self.radar_angle + 0.025) % (math.pi * 2)
        draw_radar_bg(screen, self.radar_angle, g)
        screen.blit(g, (0, 0))
        if a.terrain:
            draw_radar_shadows(screen, a.jet, a.terrain)
            for o in a.obstacles:
                o.draw(screen)
        for l in a.launchers:
            l.draw(screen)
        if not a.game_over:
            a.jet.draw(g, self.pilot.maneuver_label)
        for f in a.flares:
            f.draw(g)
        for m in a.missiles:
            m.draw(g)
        for aim in a.ai_missiles:
            aim.draw(g)
        for e in a.explosions:
            e.draw(screen)
        for s in a.shockwaves:
            s.update()
            s.draw(g)
        a.shockwaves[:] = [s for s in a.shockwaves if not s.is_dead()]
        screen.blit(g, (0, 0))
        self.pilot.draw_hud(screen, a, self.show_overlay)
        draw_magazine(screen, a)
        screen.blit(self.tag, (WIDTH // 2 - self.tag.get_width() // 2, HEIGHT - HUD_HEIGHT - 14))

    def run(self):
        self.pilot.start(self.arena)
        try:
            while True:
                if not self.arena.game_over:
                    self.arena.begin_frame()
                route = self.handle_events()
                if route is not None:
                    return route
                if not self.arena.game_over:
                    self.arena.advance(self.pilot)

                self.render()

                if self.arena.game_over:
                    if self.game_over_time is None:
                        self.game_over_time = pygame.time.get_ticks()
                        self.pilot.stop()
                    screen.blit(self.dim, (0, 0))
                    if pygame.time.get_ticks() - self.game_over_time > GAMEOVER_HOLD_MS:
                        a = self.arena
                        return run_gameover_screen(a.user_won, a.stats, {
                            'level': "L%d %s" % (self.level.number, self.level.name),
                            'version': VERSIONS[self.version - 1].name,
                            'cause': a.cause,
                            'seconds': (a.end_frame or a.frame) / 60.0,
                        })

                draw_scanlines(screen)
                pygame.display.flip()
                clock.tick(FPS)
        finally:
            self.pilot.stop()


def run_game(level=None, version=3):
    from .levels import LEVEL_1
    return Match(level or LEVEL_1, version).run()
