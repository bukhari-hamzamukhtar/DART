"""
Menu and result screens (main3.py sections 7 and 8).

Both run their own event/render loop and return a routing string:
"play", "quit" (and, from the match loop, "menu").
"""

import math

import pygame

from ..config import FPS, HEIGHT, JET_COLOR, MISSILE_COLOR, WIDTH
from ..display import clock, screen
from .effects import (draw_bg_particles, draw_glow_text, draw_radar_bg,
                      draw_scanlines, update_bg_particles)
from .fonts import (font_btn, font_hud, font_result, font_sm, font_sub,
                    font_tiny, font_title)
from .widgets import NeonButton

CORNER_BRACKETS = [
    ((10,10),(40,10)),                     ((10,10),(10,40)),
    ((WIDTH-10,10),(WIDTH-40,10)),         ((WIDTH-10,10),(WIDTH-10,40)),
    ((10,HEIGHT-10),(40,HEIGHT-10)),       ((10,HEIGHT-10),(10,HEIGHT-40)),
    ((WIDTH-10,HEIGHT-10),(WIDTH-40,HEIGHT-10)),
    ((WIDTH-10,HEIGHT-10),(WIDTH-10,HEIGHT-40)),
]

CONTROL_LEGEND = [
    ("H", "Homing missile",        JET_COLOR),
    ("B", "Ballistic (predictive)", (255,100,255)),
    ("C", "Cluster swarm  (6s cd)", MISSILE_COLOR),
]

GAMEOVER_LOCKOUT_MS = 800   # ignore clicks briefly so a stray click can't skip


def _draw_corner_brackets(surface):
    for (x1,y1),(x2,y2) in CORNER_BRACKETS:
        pygame.draw.line(surface,(0,180,140),(x1,y1),(x2,y2),2)


def run_start_screen():
    btn_play = NeonButton(WIDTH//2, HEIGHT//2+30,  220, 52, "[ LAUNCH MISSION ]", JET_COLOR,     JET_COLOR)
    btn_quit = NeonButton(WIDTH//2, HEIGHT//2+100, 220, 52, "[ ABORT ]",          MISSILE_COLOR, MISSILE_COLOR)
    radar_a  = 0.0
    glow_s   = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tick_sin = 0.0

    while True:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if btn_play.is_clicked(event): return "play"
            if btn_quit.is_clicked(event): return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return "play"

        # Update
        radar_a  = (radar_a + 0.02) % (math.pi*2)
        tick_sin = (tick_sin + 0.05) % (math.pi*2)
        update_bg_particles()
        btn_play.update(mx, my)
        btn_quit.update(mx, my)

        # Draw base
        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0,0))
        draw_bg_particles(screen)
        _draw_corner_brackets(screen)

        # Title pulse
        title_y   = HEIGHT//2 - 120
        pulse_off = int(math.sin(tick_sin)*3)
        draw_glow_text(screen, "NEURAL", font_title, WIDTH//2, title_y+pulse_off,
                       JET_COLOR, center=True, layers=4)
        draw_glow_text(screen, "EVASION", font_title, WIDTH//2, title_y+78+pulse_off,
                       (0,200,160), center=True, layers=4)

        # Subtitle
        sub_a = int(160+95*math.sin(tick_sin))
        sub_s = font_sub.render("HIERARCHICAL GNN  ·  CROSS-THREAT ATTENTION", True, (0,180,140))
        sub_s.set_alpha(sub_a)
        screen.blit(sub_s, (WIDTH//2-sub_s.get_width()//2, title_y+155))

        # Divider line
        div_w = int(300+80*math.sin(tick_sin*0.7))
        pygame.draw.line(screen, (0,140,110), (WIDTH//2-div_w//2, title_y+178),
                         (WIDTH//2+div_w//2, title_y+178), 1)

        # Controls legend below buttons
        leg_y = HEIGHT//2 + 165
        for i,(key,desc,col) in enumerate(CONTROL_LEGEND):
            ks  = font_btn.render(" %s " % key, True, (10,20,30))
            bg  = pygame.Surface((ks.get_width(), ks.get_height()), pygame.SRCALPHA)
            bg.fill((*col,220))
            screen.blit(bg, (WIDTH//2-140, leg_y+i*22))
            screen.blit(ks, (WIDTH//2-140, leg_y+i*22))
            ds = font_sm.render(desc, True, (100,160,140))
            screen.blit(ds, (WIDTH//2-140+ks.get_width()+8, leg_y+i*22+2))

        # Version tag
        vs = font_tiny.render("v2.0  |  GNN + ATTENTION  |  5 MANEUVERS", True, (30,70,60))
        screen.blit(vs, (WIDTH//2-vs.get_width()//2, HEIGHT-22))

        # Buttons
        btn_play.draw(screen)
        btn_quit.draw(screen)

        # Scanlines on top
        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)


def run_gameover_screen(user_won, stats):
    btn_again = NeonButton(WIDTH//2, HEIGHT//2+60,  220, 52, "[ PLAY AGAIN ]", JET_COLOR,     JET_COLOR)
    btn_exit  = NeonButton(WIDTH//2, HEIGHT//2+125, 220, 52, "[ EXIT ]",       MISSILE_COLOR, MISSILE_COLOR)
    radar_a   = 0.0
    glow_s    = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tick_sin  = 0.0
    enter_t   = pygame.time.get_ticks()

    if user_won:
        headline = "JET DESTROYED"
        sub_line = "YOU WIN — AI ELIMINATED"
        h_color  = MISSILE_COLOR
        s_color  = (200, 80, 80)
    else:
        headline = "BASES OBLITERATED"
        sub_line = "AI WINS — HUMANITY FALLS"
        h_color  = JET_COLOR
        s_color  = (0, 180, 140)

    while True:
        mx, my = pygame.mouse.get_pos()
        elapsed = pygame.time.get_ticks() - enter_t

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if elapsed > GAMEOVER_LOCKOUT_MS:
                if btn_again.is_clicked(event): return "play"
                if btn_exit.is_clicked(event):  return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and elapsed > GAMEOVER_LOCKOUT_MS: return "play"
                if event.key == pygame.K_ESCAPE: return "quit"

        radar_a  = (radar_a + 0.025) % (math.pi*2)
        tick_sin = (tick_sin + 0.06) % (math.pi*2)
        update_bg_particles()
        btn_again.update(mx, my)
        btn_exit.update(mx, my)

        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0,0))
        draw_bg_particles(screen)

        # Dark overlay
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        a_ov = min(160, int((elapsed/600)*160))
        ov.fill((0,0,0,a_ov))
        screen.blit(ov, (0,0))

        # Headline
        pulse = int(math.sin(tick_sin)*4)
        draw_glow_text(screen, headline, font_result, WIDTH//2, HEIGHT//2-90+pulse,
                       h_color, center=True, layers=4)
        draw_glow_text(screen, sub_line, font_sub, WIDTH//2, HEIGHT//2-28,
                       s_color, center=True, layers=2)

        # Stats panel
        stat_y = HEIGHT//2 + 5
        stat_entries = [
            ("MISSILES FIRED : %d" % stats.get('fired',0),     (80,160,140)),
            ("FLARES DEPLOYED: %d" % stats.get('flares',0),    (80,130,100)),
            ("MANEUVERS USED : %d" % stats.get('maneuvers',0), (60,100,90)),
        ]
        for i,(txt,col) in enumerate(stat_entries):
            ss = font_hud.render(txt, True, col)
            screen.blit(ss, (WIDTH//2-ss.get_width()//2, stat_y+i*18))

        # Hint
        if elapsed > 900:
            hint_a = int(80+80*math.sin(tick_sin))
            hs = font_sm.render("R — Play Again  |  ESC — Exit", True, (60,100,90))
            hs.set_alpha(hint_a)
            screen.blit(hs, (WIDTH//2-hs.get_width()//2, HEIGHT-22))

        if elapsed > GAMEOVER_LOCKOUT_MS:
            btn_again.draw(screen)
            btn_exit.draw(screen)

        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)
