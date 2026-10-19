"""
Menu, mission setup, briefing and result screens.

Each runs its own event/render loop and returns a routing string:

  run_start_screen     "play" | "quit"
  run_setup_screen     "start" | "back" | "quit"   (mutates the choice dict)
  run_briefing_screen  "start" | "back" | "quit"   (launch keys and rules, just before a round)
  run_gameover_screen  "again" | "setup" | "quit"
"""

import math

import pygame

from ..config import (FLARE_COLOR, FPS, HEIGHT, HUD_HEIGHT, JET_COLOR,
                      MISSILE_COLOR, WIDTH)
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
    ("1-3", "choose theatre",         JET_COLOR),
    ("<>",  "choose pilot version",   (255,100,255)),
    ("G",   "toggle graph overlay",   MISSILE_COLOR),
]

STYLE_NOTES = [
    ("PURSUIT", "straight at the jet",           (255, 90, 90)),
    ("FLANKER", "swings wide, hits from a side", (255, 170, 60)),
    ("WEAVER",  "swings left and right",         (120, 200, 255)),
    ("ERRATIC", "darts to random sides",         (220, 120, 255)),
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
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "quit"

        radar_a  = (radar_a + 0.02) % (math.pi*2)
        tick_sin = (tick_sin + 0.05) % (math.pi*2)
        update_bg_particles()
        btn_play.update(mx, my)
        btn_quit.update(mx, my)

        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0,0))
        draw_bg_particles(screen)
        _draw_corner_brackets(screen)

        title_y   = HEIGHT//2 - 120
        pulse_off = int(math.sin(tick_sin)*3)
        draw_glow_text(screen, "NEURAL", font_title, WIDTH//2, title_y+pulse_off,
                       JET_COLOR, center=True, layers=4)
        draw_glow_text(screen, "EVASION", font_title, WIDTH//2, title_y+78+pulse_off,
                       (0,200,160), center=True, layers=4)

        sub_a = int(160+95*math.sin(tick_sin))
        sub_s = font_sub.render("DYNAMIC SCENE GRAPH  ·  TEMPORAL GNN  ·  WORLD MODEL", True, (0,180,140))
        sub_s.set_alpha(sub_a)
        screen.blit(sub_s, (WIDTH//2-sub_s.get_width()//2, title_y+155))

        div_w = int(300+80*math.sin(tick_sin*0.7))
        pygame.draw.line(screen, (0,140,110), (WIDTH//2-div_w//2, title_y+178),
                         (WIDTH//2+div_w//2, title_y+178), 1)

        leg_y = HEIGHT//2 + 165
        for i,(key,desc,col) in enumerate(CONTROL_LEGEND):
            ks  = font_btn.render(" %s " % key, True, (10,20,30))
            bg  = pygame.Surface((ks.get_width(), ks.get_height()), pygame.SRCALPHA)
            bg.fill((*col,220))
            screen.blit(bg, (WIDTH//2-140, leg_y+i*22))
            screen.blit(ks, (WIDTH//2-140, leg_y+i*22))
            ds = font_sm.render(desc, True, (100,160,140))
            screen.blit(ds, (WIDTH//2-140+ks.get_width()+8, leg_y+i*22+2))

        vs = font_tiny.render("v3.0  |  3 THEATRES  |  3 PILOT VERSIONS", True, (30,70,60))
        screen.blit(vs, (WIDTH//2-vs.get_width()//2, HEIGHT-22))

        btn_play.draw(screen)
        btn_quit.draw(screen)
        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)


# ══════════════════════════════════════════════════════════════
#  Mission setup: theatre, then pilot version
# ══════════════════════════════════════════════════════════════
LEVEL_CARD_W, LEVEL_CARD_H = 240, 178
PILOT_CARD_W, PILOT_CARD_H = 240, 92


def _card(surface, rect, selected, hovered, color):
    fill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    fill.fill((12, 30, 40, 235) if selected else (6, 14, 24, 215))
    surface.blit(fill, rect.topleft)
    border = color if selected else ((70, 120, 110) if hovered else (30, 60, 60))
    pygame.draw.rect(surface, border, rect, 2 if selected else 1, 8)
    if selected:
        glow = rect.inflate(8, 8)
        pygame.draw.rect(surface, tuple(int(c * 0.35) for c in color), glow, 1, 10)


def _minimap(surface, level, rect):
    """A tiny plan view of the theatre: terrain and batteries."""
    pygame.draw.rect(surface, (4, 10, 18), rect)
    pygame.draw.rect(surface, (30, 60, 60), rect, 1)
    sx = rect.width / float(WIDTH)
    sy = rect.height / float(HEIGHT - HUD_HEIGHT)
    for ox, oy, r in level.obstacles:
        pygame.draw.circle(surface, (46, 96, 70), (int(rect.x + ox * sx), int(rect.y + oy * sy)),
                           max(2, int(r * sx)))
    for st in level.stations:
        pygame.draw.rect(surface, st.color, (int(rect.x + st.x * sx) - 3, rect.bottom - 5, 6, 4))
    pygame.draw.circle(surface, JET_COLOR, (rect.centerx, int(rect.y + (HEIGHT // 2) * sy)), 2)


def run_setup_screen(choice):
    """
    `choice` is {'level': index, 'version': 1|2|3}; updated in place.
    """
    from ..levels import LEVELS
    from ..pilots import VERSIONS
    from ..model.temporal_gnn import checkpoint_path

    has_v3_weights = checkpoint_path() is not None
    btn_start = NeonButton(WIDTH // 2, HEIGHT - 52, 240, 46, "[ LAUNCH ]", JET_COLOR, JET_COLOR)
    radar_a, tick = 0.0, 0.0
    glow_s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    gap = (WIDTH - 3 * LEVEL_CARD_W) // 4
    level_rects = [pygame.Rect(gap + i * (LEVEL_CARD_W + gap), 92, LEVEL_CARD_W, LEVEL_CARD_H)
                   for i in range(3)]
    pilot_rects = [pygame.Rect(gap + i * (PILOT_CARD_W + gap), 318, PILOT_CARD_W, PILOT_CARD_H)
                   for i in range(3)]

    while True:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if btn_start.is_clicked(event):
                return "start"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(level_rects):
                    if r.collidepoint(event.pos):
                        choice['level'] = i
                for i, r in enumerate(pilot_rects):
                    if r.collidepoint(event.pos):
                        choice['version'] = VERSIONS[i].number
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "back"
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return "start"
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    choice['level'] = event.key - pygame.K_1
                if event.key in (pygame.K_RIGHT, pygame.K_v, pygame.K_TAB):
                    choice['version'] = choice['version'] % 3 + 1
                if event.key == pygame.K_LEFT:
                    choice['version'] = (choice['version'] - 2) % 3 + 1

        radar_a = (radar_a + 0.02) % (math.pi * 2)
        tick = (tick + 0.05) % (math.pi * 2)
        update_bg_particles()
        btn_start.update(mx, my)

        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0, 0))
        draw_bg_particles(screen)
        _draw_corner_brackets(screen)

        draw_glow_text(screen, "MISSION SETUP", font_sub, WIDTH // 2, 30, JET_COLOR, layers=2)
        screen.blit(font_hud.render("1  THEATRE", True, (0, 200, 160)), (gap, 70))
        screen.blit(font_hud.render("2  PILOT", True, (0, 200, 160)), (gap, 296))

        for i, (lv, r) in enumerate(zip(LEVELS, level_rects)):
            sel = choice['level'] == i
            _card(screen, r, sel, r.collidepoint(mx, my), JET_COLOR)
            screen.blit(font_tiny.render("LEVEL %d" % lv.number, True, (90, 150, 130)), (r.x + 10, r.y + 8))
            screen.blit(font_btn.render(lv.name, True, JET_COLOR if sel else (150, 200, 190)),
                        (r.x + 10, r.y + 22))
            screen.blit(font_sm.render(lv.tagline, True, (110, 170, 150)), (r.x + 10, r.y + 44))
            for j, line in enumerate(lv.blurb):
                screen.blit(font_tiny.render(line, True, (90, 130, 120)), (r.x + 10, r.y + 62 + j * 13))
            _minimap(screen, lv, pygame.Rect(r.x + 10, r.y + 94, r.width - 20, 58))
            keys = "keys  " + "  ".join(k for k, _ in lv.legend)
            screen.blit(font_tiny.render(keys, True, (70, 110, 100)), (r.x + 10, r.y + 158))

        for i, (v, r) in enumerate(zip(VERSIONS, pilot_rects)):
            sel = choice['version'] == v.number
            _card(screen, r, sel, r.collidepoint(mx, my), (255, 100, 255))
            screen.blit(font_btn.render(v.name, True, (255, 140, 255) if sel else (190, 160, 190)),
                        (r.x + 10, r.y + 8))
            screen.blit(font_sm.render(v.summary, True, (150, 170, 160)), (r.x + 10, r.y + 32))
            screen.blit(font_tiny.render(v.knows, True, (100, 130, 120)), (r.x + 10, r.y + 52))
            if v.number == 3 and not has_v3_weights:
                screen.blit(font_tiny.render("weights missing: prior estimates", True, MISSILE_COLOR),
                            (r.x + 10, r.y + 70))

        lv = LEVELS[choice['level']]
        hint = "  ".join("[%s]" % k for k, _ in lv.legend) + "   launch keys  ·  full briefing next"
        hs = font_sm.render(hint, True, (100, 160, 140))
        screen.blit(hs, (WIDTH // 2 - hs.get_width() // 2, 432))
        foot = font_tiny.render("1-3 theatre   <- -> pilot   ENTER launch   ESC back", True, (60, 100, 90))
        screen.blit(foot, (WIDTH // 2 - foot.get_width() // 2, HEIGHT - 20))

        btn_start.draw(screen)
        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)


# ══════════════════════════════════════════════════════════════
#  Briefing: launch keys and rules, just before the round starts
# ══════════════════════════════════════════════════════════════
def _keycap(surface, x, y, text, color=JET_COLOR):
    """A key drawn as a filled cap; returns its width."""
    ks = font_btn.render(" %s " % text, True, (10, 20, 30))
    pygame.draw.rect(surface, color, (x, y, ks.get_width(), ks.get_height()), 0, 4)
    surface.blit(ks, (x, y))
    return ks.get_width()


def _style_icon(surface, x, y, style, color):
    """A 44 x 14 px sketch of how a flight style flies, left to right."""
    pts = []
    for i in range(12):
        t = i / 11.0
        px = x + 44.0 * t
        if style == "PURSUIT":
            py = y + 7.0
        elif style == "FLANKER":
            py = y + 13.0 - 12.0 * math.sin(math.pi * t)
        elif style == "WEAVER":
            py = y + 7.0 + 6.0 * math.sin(4.0 * math.pi * t)
        else:
            py = y + 7.0 + (6.0 if (i // 2) % 2 else -6.0)
        pts.append((int(px), int(py)))
    pygame.draw.lines(surface, color, False, pts, 2)
    pygame.draw.circle(surface, color, pts[-1], 3)


def run_briefing_screen(level, version):
    """The round's launch keys and rules. Returns "start" | "back" | "quit"."""
    from ..pilots import VERSIONS

    btn_start = NeonButton(WIDTH // 2, HEIGHT - 60, 240, 44, "[ START ROUND ]", JET_COLOR, JET_COLOR)
    radar_a = 0.0
    glow_s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    panel = pygame.Surface((356, 262), pygame.SRCALPHA)
    panel.fill((6, 14, 24, 215))
    vname = VERSIONS[version - 1].name
    text = (170, 200, 190)

    while True:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if btn_start.is_clicked(event):
                return "start"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "back"
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return "start"

        radar_a = (radar_a + 0.02) % (math.pi * 2)
        update_bg_particles()
        btn_start.update(mx, my)

        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0, 0))
        draw_bg_particles(screen)
        _draw_corner_brackets(screen)

        draw_glow_text(screen, "BRIEFING", font_sub, WIDTH // 2, 24, JET_COLOR, layers=2)
        sub = font_hud.render("LEVEL %d  %s   ·   %s" % (level.number, level.name, vname),
                              True, (0, 200, 160))
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 50))

        # Plan view, with each battery's keys on it.
        mm = pygame.Rect(WIDTH // 2 - 170, 74, 340, 120)
        _minimap(screen, level, mm)
        for st in level.stations:
            caps = [st.key] + ([st.cluster_key] if st.cluster_key else [])
            sx = int(mm.x + st.x * mm.width / float(WIDTH))
            cx = min(max(sx - 14 * len(caps), mm.x + 4), mm.right - 30 * len(caps))
            for k in caps:
                cx += _keycap(screen, cx, mm.bottom - 34, k, st.color) + 4

        # Left: launch keys.
        lx, ly = 36, 206
        screen.blit(panel, (lx, ly))
        pygame.draw.rect(screen, (0, 140, 110), (lx, ly, 356, 262), 1, 6)
        screen.blit(font_hud.render("YOUR LAUNCH KEYS", True, (0, 220, 170)), (lx + 12, ly + 8))
        y = ly + 32
        for st in level.stations:
            where = "WEST" if st.x < WIDTH / 2 else "EAST"
            screen.blit(font_tiny.render("%s BATTERY  ·  %s" % (where, st.label), True, (110, 170, 150)),
                        (lx + 12, y))
            y += 16
            if st.kind == "ballistic":
                what = "one ballistic shot"
            else:
                what = "one seeker missile" if level.smart else "one homing missile"
            w = _keycap(screen, lx + 16, y, st.key, st.color)
            screen.blit(font_sm.render(what, True, text), (lx + 24 + w, y + 2))
            y += 24
            if st.cluster_key:
                w = _keycap(screen, lx + 16, y, st.cluster_key, st.color)
                screen.blit(font_sm.render("cluster of five  (jet gets +1 flare)", True, text),
                            (lx + 24 + w, y + 2))
                y += 24
            y += 4
        extras = []
        if level.pincer:
            extras.append(("S", "pincer salvo from both batteries", (255, 100, 255)))
        extras.append(("G", "graph overlay (V3)", (0, 200, 160)))
        extras.append(("ESC", "abort to setup", (120, 140, 140)))
        for key, desc, col in extras:
            w = _keycap(screen, lx + 16, y, key, col)
            screen.blit(font_sm.render(desc, True, text), (lx + 24 + w, y + 2))
            y += 24

        # Right: what the missiles do, and the jet's flares.
        rx, ry = 408, 206
        screen.blit(panel, (rx, ry))
        pygame.draw.rect(screen, (0, 140, 110), (rx, ry, 356, 262), 1, 6)
        screen.blit(font_hud.render("THE MISSILES", True, (0, 220, 170)), (rx + 12, ry + 8))
        y = ry + 34
        for name, desc, col in STYLE_NOTES:
            _style_icon(screen, rx + 14, y, name, col)
            screen.blit(font_hud.render(name, True, col), (rx + 66, y - 1))
            screen.blit(font_tiny.render(desc, True, (150, 180, 170)), (rx + 150, y + 1))
            y += 22
        notes = ["Each missile's style is random."]
        if level.obstacles:
            notes.append("Seekers steer around peaks.")
        if level.smart:
            notes.append("A seeker that loses the jet flies to its last spot.")
        if level.datalink:
            notes.append("One lock guides every missile in its salvo.")
        y += 2
        for line in notes:
            screen.blit(font_tiny.render(line, True, (110, 150, 140)), (rx + 14, y))
            y += 14
        y += 8
        screen.blit(font_hud.render("THE JET'S FLARES", True, FLARE_COLOR), (rx + 12, y))
        y += 20
        for line in ("Every life starts with 0 flares.",
                     "Each cluster you launch gives it +1 volley.",
                     "A volley costs one, whatever it holds."):
            screen.blit(font_sm.render(line, True, (220, 190, 140)), (rx + 14, y))
            y += 17

        foot = font_tiny.render("ENTER / SPACE  start round      ESC  back to setup", True, (60, 100, 90))
        screen.blit(foot, (WIDTH // 2 - foot.get_width() // 2, HEIGHT - 22))
        btn_start.draw(screen)
        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)


def run_gameover_screen(user_won, stats, detail=None):
    btn_again = NeonButton(WIDTH//2, HEIGHT//2+70,  220, 46, "[ PLAY AGAIN ]", JET_COLOR,     JET_COLOR)
    btn_setup = NeonButton(WIDTH//2, HEIGHT//2+125, 220, 46, "[ SETUP ]",      (255,100,255), (255,100,255))
    btn_exit  = NeonButton(WIDTH//2, HEIGHT//2+180, 220, 46, "[ EXIT ]",       MISSILE_COLOR, MISSILE_COLOR)
    radar_a   = 0.0
    glow_s    = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tick_sin  = 0.0
    enter_t   = pygame.time.get_ticks()
    detail    = detail or {}

    if user_won:
        headline = "JET DESTROYED"
        cause = detail.get('cause')
        sub_line = {"missile": "YOU WIN — MISSILE HIT", "wall": "YOU WIN — HERDED INTO A WALL",
                    "terrain": "YOU WIN — FLEW INTO TERRAIN"}.get(cause, "YOU WIN — AI ELIMINATED")
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
                if btn_again.is_clicked(event): return "again"
                if btn_setup.is_clicked(event): return "setup"
                if btn_exit.is_clicked(event):  return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and elapsed > GAMEOVER_LOCKOUT_MS: return "again"
                if event.key == pygame.K_ESCAPE: return "setup"
                if event.key == pygame.K_q: return "quit"

        radar_a  = (radar_a + 0.025) % (math.pi*2)
        tick_sin = (tick_sin + 0.06) % (math.pi*2)
        update_bg_particles()
        btn_again.update(mx, my)
        btn_setup.update(mx, my)
        btn_exit.update(mx, my)

        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0,0))
        draw_bg_particles(screen)

        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        a_ov = min(160, int((elapsed/600)*160))
        ov.fill((0,0,0,a_ov))
        screen.blit(ov, (0,0))

        pulse = int(math.sin(tick_sin)*4)
        draw_glow_text(screen, headline, font_result, WIDTH//2, HEIGHT//2-120+pulse,
                       h_color, center=True, layers=4)
        draw_glow_text(screen, sub_line, font_sub, WIDTH//2, HEIGHT//2-62,
                       s_color, center=True, layers=2)

        stat_y = HEIGHT//2 - 38
        stat_entries = []
        if detail:
            stat_entries.append(("%s  ·  %s  ·  %.1f s" % (detail.get('level', ''),
                                                        detail.get('version', ''),
                                                        detail.get('seconds', 0.0)), (120,200,180)))
        stat_entries += [
            ("MISSILES FIRED : %d" % stats.get('fired',0),     (80,160,140)),
            ("FLARES DEPLOYED: %d" % stats.get('flares',0),    (80,130,100)),
            ("MANEUVERS USED : %d" % stats.get('maneuvers',0), (60,100,90)),
        ]
        for i,(txt,col) in enumerate(stat_entries):
            ss = font_hud.render(txt, True, col)
            screen.blit(ss, (WIDTH//2-ss.get_width()//2, stat_y+i*18))

        if elapsed > 900:
            hint_a = int(80+80*math.sin(tick_sin))
            hs = font_sm.render("R — Play Again  |  ESC — Setup  |  Q — Quit", True, (60,100,90))
            hs.set_alpha(hint_a)
            screen.blit(hs, (WIDTH//2-hs.get_width()//2, HEIGHT-22))

        if elapsed > GAMEOVER_LOCKOUT_MS:
            btn_again.draw(screen)
            btn_setup.draw(screen)
            btn_exit.draw(screen)

        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)
