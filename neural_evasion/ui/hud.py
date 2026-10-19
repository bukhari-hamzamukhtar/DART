"""
In-match HUD (the render half of main3.py section 9).

Four independent panels, each a pure function of the latest `AIDecision`:

  * threat bar + maneuver label along the bottom
  * per-missile attention histogram, top right
  * polar counterfactual heading chart, top left
  * optional [G] graph overlay showing GNN edge-message magnitudes
"""

import math

import pygame

from ..config import HEIGHT, HUD_HEIGHT, MAX_THROTTLE, MIN_THROTTLE, WIDTH
from .fonts import font_hud, font_sm, font_tiny

MANEUVER_COLORS = {
    "NORMAL":       (80, 180, 140),
    "IMMELMANN":    (255, 220, 50),
    "COBRA":        (255, 80, 80),
    "FALLING_LEAF": (255, 160, 30),
    "JINKING":      (80, 160, 255),
    "BARREL_ROLL":  (180, 80, 255),
}

CONTROLS_TEXT = "H:Homing  B:Ballistic  C:Cluster  ESC:Menu"

MAX_ATTENTION_BARS = 8
POLAR_CENTER = (60, 65)
POLAR_RADIUS = 44


def draw_gnn_overlay(surface, edges):
    """Draw every GNN edge, coloured and thickened by message magnitude."""
    if not edges:
        return
    mags = [e['mag'] for e in edges]
    max_mag = max(mags) + 1e-5
    gnn_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for e in edges:
        norm = e['mag'] / max_mag
        alpha = int(30 + norm * 180)
        width = max(1, int(norm * 3))
        r = int(norm * 255)
        g = int((1 - norm) * 180)
        x1, y1 = int(e['p1'][0]), int(e['p1'][1])
        x2, y2 = int(e['p2'][0]), int(e['p2'][1])
        pygame.draw.line(gnn_surf, (r, g, 50, alpha), (x1, y1), (x2, y2), width)
    surface.blit(gnn_surf, (0, 0))
    lbl = font_sm.render("GNN GRAPH [G]", True, (0, 200, 160))
    surface.blit(lbl, (14, HEIGHT - 95))


def draw_hud_bar(surface, danger, maneuver, throttle=None, controls=None):
    """Bottom bar: threat meter, throttle meter, current maneuver, controls."""
    hy = HEIGHT - HUD_HEIGHT
    pygame.draw.rect(surface,(6,14,26),(0,hy,WIDTH,HUD_HEIGHT))
    pygame.draw.line(surface,(0,180,140),(0,hy),(WIDTH,hy),1)

    # Threat meter
    dc = min(max(danger,0.0),1.0)
    mx_p,my_p = 14, hy+10
    pygame.draw.rect(surface,(20,30,40),(mx_p,my_p,160,12),0,4)
    fw = int(dc*160)
    if fw > 0:
        pygame.draw.rect(surface,(int(dc*255),int((1-dc)*200),0),(mx_p,my_p,fw,12),0,4)
    pygame.draw.rect(surface,(0,180,140),(mx_p,my_p,160,12),1,4)
    surface.blit(font_hud.render("THREAT",True,(0,200,160)),(mx_p,my_p+16))

    # Throttle meter — the jet picks its own speed, so show what it picked
    if throttle is not None:
        tx_p = mx_p + 186
        frac = (throttle - MIN_THROTTLE) / float(MAX_THROTTLE - MIN_THROTTLE)
        frac = min(max(frac, 0.0), 1.0)
        pygame.draw.rect(surface,(20,30,40),(tx_p,my_p,110,12),0,4)
        tw = int(frac*110)
        if tw > 0:
            pygame.draw.rect(surface,(40,180,255),(tx_p,my_p,tw,12),0,4)
        pygame.draw.rect(surface,(0,180,140),(tx_p,my_p,110,12),1,4)
        surface.blit(font_hud.render("THROTTLE %.1f" % throttle, True,(0,200,160)),
                     (tx_p,my_p+16))

    # Maneuver label
    mc = MANEUVER_COLORS.get(maneuver,(255,255,255))
    ml = font_hud.render("[ %s ]" % maneuver, True, mc)
    surface.blit(ml,(WIDTH//2-ml.get_width()//2, hy+8))

    # Controls
    cl = font_sm.render(controls or CONTROLS_TEXT,True,(60,100,120))
    surface.blit(cl,(WIDTH-cl.get_width()-14, hy+28))


def draw_attention_panel(surface, attention, danger):
    """Histogram of how much attention the brain is paying to each missile."""
    px2,py2,pw,ph = WIDTH-190, 12, 178, 130
    ps2 = pygame.Surface((pw,ph),pygame.SRCALPHA); ps2.fill((6,14,30,210))
    surface.blit(ps2,(px2,py2))
    pygame.draw.rect(surface,(0,140,110),(px2,py2,pw,ph),1,6)
    surface.blit(font_hud.render("AI ATTENTION",True,(0,220,170)),(px2+8,py2+7))
    if attention:
        nb   = min(len(attention),MAX_ATTENTION_BARS)
        bw   = max(14,(pw-16)//nb-4)
        bmh  = 60; bby = py2+100
        for i in range(nb):
            wv = attention[i]
            bx2 = px2+8+i*(bw+4); bh2 = int(wv*bmh)
            pygame.draw.rect(surface,(15,30,45),(bx2,bby-bmh,bw,bmh),0,3)
            rb=int(wv*255); gb=int((1-wv)*220)
            bc=(rb,gb,30)
            if wv>0.80 and danger>0.55 and (pygame.time.get_ticks()//150)%2==0:
                bc=(255,255,255)
            if bh2>0: pygame.draw.rect(surface,bc,(bx2,bby-bh2,bw,bh2),0,3)
            pygame.draw.rect(surface,(0,160,120),(bx2,bby-bmh,bw,bmh),1,3)
            surface.blit(font_tiny.render("M%d" % (i+1),True,(120,180,160)),(bx2+bw//2-7,bby+2))
            surface.blit(font_tiny.render("%d%%" % int(wv*100),True,(80,140,120)),(bx2-1,bby-bh2-14))
    else:
        surface.blit(font_sm.render("no threats",True,(40,80,70)),(px2+45,py2+65))


def draw_polar_hud(surface, jet_heading, candidate_scores, cx, cy, r=48):
    """Polar chart of candidate headings — spoke length = safety, color = danger."""
    if not candidate_scores:
        return
    s = pygame.Surface((r*2+20, r*2+20), pygame.SRCALPHA)
    ox, oy = r+10, r+10

    # Background rings
    for ring_r in [r//3, r*2//3, r]:
        pygame.draw.circle(s, (0, 180, 140, 18), (ox, oy), ring_r, 1)
    pygame.draw.circle(s, (0, 180, 140, 35), (ox, oy), 4)  # center dot

    max_d = max(d for _, d in candidate_scores) + 1e-5
    min_d = min(d for _, d in candidate_scores)

    for angle, danger in candidate_scores:
        norm = (danger - min_d) / (max_d - min_d + 1e-5)
        spoke_len = int((1.0 - norm) * r)  # safe = long spoke
        # Screen angle: subtract pi/2 because pygame y-axis is flipped
        screen_angle = angle - math.pi / 2
        ex = int(ox + math.cos(screen_angle) * spoke_len)
        ey = int(oy + math.sin(screen_angle) * spoke_len)
        col_r = int(norm * 255)
        col_g = int((1.0 - norm) * 200)
        alpha = 160 + int(norm * 80)
        width = 2 if danger == min_d else 1
        pygame.draw.line(s, (col_r, col_g, 40, alpha), (ox, oy), (ex, ey), width)
        # Dot at tip
        pygame.draw.circle(s, (col_r, col_g, 40, alpha), (ex, ey), 2)

    # Chosen heading arrow (bright teal)
    ba = jet_heading - math.pi / 2
    bx = int(ox + math.cos(ba) * r)
    by = int(oy + math.sin(ba) * r)
    pygame.draw.line(s, (0, 255, 200, 230), (ox, oy), (bx, by), 2)
    pygame.draw.circle(s, (0, 255, 200, 230), (bx, by), 3)

    surface.blit(s, (cx - ox, cy - oy))


def draw_hud(surface, jet, maneuver, decision, show_gnn_overlay=False, controls=None):
    """Paint every HUD element for one frame."""
    if show_gnn_overlay:
        draw_gnn_overlay(surface, decision.edges)

    draw_hud_bar(surface, decision.danger, maneuver, jet.speed, controls)
    draw_attention_panel(surface, decision.attention, decision.danger)

    polar_cx, polar_cy = POLAR_CENTER
    draw_polar_hud(surface, jet.heading, decision.candidate_scores,
                   polar_cx, polar_cy, r=POLAR_RADIUS)
    lbl_p = font_tiny.render("HEADING RISK", True, (0, 160, 120))
    surface.blit(lbl_p, (polar_cx - lbl_p.get_width()//2, polar_cy + 50))
