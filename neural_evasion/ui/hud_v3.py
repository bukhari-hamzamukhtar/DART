"""
The v3 HUD: the dynamic graph, drawn live.

  [G] overlay (on by default for v3)
    edges   coloured by type -- red SEEKER (this missile's seeker can see you),
            orange THREAT, cyan LINK (bright when the GNN believes the datalink is
            live), yellow DECOY, grey OCCLUDE (a peak blocking a missile's view),
            purple LAUNCH
    nodes   tracks labelled with their persistent ID; tracks hidden behind
            terrain drawn as hollow ghosts at the tracker's dead-reckoned position
    future  the winning candidate's imagined 72 frames: the jet's path, and each
            missile's predicted path -- red while the world model predicts it
            still has the jet, yellow once it predicts the lock is gone
  track panel   per track, what the GNN believes: guided, turn rate, lead,
                pincer side, datalink net, threat
  event ticker  the graph's event stream: tracks born / hidden / reacquired /
                destroyed, seeker edges gained and lost
  graph line    nodes, edges, insertions/deletions last tick, GNN and planner ms
"""

import math

import pygame

from ..config import HEIGHT, HUD_HEIGHT, WIDTH
from ..sim.scene_graph import (DECOY, LAUNCH, LINK, NEAR, OCCLUDE, SEEKER,
                               STATION, THREAT, TRACK, key_flare, key_track)
from .fonts import font_hud, font_sm, font_tiny
from .hud import draw_hud_bar, draw_polar_hud

EDGE_COLORS = {
    THREAT:  (255, 130, 60),
    SEEKER:  (255, 45, 45),
    LINK:    (80, 220, 255),
    NEAR:    (150, 150, 160),
    DECOY:   (255, 210, 50),
    OCCLUDE: (150, 170, 160),
    LAUNCH:  (190, 100, 255),
}
_CACHE = {}


def _layer():
    s = _CACHE.get('layer')
    if s is None:
        s = _CACHE['layer'] = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))
    return s


def _label(text, font, color):
    key = (text, id(font), color)
    surf = _CACHE.get(key)
    if surf is None:
        if len(_CACHE) > 600:
            _CACHE.clear()
        surf = _CACHE[key] = font.render(text, True, color)
    return surf


def _positions(arena, pilot):
    """Where every graph node is *now* -- the snapshot is up to GRAPH_TICK frames old."""
    g = pilot.graph
    pos = {k: (float(p[0]), float(p[1])) for k, p in zip(g.keys, g.pos)}
    pos[0] = (arena.jet.x, arena.jet.y)
    for t in pilot.tracker.tracks.values():
        pos[key_track(t.tid)] = (t.x, t.y)
    for f in arena.flares:
        k = key_flare(f.uid)
        if k in pos:
            pos[k] = (f.x, f.y)
    return pos


def draw_graph(s, arena, pilot):
    """Draw the live graph into the transparent overlay layer `s`."""
    g = pilot.graph
    if g is None:
        return
    pos = _positions(arena, pilot)
    tracker = pilot.tracker
    link_p = {}
    for a, b, p in pilot.links:
        link_p[(a, b)] = link_p[(b, a)] = p

    keys = g.keys
    for e in range(len(g.src)):
        typ = int(g.etype[e])
        col = EDGE_COLORS.get(typ)
        if col is None:
            continue
        ks, kd = keys[int(g.src[e])], keys[int(g.dst[e])]
        if ks not in pos or kd not in pos:
            continue
        width, alpha = 1, 90
        if typ == SEEKER:
            width, alpha = 2, 200
        elif typ == THREAT:
            t = tracker.tracks.get(ks - 1000000)
            thr = t.est['threat'] if t is not None else 0.0
            alpha = int(25 + 170 * thr)
            width = 1 if thr < 0.6 else 2
        elif typ == LINK:
            if ks > kd:
                continue
            p = link_p.get((ks - 1000000, kd - 1000000), 0.0)
            alpha = int(30 + 200 * p)
            width = 2 if p >= 0.5 else 1
        elif typ == NEAR:
            alpha = 45
        (x1, y1), (x2, y2) = pos[ks], pos[kd]
        pygame.draw.line(s, (*col, alpha), (int(x1), int(y1)), (int(x2), int(y2)), width)

    for t in tracker.tracks.values():
        x, y = int(t.x), int(t.y)
        thr = t.est['threat']
        col = (255, int(200 - 150 * thr), 60)
        if t.visible:
            pygame.draw.circle(s, (*col, 150), (x, y), 7, 1)
        else:
            for a in range(0, 360, 45):
                r = math.radians(a)
                pygame.draw.circle(s, (200, 200, 200, 160),
                                   (int(x + 8 * math.cos(r)), int(y + 8 * math.sin(r))), 1)
        s.blit(_label("T%d" % t.tid, font_tiny, (230, 230, 230) if t.visible else (150, 150, 150)),
               (x + 9, y - 12))


def draw_future(s, pilot):
    """The winning candidate's imagined future, into the overlay layer `s`."""
    p = pilot.plan
    if p is None or p.result is None:
        return
    res, b = p.result, p.best
    jx, jy = res.jet_x[b], res.jet_y[b]
    pts = [(int(jx[k]), int(jy[k])) for k in range(0, jx.shape[0], 3)]
    if len(pts) > 1:
        pygame.draw.lines(s, (0, 255, 200, 110), False, pts, 1)
    M = res.threat_x.shape[1]
    for i in range(M):
        tx, ty = res.threat_x[b, i], res.threat_y[b, i]
        alive = res.alive_path[b, i]
        guided = res.guided_path[b, i] if res.guided_path is not None else None
        seg, seg_col = [], None
        for k in range(0, tx.shape[0], 3):
            if not alive[k]:
                break
            col = (255, 60, 60, 120) if guided is None or guided[k] else (255, 210, 60, 120)
            if col != seg_col and seg:
                if len(seg) > 1:
                    pygame.draw.lines(s, seg_col, False, seg, 1)
                seg = [seg[-1]]
            seg_col = col
            seg.append((int(tx[k]), int(ty[k])))
        if len(seg) > 1:
            pygame.draw.lines(s, seg_col, False, seg, 1)


def draw_track_panel(surface, pilot):
    tracks = sorted(pilot.tracker.tracks.values(), key=lambda t: -t.est['threat'])[:7]
    x0, y0, w = WIDTH - 250, 10, 240
    h = 30 + 15 * max(1, len(tracks))
    panel = _CACHE.get(('panel', h))
    if panel is None:
        panel = _CACHE[('panel', h)] = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((6, 14, 30, 205))
    surface.blit(panel, (x0, y0))
    pygame.draw.rect(surface, (0, 140, 110), (x0, y0, w, h), 1, 6)
    title = "TRACK MEMORY (TGN)" if pilot.brain is not None else "TRACKS (NO GNN WEIGHTS)"
    surface.blit(_label(title, font_hud, (0, 220, 170)), (x0 + 8, y0 + 6))
    if not tracks:
        surface.blit(_label("no tracks", font_sm, (40, 80, 70)), (x0 + 80, y0 + 26))
        return
    for r, t in enumerate(tracks):
        e = t.est
        yy = y0 + 24 + r * 15
        thr = e['threat']
        pygame.draw.rect(surface, (20, 30, 40), (x0 + 8, yy + 3, 30, 7))
        pygame.draw.rect(surface, (int(80 + 175 * thr), int(200 * (1 - thr)), 40),
                         (x0 + 8, yy + 3, int(30 * thr), 7))
        if e['pursuit'] < 0.35:
            state = "BALLIS"
        else:
            state = "GUIDED" if e['guided'] >= 0.5 else "LOST  "
        side = int(round(e['side']))
        net = "n%-2d" % (t.group % 100) if t.group >= 0 else "   "
        txt = "T%-3d %s tr%.2f<%.2f ld%.1f s%+d %s" % (t.tid % 1000, state, e['turn_rate'],
                                                       e.get('turn_hi', e['turn_rate']),
                                                       e['lead'], side, net)
        col = (220, 230, 220) if t.visible else (130, 130, 130)
        surface.blit(_label(txt, font_tiny, col), (x0 + 42, yy))


def draw_ticker(surface, pilot):
    lines = list(pilot.ticker)[-6:]
    y = HEIGHT - HUD_HEIGHT - 14 * len(lines) - 6
    for i, line in enumerate(lines):
        surface.blit(_label(line, font_tiny, (110, 170, 150)), (10, y + 14 * i))


def draw_graph_line(surface, pilot):
    g = pilot.graph
    if g is None:
        return
    add, rem = pilot.edge_delta
    tm = pilot.timing
    txt = "DYNAMIC GRAPH  %d nodes  %d edges  +%d/-%d   TGN %.1fms  PLAN %.1fms" % (
        len(g.keys), len(g.src), add, rem, tm['gnn'], tm['plan'])
    lbl = _label(txt, font_tiny, (0, 190, 150))
    surface.blit(lbl, (WIDTH // 2 - lbl.get_width() // 2 - 60, 4))


def draw_hud_v3(surface, arena, pilot, show_overlay):
    if show_overlay:
        # One transparent layer for the future and the graph: a full-screen
        # alpha blit costs a few ms on its own, so it is done once.
        layer = _layer()
        draw_future(layer, pilot)
        draw_graph(layer, arena, pilot)
        surface.blit(layer, (0, 0))
        draw_ticker(surface, pilot)
    d = pilot.decision
    draw_hud_bar(surface, d.danger, pilot.maneuver_label, arena.jet.speed, arena.level.controls)
    draw_track_panel(surface, pilot)
    draw_polar_hud(surface, arena.jet.heading, d.candidate_scores, 60, 65, r=44)
    lbl = _label("HEADING RISK", font_tiny, (0, 160, 120))
    surface.blit(lbl, (60 - lbl.get_width() // 2, 115))
    draw_graph_line(surface, pilot)
