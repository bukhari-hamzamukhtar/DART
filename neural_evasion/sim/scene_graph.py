"""
The dynamic scene graph: what the jet's graph network looks at, every tick.

Nodes are everything that matters, each with a persistent key so the same
missile is the same node from one tick to the next:

    JET       the aircraft itself
    TRACK     a radar track -- visible, or coasting behind terrain
    FLARE     a decoy in the air
    OBSTACLE  a terrain peak
    STATION   a missile battery

Edges are typed relations, recomputed from geometry every tick, so the topology
changes as the scene does:

    THREAT    track -> jet     within threat range
    WATCH     jet -> track     the jet's view of the same pair
    SEEKER    track -> jet     by doctrine, the track's seeker can see the jet now
    LINK      track <-> track  within datalink range (levels with a datalink)
    NEAR      track <-> track  close enough to interact
    DECOY     flare -> track   a flare sits in the track's seeker cone
    TERRAIN   obstacle -> jet/track   close to a peak
    OCCLUDE   obstacle -> track       this peak blocks the track's view of the jet
    LAUNCH    station -> track        fired from this battery, recently

This is a continuous-time dynamic graph in the sense of Kazemi et al. (JMLR
2020) and Rossi et al.'s Temporal Graph Networks: one graph whose nodes and
edges are added and deleted as events, observed as a snapshot each tick. It is
*not* rebuilt from nothing -- node keys persist, and the GNN's memory is keyed on
them. `diff` returns the edge insertions and deletions between two snapshots,
which is the event stream the HUD shows.
"""

import math

import numpy as np

from ..config import HEIGHT, HUD_HEIGHT, WIDTH
from .doctrine import DECOY_RANGE, LINK_RANGE, SEEKER_COS_FOV
from .geometry import segment_blocked

JET, TRACK, FLARE, OBSTACLE, STATION = range(5)
NODE_TYPES = ["JET", "TRACK", "FLARE", "OBSTACLE", "STATION"]

(THREAT, WATCH, SEEKER, LINK, NEAR, DECOY, TERRAIN, OCCLUDE, LAUNCH) = range(9)
EDGE_TYPES = ["THREAT", "WATCH", "SEEKER", "LINK", "NEAR", "DECOY", "TERRAIN",
              "OCCLUDE", "LAUNCH"]

NODE_DIM = 23
EDGE_DIM = 9

THREAT_RADIUS   = 520.0
NEAR_RADIUS     = 120.0
TERRAIN_RADIUS  = 110.0     # px beyond a peak's surface
LAUNCH_AGE      = 90        # frames a fresh track stays attached to its battery
LAUNCH_SNAP     = 30.0      # px; a track born this close to a battery came from it
MAX_FLARE_NODES = 16
FIELD_H = float(HEIGHT - HUD_HEIGHT)

KEY_JET = 0


def key_track(tid):   return 1000000 + tid
def key_flare(uid):   return 2000000 + uid
def key_obstacle(k):  return 3000000 + k
def key_station(k):   return 4000000 + k


class SceneGraph(object):
    """One snapshot of the dynamic graph."""

    __slots__ = ("frame", "keys", "ntype", "x", "pos", "src", "dst", "etype",
                 "eattr", "track_rows", "track_tids", "link_rows", "names")

    def edge_set(self):
        k = self.keys
        return set(zip(self.etype.tolist(), (k[s] for s in self.src.tolist()),
                       (k[d] for d in self.dst.tolist())))


def _node_row(ntype, x, y, vx, vy, heading, visible, age, coast, turn, size,
              lk=None, speed=None, jet=None):
    row = [0.0] * NODE_DIM
    if jet is not None:
        # Where a track points relative to its line of sight to the jet: a
        # pursuer points at it, a flanker holds off to one side, a weaver or an
        # erratic missile swings across it.
        off = heading - math.atan2(jet[1] - y, jet[0] - x)
        row[20] = math.sin(off)
        row[21] = math.cos(off)
        row[22] = min(math.hypot(jet[0] - x, jet[1] - y) / 400.0, 2.0)
    row[ntype] = 1.0
    row[5] = x / WIDTH
    row[6] = y / FIELD_H
    row[7] = vx / 8.0
    row[8] = vy / 8.0
    row[9] = math.cos(heading)
    row[10] = math.sin(heading)
    row[11] = visible
    row[12] = min(age / 300.0, 1.0)
    row[13] = min(coast / 60.0, 1.0)
    row[14] = max(-2.0, min(2.0, turn / 0.15))
    row[15] = size / 60.0
    if lk is not None:
        row[16] = 1.0
        row[17] = (lk[0] - x) / 400.0
        row[18] = (lk[1] - y) / 400.0
    row[19] = (math.hypot(vx, vy) if speed is None else speed) / 8.0
    return row


class SceneGraphBuilder(object):
    """Builds a SceneGraph from the jet, the tracker and what else is in the air."""

    def __init__(self, level):
        self.level = level
        self.terrain = [tuple(o) for o in level.obstacles]

    def build(self, frame, jet, tracker, flares, launchers):
        keys, ntype, rows, pos, names = [], [], [], [], []
        # per node: (x, y, vx, vy, heading) for edge attributes
        kin = []

        def add(key, t, row, x, y, vx, vy, h, name):
            keys.append(key)
            ntype.append(t)
            rows.append(row)
            pos.append((x, y))
            kin.append((x, y, vx, vy, h))
            names.append(name)
            return len(keys) - 1

        jet_i = add(KEY_JET, JET,
                    _node_row(JET, jet.x, jet.y, jet.vx, jet.vy, jet.heading, 1.0,
                              frame, 0, 0.0, jet.radius),
                    jet.x, jet.y, jet.vx, jet.vy, jet.heading, "JET")

        tracks = sorted(tracker.tracks.values(), key=lambda t: t.tid)
        track_rows, track_tids = [], []
        for t in tracks:
            i = add(key_track(t.tid), TRACK,
                    _node_row(TRACK, t.x, t.y, t.vx, t.vy, t.heading,
                              1.0 if t.visible else 0.0, t.age, t.coast,
                              t.turn_per_frame(), 2.0, t.last_known, t.speed,
                              jet=(jet.x, jet.y)),
                    t.x, t.y, t.vx, t.vy, t.heading, "T%d" % t.tid)
            track_rows.append(i)
            track_tids.append(t.tid)

        # Only flares near some track can matter; keep the closest ones.
        if flares and tracks:
            scored = []
            for f in flares:
                dmin = min(math.hypot(f.x - t.x, f.y - t.y) for t in tracks)
                if dmin < DECOY_RANGE:
                    scored.append((dmin, f))
            scored.sort(key=lambda p: p[0])
            flares = [f for _, f in scored[:MAX_FLARE_NODES]]
        else:
            flares = []
        flare_rows = []
        for f in flares:
            age = frame - (f.born_frame if getattr(f, "born_frame", None) is not None else frame)
            flare_rows.append(add(key_flare(f.uid), FLARE,
                                  _node_row(FLARE, f.x, f.y, f.vx, f.vy,
                                            math.atan2(f.vy, f.vx), 1.0, age, 0, 0.0, f.radius),
                                  f.x, f.y, f.vx, f.vy, 0.0, "F"))

        obs_rows = []
        for k, (ox, oy, r) in enumerate(self.terrain):
            obs_rows.append(add(key_obstacle(k), OBSTACLE,
                                _node_row(OBSTACLE, ox, oy, 0.0, 0.0, 0.0, 1.0, 300, 0, 0.0, r),
                                ox, oy, 0.0, 0.0, 0.0, "P%d" % (k + 1)))

        st_rows = []
        for k, l in enumerate(launchers):
            st_rows.append(add(key_station(k), STATION,
                               _node_row(STATION, l.x, l.y, 0.0, 0.0, -math.pi / 2,
                                         1.0 if l.alive else 0.0, 300, 0, 0.0, 15.0),
                               float(l.x), float(l.y), 0.0, 0.0, -math.pi / 2,
                               "S%d" % (k + 1)))

        # ── edges ────────────────────────────────────────────
        src, dst, et = [], [], []

        def edge(a, b, typ):
            src.append(a)
            dst.append(b)
            et.append(typ)

        terrain = self.terrain
        for ti, t in zip(track_rows, tracks):
            d = math.hypot(t.x - jet.x, t.y - jet.y)
            if d < THREAT_RADIUS:
                edge(ti, jet_i, THREAT)
                edge(jet_i, ti, WATCH)
            if t.visible and t.seeker_edge:
                edge(ti, jet_i, SEEKER)
            if terrain:
                for oi, (k, (ox, oy, r)) in zip(obs_rows, enumerate(terrain)):
                    if math.hypot(t.x - ox, t.y - oy) - r < TERRAIN_RADIUS:
                        edge(oi, ti, TERRAIN)
                    if segment_blocked(t.x, t.y, jet.x, jet.y, ((ox, oy, r),)):
                        edge(oi, ti, OCCLUDE)
            if t.age < LAUNCH_AGE:
                for si, l in zip(st_rows, launchers):
                    if math.hypot(t.birth_x - l.x, t.birth_y - l.y) < LAUNCH_SNAP:
                        edge(si, ti, LAUNCH)

        link_rows = []
        for a in range(len(tracks)):
            ta = tracks[a]
            for b in range(a + 1, len(tracks)):
                tb = tracks[b]
                d = math.hypot(ta.x - tb.x, ta.y - tb.y)
                if self.level.datalink and d < LINK_RANGE:
                    link_rows.append(len(src))
                    edge(track_rows[a], track_rows[b], LINK)
                    edge(track_rows[b], track_rows[a], LINK)
                if d < NEAR_RADIUS:
                    edge(track_rows[a], track_rows[b], NEAR)
                    edge(track_rows[b], track_rows[a], NEAR)

        for fi, f in zip(flare_rows, flares):
            for ti, t in zip(track_rows, tracks):
                dx, dy = f.x - t.x, f.y - t.y
                d = math.hypot(dx, dy)
                if d < DECOY_RANGE and (dx * math.cos(t.heading) + dy * math.sin(t.heading)
                                        >= SEEKER_COS_FOV * d):
                    if not (terrain and segment_blocked(t.x, t.y, f.x, f.y, terrain)):
                        edge(fi, ti, DECOY)

        if terrain:
            for oi, (ox, oy, r) in zip(obs_rows, terrain):
                if math.hypot(jet.x - ox, jet.y - oy) - r < TERRAIN_RADIUS:
                    edge(oi, jet_i, TERRAIN)

        # ── edge attributes ──────────────────────────────────
        E = len(src)
        eattr = np.zeros((E, EDGE_DIM), dtype=np.float32)
        for e in range(E):
            sx, sy, svx, svy, _ = kin[src[e]]
            dx_, dy_, dvx, dvy, dh = kin[dst[e]]
            rx, ry = sx - dx_, sy - dy_
            d = math.hypot(rx, ry)
            rvx, rvy = svx - dvx, svy - dvy
            bearing = math.atan2(ry, rx) - dh
            clear = 1.0
            if terrain and et[e] in (THREAT, WATCH, LINK, NEAR):
                clear = 0.0 if segment_blocked(sx, sy, dx_, dy_, terrain) else 1.0
            eattr[e] = (rx / 400.0, ry / 400.0, rvx / 10.0, rvy / 10.0, d / 400.0,
                        -(rx * rvx + ry * rvy) / (d + 1e-6) / 10.0,
                        math.cos(bearing), math.sin(bearing), clear)

        g = SceneGraph()
        g.frame = frame
        g.keys = keys
        g.ntype = np.asarray(ntype, dtype=np.int64)
        g.x = np.asarray(rows, dtype=np.float32)
        g.pos = np.asarray(pos, dtype=np.float32)
        g.src = np.asarray(src, dtype=np.int64)
        g.dst = np.asarray(dst, dtype=np.int64)
        g.etype = np.asarray(et, dtype=np.int64)
        g.eattr = eattr
        g.track_rows = track_rows
        g.track_tids = track_tids
        g.link_rows = link_rows
        g.names = names
        return g


def diff(prev, cur):
    """Edge insertions and deletions between two snapshots: {etype: (added, removed)}."""
    if prev is None:
        return {}
    a, b = prev.edge_set(), cur.edge_set()
    out = {}
    for typ, s, d in b - a:
        out.setdefault(typ, [[], []])[0].append((s, d))
    for typ, s, d in a - b:
        out.setdefault(typ, [[], []])[1].append((s, d))
    return out
