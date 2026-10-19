"""
Radar track manager: the jet's persistent picture of every missile.

Each frame the radar reports the missiles it can see -- a track number, a
position, a velocity. Nothing else. The tracker turns that stream into tracks
whose identity survives from frame to frame, including frames where terrain
hides the missile and the track has to be dead-reckoned (coasted).

Per track it keeps:

  * kinematics  -- the latest radar return, or the coasted estimate
  * history     -- recent headings, from which turning is observable
  * belief      -- where the missile itself last saw the jet. This is derived
                   from events on the dynamic graph: while the seeker-geometry
                   edge track->jet exists, the missile can see the jet; when that
                   edge is deleted, the jet's position at that instant is what
                   the missile is left chasing.
  * estimates   -- the temporal GNN's latest read on the hidden guidance
                   parameters (turn rate, lead, flank side and angle, how much
                   it weaves, lock, datalink)

Lifecycle events (born, occluded, reacquired, destroyed, dropped) are appended
to `events` for the scene graph and the HUD ticker.
"""

import math
from collections import deque

from ..config import BALLISTIC_LIFE_FRAMES, HEIGHT, HOMING_LIFE_FRAMES, WIDTH
from ..utils import normalize_angle
from .doctrine import (AVOID_FAR, MEMORY_REACHED, PINCER_FLANK, SEEKER_COS_FOV,
                       SEEKER_RANGE, avoid_turn)
from .geometry import inside_any, segment_blocked

MAX_COAST_FRAMES = 120     # an unseen track is forgotten after this long
HISTORY          = 9       # headings kept per track
PURSUIT_MIN      = 0.35    # below this the jet treats a track as unguided
LEAD_MIN         = 0.08    # lead estimates below this are noise
# A weaving or darting missile can only be predicted along its average line --
# nobody can see the phase of a weave or the next dart coming. The world model
# counts it as closer than that line by this many px per unit of estimated
# swing (a swing of 0.4 -- about 22 degrees -- is a 16 px margin).
WOBBLE_MARGIN_PX = 40.0

# Before the network has seen a track: turn rate is unknown, so the planner
# assumes the most agile missile in the threat library (turn_hi).
PRIOR = {'pursuit': 0.5, 'turn_rate': 0.10, 'turn_hi': 0.16, 'lead': 0.0,
         'side': 0.0, 'guided': 0.5, 'threat': 0.0, 'bias': 0.0,
         'flank': 0.0, 'wobble': 0.0}


def doctrine_sees(mx, my, mh, px, py, terrain):
    """Could a seeker at (mx, my) pointing along mh see (px, py)? Doctrine only."""
    dx, dy = px - mx, py - my
    d = math.hypot(dx, dy)
    if d > SEEKER_RANGE:
        return False
    if dx * math.cos(mh) + dy * math.sin(mh) < SEEKER_COS_FOV * d:
        return False
    return not (terrain and segment_blocked(mx, my, px, py, terrain))


class Track(object):
    __slots__ = ("tid", "x", "y", "vx", "vy", "heading", "speed", "visible",
                 "born", "last_seen", "coast", "heads", "last_known",
                 "seeker_edge", "est", "group", "birth_x", "birth_y")

    def __init__(self, tid, frame, x, y, jet):
        self.tid = tid
        self.x = self.birth_x = x
        self.y = self.birth_y = y
        self.vx = self.vy = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.visible = True
        self.born = frame
        self.last_seen = frame
        self.coast = 0
        self.heads = deque(maxlen=HISTORY)
        # A launch is cued with the jet's position, so that is the first belief.
        self.last_known = (jet.x, jet.y)
        self.seeker_edge = False
        self.est = dict(PRIOR)
        self.group = -1

    def observe(self, frame, x, y, vx, vy):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.speed = math.hypot(vx, vy)
        self.heading = math.atan2(vy, vx)
        self.heads.append(self.heading)
        self.visible = True
        self.last_seen = frame
        self.coast = 0

    def turn_per_frame(self, span=4):
        """Mean heading change per frame over the last `span` frames."""
        if len(self.heads) < 2:
            return 0.0
        n = min(span, len(self.heads) - 1)
        return normalize_angle(self.heads[-1] - self.heads[-1 - n]) / n

    def dead_reckon(self, terrain=()):
        """
        Advance an unseen track one frame the way doctrine says it flies blind.

        Only a track the network believes is guided is steered toward the point
        it last saw the jet, and around any peak in its way; anything else
        coasts straight. Steering on a coin flip sends a hidden ballistic shot
        curving into the nearest peak, and the track gets deleted while the real
        missile flies on.
        """
        self.coast += 1
        guided = self.est['pursuit'] > 0.5
        tr = self.est['turn_rate']
        turn = 0.0
        aim_d = AVOID_FAR
        if guided and self.last_known is not None:
            lx, ly = self.last_known
            if math.hypot(lx - self.x, ly - self.y) < MEMORY_REACHED:
                self.last_known = None
            else:
                ex, ey = lx - self.x, ly - self.y
                aim_d = math.sqrt(ex * ex + ey * ey)
                diff = normalize_angle(math.atan2(ly - self.y, lx - self.x) - self.heading)
                turn = max(-tr, min(tr, diff))
        if guided and terrain:
            av = avoid_turn(self.x, self.y, math.cos(self.heading), math.sin(self.heading),
                            self.speed, tr, terrain, aim_d)
            if av != 0.0:
                turn = av * tr
        self.heading += turn
        if guided:
            self.heading += self.est.get('bias', 0.0)
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.x += self.vx
        self.y += self.vy
        self.heads.append(self.heading)

    @property
    def age(self):
        return self.last_seen - self.born + self.coast


class Tracker(object):
    """All tracks for one match."""

    def __init__(self, level):
        self.level = level
        self.terrain = [tuple(o) for o in level.obstacles]
        self.tracks = {}
        self.events = deque(maxlen=64)     # (frame, kind, tid, detail)
        self.event_count = 0               # total ever logged; the deque only keeps the last 64
        self.dropped = {}                  # tid -> frame dropped, to recognise a comeback
        # Conformal correction added to the network's turn-rate quantile
        # (training/calibrate.py); 0 flies on the raw quantile.
        self.turn_margin = 0.0

    def _event(self, frame, kind, tid, detail=""):
        self.events.append((frame, kind, tid, detail))
        self.event_count += 1

    def update(self, frame, jet, visible_missiles):
        seen = set()
        for m in visible_missiles:
            t = self.tracks.get(m.uid)
            if t is None:
                t = self.tracks[m.uid] = Track(m.uid, frame, m.x, m.y, jet)
                gone = self.dropped.pop(m.uid, None)
                if gone is not None:
                    # Same radar track number: the node comes back under its old
                    # key, but its memory was deleted when it was dropped.
                    self._event(frame, "reacquired", m.uid, "after drop")
                else:
                    self._event(frame, "born", m.uid)
            elif not t.visible:
                self._event(frame, "reacquired", m.uid, "after %d f" % t.coast)
            t.observe(frame, m.x, m.y, m.vx, m.vy)
            seen.add(m.uid)

        blind_terrain = self.terrain if self.level.smart else ()
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if tid in seen:
                self._update_belief(t, jet)
                continue
            if t.visible:
                # Gone from the radar. Behind terrain it is merely hidden; in the
                # open, a missile that vanishes has been destroyed. Test where it
                # would be *now* -- its last return was, by definition, still in
                # view, so testing that spot calls every occlusion a kill.
                if self.terrain and any(
                        segment_blocked(jet.x, jet.y, t.x + s * t.vx, t.y + s * t.vy, self.terrain)
                        for s in (0.0, 1.0, 2.0)):
                    t.visible = False
                    t.seeker_edge = False
                    self._event(frame, "occluded", tid)
                else:
                    del self.tracks[tid]
                    self._event(frame, "destroyed", tid)
                    continue
            t.dead_reckon(blind_terrain)
            if (t.coast > MAX_COAST_FRAMES or not (0 <= t.x <= WIDTH and 0 <= t.y <= HEIGHT)
                    or inside_any(t.x, t.y, self.terrain, 2.0)):
                del self.tracks[tid]
                self.dropped[tid] = frame
                self._event(frame, "dropped", tid, "coasted %d f" % t.coast)
        if len(self.dropped) > 200:
            self.dropped = {k: f for k, f in self.dropped.items() if frame - f < 600}

    def _update_belief(self, t, jet):
        """Keep the track's estimated last-known point in step with doctrine."""
        if not self.level.smart:
            t.last_known = (jet.x, jet.y)
            t.seeker_edge = True
            return
        sees = doctrine_sees(t.x, t.y, t.heading, jet.x, jet.y, self.terrain)
        if sees:
            t.last_known = (jet.x, jet.y)
        elif t.last_known is not None and math.hypot(t.last_known[0] - t.x,
                                                     t.last_known[1] - t.y) < MEMORY_REACHED:
            t.last_known = None
        t.seeker_edge = sees

    # ── estimates in, world-model inputs out ─────────────────
    def apply_estimates(self, tids, rows, link_pairs=()):
        """
        rows[i] = dict of the GNN's estimates for track tids[i]; link_pairs is
        [(tid_a, tid_b, p)]. Datalink groups are the connected components of
        links the network believes are active.
        """
        for tid, est in zip(tids, rows):
            t = self.tracks.get(tid)
            if t is not None:
                t.est = est
        parent = {tid: tid for tid in self.tracks}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b, p in link_pairs:
            if p >= 0.5 and a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        sizes = {}
        for tid in self.tracks:
            r = find(tid)
            sizes[r] = sizes.get(r, 0) + 1
        for tid, t in self.tracks.items():
            r = find(tid)
            t.group = r if sizes[r] > 1 else -1

    def threat_dicts(self, frame=None):
        """
        Every track as a world-model input, guidance parameters from the estimates.

        Turn rate uses the network's upper (90%) quantile, not its mean. A
        missile flying straight at the jet shows no turning, so its turn rate is
        unobservable exactly when it matters. The quantile is wide when the
        network has not seen the missile manoeuvre and tight once it has -- so a
        sluggish missile, once identified, is still planned around as sluggish.

        A flanker's side and angle are planned exactly; a weaver's or erratic
        missile's swing cannot be, so its estimated size becomes a `margin`.

        With `frame` (the frame the rollout's first step simulates) each track
        also carries its remaining flight time. That is doctrine, like seeker
        range: a guided missile of this class flies HOMING_LIFE_FRAMES, an
        unguided one BALLISTIC_LIFE_FRAMES. It is counted from the frame the
        radar first saw the track -- launch, in the open; later, behind terrain
        or after a drop -- so the estimate can only err on the long, safe side.
        """
        out = []
        smart = self.level.smart
        for t in self.tracks.values():
            e = t.est
            guided = e['pursuit'] >= PURSUIT_MIN
            d = {'pos': [t.x, t.y], 'vel': [t.vx, t.vy], 'heading': t.heading,
                 'speed': t.speed, 'radius': 2, 'tid': t.tid,
                 'turn_rate': e.get('turn_hi', e['turn_rate']) + self.turn_margin}
            if frame is not None:
                life = HOMING_LIFE_FRAMES if guided else BALLISTIC_LIFE_FRAMES
                d['ttl'] = life - (frame - t.born)
            if guided:
                d['lateral_bias'] = e.get('bias', 0.0)
                d['lead'] = e['lead'] if e['lead'] >= LEAD_MIN else 0.0
                d['side'] = float(round(e['side']))
                d['flank'] = e.get('flank', PINCER_FLANK)
                d['margin'] = WOBBLE_MARGIN_PX * e.get('wobble', 0.0)
                if smart:
                    d['seeker'] = True
                    d['group'] = t.group if self.level.datalink else -1
                    d['last_known'] = t.last_known
                else:
                    # classic homing missiles push apart when they bunch up
                    d['repel'] = True
            out.append(d)
        return out
