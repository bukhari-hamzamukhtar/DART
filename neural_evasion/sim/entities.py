"""
Simulation objects (main3.py section 5, plus the level 2/3 additions).

Each class owns its own physics (`update`) and rendering (`draw`). Lifetimes are
counted in frames -- `life_frames`, checked by the arena against `born_frame` --
so a headless run and a 60 fps window age things identically. The classic
classes keep their old wall-clock `is_dead()` for anything that still calls it.

Missiles and flares get a `uid` at creation: the radar's track number. The
jet's tracker keys its memory on it, and nothing on the jet's side reads any
other attribute of a missile.

Guided missiles (HomingMissile, SmartMissile) also carry a flight style --
pursuit, flanker, weaver or erratic (`sim/doctrine.py`). A missile built directly
flies plain pursuit; the arena gives every missile it launches a style.
"""

import itertools
import math
import random
from collections import deque

import pygame

from ..config import (BALLISTIC_COLOR, BALLISTIC_LIFE_FRAMES, BASE_SPEED,
                      FLARE_COLOR, FLARE_LIFE_FRAMES, GUIDANCE_BALLISTIC,
                      GUIDANCE_HOMING, GUIDANCE_SMART, HEIGHT, HOMING_LIFE_FRAMES,
                      JET_COLOR, JET_MISSILE_LIFE_FRAMES, LAUNCHER_COOLDOWN_FRAMES,
                      MAX_THROTTLE, MIN_THROTTLE, MISSILE_COLOR, TERRAIN_CONTOUR,
                      TERRAIN_FILL, WIDTH)
from ..utils import normalize_angle
from .doctrine import (AVOID_FAR, DECOY_RANGE, FLANKER, MEMORY_REACHED,
                       PINCER_FLANK, PURSUIT_STYLE, SEEKER_COS_FOV, SEEKER_RANGE,
                       SMART_SPEED, SMART_TURN_RATE, avoid_turn, guidance_aim,
                       jink_value, wobble_at)
from .geometry import segment_blocked

_UIDS = itertools.count(1)

STYLE_FIELDS = ('style', 'side', 'flank', 'weave_amp', 'weave_freq', 'weave_phase',
                'jink_amp', 'jink_period', 'jink_seed')


def new_uid():
    """Next radar track number. Unique for the life of the process."""
    return next(_UIDS)


def apply_style(m, style):
    """Give a guided missile a flight style (a dict from doctrine.sample_style)."""
    for k in STYLE_FIELDS:
        setattr(m, k, style[k])


def _style_swing(m, age):
    """A missile's aim swing (tangent of the angle) at flight age `age`."""
    jink = jink_value(m.jink_seed, age // m.jink_period) if m.jink_amp else 0.0
    return wobble_at(m.weave_amp, m.weave_freq, m.weave_phase, m.jink_amp, jink, age)


class Jet:
    """The AI-flown aircraft. Heading is set by the brain; physics is drag-based."""

    def __init__(self, x, y):
        self.x, self.y     = float(x), float(y)
        self.vx, self.vy   = 0.0, 0.0
        self.heading       = 0.0
        self.speed         = BASE_SPEED      # commanded throttle, jet controls it
        self.max_speed     = BASE_SPEED
        self.drag          = 0.92
        self.radius        = 11
        self.last_flare_time  = 0
        self.last_attack_time = pygame.time.get_ticks()
        self.trail = deque(maxlen=20)

    def set_throttle(self, speed):
        """Command a new speed, clamped to the jet's physical envelope."""
        self.speed = max(MIN_THROTTLE, min(MAX_THROTTLE, speed))

    def update(self):
        self.vx = self.vx*self.drag + math.cos(self.heading)*self.speed*0.1
        self.vy = self.vy*self.drag + math.sin(self.heading)*self.speed*0.1
        self.x += self.vx
        self.y += self.vy
        self.trail.append((self.x, self.y))

    def draw(self, surface, ai_maneuver):
        for i, (tx, ty) in enumerate(self.trail):
            a = int((i/max(len(self.trail),1))*160)
            r = max(1, int((i/max(len(self.trail),1))*3))
            pygame.draw.circle(surface, (0,255,200,a), (int(tx),int(ty)), r)
        R = self.radius
        tip    = ( R*2.2,  0.0)
        back_l = (-R*1.0,  R*1.2)
        back_r = (-R*1.0, -R*1.2)
        notch  = (-R*0.2,  0.0)
        raw    = [tip, back_l, notch, back_r]
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        def rot(px,py): return (self.x+px*ch-py*sh, self.y+px*sh+py*ch)
        pts = [rot(px,py) for px,py in raw]
        if ai_maneuver == "BARREL_ROLL":
            rs = max(0.15, abs(math.cos(pygame.time.get_ticks()*0.012)))
            cy_avg = sum(p[1] for p in pts)/len(pts)
            pts = [(p[0], cy_avg+(p[1]-cy_avg)*rs) for p in pts]
        pygame.draw.polygon(surface, (0,255,200,35),  pts)
        pygame.draw.polygon(surface, (0,255,200,100), pts)
        pygame.draw.polygon(surface, (0,255,200),     pts)
        pygame.draw.polygon(surface, (255,255,255),   pts, 1)
        tx2, ty2 = rot(-R*0.9, 0)
        pygame.draw.circle(surface, (0,255,200,180), (int(tx2),int(ty2)), 4)
        pygame.draw.circle(surface, (255,255,200,80), (int(tx2),int(ty2)), 7)


class JetMissile:
    """Straight-line missile the jet fires back at the ground launchers."""

    life_frames = JET_MISSILE_LIFE_FRAMES

    def __init__(self, x, y, tx, ty):
        self.x, self.y   = float(x), float(y)
        self.heading     = math.atan2(ty-y, tx-x)
        self.speed       = 4.0
        self.radius      = 4
        self.spawn_time  = pygame.time.get_ticks()
        self.born_frame  = 0
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead  = False
        self.trail = deque(maxlen=12)

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.trail.append((self.x, self.y))
        if self.x<0 or self.x>WIDTH or self.y<0 or self.y>HEIGHT:
            self.dead = True

    def is_dead(self): return self.dead or pygame.time.get_ticks()-self.spawn_time>4000

    def draw(self, s):
        for i,(tx,ty) in enumerate(self.trail):
            a=int((i/len(self.trail))*180)
            pygame.draw.circle(s,(*JET_COLOR,a),(int(tx),int(ty)),max(1,int((i/len(self.trail))*2)+1))
        pygame.draw.circle(s,JET_COLOR,(int(self.x),int(self.y)),self.radius)


FLARE_DRAG   = 0.85     # the world model drifts flares with these same numbers
FLARE_RADIUS = 6


class FlareParticle:
    """Decoy the jet ejects; a missile that touches one is destroyed."""

    life_frames = FLARE_LIFE_FRAMES

    def __init__(self, x, y, angle, speed):
        self.uid = new_uid()
        self.x, self.y  = float(x), float(y)
        self.vx = math.cos(angle)*speed
        self.vy = math.sin(angle)*speed
        self.drag       = FLARE_DRAG
        self.radius     = FLARE_RADIUS
        self.spawn_time = pygame.time.get_ticks()
        self.born_frame = None     # stamped by the arena the frame it appears
        self.dead       = False

    def update(self):
        self.vx*=self.drag; self.vy*=self.drag
        self.x+=self.vx; self.y+=self.vy

    def is_dead(self): return pygame.time.get_ticks()-self.spawn_time>2000

    def draw(self, s): pygame.draw.circle(s,FLARE_COLOR,(int(self.x),int(self.y)),self.radius)


class HomingMissile:
    """Tracking missile — turns toward the jet at a limited turn rate, in its flight style."""

    mtype = GUIDANCE_HOMING     # what the jet's radar reports; matches training data
    life_frames = HOMING_LIFE_FRAMES
    mode = "PURSUIT"

    def __init__(self, x, y, target, tx, ty):
        self.uid = new_uid()
        self.x, self.y   = float(x), float(y)
        self.target      = target
        self.heading     = math.atan2(ty-y, tx-x)
        self.speed       = 5.0
        self.max_speed   = 5.0
        self.turn_rate   = 0.15
        self.radius      = 2
        self.spawn_time  = pygame.time.get_ticks()
        self.born_frame  = 0
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead        = False
        self.trail       = deque(maxlen=12)
        self.spawn_delay = 0
        self.lateral_bias = 0.0   # persistent curve for cluster formation spread
        self.age         = 0      # frames flown since leaving the rail
        apply_style(self, PURSUIT_STYLE)

    def update(self):
        if self.spawn_delay>0: self.spawn_delay-=1; return
        a = self.age
        # Plain pursuit aims at the jet itself; a flanker, weaver or erratic
        # missile aims off the line of sight.
        tx, ty = guidance_aim(self.x, self.y, self.speed, self.target.x, self.target.y,
                              self.target.vx, self.target.vy, 0.0, self.side, self.flank,
                              _style_swing(self, a))
        dx,dy = tx-self.x, ty-self.y
        dh    = math.atan2(dy,dx)
        diff  = normalize_angle(dh-self.heading)
        self.heading += max(-self.turn_rate, min(self.turn_rate, diff))
        self.heading += self.lateral_bias  # cluster spread: each missile drifts its own way
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.x += self.vx; self.y += self.vy
        self.age = a + 1
        self.trail.append((self.x,self.y))
        if self.x<0 or self.x>WIDTH or self.y<0 or self.y>HEIGHT: self.dead=True

    def is_dead(self): return self.dead or pygame.time.get_ticks()-self.spawn_time>8500

    def draw(self, s):
        for i,(tx,ty) in enumerate(self.trail):
            a=int((i/len(self.trail))*180)
            pygame.draw.circle(s,(*MISSILE_COLOR,a),(int(tx),int(ty)),max(1,int((i/len(self.trail))*2)+1))
        pygame.draw.circle(s,MISSILE_COLOR,(int(self.x),int(self.y)),self.radius)


class BallisticMissile:
    """Predictive shot — no tracking, ignores flares, faster than the homing type."""

    mtype     = GUIDANCE_BALLISTIC
    turn_rate = 0.0             # never steers: the world model needs this explicitly
    lateral_bias = 0.0
    spawn_delay  = 0
    life_frames  = BALLISTIC_LIFE_FRAMES
    mode = "BALLISTIC"

    def __init__(self, x, y, tx, ty):
        self.uid = new_uid()
        self.x, self.y  = float(x), float(y)
        self.heading    = math.atan2(ty-y, tx-x)
        self.speed      = 7.0
        self.radius     = 2
        self.spawn_time = pygame.time.get_ticks()
        self.born_frame = 0
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead  = False
        self.trail = deque(maxlen=12)

    def update(self):
        self.x+=self.vx; self.y+=self.vy
        self.trail.append((self.x,self.y))
        if self.x<0 or self.x>WIDTH or self.y<0 or self.y>HEIGHT: self.dead=True

    def is_dead(self): return self.dead or pygame.time.get_ticks()-self.spawn_time>3000

    def draw(self, s):
        for i,(tx,ty) in enumerate(self.trail):
            a=int((i/len(self.trail))*180)
            pygame.draw.circle(s,(*BALLISTIC_COLOR,a),(int(tx),int(ty)),max(1,int((i/len(self.trail))*2)+1))
        pygame.draw.circle(s,BALLISTIC_COLOR,(int(self.x),int(self.y)),self.radius)


SMART_MODE_COLORS = {
    "LOCK":   (255, 68, 68),      # own seeker has the jet
    "LINK":   (90, 200, 255),     # guided by a salvo partner's lock
    "MEMORY": (255, 210, 70),     # flying to where it last knew the jet was
    "DECOY":  (255, 150, 40),     # seduced by a flare
    "SEARCH": (150, 150, 160),    # nothing to guide on
    "CUE":    (255, 120, 120),    # just launched, on the battery's cue
}


class SmartMissile:
    """
    Seeker missile for levels 2 and 3.

    Guidance runs in two passes so a salvo can share a lock:

      1. `sense`  -- every missile checks what its own seeker can see (range,
                     field of view, terrain) before anything moves this frame;
      2. the arena marks missiles whose salvo partner holds a lock as `linked`;
      3. `update` -- steer and move.

    Priority when steering: a flare it has been seduced by, then the jet (own
    lock or datalink, with lead and its flight style's deviation), then the last
    place it knew the jet was, then straight ahead. Whatever it is chasing, it
    turns hard away from a peak it is about to fly into.
    """

    mtype = GUIDANCE_SMART
    life_frames = HOMING_LIFE_FRAMES

    def __init__(self, x, y, target, cue_x, cue_y, speed=SMART_SPEED,
                 turn_rate=SMART_TURN_RATE, lead=0.0, side=0.0, network=None,
                 lateral_bias=0.0, spawn_delay=0, launch_heading=None, flank=PINCER_FLANK):
        self.uid = new_uid()
        self.x, self.y = float(x), float(y)
        self.target = target
        self.heading = (math.atan2(cue_y - y, cue_x - x)
                        if launch_heading is None else float(launch_heading))
        self.speed = self.max_speed = float(speed)
        self.turn_rate = float(turn_rate)
        self.lead = float(lead)
        apply_style(self, PURSUIT_STYLE)
        # A pincer member is a flanker at the pincer's deviation.
        self.side = float(side)
        self.flank = float(flank)
        if self.side != 0.0:
            self.style = FLANKER
        self.network = network          # salvo id shared over the datalink, or None
        self.lateral_bias = float(lateral_bias)
        self.spawn_delay = int(spawn_delay)
        self.radius = 2
        self.spawn_time = pygame.time.get_ticks()
        self.born_frame = 0
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.dead = False
        self.trail = deque(maxlen=12)
        self.last_known = (float(cue_x), float(cue_y))   # the battery's cue
        self.lock = False
        self.linked = False
        self.decoy = None
        self.mode = "CUE"
        self.age = 0                    # frames flown since leaving the rail
        self.obstacles = ()             # the terrain it steers around, from sense()
        self.avoiding = False

    # ── pass 1: what can the seeker see ──────────────────────
    def sees(self, px, py, obstacles, max_range=SEEKER_RANGE):
        # hypot, not a squared compare: the world model uses the same formula,
        # so the two agree about who sees whom down to rounding.
        dx = px - self.x
        dy = py - self.y
        d = math.hypot(dx, dy)
        if d > max_range:
            return False
        if (dx * math.cos(self.heading) + dy * math.sin(self.heading)) < SEEKER_COS_FOV * d:
            return False
        return not (obstacles and segment_blocked(self.x, self.y, px, py, obstacles))

    def sense(self, jet, flares, obstacles):
        self.obstacles = obstacles
        self.lock = False
        self.linked = False
        self.decoy = None
        if self.spawn_delay > 0:
            return
        self.lock = self.sees(jet.x, jet.y, obstacles)
        jet_d2 = (jet.x - self.x) ** 2 + (jet.y - self.y) ** 2
        best = None
        for f in flares:
            if getattr(f, "dead", False):
                continue
            fd2 = (f.x - self.x) ** 2 + (f.y - self.y) ** 2
            # With the jet in view a flare only wins if it is closer; with the
            # jet out of view the seeker grabs whatever it can see.
            if self.lock and fd2 >= jet_d2:
                continue
            if best is not None and fd2 >= best[0]:
                continue
            if self.sees(f.x, f.y, obstacles, DECOY_RANGE):
                best = (fd2, f)
        if best is not None:
            self.decoy = best[1]

    # ── pass 3: steer and move ───────────────────────────────
    def update(self):
        if self.spawn_delay > 0:
            self.spawn_delay -= 1
            return
        jet = self.target
        a = self.age
        aim = None
        if self.decoy is not None:
            aim = (self.decoy.x, self.decoy.y)
            self.mode = "DECOY"
        elif self.lock or self.linked:
            aim = guidance_aim(self.x, self.y, self.speed, jet.x, jet.y,
                               jet.vx, jet.vy, self.lead, self.side, self.flank,
                               _style_swing(self, a))
            self.last_known = (jet.x, jet.y)
            self.mode = "LOCK" if self.lock else "LINK"
        elif self.last_known is not None:
            lx, ly = self.last_known
            if math.hypot(lx - self.x, ly - self.y) < MEMORY_REACHED:
                self.last_known = None
                self.mode = "SEARCH"
            else:
                aim = self.last_known
                self.mode = "MEMORY"
        else:
            self.mode = "SEARCH"

        turn = 0.0
        aim_d = AVOID_FAR
        if aim is not None:
            ex = aim[0] - self.x
            ey = aim[1] - self.y
            aim_d = math.sqrt(ex * ex + ey * ey)
            diff = normalize_angle(math.atan2(aim[1] - self.y, aim[0] - self.x) - self.heading)
            turn = max(-self.turn_rate, min(self.turn_rate, diff))
        av = 0.0
        if self.obstacles:
            av = avoid_turn(self.x, self.y, math.cos(self.heading), math.sin(self.heading),
                            self.speed, self.turn_rate, self.obstacles, aim_d)
        self.avoiding = av != 0.0
        if av != 0.0:
            turn = av * self.turn_rate
        self.heading += turn
        self.heading += self.lateral_bias
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.x += self.vx
        self.y += self.vy
        self.age = a + 1
        self.trail.append((self.x, self.y))
        if self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            self.dead = True

    def draw(self, s):
        col = SMART_MODE_COLORS.get(self.mode, MISSILE_COLOR)
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            a = int((i / n) * 180)
            pygame.draw.circle(s, (*col, a), (int(tx), int(ty)), max(1, int((i / n) * 2) + 1))
        pygame.draw.circle(s, col, (int(self.x), int(self.y)), self.radius + 1)


class Obstacle:
    """A terrain peak: blocks line of sight, destroys whatever flies into it."""

    def __init__(self, x, y, r):
        self.x, self.y, self.r = float(x), float(y), float(r)

    def draw(self, surface):
        cx, cy, r = int(self.x), int(self.y), int(self.r)
        pygame.draw.circle(surface, TERRAIN_FILL, (cx, cy), r)
        # Topographic contours, tighter toward the summit.
        for k, frac in enumerate((1.0, 0.74, 0.5, 0.28)):
            shade = tuple(min(255, int(c * (0.7 + 0.25 * k))) for c in TERRAIN_CONTOUR)
            pygame.draw.circle(surface, shade, (cx, cy), max(2, int(r * frac)), 1)
        pygame.draw.circle(surface, (120, 190, 150), (cx, cy), 2)


class ExplosionParticle:
    def __init__(self, x, y, color):
        self.x,self.y = float(x),float(y)
        a = random.uniform(0,math.pi*2); sp = random.uniform(2,8)
        self.vx=math.cos(a)*sp; self.vy=math.sin(a)*sp
        self.radius=random.uniform(2,5); self.color=color; self.life=255

    def update(self):
        self.vx*=0.9; self.vy*=0.9
        self.x+=self.vx; self.y+=self.vy
        self.life-=15; self.radius-=0.15

    def is_dead(self): return self.life<=0 or self.radius<=0

    def draw(self, s):
        if self.life>0 and self.radius>0:
            pygame.draw.circle(s,self.color,(int(self.x),int(self.y)),int(self.radius))


class Shockwave:
    def __init__(self, x, y, color):
        self.x,self.y=float(x),float(y); self.radius=8; self.color=color; self.alpha=200

    def update(self): self.radius+=2.5; self.alpha-=15

    def is_dead(self): return self.alpha<=0

    def draw(self, s):
        if self.alpha>0:
            pygame.draw.circle(s,(*self.color,int(self.alpha)),(int(self.x),int(self.y)),int(self.radius),2)


class Launcher:
    """Ground battery. All destroyed == AI wins."""

    FIRE_COOLDOWN_MS = 2000
    FIRE_COOLDOWN_FRAMES = LAUNCHER_COOLDOWN_FRAMES

    def __init__(self, x, y, color):
        self.x,self.y=x,y; self.color=color
        self.last_fire=pygame.time.get_ticks()-2001; self.alive=True
        self.last_fire_frame = -10 ** 6

    def can_fire(self):
        return self.alive and pygame.time.get_ticks()-self.last_fire>=self.FIRE_COOLDOWN_MS

    def fire(self): self.last_fire=pygame.time.get_ticks()

    # Frame-based equivalents, used by the arena.
    def ready(self, frame):
        return self.alive and frame - self.last_fire_frame >= self.FIRE_COOLDOWN_FRAMES

    def fire_at(self, frame):
        self.last_fire_frame = frame

    def draw(self, s):
        if not self.alive:
            pygame.draw.rect(s,(40,40,40),(self.x-15,self.y-10,30,20),2)
            pygame.draw.line(s,(150,40,40),(self.x-10,self.y-15),(self.x+10,self.y+5),2)
            pygame.draw.line(s,(150,40,40),(self.x-10,self.y+5),(self.x+10,self.y-15),2)
            return
        pygame.draw.rect(s,self.color,(self.x-15,self.y-10,30,20),2)
        pygame.draw.circle(s,self.color,(self.x,self.y),5)
