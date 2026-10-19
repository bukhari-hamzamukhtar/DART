"""
Simulation objects (main3.py section 5).

Each class owns its own physics (`update`), lifetime rule (`is_dead`) and
rendering (`draw`).  Behaviour is unchanged from main3.py; only the shared
constants now come from `config`.
"""

import math
import random
from collections import deque

import pygame

from ..config import (BALLISTIC_COLOR, BASE_SPEED, FLARE_COLOR,
                      GUIDANCE_BALLISTIC, GUIDANCE_HOMING, HEIGHT, JET_COLOR,
                      MAX_THROTTLE, MIN_THROTTLE, MISSILE_COLOR, WIDTH)
from ..utils import normalize_angle


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

    def __init__(self, x, y, tx, ty):
        self.x, self.y   = float(x), float(y)
        self.heading     = math.atan2(ty-y, tx-x)
        self.speed       = 4.0
        self.radius      = 4
        self.spawn_time  = pygame.time.get_ticks()
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


class FlareParticle:
    """Decoy the jet ejects; homing missiles that touch one are destroyed."""

    def __init__(self, x, y, angle, speed):
        self.x, self.y  = float(x), float(y)
        self.vx = math.cos(angle)*speed
        self.vy = math.sin(angle)*speed
        self.drag       = 0.85
        self.radius     = 6
        self.spawn_time = pygame.time.get_ticks()

    def update(self):
        self.vx*=self.drag; self.vy*=self.drag
        self.x+=self.vx; self.y+=self.vy

    def is_dead(self): return pygame.time.get_ticks()-self.spawn_time>2000

    def draw(self, s): pygame.draw.circle(s,FLARE_COLOR,(int(self.x),int(self.y)),self.radius)


class HomingMissile:
    """Tracking missile — turns toward the jet at a limited turn rate."""

    mtype = GUIDANCE_HOMING     # what the jet's radar reports; matches training data

    def __init__(self, x, y, target, tx, ty):
        self.x, self.y   = float(x), float(y)
        self.target      = target
        self.heading     = math.atan2(ty-y, tx-x)
        self.speed       = 5.0
        self.max_speed   = 5.0
        self.turn_rate   = 0.15
        self.radius      = 2
        self.spawn_time  = pygame.time.get_ticks()
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead        = False
        self.trail       = deque(maxlen=12)
        self.spawn_delay = 0
        self.lateral_bias = 0.0   # persistent curve for cluster formation spread

    def update(self):
        if self.spawn_delay>0: self.spawn_delay-=1; return
        dx,dy = self.target.x-self.x, self.target.y-self.y
        dh    = math.atan2(dy,dx)
        diff  = normalize_angle(dh-self.heading)
        self.heading += max(-self.turn_rate, min(self.turn_rate, diff))
        self.heading += self.lateral_bias  # cluster spread: each missile drifts its own way
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.x += self.vx; self.y += self.vy
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

    def __init__(self, x, y, tx, ty):
        self.x, self.y  = float(x), float(y)
        self.heading    = math.atan2(ty-y, tx-x)
        self.speed      = 7.0
        self.radius     = 2
        self.spawn_time = pygame.time.get_ticks()
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
    """Ground battery. Two per match; both destroyed == AI wins."""

    FIRE_COOLDOWN_MS = 2000

    def __init__(self, x, y, color):
        self.x,self.y=x,y; self.color=color
        self.last_fire=pygame.time.get_ticks()-2001; self.alive=True

    def can_fire(self):
        return self.alive and pygame.time.get_ticks()-self.last_fire>=self.FIRE_COOLDOWN_MS

    def fire(self): self.last_fire=pygame.time.get_ticks()

    def draw(self, s):
        if not self.alive:
            pygame.draw.rect(s,(40,40,40),(self.x-15,self.y-10,30,20),2)
            pygame.draw.line(s,(150,40,40),(self.x-10,self.y-15),(self.x+10,self.y+5),2)
            pygame.draw.line(s,(150,40,40),(self.x-10,self.y+5),(self.x+10,self.y-15),2)
            return
        pygame.draw.rect(s,self.color,(self.x-15,self.y-10,30,20),2)
        pygame.draw.circle(s,self.color,(self.x,self.y),5)
