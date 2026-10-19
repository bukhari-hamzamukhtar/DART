"""Render a few frames of each full-screen menu."""
import os, sys, traceback
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import pygame
from neural_evasion.levels import LEVELS
from neural_evasion.ui import screens

class Done(Exception): pass

real_flip = pygame.display.flip
def run(fn, *a):
    n = {'i': 0}
    def flip():
        real_flip(); n['i'] += 1
        if n['i'] >= 5: raise Done()
    pygame.display.flip = flip
    try:
        fn(*a)
    except Done:
        print("PASS  %-22s rendered %d frames" % (fn.__name__, n['i']))
        return True
    except Exception:
        print("FAIL  " + fn.__name__); traceback.print_exc(); return False
    finally:
        pygame.display.flip = real_flip
    print("PASS  %s returned early" % fn.__name__); return True

ok  = run(screens.run_start_screen)
for lv in range(3):
    for ver in (1, 2, 3):
        ok &= run(screens.run_setup_screen, {'level': lv, 'version': ver})
for level in LEVELS:
    for ver in (1, 3):
        ok &= run(screens.run_briefing_screen, level, ver)
ok &= run(screens.run_gameover_screen, True,  {'fired': 12, 'flares': 40, 'maneuvers': 7},
          {'level': 'L2 TERRAIN MASKING', 'version': 'V3  DYNAMIC GRAPH', 'cause': 'terrain', 'seconds': 41.5})
ok &= run(screens.run_gameover_screen, False, {'fired': 3,  'flares': 9,  'maneuvers': 2})
sys.exit(0 if ok else 1)
