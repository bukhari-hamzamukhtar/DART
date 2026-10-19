"""Render a few frames of each full-screen menu."""
import os, sys, traceback
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import pygame
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
ok &= run(screens.run_gameover_screen, True,  {'fired': 12, 'flares': 40, 'maneuvers': 7})
ok &= run(screens.run_gameover_screen, False, {'fired': 3,  'flares': 9,  'maneuvers': 2})
sys.exit(0 if ok else 1)
