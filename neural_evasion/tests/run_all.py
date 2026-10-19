"""Run every check. Usage:  python neural_evasion/tests/run_all.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_equivalence.py", "test_worldmodel.py", "test_worldmodel_v3.py",
         "test_dynamic_graph.py", "test_rules.py", "test_maneuvers.py",
         "test_game_smoke.py", "test_screens.py"]

env = dict(os.environ, SDL_VIDEODRIVER="dummy")
failed = []
for name in TESTS:
    print("\n" + "=" * 62)
    print("  " + name)
    print("=" * 62)
    rc = subprocess.call([sys.executable, os.path.join(HERE, name)], env=env)
    if rc != 0:
        failed.append(name)

print("\n" + "=" * 62)
print("FAILED: %s" % failed if failed else "ALL CHECKS PASSED")
sys.exit(1 if failed else 0)
