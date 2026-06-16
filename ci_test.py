"""CI wrapper: runs both smoke tests and exits non-zero on any failure.

Usage:
    python ci_test.py

Requires: pygame, numpy (tested with Python 3.9+)
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

tests = [
    ["python", os.path.join(HERE, "test_headless.py")],
    ["python", os.path.join(HERE, "test_save_state.py")],
]

failed = 0
for cmd in tests:
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    try:
        result = subprocess.run(cmd, cwd=HERE,
                                capture_output=False,
                                timeout=60)
        if result.returncode != 0:
            print(f"FAILED: {' '.join(cmd)} (exit code {result.returncode})")
            failed += 1
        else:
            print(f"PASSED: {' '.join(cmd)}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {' '.join(cmd)}")
        failed += 1
    except Exception as e:
        print(f"ERROR: {' '.join(cmd)} -> {e}")
        failed += 1

if failed:
    print(f"\n{failed} test(s) FAILED")
    sys.exit(1)
else:
    print("\nALL TESTS PASSED")
