#!/usr/bin/env python3
"""Deprecated alias for gbc_emulator.py. Kept so old launch commands still work."""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "gbc_emulator.py")
if not os.path.isfile(_TARGET):
    sys.exit("error: gbc_emulator.py not found next to this compatibility stub")
if sys.stderr.isatty():
    print("note: gbc_emulator_skeleton.py is deprecated; use gbc_emulator.py",
          file=sys.stderr)
sys.argv[0] = _TARGET
runpy.run_path(_TARGET, run_name="__main__")
