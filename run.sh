#!/usr/bin/env bash
# Python GBC Emulator - Linux / macOS launcher
# Installs pygame and numpy if missing, then starts the emulator.
set -e

# Move into the directory containing this script so paths are stable.
cd "$(dirname "$0")"

echo "-----------------------------------------"
echo "  Python Game Boy / Game Boy Color Emulator"
echo "-----------------------------------------"
echo

# --- Locate Python 3 ---------------------------------------------------------
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "[ERROR] Python 3.9+ not found. Install it from https://python.org"
    echo "        On Debian/Ubuntu:  sudo apt install python3 python3-pip"
    echo "        On Fedora:         sudo dnf install python3 python3-pip"
    echo "        On macOS:          brew install python"
    exit 1
fi

# --- Install dependencies if missing ----------------------------------------
echo "Checking dependencies..."
if ! "$PY" -c "import pygame; import numpy" >/dev/null 2>&1; then
    echo "Installing pygame and numpy..."
    if ! "$PY" -m pip install --user -r requirements.txt; then
        echo "[WARN]  Could not auto-install dependencies."
        echo "        Run:  $PY -m pip install --user pygame numpy"
    fi
fi

echo
echo "Starting emulator..."
echo
echo "  Controls:"
echo "    Arrow keys - D-pad"
echo "    Z          - A button"
echo "    X          - B button"
echo "    Right Shift- Select"
echo "    Enter      - Start"
echo "    Escape     - Return to menu / Quit"
echo
echo "  In menu:  Arrow keys to navigate, Enter to select, Esc to go back"
echo

"$PY" "$(dirname "$0")/gbc_emulator_skeleton.py"
status=$?

if [ $status -ne 0 ]; then
    echo
    echo "[ERROR] Emulator exited with status $status. See the message above."
    exit $status
fi
