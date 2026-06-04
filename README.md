# Python GBC Emulator

A Game Boy / Game Boy Color emulator written in Python, featuring a built-in
menu system, ROM browser, and MBC1/MBC5 cartridge support.

![status](https://img.shields.io/badge/status-playable-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)

## Features

- **CPU** — Complete LR35902 instruction set (standard + CB-prefixed),
  interrupt handling, HALT / wait states.
- **PPU** — Background, window, and 8×8 / 8×16 sprite rendering with
  DMG-style 4-shade palette, X/Y flip, and sprite-over-BG priority.
- **Timers** — DIV, TIMA, TMA, TAC with all four programmable rates and
  correct overflow → interrupt signalling.
- **Cartridge** — MBC1 and MBC5 (ROM / RAM banking, battery-backed RAM,
  rumble flag detection).
- **Menu system** — ROM browser, window-scale selector, keyboard controls,
  project logo.
- **Input** — D-pad, A / B, Start, Select via the standard joypad register
  layout.
- **Performance** — ~60–90 fps on SUPERBAJTEK (Python 3.11). Precomputed
  tile / palette lookup tables, unrolled scanline writers, and a
  combined per-opcode dispatcher keep the inner loop tight.

## Requirements

- Python 3.9 or newer
- [pygame](https://pypi.org/project/pygame/) ≥ 2.0 — display, input, icon
- [numpy](https://pypi.org/project/numpy/) ≥ 1.20 — fast surface blit and
  framebuffer conversion

## Quick Start

### Windows

1. Double-click `run.bat` — it will install dependencies and launch the
   emulator with the menu.

### Linux / macOS

1. From a terminal, run `./run.sh` (or `bash run.sh` if it isn't
   executable). The launcher will install missing dependencies and start
   the emulator.

### Manual

```bash
pip install -r requirements.txt
python gbc_emulator_skeleton.py
```

### Boot a ROM directly (skip the menu)

```bash
python gbc_emulator_skeleton.py path/to/rom.gb --nomenu
```

## Controls

| Key            | GB Button     |
| -------------- | ------------- |
| Arrow keys     | D-pad         |
| Z              | A             |
| X              | B             |
| Right Shift    | Select        |
| Enter          | Start         |
| Escape         | Menu / Quit   |
| F5 (in menu)   | Refresh ROMs  |

## Menu

- **Load ROM** — Browse and select a `.gb` or `.gbc` file. Press **F5**
  to refresh the list.
- **Settings** — Cycle the window scale between 2× and 5×.
- **Exit to OS** — Quit the emulator.

Place ROM files in a `roms/` folder next to the emulator script, or
anywhere in the current / parent directory — the scanner walks the
filesystem (up to two levels deep) looking for `.gb` / `.gbc` files.

## Architecture

The emulator lives in a single ~1700-line Python file. The hot path
is:

1. `GameBoy.step_all` — combined per-opcode dispatcher.
2. `CPU.step` → `execute` (hot path: NOP / LD r,r / ALU A,r) → `_exec_low`
   / `_exec_high` (less common opcodes).
3. `PPU.step` accumulates cycles and renders scanlines on the
   456-cycle boundary; `Timers.step` advances DIV / TIMA.

Performance-critical helpers:

- `PPU._TILE_COLORS[(hi<<8) | lo]` — 64K-entry table mapping a tile row's
  lo / hi bytes to the eight 2-bit color indices, replacing a per-pixel
  bit-shift chain.
- `PPU._PALETTE_SHADES[bgp]` — 256-entry table mapping BGP / OBP values
  to the four shade indices.
- Unrolled 8-pixel writers in `_render_scanline` and `_render_window`
  for the common fully-on-screen case, with a bounds-checked fallback
  for partial overlap on the left edge.

## Project Layout

```
gbc_emulator_skeleton.py   Single-file emulator (CPU, MMU, PPU, Timers, menu, runner)
requirements.txt           Pinned dependency list (pygame, numpy)
run.bat                    Windows launcher (installs deps if missing, then runs)
run.sh                     Linux / macOS launcher (bash, installs deps if missing)
gbclogo.png                Branding logo (used in menu + window icon)
roms/                      Drop ROMs here (auto-created on first run)
LICENSE                    MIT License
.gitattributes             Enforces LF line endings on shell / source files
```

## Status

A working DMG-compatible emulator that successfully displays the title
screen of several homebrew ROMs (including **SUPERBAJTEK** by Arte Frog
FF Studio) with proper scrolling background, window layer, sprite
rendering, and interrupt timing. Runs at ~60–90 fps in the Python
interpreter on modest hardware.

### Known Limitations

- No APU (audio output is silent).
- No CGB color palettes — DMG 4-shade palette only.
- No battery save file persistence (`*.sav` is not written).
- STAT blocking mode is not emulated per-cycle.
- Unimplemented opcodes halt the CPU silently (logged as an error).
- No save-state support.

## License

This project is licensed under the **MIT License** — see the
[`LICENSE`](./LICENSE) file for the full text.

```
MIT License

Copyright (c) 2026 awest813

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Acknowledgements

- **SUPERBAJTEK** test ROM by Arte Frog FF Studio.
- Game Boy hardware reference: [Pan Docs](https://gbdev.io/pandocs/),
  the [gbdev](https://gbdev.io/) community.
- Project logo by the author.
