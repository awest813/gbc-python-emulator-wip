# Python GBC Emulator

A Game Boy / Game Boy Color emulator written in Python, featuring a built-in
menu system, ROM browser, MBC1/MBC2/MBC3/MBC5 cartridge support, and full
CGB compatibility.

![status](https://img.shields.io/badge/status-playable-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)

## Features

- **CPU** — Complete LR35902 instruction set (standard + CB-prefixed),
  interrupt handling, HALT bug, HALT / wait states, EI delay, double-speed
  mode on CGB.
- **PPU** — Background, window, and 8×8 / 8×16 sprite rendering with
  DMG 4-shade palette and CGB 8-palette × 4-color BG/OBJ palettes,
  tile-level attributes from VRAM bank 1, X/Y flip, sprite-over-BG
  priority, OPRI register, OAM / VRAM access blocking during PPU
  modes 2/3 (STAT blocking), 8x16 sprite y-flip, and correct sprite
  X / OAM priority ordering.
- **APU** — All four DMG sound channels (two squares with envelope,
  sweep on channel 1, 32-step wave channel, LFSR noise channel),
  frame sequencer for length / envelope / sweep, master volume and
  per-channel stereo panning, streamed to the host at 44.1 kHz.
- **Serial port** — FF01/FF02 register support with serial interrupt
  generation (link cable TCP client/server planned).
- **Timers** — DIV, TIMA, TMA, TAC with all four programmable rates and
  correct overflow → interrupt signalling.
- **Cartridge** — MBC1, MBC2 (4-bit RAM), MBC3 (with RTC), and MBC5
  (ROM / RAM banking, battery-backed RAM with automatic `.sav` file
  load on boot and save on exit). CGB mode auto-detected from header
  byte 0x0143. Boot ROM support (DMG 256B / CGB ~2304B).
- **DMA** — OAM DMA (FF46), CGB H-Blank DMA (FF51-FF55), and CGB GDMA.
- **CGB extras** — VRAM bank 1 (FF4F), WRAM bank 1-7 (FF70), CGB BG/OBJ
  palettes (FF68-FF6C), KEY1 double-speed (FF4D), and all write-protection
  rules (STAT read-only bits 0-2 / 6, unused bits forced to 1).
- **Save states** — Snapshot full emulator state to `<rom>.ss<slot>`
  with F6 / F8 (save) and F7 / F9 (load).
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

### Chromebook (Crostini Linux)

1. Enable Linux in Chrome OS Settings > Developers > Linux development environment
2. Open the Terminal app and run:
   ```bash
   sudo apt update && sudo apt install python3 python3-pip libsdl2-2.0-0
   pip3 install pygame numpy
   ```
3. Launch the emulator:
   ```bash
   python3 gbc_emulator_skeleton.py
   ```
   The `run.sh` launcher automatically sets `SDL_AUDIODRIVER=alsa` on ChromeOS. If
   audio doesn't work, try `SDL_AUDIODRIVER=dummy python3 gbc_emulator_skeleton.py`
   to run silently.

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
| F6 / F8        | Save state (slot 0 / 1) |
| F7 / F9        | Load state (slot 0 / 1) |

## Menu

- **Load ROM** — Browse and select a `.gb` or `.gbc` file. Press **F5**
  to refresh the list.
- **Settings** — Tweak the following options (press **Enter** to cycle each):
  - *Window Scale* — 2× … 5×
  - *Frame Rate* — 59.7 fps / 60 fps / Unlimited
  - *Volume* — Mute / Low / Medium / High / Max
  - *Palette* — DMG Green / Grayscale / Amber / Blue / Brown / Pastel
  - *Filter* — Nearest (pixel-sharp) / Smooth (bilinear)
  - *Shader* — Off / LCD Ghost / CRT Scanlines / Gamma Warm / Pixel Bloom / Pocket Green
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
   456-cycle boundary; `Timers.step` advances DIV / TIMA;
   `APU.step` clocks the frame sequencer, channel waveforms, and
   emits 44.1 kHz PCM into a buffer that the run loop drains to
   SDL's audio queue after each frame.

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
*.sav                      Battery-backed cartridge save files (auto-created next to ROM)
```

## Status

A working DMG/CGB-compatible emulator that successfully displays the title
screen of several homebrew ROMs (including **SUPERBAJTEK** by Arte Frog
FF Studio) and commercial CGB games (such as **Dragon Warrior III**)
with proper scrolling background, window layer, sprite rendering, CGB
palettes, and interrupt timing. Runs at ~60–90 fps in the Python
interpreter on modest hardware.

### Known Limitations

- CGB double-speed mode is implemented at the cycle-accurate level
  (PPU / APU / timers all see 2× cycles), but the cartridge bus
  access-time penalty (one extra wait state per cartridge read at
  2× speed) is not emulated.
- Per-cycle STAT blocking for non-CPU bus activity (e.g. during DMA)
  is approximated; the OAM / VRAM access blocking during PPU modes
  2/3 is correctly enforced.
- Unimplemented opcodes are logged as errors and treated as a 4-cycle
  NOP (the original DMG behaviour is to fully fault on them).
- SGB (Super Game Boy) features are not emulated.

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
