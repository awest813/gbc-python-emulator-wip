# Python GBC Emulator

A Game Boy / Game Boy Color emulator written in Python, featuring a built-in
menu system, ROM browser, and MBC1/MBC5 cartridge support.

![status](https://img.shields.io/badge/status-working-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

## Features

- **CPU** — Complete LR35902 instruction set (standard + CB-prefixed),
  interrupt handling, halt/wait states.
- **PPU** — Background, window, and 8×8 / 8×16 sprite rendering with
  DMG-style 4-shade palette, sprite priority, and BG-attribute overlay.
- **Timers** — DIV, TIMA, TMA, TAC with all four programmable rates and
  correct overflow → interrupt signalling.
- **Cartridge** — MBC1 and MBC5 (ROM/RAM banking, battery RAM, rumble).
- **Menu system** — ROM browser, window-scale selector, keyboard controls,
  project logo.
- **Input** — D-pad, A/B, Start, Select with the standard joypad register
  layout.
- **Performance** — ~60–90 fps on SUPERBAJTEK (Python 3.11); precomputed
  tile / palette lookup tables, unrolled scanline writers, and a
  combined per-opcode dispatcher keep the inner loop tight.

## Requirements

- Python 3.9 or newer
- [pygame](https://pypi.org/project/pygame/) — display, input, icon
- [numpy](https://pypi.org/project/numpy/) — fast surface blit and
  framebuffer conversion (optional but recommended)

## Quick Start

### Windows

1. Double-click `run.bat` — it will install dependencies and launch the
   emulator with the menu.

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

| Key            | GB Button |
| -------------- | --------- |
| Arrow keys     | D-pad     |
| Z              | A         |
| X              | B         |
| Right Shift    | Select    |
| Enter          | Start     |
| Escape         | Menu / Quit |

## Menu

- **Load ROM** — Browse and select a `.gb` or `.gbc` file. Press **F5**
  to refresh the list.
- **Settings** — Cycle the window scale between 2× and 5×.
- **Exit to OS** — Quit the emulator.

Place ROM files in a `roms/` folder next to the emulator script, or
anywhere in the current / parent directory — the scanner walks the
filesystem looking for `.gb` / `.gbc` files.

## Project Layout

```
gbc_emulator_skeleton.py   Single-file emulator (CPU, MMU, PPU, Timers, menu, runner)
requirements.txt           Pinned dependency list (pygame, numpy)
run.bat                    Windows launcher (installs deps if missing, then runs)
gbclogo.png                Branding logo (used in menu + window icon)
roms/                      Drop ROMs here (auto-created on first run)
```

## Status

A working DMG-compatible emulator that successfully displays the title
screen of several homebrew ROMs (including **SUPERBAJTEK** by Arte Frog
FF Studio) with proper scrolling background, window layer, sprite
rendering, and interrupt timing. Runs at ~60 fps in the Python
interpreter on modest hardware.

### Known Limitations

- No APU (audio output is silent).
- No CGB color palettes — DMG 4-shade palette only.
- No battery save file persistence.
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

- SUPERBAJTEK test ROM by Arte Frog FF Studio.
- Game Boy hardware reference: [Pan Docs](https://gbdev.io/pandocs/),
  [gbdev](https://gbdev.io/) community.
