# Python GBC Emulator

A Game Boy / Game Boy Color emulator written in Python, featuring a built-in
menu system, ROM browser, MBC1/MBC2/MBC3/MBC5/MBC6/MBC7 cartridge support,
Super Game Boy palettes, and full CGB compatibility.

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
  per-channel stereo panning, streamed to the host at 44.1 kHz.  Full
  CGB audio behaviour (wave RAM access rules, frame-sequencer reset).
- **Serial port** — FF01/FF02 with cycle-accurate bit-clocking (512 T-cycles
  per bit, or 16 in CGB fast mode), serial interrupt after 8 bits, and a
  TCP link cable for local two-player multiplayer.
- **Timers** — DIV, TIMA, TMA, TAC with all four programmable rates and
  correct overflow → interrupt signalling.
- **Cartridge** — MBC1, MBC2 (4-bit RAM), MBC3 (with RTC), MBC5,
  MBC6 (dual 8 KB ROM/flash windows + SRAM) and MBC7 (EEPROM +
  accelerometer; D-pad tilts *Kirby Tilt 'n' Tumble*). Battery-backed
  RAM / EEPROM / flash with automatic `.sav` load on boot and save on
  exit. CGB mode auto-detected from header byte 0x0143; SGB mode from
  byte 0x0146 when the cart is not CGB.
- **SGB** — Packet transfer via P1, PAL01/23/03/12, PAL_SET / PAL_TRN,
  ATTR_BLK/LIN/DIV/CHR, ATTR_SET / ATTR_TRN, MASK_EN and MLT_REQ
  (SGB detection + 2/4-player IDs). The 20×18 attribute map recolors
  the DMG picture with SNES RGB555 palettes.
- **DMA** — OAM DMA (FF46), CGB H-Blank DMA (FF51-FF55), and CGB GDMA.
- **CGB extras** — VRAM bank 1 (FF4F), WRAM bank 1-7 (FF70), CGB BG/OBJ
  palettes (FF68-FF6C), KEY1 double-speed (FF4D) including one extra
  T-cycle wait-state per cartridge ROM/RAM access, and all
  write-protection rules (STAT read-only bits 0-2 / 6, unused bits
  forced to 1). Boot ROM support (DMG 256B / CGB ~2304B).
- **Save states** — Snapshot full emulator state (including cartridge SRAM)
  to `<rom>.ss<slot>` with F6 / F8 (save) and F7 / F9 (load). v8 saves
  record the ROM basename plus a CRC32 fingerprint and refuse to load into
  a different game or a same-name file swap.
- **Menu system** — ROM browser, window-scale selector, keyboard controls,
  project logo.
- **Input** — D-pad, A / B, Start, Select via keyboard (customisable
  key bindings) and gamepad (auto-detected Xbox/PlayStation layout).
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
python gbc_emulator.py
```

### Boot a ROM directly (skip the menu)

```bash
python gbc_emulator.py path/to/rom.gb
python gbc_emulator.py path/to/rom.gbc --bootrom path/to/boot.bin
```

`--nomenu` is still accepted and means the same thing. A missing ROM path
prints an error instead of dropping into the menu. `--bootrom` loads an
optional DMG (256 B) or CGB (~2304 B) boot ROM before the cartridge starts.

### Chromebook (Crostini Linux)

1. Enable Linux in Chrome OS Settings > Developers > Linux development environment
2. Open the Terminal app and run:
   ```bash
   sudo apt update && sudo apt install python3 python3-pip libsdl2-2.0-0
   pip3 install pygame numpy
   ```
3. Launch the emulator:
   ```bash
   python3 gbc_emulator.py
   ```
   The `run.sh` launcher automatically sets `SDL_AUDIODRIVER=alsa` on ChromeOS. If
   audio doesn't work, try `SDL_AUDIODRIVER=dummy python3 gbc_emulator.py`
   to run silently.

## Controls

### Keyboard
| Key            | GB Button     |
| -------------- | ------------- |
| Arrow keys     | D-pad         |
| W A S D        | D-pad (toggle in Controls) |
| Z              | A             |
| X              | B             |
| Right Shift    | Select        |
| Enter          | Start         |
| Escape         | Pause menu (in game) / Back |
| Tab            | Fast-forward (hold) |
| Q / `,`        | Turbo A (remappable) |
| E / `.`        | Turbo B (remappable) |
| F3             | Toggle FPS overlay |
| F4             | Toggle input overlay |
| F11            | Toggle fullscreen |
| Ctrl+R         | Soft reset |
| F5 (in menu)   | Refresh ROMs  |
| F6 / F8        | Save state (slot 0 / 1) |
| F7 / F9        | Load state (slot 0 / 1) |

When a link cable is connected, a **LINK** badge appears in the HUD
(top-right, with the FPS / input overlays).

Keys are customisable: **Settings → Controls...** (also available from the
in-game pause menu). Press Enter on a button to capture a new key. Esc, Tab,
F2–F9, and F11 are reserved. Bindings (including turbo A/B) are stored in
`gbc_config.json`. Click a row or use the analog stick (with a repeat delay)
to move; mouse wheel scrolls lists.

### Gamepad / Controller
Gamepads are auto-detected and use Xbox/PlayStation layout by default:

| Gamepad        | GB Button     |
| -------------- | ------------- |
| D-pad          | D-pad         |
| Left stick     | D-pad         |
| A / Cross / Y / RB | A         |
| B / Circle / X / LB | B        |
| Select / Share | Select        |
| Start / Options | Start        |
| R3 (stick click)| Fast-forward |
| Select + Start | Pause menu    |
| Escape         | Pause menu    |
| F11            | Fullscreen    |

D-pad, analog stick, and keyboard are tracked as separate sources so releasing
the stick cannot un-press a still-held D-pad or key. Opposite directions on the
same axis use last-wins cleaning (hardware cannot press Left+Right together).
A held analog stick in menus moves once, then repeats after a short delay
instead of scrolling every SDL axis event. Hot-plugging a pad shows a toast.

## Link Cable (Local Multiplayer)

Two emulator instances can connect via TCP for local link cable gameplay:

```bash
# Player 1 (server):
python gbc_emulator.py rom.gbc --link-server 12345

# Player 2 (client):
python gbc_emulator.py rom.gbc --link-connect 127.0.0.1:12345
```

## Settings Persistence

All menu settings (scale, fullscreen, volume, palette, shader, audio toggle, frame rate,
key bindings, last ROM) are saved to `gbc_config.json` and reloaded on next launch.  The
file is created automatically in the emulator directory.

## Menu

- **Continue** — Shown when a ROM was loaded previously; boots that file immediately.
- **Load ROM** — Browse and select a `.gb` or `.gbc` file. Each entry
  shows a **DMG** or **CGB** badge from the ROM header. Press **F5**
  to refresh the list, **PgUp/PgDn/Home/End** to jump, or click a row
  (mouse wheel scrolls).
- **Settings** — Tweak the following options (press **Enter** or **Left/Right** to cycle):
  - *Window Scale* — 2× … 5× (windowed mode)
  - *Display* — Window / Fullscreen (F11 also toggles; integer-scaled and letterboxed)
  - *Frame Rate* — 59.7 fps / 60 fps / Unlimited
  - *Audio* — On / Off
  - *Volume* — Mute / Low / Medium / High / Max
  - *Palette* — DMG Green / Grayscale / Amber / Blue / Brown / Pastel
  - *Filter* — Nearest (pixel-sharp) / Smooth (bilinear)
  - *Shader* — Off / LCD Ghost / CRT Scanlines / Gamma Warm / Pixel Bloom / Pocket Green
  - *Controls...* — Remap keyboard keys (including turbo A/B), toggle WASD D-pad, reset to defaults
- **Exit to OS** — Quit the emulator (with a confirmation prompt).

## Pause menu

Press **Escape** while a game is running to open the in-game pause menu.
Emulation and audio halt, and the current frame is dimmed behind the menu.
**F6–F9** quick-save/load and **Ctrl+R** soft reset still work while the
pause overlay is open:

- **Resume** — Return to the game (Escape also resumes).
- **Save States...** — Sub-menu with save/load for slots 0 and 1
  (same as F6–F9). Occupied slots show a timestamp; empty slots say
  **empty**. Status toasts appear in a dedicated strip above
  the navigation hint.
- **Settings** — The same options as the main settings page, applied
  **live** to the running game (palette, shader, filter, volume, audio,
  frame rate, window scale, and Controls remapping all update immediately).
- **Exit to Menu** — Return to the main menu, with a confirmation prompt.
  Your battery save (`.sav`) is written out on the way back.

Place ROM files in a `roms/` folder next to the emulator script, or
anywhere in the current / parent directory — the scanner walks the
filesystem (up to two levels deep) looking for `.gb` / `.gbc` files.

## Architecture

The emulator lives in a single Python file. The hot path
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
- Packed 24-bit framebuffer: each pixel is a single `(r<<16)|(g<<8)|b`
  integer rather than an `(r, g, b)` tuple. Scanline writers stay at one
  assignment per pixel, but the per-frame host blit unpacks the whole
  frame with a vectorised numpy shift instead of iterating 23 040 tuples
  (~2.5× faster render-to-surface path).
- Direct WRAM (`C000–DFFF`) and HRAM/IE (`FF80–FFFF`) memory accessors skip
  the I/O decode chain on the hottest CPU read/write path.
- Silent APU frames emit a single bulk zero-fill instead of mixing 700+
  empty samples per video frame.
- Halted CPUs skip ahead to the next PPU mode or TIMA event instead of
  burning 4 T-cycles per `step_all` call (~17k times per frame).
- APU frame-sequencer countdown and reused OAM scans (~10% higher synthetic
  throughput vs. the pre-audit baseline on `smoke_test.py`).
- On boot, invalid Nintendo logos / header checksums and unsupported mappers
  show a warning toast without blocking the load.

## Project Layout

```
gbc_emulator.py            Single-file emulator (CPU, MMU, PPU, Timers, menu, runner)
gbc_emulator_skeleton.py   Deprecated CLI alias that forwards to gbc_emulator.py
test_headless.py           Self-contained smoke test (synthetic ROM, no display)
test_save_state.py         Save-state round-trip test (synthetic CGB ROM)
test_controls.py           Joypad sources, SOCD, key bindings, WRAM fast-path
test_hw_features.py        SGB packets, MBC6/7, serial bit-clock, CGB wait-states
ci_test.py                 Runs the portable tests (used by GitHub Actions)
requirements.txt           Dependency list (pygame, numpy)
run.bat                    Windows launcher (installs deps if missing, then runs)
run.sh                     Linux / macOS launcher (bash, installs deps if missing)
gbclogo.png                Branding logo (used in menu + window icon)
roms/                      Drop ROMs here (auto-created on first run)
LICENSE                    MIT License
.gitattributes             Enforces LF line endings on shell / source files
*.sav                      Battery-backed cartridge save files (auto-created next to ROM)
```

## Testing

A portable smoke test builds a synthetic ROM in memory and exercises the CPU,
PPU, timers and APU without a display or any local ROM files:

```bash
python test_headless.py
```

It verifies representative opcode flag behaviour (ADD / SUB / DAA / CB
SWAP / SRL / BIT), illegal-opcode handling, a full multi-step run, and
CGB double-speed dot scaling. Exits non-zero on failure.

A second portable test covers the save-state system end to end — it
snapshots a running synthetic CGB machine, mutates live state, reloads,
and asserts the CPU / PPU / timer state and the derived CGB colour tables
are restored exactly:

```bash
python test_save_state.py
```

A third portable test covers input: multi-source joypad combining (keyboard
does not get un-pressed when an analog stick recenters), last-wins opposite
D-pad cleaning, customisable key bindings, and WRAM/HRAM write fast-paths:

```bash
python test_controls.py
```

All four portable tests are self-contained (no display, no local ROMs) and exit
non-zero on failure, so they work as CI checks. The fourth test covers Super
Game Boy packets, MBC6/MBC7 mappers, cycle-accurate serial bit-clocking, and
CGB double-speed cartridge wait-states:

```bash
python test_hw_features.py
```

A single wrapper script runs all tests sequentially:

```bash
python ci_test.py
```

GitHub Actions (`.github/workflows/ci.yml`) runs that wrapper on Python
3.9, 3.11, and 3.12.

## Status

A working DMG/CGB-compatible emulator that successfully displays the title
screen of several homebrew ROMs (including **SUPERBAJTEK** by Arte Frog
FF Studio) and commercial CGB games (such as **Dragon Warrior III**)
with proper scrolling background, window layer, sprite rendering, CGB
palettes, and interrupt timing. Runs at ~60–90 fps in the Python
interpreter on modest hardware.

### Known Limitations

- SGB border graphics (`CHR_TRN` / `PCT_TRN`) are accepted but not drawn;
  the emulated picture stays at 160×144.
- MBC7 tilt is mapped from the D-pad (and analog stick) rather than a
  real accelerometer. EEPROM programming is immediate (no busy delay).
- Link-cable multiplayer is CLI-only (`--link-server PORT` /
  `--link-connect HOST:PORT` with a ROM path). Bytes are exchanged at
  transfer start, then shifted locally one bit at a time. A stalled
  partner times out instead of freezing the emulator. The HUD shows
  **LINK** while the socket is connected.
- Save states (slots 0 and 1) include cartridge SRAM, MBC6 flash, SGB
  palettes, and in-flight serial state. v8 saves store the ROM basename,
  byte length, and CRC32 fingerprint (with length guard), APU
  frame-sequencer timing, GDMA stall, double-speed remainder, boot-ROM
  map flag, per-source joypad state, MBC7 EEPROM shift progress, and
  in-progress Super Game Boy packet assembly; v4–v7 saves still load
  without a CRC check; v1–v3 load without a ROM identity check. Loading
  v6 or older clears any partial SGB packet state. Failed
  loads roll back live CPU/MMU/PPU/APU state. In-flight serial shifts are
  cleared on load unless a link partner is connected.
  Battery-backed `.sav` files are
  written on exit. MBC3 RTC is stored after SRAM in VBA-M's 44-byte
  format so day/night continues while the emulator is closed. Older
  48-byte custom trailers still load.

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
