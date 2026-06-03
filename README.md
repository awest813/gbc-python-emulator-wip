# Python GBC Emulator

A Game Boy / Game Boy Color emulator written in Python, featuring a built-in menu system, ROM browser, and MBC1/MBC5 cartridge support.

![menu](https://img.shields.io/badge/status-work_in_progress-yellow)

## Features

- **CPU**: Complete LR35902 instruction set (standard + CB-prefixed), interrupt handling
- **PPU**: BG, window, and sprite rendering with DMG-style 4-shade palette
- **Timers**: DIV, TIMA, TMA, TAC with programmable rates
- **Cartridge**: MBC1 and MBC5 (ROM/RAM banking, battery RAM)
- **Menu system**: ROM browser, window scale settings, keyboard controls
- **Input**: Keyboard mapping for D-pad, A/B, Start, Select

## Requirements

- Python 3.9+
- [pygame](https://pypi.org/project/pygame/)
- [numpy](https://pypi.org/project/numpy/)

## Quick Start

### Windows

1. Double-click `run.bat` — it will install dependencies and launch the emulator.

### Manual

```bash
pip install -r requirements.txt
python gbc_emulator_skeleton.py
```

### Skip the menu (boot directly)

```bash
python gbc_emulator_skeleton.py path/to/rom.gb --nomenu
```

## Controls

| Key | GB Button |
|-----|-----------|
| Arrow keys | D-pad |
| Z | A |
| X | B |
| Right Shift | Select |
| Enter | Start |
| Escape | Return to menu / Quit |

## Menu

- **Load ROM** — Browse and select a `.gb` or `.gbc` file. Press F5 to refresh.
- **Settings** — Toggle window scale (2x–5x).
- **Exit to OS** — Quit.

Place ROM files in a `roms/` folder next to the emulator, or anywhere in the current or parent directory.

## Status

This is a work-in-progress emulator. It can successfully display the title screen of several homebrew ROMs with proper scrolling background, window layer, and interrupt timing. Known limitations:

- No APU (audio) — only runs at ~33 fps
- No CGB color mode — DMG palette only
- No battery save file persistence
- STAT blocking mode not emulated per-cycle
