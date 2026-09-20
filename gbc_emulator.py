"""
Game Boy / Game Boy Color emulator written in Python.

Supports MBC1/MBC2/MBC3/MBC5/MBC6/MBC7 cartridges, Super Game Boy palettes,
cycle-accurate serial bit-clocking, CGB double-speed cart wait-states,
BG / Window / Sprite rendering, and a built-in menu with ROM browser.
Performance-tuned via precomputed tile / palette LUTs, unrolled scanline
writers, a combined per-opcode dispatcher, and fast-path WRAM/HRAM memory
access. Keyboard bindings are customisable.

Usage:
    python gbc_emulator.py                  # launch the menu
    python gbc_emulator.py rom.gb           # boot a ROM directly
    python gbc_emulator.py rom.gb --nomenu  # same, skip the menu explicitly

Requires: pygame >= 2.0, numpy >= 1.20
          (pip install pygame numpy)

SPDX-License-Identifier: MIT
Copyright (c) 2026 awest813
"""
__version__ = "1.0.0"
import sys
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import time
import logging
import argparse
import struct
import socket
import zlib
import json
from collections import deque

# On Windows, request 1 ms timer resolution so time.sleep() is more precise.
if sys.platform == 'win32':
    try:
        import ctypes
        _winmm = ctypes.windll.winmm
        _winmm.timeBeginPeriod(1)
    except Exception:
        pass

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed. Run 'pip install numpy' for faster rendering.")
    np = None

try:
    import pygame
except ImportError:
    print("Warning: pygame not installed. Run 'pip install pygame' for display support.")
    pygame = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _noop_trace(*_args, **_kwargs):
    """No-op branch trace hook used when tracing is disabled (avoids call overhead)."""
    return

# --- CONSTANTS ---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
FPS = 4194304 / 70224  # ≈ 59.7275 Hz
CYCLES_PER_FRAME = 70224  # 4.194304 MHz / 59.73 frames per second

# Interrupt flag bits (used in IF and IE registers at 0xFF0F / 0xFFFF)
IF_VBLANK  = 0x01
IF_LCD     = 0x02
IF_TIMER   = 0x04
IF_SERIAL  = 0x08
IF_JOYPAD  = 0x10

# PPU timing (in dot-cycles)
DOTS_PER_SCANLINE     = 456
TOTAL_SCANLINES       = 154
MODE3_START_DOT       = 80
MODE3_BASE_DURATION   = 172
MAX_SPRITES_PER_LINE  = 10
TOTAL_OAM_SPRITES     = 40
OAM_SIZE              = 160  # bytes
OAM_DMA_CYCLES        = 640  # CPU T-cycles (160 bytes × 4)

# Serial bit-clock: 8192 Hz (512 T-cycles/bit) or CGB fast 262144 Hz (16 T-cycles/bit).
SERIAL_BIT_CYCLES_NORMAL = 512
SERIAL_BIT_CYCLES_FAST   = 16

# MBC7 accelerometer: 16-bit samples centred at 0x81D0; 1g ≈ 0x70.
MBC7_ACCEL_CENTER = 0x81D0
MBC7_ACCEL_G      = 0x70
MBC6_FLASH_SIZE   = 0x100000  # 1 MiB MX29F008
MBC7_EEPROM_SIZE  = 256       # 93LC56, 128 × 16-bit words

# APU constants
CH_PULSE_MAX_LENGTH = 64
CH3_MAX_LENGTH      = 256
MAX_FREQUENCY       = 2048
DEFAULT_ENV_PERIOD  = 8
WAVE_RAM_SIZE       = 16

# Audio
INT16_MAX = 32767
INT16_MIN = -32768
AUDIO_SCALE_FACTOR = 70

def _sign_extend_byte(b):
    """Sign-extend an 8-bit value to a signed Python int (range -128..127)."""
    return b - 256 if b & 0x80 else b

# --- PALETTES (DMG 4-shade colour sets) ---
PALETTE_DMG = ((224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32))
PALETTE_GRAY = ((255, 255, 255), (170, 170, 170), (85, 85, 85), (0, 0, 0))
PALETTE_GREEN = PALETTE_DMG
PALETTE_AMBER = ((255, 245, 200), (200, 175, 100), (140, 110, 50), (60, 40, 10))
PALETTE_BLUE = ((200, 230, 255), (100, 160, 220), (40, 80, 160), (10, 20, 60))
PALETTE_BROWN = ((240, 220, 180), (180, 140, 100), (110, 70, 40), (40, 20, 10))
PALETTE_PASTEL = ((230, 220, 255), (180, 160, 230), (120, 110, 190), (50, 40, 100))
PALETTE_LIST = [
    ("DMG Green",  PALETTE_DMG),
    ("Grayscale",  PALETTE_GRAY),
    ("Amber",      PALETTE_AMBER),
    ("Blue",       PALETTE_BLUE),
    ("Brown",      PALETTE_BROWN),
    ("Pastel",     PALETTE_PASTEL),
]

# Post-process shaders: each takes (np.uint8 array 160x144x3) and returns same shape
def _shader_none(fb):
    return fb

def _shader_lcd_ghost(fb, prev=None, out=None, scratch=None, scratch_b=None):
    if prev is None:
        return fb
    if scratch is not None and scratch_b is not None and out is not None:
        np.multiply(fb, 0.80, out=scratch)
        np.multiply(prev, 0.20, out=scratch_b)
        np.add(scratch, scratch_b, out=scratch)
        np.clip(scratch, 0, 255, out=out)
        return out
    return np.clip(fb * 0.80 + prev * 0.20, 0, 255).astype(np.uint8)

def _shader_crt_scanlines(fb, out=None):
    if out is None:
        out = fb.copy()
    else:
        np.copyto(out, fb)
    out[1::2] = (out[1::2].astype(np.uint16) * 65 // 100).astype(np.uint8)
    return out

def _shader_gamma_warm(fb, out=None, scratch=None):
    if scratch is None or out is None:
        return np.clip(np.power(fb.astype(np.float32) / 255.0, 1.2) * 255.0, 0, 255).astype(np.uint8)
    np.divide(fb, 255.0, out=scratch)
    np.power(scratch, 1.2, out=scratch)
    np.multiply(scratch, 255.0, out=scratch)
    np.clip(scratch, 0, 255, out=out)
    return out

def _shader_pixel_bloom(fb, out=None, acc=None, blend=None):
    if acc is None or blend is None or out is None:
        blurred = (np.roll(fb, 1, 0) + np.roll(fb, -1, 0) +
                   np.roll(fb, 1, 1) + np.roll(fb, -1, 1)) // 4
        return np.clip(fb * 0.70 + blurred * 0.30, 0, 255).astype(np.uint8)
    acc.fill(0)
    acc[1:-1] += fb[:-2]
    acc[1:-1] += fb[2:]
    acc[:, 1:-1] += fb[:, :-2]
    acc[:, 1:-1] += fb[:, 2:]
    acc[0] += fb[-1]
    acc[-1] += fb[0]
    acc[:, 0] += fb[:, -1]
    acc[:, -1] += fb[:, 0]
    acc *= 0.25
    np.multiply(fb, 0.70, out=blend)
    np.multiply(acc, 0.30, out=acc)
    np.add(blend, acc, out=blend)
    np.clip(blend, 0, 255, out=out)
    return out

def _shader_pocket_green(fb):
    lum = fb.astype(np.float32).dot([0.299, 0.587, 0.114])
    r = np.clip(lum * 0.65, 0, 255).astype(np.uint8)
    g = np.clip(lum * 0.85, 0, 255).astype(np.uint8)
    b = np.clip(lum * 0.45, 0, 255).astype(np.uint8)
    return np.dstack((r, g, b))

SHADER_LIST = [
    ("Off",           _shader_none),
    ("LCD Ghost",     _shader_lcd_ghost),
    ("CRT Scanlines", _shader_crt_scanlines),
    ("Gamma Warm",    _shader_gamma_warm),
    ("Pixel Bloom",   _shader_pixel_bloom),
    ("Pocket Green",  _shader_pocket_green),
]

FPS_LIMIT_OPTIONS = [("59.7 fps", 59.73), ("60 fps", 60.0), ("Unlimited", 0)]
AUDIO_OPTIONS = [("On", True), ("Off", False)]
VOLUME_OPTIONS = [("Mute", 0.0), ("Low", 0.25), ("Medium", 0.5), ("High", 0.75), ("Max", 1.0)]
FILTER_OPTIONS = [("Nearest", False), ("Smooth", True)]
WINDOW_SCALE_OPTIONS = [("2x", 2), ("3x", 3), ("4x", 4), ("5x", 5)]


_NINTENDO_LOGO = bytes([
    0x48, 0x06, 0x0E, 0x76, 0xFE, 0xB3, 0x1A, 0x0F, 0xCE, 0x6B, 0xB3, 0x83,
    0x2D, 0xC1, 0xE5, 0xD6, 0xC9, 0x19, 0x7D, 0x07, 0x4F, 0x1B, 0x7E, 0x33,
    0x9D, 0xBE, 0x9C, 0xD3, 0x09, 0x6C, 0xD2, 0xA1, 0x4A, 0x9F, 0x53, 0x1A,
    0x5C, 0x1B, 0x78, 0x20, 0x86, 0xE0, 0x49, 0x38, 0x84, 0xB3, 0x1C,
])

def _validate_rom_header(rom_data):
    """Return a list of header problems ('logo', 'checksum') or [] if OK."""
    if len(rom_data) < 0x150:
        return ['size']
    issues = []
    # Compare the 47-byte logo bitmap; byte 0x133 is the last title byte on HW.
    if rom_data[0x104:0x104 + len(_NINTENDO_LOGO)] != _NINTENDO_LOGO:
        issues.append('logo')
    chk = 0
    for b in rom_data[0x134:0x14D]:
        chk = (chk - b - 1) & 0xFF
    if chk != rom_data[0x14D]:
        issues.append('checksum')
    return issues

def _parse_rom_header(rom_path):
    """Parse ROM header bytes and return a short info string, or None on error."""
    try:
        with open(rom_path, 'rb') as f:
            f.seek(0x0143)
            cgb = f.read(1)[0]
            f.seek(0x0147)
            cart_type = f.read(1)[0]
            f.seek(0x0148)
            rom_size = f.read(1)[0]
            f.seek(0x0149)
            ram_size = f.read(1)[0]
    except OSError:
        return None
    _MBC_NAMES = {
        0x00: "ROM ONLY", 0x01: "MBC1", 0x02: "MBC1+RAM", 0x03: "MBC1+RAM+BATT",
        0x05: "MBC2", 0x06: "MBC2+BATT",
        0x0F: "MBC3+TIMER+BATT", 0x10: "MBC3+TIMER+RAM+BATT", 0x11: "MBC3",
        0x12: "MBC3+RAM", 0x13: "MBC3+RAM+BATT",
        0x19: "MBC5", 0x1A: "MBC5+RAM", 0x1B: "MBC5+RAM+BATT",
        0x1C: "MBC5+RUMBLE", 0x1D: "MBC5+RUMBLE+RAM", 0x1E: "MBC5+RUMBLE+RAM+BATT",
        0x20: "MBC6", 0x22: "MBC7",
    }
    mbc = _MBC_NAMES.get(cart_type, f"Unknown ({cart_type:02X})")
    if cart_type not in _SUPPORTED_CART_TYPES:
        mbc = f"{mbc} (unsupported)"
    rom_kb = (32 << rom_size) if rom_size <= 8 else 0
    ram_info = ""
    if ram_size == 0x00: ram_info = "No RAM"
    elif ram_size == 0x01: ram_info = "2 KB RAM"
    elif ram_size == 0x02: ram_info = "8 KB RAM"
    elif ram_size == 0x03: ram_info = "32 KB RAM"
    elif ram_size == 0x04: ram_info = "128 KB RAM"
    elif ram_size == 0x05: ram_info = "64 KB RAM"
    else: ram_info = f"RAM: {ram_size:02X}"
    cgb_str = "CGB" if cgb & 0x80 else "DMG"
    return f"{cgb_str}  |  {mbc}  |  {rom_kb} KB  |  {ram_info}"

def _opt_index(options, value, default=0):
    """Return the index in a (label, value) option list whose value matches, else default."""
    for i, opt in enumerate(options):
        if opt[1] == value:
            return i
    return default


def _clamp_index(idx, n, default=0):
    """Coerce a stored config index into ``range(n)``, else ``default``."""
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return default
    if n <= 0 or idx < 0 or idx >= n:
        return default
    return idx


def _clamp_choice(value, allowed, default):
    """Return ``value`` if it is in ``allowed``, else ``default``."""
    if value in allowed:
        return value
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value in allowed else default


# Cartridge types with working mapper emulation. Others are recognised in the
# ROM browser but will not run correctly (camera, HuC, TAMA5, …).
_SUPPORTED_CART_TYPES = frozenset({
    0x00, 0x01, 0x02, 0x03,
    0x05, 0x06,
    0x0F, 0x10, 0x11, 0x12, 0x13,
    0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
    0x20, 0x22,
})
_WINDOW_SCALES = (2, 3, 4, 5)

# Flag bit positions in the F register
FLAG_Z = 7  # Zero flag
FLAG_N = 6  # Subtract flag
FLAG_H = 5  # Half Carry flag
FLAG_C = 4  # Carry flag

# Joypad bit order matches P1 (FF00): Right, Left, Up, Down, A, B, Select, Start
JOYPAD_BUTTON_KEYS = ('right', 'left', 'up', 'down', 'a', 'b', 'select', 'start')
CONTROLS_WASD_ROW = len(JOYPAD_BUTTON_KEYS)
CONTROLS_TURBO_A_ROW = CONTROLS_WASD_ROW + 1
CONTROLS_TURBO_B_ROW = CONTROLS_WASD_ROW + 2
CONTROLS_RESET_ROW = CONTROLS_WASD_ROW + 3
CONTROLS_GAMEPAD_ROW = CONTROLS_WASD_ROW + 4
CONTROLS_SKIP_ROWS = frozenset({CONTROLS_GAMEPAD_ROW})
# Back-compat alias used by older tests / comments: first turbo row.
CONTROLS_TURBO_ROW = CONTROLS_TURBO_A_ROW
SETTINGS_ROW_IDS = (
    'scale', 'display', 'fps', 'audio', 'volume', 'palette', 'filter', 'shader', 'controls',
)
DISPLAY_OPTIONS = [("Window", False), ("Fullscreen", True)]


def _advance_controls_cursor(cursor, delta, n_items):
    """Move the controls-menu cursor, skipping informational rows."""
    n = max(int(n_items), 1)
    nxt = int(cursor)
    for _ in range(n):
        nxt = (nxt + delta) % n
        if nxt not in CONTROLS_SKIP_ROWS:
            return nxt
    return int(cursor) % n
JOYPAD_BUTTON_LABELS = ('Right', 'Left', 'Up', 'Down', 'A', 'B', 'Select', 'Start')
DEFAULT_KEY_BINDINGS = {
    'right':  ['right', 'd'],
    'left':   ['left', 'a'],
    'up':     ['up', 'w'],
    'down':   ['down', 's'],
    'a':      ['z'],
    'b':      ['x'],
    'select': ['right shift'],
    'start':  ['return'],
}
DEFAULT_TURBO_BINDINGS = {
    'turbo_a': ['q', ','],
    'turbo_b': ['e', '.'],
}
TURBO_BUTTON_KEYS = ('turbo_a', 'turbo_b')
TURBO_BUTTON_LABELS = ('Turbo A', 'Turbo B')
WASD_ALIASES = (('d', 'right'), ('a', 'left'), ('w', 'up'), ('s', 'down'))

# pygame key -> joypad bit; rebuilt from config / the Controls page.
KEY_TO_JOYPAD_BIT = {}

# Gamepad mapping: joystick button index -> joypad bit (Xbox / PlayStation layout)
_GAMEPAD_BUTTON_MAP = {
    0: 4,   # A / Cross  -> A
    1: 5,   # B / Circle -> B
    2: 5,   # X / Square -> B
    3: 4,   # Y / Triangle -> A
    4: 5,   # LB / L1    -> B
    5: 4,   # RB / R1    -> A
    6: 6,   # Select/Back/Share -> Select
    7: 7,   # Start/Options     -> Start
}
_GAMEPAD_AXIS_THRESHOLD = 0.5
_GAMEPAD_DPAD_BITS = {0: 0, 1: 1, 2: 2, 3: 3}  # hat direction index -> joypad bit
_JOY_SRC_KB = 0
_JOY_SRC_HAT = 1
_JOY_SRC_AXIS = 2
_JOY_SRC_BTN = 3
_JOY_SRC_TURBO = 4
_RESERVED_REMAP_KEY_NAMES = frozenset({
    'escape', 'tab', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f11',
})
_RESERVED_KEY_MSG = "That key is reserved (Esc, Tab, F2-F9, F11)"
KEY_TO_TURBO_BIT = {}

def _init_joysticks():
    """Initialise all connected joysticks. Safe to call multiple times."""
    if not pygame or not pygame.get_init():
        return
    try:
        pygame.joystick.init()
    except pygame.error:
        return
    for i in range(pygame.joystick.get_count()):
        try:
            pygame.joystick.Joystick(i).init()
        except pygame.error:
            pass

def _joystick_dpad_from_hat(event):
    """Convert a JOYHATMOTION event to a list of (joypad_bit, pressed) tuples."""
    x, y = event.value
    results = []
    dj = {0: (x == 1), 1: (x == -1), 2: (y == 1), 3: (y == -1)}
    for di, pressed in dj.items():
        results.append((_GAMEPAD_DPAD_BITS[di], pressed))
    return results

def _joystick_dpad_from_axis(event):
    """Convert a JOYAXISMOTION event to (joypad_bit, pressed) tuples.

    Axes 0/1 are the left stick; 6/7 are the D-pad on some Xbox-style pads.
    """
    T = _GAMEPAD_AXIS_THRESHOLD
    if event.axis in (0, 6):
        if event.value > T:      return [(0, True), (1, False)]
        elif event.value < -T:   return [(1, True), (0, False)]
        else:                    return [(0, False), (1, False)]
    if event.axis in (1, 7):
        if event.value > T:      return [(3, True), (2, False)]
        elif event.value < -T:   return [(2, True), (3, False)]
        else:                    return [(2, False), (3, False)]
    return []

def _gamepad_menu_action(event):
    """Map a joystick event to a menu keystroke string.

    Analog stick motion is handled separately by `_StickNav` so a held stick
    cannot flood the menu with one event per SDL axis sample.
    """
    if event.type == pygame.JOYBUTTONDOWN:
        if event.button == 0:  return 'select'
        if event.button == 1:  return 'back'
        if event.button == 7:  return 'select'  # Start == select
        if event.button == 6:  return 'back'    # Select == back
    if event.type == pygame.JOYHATMOTION:
        x, y = event.value
        if y == 1:  return 'up'
        if y == -1: return 'down'
        if x == -1: return 'left'
        if x == 1:  return 'right'
    return None


class _StickNav:
    """Edge-detect + repeat analog-stick menu navigation (one action per tick)."""
    __slots__ = ('x', 'y', 'x_due', 'y_due', 'delay', 'repeat')

    def __init__(self, delay=280, repeat=70):
        self.x = 0
        self.y = 0
        self.x_due = 0
        self.y_due = 0
        self.delay = delay
        self.repeat = repeat

    def reset(self):
        self.x = self.y = 0
        self.x_due = self.y_due = 0

    def update(self, x_dir, y_dir, now):
        """Return one of 'up'/'down'/'left'/'right', or None.

        ``x_dir`` / ``y_dir`` are -1, 0, or 1. Newly pressed directions fire
        immediately; held directions repeat after ``delay`` then ``repeat``.
        """
        edges = []
        repeats = []
        if y_dir != self.y:
            self.y = y_dir
            self.y_due = now + self.delay
            if y_dir:
                edges.append('up' if y_dir < 0 else 'down')
        elif y_dir and now >= self.y_due:
            self.y_due = now + self.repeat
            repeats.append('up' if y_dir < 0 else 'down')
        if x_dir != self.x:
            self.x = x_dir
            self.x_due = now + self.delay
            if x_dir:
                edges.append('left' if x_dir < 0 else 'right')
        elif x_dir and now >= self.x_due:
            self.x_due = now + self.repeat
            repeats.append('left' if x_dir < 0 else 'right')
        if edges:
            return edges[0]
        if repeats:
            return repeats[0]
        return None


def _read_analog_menu_dirs():
    """Return (x_dir, y_dir) from left stick / analog D-pad axes, or (0, 0)."""
    x_dir = y_dir = 0
    if not pygame:
        return 0, 0
    try:
        count = pygame.joystick.get_count()
    except pygame.error:
        return 0, 0
    T = _GAMEPAD_AXIS_THRESHOLD
    for i in range(count):
        try:
            js = pygame.joystick.Joystick(i)
            naxes = js.get_numaxes()
            for ax in (0, 6):
                if ax < naxes:
                    v = js.get_axis(ax)
                    if v > T:
                        x_dir = 1
                    elif v < -T:
                        x_dir = -1
            for ax in (1, 7):
                if ax < naxes:
                    v = js.get_axis(ax)
                    if v > T:
                        y_dir = 1
                    elif v < -T:
                        y_dir = -1
        except pygame.error:
            continue
    return x_dir, y_dir


def _open_display(size, fullscreen=False):
    """Open a windowed or desktop-fullscreen display surface."""
    flags = pygame.FULLSCREEN if fullscreen else 0
    if fullscreen:
        try:
            return pygame.display.set_mode((0, 0), flags)
        except pygame.error:
            return pygame.display.set_mode(size)
    return pygame.display.set_mode(size)


def _present_integer_scale(display, canvas, fill=(0, 0, 0)):
    """Blit ``canvas`` onto ``display`` with integer scale, letterboxed.

    Returns ``(scale, ox, oy, dest_w, dest_h)``.
    """
    dw, dh = display.get_size()
    cw, ch = canvas.get_size()
    if cw <= 0 or ch <= 0 or dw <= 0 or dh <= 0:
        return 1, 0, 0, max(cw, 1), max(ch, 1)
    scale = max(1, min(dw // cw, dh // ch))
    w, h = cw * scale, ch * scale
    ox, oy = (dw - w) // 2, (dh - h) // 2
    if (dw, dh) != (w, h) or (ox, oy) != (0, 0):
        display.fill(fill)
    if scale == 1:
        display.blit(canvas, (ox, oy))
    else:
        display.blit(pygame.transform.scale(canvas, (w, h)), (ox, oy))
    return scale, ox, oy, w, h


def _map_mouse_to_canvas(pos, scale, ox, oy, cw, ch):
    """Map a display-space mouse position onto a letterboxed canvas, or None."""
    if not pos or scale < 1:
        return None
    x = (pos[0] - ox) // scale
    y = (pos[1] - oy) // scale
    if x < 0 or y < 0 or x >= cw or y >= ch:
        return None
    return int(x), int(y)


def _hit_list_index(hits, pos):
    """Return the item index for a point in ``hits`` ``(x, y, w, h, index)`` rows."""
    if pos is None:
        return None
    x, y = pos
    for hx, hy, hw, hh, idx in hits:
        if hx <= x < hx + hw and hy <= y < hy + hh:
            return idx
    return None


def _move_list_cursor(cursor, n, action, page_size=10):
    """Move a non-wrapping list cursor (ROM browser, etc.)."""
    if n <= 0:
        return 0
    page = max(1, int(page_size))
    if action == 'home':
        return 0
    if action == 'end':
        return n - 1
    if action == 'pageup':
        return max(0, int(cursor) - page)
    if action == 'pagedown':
        return min(n - 1, int(cursor) + page)
    if action == 'up':
        return max(0, int(cursor) - 1)
    if action == 'down':
        return min(n - 1, int(cursor) + 1)
    return max(0, min(int(cursor), n - 1))


def _slot_status_suffix(path):
    """Return ``empty`` or a local ``YYYY-MM-DD HH:MM`` timestamp for a save file."""
    if not path or not os.path.isfile(path):
        return "empty"
    try:
        ts = os.path.getmtime(path)
    except OSError:
        return "empty"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _decorate_submenu_row(text, selected):
    return f"> {text}" if selected else text


def _connected_gamepad_label():
    if not pygame:
        return "Gamepad: none"
    try:
        n = pygame.joystick.get_count()
    except pygame.error:
        return "Gamepad: none"
    names = []
    for i in range(n):
        try:
            names.append(pygame.joystick.Joystick(i).get_name())
        except pygame.error:
            continue
    if not names:
        return "Gamepad: none"
    label = names[0] if len(names) == 1 else f"{names[0]} +{len(names) - 1}"
    return f"Gamepad: {label}"


def _gamepad_hotplug_message(event):
    """Human-readable toast for JOYDEVICEADDED / JOYDEVICEREMOVED, or None."""
    if not pygame:
        return None
    added = getattr(pygame, 'JOYDEVICEADDED', None)
    removed = getattr(pygame, 'JOYDEVICEREMOVED', None)
    if added is not None and event.type == added:
        name = None
        idx = getattr(event, 'device_index', None)
        if idx is not None:
            try:
                name = pygame.joystick.Joystick(idx).get_name()
            except pygame.error:
                name = None
        return f"Gamepad connected{': ' + name if name else ''}"
    if removed is not None and event.type == removed:
        return "Gamepad disconnected"
    return None


def _normalize_key_name(name):
    """Canonicalise a pygame key name for config storage."""
    return str(name).strip().lower().replace('_', ' ')


_KEY_NAME_ALIASES = {
    'right shift': 'K_RSHIFT',
    'left shift': 'K_LSHIFT',
    'return': 'K_RETURN',
    'enter': 'K_RETURN',
    'space': 'K_SPACE',
    'right': 'K_RIGHT',
    'left': 'K_LEFT',
    'up': 'K_UP',
    'down': 'K_DOWN',
    ',': 'K_COMMA',
    'comma': 'K_COMMA',
    '.': 'K_PERIOD',
    'period': 'K_PERIOD',
    'full stop': 'K_PERIOD',
}


def _key_constant(name):
    """Resolve a stored key name to a pygame key constant, or None."""
    if not pygame or not name:
        return None
    name = _normalize_key_name(name)
    alias = _KEY_NAME_ALIASES.get(name)
    if alias:
        return getattr(pygame, alias, None)
    attr = 'K_' + name.upper().replace(' ', '_')
    val = getattr(pygame, attr, None)
    if val is not None:
        return val
    # Unusual names (e.g. 'left ctrl') need pygame.init(); skip before then
    # so import doesn't warn.
    if not pygame.get_init():
        return None
    try:
        return pygame.key.key_code(name)
    except (ValueError, OverflowError, pygame.error, TypeError):
        return None


def _key_display_name(name):
    """Pretty-print a stored key name for menus."""
    name = _normalize_key_name(name)
    aliases = {
        'return': 'Enter',
        'right shift': 'R-Shift',
        'left shift': 'L-Shift',
        'space': 'Space',
        'right': 'Right',
        'left': 'Left',
        'up': 'Up',
        'down': 'Down',
        ',': 'Comma',
        'comma': 'Comma',
        '.': 'Period',
        'period': 'Period',
        'full stop': 'Period',
    }
    if name in aliases:
        return aliases[name]
    if len(name) == 1:
        return name.upper()
    return ' '.join(part.capitalize() for part in name.split())


def _default_key_bindings(wasd_enabled=True):
    bindings = {k: list(v) for k, v in DEFAULT_KEY_BINDINGS.items()}
    if not wasd_enabled:
        for alias, button in WASD_ALIASES:
            if alias in bindings[button]:
                bindings[button] = [n for n in bindings[button] if n != alias]
    return bindings


def _sanitize_key_bindings(raw, wasd_enabled=True):
    """Return a validated bit-name -> [key-name] map, filling any gaps."""
    bindings = _default_key_bindings(wasd_enabled)
    if not isinstance(raw, dict):
        return bindings
    for button in JOYPAD_BUTTON_KEYS:
        value = raw.get(button)
        if not isinstance(value, (list, tuple)):
            continue
        names = []
        for item in value:
            name = _normalize_key_name(item)
            if not name or name in _RESERVED_REMAP_KEY_NAMES or name in names:
                continue
            if pygame and _key_constant(name) is None:
                continue
            names.append(name)
        if names:
            bindings[button] = names
    if wasd_enabled:
        used = {name for keys in bindings.values() for name in keys}
        for alias, button in WASD_ALIASES:
            if alias not in used and alias not in bindings[button]:
                bindings[button].append(alias)
    else:
        wasd = {alias for alias, _ in WASD_ALIASES}
        for button in ('right', 'left', 'up', 'down'):
            bindings[button] = [n for n in bindings[button] if n not in wasd]
            if not bindings[button]:
                bindings[button] = list(DEFAULT_KEY_BINDINGS[button][:1])
    return bindings


def _rebuild_key_map(bindings=None, wasd_enabled=True):
    """Rebuild KEY_TO_JOYPAD_BIT from a bindings dict. Returns the map."""
    global KEY_TO_JOYPAD_BIT
    bindings = _sanitize_key_bindings(bindings, wasd_enabled)
    mapping = {}
    if pygame:
        for bit, button in enumerate(JOYPAD_BUTTON_KEYS):
            for name in bindings.get(button, ()):
                key = _key_constant(name)
                if key is not None and key not in mapping:
                    mapping[key] = bit
    KEY_TO_JOYPAD_BIT = mapping
    return bindings


def _binding_label(bindings, button):
    names = bindings.get(button) or _default_fallbacks_for(button)
    return ', '.join(_key_display_name(n) for n in names)


def _default_fallbacks_for(button):
    if button in DEFAULT_TURBO_BINDINGS:
        return DEFAULT_TURBO_BINDINGS[button]
    return DEFAULT_KEY_BINDINGS.get(button, ['space'])


def _steal_key_from_map(amap, key_name, except_button, old_primary=None):
    """Remove ``key_name`` from another button in ``amap``, swapping primaries."""
    owner = None
    for other, names in amap.items():
        if other == except_button:
            continue
        if key_name in names:
            owner = other
            break
    if owner is None:
        return
    amap[owner] = [n for n in amap[owner] if n != key_name]
    if old_primary and old_primary != key_name and old_primary not in amap[owner]:
        amap[owner] = [old_primary] + amap[owner]
    if not amap[owner]:
        fallback = _default_fallbacks_for(owner)[0]
        if fallback != key_name:
            amap[owner] = [fallback]


def _assign_binding_key(bindings, button, key_name, other_maps=None):
    """Set `button`'s primary key. If another button owns it, swap primaries."""
    key_name = _normalize_key_name(key_name)
    if not key_name or key_name in _RESERVED_REMAP_KEY_NAMES:
        return False
    if pygame and _key_constant(key_name) is None:
        return False
    old_primary = (bindings.get(button) or [None])[0]
    _steal_key_from_map(bindings, key_name, button, old_primary)
    if other_maps:
        for extra in other_maps:
            _steal_key_from_map(extra, key_name, None, old_primary)
    rest = [n for n in bindings.get(button, []) if n != key_name]
    bindings[button] = [key_name] + rest
    return True


def _sanitize_turbo_bindings(raw):
    """Return a validated turbo_a/turbo_b -> [key-name] map."""
    bindings = {k: list(v) for k, v in DEFAULT_TURBO_BINDINGS.items()}
    if not isinstance(raw, dict):
        return bindings
    for button in TURBO_BUTTON_KEYS:
        value = raw.get(button)
        if not isinstance(value, (list, tuple)):
            continue
        names = []
        for item in value:
            name = _normalize_key_name(item)
            if not name or name in _RESERVED_REMAP_KEY_NAMES or name in names:
                continue
            if pygame and _key_constant(name) is None:
                continue
            names.append(name)
        if names:
            bindings[button] = names
    return bindings


def _rebuild_turbo_map(bindings=None):
    """Rebuild KEY_TO_TURBO_BIT (pygame key -> joypad A/B bit). Returns the map."""
    global KEY_TO_TURBO_BIT
    bindings = _sanitize_turbo_bindings(bindings)
    mapping = {}
    bit_for = {'turbo_a': 4, 'turbo_b': 5}
    if pygame:
        for button, bit in bit_for.items():
            for name in bindings.get(button, ()):
                key = _key_constant(name)
                if key is not None and key not in mapping and key not in KEY_TO_JOYPAD_BIT:
                    mapping[key] = bit
    KEY_TO_TURBO_BIT = mapping
    return bindings


def _menu_nav_action(action):
    """Map WASD / gamepad strings onto the arrow-key constants used by menus."""
    if action in ('up', 'down', 'left', 'right', 'select', 'back',
                  'home', 'end', 'pageup', 'pagedown', 'toggle_fullscreen'):
        return action
    if not pygame:
        return action
    if action in (pygame.K_UP, pygame.K_w):
        return pygame.K_UP
    if action in (pygame.K_DOWN, pygame.K_s):
        return pygame.K_DOWN
    if action in (pygame.K_LEFT, pygame.K_a):
        return pygame.K_LEFT
    if action in (pygame.K_RIGHT, pygame.K_d):
        return pygame.K_RIGHT
    if action == pygame.K_HOME:
        return 'home'
    if action == pygame.K_END:
        return 'end'
    if action == pygame.K_PAGEUP:
        return 'pageup'
    if action == pygame.K_PAGEDOWN:
        return 'pagedown'
    if action == pygame.K_F11:
        return 'toggle_fullscreen'
    return action


# ── Persistent settings configuration ─────────────────────────────────
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()
_CONFIG_PATH = os.path.join(_THIS_DIR, "gbc_config.json")

def _load_config():
    """Load user settings from the JSON config file. Returns {} on any error."""
    try:
        with open(_CONFIG_PATH, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

def _save_config(cfg):
    """Merge ``cfg`` into the existing JSON config and write it back."""
    current = _load_config()
    current.update(cfg)
    try:
        with open(_CONFIG_PATH, 'w') as f:
            json.dump(current, f, indent=2)
        return True
    except OSError:
        return False


def _rebuild_input_maps(key_bindings=None, wasd_enabled=True, turbo_bindings=None):
    """Rebuild joypad and turbo key maps together so they cannot overlap."""
    keys = _rebuild_key_map(key_bindings, wasd_enabled)
    turbo = _rebuild_turbo_map(turbo_bindings)
    return keys, turbo


# Apply persisted (or default) key bindings as soon as pygame is importable.
_cfg0 = _load_config()
_rebuild_input_maps(_cfg0.get('key_bindings'), bool(_cfg0.get('wasd_enabled', True)),
                    _cfg0.get('turbo_bindings'))
del _cfg0


class Registers:
    """Manages the 8-bit and 16-bit paired registers of the LR35902 CPU."""
    __slots__ = ('a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'sp', 'pc')

    def __init__(self):
        self.a = 0x00
        self.f = 0x00
        self.b = 0x00
        self.c = 0x00
        self.d = 0x00
        self.e = 0x00
        self.h = 0x00
        self.l = 0x00
        self.sp = 0xFFFE  # Stack pointer usually starts here
        self.pc = 0x0100  # Execution usually begins at 0x0100 after boot ROM

    # 16-bit register property pairs
    @property
    def af(self):
        return (self.a << 8) | self.f

    @af.setter
    def af(self, value):
        self.a = (value >> 8) & 0xFF
        self.f = value & 0xF0  # Lower 4 bits of F are always 0

    @property
    def bc(self):
        return (self.b << 8) | self.c

    @bc.setter
    def bc(self, value):
        self.b = (value >> 8) & 0xFF
        self.c = value & 0xFF

    @property
    def de(self):
        return (self.d << 8) | self.e

    @de.setter
    def de(self, value):
        self.d = (value >> 8) & 0xFF
        self.e = value & 0xFF

    @property
    def hl(self):
        return (self.h << 8) | self.l

    @hl.setter
    def hl(self, value):
        self.h = (value >> 8) & 0xFF
        self.l = value & 0xFF

    # Flag Helpers
    def set_flag(self, flag_bit, state):
        if state:
            self.f |= (1 << flag_bit)
        else:
            self.f &= ~(1 << flag_bit)

    def get_flag(self, flag_bit):
        return (self.f >> flag_bit) & 1

    def set_znhc(self, z, n, h, c):
        """Write Z/N/H/C in one assignment (hot path)."""
        self.f = ((0x80 if z else 0) | (0x40 if n else 0) |
                  (0x20 if h else 0) | (0x10 if c else 0))


class CPU:
    """The LR35902 CPU."""
    # Opcodes with no defined behaviour on real hardware (they hang the CPU).
    INVALID_OPS = frozenset({0xD3, 0xDB, 0xDD, 0xE3, 0xE4, 0xEB, 0xEC, 0xED, 0xF4, 0xFC, 0xFD})

    def __init__(self, mmu):
        self.mmu = mmu
        self.mem = mmu.memory
        self.reg = Registers()
        self.halted = False
        self.interrupts_master_enabled = False
        self.ime_pending = False
        self.halt_bug_pending = False  # HALT bug: next opcode fetch should not advance PC
        self.trace_enabled = False
        self.branch_trace = deque(maxlen=8192)
        self.invalid_opcode_count = 0
        self._logged_opcodes = set()
        self.trace_branch = _noop_trace
        self.current_opcode_pc = 0

    def _record_branch(self, kind, old_pc, new_pc, opcode):
        self.branch_trace.append({
            "kind": kind,
            "from": old_pc,
            "to": new_pc,
            "opcode": opcode,
            "rom_bank": self.mmu.rom_bank,
            "sp": self.reg.sp,
            "af": (self.reg.a << 8) | self.reg.f,
            "bc": self.reg.bc,
            "de": self.reg.de,
            "hl": self.reg.hl,
            "ime": self.interrupts_master_enabled,
            "ie": self.mmu.read_byte(0xFFFF),
            "if": self.mmu.read_byte(0xFF0F),
        })

    def set_branch_trace(self, enabled):
        """Enable or disable branch tracing (swaps in a no-op when off)."""
        self.trace_enabled = enabled
        self.trace_branch = self._record_branch if enabled else _noop_trace

    def dump_branch_trace(self):
        print("=== Last 64 branches ===")
        for e in self.branch_trace[-64:]:
            print(f"  {e['kind']:12s} 0x{e['from']:04X}->0x{e['to']:04X} op=0x{e['opcode']:02X} bank={e['rom_bank']:02X} SP=0x{e['sp']:04X} AF=0x{e['af']:04X} BC=0x{e['bc']:04X} DE=0x{e['de']:04X} HL=0x{e['hl']:04X} IME={e['ime']} IE=0x{e['ie']:02X} IF=0x{e['if']:02X}")

    def _get_r8(self, idx):
        if idx == 0: return self.reg.b
        elif idx == 1: return self.reg.c
        elif idx == 2: return self.reg.d
        elif idx == 3: return self.reg.e
        elif idx == 4: return self.reg.h
        elif idx == 5: return self.reg.l
        elif idx == 6: return self.mmu.read_byte(self.reg.hl)
        elif idx == 7: return self.reg.a

    def _set_r8(self, idx, val):
        val &= 0xFF
        if idx == 0: self.reg.b = val
        elif idx == 1: self.reg.c = val
        elif idx == 2: self.reg.d = val
        elif idx == 3: self.reg.e = val
        elif idx == 4: self.reg.h = val
        elif idx == 5: self.reg.l = val
        elif idx == 6: self.mmu.write_byte(self.reg.hl, val)
        elif idx == 7: self.reg.a = val

    def _get_r16_stk(self, idx):
        if idx == 0: return self.reg.bc
        elif idx == 1: return self.reg.de
        elif idx == 2: return self.reg.hl
        elif idx == 3: return self.reg.sp

    def _set_r16_stk(self, idx, val):
        val &= 0xFFFF
        if idx == 0: self.reg.bc = val
        elif idx == 1: self.reg.de = val
        elif idx == 2: self.reg.hl = val
        elif idx == 3: self.reg.sp = val

    def _check_cond(self, cc):
        if cc == 0: return self.reg.get_flag(FLAG_Z) == 0
        elif cc == 1: return self.reg.get_flag(FLAG_Z) == 1
        elif cc == 2: return self.reg.get_flag(FLAG_C) == 0
        elif cc == 3: return self.reg.get_flag(FLAG_C) == 1

    def _alu_a_op(self, op_type, operand):
        a = self.reg.a
        carry = (self.reg.f >> FLAG_C) & 1
        set_znhc = self.reg.set_znhc
        if op_type == 0:
            result = a + operand
            set_znhc((result & 0xFF) == 0, 0,
                     (a & 0xF) + (operand & 0xF) > 0xF, result > 0xFF)
            self.reg.a = result & 0xFF
        elif op_type == 1:
            result = a + operand + carry
            set_znhc((result & 0xFF) == 0, 0,
                     (a & 0xF) + (operand & 0xF) + carry > 0xF, result > 0xFF)
            self.reg.a = result & 0xFF
        elif op_type == 2:
            result = a - operand
            set_znhc((result & 0xFF) == 0, 1,
                     (a & 0xF) < (operand & 0xF), a < operand)
            self.reg.a = result & 0xFF
        elif op_type == 3:
            result = a - operand - carry
            set_znhc((result & 0xFF) == 0, 1,
                     (a & 0xF) < (operand & 0xF) + carry, a < operand + carry)
            self.reg.a = result & 0xFF
        elif op_type == 4:
            result = a & operand
            set_znhc(result == 0, 0, 1, 0)
            self.reg.a = result
        elif op_type == 5:
            result = a ^ operand
            set_znhc(result == 0, 0, 0, 0)
            self.reg.a = result
        elif op_type == 6:
            result = a | operand
            set_znhc(result == 0, 0, 0, 0)
            self.reg.a = result
        elif op_type == 7:
            result = a - operand
            set_znhc((result & 0xFF) == 0, 1,
                     (a & 0xF) < (operand & 0xF), a < operand)

    def _inc_r8(self, val):
        result = (val + 1) & 0xFF
        self.reg.set_znhc(result == 0, 0, (val & 0xF) == 0xF, (self.reg.f >> FLAG_C) & 1)
        return result

    def _dec_r8(self, val):
        result = (val - 1) & 0xFF
        self.reg.set_znhc(result == 0, 1, (val & 0xF) == 0, (self.reg.f >> FLAG_C) & 1)
        return result

    def _add_hl_rr(self, rr_val):
        hl = self.reg.hl
        result = hl + rr_val
        self.reg.set_flag(FLAG_N, 0)
        self.reg.set_flag(FLAG_H, (hl & 0xFFF) + (rr_val & 0xFFF) > 0xFFF)
        self.reg.set_flag(FLAG_C, result > 0xFFFF)
        self.reg.hl = result & 0xFFFF

    def _daa(self):
        a = self.reg.a
        c = self.reg.get_flag(FLAG_C)
        h = self.reg.get_flag(FLAG_H)
        n = self.reg.get_flag(FLAG_N)
        if n:
            if c: a = (a - 0x60) & 0xFF
            if h: a = (a - 0x06) & 0xFF
        else:
            if c or a > 0x99:
                a = (a + 0x60) & 0xFF
                c = 1
            if h or (a & 0x0F) > 0x09:
                a = (a + 0x06) & 0xFF
        self.reg.a = a
        self.reg.set_flag(FLAG_Z, a == 0)
        self.reg.set_flag(FLAG_H, 0)
        self.reg.set_flag(FLAG_C, c)

    def _push(self, val):
        self.reg.sp = (self.reg.sp - 1) & 0xFFFF
        self.mmu.write_byte(self.reg.sp, (val >> 8) & 0xFF)
        self.reg.sp = (self.reg.sp - 1) & 0xFFFF
        self.mmu.write_byte(self.reg.sp, val & 0xFF)

    def _pop(self):
        lo = self.mmu.read_byte(self.reg.sp)
        self.reg.sp = (self.reg.sp + 1) & 0xFFFF
        hi = self.mmu.read_byte(self.reg.sp)
        self.reg.sp = (self.reg.sp + 1) & 0xFFFF
        return (hi << 8) | lo

    def fetch_byte(self):
        """Fetches the next byte at PC and increments PC."""
        pc = self.reg.pc
        mmu = self.mmu
        if mmu.key1 & 0x80:
            mmu._add_cart_wait(pc)
        val = self.mem[pc]
        self.reg.pc = (pc + 1) & 0xFFFF
        return val

    def fetch_word(self):
        """Fetches the next 16-bit word at PC and increments PC twice."""
        pc = self.reg.pc
        mmu = self.mmu
        if mmu.key1 & 0x80:
            mmu._add_cart_wait(pc)
            mmu._add_cart_wait((pc + 1) & 0xFFFF)
        mem = self.mem
        val = mem[pc] | (mem[(pc + 1) & 0xFFFF] << 8)
        self.reg.pc = (pc + 2) & 0xFFFF
        return val

    def step(self):
        """Fetches, decodes, and executes a single instruction."""
        mem = self.mem
        mmu = self.mmu
        if self.halted:
            if mem[0xFFFF] & mem[0xFF0F]:
                self.halted = False
                if self.interrupts_master_enabled:
                    return 4 + self._handle_interrupts()
            return 4

        if self.ime_pending:
            self.interrupts_master_enabled = True
            self.ime_pending = False

        reg = self.reg
        pc = reg.pc
        if mmu.key1 & 0x80:
            mmu._add_cart_wait(pc)
        if self.halt_bug_pending:
            self.halt_bug_pending = False
            opcode = mem[pc]
        else:
            opcode = mem[pc]
            reg.pc = (pc + 1) & 0xFFFF
        if self.trace_enabled:
            self.current_opcode_pc = pc
        cycles = self.execute(opcode)
        wait = mmu.cart_wait_cycles
        if wait:
            mmu.cart_wait_cycles = 0
            cycles += wait
        if self.interrupts_master_enabled:
            cycles += self._handle_interrupts()
        return cycles

    def _handle_interrupts(self):
        mem = self.mem
        pending = mem[0xFFFF] & mem[0xFF0F]
        if pending == 0:
            return 0
        self.interrupts_master_enabled = False
        for bit in range(5):
            if pending & (1 << bit):
                self.mem[0xFF0F] &= ~(1 << bit)
                vector = 0x0040 + (bit * 8)
                self.trace_branch(f"IRQ{bit}", self.reg.pc, vector, 0xFF)
                self._push(self.reg.pc)
                self.reg.pc = vector
                return 20
        return 0

    def execute(self, opcode):
        # Fast path: inline top ~15 opcodes (~52% of all) to bypass if/elif chains.
        if opcode == 0xEA:  # LD (nn),A - 11.5%
            self.mmu.write_byte(self.fetch_word(), self.reg.a); return 16
        if opcode == 0xFA:  # LD A,(nn) - 9.8%
            self.reg.a = self.mmu.read_byte(self.fetch_word()); return 16
        if opcode == 0xCB:  # CB prefix - 7.7%
            return self.execute_cb(self.fetch_byte())
        if opcode == 0x3A:  # LD A,(HL-) - 3.4%
            self.reg.a = self.mmu.read_byte(self.reg.hl)
            self.reg.hl = (self.reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x20:  # JR NZ - 3.3%
            offset = self.fetch_byte()
            if (self.reg.f & 0x80) == 0:
                offset = _sign_extend_byte(offset)
                self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
            return 8
        if opcode == 0x19:  # ADD HL,DE - 2.7%
            self._add_hl_rr(self.reg.de); return 8
        if opcode == 0xD5:  # PUSH DE - 2.4%
            self._push(self.reg.de); return 16
        if opcode == 0xE1:  # POP HL - 2.2%
            self.reg.hl = self._pop(); return 12
        if opcode == 0xCE:  # ADC A,n - 2.2%
            self._alu_a_op(1, self.fetch_byte()); return 8
        if opcode == 0x22:  # LD (HL+),A - 2.1%
            self.mmu.write_byte(self.reg.hl, self.reg.a)
            self.reg.hl = (self.reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x0B:  # DEC BC - 2.1%
            self.reg.bc = (self.reg.bc - 1) & 0xFFFF; return 8
        if opcode == 0x28:  # JR Z - 1.6%
            offset = self.fetch_byte()
            if self.reg.f & 0x80:
                offset = _sign_extend_byte(offset)
                self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
            return 8
        if opcode == 0xC9:  # RET - 1.5%
            self.reg.pc = self._pop(); return 16
        if opcode == 0xCD:  # CALL nn - 1.5%
            addr = self.fetch_word()
            self._push(self.reg.pc)
            self.reg.pc = addr; return 24
        if opcode == 0x00:  # NOP
            return 4
        if opcode == 0x18:  # JR - common in tight loops
            offset = self.fetch_byte()
            if offset & 0x80:
                offset -= 256
            self.reg.pc = (self.reg.pc + offset) & 0xFFFF
            return 12
        if opcode == 0xE0:  # LDH (a8),A - frequent IO writes
            self.mmu.write_byte(0xFF00 + self.fetch_byte(), self.reg.a)
            return 12
        if opcode == 0xF0:  # LD A,(a8) - frequent IO reads
            self.reg.a = self.mmu.read_byte(0xFF00 + self.fetch_byte())
            return 12
        # Hot-path opcodes: LD r,r (0x40-0x7F), ALU A,r (0x80-0xBF).
        if opcode < 0x40:
            return self._exec_low(opcode)
        if opcode < 0x80:
            if opcode == 0x76:
                if not self.interrupts_master_enabled and (self.mem[0xFFFF] & self.mem[0xFF0F]):
                    # HALT bug: IME=0 with pending interrupt -> PC not incremented
                    self.halt_bug_pending = True
                else:
                    self.halted = True
                return 4
            dst = (opcode >> 3) & 0x7
            src = opcode & 0x7
            self._set_r8(dst, self._get_r8(src))
            return 4
        if opcode < 0xC0:
            op_type = (opcode >> 3) & 0x7
            operand_idx = opcode & 0x7
            self._alu_a_op(op_type, self._get_r8(operand_idx))
            return 8 if operand_idx == 6 else 4
        return self._exec_high(opcode)

    def _exec_low(self, opcode):
        # 0x00-0x3F: less common opcodes.
        reg = self.reg; mmu = self.mmu
        if opcode == 0x00:
            return 4
        if opcode == 0x01:
            reg.bc = self.fetch_word(); return 12
        if opcode == 0x02:
            mmu.write_byte(reg.bc, reg.a); return 8
        if opcode == 0x03:
            reg.bc = (reg.bc + 1) & 0xFFFF; return 8
        if opcode == 0x04:
            reg.b = self._inc_r8(reg.b); return 4
        if opcode == 0x05:
            reg.b = self._dec_r8(reg.b); return 4
        if opcode == 0x06:
            reg.b = self.fetch_byte(); return 8
        if opcode == 0x07:
            a = reg.a; carry = (a >> 7) & 1
            reg.a = ((a << 1) | carry) & 0xFF
            reg.set_flag(FLAG_Z, 0); reg.set_flag(FLAG_N, 0)
            reg.set_flag(FLAG_H, 0); reg.set_flag(FLAG_C, carry); return 4
        if opcode == 0x08:
            addr = self.fetch_word()
            mmu.write_word(addr, reg.sp); return 20
        if opcode == 0x09:
            self._add_hl_rr(reg.bc); return 8
        if opcode == 0x0A:
            reg.a = mmu.read_byte(reg.bc); return 8
        if opcode == 0x0B:
            reg.bc = (reg.bc - 1) & 0xFFFF; return 8
        if opcode == 0x0C:
            reg.c = self._inc_r8(reg.c); return 4
        if opcode == 0x0D:
            reg.c = self._dec_r8(reg.c); return 4
        if opcode == 0x0E:
            reg.c = self.fetch_byte(); return 8
        if opcode == 0x0F:
            a = reg.a; carry = a & 1
            reg.a = ((a >> 1) | (carry << 7)) & 0xFF
            reg.set_flag(FLAG_Z, 0); reg.set_flag(FLAG_N, 0)
            reg.set_flag(FLAG_H, 0); reg.set_flag(FLAG_C, carry); return 4
        if opcode == 0x10:
            self.fetch_byte()  # consume 0x00 padding
            if mmu.is_cgb and (mmu.key1 & 0x01):
                mmu.key1 ^= 0x80  # toggle double-speed
                mmu.key1 &= ~0x01  # clear prepare flag
                apu = getattr(mmu, 'apu', None)
                if apu is not None:
                    apu._sync_fs_remain(apu.fs_div, bool(mmu.key1 & 0x80))
            else:
                mmu.memory[0xFF40] &= 0x7F  # disable LCD
                self.halted = True  # wakes on any pending interrupt (joypad)
            return 4  # STOP
        if opcode == 0x11:
            reg.de = self.fetch_word(); return 12
        if opcode == 0x12:
            mmu.write_byte(reg.de, reg.a); return 8
        if opcode == 0x13:
            reg.de = (reg.de + 1) & 0xFFFF; return 8
        if opcode == 0x14:
            reg.d = self._inc_r8(reg.d); return 4
        if opcode == 0x15:
            reg.d = self._dec_r8(reg.d); return 4
        if opcode == 0x16:
            reg.d = self.fetch_byte(); return 8
        if opcode == 0x17:
            a = reg.a; old_c = reg.get_flag(FLAG_C)
            new_c = (a >> 7) & 1
            reg.a = ((a << 1) | old_c) & 0xFF
            reg.set_flag(FLAG_Z, 0); reg.set_flag(FLAG_N, 0)
            reg.set_flag(FLAG_H, 0); reg.set_flag(FLAG_C, new_c); return 4
        if opcode == 0x18:
            offset = self.fetch_byte()
            offset = _sign_extend_byte(offset)
            new_pc = (reg.pc + offset) & 0xFFFF
            self.trace_branch("JR", self.current_opcode_pc, new_pc, opcode)
            reg.pc = new_pc; return 12
        if opcode == 0x19:
            self._add_hl_rr(reg.de); return 8
        if opcode == 0x1A:
            reg.a = mmu.read_byte(reg.de); return 8
        if opcode == 0x1B:
            reg.de = (reg.de - 1) & 0xFFFF; return 8
        if opcode == 0x1C:
            reg.e = self._inc_r8(reg.e); return 4
        if opcode == 0x1D:
            reg.e = self._dec_r8(reg.e); return 4
        if opcode == 0x1E:
            reg.e = self.fetch_byte(); return 8
        if opcode == 0x1F:
            a = reg.a; old_c = reg.get_flag(FLAG_C)
            new_c = a & 1
            reg.a = ((a >> 1) | (old_c << 7)) & 0xFF
            reg.set_flag(FLAG_Z, 0); reg.set_flag(FLAG_N, 0)
            reg.set_flag(FLAG_H, 0); reg.set_flag(FLAG_C, new_c); return 4
        if opcode == 0x20:
            offset = self.fetch_byte()
            if self._check_cond(0):
                offset = _sign_extend_byte(offset)
                new_pc = (reg.pc + offset) & 0xFFFF
                self.trace_branch("JR NZ", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 12
            return 8
        if opcode == 0x21:
            reg.hl = self.fetch_word(); return 12
        if opcode == 0x22:
            mmu.write_byte(reg.hl, reg.a)
            reg.hl = (reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x23:
            reg.hl = (reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x24:
            reg.h = self._inc_r8(reg.h); return 4
        if opcode == 0x25:
            reg.h = self._dec_r8(reg.h); return 4
        if opcode == 0x26:
            reg.h = self.fetch_byte(); return 8
        if opcode == 0x27:
            self._daa(); return 4
        if opcode == 0x28:
            offset = self.fetch_byte()
            if self._check_cond(1):
                offset = _sign_extend_byte(offset)
                new_pc = (reg.pc + offset) & 0xFFFF
                self.trace_branch("JR Z", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 12
            return 8
        if opcode == 0x29:
            self._add_hl_rr(reg.hl); return 8
        if opcode == 0x2A:
            reg.a = mmu.read_byte(reg.hl)
            reg.hl = (reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x2B:
            reg.hl = (reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x2C:
            reg.l = self._inc_r8(reg.l); return 4
        if opcode == 0x2D:
            reg.l = self._dec_r8(reg.l); return 4
        if opcode == 0x2E:
            reg.l = self.fetch_byte(); return 8
        if opcode == 0x2F:
            reg.a ^= 0xFF
            reg.set_flag(FLAG_N, 1); reg.set_flag(FLAG_H, 1); return 4
        if opcode == 0x30:
            offset = self.fetch_byte()
            if self._check_cond(2):
                offset = _sign_extend_byte(offset)
                new_pc = (reg.pc + offset) & 0xFFFF
                self.trace_branch("JR NC", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 12
            return 8
        if opcode == 0x31:
            reg.sp = self.fetch_word(); return 12
        if opcode == 0x32:
            mmu.write_byte(reg.hl, reg.a)
            reg.hl = (reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x33:
            reg.sp = (reg.sp + 1) & 0xFFFF; return 8
        if opcode == 0x34:
            val = mmu.read_byte(reg.hl)
            mmu.write_byte(reg.hl, self._inc_r8(val)); return 12
        if opcode == 0x35:
            val = mmu.read_byte(reg.hl)
            mmu.write_byte(reg.hl, self._dec_r8(val)); return 12
        if opcode == 0x36:
            mmu.write_byte(reg.hl, self.fetch_byte()); return 12
        if opcode == 0x37:
            reg.set_flag(FLAG_N, 0); reg.set_flag(FLAG_H, 0)
            reg.set_flag(FLAG_C, 1); return 4
        if opcode == 0x38:
            offset = self.fetch_byte()
            if self._check_cond(3):
                offset = _sign_extend_byte(offset)
                new_pc = (reg.pc + offset) & 0xFFFF
                self.trace_branch("JR C", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 12
            return 8
        if opcode == 0x39:
            self._add_hl_rr(reg.sp); return 8
        if opcode == 0x3A:
            reg.a = mmu.read_byte(reg.hl)
            reg.hl = (reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x3B:
            reg.sp = (reg.sp - 1) & 0xFFFF; return 8
        if opcode == 0x3C:
            reg.a = self._inc_r8(reg.a); return 4
        if opcode == 0x3D:
            reg.a = self._dec_r8(reg.a); return 4
        if opcode == 0x3E:
            reg.a = self.fetch_byte(); return 8
        # 0x3F
        carry = reg.get_flag(FLAG_C)
        reg.set_flag(FLAG_N, 0); reg.set_flag(FLAG_H, 0)
        reg.set_flag(FLAG_C, carry ^ 1)
        return 4

    def _exec_high(self, opcode):
        # 0xC0-0xFF: control flow and stack opcodes.
        reg = self.reg; mmu = self.mmu
        if opcode == 0xC0:
            if self._check_cond(0):
                new_pc = self._pop()
                self.trace_branch("RET NZ", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 20
            return 8
        if opcode == 0xC1:
            reg.bc = self._pop(); return 12
        if opcode == 0xC2:
            addr = self.fetch_word()
            if self._check_cond(0):
                self.trace_branch("JP NZ", self.current_opcode_pc, addr, opcode)
                reg.pc = addr; return 16
            return 12
        if opcode == 0xC3:
            new_pc = self.fetch_word()
            self.trace_branch("JP", self.current_opcode_pc, new_pc, opcode)
            reg.pc = new_pc; return 16
        if opcode == 0xC4:
            addr = self.fetch_word()
            if self._check_cond(0):
                self.trace_branch("CALL NZ", self.current_opcode_pc, addr, opcode)
                self._push(reg.pc)
                reg.pc = addr; return 24
            return 12
        if opcode == 0xC5:
            self._push(reg.bc); return 16
        if opcode == 0xC6:
            self._alu_a_op(0, self.fetch_byte()); return 8
        if opcode == 0xC7:
            self.trace_branch("RST 00", self.current_opcode_pc, 0x00, opcode)
            self._push(reg.pc); reg.pc = 0x00; return 16
        if opcode == 0xC8:
            if self._check_cond(1):
                new_pc = self._pop()
                self.trace_branch("RET Z", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 20
            return 8
        if opcode == 0xC9:
            new_pc = self._pop()
            self.trace_branch("RET", self.current_opcode_pc, new_pc, opcode)
            reg.pc = new_pc; return 16
        if opcode == 0xCA:
            addr = self.fetch_word()
            if self._check_cond(1):
                self.trace_branch("JP Z", self.current_opcode_pc, addr, opcode)
                reg.pc = addr; return 16
            return 12
        if opcode == 0xCC:
            addr = self.fetch_word()
            if self._check_cond(1):
                self.trace_branch("CALL Z", self.current_opcode_pc, addr, opcode)
                self._push(reg.pc)
                reg.pc = addr; return 24
            return 12
        if opcode == 0xCD:
            addr = self.fetch_word()
            self.trace_branch("CALL", self.current_opcode_pc, addr, opcode)
            self._push(reg.pc)
            reg.pc = addr; return 24
        if opcode == 0xCE:
            self._alu_a_op(1, self.fetch_byte()); return 8
        if opcode == 0xCF:
            self.trace_branch("RST 08", self.current_opcode_pc, 0x08, opcode)
            self._push(reg.pc); reg.pc = 0x08; return 16
        if opcode == 0xD0:
            if self._check_cond(2):
                new_pc = self._pop()
                self.trace_branch("RET NC", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 20
            return 8
        if opcode == 0xD1:
            reg.de = self._pop(); return 12
        if opcode == 0xD2:
            addr = self.fetch_word()
            if self._check_cond(2):
                self.trace_branch("JP NC", self.current_opcode_pc, addr, opcode)
                reg.pc = addr; return 16
            return 12
        if opcode == 0xD4:
            addr = self.fetch_word()
            if self._check_cond(2):
                self.trace_branch("CALL NC", self.current_opcode_pc, addr, opcode)
                self._push(reg.pc)
                reg.pc = addr; return 24
            return 12
        if opcode == 0xD5:
            self._push(reg.de); return 16
        if opcode == 0xD6:
            self._alu_a_op(2, self.fetch_byte()); return 8
        if opcode == 0xD7:
            self.trace_branch("RST 10", self.current_opcode_pc, 0x10, opcode)
            self._push(reg.pc); reg.pc = 0x10; return 16
        if opcode == 0xD8:
            if self._check_cond(3):
                new_pc = self._pop()
                self.trace_branch("RET C", self.current_opcode_pc, new_pc, opcode)
                reg.pc = new_pc; return 20
            return 8
        if opcode == 0xD9:
            new_pc = self._pop()
            self.trace_branch("RETI", self.current_opcode_pc, new_pc, opcode)
            reg.pc = new_pc
            self.interrupts_master_enabled = True; return 16
        if opcode == 0xDA:
            addr = self.fetch_word()
            if self._check_cond(3):
                self.trace_branch("JP C", self.current_opcode_pc, addr, opcode)
                reg.pc = addr; return 16
            return 12
        if opcode == 0xDC:
            addr = self.fetch_word()
            if self._check_cond(3):
                self.trace_branch("CALL C", self.current_opcode_pc, addr, opcode)
                self._push(reg.pc)
                reg.pc = addr; return 24
            return 12
        if opcode == 0xDE:
            self._alu_a_op(3, self.fetch_byte()); return 8
        if opcode == 0xDF:
            self.trace_branch("RST 18", self.current_opcode_pc, 0x18, opcode)
            self._push(reg.pc); reg.pc = 0x18; return 16
        if opcode == 0xE0:
            mmu.write_byte(0xFF00 | self.fetch_byte(), reg.a); return 12
        if opcode == 0xE1:
            reg.hl = self._pop(); return 12
        if opcode == 0xE2:
            mmu.write_byte(0xFF00 | reg.c, reg.a); return 8
        if opcode == 0xE5:
            self._push(reg.hl); return 16
        if opcode == 0xE6:
            self._alu_a_op(4, self.fetch_byte()); return 8
        if opcode == 0xE7:
            self.trace_branch("RST 20", self.current_opcode_pc, 0x20, opcode)
            self._push(reg.pc); reg.pc = 0x20; return 16
        if opcode == 0xE8:
            offset = self.fetch_byte()
            offset = _sign_extend_byte(offset)
            result = reg.sp + offset
            reg.set_flag(FLAG_Z, 0); reg.set_flag(FLAG_N, 0)
            reg.set_flag(FLAG_H, (reg.sp & 0xF) + (offset & 0xF) > 0xF)
            reg.set_flag(FLAG_C, (reg.sp & 0xFF) + (offset & 0xFF) > 0xFF)
            reg.sp = result & 0xFFFF; return 16
        if opcode == 0xE9:
            self.trace_branch("JP HL", self.current_opcode_pc, reg.hl, opcode)
            reg.pc = reg.hl; return 4
        if opcode == 0xEA:
            mmu.write_byte(self.fetch_word(), reg.a); return 16
        if opcode == 0xEE:
            self._alu_a_op(5, self.fetch_byte()); return 8
        if opcode == 0xEF:
            self.trace_branch("RST 28", self.current_opcode_pc, 0x28, opcode)
            self._push(reg.pc); reg.pc = 0x28; return 16
        if opcode == 0xF0:
            reg.a = mmu.read_byte(0xFF00 | self.fetch_byte()); return 12
        if opcode == 0xF1:
            reg.af = self._pop(); return 12
        if opcode == 0xF2:
            reg.a = mmu.read_byte(0xFF00 | reg.c); return 8
        if opcode == 0xF3:
            self.interrupts_master_enabled = False; return 4
        if opcode == 0xF5:
            self._push(reg.af); return 16
        if opcode == 0xF6:
            self._alu_a_op(6, self.fetch_byte()); return 8
        if opcode == 0xF7:
            self.trace_branch("RST 30", self.current_opcode_pc, 0x30, opcode)
            self._push(reg.pc); reg.pc = 0x30; return 16
        if opcode == 0xF8:
            offset = self.fetch_byte()
            offset = _sign_extend_byte(offset)
            result = reg.sp + offset
            reg.set_flag(FLAG_Z, 0); reg.set_flag(FLAG_N, 0)
            reg.set_flag(FLAG_H, (reg.sp & 0xF) + (offset & 0xF) > 0xF)
            reg.set_flag(FLAG_C, (reg.sp & 0xFF) + (offset & 0xFF) > 0xFF)
            reg.hl = result & 0xFFFF; return 12
        if opcode == 0xF9:
            reg.sp = reg.hl; return 8
        if opcode == 0xFA:
            reg.a = mmu.read_byte(self.fetch_word()); return 16
        if opcode == 0xFB:
            self.ime_pending = True; return 4
        if opcode == 0xFE:
            self._alu_a_op(7, self.fetch_byte()); return 8
        if opcode == 0xFF:
            self.trace_branch("RST 38", self.current_opcode_pc, 0x38, opcode)
            self._push(reg.pc); reg.pc = 0x38; return 16
        if opcode in self.INVALID_OPS:
            # Hardware-illegal opcode (locks up a real DMG/CGB); treat as NOP.
            self.invalid_opcode_count += 1
            return 4
        self.invalid_opcode_count += 1
        if opcode not in self._logged_opcodes:
            self._logged_opcodes.add(opcode)
            logging.warning(
                f"Unimplemented opcode: {opcode:02X} at PC: {self.current_opcode_pc:04X}")
        return 4

    def execute_cb(self, cb_opcode):
        """Executes prefixed CB instructions."""
        reg_idx = cb_opcode & 0x7
        bit_pos = (cb_opcode >> 3) & 0x7
        op_group = (cb_opcode >> 6) & 0x3

        # Determine operand (read from register or (HL), with cycle info)
        # BIT takes 12 cycles for (HL), 8 for r. Other CB ops take 16 for (HL), 8 for r.
        if reg_idx == 6:
            val = self.mmu.read_byte(self.reg.hl)
            is_hl = True
        else:
            val = self._get_r8(reg_idx)
            is_hl = False

        if op_group == 0:
            ops = (self._cb_rlc, self._cb_rrc, self._cb_rl, self._cb_rr,
                   self._cb_sla, self._cb_sra, self._cb_swap, self._cb_srl)
            result = ops[bit_pos](val)
            cycles = 16 if is_hl else 8
        elif op_group == 1:
            z = self._cb_bit(val, bit_pos)
            # BIT: Z from tested bit, N=0, H=1, C preserved
            self.reg.f = (self.reg.f & 0x10) | (0x80 if z else 0) | 0x20
            return 12 if is_hl else 8
        elif op_group == 2:
            result = self._cb_res(val, bit_pos)
            cycles = 16 if is_hl else 8
        else:
            result = self._cb_set(val, bit_pos)
            cycles = 16 if is_hl else 8

        if is_hl:
            self.mmu.write_byte(self.reg.hl, result)
        else:
            self._set_r8(reg_idx, result)
        return cycles

    # ── CB-prefix rotate / shift helpers (extracted from execute_cb to
    #    avoid recreating inner functions 900+ times per frame) ────────

    def _cb_rlc(self, val):
        carry = (val >> 7) & 1
        result = ((val << 1) | carry) & 0xFF
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_rrc(self, val):
        carry = val & 1
        result = ((val >> 1) | (carry << 7)) & 0xFF
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_rl(self, val):
        old_c = (self.reg.f >> FLAG_C) & 1
        carry = (val >> 7) & 1
        result = ((val << 1) | old_c) & 0xFF
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_rr(self, val):
        old_c = (self.reg.f >> FLAG_C) & 1
        carry = val & 1
        result = ((val >> 1) | (old_c << 7)) & 0xFF
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_sla(self, val):
        carry = (val >> 7) & 1
        result = (val << 1) & 0xFF
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_sra(self, val):
        carry = val & 1
        result = (val >> 1) | (val & 0x80)
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_srl(self, val):
        carry = val & 1
        result = val >> 1
        self.reg.set_znhc(result == 0, 0, 0, carry)
        return result

    def _cb_swap(self, val):
        result = ((val & 0x0F) << 4) | ((val & 0xF0) >> 4)
        self.reg.set_znhc(result == 0, 0, 0, 0)
        return result

    @staticmethod
    def _cb_bit(val, bit):
        return (val & (1 << bit)) == 0

    @staticmethod
    def _cb_res(val, bit):
        return val & ~(1 << bit)

    @staticmethod
    def _cb_set(val, bit):
        return val | (1 << bit)


class LinkCable:
    """TCP serial link between two emulator instances (local multiplayer).

    The partner byte is fetched when the transfer starts; local SB is then
    bit-clocked via ``MMU._serial_step`` like hardware. Sockets use a short
    timeout so a stalled partner cannot freeze the emulator indefinitely.
    """
    TRANSFER_TIMEOUT = 0.25

    def __init__(self):
        self.sock = None
        self.server_sock = None
        self._buf = bytearray()

    def start_server(self, port):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('127.0.0.1', port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(30)
        try:
            self.sock, _ = self.server_sock.accept()
            self.sock.settimeout(self.TRANSFER_TIMEOUT)
        except socket.timeout:
            self.sock = None

    def connect(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        try:
            self.sock.connect((host, port))
            self.sock.settimeout(self.TRANSFER_TIMEOUT)
        except (socket.timeout, ConnectionRefusedError, OSError):
            self.sock = None

    def transfer(self, sb_out):
        """Send one byte, receive the partner's byte, return it."""
        if self.sock is None:
            return 0xFF
        try:
            self.sock.settimeout(self.TRANSFER_TIMEOUT)
            self.sock.sendall(bytes([sb_out]))
            resp = self.sock.recv(1)
            return resp[0] if resp else 0xFF
        except (OSError, socket.timeout):
            # Drop a dead link so the game can keep running rather than hang.
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
            return 0xFF

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass
            self.server_sock = None

    @property
    def is_connected(self):
        return self.sock is not None


class MMU:
    """Memory Management Unit handling the 64KB address space with MBC1/MBC2/MBC3/MBC5."""
    def __init__(self):
        self.memory = bytearray(0x10000)
        self.rom_data = bytearray()
        self.ram_data = bytearray()
        self.mbc_type = 0x00
        self.ram_enabled = False
        self.rom_bank = 1
        self.ram_bank = 0
        self.mbc1_mode = 0
        self.mbc1_upper_bank = 0
        self.has_ram = False
        self.has_battery = False
        self.num_rom_banks = 2
        self.num_ram_banks = 0
        self.joypad_buttons = 0xFF
        # Independent input sources (keyboard / hat / stick / pad buttons / turbo).
        # Combined as AND because 0 = pressed. Stops analog-stick release
        # from eating a still-held D-pad or keyboard direction.
        self._joy_src = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        self._dpad_last = [0, 2]  # last-wins SOCD: horiz bit, vert bit
        self.div_reset_callback = None
        self.apu = None
        self.ppu = None
        self.rom_path = None
        self.serial_data = 0x00
        self.serial_control = 0x00
        self.link_cable = None
        self.vram_bank1 = bytearray(0x2000)
        self.vram_bank_select = 0
        self.is_cgb = False
        self.key1 = 0x00
        self.rp = 0x00
        self.wram_banks = [bytearray(0x1000) for _ in range(7)]
        self.svbk = 1
        self.hdma_active = False
        self.hdma_src = 0
        self.hdma_dst = 0
        self.hdma_remaining = 0
        # OAM DMA (0xFF46): 160 bytes, one per 4 CPU T-cycles. CPU keeps
        # running from HRAM while the DMA unit owns the rest of the bus.
        self.dma_remaining = 0  # remaining CPU T-cycles of the transfer
        self.dma_src = 0       # source page high byte
        self.dma_index = 0     # bytes copied so far (0..160)
        self.dma_cycle_acc = 0
        self.dma_buffer = bytearray()  # leftover for older save-state paths
        self.gdma_stall = 0
        # MBC3 RTC (real-time clock, battery-backed)
        self.has_rtc = False
        self.rtc_s = 0
        self.rtc_m = 0
        self.rtc_h = 0
        self.rtc_dl = 0
        self.rtc_dh = 0
        self.rtc_latch_s = 0
        self.rtc_latch_m = 0
        self.rtc_latch_h = 0
        self.rtc_latch_dl = 0
        self.rtc_latch_dh = 0
        self.rtc_latch_state = 0xFF
        self.rtc_last_time = time.time()
        # Boot ROM
        self.bootrom = bytearray()
        self.bootrom_enabled = False
        # CGB double-speed: extra T-cycle per cartridge bus access.
        self.cart_wait_cycles = 0
        # Serial shift register (bit-clocked; not an instant transfer).
        self.serial_bits_left = 0
        self.serial_cycle_accum = 0
        self.serial_incoming = 0xFF
        # Super Game Boy
        self.is_sgb = False
        self.sgb_in_packet = False
        self.sgb_bit_count = 0
        self.sgb_packet = bytearray(16)
        self.sgb_cmd = 0
        self.sgb_cmd_data = bytearray()
        self.sgb_packets_left = 0
        self.sgb_player_count = 1
        self.sgb_current_player = 0
        self.sgb_mask = 0
        self.sgb_pal_rgb = [0] * 16
        self.sgb_attr = bytearray(20 * 18)
        self.sgb_sys_pal = bytearray(512 * 8)
        self.sgb_atf = bytearray(45 * 90)
        self._sgb_init_default_palettes()
        # MBC6
        self.mbc6_rom_bank_a = 0
        self.mbc6_rom_bank_b = 1
        self.mbc6_ram_bank_a = 0
        self.mbc6_ram_bank_b = 0
        self.mbc6_flash_a = False
        self.mbc6_flash_b = False
        self.mbc6_flash_enable = False
        self.mbc6_flash_we = False
        self.flash_data = bytearray()
        self.flash_mode = 'ready'
        self.flash_cmd = 0
        # MBC7
        self.mbc7_ram_enable2 = False
        self.mbc7_latch_ready = False
        self.mbc7_latch_x = 0x8000
        self.mbc7_latch_y = 0x8000
        self.eeprom_pins = 0
        self.eeprom_cs = False
        self.eeprom_clk = False
        self.eeprom_do = 1
        self.eeprom_state = 0  # 0 wait-start, 1 command, 2 dummy, 3 read, 4 write
        self.eeprom_bits = 0
        self.eeprom_shift = 0
        self.eeprom_addr = 0
        self.eeprom_write_en = False

    def load_bootrom(self, bootrom_data):
        """Install a boot ROM that will shadow cartridge reads at 0x0000-N.

        Length 256 = DMG boot ROM, length ~2304 = CGB boot ROM.
        Sets bootrom_enabled = True; the boot ROM itself un-maps by writing
        to 0xFF50."""
        if not bootrom_data:
            return
        self.bootrom = bytearray(bootrom_data)
        self.bootrom_enabled = True

    def load_rom(self, rom_data):
        self.rom_data = bytearray(rom_data)
        self.mbc_type = self.rom_data[0x0147] if len(self.rom_data) > 0x0147 else 0x00

        rom_size_code = self.rom_data[0x0148] if len(self.rom_data) > 0x0148 else 0
        ram_size_code = self.rom_data[0x0149] if len(self.rom_data) > 0x0149 else 0
        rom_size_map = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128, 7: 256, 8: 512}
        ram_size_map = {0: 0, 1: 1, 2: 1, 3: 4, 4: 16, 5: 8}
        self.num_rom_banks = rom_size_map.get(rom_size_code, 2)
        self.num_ram_banks = ram_size_map.get(ram_size_code, 0)

        mbc_ram_types = {0x02, 0x03, 0x05, 0x06, 0x0F, 0x10, 0x12, 0x13, 0x1A, 0x1B, 0x1D, 0x1E, 0x20, 0x22}
        mbc_battery_types = {0x03, 0x06, 0x0F, 0x10, 0x13, 0x1B, 0x1E, 0x20, 0x22}
        self.has_ram = self.mbc_type in mbc_ram_types or self.num_ram_banks > 0
        self.has_battery = self.mbc_type in mbc_battery_types
        # MBC3 RTC is present on types 0x0F (timer+batt) and 0x10 (timer+ram+batt)
        self.has_rtc = self.mbc_type in (0x0F, 0x10)
        cgb_flag = self.rom_data[0x0143] if len(self.rom_data) > 0x0143 else 0x00
        sgb_flag = self.rom_data[0x0146] if len(self.rom_data) > 0x0146 else 0x00
        self.is_cgb = bool(cgb_flag & 0x80)
        # SGB only when the cart is not CGB-exclusive/compatible (CGB wins).
        self.is_sgb = (not self.is_cgb) and (sgb_flag == 0x03)
        if self.mbc_type in (0x05, 0x06):
            # MBC2 has 512 nibbles = 256 bytes of 4-bit RAM
            self.ram_data = bytearray(256)
        elif self.mbc_type == 0x22:
            self.ram_data = bytearray(MBC7_EEPROM_SIZE)
            self.has_ram = True
        elif self.mbc_type == 0x20:
            ram_bytes = self.num_ram_banks * 0x2000 if self.num_ram_banks else 0x8000
            self.ram_data = bytearray(ram_bytes)
            self.flash_data = bytearray(b'\xFF' * MBC6_FLASH_SIZE)
            self.has_ram = True
            self.mbc6_rom_bank_a = 0
            self.mbc6_rom_bank_b = 1
        elif self.has_ram and self.num_ram_banks > 0:
            self.ram_data = bytearray(self.num_ram_banks * 0x2000)

        self.memory[0:0x8000] = self.rom_data[0:min(0x8000, len(self.rom_data))]
        mbc_name = {0x00:"ROM ONLY", 0x01:"MBC1", 0x02:"MBC1+RAM", 0x03:"MBC1+RAM+BATT",
                    0x05:"MBC2", 0x06:"MBC2+BATT", 0x0F:"MBC3+TIMER+BATT",
                    0x10:"MBC3+TIMER+RAM+BATT", 0x11:"MBC3", 0x12:"MBC3+RAM",
                    0x13:"MBC3+RAM+BATT", 0x19:"MBC5", 0x1A:"MBC5+RAM",
                    0x1B:"MBC5+RAM+BATT", 0x1C:"MBC5+RUMBLE", 0x1D:"MBC5+RUMBLE+RAM",
                    0x1E:"MBC5+RUMBLE+RAM+BATT",
                    0x20:"MBC6", 0x22:"MBC7"}.get(self.mbc_type, f"UNKNOWN(0x{self.mbc_type:02X})")
        extra = "" if self.mbc_type in _SUPPORTED_CART_TYPES else " — mapper not emulated"
        mode = " CGB" if self.is_cgb else (" SGB" if self.is_sgb else "")
        logging.info(f"Loaded ROM: {len(rom_data)} bytes [{mbc_name}, {self.num_rom_banks} ROM banks, {self.num_ram_banks} RAM banks{mode}{extra}]")
        if extra:
            logging.warning("This cartridge type is not emulated; the game will not run correctly.")

    def read_byte(self, address):
        # During OAM DMA the CPU can only see HRAM / IE (FF80-FFFF).
        if self.dma_remaining > 0 and address < 0xFF80:
            return 0xFF
        # Boot ROM shadows cartridge ROM at 0x0000-N while enabled
        if self.bootrom_enabled and address < len(self.bootrom):
            return self.bootrom[address]
        # Fast path: WRAM (C000-DFFF) and HRAM/IE (FF80-FFFF) dominate game traffic
        if 0xC000 <= address < 0xE000:
            return self.memory[address]
        if address >= 0xFF80:
            return self.memory[address]
        # Fast path: remaining high addresses (echo RAM, OAM, I/O)
        if address >= 0xC000:
            if address < 0xFE00:
                return self.memory[address - 0x2000]  # Echo RAM
            # OAM (0xFE00-0xFE9F): blocked during PPU modes 2 and 3, or during OAM DMA
            if address < 0xFEA0:
                if self.dma_remaining > 0:
                    return 0xFF
                if self.ppu is not None and self.ppu.mode >= 2:
                    return 0xFF
            elif address < 0xFF00:
                return 0xFF  # 0xFEA0-0xFEFF is unusable; reads back as 0xFF on DMG
            if address == 0xFF00:
                return self._read_joypad()
            if address == 0xFF0F:
                # IF: only bits 0-4 are implemented; the top 3 read back as 1.
                return self.memory[0xFF0F] | 0xE0
            if address == 0xFF01:
                return self.serial_data
            if address == 0xFF02:
                return (self.serial_control & 0x83) | 0x7C
            if address == 0xFF4F:
                if not self.is_cgb:
                    return 0xFF
                return self.vram_bank_select | 0xFE
            if address == 0xFF70:
                if not self.is_cgb:
                    return 0xFF
                return self.svbk | 0xF8
            if address == 0xFF68:
                return self.ppu.bg_palette_addr if self.ppu else 0x00
            if address == 0xFF69:
                return self.ppu.bg_palette_data[self.ppu.bg_palette_addr & 0x3F] if self.ppu else 0xFF
            if address == 0xFF6A:
                return self.ppu.obj_palette_addr if self.ppu else 0x00
            if address == 0xFF6B:
                return self.ppu.obj_palette_data[self.ppu.obj_palette_addr & 0x3F] if self.ppu else 0xFF
            if address == 0xFF6C:
                return self.ppu.cgb_opri | 0xFE if self.ppu else 0xFE
            if address == 0xFF4D:
                if not self.is_cgb:
                    return 0xFF
                return self.key1 | 0x7E
            if address == 0xFF56:
                if not self.is_cgb:
                    return 0xFF
                return self.rp | 0x3C
            if 0xFF10 <= address <= 0xFF3F:
                if self.apu is not None:
                    return self.apu.read_register(address)
                return self.memory[address]
            return self.memory[address]
        # VRAM (0x8000-0x9FFF): blocked during PPU mode 3
        if 0x8000 <= address <= 0x9FFF:
            if self.ppu is not None and self.ppu.mode == 3:
                return 0xFF
            if self.vram_bank_select:
                return self.vram_bank1[address - 0x8000]
            return self.memory[address]
        # Cartridge RAM (0xA000-0xBFFF)
        if 0xA000 <= address <= 0xBFFF:
            self._add_cart_wait(address)
            if self.mbc_type == 0x22:
                return self._mbc7_read_reg(address)
            if self.mbc_type == 0x20:
                return self._mbc6_read_ram(address)
            if self.mbc_type in (0x05, 0x06):
                if not self.ram_enabled or len(self.ram_data) == 0:
                    return 0xFF
                # MBC2: lower 4 bits stored, upper 4 bits read as 1
                offset = address & 0x1FF
                return (self.ram_data[offset] & 0x0F) | 0xF0
            if self.mbc_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                if not self.ram_enabled:
                    return 0xFF
                if self.ram_bank <= 0x03:
                    if self.has_ram and len(self.ram_data) > 0:
                        offset = self.ram_bank * 0x2000 + (address - 0xA000)
                        return self.ram_data[offset] if offset < len(self.ram_data) else 0xFF
                    return 0xFF
                if 0x08 <= self.ram_bank <= 0x0C:
                    return self._rtc_read(self.ram_bank)
                return 0xFF
            if self.mbc_type in (0x01, 0x02, 0x03):
                if not self.ram_enabled or not self.has_ram or len(self.ram_data) == 0:
                    return 0xFF
                effective_bank = 0 if not self.mbc1_mode else self.ram_bank
                offset = effective_bank * 0x2000 + (address - 0xA000)
                return self.ram_data[offset] if offset < len(self.ram_data) else 0xFF
            if self.has_ram and self.ram_enabled and len(self.ram_data) > 0:
                offset = self.ram_bank * 0x2000 + (address - 0xA000)
                return self.ram_data[offset] if offset < len(self.ram_data) else 0xFF
            return 0xFF
        # ROM (0x0000-0x7FFF)
        if address < 0x8000:
            self._add_cart_wait(address)
            return self._read_rom(address)

    def _read_joypad(self):
        sel = self.memory[0xFF00] & 0x30
        if self.sgb_player_count > 1 and sel == 0x30:
            return 0xC0 | 0x30 | (0x0F - self.sgb_current_player)
        line_dir = 0x0F
        line_act = 0x0F
        buttons = self.joypad_buttons
        if self.sgb_player_count > 1 and self.sgb_current_player != 0:
            buttons = 0xFF  # only player 1 is wired to the host keyboard/gamepad
        if not (buttons & 0x01): line_dir &= ~0x01
        if not (buttons & 0x02): line_dir &= ~0x02
        if not (buttons & 0x04): line_dir &= ~0x04
        if not (buttons & 0x08): line_dir &= ~0x08
        if not (buttons & 0x10): line_act &= ~0x01
        if not (buttons & 0x20): line_act &= ~0x02
        if not (buttons & 0x40): line_act &= ~0x04
        if not (buttons & 0x80): line_act &= ~0x08
        if not (sel & 0x10) and not (sel & 0x20):
            result = line_dir & line_act
        elif not (sel & 0x10):
            result = line_dir
        elif not (sel & 0x20):
            result = line_act
        else:
            result = 0x0F
        return 0xC0 | sel | result

    def set_joypad_button(self, bit, pressed, source=_JOY_SRC_KB):
        """Update one input source and recompute the combined P1 state.

        `source` is one of _JOY_SRC_KB / _JOY_SRC_HAT / _JOY_SRC_AXIS / _JOY_SRC_BTN
        so analog-stick release cannot un-press a still-held hat or key.
        Opposite D-pad directions use last-wins SOCD cleaning (hardware cannot
        press Left+Right or Up+Down together).
        """
        mask = self._joy_src[source]
        if pressed:
            mask &= ~(1 << bit)
            if bit <= 3:
                self._dpad_last[bit >> 1] = bit
        else:
            mask |= (1 << bit)
        self._joy_src[source] = mask
        self._recompute_joypad()

    def release_all_joypad(self):
        """Clear every input source (used when opening the pause menu)."""
        self._joy_src = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        self.joypad_buttons = 0xFF

    def _recompute_joypad(self):
        src = self._joy_src
        new_state = src[0] & src[1] & src[2] & src[3]
        if len(src) > 4:
            new_state &= src[4]
        # Last-wins SOCD: if both sides of an axis are down, keep the latest.
        if (new_state & 0x03) == 0:
            new_state |= (0x02 if self._dpad_last[0] == 0 else 0x01)
        if (new_state & 0x0C) == 0:
            new_state |= (0x08 if self._dpad_last[1] == 2 else 0x04)
        newly_pressed = self.joypad_buttons & ~new_state
        if newly_pressed:
            self.memory[0xFF0F] |= IF_JOYPAD
        self.joypad_buttons = new_state

    def read_word(self, address):
        lo = self.read_byte(address)
        hi = self.read_byte((address + 1) & 0xFFFF)
        return (hi << 8) | lo

    def write_byte(self, address, value):
        value &= 0xFF
        # During OAM DMA the CPU may only write HRAM / IE.
        if self.dma_remaining > 0 and address < 0xFF80:
            return
        # Fast path: WRAM (C000-DFFF) and HRAM/IE (FF80-FFFF)
        if 0xC000 <= address < 0xE000:
            self.memory[address] = value
            return
        if address >= 0xFF80:
            self.memory[address] = value
            return
        if address < 0x8000:
            self._add_cart_wait(address)
            self._handle_mbc_write(address, value)
        elif 0xA000 <= address <= 0xBFFF:
            self._add_cart_wait(address)
            if self.mbc_type == 0x22:
                self._mbc7_write_reg(address, value)
            elif self.mbc_type == 0x20:
                self._mbc6_write_ram(address, value)
            elif self.mbc_type in (0x05, 0x06):
                if not self.ram_enabled or len(self.ram_data) == 0:
                    return
                # MBC2: only lower 4 bits of address matter (256 entries), lower 4 bits stored
                offset = address & 0x1FF
                self.ram_data[offset] = value & 0x0F
            elif self.mbc_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                if not self.ram_enabled:
                    return
                if self.ram_bank <= 0x03:
                    if self.has_ram and len(self.ram_data) > 0:
                        offset = self.ram_bank * 0x2000 + (address - 0xA000)
                        if offset < len(self.ram_data):
                            self.ram_data[offset] = value
                elif 0x08 <= self.ram_bank <= 0x0C:
                    self._rtc_write(self.ram_bank, value)
            elif self.mbc_type in (0x01, 0x02, 0x03):
                if not self.ram_enabled or not self.has_ram or len(self.ram_data) == 0:
                    return
                effective_bank = 0 if not self.mbc1_mode else self.ram_bank
                offset = effective_bank * 0x2000 + (address - 0xA000)
                if offset < len(self.ram_data):
                    self.ram_data[offset] = value
            elif self.has_ram and self.ram_enabled and len(self.ram_data) > 0:
                offset = self.ram_bank * 0x2000 + (address - 0xA000)
                if offset < len(self.ram_data):
                    self.ram_data[offset] = value
        elif 0xE000 <= address <= 0xFDFF:
            self.memory[0xC000 + (address - 0xE000)] = value
        elif address == 0xFF00:
            old = self.memory[0xFF00] & 0x30
            new = value & 0x30
            self.memory[0xFF00] = (self.memory[0xFF00] & 0x0F) | new | 0xC0
            if not self.is_cgb:
                self._sgb_write_joyp(old, new)
        elif address == 0xFF04:
            self.memory[0xFF04] = 0x00
            if self.div_reset_callback:
                self.div_reset_callback()
        elif address == 0xFF07:
            self.memory[0xFF07] = value | 0xF8
        elif address == 0xFF41:
            self.memory[0xFF41] = (value & 0x78) | (self.memory[0xFF41] & 0x07) | 0x80
        elif address == 0xFF01:
            self.serial_data = value
        elif address == 0xFF02:
            self.serial_control = value & 0x83
            if value & 0x80:
                if value & 0x01:
                    self._serial_start()
            else:
                # Bit 7 is hardware-clear-only; ignore attempts to abort.
                pass
        elif 0xFF10 <= address <= 0xFF3F:
            if self.apu is not None:
                self.apu.write_register(address, value)
            else:
                self.memory[address] = value
        elif address == 0xFF46:
            self._dma_transfer(value)
        elif address == 0xFF44:
            self.memory[0xFF44] = 0
        elif address == 0xFF50:
            # Boot ROM unmap: any write disables the boot ROM
            self.bootrom_enabled = False
        elif address == 0xFF4D:
            self.key1 = (self.key1 & 0x80) | (value & 0x01)
        elif address == 0xFF56:
            self.rp = value & 0xC1
        elif 0xFF68 <= address <= 0xFF6C:
            if self.ppu is not None:
                self.ppu.write_cgb_register(address, value)
            self.memory[address] = value
        elif address == 0xFF55:
            if value & 0x80:
                # H-Blank DMA request
                if self.hdma_active:
                    # Cancel active HDMA
                    self.hdma_active = False
                    remaining_blocks = (self.hdma_remaining + 15) // 16
                    self.memory[0xFF55] = 0x80 | (remaining_blocks & 0x7F)
                else:
                    # Start new H-Blank DMA
                    self.hdma_src = ((self.memory[0xFF51] << 8) | self.memory[0xFF52]) & 0xFFF0
                    self.hdma_dst = (((self.memory[0xFF53] << 8) | self.memory[0xFF54]) & 0x1FF0) | 0x8000
                    self.hdma_remaining = ((value & 0x7F) + 1) * 16
                    self.hdma_active = True
                    self.memory[0xFF55] = value & 0x7F
            else:
                # General Purpose DMA (GDMA) - immediate bulk transfer
                self._hdma_transfer()
                self.hdma_active = False
        elif address == 0xFF4F:
            self.vram_bank_select = value & 0x01
        elif address == 0xFF70 and self.is_cgb:
            new_bank = value & 0x07
            if new_bank == 0:
                new_bank = 1
            if new_bank != self.svbk:
                self.wram_banks[self.svbk - 1][:] = self.memory[0xD000:0xE000]
                self.memory[0xD000:0xE000] = self.wram_banks[new_bank - 1]
                self.svbk = new_bank
        elif 0x8000 <= address <= 0x9FFF:
            if self.ppu is not None and self.ppu.mode == 3:
                return
            if self.vram_bank_select:
                self.vram_bank1[address - 0x8000] = value
            else:
                self.memory[address] = value
        else:
            # OAM (0xFE00-0xFE9F): blocked during PPU modes 2/3 or during OAM DMA
            if 0xFE00 <= address <= 0xFE9F:
                if self.dma_remaining > 0:
                    return
                if self.ppu is not None and self.ppu.mode >= 2:
                    return
            self.memory[address] = value

    def write_word(self, address, value):
        self.write_byte(address, value & 0xFF)
        self.write_byte((address + 1) & 0xFFFF, (value >> 8) & 0xFF)

    def _serial_start(self):
        """Begin an internal-clock 8-bit transfer; incoming bits default to 1."""
        self.serial_bits_left = 8
        self.serial_cycle_accum = 0
        if self.link_cable is not None:
            self.serial_incoming = self.link_cable.transfer(self.serial_data)
        else:
            self.serial_incoming = 0xFF

    def _serial_bit_period(self):
        if self.is_cgb and (self.serial_control & 0x02):
            return SERIAL_BIT_CYCLES_FAST
        return SERIAL_BIT_CYCLES_NORMAL

    def _serial_step(self, cpu_cycles):
        """Shift one bit every 512 (or 16, CGB fast) CPU T-cycles."""
        if self.serial_bits_left <= 0 or not (self.serial_control & 0x81) == 0x81:
            return
        period = self._serial_bit_period()
        self.serial_cycle_accum += cpu_cycles
        while self.serial_bits_left > 0 and self.serial_cycle_accum >= period:
            self.serial_cycle_accum -= period
            in_bit = (self.serial_incoming >> 7) & 1
            self.serial_incoming = ((self.serial_incoming << 1) | 1) & 0xFF
            self.serial_data = ((self.serial_data << 1) | in_bit) & 0xFF
            self.serial_bits_left -= 1
        if self.serial_bits_left <= 0:
            self.serial_control &= 0x7F
            self.serial_bits_left = 0
            self.memory[0xFF0F] |= IF_SERIAL

    def _add_cart_wait(self, address):
        """One extra T-cycle per cartridge access in CGB double-speed."""
        if not (self.key1 & 0x80):
            return
        if self.bootrom_enabled and address < len(self.bootrom):
            return
        if address < 0x8000 or 0xA000 <= address <= 0xBFFF:
            self.cart_wait_cycles += 1

    def _hdma_transfer(self):
        src = ((self.memory[0xFF51] << 8) | self.memory[0xFF52]) & 0xFFF0
        dst = (((self.memory[0xFF53] << 8) | self.memory[0xFF54]) & 0x1FF0) | 0x8000
        blocks = (self.memory[0xFF55] & 0x7F) + 1
        length = blocks * 16
        saved_wait = self.cart_wait_cycles
        for i in range(length):
            val = self.read_byte(src + i)
            addr = 0x8000 + ((dst + i) & 0x1FFF)
            if self.vram_bank_select:
                self.vram_bank1[addr - 0x8000] = val
            else:
                self.memory[addr] = val
        self.cart_wait_cycles = saved_wait
        self.memory[0xFF55] = 0xFF
        self.gdma_stall = (blocks * 2 + 1) * 4

    def _hdma_hblank_step(self):
        if not self.hdma_active:
            return
        chunk = min(16, self.hdma_remaining)
        saved_wait = self.cart_wait_cycles
        for i in range(chunk):
            val = self.read_byte(self.hdma_src + i)
            addr = 0x8000 + ((self.hdma_dst + i) & 0x1FFF)
            if self.vram_bank_select:
                self.vram_bank1[addr - 0x8000] = val
            else:
                self.memory[addr] = val
        self.cart_wait_cycles = saved_wait
        self.hdma_src += 16
        self.hdma_dst += 16
        self.hdma_remaining -= chunk
        if self.hdma_remaining <= 0:
            self.hdma_active = False
            self.memory[0xFF55] = 0xFF
        else:
            remaining_blocks = (self.hdma_remaining + 15) // 16
            self.memory[0xFF55] = remaining_blocks & 0x7F

    def _dma_transfer(self, value):
        """Start OAM DMA: 160 bytes from page `value`, one byte per 4 T-cycles."""
        self.dma_src = value & 0xFF
        self.dma_index = 0
        self.dma_cycle_acc = 0
        self.dma_remaining = OAM_DMA_CYCLES

    def _dma_read_source(self, address):
        """Bus-master read used by the OAM DMA unit (not subject to the CPU lock)."""
        address &= 0xFFFF
        if address < 0x8000:
            if self.bootrom_enabled and address < len(self.bootrom):
                return self.bootrom[address]
            return self._read_rom(address)
        if address < 0xA000:
            if self.vram_bank_select:
                return self.vram_bank1[address - 0x8000]
            return self.memory[address]
        if address < 0xC000:
            saved, self.dma_remaining = self.dma_remaining, 0
            try:
                return self.read_byte(address)
            finally:
                self.dma_remaining = saved
        if address < 0xE000:
            return self.memory[address]
        if address < 0xFE00:
            return self.memory[address - 0x2000]
        return 0xFF

    def _dma_advance(self, cpu_cycles):
        """Copy one OAM byte every 4 CPU T-cycles."""
        remaining = self.dma_remaining
        if remaining <= 0 or cpu_cycles <= 0:
            return
        self.dma_cycle_acc += cpu_cycles
        src_base = self.dma_src << 8
        mem = self.memory
        while self.dma_cycle_acc >= 4 and self.dma_index < 160:
            self.dma_cycle_acc -= 4
            mem[0xFE00 + self.dma_index] = self._dma_read_source(src_base + self.dma_index)
            self.dma_index += 1
            remaining -= 4
        if self.dma_index >= 160:
            remaining = 0
            self.dma_cycle_acc = 0
        self.dma_remaining = remaining if remaining > 0 else 0

    def _read_rom(self, address):
        """Read a cartridge ROM byte, honouring the current mapper bank."""
        address &= 0xFFFF
        rom = self.rom_data
        n = len(rom)
        if address < 0x4000:
            if self.mbc_type in (0x01, 0x02, 0x03) and self.mbc1_mode and self.mbc1_upper_bank:
                banks = self.num_rom_banks if self.num_rom_banks else 1
                bank = (self.mbc1_upper_bank << 5) % banks
                offset = bank * 0x4000 + address
            else:
                offset = address
            return rom[offset] if offset < n else 0xFF
        if self.mbc_type == 0x20:
            return self._mbc6_read_window(address)
        if self.mbc_type == 0x00:
            return rom[address] if address < n else 0xFF
        banks = self.num_rom_banks if self.num_rom_banks else 1
        bank = self.rom_bank % banks
        offset = bank * 0x4000 + (address - 0x4000)
        return rom[offset] if offset < n else 0xFF

    def _remap_rom_bank(self):
        if self.mbc_type == 0x20:
            for i in range(0x4000):
                self.memory[0x4000 + i] = self._mbc6_read_window(0x4000 + i)
            return
        banks = self.num_rom_banks if self.num_rom_banks else 1
        bank = self.rom_bank % banks
        src_offset = bank * 0x4000
        end = min(src_offset + 0x4000, len(self.rom_data))
        length = end - src_offset
        self.memory[0x4000:0x4000 + length] = self.rom_data[src_offset:end]
        if length < 0x4000:
            for i in range(length, 0x4000):
                self.memory[0x4000 + i] = 0xFF

    def _handle_mbc_write(self, address, value):
        mbc = self.mbc_type
        if mbc == 0x20:
            self._mbc6_write_control(address, value)
            return
        if mbc == 0x22:
            self._mbc7_write_control(address, value)
            return
        if mbc in (0x05, 0x06):
            if address & 0x0100:
                bank = value & 0x0F
                if bank == 0:
                    bank = 1
                self.rom_bank = bank
                self._remap_rom_bank()
            else:
                self.ram_enabled = (value & 0x0F) == 0x0A
            return
        if 0x2000 <= address <= 0x2FFF:
            if mbc in (0x01, 0x02, 0x03):
                bank = value & 0x1F
                if bank == 0:
                    bank = 1
                self.rom_bank = (self.rom_bank & 0x60) | bank
                self._remap_rom_bank()
            elif mbc in (0x0F, 0x10, 0x11, 0x12, 0x13):
                bank = value & 0x7F
                if bank == 0:
                    bank = 1
                self.rom_bank = bank
                self._remap_rom_bank()
            elif mbc in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                self.rom_bank = (self.rom_bank & 0x100) | value
                self._remap_rom_bank()
        elif 0x3000 <= address <= 0x3FFF:
            if mbc in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                self.rom_bank = (self.rom_bank & 0xFF) | ((value & 0x01) << 8)
                self._remap_rom_bank()
            elif mbc in (0x01, 0x02, 0x03):
                pass
        elif 0x4000 <= address <= 0x5FFF and mbc in (0x01, 0x02, 0x03):
            self.mbc1_upper_bank = value & 0x03
            if self.mbc1_mode:
                self.ram_bank = self.mbc1_upper_bank
            else:
                self.rom_bank = (self.rom_bank & 0x1F) | (self.mbc1_upper_bank << 5)
                self._remap_rom_bank()
        elif 0x4000 <= address <= 0x5FFF and mbc in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
            self.ram_bank = value & 0x0F
        elif 0x4000 <= address <= 0x5FFF and mbc in (0x0F, 0x10, 0x11, 0x12, 0x13):
            # MBC3: 0x00-0x03 -> RAM bank 0-3, 0x08-0x0C -> RTC register
            v = value & 0x0F
            if v <= 0x03 or (0x08 <= v <= 0x0C):
                self.ram_bank = v
        elif 0x6000 <= address <= 0x7FFF and mbc in (0x01, 0x02, 0x03):
            mode = value & 0x01
            if mode != self.mbc1_mode:
                self.mbc1_mode = mode
                if mode:
                    self.ram_bank = self.mbc1_upper_bank
                    self.rom_bank = self.rom_bank & 0x1F
                else:
                    self.rom_bank = (self.rom_bank & 0x1F) | (self.mbc1_upper_bank << 5)
                    self.ram_bank = 0
                self._remap_rom_bank()
        elif 0x6000 <= address <= 0x7FFF and mbc in (0x0F, 0x10, 0x11, 0x12, 0x13):
            # MBC3 latch: write 0x00 then 0x01 to copy RTC -> latched registers
            if self.rtc_latch_state == 0x00 and value == 0x01:
                self._rtc_latch()
            self.rtc_latch_state = value
        elif 0x0000 <= address <= 0x1FFF:
            self.ram_enabled = (value & 0x0F) == 0x0A

    # ===== MBC3 RTC =====
    def _rtc_update(self):
        """Advance MBC3 RTC state based on wall-clock time since last update."""
        if not self.has_rtc:
            return
        if self.rtc_dh & 0x40:  # halted
            self.rtc_last_time = time.time()
            return
        now = time.time()
        delta = int(now - self.rtc_last_time)
        if delta <= 0:
            return
        self.rtc_last_time = now
        # Bulk-advance: seconds -> minutes -> hours -> days -> day-carry
        # This replaces the old per-second loop which was capped at 86400.
        self.rtc_s += delta
        carry = self.rtc_s // 60
        self.rtc_s %= 60
        self.rtc_m += carry
        carry = self.rtc_m // 60
        self.rtc_m %= 60
        self.rtc_h += carry
        carry = self.rtc_h // 24
        self.rtc_h %= 24
        total_days = self.rtc_dl | ((self.rtc_dh & 0x01) << 8)
        total_days += carry
        if total_days > 511:
            total_days %= 512
            self.rtc_dh |= 0x80  # day carry bit
        self.rtc_dl = total_days & 0xFF
        self.rtc_dh = (self.rtc_dh & 0xC0) | ((total_days >> 8) & 0x01)

    def _rtc_latch(self):
        """Snapshot current RTC state into the latched registers."""
        self._rtc_update()
        self.rtc_latch_s = self.rtc_s
        self.rtc_latch_m = self.rtc_m
        self.rtc_latch_h = self.rtc_h
        self.rtc_latch_dl = self.rtc_dl
        self.rtc_latch_dh = self.rtc_dh

    def _rtc_read(self, reg):
        """Return the latched value of the given RTC register (0x08-0x0C)."""
        return {
            0x08: self.rtc_latch_s,
            0x09: self.rtc_latch_m,
            0x0A: self.rtc_latch_h,
            0x0B: self.rtc_latch_dl,
            0x0C: self.rtc_latch_dh,
        }[reg]

    def _rtc_write(self, reg, value):
        """Write a value to the current (not latched) RTC register."""
        if reg == 0x08:
            self.rtc_s = value & 0x3F
        elif reg == 0x09:
            self.rtc_m = value & 0x3F
        elif reg == 0x0A:
            self.rtc_h = value & 0x1F
        elif reg == 0x0B:
            self.rtc_dl = value
        elif reg == 0x0C:
            # bit 0 = day high, bit 6 = halt, bit 7 = overflow
            self.rtc_dh = value & 0xC1

    def pack_rtc_blob(self):
        """VBA-M 44-byte RTC trailer: 10× int32 registers + unix timestamp."""
        self._rtc_update()
        days = self.rtc_dl | ((self.rtc_dh & 0x01) << 8)
        latched_days = self.rtc_latch_dl | ((self.rtc_latch_dh & 0x01) << 8)
        return struct.pack(
            '<11i',
            self.rtc_s & 0x3F,
            self.rtc_m & 0x3F,
            self.rtc_h & 0x1F,
            days,
            self.rtc_dh & 0xC1,
            self.rtc_latch_s & 0x3F,
            self.rtc_latch_m & 0x3F,
            self.rtc_latch_h & 0x1F,
            latched_days,
            self.rtc_latch_dh & 0xC1,
            int(self.rtc_last_time) & 0x7FFFFFFF,
        )

    def unpack_rtc_blob(self, blob):
        """Load RTC from a VBA-M trailer or the previous 48-byte custom format."""
        if not blob:
            self.rtc_last_time = time.time()
            return
        # VBA-M: minutes stored as a 32-bit int, so bytes 5-7 are zero.
        if len(blob) >= 44 and blob[5] == 0 and blob[6] == 0 and blob[7] == 0:
            (s, m, h, days, ctrl, ls, lm, lh, ld, lctrl, ts) = struct.unpack_from('<11i', blob, 0)
            self.rtc_s = s & 0x3F
            self.rtc_m = m & 0x3F
            self.rtc_h = h & 0x1F
            self.rtc_dl = days & 0xFF
            self.rtc_dh = (ctrl & 0xC0) | ((days >> 8) & 0x01)
            self.rtc_latch_s = ls & 0x3F
            self.rtc_latch_m = lm & 0x3F
            self.rtc_latch_h = lh & 0x1F
            self.rtc_latch_dl = ld & 0xFF
            self.rtc_latch_dh = (lctrl & 0xC0) | ((ld >> 8) & 0x01)
            self.rtc_last_time = float(ts) if ts > 0 else time.time()
            self._rtc_update()
            return
        # Legacy: 4 zero bytes + 5 RTC bytes.
        if len(blob) >= 9:
            self.rtc_s = blob[4] & 0x3F
            self.rtc_m = blob[5] & 0x3F
            self.rtc_h = blob[6] & 0x1F
            self.rtc_dl = blob[7]
            self.rtc_dh = blob[8] & 0xC1
        self.rtc_last_time = time.time()

    # ===== CGB cart wait / SGB / MBC6 / MBC7 =====
    def _sgb_init_default_palettes(self):
        """Four palettes matching the default DMG greens until a PAL command."""
        dmg = PALETTE_DMG
        for pal in range(4):
            for c, (r, g, b) in enumerate(dmg):
                self.sgb_pal_rgb[pal * 4 + c] = (r << 16) | (g << 8) | b
        self.sgb_attr[:] = b'\x00' * (20 * 18)

    @staticmethod
    def _rgb555_to_packed(color):
        r5 = color & 0x1F
        g5 = (color >> 5) & 0x1F
        b5 = (color >> 10) & 0x1F
        r = (r5 << 3) | (r5 >> 2)
        g = (g5 << 3) | (g5 >> 2)
        b = (b5 << 3) | (b5 >> 2)
        return (r << 16) | (g << 8) | b

    def _sgb_set_color(self, pal, idx, color555, share_zero=False):
        packed = self._rgb555_to_packed(color555)
        self.sgb_pal_rgb[(pal & 3) * 4 + (idx & 3)] = packed
        if share_zero or idx == 0:
            for p in range(4):
                self.sgb_pal_rgb[p * 4] = packed

    def _sgb_write_joyp(self, old, new):
        if self.sgb_player_count > 1 and (old & 0x20) == 0 and (new & 0x20):
            self.sgb_current_player = (self.sgb_current_player + 1) % self.sgb_player_count
        if new == 0x00:
            self.sgb_in_packet = True
            self.sgb_bit_count = 0
            self.sgb_packet = bytearray(16)
            return
        if not self.sgb_in_packet:
            return
        if old == 0x30 and new in (0x10, 0x20):
            bit = 1 if new == 0x20 else 0
            if self.sgb_bit_count < 128:
                byte_i = self.sgb_bit_count >> 3
                self.sgb_packet[byte_i] |= (bit << (self.sgb_bit_count & 7))
                self.sgb_bit_count += 1
                if self.sgb_bit_count == 128:
                    self.sgb_in_packet = False
                    self._sgb_finish_packet()

    def _sgb_finish_packet(self):
        pkt = self.sgb_packet
        if self.sgb_packets_left <= 0:
            length = pkt[0] & 7
            if length == 0:
                return
            self.sgb_cmd = pkt[0] >> 3
            self.sgb_cmd_data = bytearray(pkt)
            self.sgb_packets_left = length - 1
        else:
            self.sgb_cmd_data.extend(pkt)
            self.sgb_packets_left -= 1
        if self.sgb_packets_left <= 0:
            self._sgb_exec(self.sgb_cmd, self.sgb_cmd_data)
            if not self.is_cgb:
                self.is_sgb = True
                if self.ppu is not None:
                    self.ppu.is_sgb = True

    def _sgb_exec(self, cmd, data):
        if cmd == 0x00:
            self._sgb_pal_pair(data, 0, 1)
        elif cmd == 0x01:
            self._sgb_pal_pair(data, 2, 3)
        elif cmd == 0x02:
            self._sgb_pal_pair(data, 0, 3)
        elif cmd == 0x03:
            self._sgb_pal_pair(data, 1, 2)
        elif cmd == 0x04:
            self._sgb_attr_blk(data)
        elif cmd == 0x05:
            self._sgb_attr_lin(data)
        elif cmd == 0x06:
            self._sgb_attr_div(data)
        elif cmd == 0x07:
            self._sgb_attr_chr(data)
        elif cmd == 0x0A:
            self._sgb_pal_set(data)
        elif cmd == 0x0B:
            n = min(len(self.sgb_sys_pal), 0x1000)
            self.sgb_sys_pal[:n] = self.memory[0x8000:0x8000 + n]
        elif cmd == 0x11:
            ctrl = data[1] if len(data) > 1 else 0
            if ctrl == 0:
                self.sgb_player_count = 1
            elif ctrl == 1:
                self.sgb_player_count = 2
            else:
                self.sgb_player_count = 4
            self.sgb_current_player = 0
        elif cmd == 0x15:
            n = min(len(self.sgb_atf), 4050)
            self.sgb_atf[:n] = self.memory[0x8000:0x8000 + n]
        elif cmd == 0x16:
            self._sgb_attr_set(data[1] if len(data) > 1 else 0)
        elif cmd == 0x17:
            self.sgb_mask = data[1] & 3 if len(data) > 1 else 0

    def _sgb_pal_pair(self, data, pal_a, pal_b):
        if len(data) < 15:
            return
        def col(off):
            return data[off] | (data[off + 1] << 8)
        c0 = col(1)
        self._sgb_set_color(pal_a, 0, c0, share_zero=True)
        self._sgb_set_color(pal_a, 1, col(3))
        self._sgb_set_color(pal_a, 2, col(5))
        self._sgb_set_color(pal_a, 3, col(7))
        self._sgb_set_color(pal_b, 1, col(9))
        self._sgb_set_color(pal_b, 2, col(11))
        self._sgb_set_color(pal_b, 3, col(13))

    def _sgb_pal_set(self, data):
        if len(data) < 10:
            return
        for i in range(4):
            pid = data[1 + i * 2] | (data[2 + i * 2] << 8)
            pid &= 0x1FF
            off = pid * 8
            for c in range(4):
                color = self.sgb_sys_pal[off + c * 2] | (self.sgb_sys_pal[off + c * 2 + 1] << 8)
                self.sgb_pal_rgb[i * 4 + c] = self._rgb555_to_packed(color)
        flags = data[9]
        if flags & 0x40:
            self.sgb_mask = 0
        if flags & 0x80:
            self._sgb_apply_atf(flags & 0x3F)

    def _sgb_apply_atf(self, n):
        if n > 0x2C:
            return
        src = self.sgb_atf[n * 90:(n + 1) * 90]
        if len(src) < 90:
            return
        i = 0
        for y in range(18):
            for b in range(5):
                byte = src[i]
                i += 1
                for t in range(4):
                    self.sgb_attr[y * 20 + b * 4 + t] = (byte >> 6) & 3
                    byte = (byte << 2) & 0xFF

    def _sgb_attr_set(self, value):
        self._sgb_apply_atf(value & 0x3F)
        if value & 0x40:
            self.sgb_mask = 0

    def _sgb_attr_blk(self, data):
        nsets = data[1] if len(data) > 1 else 0
        off = 2
        for _ in range(nsets):
            if off + 6 > len(data):
                break
            ctrl, pals, x1, y1, x2, y2 = data[off:off + 6]
            off += 6
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            inside = pals & 3
            around = (pals >> 2) & 3
            outside = (pals >> 4) & 3
            do_in = bool(ctrl & 1)
            do_line = bool(ctrl & 2)
            do_out = bool(ctrl & 4)
            if do_in and not do_line and not do_out:
                do_line = True
                around = inside
            if do_out and not do_line and not do_in:
                do_line = True
                around = outside
            for y in range(18):
                for x in range(20):
                    in_box = x1 <= x <= x2 and y1 <= y <= y2
                    on_border = in_box and (x == x1 or x == x2 or y == y1 or y == y2)
                    interior = in_box and not on_border
                    if interior and do_in:
                        self.sgb_attr[y * 20 + x] = inside
                    elif on_border and do_line:
                        self.sgb_attr[y * 20 + x] = around
                    elif not in_box and do_out:
                        self.sgb_attr[y * 20 + x] = outside

    def _sgb_attr_lin(self, data):
        nsets = data[1] if len(data) > 1 else 0
        for i in range(nsets):
            if 2 + i >= len(data):
                break
            d = data[2 + i]
            pal = (d >> 5) & 3
            line = d & 0x1F
            if d & 0x80:
                if line < 18:
                    for x in range(20):
                        self.sgb_attr[line * 20 + x] = pal
            else:
                if line < 20:
                    for y in range(18):
                        self.sgb_attr[y * 20 + line] = pal

    def _sgb_attr_div(self, data):
        if len(data) < 3:
            return
        pals = data[1]
        coord = data[2]
        below = pals & 3
        above = (pals >> 2) & 3
        line = (pals >> 4) & 3
        if pals & 0x40:
            for y in range(18):
                pal = above if y < coord else (line if y == coord else below)
                for x in range(20):
                    self.sgb_attr[y * 20 + x] = pal
        else:
            for x in range(20):
                pal = above if x < coord else (line if x == coord else below)
                for y in range(18):
                    self.sgb_attr[y * 20 + x] = pal

    def _sgb_attr_chr(self, data):
        if len(data) < 6:
            return
        x = data[1]
        y = data[2]
        count = data[3] | (data[4] << 8)
        vertical = data[5] & 1
        bits = data[6:]
        bi = 0
        for n in range(count):
            byte = bits[n >> 2] if (n >> 2) < len(bits) else 0
            shift = 6 - ((n & 3) << 1)
            pal = (byte >> shift) & 3
            if 0 <= x < 20 and 0 <= y < 18:
                self.sgb_attr[y * 20 + x] = pal
            if vertical:
                y += 1
                if y >= 18:
                    y = 0
                    x += 1
            else:
                x += 1
                if x >= 20:
                    x = 0
                    y += 1

    def _mbc6_read_window(self, address):
        if address < 0x4000:
            return self.rom_data[address] if address < len(self.rom_data) else 0xFF
        if address < 0x6000:
            bank, flash, offset = self.mbc6_rom_bank_a, self.mbc6_flash_a, address - 0x4000
        else:
            bank, flash, offset = self.mbc6_rom_bank_b, self.mbc6_flash_b, address - 0x6000
        if flash and self.mbc6_flash_enable:
            linear = (bank & 0x7F) * 0x2000 + offset
            if self.flash_mode == 'id':
                return 0xC2 if (offset & 1) == 0 else 0x81
            if linear < len(self.flash_data):
                return self.flash_data[linear]
            return 0xFF
        n8 = max(1, self.num_rom_banks * 2)
        linear = (bank % n8) * 0x2000 + offset
        return self.rom_data[linear] if linear < len(self.rom_data) else 0xFF

    def _mbc6_read_ram(self, address):
        if not self.ram_enabled or not self.ram_data:
            return 0xFF
        if address < 0xB000:
            idx = (self.mbc6_ram_bank_a & 7) * 0x1000 + (address - 0xA000)
        else:
            idx = (self.mbc6_ram_bank_b & 7) * 0x1000 + (address - 0xB000)
        return self.ram_data[idx] if idx < len(self.ram_data) else 0xFF

    def _mbc6_write_ram(self, address, value):
        if not self.ram_enabled or not self.ram_data:
            return
        if address < 0xB000:
            idx = (self.mbc6_ram_bank_a & 7) * 0x1000 + (address - 0xA000)
        else:
            idx = (self.mbc6_ram_bank_b & 7) * 0x1000 + (address - 0xB000)
        if idx < len(self.ram_data):
            self.ram_data[idx] = value

    def _mbc6_write_control(self, address, value):
        if address <= 0x03FF:
            self.ram_enabled = (value & 0x0F) == 0x0A
        elif address <= 0x07FF:
            self.mbc6_ram_bank_a = value & 0x07
        elif address <= 0x0BFF:
            self.mbc6_ram_bank_b = value & 0x07
        elif address <= 0x0FFF:
            self.mbc6_flash_enable = bool(value & 1)
            self._remap_rom_bank()
        elif address == 0x1000:
            self.mbc6_flash_we = bool(value & 1)
        elif address <= 0x27FF:
            self.mbc6_rom_bank_a = value & 0x7F
            self._remap_rom_bank()
        elif address <= 0x2FFF:
            self.mbc6_flash_a = (value & 0x08) != 0
            self._remap_rom_bank()
        elif address <= 0x37FF:
            self.mbc6_rom_bank_b = value & 0x7F
            self._remap_rom_bank()
        elif address <= 0x3FFF:
            self.mbc6_flash_b = (value & 0x08) != 0
            self._remap_rom_bank()
        elif 0x4000 <= address <= 0x7FFF:
            self._mbc6_flash_write(address, value)

    def _mbc6_flash_write(self, address, value):
        if not self.mbc6_flash_enable:
            return
        if address < 0x6000:
            if not self.mbc6_flash_a:
                return
            bank = self.mbc6_rom_bank_a
            offset = address - 0x4000
        else:
            if not self.mbc6_flash_b:
                return
            bank = self.mbc6_rom_bank_b
            offset = address - 0x6000
        linear = (bank & 0x7F) * 0x2000 + offset
        off = offset & 0x1FFF
        if self.flash_mode == 'program':
            if value == 0xF0:
                self.flash_mode = 'ready'
                return
            if linear < len(self.flash_data):
                self.flash_data[linear] &= value
                self.memory[address] = self.flash_data[linear]
            return
        if value == 0xF0:
            self.flash_mode = 'ready'
            self.flash_cmd = 0
            return
        if self.flash_cmd == 0 and off == 0x1555 and value == 0xAA:
            self.flash_cmd = 1
        elif self.flash_cmd == 1 and off == 0x0AAA and value == 0x55:
            self.flash_cmd = 2
        elif self.flash_cmd == 2 and off == 0x1555:
            if value == 0x90:
                self.flash_mode = 'id'
            elif value == 0xA0:
                self.flash_mode = 'program'
            elif value == 0x80:
                self.flash_cmd = 3
                return
            self.flash_cmd = 0
        elif self.flash_cmd == 3 and off == 0x1555 and value == 0xAA:
            self.flash_cmd = 4
        elif self.flash_cmd == 4 and off == 0x0AAA and value == 0x55:
            self.flash_cmd = 5
        elif self.flash_cmd == 5:
            if value == 0x10:
                if self.mbc6_flash_we:
                    self.flash_data[:] = b'\xFF' * len(self.flash_data)
                elif len(self.flash_data) > 0x20000:
                    self.flash_data[0x20000:] = b'\xFF' * (len(self.flash_data) - 0x20000)
            elif value == 0x30:
                sector = (linear // 0x20000) * 0x20000
                if sector != 0 or self.mbc6_flash_we:
                    end = min(sector + 0x20000, len(self.flash_data))
                    if sector < len(self.flash_data):
                        self.flash_data[sector:end] = b'\xFF' * (end - sector)
            self.flash_cmd = 0
            self.flash_mode = 'ready'
        else:
            self.flash_cmd = 0

    def _mbc7_write_control(self, address, value):
        if address <= 0x1FFF:
            self.ram_enabled = (value & 0x0F) == 0x0A
        elif address <= 0x3FFF:
            banks = self.num_rom_banks if self.num_rom_banks else 1
            self.rom_bank = value % banks
            self._remap_rom_bank()
        elif address <= 0x5FFF:
            self.mbc7_ram_enable2 = (value == 0x40)

    def _mbc7_regs_enabled(self):
        return self.ram_enabled and self.mbc7_ram_enable2

    def _mbc7_read_reg(self, address):
        if not self._mbc7_regs_enabled():
            return 0xFF
        reg = (address >> 4) & 0x0F
        if reg <= 1:
            return 0xFF
        if reg == 2:
            return self.mbc7_latch_x & 0xFF
        if reg == 3:
            return (self.mbc7_latch_x >> 8) & 0xFF
        if reg == 4:
            return self.mbc7_latch_y & 0xFF
        if reg == 5:
            return (self.mbc7_latch_y >> 8) & 0xFF
        if reg == 6:
            return 0x00
        if reg == 8:
            return (self.eeprom_pins & 0xC2) | (self.eeprom_do & 1)
        return 0xFF

    def _mbc7_write_reg(self, address, value):
        if not self._mbc7_regs_enabled():
            return
        reg = (address >> 4) & 0x0F
        if reg == 0 and value == 0x55:
            self.mbc7_latch_x = 0x8000
            self.mbc7_latch_y = 0x8000
            self.mbc7_latch_ready = True
        elif reg == 1 and value == 0xAA and self.mbc7_latch_ready:
            self.mbc7_latch_x, self.mbc7_latch_y = self._mbc7_sample_accel()
            self.mbc7_latch_ready = False
        elif reg == 8:
            self._eeprom_write(value)

    def _mbc7_sample_accel(self):
        x = y = MBC7_ACCEL_CENTER
        b = self.joypad_buttons
        if not (b & 0x01):  # Right → lower X
            x -= MBC7_ACCEL_G
        if not (b & 0x02):  # Left → higher X
            x += MBC7_ACCEL_G
        if not (b & 0x04):  # Up → higher Y
            y += MBC7_ACCEL_G
        if not (b & 0x08):  # Down → lower Y
            y -= MBC7_ACCEL_G
        return x & 0xFFFF, y & 0xFFFF

    def _eeprom_write(self, value):
        cs = bool(value & 0x80)
        clk = bool(value & 0x40)
        di = 1 if (value & 0x02) else 0
        self.eeprom_pins = value & 0xC2
        if not cs:
            self.eeprom_cs = False
            self.eeprom_clk = clk
            self.eeprom_do = 1
            self.eeprom_state = 0
            self.eeprom_bits = 0
            self.eeprom_shift = 0
            return
        rising = clk and not self.eeprom_clk
        self.eeprom_cs = True
        self.eeprom_clk = clk
        if not rising:
            return
        if self.eeprom_state == 0:
            if di:
                self.eeprom_state = 1
                self.eeprom_bits = 0
                self.eeprom_shift = 0
            return
        if self.eeprom_state == 1:
            self.eeprom_shift = ((self.eeprom_shift << 1) | di) & 0x3FF
            self.eeprom_bits += 1
            if self.eeprom_bits == 10:
                self._eeprom_decode(self.eeprom_shift)
            return
        if self.eeprom_state == 2:  # dummy 0 then 16 read bits
            self.eeprom_do = 0
            self.eeprom_state = 3
            self.eeprom_bits = 0
            return
        if self.eeprom_state == 3:
            word = (self.ram_data[self.eeprom_addr * 2] << 8) | self.ram_data[self.eeprom_addr * 2 + 1]
            self.eeprom_do = (word >> (15 - self.eeprom_bits)) & 1
            self.eeprom_bits += 1
            if self.eeprom_bits >= 16:
                self.eeprom_addr = (self.eeprom_addr + 1) & 0x7F
                self.eeprom_bits = 0
            return
        if self.eeprom_state == 4:
            self.eeprom_shift = ((self.eeprom_shift << 1) | di) & 0xFFFF
            self.eeprom_bits += 1
            if self.eeprom_bits == 16:
                if self.eeprom_write_en and self.eeprom_addr < 128:
                    self.ram_data[self.eeprom_addr * 2] = (self.eeprom_shift >> 8) & 0xFF
                    self.ram_data[self.eeprom_addr * 2 + 1] = self.eeprom_shift & 0xFF
                self.eeprom_do = 1
                self.eeprom_state = 0

    def _eeprom_decode(self, cmd):
        top2 = (cmd >> 8) & 3
        addr = cmd & 0x7F
        if top2 == 0:
            top4 = (cmd >> 6) & 0x0F
            if top4 == 0x00:
                self.eeprom_write_en = False
                self.eeprom_state = 0
            elif top4 == 0x01:
                self.eeprom_state = 4
                self.eeprom_bits = 0
                self.eeprom_shift = 0
                self.eeprom_addr = 0
            elif top4 == 0x02:
                if self.eeprom_write_en:
                    self.ram_data[:] = b'\xFF' * len(self.ram_data)
                self.eeprom_do = 1
                self.eeprom_state = 0
            elif top4 == 0x03:
                self.eeprom_write_en = True
                self.eeprom_state = 0
            else:
                self.eeprom_state = 0
        elif top2 == 1:
            self.eeprom_addr = addr
            self.eeprom_state = 4
            self.eeprom_bits = 0
            self.eeprom_shift = 0
        elif top2 == 2:
            self.eeprom_addr = addr
            self.eeprom_state = 2
            self.eeprom_bits = 0
        else:
            if self.eeprom_write_en and addr < 128:
                self.ram_data[addr * 2] = 0xFF
                self.ram_data[addr * 2 + 1] = 0xFF
            self.eeprom_do = 1
            self.eeprom_state = 0


class PPU:
    """Picture Processing Unit with background rendering."""
    _TILE_COLORS = tuple(
        tuple((((hi >> p) & 1) << 1 | ((lo >> p) & 1) for p in range(7, -1, -1)))
        for hi in range(256) for lo in range(256)
    )
    _PALETTE_SHADES = tuple(
        tuple((bgp >> (c << 1)) & 0x03 for c in range(4))
        for bgp in range(256)
    )

    def __init__(self, mmu):
        self.mmu = mmu
        self.cycles = 0
        self.scanline_dot = 0
        self.mode3_duration = 172
        self.mode = 2
        # The framebuffer stores one packed 24-bit colour per pixel
        # ((r << 16) | (g << 8) | b).  A single packed int per pixel keeps the
        # hot scanline writers at one assignment per pixel (same as RGB tuples)
        # while letting the display path convert the whole frame to a numpy
        # surface with a vectorised unpack instead of iterating 23 040 tuples.
        self.framebuffer = [0xFFFFFF] * (SCREEN_WIDTH * SCREEN_HEIGHT)
        self.bg_palette_idx = bytearray(SCREEN_WIDTH * SCREEN_HEIGHT)
        # Reusable all-zero scanline for clearing BG priority when BG is disabled.
        self._zero_row = bytes(SCREEN_WIDTH)
        self.shades = [(r << 16) | (g << 8) | b for (r, g, b) in PALETTE_DMG]
        self._rebuild_dmg_lut()
        self.is_cgb = False
        self.is_sgb = False
        self.bg_palette_data = bytearray(64)
        self.obj_palette_data = bytearray(64)
        self.bg_palette_addr = 0x00
        self.obj_palette_addr = 0x00
        self.cgb_opri = 0x00
        self.prev_stat_irq = False
        self.lcd_was_on = False
        self.window_line_counter = 0
        # Latched once LY==WY occurs in a frame; the window then stays active for
        # the rest of the frame even if WY is changed, matching hardware.
        self.window_active = False
        self._scanline_sprites = None  # OAM hits reused between mode-3 entry and sprite pass
        self._scanline_sprite_height = None
        self._bg_rgb = [0] * 32
        self._obj_rgb = [0] * 32
        unsigned_addrs = tuple(0x8000 + i * 16 for i in range(256))
        signed_addrs = tuple(0x9000 + ((i if i < 128 else i - 256) << 4) for i in range(256))
        self._tile_base_addrs = (unsigned_addrs, signed_addrs)

    def set_palette(self, palette):
        self.shades = [(r << 16) | (g << 8) | b for (r, g, b) in palette]
        self._rebuild_dmg_lut()

    def _rebuild_dmg_lut(self):
        """Pre-map BGP/OBP register values to packed colours for the active DMG palette."""
        shades = self.shades
        self._dmg_bgp_rgb = tuple(
            tuple(shades[self._PALETTE_SHADES[bgp][c]] for c in range(4))
            for bgp in range(256)
        )

    @staticmethod
    def _cgb_rgb555_to_rgb888(low, high):
        r5 = low & 0x1F
        g5 = ((low >> 5) & 0x07) | ((high & 0x03) << 3)
        b5 = (high >> 2) & 0x1F
        # Apply GBC color correction (Gambatte integer approximation)
        r = (r5 * 26 + g5 * 4 + b5 * 2) >> 5
        g = (g5 * 24 + b5 * 8) >> 5
        b = (r5 * 6 + g5 * 14 + b5 * 12) >> 5
        return ((r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2))

    def _update_cgb_bg_color(self, color_idx):
        off = color_idx * 2
        r, g, b = self._cgb_rgb555_to_rgb888(self.bg_palette_data[off],
                                              self.bg_palette_data[off + 1])
        self._bg_rgb[color_idx] = (r << 16) | (g << 8) | b

    def _update_cgb_obj_color(self, color_idx):
        off = color_idx * 2
        r, g, b = self._cgb_rgb555_to_rgb888(self.obj_palette_data[off],
                                              self.obj_palette_data[off + 1])
        self._obj_rgb[color_idx] = (r << 16) | (g << 8) | b

    def _cgb_init_palettes(self):
        dm = [0xFFFF, 0xAD55, 0x52AA, 0x0000]
        for pal in range(8):
            for c in range(4):
                col = dm[c] if pal == 0 else 0x0000
                off = pal * 8 + c * 2
                self.bg_palette_data[off] = col & 0xFF
                self.bg_palette_data[off + 1] = (col >> 8) & 0xFF
                self.obj_palette_data[off] = col & 0xFF
                self.obj_palette_data[off + 1] = (col >> 8) & 0xFF
        for i in range(32):
            self._update_cgb_bg_color(i)
            self._update_cgb_obj_color(i)

    def write_cgb_register(self, address, value):
        if address == 0xFF68:
            self.bg_palette_addr = value
        elif address == 0xFF69:
            idx = self.bg_palette_addr & 0x3F
            self.bg_palette_data[idx] = value
            self._update_cgb_bg_color(idx // 2)
            if self.bg_palette_addr & 0x80:
                self.bg_palette_addr = 0x80 | ((idx + 1) & 0x3F)
        elif address == 0xFF6A:
            self.obj_palette_addr = value
        elif address == 0xFF6B:
            idx = self.obj_palette_addr & 0x3F
            self.obj_palette_data[idx] = value
            self._update_cgb_obj_color(idx // 2)
            if self.obj_palette_addr & 0x80:
                self.obj_palette_addr = 0x80 | ((idx + 1) & 0x3F)
        elif address == 0xFF6C:
            self.cgb_opri = value & 0x01

    def step(self, cycles_passed):
        mem = self.mmu.memory
        lcdc = mem[0xFF40]

        if not (lcdc & 0x80):
            if self.lcd_was_on:
                mem[0xFF44] = 0
                self.mode = 0
                self.scanline_dot = 0
                self.window_line_counter = 0
                self.window_active = False
                self.prev_stat_irq = False
                stat = mem[0xFF41] & 0xF8
                mem[0xFF41] = stat
                self.lcd_was_on = False
            return

        if not self.lcd_was_on:
            mem[0xFF44] = 0
            self.scanline_dot = 0
            self.mode = 2
            self._update_stat_mode(2)
            self._check_lyc(0)
            self.lcd_was_on = True

        dot = self.scanline_dot + cycles_passed
        mode3_end = 80 + self.mode3_duration
        sd = self.scanline_dot

        while dot >= 456:
            ly = mem[0xFF44]
            if ly < 144:
                if sd < 80:
                    self._enter_mode3(ly, lcdc)
                if sd < mode3_end:
                    self._update_stat_mode(0)
                    if self.mmu.hdma_active:
                        self.mmu._hdma_hblank_step()
            self._finish_scanline()
            sd = 0
            self.scanline_dot = 0
            if not (mem[0xFF40] & 0x80):
                return
            dot -= 456
            dot += self._begin_scanline()

        if dot > 0:
            ly = mem[0xFF44]
            if ly < 144:
                mode3_end = 80 + self.mode3_duration
                if sd < 80 <= dot:
                    self._enter_mode3(ly, lcdc)
                if sd < mode3_end <= dot:
                    self._update_stat_mode(0)
                    if self.mmu.hdma_active:
                        self.mmu._hdma_hblank_step()
            self.scanline_dot = dot

    def _begin_scanline(self):
        mem = self.mmu.memory
        ly = mem[0xFF44]

        if ly < 144:
            self._update_stat_mode(2)
            self._check_lyc(ly)
        else:
            self._update_stat_mode(1)
            if ly == 144:
                mem[0xFF0F] |= IF_VBLANK
            self._check_lyc(ly)
        return 0

    def _advance_dots(self, target_dot):
        mem = self.mmu.memory
        ly = mem[0xFF44]
        lcdc = mem[0xFF40]

        if ly >= 144:
            return

        mode3_start = 80
        mode3_end = 80 + self.mode3_duration

        if self.scanline_dot < mode3_start <= target_dot:
            self._enter_mode3(ly, lcdc)

        if self.scanline_dot < mode3_end <= target_dot:
            self._update_stat_mode(0)
            if self.mmu.hdma_active:
                self.mmu._hdma_hblank_step()

    def _scanline_oam(self, ly, sprite_height):
        """Return up to 10 OAM entries overlapping scanline ``ly``."""
        mem = self.mmu.memory
        sprites = []
        for i in range(40):
            oam_addr = 0xFE00 + i * 4
            y = mem[oam_addr]
            if y == 0 or y >= 160:
                continue
            spr_y = y - 16
            if spr_y > ly or spr_y + sprite_height <= ly:
                continue
            sprites.append((
                mem[oam_addr + 1] - 8,
                spr_y,
                mem[oam_addr + 2],
                mem[oam_addr + 3],
            ))
            if len(sprites) >= 10:
                break
        return sprites

    def _enter_mode3(self, ly, lcdc):
        mem = self.mmu.memory
        scx = mem[0xFF43]
        sprite_height = 16 if (lcdc & 0x04) else 8
        sprites = self._scanline_oam(ly, sprite_height)
        self.mode3_duration = 172 + (scx & 7) + len(sprites) * 11
        if self.is_cgb or (lcdc & 0x01) or (lcdc & 0x20) or (lcdc & 0x02):
            # Reuse the OAM scan for the sprite render pass on this line.
            self._scanline_sprites = sprites if (lcdc & 0x02) else None
            self._scanline_sprite_height = sprite_height if (lcdc & 0x02) else None
            self._render_scanline(ly, lcdc)
            self._scanline_sprites = None
            self._scanline_sprite_height = None
        self.mode = 3
        stat = (mem[0xFF41] & 0xFC) | 3
        mem[0xFF41] = stat
        # Mode 3 has no STAT source, so the interrupt line drops here. Re-evaluate
        # it (unless LY=LYC holds) so the rising edge into mode 0 fires the HBlank
        # STAT interrupt even when the mode-2 OAM source was also enabled.
        self._fire_stat_irq(stat)

    def _finish_scanline(self):
        mem = self.mmu.memory
        ly = mem[0xFF44]
        new_ly = ly + 1
        if new_ly == 154:
            new_ly = 0
            self.window_line_counter = 0
            self.window_active = False
        mem[0xFF44] = new_ly

    def _update_stat_mode(self, mode):
        self.mode = mode
        mem = self.mmu.memory
        stat = (mem[0xFF41] & 0xFC) | mode
        mem[0xFF41] = stat
        self._fire_stat_irq(stat)

    def _check_lyc(self, ly):
        mem = self.mmu.memory
        lyc = mem[0xFF45]
        stat = mem[0xFF41]
        if ly == lyc:
            stat |= 0x04
        else:
            stat &= ~0x04
        mem[0xFF41] = stat
        self._fire_stat_irq(stat)

    def _fire_stat_irq(self, stat):
        mem = self.mmu.memory
        ly = mem[0xFF44]
        lyc = mem[0xFF45]
        mode = self.mode
        # Combined STAT interrupt line: OR of all enabled sources
        current_irq = (
            (mode == 0 and (stat & 0x08)) or
            (mode == 1 and (stat & 0x10)) or
            (mode == 2 and (stat & 0x20)) or
            (ly == lyc and (stat & 0x40))
        )
        # Fire only on rising edge (0->1)
        if current_irq and not self.prev_stat_irq:
            mem[0xFF0F] |= IF_LCD
        self.prev_stat_irq = current_irq

    def _render_scanline(self, ly, lcdc):
        mem = self.mmu.memory
        is_cgb = self.is_cgb
        is_sgb = self.is_sgb and not is_cgb
        fb_row = ly * SCREEN_WIDTH
        if is_sgb:
            mask = self.mmu.sgb_mask
            if mask == 1:
                return
            if mask == 2 or mask == 3:
                fill = 0 if mask == 2 else self.mmu.sgb_pal_rgb[0]
                fb = self.framebuffer
                for i in range(SCREEN_WIDTH):
                    fb[fb_row + i] = fill
                self.bg_palette_idx[fb_row:fb_row + SCREEN_WIDTH] = self._zero_row
                return
        # On CGB, LCDC bit 0 is BG/OBJ priority flag, not BG enable.
        # BG is always rendered in CGB mode regardless of bit 0.
        bg_enabled = (lcdc & 0x01) or is_cgb
        fb_row = ly * SCREEN_WIDTH

        if not bg_enabled:
            # DMG: BG/Window off - the background outputs colour 0 through BGP
            # (usually white, but BGP can remap it), then sprites draw on top.
            white = self._dmg_bgp_rgb[mem[0xFF47]][0]
            fb = self.framebuffer
            bg_pri = self.bg_palette_idx
            for i in range(SCREEN_WIDTH):
                fb[fb_row + i] = white
            bg_pri[fb_row:fb_row + SCREEN_WIDTH] = self._zero_row
            if self.is_sgb and not self.is_cgb:
                self._apply_sgb_scanline(ly, mem[0xFF47])
            if lcdc & 0x02:
                self._render_sprites(ly)
            return

        bg_map_base = 0x9C00 if (lcdc & 0x08) else 0x9800
        signed_tiles = not (lcdc & 0x10)
        tile_base_idx = 1 if signed_tiles else 0
        tile_addrs = self._tile_base_addrs[tile_base_idx]

        scy = mem[0xFF42]
        scx = mem[0xFF43]
        bgp = mem[0xFF47]

        bg_y = (scy + ly) & 0xFF
        tile_row = bg_y >> 3
        pixel_row = bg_y & 7

        fb = self.framebuffer
        bg_pri = self.bg_palette_idx
        tile_colors = self._TILE_COLORS
        dmg_rgb = self._dmg_bgp_rgb[bgp]
        scx_mod = scx & 7
        first_tile_col = scx >> 3
        # Fast path: when scx is 8-aligned, every tile column is fully on-screen,
        # so the inner 8-pixel loop has no bounds checks.
        if scx_mod == 0:
            if is_cgb:
                vram_bank1 = self.mmu.vram_bank1
                pr = self._bg_rgb
                for tile_col_offset in range(20):
                    tile_col = (first_tile_col + tile_col_offset) & 0x1F
                    map_addr = bg_map_base + (tile_row << 5) + tile_col
                    tile_idx = mem[map_addr]
                    attr = vram_bank1[map_addr - 0x8000]
                    pal = attr & 0x07
                    pri_mask = attr & 0x80
                    vram_bank = attr & 0x08
                    addr = tile_addrs[tile_idx]
                    row = pixel_row
                    if attr & 0x40:
                        row = 7 - row
                    addr += row << 1
                    if vram_bank:
                        lo = vram_bank1[addr - 0x8000]
                        hi = vram_bank1[addr - 0x8000 + 1]
                    else:
                        lo = mem[addr]
                        hi = mem[addr + 1]
                    colors = tile_colors[(hi << 8) | lo]
                    if attr & 0x20:
                        c0, c1, c2, c3, c4, c5, c6, c7 = colors[7], colors[6], colors[5], colors[4], colors[3], colors[2], colors[1], colors[0]
                    else:
                        c0, c1, c2, c3, c4, c5, c6, c7 = colors
                    base = fb_row + tile_col_offset * 8
                    off = pal * 4
                    fb[base]     = pr[off + c0]
                    fb[base + 1] = pr[off + c1]
                    fb[base + 2] = pr[off + c2]
                    fb[base + 3] = pr[off + c3]
                    fb[base + 4] = pr[off + c4]
                    fb[base + 5] = pr[off + c5]
                    fb[base + 6] = pr[off + c6]
                    fb[base + 7] = pr[off + c7]
                    bp = bg_pri
                    bp[base]     = c0 | pri_mask
                    bp[base + 1] = c1 | pri_mask
                    bp[base + 2] = c2 | pri_mask
                    bp[base + 3] = c3 | pri_mask
                    bp[base + 4] = c4 | pri_mask
                    bp[base + 5] = c5 | pri_mask
                    bp[base + 6] = c6 | pri_mask
                    bp[base + 7] = c7 | pri_mask
            else:
                for tile_col_offset in range(20):
                    tile_col = (first_tile_col + tile_col_offset) & 0x1F
                    map_addr = bg_map_base + (tile_row << 5) + tile_col
                    tile_idx = mem[map_addr]
                    addr = tile_addrs[tile_idx] + (pixel_row << 1)
                    lo = mem[addr]
                    hi = mem[addr + 1]
                    c0, c1, c2, c3, c4, c5, c6, c7 = tile_colors[(hi << 8) | lo]
                    base = fb_row + tile_col_offset * 8
                    fb[base]     = dmg_rgb[c0]
                    fb[base + 1] = dmg_rgb[c1]
                    fb[base + 2] = dmg_rgb[c2]
                    fb[base + 3] = dmg_rgb[c3]
                    fb[base + 4] = dmg_rgb[c4]
                    fb[base + 5] = dmg_rgb[c5]
                    fb[base + 6] = dmg_rgb[c6]
                    fb[base + 7] = dmg_rgb[c7]
                    bp = bg_pri
                    bp[base]     = c0
                    bp[base + 1] = c1
                    bp[base + 2] = c2
                    bp[base + 3] = c3
                    bp[base + 4] = c4
                    bp[base + 5] = c5
                    bp[base + 6] = c6
                    bp[base + 7] = c7
        else:
            if is_cgb:
                vram_bank1 = self.mmu.vram_bank1
                pr = self._bg_rgb
                for tile_col_offset in range(21):
                    tile_col = (first_tile_col + tile_col_offset) & 0x1F
                    map_addr = bg_map_base + (tile_row << 5) + tile_col
                    tile_idx = mem[map_addr]
                    attr = vram_bank1[map_addr - 0x8000]
                    pal = attr & 0x07
                    pri_mask = attr & 0x80
                    vram_bank = attr & 0x08
                    addr = tile_addrs[tile_idx]
                    row = pixel_row
                    if attr & 0x40:
                        row = 7 - row
                    addr += row << 1
                    if vram_bank:
                        lo = vram_bank1[addr - 0x8000]
                        hi = vram_bank1[addr - 0x8000 + 1]
                    else:
                        lo = mem[addr]
                        hi = mem[addr + 1]
                    colors = tile_colors[(hi << 8) | lo]
                    if attr & 0x20:
                        colors = (colors[7], colors[6], colors[5], colors[4], colors[3], colors[2], colors[1], colors[0])
                    tile_x_start = tile_col_offset * 8 - scx_mod
                    for p in range(8):
                        x = tile_x_start + p
                        if x < 0 or x >= SCREEN_WIDTH:
                            continue
                        fb[fb_row + x] = pr[pal * 4 + colors[p]]
                        bg_pri[fb_row + x] = colors[p] | pri_mask
            else:
                for tile_col_offset in range(21):
                    tile_col = (first_tile_col + tile_col_offset) & 0x1F
                    map_addr = bg_map_base + (tile_row << 5) + tile_col
                    tile_idx = mem[map_addr]
                    addr = tile_addrs[tile_idx] + (pixel_row << 1)
                    lo = mem[addr]
                    hi = mem[addr + 1]
                    colors = tile_colors[(hi << 8) | lo]
                    tile_x_start = tile_col_offset * 8 - scx_mod
                    for p in range(8):
                        x = tile_x_start + p
                        if x < 0 or x >= SCREEN_WIDTH:
                            continue
                        c = colors[p]
                        fb[fb_row + x] = dmg_rgb[c]
                        bg_pri[fb_row + x] = c

        # The window's WY==LY trigger latches for the whole frame; once latched the
        # window keeps rendering even if WY is later moved past the current line.
        if ly == mem[0xFF4A]:
            self.window_active = True
        if (lcdc & 0x20) and self.window_active:
            wx_raw = mem[0xFF4B]
            if (wx_raw - 7) < SCREEN_WIDTH:
                self._render_window(ly)
                self.window_line_counter += 1
        if self.is_sgb and not self.is_cgb:
            self._apply_sgb_scanline(ly, mem[0xFF47])
        if lcdc & 0x02:
            self._render_sprites(ly)

    def _apply_sgb_scanline(self, ly, bgp):
        """Recolour a DMG scanline using the 20×18 SGB attribute map."""
        shades = self._PALETTE_SHADES[bgp]
        fb = self.framebuffer
        bg_pri = self.bg_palette_idx
        attr = self.mmu.sgb_attr
        pal_rgb = self.mmu.sgb_pal_rgb
        row = ly * SCREEN_WIDTH
        attr_row = (ly >> 3) * 20
        for x in range(SCREEN_WIDTH):
            pal = attr[attr_row + (x >> 3)]
            c = bg_pri[row + x] & 3
            fb[row + x] = pal_rgb[pal * 4 + shades[c]]

    def _render_window(self, ly):
        mem = self.mmu.memory
        lcdc = mem[0xFF40]
        wy = mem[0xFF4A]
        wx_raw = mem[0xFF4B]
        win_x_offset = wx_raw - 7
        if win_x_offset >= SCREEN_WIDTH:
            return
        win_map_base = 0x9C00 if (lcdc & 0x40) else 0x9800
        signed_tiles = not (lcdc & 0x10)
        tile_base_idx = 1 if signed_tiles else 0
        tile_addrs = self._tile_base_addrs[tile_base_idx]
        bgp = mem[0xFF47]
        win_y = self.window_line_counter
        tile_row = win_y >> 3
        pixel_row = win_y & 7
        fb_row = ly * SCREEN_WIDTH
        fb = self.framebuffer
        bg_pri = self.bg_palette_idx
        tile_colors = self._TILE_COLORS
        dmg_rgb = self._dmg_bgp_rgb[bgp]
        is_cgb = self.is_cgb
        if is_cgb:
            vram_bank1 = self.mmu.vram_bank1
            pr = self._bg_rgb
        for tile_col in range(21):
            x = tile_col * 8 + win_x_offset
            if x >= SCREEN_WIDTH:
                break
            if x + 8 <= 0:
                continue
            map_addr = win_map_base + (tile_row << 5) + tile_col
            tile_idx = mem[map_addr]
            if is_cgb:
                attr = vram_bank1[map_addr - 0x8000]
                pal = attr & 0x07
                pri_mask = attr & 0x80
                vram_bank = attr & 0x08
                addr = tile_addrs[tile_idx]
                row = pixel_row
                if attr & 0x40:
                    row = 7 - row
                addr += row << 1
                if vram_bank:
                    lo = vram_bank1[addr - 0x8000]
                    hi = vram_bank1[addr - 0x8000 + 1]
                else:
                    lo = mem[addr]
                    hi = mem[addr + 1]
                colors = tile_colors[(hi << 8) | lo]
                if attr & 0x20:
                    colors = (colors[7], colors[6], colors[5], colors[4], colors[3], colors[2], colors[1], colors[0])
            else:
                addr = tile_addrs[tile_idx] + (pixel_row << 1)
                lo = mem[addr]
                hi = mem[addr + 1]
                colors = tile_colors[(hi << 8) | lo]
            if x < 0:
                for p in range(8):
                    px = x + p
                    if 0 <= px < SCREEN_WIDTH:
                        c = colors[p]
                        if is_cgb:
                            fb[fb_row + px] = pr[pal * 4 + c]
                            bg_pri[fb_row + px] = c | pri_mask
                        else:
                            fb[fb_row + px] = dmg_rgb[c]
                            bg_pri[fb_row + px] = c
            elif x + 8 > SCREEN_WIDTH:
                for p in range(SCREEN_WIDTH - x):
                    c = colors[p]
                    if is_cgb:
                        fb[fb_row + x + p] = pr[pal * 4 + c]
                        bg_pri[fb_row + x + p] = c | pri_mask
                    else:
                        fb[fb_row + x + p] = dmg_rgb[c]
                        bg_pri[fb_row + x + p] = c
            else:
                c0, c1, c2, c3, c4, c5, c6, c7 = colors
                base = fb_row + x
                if is_cgb:
                    off = pal * 4
                    fb[base]     = pr[off + c0]
                    fb[base + 1] = pr[off + c1]
                    fb[base + 2] = pr[off + c2]
                    fb[base + 3] = pr[off + c3]
                    fb[base + 4] = pr[off + c4]
                    fb[base + 5] = pr[off + c5]
                    fb[base + 6] = pr[off + c6]
                    fb[base + 7] = pr[off + c7]
                    bp = bg_pri
                    bp[base]     = c0 | pri_mask
                    bp[base + 1] = c1 | pri_mask
                    bp[base + 2] = c2 | pri_mask
                    bp[base + 3] = c3 | pri_mask
                    bp[base + 4] = c4 | pri_mask
                    bp[base + 5] = c5 | pri_mask
                    bp[base + 6] = c6 | pri_mask
                    bp[base + 7] = c7 | pri_mask
                else:
                    fb[base]     = dmg_rgb[c0]
                    fb[base + 1] = dmg_rgb[c1]
                    fb[base + 2] = dmg_rgb[c2]
                    fb[base + 3] = dmg_rgb[c3]
                    fb[base + 4] = dmg_rgb[c4]
                    fb[base + 5] = dmg_rgb[c5]
                    fb[base + 6] = dmg_rgb[c6]
                    fb[base + 7] = dmg_rgb[c7]
                    bp = bg_pri
                    bp[base]     = c0
                    bp[base + 1] = c1
                    bp[base + 2] = c2
                    bp[base + 3] = c3
                    bp[base + 4] = c4
                    bp[base + 5] = c5
                    bp[base + 6] = c6
                    bp[base + 7] = c7

    def _cgb_sprite_plot(self, fb, bg_pri, idx, lcdc, bg_priority, pr, pal, c):
        if c == 0:
            return
        bg_color_idx = bg_pri[idx] & 0x7F
        if (lcdc & 0x01) and bg_color_idx != 0 and ((bg_pri[idx] & 0x80) or bg_priority):
            return
        fb[idx] = pr[pal * 4 + c]

    def _cgb_sprite_plot8(self, fb, bg_pri, fb_row, spr_x, lcdc, bg_priority, pr, pal, colors):
        """Unrolled CGB sprite row for sprites fully within the 160px scanline."""
        off = pal * 4
        cgb_pri = lcdc & 0x01
        c0, c1, c2, c3, c4, c5, c6, c7 = colors
        base = fb_row + spr_x
        for sx, c in enumerate((c0, c1, c2, c3, c4, c5, c6, c7)):
            if c == 0:
                continue
            idx = base + sx
            if cgb_pri:
                bg_color_idx = bg_pri[idx] & 0x7F
                if bg_color_idx != 0 and ((bg_pri[idx] & 0x80) or bg_priority):
                    continue
            fb[idx] = pr[off + c]

    def _dmg_sprite_plot8(self, fb, bg_pri, fb_row, spr_x, bg_priority, dmg_obj_rgb, colors):
        """Unrolled DMG sprite row for sprites fully within the 160px scanline."""
        base = fb_row + spr_x
        for sx, c in enumerate(colors):
            if c == 0:
                continue
            idx = base + sx
            if bg_priority and bg_pri[idx] != 0:
                continue
            fb[idx] = dmg_obj_rgb[c]

    def _render_sprites(self, ly):
        mem = self.mmu.memory
        lcdc = mem[0xFF40]
        # LCDC bit 1: OBJ display enable.  Skip the entire 40-entry OAM scan
        # when sprites are disabled — saves ~11,520 OAM reads/frame.
        if not (lcdc & 0x02):
            return
        sprite_height = 16 if (lcdc & 0x04) else 8
        sprites = self._scanline_sprites
        if sprites is None or self._scanline_sprite_height != sprite_height:
            sprites = self._scanline_oam(ly, sprite_height)
        # CGB: OAM index priority (OPRI=0). DMG / CGB with OPRI=1: sort by X.
        if not self.is_cgb or self.cgb_opri:
            sprites.sort(key=lambda s: s[0])
        obp0 = mem[0xFF48]
        obp1 = mem[0xFF49]
        fb_row = ly * SCREEN_WIDTH
        bg_pri = self.bg_palette_idx
        fb = self.framebuffer
        tile_colors = self._TILE_COLORS
        obp_rgb = (self._dmg_bgp_rgb[obp0], self._dmg_bgp_rgb[obp1])
        if self.is_sgb and not self.is_cgb:
            pal_rgb = self.mmu.sgb_pal_rgb
            sh0 = self._PALETTE_SHADES[obp0]
            sh1 = self._PALETTE_SHADES[obp1]
            obp_rgb = (
                tuple(pal_rgb[sh0[i]] for i in range(4)),
                tuple(pal_rgb[4 + sh1[i]] for i in range(4)),
            )
        is_cgb = self.is_cgb
        unsigned_addrs = self._tile_base_addrs[0]
        if is_cgb:
            vram_bank1 = self.mmu.vram_bank1
            pr = self._obj_rgb
            # LCDC bit 0: when 1, BG tile attributes control sprite priority.
            # When 0, sprites always draw on top of BG (OAM priority ignored).
            lcdc = mem[0xFF40]
        for spr_x, spr_y, tile, flags in reversed(sprites):
            sprite_pixel_y = ly - spr_y
            if flags & 0x40:
                sprite_pixel_y = sprite_height - 1 - sprite_pixel_y
            if sprite_height == 16:
                tile_row_offset = sprite_pixel_y >> 3
                tile_idx_used = (tile & 0xFE) + tile_row_offset
            else:
                tile_idx_used = tile
            tile_addr = unsigned_addrs[tile_idx_used] + (sprite_pixel_y & 7) * 2
            if is_cgb and (flags & 0x08):
                lo = vram_bank1[tile_addr - 0x8000]
                hi = vram_bank1[tile_addr - 0x8000 + 1]
            else:
                lo = mem[tile_addr]
                hi = mem[tile_addr + 1]
            colors = tile_colors[(hi << 8) | lo]
            x_flip = flags & 0x20
            bg_priority = flags & 0x80
            if is_cgb:
                pal = flags & 0x07
                use_cgb_obj = True
            else:
                use_obp1 = bool(flags & 0x10)
                dmg_obj_rgb = obp_rgb[use_obp1]
                use_cgb_obj = False
            on_screen = spr_x >= 0 and spr_x + 8 <= SCREEN_WIDTH
            if use_cgb_obj and on_screen and not x_flip:
                self._cgb_sprite_plot8(fb, bg_pri, fb_row, spr_x, lcdc, bg_priority, pr, pal, colors)
            elif use_cgb_obj and on_screen and x_flip:
                flipped = (colors[7], colors[6], colors[5], colors[4],
                             colors[3], colors[2], colors[1], colors[0])
                self._cgb_sprite_plot8(fb, bg_pri, fb_row, spr_x, lcdc, bg_priority, pr, pal, flipped)
            elif on_screen:
                row = (colors[7], colors[6], colors[5], colors[4],
                       colors[3], colors[2], colors[1], colors[0]) if x_flip else colors
                self._dmg_sprite_plot8(fb, bg_pri, fb_row, spr_x, bg_priority, dmg_obj_rgb, row)
            elif x_flip:
                for sx in range(8):
                    pixel_x = spr_x + sx
                    if pixel_x < 0 or pixel_x >= SCREEN_WIDTH:
                        continue
                    c = colors[7 - sx]
                    if use_cgb_obj:
                        self._cgb_sprite_plot(fb, bg_pri, fb_row + pixel_x, lcdc, bg_priority, pr, pal, c)
                    elif c != 0:
                        idx = fb_row + pixel_x
                        if bg_priority and bg_pri[idx] != 0:
                            continue
                        fb[idx] = dmg_obj_rgb[c]
            else:
                for sx in range(8):
                    pixel_x = spr_x + sx
                    if pixel_x < 0 or pixel_x >= SCREEN_WIDTH:
                        continue
                    c = colors[sx]
                    if use_cgb_obj:
                        self._cgb_sprite_plot(fb, bg_pri, fb_row + pixel_x, lcdc, bg_priority, pr, pal, c)
                    elif c != 0:
                        idx = fb_row + pixel_x
                        if bg_priority and bg_pri[idx] != 0:
                            continue
                        fb[idx] = dmg_obj_rgb[c]


class Timers:
    """Timer registers: DIV, TIMA, TMA, TAC."""
    _TIMA_RATES = (1024, 16, 64, 256)

    def __init__(self, mmu):
        self.mmu = mmu
        self.div_counter = 0
        self.tima_accum = 0

    def reset_div(self):
        old = self.div_counter
        self.div_counter = 0
        apu = getattr(self.mmu, 'apu', None)
        if apu is not None:
            bit = 13 if (self.mmu.key1 & 0x80) else 12
            if (old >> bit) & 1:
                if apu.power or not apu.is_cgb:
                    apu._frame_seq_tick()
            apu.fs_div = 0
            apu._fs_remain = apu._frame_seq_period(bool(self.mmu.key1 & 0x80))

    def step(self, cycles):
        mem = self.mmu.memory
        div_counter = self.div_counter + cycles
        self.div_counter = div_counter
        mem[0xFF04] = (div_counter >> 8) & 0xFF
        tac = mem[0xFF07]
        if not (tac & 0x04):
            return
        self._tima_step(cycles, tac)

    def _tima_step(self, cycles, tac):
        mem = self.mmu.memory
        step_cyc = self._TIMA_RATES[tac & 0x03]
        tima_accum = self.tima_accum + cycles
        if tima_accum < step_cyc:
            self.tima_accum = tima_accum
            return
        overflows = tima_accum // step_cyc
        self.tima_accum = tima_accum - overflows * step_cyc
        tima = mem[0xFF05]
        tma = mem[0xFF06]
        for _ in range(overflows):
            if tima > 0xFF - 1:
                mem[0xFF0F] |= IF_TIMER
                tima = tma
            else:
                tima += 1
        mem[0xFF05] = tima & 0xFF


# ── Audio Processing Unit ───────────────────────────────────────────

_DUTY_PATTERNS = (
    (0, 0, 0, 0, 0, 0, 0, 1),  # 12.5%
    (1, 0, 0, 0, 0, 0, 0, 1),  # 25%
    (1, 0, 0, 0, 0, 1, 1, 1),  # 50%
    (0, 1, 1, 1, 1, 1, 1, 0),  # 75%
)

# Channel 4 noise divisor table (indexed by NR43 bits 2-0)
_NOISE_DIVISORS = (8, 16, 32, 48, 64, 80, 96, 112)

# Wave-channel volume shifts: 0=mute, 1=full (>>0), 2=half (>>1), 3=quarter (>>2)
_WAVE_VOL_SHIFT = (4, 0, 1, 2)

# OR masks for register reads (unused bits read as 1)
_APU_READ_OR = {
    0xFF10: 0x80, 0xFF11: 0x3F, 0xFF12: 0x00, 0xFF13: 0xFF, 0xFF14: 0xB8,
    0xFF15: 0xFF,
    0xFF16: 0x3F, 0xFF17: 0x00, 0xFF18: 0xFF, 0xFF19: 0xB8,
    0xFF1A: 0x7F, 0xFF1B: 0xFF, 0xFF1C: 0x9F, 0xFF1D: 0xFF, 0xFF1E: 0xB8,
    0xFF1F: 0xFF,
    0xFF20: 0xFF, 0xFF21: 0x00, 0xFF22: 0x00, 0xFF23: 0xB8,
    0xFF24: 0x00, 0xFF25: 0x00, 0xFF26: 0x70,
}

# Post-BIOS register defaults (DMG)
_APU_BOOT_VALUES = {
    0xFF10: 0x80, 0xFF11: 0xBF, 0xFF12: 0xF3, 0xFF13: 0xFF, 0xFF14: 0xBF,
    0xFF15: 0xFF,
    0xFF16: 0x3F, 0xFF17: 0x00, 0xFF18: 0xFF, 0xFF19: 0xBF,
    0xFF1A: 0x7F, 0xFF1B: 0xFF, 0xFF1C: 0x9F, 0xFF1D: 0xFF, 0xFF1E: 0xBF,
    0xFF1F: 0xFF,
    0xFF20: 0xFF, 0xFF21: 0x00, 0xFF22: 0x00, 0xFF23: 0xBF,
    0xFF24: 0x77, 0xFF25: 0xF3, 0xFF26: 0xF1,
}


class APU:
    """Game Boy Audio Processing Unit: two square waves, wave channel, noise channel.

    Drives a 44.1 kHz signed-16-bit stereo PCM stream that the host audio
    backend consumes. The frame sequencer runs at 512 Hz (one tick every
    8192 CPU cycles) and clocks the length counters, envelopes, and the
    channel-1 sweep.

    CGB differences from DMG:
    - Wave RAM reads always return the addressed byte even when CH3 is active.
    - Wave RAM writes while CH3 is active go to the byte CH3 is currently accessing.
    - Frame sequencer is reset to step 0 on APU power-on.
    """

    SAMPLE_RATE = 44100
    CPU_CLOCK = 4194304
    FRAME_SEQ_PERIOD = 8192  # CPU cycles per frame-sequencer tick (= 4194304 / 512)
    SOFT_BUFFER_CAP = 65536  # bytes (~370 ms stereo at 44.1 kHz); emergency drop threshold

    def __init__(self, mmu, is_cgb=False):
        self.mmu = mmu
        self.is_cgb = is_cgb
        self.power = True
        self.buffer = bytearray()

        # Frame sequencer — countdown to the next DIV bit-12/13 falling edge.
        # ``fs_div`` mirrors the CPU DIV counter for save-state compatibility.
        self.fs_div = 0
        self._fs_remain = self.FRAME_SEQ_PERIOD
        self.frame_seq_step = 0   # 0-7 step index

        # Fractional sample timer: produce one sample every CPU_CLOCK / SAMPLE_RATE cycles
        self.sample_accum = 0
        self.sample_num = self.CPU_CLOCK
        self.sample_den = self.SAMPLE_RATE

        # Master / mixer (NR50, NR51)
        self.vol_left = 7   # bits 6-4 of NR50
        self.vol_right = 7  # bits 2-0 of NR50
        self.pan_left = 0xF0  # bits 7-4 of NR51 (one bit per channel)
        self.pan_right = 0x30  # (NR51 post-boot 0xF3: lower nibble 0x3 << 4 = 0x30)

        # Channel 1: square + sweep
        self.ch1_enabled = False
        self.ch1_dac = False
        self.ch1_freq = 0
        self.ch1_freq_timer = 1
        self.ch1_duty = 2
        self.ch1_duty_step = 0
        self.ch1_length_enabled = False
        self.ch1_length = 0
        self.ch1_volume = 0
        self.ch1_env_initial = 0
        self.ch1_env_direction = 0
        self.ch1_env_period = 0
        self.ch1_env_timer = 0
        self.ch1_sweep_period = 0
        self.ch1_sweep_direction = 0
        self.ch1_sweep_shift = 0
        self.ch1_sweep_timer = 0
        self.ch1_sweep_enabled = False
        self.ch1_sweep_shadow = 0

        # Channel 2: square (no sweep)
        self.ch2_enabled = False
        self.ch2_dac = False
        self.ch2_freq = 0
        self.ch2_freq_timer = 1
        self.ch2_duty = 2
        self.ch2_duty_step = 0
        self.ch2_length_enabled = False
        self.ch2_length = 0
        self.ch2_volume = 0
        self.ch2_env_initial = 0
        self.ch2_env_direction = 0
        self.ch2_env_period = 0
        self.ch2_env_timer = 0

        # Channel 3: wave
        self.ch3_enabled = False
        self.ch3_dac = False
        self.ch3_freq = 0
        self.ch3_freq_timer = 1
        self.ch3_length_enabled = False
        self.ch3_length = 0
        self.ch3_vol_shift = 4
        self.ch3_wave_pos = 0
        self.wave_ram = bytearray(16)

        # Channel 4: noise (LFSR)
        self.ch4_enabled = False
        self.ch4_dac = False
        self.ch4_freq_timer = 1
        self.ch4_length_enabled = False
        self.ch4_length = 0
        self.ch4_volume = 0
        self.ch4_env_initial = 0
        self.ch4_env_direction = 0
        self.ch4_env_period = 0
        self.ch4_env_timer = 0
        self.ch4_lfsr = 0x7FFF
        self.ch4_shift = 0
        self.ch4_width_mode = 0
        self.ch4_divisor_code = 0

        # Seed memory with post-boot register values so reads work even before
        # the ROM writes them.
        for addr, val in _APU_BOOT_VALUES.items():
            self.mmu.memory[addr] = val
        # Mirror wave RAM into mmu memory so CPU reads at FF30-FF3F return
        # the initial (silent) wave pattern even before the ROM writes it.
        for i in range(16):
            self.mmu.memory[0xFF30 + i] = self.wave_ram[i]
        self._refresh_nr52()

    # ── Register access ─────────────────────────────────────────────

    def write_register(self, addr, value):
        value &= 0xFF
        mem = self.mmu.memory

        # NR52 master power (writable any time)
        if addr == 0xFF26:
            new_power = bool(value & 0x80)
            if self.power and not new_power:
                self._power_off()
            elif not self.power and new_power:
                self.power = True
                if self.is_cgb:
                    self.frame_seq_step = 0
                    self.fs_div = 0
                    self._fs_remain = self._frame_seq_period(bool(self.mmu.key1 & 0x80))
            self.power = new_power
            self._refresh_nr52()
            return

        # When powered off, ignore writes to most registers. DMG still accepts
        # length-counter loads on NRx1, and wave RAM is always reachable.
        if not self.power and addr != 0xFF26:
            if 0xFF30 <= addr <= 0xFF3F:
                self.wave_ram[addr - 0xFF30] = value
                mem[addr] = value
            elif not self.is_cgb:
                if addr == 0xFF11:
                    self.ch1_length = 64 - (value & 0x3F)
                elif addr == 0xFF16:
                    self.ch2_length = 64 - (value & 0x3F)
                elif addr == 0xFF1B:
                    self.ch3_length = 256 - (value & 0xFF)
                elif addr == 0xFF20:
                    self.ch4_length = 64 - (value & 0x3F)
            return

        # Wave RAM
        if 0xFF30 <= addr <= 0xFF3F:
            if self.is_cgb and self.ch3_enabled:
                self.wave_ram[self.ch3_wave_pos >> 1] = value
            elif not self.ch3_enabled:
                self.wave_ram[addr - 0xFF30] = value
            mem[addr] = value
            return

        # Channel 1
        if addr == 0xFF10:  # NR10: sweep
            self.ch1_sweep_period = (value >> 4) & 0x07
            self.ch1_sweep_direction = (value >> 3) & 0x01
            self.ch1_sweep_shift = value & 0x07
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF11:  # NR11: duty / length
            self.ch1_duty = (value >> 6) & 0x03
            self.ch1_length = 64 - (value & 0x3F)
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF12:  # NR12: volume envelope
            self.ch1_env_initial = (value >> 4) & 0x0F
            self.ch1_env_direction = (value >> 3) & 0x01
            self.ch1_env_period = value & 0x07
            self.ch1_dac = (value & 0xF8) != 0
            if not self.ch1_dac:
                self.ch1_enabled = False
                self._refresh_nr52()
            mem[addr] = value
        elif addr == 0xFF13:  # NR13: freq lo
            self.ch1_freq = (self.ch1_freq & 0x700) | value
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF14:  # NR14: trigger / length-en / freq hi
            self.ch1_freq = (self.ch1_freq & 0xFF) | ((value & 0x07) << 8)
            self.ch1_length_enabled = bool(value & 0x40)
            if value & 0x80:
                self._trigger_ch1()
            mem[addr] = value | _APU_READ_OR[addr]

        # Channel 2
        elif addr == 0xFF16:  # NR21
            self.ch2_duty = (value >> 6) & 0x03
            self.ch2_length = 64 - (value & 0x3F)
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF17:  # NR22
            self.ch2_env_initial = (value >> 4) & 0x0F
            self.ch2_env_direction = (value >> 3) & 0x01
            self.ch2_env_period = value & 0x07
            self.ch2_dac = (value & 0xF8) != 0
            if not self.ch2_dac:
                self.ch2_enabled = False
                self._refresh_nr52()
            mem[addr] = value
        elif addr == 0xFF18:  # NR23
            self.ch2_freq = (self.ch2_freq & 0x700) | value
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF19:  # NR24
            self.ch2_freq = (self.ch2_freq & 0xFF) | ((value & 0x07) << 8)
            self.ch2_length_enabled = bool(value & 0x40)
            if value & 0x80:
                self._trigger_ch2()
            mem[addr] = value | _APU_READ_OR[addr]

        # Channel 3
        elif addr == 0xFF1A:  # NR30
            self.ch3_dac = bool(value & 0x80)
            if not self.ch3_dac:
                self.ch3_enabled = False
                self._refresh_nr52()
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF1B:  # NR31
            self.ch3_length = 256 - value
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF1C:  # NR32
            self.ch3_vol_shift = _WAVE_VOL_SHIFT[(value >> 5) & 0x03]
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF1D:  # NR33
            self.ch3_freq = (self.ch3_freq & 0x700) | value
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF1E:  # NR34
            self.ch3_freq = (self.ch3_freq & 0xFF) | ((value & 0x07) << 8)
            self.ch3_length_enabled = bool(value & 0x40)
            if value & 0x80:
                self._trigger_ch3()
            mem[addr] = value | _APU_READ_OR[addr]

        # Channel 4
        elif addr == 0xFF20:  # NR41
            self.ch4_length = 64 - (value & 0x3F)
            mem[addr] = value | _APU_READ_OR[addr]
        elif addr == 0xFF21:  # NR42
            self.ch4_env_initial = (value >> 4) & 0x0F
            self.ch4_env_direction = (value >> 3) & 0x01
            self.ch4_env_period = value & 0x07
            self.ch4_dac = (value & 0xF8) != 0
            if not self.ch4_dac:
                self.ch4_enabled = False
                self._refresh_nr52()
            mem[addr] = value
        elif addr == 0xFF22:  # NR43
            self.ch4_shift = (value >> 4) & 0x0F
            self.ch4_width_mode = (value >> 3) & 0x01
            self.ch4_divisor_code = value & 0x07
            mem[addr] = value
        elif addr == 0xFF23:  # NR44
            self.ch4_length_enabled = bool(value & 0x40)
            if value & 0x80:
                self._trigger_ch4()
            mem[addr] = value | _APU_READ_OR[addr]

        # Master mixer
        elif addr == 0xFF24:  # NR50
            self.vol_left = (value >> 4) & 0x07
            self.vol_right = value & 0x07
            mem[addr] = value
        elif addr == 0xFF25:  # NR51
            self.pan_left = value & 0xF0
            self.pan_right = (value & 0x0F) << 4
            mem[addr] = value

        # FF15 / FF1F are unused; just store
        else:
            mem[addr] = value | _APU_READ_OR.get(addr, 0)

    def read_register(self, addr):
        """Return the current value of an APU register, computing dynamic
        fields (current channel volume, sweep state) on the fly."""
        if not self.power:
            if 0xFF30 <= addr <= 0xFF3F:
                return self.wave_ram[addr - 0xFF30]
            if 0xFF10 <= addr <= 0xFF25:
                return 0xFF
            if addr == 0xFF26:
                return 0x70
            return 0xFF
        mem = self.mmu.memory
        if 0xFF30 <= addr <= 0xFF3F:
            if not self.is_cgb and self.ch3_enabled:
                return self.wave_ram[self.ch3_wave_pos >> 1]
            return self.wave_ram[addr - 0xFF30]
        # NRx2 (envelope) registers read back the last written value, not the live
        # envelope volume — the running volume is internal and not exposed.
        if addr == 0xFF14:
            return mem[0xFF14] & 0xBF | (0x40 if self.ch1_length_enabled else 0)
        if addr == 0xFF19:
            return mem[0xFF19] & 0xBF | (0x40 if self.ch2_length_enabled else 0)
        if addr == 0xFF1E:
            return mem[0xFF1E] & 0xBF | (0x40 if self.ch3_length_enabled else 0)
        if addr == 0xFF23:
            return mem[0xFF23] & 0xBF | (0x40 if self.ch4_length_enabled else 0)
        return mem[addr] | _APU_READ_OR.get(addr, 0)

    def _refresh_nr52(self):
        flags = 0
        if self.ch1_enabled: flags |= 0x01
        if self.ch2_enabled: flags |= 0x02
        if self.ch3_enabled: flags |= 0x04
        if self.ch4_enabled: flags |= 0x08
        master = 0x80 if self.power else 0x00
        self.mmu.memory[0xFF26] = master | 0x70 | flags

    def _power_off(self):
        for addr in range(0xFF10, 0xFF26):
            self.mmu.memory[addr] = 0
        for i in range(16):
            self.wave_ram[i] = 0
            self.mmu.memory[0xFF30 + i] = 0
        self.ch1_enabled = self.ch2_enabled = self.ch3_enabled = self.ch4_enabled = False
        self.ch1_dac = self.ch2_dac = self.ch3_dac = self.ch4_dac = False
        self.vol_left = self.vol_right = 0
        self.pan_left = self.pan_right = 0
        self.ch1_freq = self.ch2_freq = self.ch3_freq = 0
        self.ch1_duty = self.ch2_duty = 0
        self.ch1_duty_step = self.ch2_duty_step = 0
        self.ch1_length_enabled = self.ch2_length_enabled = False
        self.ch3_length_enabled = self.ch4_length_enabled = False
        self.ch1_length = self.ch2_length = self.ch4_length = 0
        self.ch3_length = 0
        self.ch1_volume = self.ch2_volume = self.ch4_volume = 0
        self.ch1_env_timer = self.ch2_env_timer = self.ch4_env_timer = 0
        self.ch1_freq_timer = self.ch2_freq_timer = self.ch3_freq_timer = self.ch4_freq_timer = 4
        self.ch1_sweep_enabled = False
        self.ch1_sweep_shadow = 0
        self.ch1_sweep_timer = 0
        self.ch3_vol_shift = 4
        self.ch3_wave_pos = 0
        self.ch4_lfsr = 0x7FFF
        # On DMG the frame sequencer is free-running even when the APU is
        # off; on CGB it resets on power-off.
        if self.is_cgb:
            self.fs_div = 0
            self.frame_seq_step = 0
            self._fs_remain = self._frame_seq_period(bool(self.mmu.key1 & 0x80))

    @staticmethod
    def _frame_seq_period(double_speed):
        return 16384 if double_speed else 8192

    def _sync_fs_remain(self, div_val, double_speed=False):
        period = self._frame_seq_period(double_speed)
        rem = period - (div_val & (period - 1))
        self._fs_remain = period if rem == 0 else rem

    def _clock_frame_sequencer(self, cpu_cycles, period):
        if cpu_cycles <= 0:
            return
        self._fs_remain -= cpu_cycles
        if self._fs_remain > 0:
            return
        ticks = (-self._fs_remain) // period + 1
        self._fs_remain += ticks * period
        for _ in range(ticks):
            self._frame_seq_tick()

    # ── Channel triggers ─────────────────────────────────────────────

    def _next_fs_clocks_length(self):
        """Return True if the next frame-sequencer tick will clock length counters."""
        return (self.frame_seq_step + 1) & 7 in (0, 2, 4, 6)

    def _trigger_ch1(self):
        if self.ch1_dac:
            self.ch1_enabled = True
        if self.ch1_length == 0:
            self.ch1_length = 63 if self._next_fs_clocks_length() else 64
        period = max((MAX_FREQUENCY - self.ch1_freq) * 4, 4)
        self.ch1_freq_timer = period
        self.ch1_duty_step = 0
        self.ch1_volume = self.ch1_env_initial
        self.ch1_env_timer = self.ch1_env_period if self.ch1_env_period else 8
        self.ch1_sweep_shadow = self.ch1_freq
        self.ch1_sweep_timer = self.ch1_sweep_period if self.ch1_sweep_period else 8
        self.ch1_sweep_enabled = (self.ch1_sweep_period != 0) or (self.ch1_sweep_shift != 0)
        if self.ch1_sweep_shift != 0:
            self._ch1_sweep_calc(apply_result=False)
        self._refresh_nr52()

    def _trigger_ch2(self):
        if self.ch2_dac:
            self.ch2_enabled = True
        if self.ch2_length == 0:
            self.ch2_length = 63 if self._next_fs_clocks_length() else 64
        period = max((MAX_FREQUENCY - self.ch2_freq) * 4, 4)
        self.ch2_freq_timer = period
        self.ch2_duty_step = 0
        self.ch2_volume = self.ch2_env_initial
        self.ch2_env_timer = self.ch2_env_period if self.ch2_env_period else 8
        self._refresh_nr52()

    def _trigger_ch3(self):
        if self.ch3_dac:
            self.ch3_enabled = True
        if self.ch3_length == 0:
            self.ch3_length = 255 if self._next_fs_clocks_length() else 256
        period = max((MAX_FREQUENCY - self.ch3_freq) * 2, 2)
        self.ch3_freq_timer = period
        self.ch3_wave_pos = 0
        self._refresh_nr52()

    def _trigger_ch4(self):
        if self.ch4_dac:
            self.ch4_enabled = True
        if self.ch4_length == 0:
            self.ch4_length = 63 if self._next_fs_clocks_length() else 64
        period = max(_NOISE_DIVISORS[self.ch4_divisor_code] << self.ch4_shift, 8)
        self.ch4_freq_timer = period
        self.ch4_volume = self.ch4_env_initial
        self.ch4_env_timer = self.ch4_env_period if self.ch4_env_period else 8
        self.ch4_lfsr = 0x7FFF
        self._refresh_nr52()

    def _ch1_sweep_calc(self, apply_result):
        shift = self.ch1_sweep_shift
        delta = self.ch1_sweep_shadow >> shift
        new_freq = self.ch1_sweep_shadow - delta if self.ch1_sweep_direction else self.ch1_sweep_shadow + delta
        if new_freq > MAX_FREQUENCY - 1:
            self.ch1_enabled = False
            self.ch1_sweep_enabled = False
            self._refresh_nr52()
            return
        if apply_result and shift != 0:
            self.ch1_sweep_shadow = new_freq
            self.ch1_freq = new_freq
            # Overflow check the second time per spec
            delta2 = new_freq >> shift
            check2 = new_freq - delta2 if self.ch1_sweep_direction else new_freq + delta2
            if check2 > MAX_FREQUENCY - 1:
                self.ch1_enabled = False
                self.ch1_sweep_enabled = False
                self._refresh_nr52()

    # ── Frame sequencer ─────────────────────────────────────────────

    def _frame_seq_tick(self):
        step = self.frame_seq_step
        if step in (0, 2, 4, 6):
            self._clock_length()
        if step in (2, 6):
            self._clock_sweep()
        if step == 7:
            self._clock_envelope()
        self.frame_seq_step = (step + 1) & 7

    def _clock_length(self):
        changed = False
        if self.ch1_length_enabled and self.ch1_length > 0:
            self.ch1_length -= 1
            if self.ch1_length == 0:
                self.ch1_enabled = False
                changed = True
        if self.ch2_length_enabled and self.ch2_length > 0:
            self.ch2_length -= 1
            if self.ch2_length == 0:
                self.ch2_enabled = False
                changed = True
        if self.ch3_length_enabled and self.ch3_length > 0:
            self.ch3_length -= 1
            if self.ch3_length == 0:
                self.ch3_enabled = False
                changed = True
        if self.ch4_length_enabled and self.ch4_length > 0:
            self.ch4_length -= 1
            if self.ch4_length == 0:
                self.ch4_enabled = False
                changed = True
        if changed:
            self._refresh_nr52()

    def _clock_sweep(self):
        if not self.ch1_sweep_enabled:
            return
        self.ch1_sweep_timer -= 1
        if self.ch1_sweep_timer > 0:
            return
        self.ch1_sweep_timer = self.ch1_sweep_period or 8
        if self.ch1_sweep_period != 0:
            self._ch1_sweep_calc(apply_result=True)

    def _clock_envelope(self):
        period = self.ch1_env_period or 8
        self.ch1_env_timer -= 1
        if self.ch1_env_timer <= 0:
            self.ch1_env_timer = period
            v = self.ch1_volume
            if self.ch1_env_direction and v < 15:
                self.ch1_volume = v + 1
            elif not self.ch1_env_direction and v > 0:
                self.ch1_volume = v - 1
        period = self.ch2_env_period or 8
        self.ch2_env_timer -= 1
        if self.ch2_env_timer <= 0:
            self.ch2_env_timer = period
            v = self.ch2_volume
            if self.ch2_env_direction and v < 15:
                self.ch2_volume = v + 1
            elif not self.ch2_env_direction and v > 0:
                self.ch2_volume = v - 1
        period = self.ch4_env_period or 8
        self.ch4_env_timer -= 1
        if self.ch4_env_timer <= 0:
            self.ch4_env_timer = period
            v = self.ch4_volume
            if self.ch4_env_direction and v < 15:
                self.ch4_volume = v + 1
            elif not self.ch4_env_direction and v > 0:
                self.ch4_volume = v - 1

    # ── Per-step channel timers ─────────────────────────────────────

    def _step_channels(self, cycles):
        # Channel 1
        if self.ch1_enabled:
            t = self.ch1_freq_timer - cycles
            if t <= 0:
                period = max((MAX_FREQUENCY - self.ch1_freq) * 4, 4)
                advances = (-t) // period + 1
                self.ch1_duty_step = (self.ch1_duty_step + advances) & 7
                t = period - ((-t) % period)
            self.ch1_freq_timer = t
        # Channel 2
        if self.ch2_enabled:
            t = self.ch2_freq_timer - cycles
            if t <= 0:
                period = max((MAX_FREQUENCY - self.ch2_freq) * 4, 4)
                advances = (-t) // period + 1
                self.ch2_duty_step = (self.ch2_duty_step + advances) & 7
                t = period - ((-t) % period)
            self.ch2_freq_timer = t
        # Channel 3
        if self.ch3_enabled:
            t = self.ch3_freq_timer - cycles
            if t <= 0:
                period = max((MAX_FREQUENCY - self.ch3_freq) * 2, 2)
                advances = (-t) // period + 1
                self.ch3_wave_pos = (self.ch3_wave_pos + advances) & 31
                t = period - ((-t) % period)
            self.ch3_freq_timer = t
        # Channel 4
        if self.ch4_enabled:
            t = self.ch4_freq_timer - cycles
            if t <= 0:
                period = max(_NOISE_DIVISORS[self.ch4_divisor_code] << self.ch4_shift, 8)
                advances = (-t) // period + 1
                lfsr = self.ch4_lfsr
                width = self.ch4_width_mode
                for _ in range(advances):
                    bit = (lfsr & 1) ^ ((lfsr >> 1) & 1)
                    # The LFSR is always 15 bits wide; the feedback bit goes to bit
                    # 14. In width ("7-bit") mode it is ALSO copied into bit 6, which
                    # shortens the period without discarding the upper state.
                    lfsr = (lfsr >> 1) | (bit << 14)
                    if width:
                        lfsr = (lfsr & ~(1 << 6)) | (bit << 6)
                self.ch4_lfsr = lfsr
                t = period - ((-t) % period)
            self.ch4_freq_timer = t

    # ── Sample generation ───────────────────────────────────────────

    def _sample_outputs(self):
        # Each channel returns a signed value in roughly -15..+15 so the
        # mixer stays AC-coupled (silent channels contribute exactly 0).
        if self.ch1_enabled and self.ch1_dac:
            s1 = (_DUTY_PATTERNS[self.ch1_duty][self.ch1_duty_step] * 2 - 1) * self.ch1_volume
        else:
            s1 = 0
        if self.ch2_enabled and self.ch2_dac:
            s2 = (_DUTY_PATTERNS[self.ch2_duty][self.ch2_duty_step] * 2 - 1) * self.ch2_volume
        else:
            s2 = 0
        if self.ch3_enabled and self.ch3_dac and self.ch3_vol_shift < 4:
            byte = self.wave_ram[self.ch3_wave_pos >> 1]
            nib = (byte >> 4) if (self.ch3_wave_pos & 1) == 0 else (byte & 0x0F)
            s3 = (nib - 8) >> self.ch3_vol_shift
        else:
            s3 = 0
        if self.ch4_enabled and self.ch4_dac:
            s4 = ((~self.ch4_lfsr & 1) * 2 - 1) * self.ch4_volume
        else:
            s4 = 0
        return s1, s2, s3, s4

    def _mix_sample(self):
        s1, s2, s3, s4 = self._sample_outputs()
        pl = self.pan_left
        pr = self.pan_right
        left = 0
        right = 0
        if pl & 0x10: left += s1
        if pl & 0x20: left += s2
        if pl & 0x40: left += s3
        if pl & 0x80: left += s4
        if pr & 0x10: right += s1
        if pr & 0x20: right += s2
        if pr & 0x40: right += s3
        if pr & 0x80: right += s4
        # left/right ranges across the four channels: roughly +-(15+15+8+15) = +-53.
        # Master vol+1 is 1..8; scale factor ~70 keeps int16 headroom.
        left = left * (self.vol_left + 1) * AUDIO_SCALE_FACTOR
        right = right * (self.vol_right + 1) * AUDIO_SCALE_FACTOR
        if left > INT16_MAX: left = INT16_MAX
        elif left < INT16_MIN: left = INT16_MIN
        if right > INT16_MAX: right = INT16_MAX
        elif right < INT16_MIN: right = INT16_MIN
        lv = left & 0xFFFF
        rv = right & 0xFFFF
        buf = self.buffer
        buf.append(lv & 0xFF)
        buf.append(lv >> 8)
        buf.append(rv & 0xFF)
        buf.append(rv >> 8)

    # ── Main step ───────────────────────────────────────────────────

    def step(self, cycles, div_old=None, div_new=None, double_speed=False):
        sn = self.sample_num
        sd = self.sample_den
        if div_old is not None:
            cpu_cycles = div_new - div_old
            self.fs_div = div_new
        else:
            cpu_cycles = cycles
            self.fs_div += cycles
        # DMG keeps the sequencer running while the APU is off; CGB does not.
        if self.power or not self.is_cgb:
            self._clock_frame_sequencer(
                cpu_cycles, 16384 if double_speed else 8192)

        if not self.power:
            # APU off: still produce silence so the audio buffer keeps flowing.
            self.sample_accum += cycles * sd
            n = self.sample_accum // sn
            if n:
                self.sample_accum -= n * sn
                self.buffer.extend(b'\x00\x00\x00\x00' * n)
            return

        active = self.ch1_enabled or self.ch2_enabled or self.ch3_enabled or self.ch4_enabled
        if active:
            self._step_channels(cycles)

        # Sample emission
        self.sample_accum += cycles * sd
        if active:
            while self.sample_accum >= sn:
                self.sample_accum -= sn
                self._mix_sample()
        else:
            n = self.sample_accum // sn
            if n:
                self.sample_accum -= n * sn
                self.buffer.extend(b'\x00\x00\x00\x00' * n)

    def drain(self):
        """Returns and clears the accumulated PCM bytes (signed-16 stereo LE)."""
        buf = self.buffer
        if len(buf) > self.SOFT_BUFFER_CAP:
            del buf[:len(buf) - self.SOFT_BUFFER_CAP]
        data = bytes(buf)
        del buf[:]
        return data

    def peek_samples_for_cycles(self, cycles):
        """Return how many stereo samples *cycles* base-clock dots would emit."""
        accum = self.sample_accum + cycles * self.sample_den
        sn = self.sample_num
        count = 0
        while accum >= sn:
            accum -= sn
            count += 1
        return count


# Each emulated video frame is CYCLES_PER_FRAME base-clock dots.  At 44.1 kHz the
# fractional sample timer alternates 738 and 739 stereo samples per frame.
APU_BYTES_PER_STEREO_SAMPLE = 4
APU_MIN_SAMPLES_PER_FRAME = (CYCLES_PER_FRAME * APU.SAMPLE_RATE) // APU.CPU_CLOCK
APU_MAX_SAMPLES_PER_FRAME = APU_MIN_SAMPLES_PER_FRAME + (
    1 if (CYCLES_PER_FRAME * APU.SAMPLE_RATE) % APU.CPU_CLOCK else 0
)


# ── Menu system ──────────────────────────────────────────────────────

MENU_W = 640
MENU_H = 480
MENU_BG = (8, 24, 32)      # DMG darkest: (8, 24, 32)
MENU_FG = (136, 192, 112)  # DMG light-mid: (136, 192, 112)
MENU_HI = (224, 248, 208)  # DMG lightest: (224, 248, 208)
MENU_DIM = (52, 104, 86)   # DMG dark-mid: (52, 104, 86)
MENU_FOOTER_Y = MENU_H - 22
MENU_SECONDARY_Y = MENU_H - 48
MENU_RULE_Y = MENU_H - 62

_font_cache = {}
def get_font(size, bold=True):
    key = (size, bold)
    if key not in _font_cache:
        try:
            _font_cache[key] = pygame.font.SysFont("Courier New", size, bold=bold)
        except (pygame.error, OSError):
            _font_cache[key] = pygame.font.Font(None, size)
    return _font_cache[key]


def _fit_text(font, text, max_width):
    """Ellipsize `text` so it fits within max_width pixels."""
    text = str(text)
    if max_width <= 0:
        return ''
    if font.size(text)[0] <= max_width:
        return text
    ell = '...'
    if font.size(ell)[0] >= max_width:
        return ell
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.size(text[:mid] + ell)[0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ell


def _decorate_cyclic_setting(text, selected):
    """Wrap the value of a selected cyclic setting in < > so Left/Right is obvious."""
    if not selected or ': ' not in text:
        return text
    name, value = text.split(': ', 1)
    return f"{name}: < {value} >"


def _sync_list_scroll(cursor, scroll, capacity, n_items):
    """Keep ``cursor`` inside the visible window ``[scroll, scroll + capacity)``."""
    n = max(int(n_items), 1)
    cap = max(1, min(int(capacity), n))
    cur = int(cursor) % n
    scr = max(0, int(scroll))
    if cur < scr:
        scr = cur
    elif cur >= scr + cap:
        scr = cur - cap + 1
    scr = max(0, min(scr, max(0, n - cap)))
    return scr, cap


def _overlay_layout(w, h, n_items, has_hint=True, has_status=False):
    """Pause-panel metrics that keep title, rows, hint, and status inside the window."""
    n = max(int(n_items), 1)
    margin = 8
    panel_w = max(220, w - margin * 2)
    if h >= 500:
        title_size, base_item, hint_size = 28, 24, 16
    elif h >= 400:
        title_size, base_item, hint_size = 22, 18, 14
    elif h >= 320:
        title_size, base_item, hint_size = 18, 15, 12
    else:
        title_size, base_item, hint_size = 16, 13, 11
    if n >= 9:
        base_item = min(base_item, 16 if h >= 400 else 13)
    title_band = title_size + 14
    hint_band = (hint_size + 14) if has_hint else 8
    status_band = (hint_size + 10) if has_status else 0
    max_panel_h = max(48, h - margin * 2)
    item_h = min(36, max(14, (max_panel_h - title_band - hint_band - status_band) // n))
    item_size = min(base_item, max(11, item_h - 3))
    while title_band + n * item_h + hint_band + status_band > max_panel_h and item_h > 13:
        item_h -= 1
        item_size = min(item_size, max(11, item_h - 3))
    panel_h = min(max_panel_h, title_band + n * item_h + hint_band + status_band)
    px = (w - panel_w) // 2
    py = max(margin, (h - panel_h) // 2)
    return {
        'panel_w': panel_w,
        'panel_h': panel_h,
        'px': px,
        'py': py,
        'title_size': title_size,
        'item_size': item_size,
        'hint_size': hint_size,
        'title_band': title_band,
        'hint_band': hint_band,
        'status_band': status_band,
        'item_h': item_h,
        'max_panel_h': max_panel_h,
    }


def _overlay_scroll_capacity(w, h, has_hint=True, has_status=False):
    """Maximum menu rows that fit in an overlay at minimum row height."""
    L = _overlay_layout(w, h, 1, has_hint=has_hint, has_status=has_status)
    fixed = L['title_band'] + L['hint_band'] + L['status_band']
    return max(1, (L['max_panel_h'] - fixed) // 13)


def _read_rom_system_tag(rom_path):
    """Return ``CGB`` or ``DMG`` from the ROM header CGB flag byte."""
    try:
        with open(rom_path, 'rb') as f:
            f.seek(0x0143)
            return "CGB" if f.read(1)[0] & 0x80 else "DMG"
    except OSError:
        return "???"


def _blit_selection_bar(surface, x, y, w, h):
    """Highlight the selected menu row."""
    if w < 2 or h < 2:
        return
    bar = pygame.Surface((w, h))
    bar.fill(MENU_DIM)
    bar.set_alpha(120)
    surface.blit(bar, (x, y))
    pygame.draw.rect(surface, MENU_HI, (x, y, w, h), 1)


def _set_window_icon():
    """Load gbclogo.png as the window icon if pygame is ready."""
    if not pygame:
        return
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    path = os.path.join(here, "gbclogo.png")
    if not os.path.isfile(path):
        return
    try:
        raw = pygame.image.load(path)
        pygame.display.set_icon(pygame.transform.smoothscale(raw, (32, 32)))
    except (OSError, pygame.error):
        return


class EmulatorMenu:
    def __init__(self, bootrom_path=None):
        self.selected = 0
        self.roms = []
        self.rom_cursor = 0
        self.rom_scroll = 0
        self.max_visible = 10
        self.settings_items = ["Window Scale: 4x"]
        self.settings_cursor = 0
        self.exit_cursor = 0
        self.window_scale = 4
        self.bootrom_path = bootrom_path
        self.fps_limit_idx = 0
        self.audio_idx = 0
        self.volume_idx = 4
        self.palette_idx = 0
        self.filter_idx = 0
        self.shader_idx = 0
        self.status_line = ""
        self.status_ttl = 0
        self._menu_hits = []
        self._stick_nav = _StickNav()
        self._blit_scale = 1
        self._blit_ox = 0
        self._blit_oy = 0
        # Load persisted settings from config file
        cfg = _load_config()
        self.window_scale = _clamp_choice(cfg.get("window_scale", self.window_scale),
                                          _WINDOW_SCALES, 4)
        self.fullscreen = bool(cfg.get("fullscreen", False))
        self.fps_limit_idx = _opt_index(FPS_LIMIT_OPTIONS, cfg.get("fps_limit", 59.73), 0)
        self.audio_idx = _opt_index(AUDIO_OPTIONS, cfg.get("audio_enabled", True), 0)
        self.volume_idx = _opt_index(VOLUME_OPTIONS, cfg.get("volume", 1.0), 4)
        self.palette_idx = _clamp_index(cfg.get("palette", 0), len(PALETTE_LIST), 0)
        self.filter_idx = _opt_index(FILTER_OPTIONS, cfg.get("smooth_scale", False), 0)
        self.shader_idx = _clamp_index(cfg.get("shader", 0), len(SHADER_LIST), 0)
        self.wasd_enabled = bool(cfg.get("wasd_enabled", True))
        last = cfg.get("last_rom")
        self.last_rom = last if isinstance(last, str) and os.path.isfile(last) else None
        self.key_bindings, self.turbo_bindings = _rebuild_input_maps(
            cfg.get("key_bindings"), self.wasd_enabled, cfg.get("turbo_bindings"))
        self.controls_cursor = 0
        self.controls_scroll = 0
        self.controls_capture = None  # button key being remapped, or None
        self._sync_settings_items()
        if pygame is None:
            print("=" * 55)
            print("  ERROR: pygame is required for the emulator menu.")
            print("  Install it with:  pip install pygame numpy")
            print("=" * 55)
            sys.exit(1)
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        try:
            pygame.init()
        except pygame.error:
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            pygame.mixer.pre_init(44100, -16, 2, 1024)
            pygame.init()
            logging.warning("Audio init failed — running silent (dummy audio driver).")
        _init_joysticks()
        self.key_bindings, self.turbo_bindings = _rebuild_input_maps(
            self.key_bindings, self.wasd_enabled, self.turbo_bindings)
        self.screen = pygame.Surface((MENU_W, MENU_H))
        self._apply_display_mode()
        pygame.display.set_caption("Python GBC Emulator")
        self.logo = None
        self._load_logo()
        self._scan_roms()
        self._select_last_rom()

    def _main_items(self):
        items = []
        if self.last_rom and os.path.isfile(self.last_rom):
            items.append("Continue")
        items.extend(["Load ROM", "Settings", "Exit to OS"])
        return items

    def _apply_display_mode(self):
        self._display = _open_display((MENU_W, MENU_H), self.fullscreen)
        _set_window_icon()
        self._stick_nav.reset()

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._apply_display_mode()
        if not _save_config({'fullscreen': self.fullscreen}):
            self._status("Could not save settings")
        else:
            self._status("Fullscreen" if self.fullscreen else "Windowed")

    def _present(self):
        self._blit_scale, self._blit_ox, self._blit_oy, _, _ = _present_integer_scale(
            self._display, self.screen, MENU_BG)
        pygame.display.flip()

    def _canvas_mouse(self, pos=None):
        if pos is None:
            pos = pygame.mouse.get_pos()
        return _map_mouse_to_canvas(
            pos, self._blit_scale, self._blit_ox, self._blit_oy, MENU_W, MENU_H)

    def _select_last_rom(self):
        if not self.last_rom or not self.roms:
            return
        target = os.path.realpath(self.last_rom)
        for i, path in enumerate(self.roms):
            if os.path.realpath(path) == target:
                self.rom_cursor = i
                self.rom_scroll, _ = _sync_list_scroll(
                    i, self.rom_scroll, self.max_visible, len(self.roms))
                return

    def _remember_rom(self, path):
        try:
            self.last_rom = os.path.realpath(path)
        except OSError:
            self.last_rom = path
        _save_config({'last_rom': self.last_rom})

    def _scan_roms(self):
        self._rom_info_cache = {}
        self._rom_tag_cache = {}
        seen = set()
        self.roms = []
        if not os.path.isdir("roms") and not os.path.isdir("rom"):
            try:
                os.makedirs("roms", exist_ok=True)
            except OSError:
                pass
        for base in [".", ".."]:
            if not os.path.isdir(base):
                continue
            for f in sorted(os.listdir(base)):
                fp = os.path.join(base, f)
                if os.path.isfile(fp) and f.lower().endswith(('.gb', '.gbc')):
                    rp = os.path.realpath(fp)
                    if rp not in seen:
                        seen.add(rp)
                        self.roms.append(fp)
            for sub in sorted(os.listdir(base)):
                d1 = os.path.join(base, sub)
                if not os.path.isdir(d1):
                    continue
                try:
                    for f in sorted(os.listdir(d1)):
                        fp = os.path.join(d1, f)
                        if os.path.isfile(fp) and f.lower().endswith(('.gb', '.gbc')):
                            rp = os.path.realpath(fp)
                            if rp not in seen:
                                seen.add(rp)
                                self.roms.append(fp)
                except OSError:
                    pass
                try:
                    for sub2 in sorted(os.listdir(d1)):
                        d2 = os.path.join(d1, sub2)
                        if not os.path.isdir(d2):
                            continue
                        try:
                            for f in sorted(os.listdir(d2)):
                                fp = os.path.join(d2, f)
                                if os.path.isfile(fp) and f.lower().endswith(('.gb', '.gbc')):
                                    rp = os.path.realpath(fp)
                                    if rp not in seen:
                                        seen.add(rp)
                                        self.roms.append(fp)
                        except OSError:
                            pass
                except OSError:
                    pass

    def _sync_settings_items(self):
        self.settings_items = [
            f"Window Scale: {self.window_scale}x",
            f"Display: {DISPLAY_OPTIONS[1][0] if self.fullscreen else DISPLAY_OPTIONS[0][0]}",
            f"Frame Rate: {FPS_LIMIT_OPTIONS[self.fps_limit_idx][0]}",
            f"Audio: {AUDIO_OPTIONS[self.audio_idx][0]}",
            f"Volume: {VOLUME_OPTIONS[self.volume_idx][0]}",
            f"Palette: {PALETTE_LIST[self.palette_idx][0]}",
            f"Filter: {FILTER_OPTIONS[self.filter_idx][0]}",
            f"Shader: {SHADER_LIST[self.shader_idx][0]}",
            "Controls...",
        ]

    def _persist_menu_settings(self):
        if not _save_config(dict(
            window_scale=self.window_scale,
            fullscreen=self.fullscreen,
            fps_limit=FPS_LIMIT_OPTIONS[self.fps_limit_idx][1],
            audio_enabled=AUDIO_OPTIONS[self.audio_idx][1],
            volume=VOLUME_OPTIONS[self.volume_idx][1],
            palette=self.palette_idx,
            smooth_scale=FILTER_OPTIONS[self.filter_idx][1],
            shader=self.shader_idx,
            wasd_enabled=self.wasd_enabled,
            key_bindings=self.key_bindings,
            turbo_bindings=self.turbo_bindings,
            last_rom=self.last_rom,
        )):
            self._status("Could not save settings")

    def _persist_controls(self):
        self.key_bindings, self.turbo_bindings = _rebuild_input_maps(
            self.key_bindings, self.wasd_enabled, self.turbo_bindings)
        if not _save_config({
            'wasd_enabled': self.wasd_enabled,
            'key_bindings': self.key_bindings,
            'turbo_bindings': self.turbo_bindings,
        }):
            self._status("Could not save settings")

    def _controls_items(self):
        items = [
            f"{JOYPAD_BUTTON_LABELS[i]}: {_binding_label(self.key_bindings, key)}"
            for i, key in enumerate(JOYPAD_BUTTON_KEYS)
        ]
        items.append(f"WASD as D-Pad: {'On' if self.wasd_enabled else 'Off'}")
        items.append(f"Turbo A: {_binding_label(self.turbo_bindings, 'turbo_a')}")
        items.append(f"Turbo B: {_binding_label(self.turbo_bindings, 'turbo_b')}")
        items.append("Reset to Default")
        items.append(_connected_gamepad_label())
        return items

    def _status(self, msg):
        self.status_line = msg
        self.status_ttl = 120

    def _centre_text(self, text, y, colour=MENU_FG, size=28, shadow=False):
        f = get_font(size)
        s = f.render(text, True, colour)
        x = (MENU_W - s.get_width()) // 2
        if shadow:
            shadow_color = (0, 0, 0)
            s_shadow = f.render(text, True, shadow_color)
            self.screen.blit(s_shadow, (x + 2, y + 2))
        self.screen.blit(s, (x, y))
        return s.get_width()

    def _draw_chrome(self, primary, secondary=None):
        """Footer plus optional secondary hint. Status toasts replace the hint."""
        pygame.draw.line(self.screen, MENU_DIM, (48, MENU_RULE_Y), (MENU_W - 48, MENU_RULE_Y), 1)
        if self.status_ttl and self.status_line:
            self._centre_text(self.status_line, MENU_SECONDARY_Y, MENU_HI, 18)
        elif secondary:
            self._centre_text(secondary, MENU_SECONDARY_Y, MENU_DIM, 15)
        self._centre_text(primary, MENU_FOOTER_Y, MENU_DIM, 16)

    def _draw_menu(self, items, cursor, start_y, gap, size=28, scroll=0, max_visible=None):
        f = get_font(size)
        n = len(items)
        self._menu_hits = []
        if max_visible is not None and n > max_visible:
            scroll, cap = _sync_list_scroll(cursor, scroll, max_visible, n)
            visible = items[scroll:scroll + cap]
            display_cursor = cursor - scroll
        else:
            visible = items
            display_cursor = cursor
            scroll = 0
            cap = n
        max_w = 0
        for item in visible:
            max_w = max(max_w, f.size(item)[0])
        bar_w = min(MENU_W - 80, max(220, max_w + 72))
        bar_h = size + 10
        bar_x = (MENU_W - bar_w) // 2
        if scroll > 0:
            self._centre_text("^ more", start_y - 16, MENU_DIM, 14)
        for i, item in enumerate(visible):
            y = start_y + i * gap
            idx = scroll + i
            self._menu_hits.append((bar_x, y - 4, bar_w, bar_h, idx))
            if i == display_cursor:
                _blit_selection_bar(self.screen, bar_x, y - 4, bar_w, bar_h)
            colour = MENU_HI if i == display_cursor else MENU_FG
            w = self._centre_text(item, y, colour, size, shadow=(i == display_cursor))
            if i == display_cursor:
                cursor_surf = f.render(">", True, MENU_HI)
                cursor_x = (MENU_W - w) // 2 - cursor_surf.get_width() - 10
                self.screen.blit(cursor_surf, (cursor_x, y))
        if scroll + cap < n:
            self._centre_text("v more", start_y + cap * gap - 6, MENU_DIM, 14)
        return scroll

    def _settings_id(self, cursor=None):
        if cursor is None:
            cursor = self.settings_cursor
        if 0 <= cursor < len(SETTINGS_ROW_IDS):
            return SETTINGS_ROW_IDS[cursor]
        return None

    def _cycle_menu_setting(self, cursor, direction=1):
        """Cycle a main-menu setting forward (1) or backward (-1)."""
        sid = self._settings_id(cursor)
        if sid == 'scale':
            scales = list(_WINDOW_SCALES)
            try:
                idx = scales.index(self.window_scale)
            except ValueError:
                idx = 2
            self.window_scale = scales[(idx + direction) % len(scales)]
        elif sid == 'display':
            self.fullscreen = not self.fullscreen
            self._apply_display_mode()
        elif sid == 'fps':
            self.fps_limit_idx = (self.fps_limit_idx + direction) % len(FPS_LIMIT_OPTIONS)
        elif sid == 'audio':
            self.audio_idx = (self.audio_idx + direction) % len(AUDIO_OPTIONS)
        elif sid == 'volume':
            self.volume_idx = (self.volume_idx + direction) % len(VOLUME_OPTIONS)
        elif sid == 'palette':
            self.palette_idx = (self.palette_idx + direction) % len(PALETTE_LIST)
        elif sid == 'filter':
            self.filter_idx = (self.filter_idx + direction) % len(FILTER_OPTIONS)
        elif sid == 'shader':
            self.shader_idx = (self.shader_idx + direction) % len(SHADER_LIST)
        else:
            return
        self._sync_settings_items()
        self._persist_menu_settings()

    def run(self):
        clock = pygame.time.Clock()
        pygame.key.set_repeat(200, 50)
        page = "main"
        while True:
            page = self._handle(page)
            self.status_ttl = max(0, self.status_ttl - 1)

            self.screen.fill(MENU_BG)
            if page == "main":
                self._render_main()
            elif page == "load_rom":
                self._render_load_rom()
            elif page == "settings":
                self._render_settings()
            elif page == "controls":
                self._render_controls()
            elif page == "confirm_exit":
                self._render_confirm_exit()
            self._present()
            clock.tick(60)

    def _load_logo(self):
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = os.getcwd()
        for candidate in ("gbclogo.png", os.path.join(here, "gbclogo.png")):
            if os.path.isfile(candidate):
                try:
                    raw = pygame.image.load(candidate).convert_alpha()
                except (OSError, pygame.error):
                    continue
                scaled = pygame.transform.smoothscale(raw, (112, 112))
                if np is not None:
                    arr = pygame.surfarray.array3d(scaled).transpose(1, 0, 2)
                    mask = (arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220)
                    arr[mask] = MENU_BG
                    new_surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
                else:
                    new_surf = scaled
                block = pygame.Surface((132, 132))
                block.fill(MENU_BG)
                block.blit(new_surf, (10, 10))
                self.logo = block
                try:
                    pygame.display.set_icon(pygame.transform.smoothscale(raw, (32, 32)))
                except pygame.error:
                    pass
                return
        self.logo = None

    def _restore_after_game(self, gb=None):
        if gb is not None:
            si = gb._set_idx
            self.window_scale = WINDOW_SCALE_OPTIONS[si['scale']][1]
            self.fps_limit_idx = si['fps']
            self.audio_idx = si['audio']
            self.volume_idx = si['volume']
            self.palette_idx = si['palette']
            self.filter_idx = si['filter']
            self.shader_idx = si['shader']
            self.wasd_enabled = gb.wasd_enabled
            self.fullscreen = getattr(gb, 'fullscreen', self.fullscreen)
            self.key_bindings = gb.key_bindings
            self.turbo_bindings = getattr(gb, 'turbo_bindings', self.turbo_bindings)
            self.key_bindings, self.turbo_bindings = _rebuild_input_maps(
                self.key_bindings, self.wasd_enabled, self.turbo_bindings)
            self._sync_settings_items()
        self.screen = pygame.Surface((MENU_W, MENU_H))
        self._apply_display_mode()
        pygame.display.set_caption("Python GBC Emulator")
        pygame.event.clear()
        pygame.key.set_repeat(200, 50)
        _init_joysticks()
        self._stick_nav.reset()

    def _launch_rom(self, path):
        if not path or not os.path.isfile(path):
            self._status(f"ROM not found: {os.path.basename(path or '')}")
            return "main"
        self._remember_rom(path)
        self._status(f"Loading: {os.path.basename(path)}")
        self._present()
        gb = None
        try:
            gb = GameBoy(
                path,
                window_scale=self.window_scale,
                fps_limit=FPS_LIMIT_OPTIONS[self.fps_limit_idx][1],
                audio_enabled=AUDIO_OPTIONS[self.audio_idx][1],
                volume=VOLUME_OPTIONS[self.volume_idx][1],
                palette=PALETTE_LIST[self.palette_idx][1],
                smooth_scale=FILTER_OPTIONS[self.filter_idx][1],
                shader=SHADER_LIST[self.shader_idx][1],
                bootrom_path=self.bootrom_path,
                fullscreen=self.fullscreen,
            )
            gb.run()
        except FileNotFoundError:
            self._status(f"ROM not found: {os.path.basename(path)}")
        except (OSError, ValueError, pygame.error, RuntimeError, MemoryError) as e:
            self._status(f"Error loading ROM: {e}")
        self._restore_after_game(gb)
        return "main"

    def _capture_binding(self, name):
        target = self.controls_capture
        if target in TURBO_BUTTON_KEYS:
            ok = _assign_binding_key(
                self.turbo_bindings, target, name, other_maps=[self.key_bindings])
        else:
            ok = _assign_binding_key(
                self.key_bindings, target, name, other_maps=[self.turbo_bindings])
        if ok:
            self._persist_controls()
            label = target.replace('_', ' ').title()
            self._status(f"{label} -> {_key_display_name(name)}")
        else:
            self._status(_RESERVED_KEY_MSG)
        self.controls_capture = None
        pygame.key.set_repeat(200, 50)

    def _handle(self, page):
        actions = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            plug = _gamepad_hotplug_message(event)
            if event.type in (getattr(pygame, 'JOYDEVICEADDED', -1),
                              getattr(pygame, 'JOYDEVICEREMOVED', -2)):
                _init_joysticks()
                if plug:
                    self._status(plug)
                continue
            if page == "controls" and self.controls_capture is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.controls_capture = None
                        pygame.key.set_repeat(200, 50)
                    elif event.key != pygame.K_F11:
                        self._capture_binding(_normalize_key_name(pygame.key.name(event.key)))
                    else:
                        actions.append('toggle_fullscreen')
                continue
            if event.type == pygame.KEYDOWN:
                actions.append(event.key)
            elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION):
                ga = _gamepad_menu_action(event)
                if ga:
                    actions.append(ga)
            elif event.type == pygame.MOUSEMOTION:
                idx = _hit_list_index(self._menu_hits, self._canvas_mouse(event.pos))
                if idx is not None:
                    self._hover_select(page, idx)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                idx = _hit_list_index(self._menu_hits, self._canvas_mouse(event.pos))
                if idx is not None:
                    if self._hover_select(page, idx) is False:
                        actions.append('select')
                    else:
                        actions.append('select')
            elif event.type == getattr(pygame, 'MOUSEWHEEL', -3):
                if getattr(event, 'y', 0) > 0:
                    actions.append('up')
                elif getattr(event, 'y', 0) < 0:
                    actions.append('down')
        if self.controls_capture is None:
            stick = self._stick_nav.update(*_read_analog_menu_dirs(), pygame.time.get_ticks())
            if stick:
                actions.append(stick)
        for raw in actions:
            page = self._dispatch(page, _menu_nav_action(raw))
        return page

    def _hover_select(self, page, idx):
        """Move the page cursor to ``idx``. Returns True if it changed."""
        if page == "main":
            items = self._main_items()
            if 0 <= idx < len(items) and idx != self.selected:
                self.selected = idx
                return True
        elif page == "load_rom" and self.roms:
            if 0 <= idx < len(self.roms) and idx != self.rom_cursor:
                self.rom_cursor = idx
                self.rom_scroll, _ = _sync_list_scroll(
                    idx, self.rom_scroll, self.max_visible, len(self.roms))
                return True
        elif page == "settings":
            if 0 <= idx < len(self.settings_items) and idx != self.settings_cursor:
                self.settings_cursor = idx
                return True
        elif page == "controls":
            items = self._controls_items()
            if 0 <= idx < len(items) and idx != self.controls_cursor:
                self.controls_cursor = idx
                self.controls_scroll, _ = _sync_list_scroll(
                    idx, self.controls_scroll, 8, len(items))
                return True
        elif page == "confirm_exit":
            if idx in (0, 1) and idx != self.exit_cursor:
                self.exit_cursor = idx
                return True
        return False

    def _dispatch(self, page, action):
        if action == 'toggle_fullscreen' or action == pygame.K_F11:
            self._toggle_fullscreen()
            return page
        if page == "main":
            items = self._main_items()
            n = len(items)
            if self.selected >= n:
                self.selected = 0
            if action == pygame.K_UP or action == 'up':
                self.selected = (self.selected - 1) % n
            elif action == pygame.K_DOWN or action == 'down':
                self.selected = (self.selected + 1) % n
            elif action == pygame.K_RETURN or action == 'select':
                choice = items[self.selected]
                if choice == "Continue":
                    return self._launch_rom(self.last_rom)
                if choice == "Load ROM":
                    self._scan_roms()
                    self._select_last_rom()
                    if not self.roms:
                        self.rom_cursor = 0
                        self.rom_scroll = 0
                    return "load_rom"
                if choice == "Settings":
                    self._sync_settings_items()
                    self.settings_cursor = 0
                    return "settings"
                if choice == "Exit to OS":
                    self.exit_cursor = 0
                    return "confirm_exit"
            elif action == pygame.K_ESCAPE or action == 'back':
                self.exit_cursor = 0
                return "confirm_exit"
            return page
        if page == "load_rom":
            n = len(self.roms)
            if action in (pygame.K_UP, 'up', pygame.K_DOWN, 'down',
                          'home', 'end', 'pageup', 'pagedown'):
                nav = {pygame.K_UP: 'up', pygame.K_DOWN: 'down'}.get(action, action)
                self.rom_cursor = _move_list_cursor(
                    self.rom_cursor, n, nav, self.max_visible)
                self.rom_scroll, _ = _sync_list_scroll(
                    self.rom_cursor, self.rom_scroll, self.max_visible, max(n, 1))
            elif action == pygame.K_RETURN or action == 'select':
                if self.roms:
                    return self._launch_rom(self.roms[self.rom_cursor])
            elif action == pygame.K_ESCAPE or action == 'back':
                return "main"
            elif action == pygame.K_F5:
                prev = self.roms[self.rom_cursor] if self.roms else None
                self._scan_roms()
                if prev and prev in self.roms:
                    self.rom_cursor = self.roms.index(prev)
                else:
                    self._select_last_rom()
                    if not self.roms:
                        self.rom_cursor = 0
                self.rom_scroll, _ = _sync_list_scroll(
                    self.rom_cursor, self.rom_scroll, self.max_visible, max(len(self.roms), 1))
                self._status("ROM list refreshed")
            return page
        if page == "settings":
            n = len(self.settings_items)
            if action == pygame.K_UP or action == 'up':
                self.settings_cursor = (self.settings_cursor - 1) % n
            elif action == pygame.K_DOWN or action == 'down':
                self.settings_cursor = (self.settings_cursor + 1) % n
            elif action == pygame.K_RETURN or action == 'select':
                if self._settings_id() == 'controls':
                    self.controls_cursor = 0
                    self.controls_scroll = 0
                    self.controls_capture = None
                    return "controls"
                self._cycle_menu_setting(self.settings_cursor, 1)
            elif action in (pygame.K_RIGHT, 'right'):
                if self._settings_id() != 'controls':
                    self._cycle_menu_setting(self.settings_cursor, 1)
            elif action in (pygame.K_LEFT, 'left'):
                if self._settings_id() != 'controls':
                    self._cycle_menu_setting(self.settings_cursor, -1)
            elif action == pygame.K_ESCAPE or action == 'back':
                return "main"
            return page
        if page == "controls":
            items = self._controls_items()
            n = len(items)
            if action == pygame.K_UP or action == 'up':
                self.controls_cursor = _advance_controls_cursor(self.controls_cursor, -1, n)
                self.controls_scroll, _ = _sync_list_scroll(
                    self.controls_cursor, self.controls_scroll, 8, n)
            elif action == pygame.K_DOWN or action == 'down':
                self.controls_cursor = _advance_controls_cursor(self.controls_cursor, 1, n)
                self.controls_scroll, _ = _sync_list_scroll(
                    self.controls_cursor, self.controls_scroll, 8, n)
            elif action in (pygame.K_LEFT, pygame.K_RIGHT, 'left', 'right'):
                if self.controls_cursor == CONTROLS_WASD_ROW:
                    self.wasd_enabled = not self.wasd_enabled
                    self.key_bindings = _sanitize_key_bindings(
                        self.key_bindings, self.wasd_enabled)
                    self._persist_controls()
            elif action == pygame.K_RETURN or action == 'select':
                if self.controls_cursor < CONTROLS_WASD_ROW:
                    self.controls_capture = JOYPAD_BUTTON_KEYS[self.controls_cursor]
                    pygame.key.set_repeat()
                elif self.controls_cursor == CONTROLS_WASD_ROW:
                    self.wasd_enabled = not self.wasd_enabled
                    self.key_bindings = _sanitize_key_bindings(
                        self.key_bindings, self.wasd_enabled)
                    self._persist_controls()
                elif self.controls_cursor == CONTROLS_TURBO_A_ROW:
                    self.controls_capture = 'turbo_a'
                    pygame.key.set_repeat()
                elif self.controls_cursor == CONTROLS_TURBO_B_ROW:
                    self.controls_capture = 'turbo_b'
                    pygame.key.set_repeat()
                elif self.controls_cursor == CONTROLS_RESET_ROW:
                    self.wasd_enabled = True
                    self.key_bindings = _default_key_bindings(True)
                    self.turbo_bindings = {k: list(v) for k, v in DEFAULT_TURBO_BINDINGS.items()}
                    self._persist_controls()
                    self._status("Controls reset to default")
                elif self.controls_cursor == CONTROLS_GAMEPAD_ROW:
                    self._status(_connected_gamepad_label())
            elif action == pygame.K_ESCAPE or action == 'back':
                self.controls_capture = None
                return "settings"
            return page
        if page == "confirm_exit":
            if action in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                          'up', 'down', 'left', 'right'):
                self.exit_cursor ^= 1
            elif action == pygame.K_RETURN or action == 'select':
                if self.exit_cursor == 1:
                    pygame.quit()
                    sys.exit()
                return "main"
            elif action == pygame.K_ESCAPE or action == 'back':
                return "main"
        return page

    def _render_main(self):
        if self.logo is not None:
            lx = (MENU_W - self.logo.get_width()) // 2
            self.screen.blit(self.logo, (lx, 10))
            title_y = 10 + self.logo.get_height() + 6
        else:
            title_y = 72
        self._centre_text("Python GBC Emulator", title_y, MENU_HI, 36, shadow=True)
        self._centre_text(f"v{__version__}", title_y + 42, MENU_DIM, 18)
        items = self._main_items()
        if self.selected >= len(items):
            self.selected = 0
        gap = 42 if len(items) > 3 else 48
        self._draw_menu(items, self.selected, title_y + 80, gap, size=26)
        secondary = None
        if items and items[self.selected] == "Continue" and self.last_rom:
            secondary = os.path.basename(self.last_rom)
        self._draw_chrome(
            "Enter: Select   F11: Fullscreen   Esc: Quit",
            secondary=secondary or "Up/Down or click to move")

    def _render_load_rom(self):
        self._centre_text("Select ROM", 22, MENU_HI, 32, shadow=True)
        self._menu_hits = []
        if not self.roms:
            self._centre_text("No .gb / .gbc files found", 160, MENU_DIM, 24)
            self._centre_text("Put ROM files in the roms folder", 210, MENU_DIM, 20)
            self._centre_text("(or this directory / its parent)", 240, MENU_DIM, 20)
            self._centre_text("Press F5 to rescan", 300, MENU_HI, 22)
            self._draw_chrome("F5: Refresh   Esc: Back")
            return
        list_y = 68
        row_h = 32
        visible = self.roms[self.rom_scroll:self.rom_scroll + self.max_visible]
        name_font = get_font(22)
        if self.rom_scroll > 0:
            self._centre_text("^ more", list_y - 18, MENU_DIM, 14)
        tag_font = get_font(14)
        tag_cache = getattr(self, '_rom_tag_cache', None)
        if tag_cache is None:
            tag_cache = {}
            self._rom_tag_cache = tag_cache
        for i, rom_path in enumerate(visible):
            y = list_y + i * row_h
            idx = self.rom_scroll + i
            self._menu_hits.append((24, y - 3, MENU_W - 48, 28, idx))
            selected = idx == self.rom_cursor
            if selected:
                _blit_selection_bar(self.screen, 24, y - 3, MENU_W - 48, 28)
            tag = tag_cache.get(rom_path)
            if tag is None:
                tag = _read_rom_system_tag(rom_path)
                tag_cache[rom_path] = tag
            tag_s = tag_font.render(tag, True, MENU_HI if selected else MENU_DIM)
            tag_w = tag_s.get_width() + 10
            tag_x = 34
            tag_bg = pygame.Surface((tag_w, tag_s.get_height() + 4))
            tag_bg.fill(MENU_BG)
            tag_bg.set_alpha(200)
            self.screen.blit(tag_bg, (tag_x, y + 1))
            pygame.draw.rect(self.screen, MENU_HI if selected else MENU_DIM,
                             (tag_x, y + 1, tag_w, tag_bg.get_height()), 1)
            self.screen.blit(tag_s, (tag_x + 5, y + 3))
            name = os.path.basename(rom_path)
            name_x = tag_x + tag_w + 8
            label = _fit_text(name_font, name, MENU_W - name_x - 40)
            colour = MENU_HI if selected else MENU_FG
            s = name_font.render(label, True, colour)
            self.screen.blit(s, (name_x, y))
        more_below = self.rom_scroll + self.max_visible < len(self.roms)
        info_y = list_y + len(visible) * row_h + 8
        if more_below:
            self._centre_text("v more", info_y, MENU_DIM, 14)
            info_y += 16
        path = self.roms[self.rom_cursor]
        folder = os.path.dirname(path) or '.'
        info = getattr(self, '_rom_info_cache', {}).get(path)
        if info is None:
            info = _parse_rom_header(path) or "Could not read header"
            self._rom_info_cache = getattr(self, '_rom_info_cache', {})
            self._rom_info_cache[path] = info
        fi = get_font(16)
        meta = _fit_text(fi, f"{folder}  |  {info}", MENU_W - 64)
        self._centre_text(meta, min(info_y, MENU_RULE_Y - 22), MENU_DIM, 16)
        self._draw_chrome("Enter/click: Load   PgUp/PgDn: Jump   F5: Refresh   Esc: Back")

    def _render_settings(self):
        self._centre_text("Settings", 24, MENU_HI, 32, shadow=True)
        items = []
        for i, text in enumerate(self.settings_items):
            selected = i == self.settings_cursor
            if self._settings_id(i) == 'controls':
                items.append(_decorate_submenu_row(text, selected))
            else:
                items.append(_decorate_cyclic_setting(text, selected))
        self._draw_menu(items, self.settings_cursor, 72, 34, size=22)
        footer = ("Enter: Open Controls   Esc: Back"
                  if self._settings_id() == 'controls'
                  else "Left/Right or Enter: Change   F11: Fullscreen   Esc: Back")
        self._draw_chrome(footer)

    def _render_controls(self):
        self._centre_text("Controls", 16, MENU_HI, 30, shadow=True)
        items = self._controls_items()
        if self.controls_capture is not None:
            items = list(items)
            if self.controls_capture in JOYPAD_BUTTON_KEYS:
                idx = JOYPAD_BUTTON_KEYS.index(self.controls_capture)
                items[idx] = f"{JOYPAD_BUTTON_LABELS[idx]}: press a key..."
            elif self.controls_capture == 'turbo_a':
                items[CONTROLS_TURBO_A_ROW] = "Turbo A: press a key..."
            elif self.controls_capture == 'turbo_b':
                items[CONTROLS_TURBO_B_ROW] = "Turbo B: press a key..."
        self.controls_scroll = self._draw_menu(
            items, self.controls_cursor, 70, 28, size=18,
            scroll=self.controls_scroll, max_visible=8)
        hint = "Press a new key   Esc: Cancel" if self.controls_capture else \
            "Enter: Remap / Toggle   Left/Right: WASD   Esc: Back"
        self._draw_chrome(
            hint,
            secondary=None if self.controls_capture else
            "In-game: Tab FF, F3 FPS, F4 input, F11, Ctrl+R")

    def _render_confirm_exit(self):
        self._centre_text("Exit Emulator?", 120, MENU_HI, 36, shadow=True)
        self._centre_text("Quit to the operating system?", 178, MENU_DIM, 22)
        self._draw_menu(["Keep Playing", "Exit to OS"], self.exit_cursor, 250, 50, size=28)
        self._draw_chrome("Left/Right: Choose   Enter: Confirm   Esc: Cancel")


class GameBoy:
    """The main emulator orchestrator class."""
    def __init__(self, rom_path=None, window_scale=4, fps_limit=59.73,
                 audio_enabled=True, volume=1.0, palette=PALETTE_DMG,
                 smooth_scale=False, shader=None, bootrom_path=None,
                 link_cable=None, fullscreen=None):
        self.mmu = MMU()
        self.mmu.link_cable = link_cable
        self.cpu = CPU(self.mmu)
        self.ppu = PPU(self.mmu)
        self.timers = Timers(self.mmu)
        self.apu = APU(self.mmu, self.mmu.is_cgb)
        self.mmu.apu = self.apu
        self.mmu.ppu = self.ppu
        self.mmu.div_reset_callback = self.timers.reset_div
        self.mmu.memory[0xFF04] = 0x00
        self.mmu.memory[0xFF05] = 0x00
        self.mmu.memory[0xFF06] = 0x00
        self.mmu.memory[0xFF07] = 0xF8

        # Optional boot ROM: takes precedence over post-boot register init
        if bootrom_path and os.path.isfile(bootrom_path):
            try:
                with open(bootrom_path, 'rb') as f:
                    self.mmu.load_bootrom(f.read())
                logging.info(f"Loaded boot ROM: {os.path.basename(bootrom_path)} ({len(self.mmu.bootrom)} bytes)")
                self.cpu.reg.pc = 0x0000
            except OSError as e:
                logging.warning(f"Could not load boot ROM: {e}")

        if rom_path:
            if not os.path.isfile(rom_path):
                raise FileNotFoundError(f"ROM not found: {rom_path}")
            try:
                with open(rom_path, 'rb') as f:
                    rom_data = f.read()
            except OSError as e:
                raise FileNotFoundError(f"Could not open ROM: {rom_path}") from e
            if len(rom_data) < 0x150:
                raise ValueError(f"Not a valid Game Boy ROM (too small): {rom_path}")
            self.mmu.load_rom(rom_data)
            self.mmu.rom_path = rom_path
            if self.mmu.is_cgb:
                self.ppu.is_cgb = True
                self.ppu._cgb_init_palettes()
                self.apu.is_cgb = True
            elif self.mmu.is_sgb:
                self.ppu.is_sgb = True
            if not self.mmu.bootrom_enabled:
                # Set post-boot register state only if no boot ROM will run
                if self.mmu.is_cgb:
                    self.cpu.reg.a = 0x11
                    self.cpu.reg.f = 0x80
                    self.cpu.reg.d = 0xFF
                    self.cpu.reg.e = 0x56
                    self.cpu.reg.l = 0x0D
                else:
                    # DMG boot ROM register state.
                    self.cpu.reg.a = 0x01
                    self.cpu.reg.f = 0xB0
                    self.cpu.reg.c = 0x13
                    self.cpu.reg.e = 0xD8
                    self.cpu.reg.h = 0x01
                    self.cpu.reg.l = 0x4D
                self._apply_post_boot_io()
            self._load_sav()
            boot_warnings = []
            header_issues = _validate_rom_header(rom_data)
            if header_issues:
                boot_warnings.append("invalid ROM header")
            if self.mmu.mbc_type not in _SUPPORTED_CART_TYPES:
                boot_warnings.append("mapper not emulated")
            if boot_warnings:
                self._status_msg = " — ".join(w.capitalize() for w in boot_warnings)
                self._status_ttl = 240
            else:
                self._status_msg = ''
                self._status_ttl = 0
        else:
            logging.info("No ROM provided. Running dummy infinite loop.")
            self.mmu.memory[0x0100] = 0x00
            self.mmu.memory[0x0101] = 0xC3
            self.mmu.memory[0x0102] = 0x00
            self.mmu.memory[0x0103] = 0x01
            self._status_msg = ''
            self._status_ttl = 0

        self.speed_remainder = 0  # carries the odd base-clock dot in double-speed mode
        self._rtc_cycle_accum = 0  # throttle MBC3 RTC wall-clock updates to once per frame
        self._has_rtc = self.mmu.has_rtc  # cached flag for step_all hot path
        self.fps_limit = fps_limit
        self.smooth_scale = smooth_scale
        self.shader = shader if shader is not None else _shader_none
        self._prev_shader_frame = None
        self.ppu.set_palette(palette)
        # Reused display buffers avoid per-frame numpy allocations in render().
        if np is not None:
            self._frame_np = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            self._blit_np = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)
            self._shader_out = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            self._shader_f32 = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.float32)
            self._shader_f32_b = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.float32)
            self._ghost_store = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        else:
            self._frame_np = None
            self._blit_np = None
            self._shader_out = None
            self._shader_f32 = None
            self._shader_f32_b = None
            self._ghost_store = None
        self._numpy_slow_warned = False

        self.fullscreen = False
        self._caption = f"Python GBC Emulator - {os.path.basename(rom_path) if rom_path else 'No ROM'}"
        self._present_rect = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self._overlay_hits = []
        self._stick_nav = _StickNav()
        if pygame:
            self.window_scale = _clamp_choice(window_scale, _WINDOW_SCALES, 4)
            pre_cfg = _load_config()
            if fullscreen is None:
                self.fullscreen = bool(pre_cfg.get('fullscreen', False))
            else:
                self.fullscreen = bool(fullscreen)
            self._apply_game_display()
            _init_joysticks()
            if rom_path:
                try:
                    _save_config({'last_rom': os.path.realpath(rom_path)})
                except OSError:
                    _save_config({'last_rom': rom_path})
        elif not rom_path:
            logging.warning("pygame not available — running headless is not useful without a ROM.")

        self._audio_on = audio_enabled
        self._init_audio()
        self._set_volume(volume)

        # In-game pause menu state. Selection indices mirror the global option
        # lists so the pause "Settings" page can cycle them and apply changes live.
        self.paused = False
        self.pause_cursor = 0
        self.pause_settings_cursor = 0
        self.pause_exit_cursor = 0
        self._pause_msg = ''
        self._pause_msg_ttl = 0
        cfg = _load_config()
        self.wasd_enabled = bool(cfg.get('wasd_enabled', True))
        self.key_bindings, self.turbo_bindings = _rebuild_input_maps(
            cfg.get('key_bindings'), self.wasd_enabled, cfg.get('turbo_bindings'))
        self.controls_capture = None
        self.pause_controls_cursor = 0
        self.pause_controls_scroll = 0
        self.pause_save_cursor = 0
        self._fast_forward = False
        self._show_fps = False
        self._show_input = False
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()
        self._fps_value = 0.0
        self._turbo_phase = 0
        self._frame_index = 0
        self._set_idx = {
            'scale':   _opt_index(WINDOW_SCALE_OPTIONS, self.window_scale if pygame else window_scale, 2),
            'fps':     _opt_index(FPS_LIMIT_OPTIONS, fps_limit, 0),
            'audio':   0 if audio_enabled else 1,
            'volume':  _opt_index(VOLUME_OPTIONS, volume, 4),
            'palette': _opt_index(PALETTE_LIST, palette, 0),
            'filter':  _opt_index(FILTER_OPTIONS, smooth_scale, 0),
            'shader':  _opt_index(SHADER_LIST, self.shader, 0),
        }

    def _apply_post_boot_io(self):
        """Hardware IO / mapper state after the boot ROM would have finished.

        Cartridge RAM is left intact so a soft reset does not wipe battery saves.
        """
        mmu = self.mmu
        mem = mmu.memory
        mmu.rom_bank = 1
        mmu.ram_bank = 0
        mmu.ram_enabled = False
        mmu.mbc1_mode = 0
        mmu.mbc1_upper_bank = 0
        if mmu.rom_data:
            mmu._remap_rom_bank()
        mmu.key1 = 0
        mmu.vram_bank_select = 0
        if mmu.svbk != 1:
            mmu.wram_banks[mmu.svbk - 1][:] = mem[0xD000:0xE000]
            mem[0xD000:0xE000] = mmu.wram_banks[0]
            mmu.svbk = 1
        mmu.hdma_active = False
        mmu.hdma_remaining = 0
        mmu.hdma_src = 0
        mmu.hdma_dst = 0
        mmu.dma_remaining = 0
        mmu.dma_index = 0
        mmu.dma_cycle_acc = 0
        mmu.gdma_stall = 0
        mmu.serial_control = 0
        mem[0xFF00] = 0xCF
        mem[0xFF04] = 0x00
        mem[0xFF05] = 0x00
        mem[0xFF06] = 0x00
        mem[0xFF07] = 0xF8
        mem[0xFF0F] = 0x00
        mem[0xFF40] = 0x91
        mem[0xFF41] = 0x85
        mem[0xFF42] = 0x00
        mem[0xFF43] = 0x00
        mem[0xFF44] = 0x00
        mem[0xFF45] = 0x00
        mem[0xFF47] = 0xFC
        mem[0xFF48] = 0xFF
        mem[0xFF49] = 0xFF
        mem[0xFF4A] = 0x00
        mem[0xFF4B] = 0x00
        mem[0xFFFF] = 0x00
        apu = self.apu
        apu.write_register(0xFF26, 0x00)
        apu.write_register(0xFF26, 0x80)
        for addr, val in _APU_BOOT_VALUES.items():
            if addr != 0xFF26:
                apu.write_register(addr, val)
        apu.drain()

    def _init_audio(self):
        """Bring up the SDL audio mixer at the APU's sample rate (signed 16-bit stereo)."""
        self.audio_enabled = False
        self.audio_channel = None
        self._audio_refs = []
        self._audio_pending = deque(maxlen=16)
        if not pygame or not self._audio_on:
            return
        desired = (self.apu.SAMPLE_RATE, -16, 2)
        init_state = pygame.mixer.get_init()
        try:
            if init_state is None:
                pygame.mixer.init(desired[0], desired[1], desired[2], 1024)
            elif init_state[:3] != desired:
                pygame.mixer.quit()
                pygame.mixer.init(desired[0], desired[1], desired[2], 1024)
        except pygame.error as e:
            logging.warning(f"Audio init failed, running silent: {e}")
            return
        try:
            self.audio_channel = pygame.mixer.Channel(0)
            self.audio_enabled = True
        except pygame.error as e:
            logging.warning(f"Audio channel unavailable, running silent: {e}")

    def _set_volume(self, volume):
        self._volume = volume
        if pygame and getattr(self, 'audio_channel', None) is not None:
            try:
                self.audio_channel.set_volume(volume)
            except pygame.error:
                pass

    def _apply_game_display(self):
        """Open the game window at the current scale, or desktop fullscreen."""
        w = SCREEN_WIDTH * getattr(self, 'window_scale', 4)
        h = SCREEN_HEIGHT * getattr(self, 'window_scale', 4)
        self.screen = _open_display((w, h), getattr(self, 'fullscreen', False))
        pygame.display.set_caption(getattr(self, '_caption', 'Python GBC Emulator'))
        _set_window_icon()
        sw, sh = self.screen.get_size()
        self._present_rect = (0, 0, sw, sh)
        if hasattr(self, '_stick_nav'):
            self._stick_nav.reset()

    def _toggle_fullscreen(self):
        self.fullscreen = not bool(getattr(self, 'fullscreen', False))
        self._apply_game_display()
        if not _save_config({'fullscreen': self.fullscreen}):
            self._status_msg = "Could not save settings"
            self._status_ttl = 90
        else:
            self._status_msg = "Fullscreen" if self.fullscreen else "Windowed"
            self._status_ttl = 90
        return True

    def _sav_path(self):
        if not self.mmu.rom_path:
            return None
        base = os.path.splitext(self.mmu.rom_path)[0]
        return base + ".sav"

    def _load_sav(self):
        if not self.mmu.has_battery:
            return
        if not self.mmu.has_ram and not self.mmu.has_rtc:
            return
        path = self._sav_path()
        if path and os.path.isfile(path):
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                ram_len = len(self.mmu.ram_data)
                if ram_len:
                    n = min(ram_len, len(data))
                    self.mmu.ram_data[:n] = data[:n]
                if self.mmu.has_rtc and len(data) >= ram_len + 8:
                    self.mmu.unpack_rtc_blob(data[ram_len:])
                elif self.mmu.mbc_type == 0x20 and len(data) > ram_len and self.mmu.flash_data:
                    blob = data[ram_len:]
                    n = min(len(self.mmu.flash_data), len(blob))
                    self.mmu.flash_data[:n] = blob[:n]
                    self.mmu._remap_rom_bank()
                logging.info(f"Loaded save: {os.path.basename(path)} ({len(data)} bytes)")
            except OSError as e:
                logging.warning(f"Could not load save: {e}")

    def _save_sav(self):
        if not self.mmu.has_battery:
            return
        if not self.mmu.has_ram and not self.mmu.has_rtc:
            return
        path = self._sav_path()
        if not path:
            return
        try:
            with open(path, 'wb') as f:
                f.write(self.mmu.ram_data)
                extra = 0
                if self.mmu.has_rtc:
                    blob = self.mmu.pack_rtc_blob()
                    f.write(blob)
                    extra = len(blob)
                elif self.mmu.mbc_type == 0x20 and self.mmu.flash_data:
                    f.write(self.mmu.flash_data)
                    extra = len(self.mmu.flash_data)
            logging.info(f"Saved: {os.path.basename(path)} ({len(self.mmu.ram_data) + extra} bytes)")
        except OSError as e:
            logging.warning(f"Could not save: {e}")


    # ── Save state support ─────────────────────────────────────────
    SAVE_STATE_MAGIC = b'GBST'
    SAVE_STATE_VERSION = 8
    SAVE_STATE_VERSION_MIN = 1
    _STATE_ROM_ID_BYTES = 32
    _STATE_ROM_FP_BYTES = 8   # v8+: ROM byte length + CRC32 fingerprint
    _STATE_TIMING_MAGIC = 0xA5
    _STATE_TIMING_TAIL_V5 = 3   # gdma_stall u16 + speed_remainder u8
    _STATE_EXTRA_TAIL_V6 = 16   # v5 timing + bootrom + joypad + MBC7 EEPROM shift
    _STATE_SGB_CMD_MAX = 256    # cap assembled SGB command buffer in saves
    _STATE_CPU_FMT = '<BBBB BB BB HHHH'

    @classmethod
    def _state_header_bytes(cls, version):
        extra = cls._STATE_ROM_ID_BYTES if version >= 4 else 0
        fp = cls._STATE_ROM_FP_BYTES if version >= 8 else 0
        return 8 + extra + fp

    @classmethod
    def _state_min_bytes(cls, version=None):
        ver = cls.SAVE_STATE_VERSION if version is None else version
        return (cls._state_header_bytes(ver)
                + struct.calcsize(cls._STATE_CPU_FMT) + 4 + 0x10000)

    def _capture_subsystems(self):
        """Snapshot MMU/PPU/APU fields that load_state mutates (for rollback)."""
        mmu = self.mmu
        ppu = self.ppu
        apu = self.apu
        return {
            'has_rtc': getattr(self, '_has_rtc', mmu.has_rtc),
            'mmu': {
                'mbc_type': mmu.mbc_type, 'ram_enabled': mmu.ram_enabled,
                'rom_bank': mmu.rom_bank, 'ram_bank': mmu.ram_bank,
                'mbc1_mode': mmu.mbc1_mode, 'num_rom_banks': mmu.num_rom_banks,
                'num_ram_banks': mmu.num_ram_banks, 'has_ram': mmu.has_ram,
                'has_battery': mmu.has_battery, 'has_rtc': mmu.has_rtc,
                'is_cgb': mmu.is_cgb, 'joypad_buttons': mmu.joypad_buttons,
                '_joy_src': list(mmu._joy_src), 'serial_data': mmu.serial_data,
                'serial_control': mmu.serial_control, 'vram_bank_select': mmu.vram_bank_select,
                'key1': mmu.key1, 'rp': mmu.rp, 'svbk': mmu.svbk,
                'hdma_remaining': mmu.hdma_remaining, 'hdma_src': mmu.hdma_src,
                'hdma_dst': mmu.hdma_dst, 'dma_remaining': mmu.dma_remaining,
                'dma_src': mmu.dma_src, 'dma_index': mmu.dma_index,
                'dma_cycle_acc': mmu.dma_cycle_acc, 'hdma_active': mmu.hdma_active,
                'mbc1_upper_bank': mmu.mbc1_upper_bank,
                'rtc_s': mmu.rtc_s, 'rtc_m': mmu.rtc_m, 'rtc_h': mmu.rtc_h,
                'rtc_dl': mmu.rtc_dl, 'rtc_dh': mmu.rtc_dh,
                'rtc_latch_state': mmu.rtc_latch_state, 'rtc_last_time': mmu.rtc_last_time,
                'rtc_latch_s': mmu.rtc_latch_s, 'rtc_latch_m': mmu.rtc_latch_m,
                'rtc_latch_h': mmu.rtc_latch_h, 'rtc_latch_dl': mmu.rtc_latch_dl,
                'rtc_latch_dh': mmu.rtc_latch_dh,
                'vram_bank1': bytes(mmu.vram_bank1),
                'wram_banks': [bytes(b) for b in mmu.wram_banks],
                'ram_data': bytes(mmu.ram_data),
                'serial_bits_left': mmu.serial_bits_left,
                'serial_cycle_accum': mmu.serial_cycle_accum,
                'serial_incoming': mmu.serial_incoming, 'is_sgb': mmu.is_sgb,
                'sgb_mask': mmu.sgb_mask, 'sgb_player_count': mmu.sgb_player_count,
                'sgb_current_player': mmu.sgb_current_player,
                'sgb_pal_rgb': list(mmu.sgb_pal_rgb),
                'sgb_attr': bytes(mmu.sgb_attr),
                'mbc6_rom_bank_a': mmu.mbc6_rom_bank_a, 'mbc6_rom_bank_b': mmu.mbc6_rom_bank_b,
                'mbc6_ram_bank_a': mmu.mbc6_ram_bank_a, 'mbc6_ram_bank_b': mmu.mbc6_ram_bank_b,
                'mbc6_flash_a': mmu.mbc6_flash_a, 'mbc6_flash_b': mmu.mbc6_flash_b,
                'mbc6_flash_enable': mmu.mbc6_flash_enable, 'mbc6_flash_we': mmu.mbc6_flash_we,
                'flash_data': bytes(mmu.flash_data),
                'mbc7_ram_enable2': mmu.mbc7_ram_enable2,
                'mbc7_latch_ready': mmu.mbc7_latch_ready,
                'mbc7_latch_x': mmu.mbc7_latch_x, 'mbc7_latch_y': mmu.mbc7_latch_y,
                'eeprom_state': mmu.eeprom_state, 'eeprom_do': mmu.eeprom_do,
                'eeprom_write_en': mmu.eeprom_write_en, 'eeprom_addr': mmu.eeprom_addr,
                'gdma_stall': mmu.gdma_stall,
                'bootrom_enabled': mmu.bootrom_enabled,
                '_dpad_last': list(mmu._dpad_last),
                'eeprom_cs': mmu.eeprom_cs,
                'eeprom_clk': mmu.eeprom_clk,
                'eeprom_bits': mmu.eeprom_bits,
                'eeprom_shift': mmu.eeprom_shift,
                'sgb_in_packet': mmu.sgb_in_packet,
                'sgb_bit_count': mmu.sgb_bit_count,
                'sgb_packet': bytes(mmu.sgb_packet),
                'sgb_cmd': mmu.sgb_cmd,
                'sgb_packets_left': mmu.sgb_packets_left,
                'sgb_cmd_data': bytes(mmu.sgb_cmd_data),
            },
            'ppu': {
                '_scanline_sprites': ppu._scanline_sprites,
                'mode': ppu.mode, 'scanline_dot': ppu.scanline_dot,
                'mode3_duration': ppu.mode3_duration,
                'bg_palette_addr': ppu.bg_palette_addr, 'obj_palette_addr': ppu.obj_palette_addr,
                'cgb_opri': ppu.cgb_opri, 'lcd_was_on': ppu.lcd_was_on, 'is_cgb': ppu.is_cgb,
                'prev_stat_irq': ppu.prev_stat_irq,
                'window_line_counter': ppu.window_line_counter,
                'window_active': ppu.window_active,
                'bg_palette_data': bytes(ppu.bg_palette_data),
                'obj_palette_data': bytes(ppu.obj_palette_data),
            },
            'apu': {
                'power': apu.power, 'is_cgb': apu.is_cgb,
                'frame_seq_step': apu.frame_seq_step, 'fs_div': apu.fs_div,
                '_fs_remain': apu._fs_remain,
                'vol_left': apu.vol_left, 'vol_right': apu.vol_right,
                'pan_left': apu.pan_left, 'pan_right': apu.pan_right,
                'sample_accum': apu.sample_accum,
                'ch1_enabled': apu.ch1_enabled, 'ch1_dac': apu.ch1_dac,
                'ch1_freq': apu.ch1_freq, 'ch1_freq_timer': apu.ch1_freq_timer,
                'ch1_duty': apu.ch1_duty, 'ch1_duty_step': apu.ch1_duty_step,
                'ch1_length_enabled': apu.ch1_length_enabled, 'ch1_length': apu.ch1_length,
                'ch1_volume': apu.ch1_volume, 'ch1_env_initial': apu.ch1_env_initial,
                'ch1_env_direction': apu.ch1_env_direction, 'ch1_env_period': apu.ch1_env_period,
                'ch1_env_timer': apu.ch1_env_timer, 'ch1_sweep_period': apu.ch1_sweep_period,
                'ch1_sweep_direction': apu.ch1_sweep_direction,
                'ch1_sweep_shift': apu.ch1_sweep_shift, 'ch1_sweep_timer': apu.ch1_sweep_timer,
                'ch1_sweep_shadow': apu.ch1_sweep_shadow,
                'ch1_sweep_enabled': apu.ch1_sweep_enabled,
                'ch2_enabled': apu.ch2_enabled, 'ch2_dac': apu.ch2_dac,
                'ch2_freq': apu.ch2_freq, 'ch2_freq_timer': apu.ch2_freq_timer,
                'ch2_duty': apu.ch2_duty, 'ch2_duty_step': apu.ch2_duty_step,
                'ch2_length_enabled': apu.ch2_length_enabled, 'ch2_length': apu.ch2_length,
                'ch2_volume': apu.ch2_volume, 'ch2_env_initial': apu.ch2_env_initial,
                'ch2_env_direction': apu.ch2_env_direction, 'ch2_env_period': apu.ch2_env_period,
                'ch2_env_timer': apu.ch2_env_timer,
                'ch3_enabled': apu.ch3_enabled, 'ch3_dac': apu.ch3_dac,
                'ch3_freq': apu.ch3_freq, 'ch3_freq_timer': apu.ch3_freq_timer,
                'ch3_length_enabled': apu.ch3_length_enabled, 'ch3_length': apu.ch3_length,
                'ch3_vol_shift': apu.ch3_vol_shift, 'ch3_wave_pos': apu.ch3_wave_pos,
                'ch4_enabled': apu.ch4_enabled, 'ch4_dac': apu.ch4_dac,
                'ch4_freq_timer': apu.ch4_freq_timer,
                'ch4_length_enabled': apu.ch4_length_enabled, 'ch4_length': apu.ch4_length,
                'ch4_volume': apu.ch4_volume, 'ch4_env_initial': apu.ch4_env_initial,
                'ch4_env_direction': apu.ch4_env_direction, 'ch4_env_period': apu.ch4_env_period,
                'ch4_env_timer': apu.ch4_env_timer, 'ch4_lfsr': apu.ch4_lfsr,
                'ch4_shift': apu.ch4_shift, 'ch4_width_mode': apu.ch4_width_mode,
                'ch4_divisor_code': apu.ch4_divisor_code,
                'wave_ram': bytes(apu.wave_ram),
            },
            'speed_remainder': getattr(self, 'speed_remainder', 0),
        }

    def _restore_subsystems(self, sub):
        if not sub:
            return
        mmu = self.mmu
        ppu = self.ppu
        apu = self.apu
        m = sub['mmu']
        mmu.mbc_type = m['mbc_type']
        mmu.ram_enabled = m['ram_enabled']
        mmu.rom_bank = m['rom_bank']
        mmu.ram_bank = m['ram_bank']
        mmu.mbc1_mode = m['mbc1_mode']
        mmu.num_rom_banks = m['num_rom_banks']
        mmu.num_ram_banks = m['num_ram_banks']
        mmu.has_ram = m['has_ram']
        mmu.has_battery = m['has_battery']
        mmu.has_rtc = m['has_rtc']
        mmu.is_cgb = m['is_cgb']
        mmu.joypad_buttons = m['joypad_buttons']
        mmu._joy_src = list(m['_joy_src'])
        mmu.serial_data = m['serial_data']
        mmu.serial_control = m['serial_control']
        mmu.vram_bank_select = m['vram_bank_select']
        mmu.key1 = m['key1']
        mmu.rp = m['rp']
        mmu.svbk = m['svbk']
        mmu.hdma_remaining = m['hdma_remaining']
        mmu.hdma_src = m['hdma_src']
        mmu.hdma_dst = m['hdma_dst']
        mmu.dma_remaining = m['dma_remaining']
        mmu.dma_src = m['dma_src']
        mmu.dma_index = m['dma_index']
        mmu.dma_cycle_acc = m['dma_cycle_acc']
        mmu.hdma_active = m['hdma_active']
        mmu.mbc1_upper_bank = m['mbc1_upper_bank']
        mmu.rtc_s = m['rtc_s']
        mmu.rtc_m = m['rtc_m']
        mmu.rtc_h = m['rtc_h']
        mmu.rtc_dl = m['rtc_dl']
        mmu.rtc_dh = m['rtc_dh']
        mmu.rtc_latch_state = m['rtc_latch_state']
        mmu.rtc_last_time = m['rtc_last_time']
        mmu.rtc_latch_s = m['rtc_latch_s']
        mmu.rtc_latch_m = m['rtc_latch_m']
        mmu.rtc_latch_h = m['rtc_latch_h']
        mmu.rtc_latch_dl = m['rtc_latch_dl']
        mmu.rtc_latch_dh = m['rtc_latch_dh']
        mmu.vram_bank1[:] = m['vram_bank1']
        for i, bank in enumerate(m['wram_banks']):
            mmu.wram_banks[i][:] = bank
        if len(mmu.ram_data) == len(m['ram_data']):
            mmu.ram_data[:] = m['ram_data']
        mmu.serial_bits_left = m['serial_bits_left']
        mmu.serial_cycle_accum = m['serial_cycle_accum']
        mmu.serial_incoming = m['serial_incoming']
        mmu.is_sgb = m['is_sgb']
        mmu.sgb_mask = m['sgb_mask']
        mmu.sgb_player_count = m['sgb_player_count']
        mmu.sgb_current_player = m['sgb_current_player']
        mmu.sgb_pal_rgb[:] = m['sgb_pal_rgb']
        mmu.sgb_attr[:] = m['sgb_attr']
        mmu.mbc6_rom_bank_a = m['mbc6_rom_bank_a']
        mmu.mbc6_rom_bank_b = m['mbc6_rom_bank_b']
        mmu.mbc6_ram_bank_a = m['mbc6_ram_bank_a']
        mmu.mbc6_ram_bank_b = m['mbc6_ram_bank_b']
        mmu.mbc6_flash_a = m['mbc6_flash_a']
        mmu.mbc6_flash_b = m['mbc6_flash_b']
        mmu.mbc6_flash_enable = m['mbc6_flash_enable']
        mmu.mbc6_flash_we = m['mbc6_flash_we']
        if len(mmu.flash_data) >= len(m['flash_data']):
            mmu.flash_data[:len(m['flash_data'])] = m['flash_data']
        mmu.mbc7_ram_enable2 = m['mbc7_ram_enable2']
        mmu.mbc7_latch_ready = m['mbc7_latch_ready']
        mmu.mbc7_latch_x = m['mbc7_latch_x']
        mmu.mbc7_latch_y = m['mbc7_latch_y']
        mmu.eeprom_state = m['eeprom_state']
        mmu.eeprom_do = m['eeprom_do']
        mmu.eeprom_write_en = m['eeprom_write_en']
        mmu.eeprom_addr = m['eeprom_addr']
        mmu.gdma_stall = m.get('gdma_stall', 0)
        mmu.bootrom_enabled = m.get('bootrom_enabled', False)
        mmu._dpad_last = list(m.get('_dpad_last', [0, 2]))
        mmu.eeprom_cs = m.get('eeprom_cs', False)
        mmu.eeprom_clk = m.get('eeprom_clk', False)
        mmu.eeprom_bits = m.get('eeprom_bits', 0)
        mmu.eeprom_shift = m.get('eeprom_shift', 0)
        mmu.sgb_in_packet = m.get('sgb_in_packet', False)
        mmu.sgb_bit_count = m.get('sgb_bit_count', 0)
        if 'sgb_packet' in m:
            mmu.sgb_packet[:] = m['sgb_packet']
        mmu.sgb_cmd = m.get('sgb_cmd', 0)
        mmu.sgb_packets_left = m.get('sgb_packets_left', 0)
        if 'sgb_cmd_data' in m:
            mmu.sgb_cmd_data[:] = m['sgb_cmd_data']
        p = sub['ppu']
        ppu.mode = p['mode']
        ppu.scanline_dot = p['scanline_dot']
        ppu.mode3_duration = p['mode3_duration']
        ppu.bg_palette_addr = p['bg_palette_addr']
        ppu.obj_palette_addr = p['obj_palette_addr']
        ppu.cgb_opri = p['cgb_opri']
        ppu.lcd_was_on = p['lcd_was_on']
        ppu.is_cgb = p['is_cgb']
        ppu.prev_stat_irq = p['prev_stat_irq']
        ppu.window_line_counter = p['window_line_counter']
        ppu.window_active = p['window_active']
        ppu.bg_palette_data[:] = p['bg_palette_data']
        ppu.obj_palette_data[:] = p['obj_palette_data']
        ppu._scanline_sprites = p.get('_scanline_sprites')
        for i in range(32):
            ppu._update_cgb_bg_color(i)
            ppu._update_cgb_obj_color(i)
        a = sub['apu']
        apu.power = a['power']
        apu.is_cgb = a['is_cgb']
        apu.frame_seq_step = a['frame_seq_step']
        apu.fs_div = a['fs_div']
        apu._fs_remain = a['_fs_remain']
        apu.vol_left = a['vol_left']
        apu.vol_right = a['vol_right']
        apu.pan_left = a['pan_left']
        apu.pan_right = a['pan_right']
        apu.sample_accum = a['sample_accum']
        apu.ch1_enabled = a['ch1_enabled']
        apu.ch1_dac = a['ch1_dac']
        apu.ch1_freq = a['ch1_freq']
        apu.ch1_freq_timer = a['ch1_freq_timer']
        apu.ch1_duty = a['ch1_duty']
        apu.ch1_duty_step = a['ch1_duty_step']
        apu.ch1_length_enabled = a['ch1_length_enabled']
        apu.ch1_length = a['ch1_length']
        apu.ch1_volume = a['ch1_volume']
        apu.ch1_env_initial = a['ch1_env_initial']
        apu.ch1_env_direction = a['ch1_env_direction']
        apu.ch1_env_period = a['ch1_env_period']
        apu.ch1_env_timer = a['ch1_env_timer']
        apu.ch1_sweep_period = a['ch1_sweep_period']
        apu.ch1_sweep_direction = a['ch1_sweep_direction']
        apu.ch1_sweep_shift = a['ch1_sweep_shift']
        apu.ch1_sweep_timer = a['ch1_sweep_timer']
        apu.ch1_sweep_shadow = a['ch1_sweep_shadow']
        apu.ch1_sweep_enabled = a['ch1_sweep_enabled']
        apu.ch2_enabled = a['ch2_enabled']
        apu.ch2_dac = a['ch2_dac']
        apu.ch2_freq = a['ch2_freq']
        apu.ch2_freq_timer = a['ch2_freq_timer']
        apu.ch2_duty = a['ch2_duty']
        apu.ch2_duty_step = a['ch2_duty_step']
        apu.ch2_length_enabled = a['ch2_length_enabled']
        apu.ch2_length = a['ch2_length']
        apu.ch2_volume = a['ch2_volume']
        apu.ch2_env_initial = a['ch2_env_initial']
        apu.ch2_env_direction = a['ch2_env_direction']
        apu.ch2_env_period = a['ch2_env_period']
        apu.ch2_env_timer = a['ch2_env_timer']
        apu.ch3_enabled = a['ch3_enabled']
        apu.ch3_dac = a['ch3_dac']
        apu.ch3_freq = a['ch3_freq']
        apu.ch3_freq_timer = a['ch3_freq_timer']
        apu.ch3_length_enabled = a['ch3_length_enabled']
        apu.ch3_length = a['ch3_length']
        apu.ch3_vol_shift = a['ch3_vol_shift']
        apu.ch3_wave_pos = a['ch3_wave_pos']
        apu.ch4_enabled = a['ch4_enabled']
        apu.ch4_dac = a['ch4_dac']
        apu.ch4_freq_timer = a['ch4_freq_timer']
        apu.ch4_length_enabled = a['ch4_length_enabled']
        apu.ch4_length = a['ch4_length']
        apu.ch4_volume = a['ch4_volume']
        apu.ch4_env_initial = a['ch4_env_initial']
        apu.ch4_env_direction = a['ch4_env_direction']
        apu.ch4_env_period = a['ch4_env_period']
        apu.ch4_env_timer = a['ch4_env_timer']
        apu.ch4_lfsr = a['ch4_lfsr']
        apu.ch4_shift = a['ch4_shift']
        apu.ch4_width_mode = a['ch4_width_mode']
        apu.ch4_divisor_code = a['ch4_divisor_code']
        apu.wave_ram[:] = a['wave_ram']
        apu._refresh_nr52()
        self._has_rtc = sub['has_rtc']
        self.speed_remainder = sub.get('speed_remainder', 0)

    def _reset_sgb_fsm(self):
        """Clear in-progress Super Game Boy packet assembly state."""
        mmu = self.mmu
        mmu.sgb_in_packet = False
        mmu.sgb_bit_count = 0
        mmu.sgb_packet = bytearray(16)
        mmu.sgb_cmd = 0
        mmu.sgb_cmd_data = bytearray()
        mmu.sgb_packets_left = 0

    def _sanitize_serial_after_load(self):
        """Drop in-flight serial shifts that cannot resume without a live link partner."""
        mmu = self.mmu
        if mmu.serial_bits_left <= 0 or (mmu.serial_control & 0x81) != 0x81:
            return
        lc = getattr(mmu, 'link_cable', None)
        if lc is not None and lc.is_connected:
            return
        mmu.serial_bits_left = 0
        mmu.serial_cycle_accum = 0
        mmu.serial_control &= ~0x80
        mmu.serial_incoming = 0xFF

    def _restore_post_load(self, ver=None):
        """Clear stale caches and re-sync live input sources after load_state."""
        ppu = self.ppu
        ppu._scanline_sprites = None
        ppu._scanline_sprite_height = None
        ppu.bg_palette_idx[:] = b'\x00' * len(ppu.bg_palette_idx)
        self._sanitize_serial_after_load()
        if ver is not None and ver < 7:
            self._reset_sgb_fsm()
        if pygame and getattr(self, 'screen', None) is not None:
            self.mmu.release_all_joypad()
            self._sync_held_inputs()

    def _state_backup(self):
        """Capture live CPU + memory before a load attempt (rollback on failure)."""
        cpu = self.cpu
        reg = cpu.reg
        return {
            'memory': bytes(self.mmu.memory),
            'cpu': (
                reg.a, reg.f, reg.b, reg.c, reg.d, reg.e, reg.h, reg.l, reg.sp, reg.pc,
                cpu.halted, cpu.interrupts_master_enabled, cpu.ime_pending, cpu.halt_bug_pending,
            ),
            'div_counter': self.timers.div_counter,
            'tima_accum': self.timers.tima_accum,
            'subsystems': self._capture_subsystems(),
        }

    def _state_restore(self, snap):
        if not snap:
            return
        mmu = self.mmu
        if len(mmu.memory) != 0x10000:
            mmu.memory = bytearray(0x10000)
        mmu.memory[:] = snap['memory']
        (a, f_, b, c, d, e, h, l, sp, pc,
         halted, ime, ime_pending, halt_bug) = snap['cpu']
        reg = self.cpu.reg
        reg.a, reg.f, reg.b, reg.c = a, f_ & 0xF0, b, c
        reg.d, reg.e, reg.h, reg.l = d, e, h, l
        reg.sp, reg.pc = sp, pc
        self.cpu.halted = halted
        self.cpu.interrupts_master_enabled = ime
        self.cpu.ime_pending = ime_pending
        self.cpu.halt_bug_pending = halt_bug
        self.timers.div_counter = snap['div_counter']
        self.timers.tima_accum = snap['tima_accum']
        mmu.memory[0xFF04] = (snap['div_counter'] >> 8) & 0xFF
        self._restore_subsystems(snap.get('subsystems'))

    @staticmethod
    def _format_state_message(action, slot, error_code, detail=None):
        if action == 'save' and error_code is None:
            return f"State saved to slot {slot}"
        if action == 'load' and error_code is None:
            return f"State loaded from slot {slot}"
        if error_code == 'missing':
            return f"No save state in slot {slot}"
        if error_code == 'version':
            return f"Unsupported save version in slot {slot}"
        if error_code == 'corrupt':
            return f"Save file in slot {slot} is corrupted"
        if error_code == 'wrong_rom':
            return f"Save state in slot {slot} is for a different ROM"
        if error_code == 'io':
            base = f"Could not {action} slot {slot}"
            return f"{base}: {detail}" if detail else base
        if action == 'save':
            return "Failed to save state" + (f": {detail}" if detail else "")
        return f"Could not load slot {slot}" + (f": {detail}" if detail else "")

    def _state_path(self, slot):
        if not self.mmu.rom_path:
            return None
        base = os.path.splitext(self.mmu.rom_path)[0]
        return f"{base}.ss{slot}"

    def _rom_identity_ok(self, name_len, full_len, saved_name):
        """Return True when a v4+ save header matches the loaded ROM basename."""
        if not name_len:
            return True
        current_base = os.path.basename(self.mmu.rom_path or '')
        current_name = current_base.encode('utf-8', 'replace')[:31]
        current_full_len = min(len(current_base.encode('utf-8', 'replace')), 255)
        if saved_name != current_name[:name_len]:
            return False
        return not (full_len and current_full_len != full_len)

    def _rom_fingerprint_ok(self, rom_size, rom_crc):
        """Return True when a v8+ save ROM fingerprint matches loaded ROM bytes."""
        rom = self.mmu.rom_data
        if len(rom) != rom_size:
            return False
        if rom_size == 0:
            return True
        return (zlib.crc32(rom) & 0xFFFFFFFF) == (rom_crc & 0xFFFFFFFF)

    def _state_toast(self, action, slot, ok):
        detail = getattr(self, '_last_state_detail', None)
        default_err = 'corrupt' if action == 'load' else 'io'
        err = None if ok else getattr(self, '_last_state_error', default_err)
        self._status_msg = self._format_state_message(action, slot, err, detail)
        self._status_ttl = 90

    def save_state(self, slot=0):
        self._last_state_error = None
        self._last_state_detail = None
        path = self._state_path(slot)
        if not path:
            self._last_state_error = 'io'
            return False
        try:
            apu = self.apu
            mmu = self.mmu
            ppu = self.ppu
            cpu = self.cpu
            timers = self.timers
            # Sync CGB WRAM bank back to main memory so it round-trips.
            if 1 <= mmu.svbk <= 7:
                mmu.wram_banks[mmu.svbk - 1][:] = mmu.memory[0xD000:0xE000]
            # Flush APU to silence (state captures the buffer's tail).
            apu.drain()
            parts = []
            parts.append(self.SAVE_STATE_MAGIC)
            rom_base = os.path.basename(mmu.rom_path or '')
            rom_name = rom_base.encode('utf-8', 'replace')[:31]
            rom_full_len = min(len(rom_base.encode('utf-8', 'replace')), 255)
            parts.append(struct.pack(
                '<BBBB', self.SAVE_STATE_VERSION, slot, len(rom_name), rom_full_len))
            parts.append(rom_name + b'\x00' * (self._STATE_ROM_ID_BYTES - len(rom_name)))
            rom_crc = zlib.crc32(mmu.rom_data) & 0xFFFFFFFF if mmu.rom_data else 0
            parts.append(struct.pack('<II', len(mmu.rom_data), rom_crc))
            # CPU
            reg = cpu.reg
            parts.append(struct.pack('<BBBB BB BB HHHH',
                                     reg.a, reg.f, reg.b, reg.c, reg.d, reg.e, reg.h, reg.l,
                                     reg.sp, reg.pc, 0, 0))
            parts.append(struct.pack('<BBBB',
                                     1 if cpu.halted else 0,
                                     1 if cpu.interrupts_master_enabled else 0,
                                     1 if cpu.ime_pending else 0,
                                     1 if cpu.halt_bug_pending else 0))
            # MMU
            parts.append(mmu.memory)  # 64KB
            parts.append(struct.pack('<BBBBBBBBBBBBBBBBBBB',
                                     mmu.mbc_type, 1 if mmu.ram_enabled else 0,
                                     mmu.rom_bank & 0xFF, (mmu.rom_bank >> 8) & 0xFF,
                                     mmu.ram_bank & 0xFF, mmu.mbc1_mode & 0xFF,
                                     mmu.num_rom_banks, mmu.num_ram_banks,
                                     1 if mmu.has_ram else 0, 1 if mmu.has_battery else 0,
                                     1 if mmu.has_rtc else 0, 1 if mmu.is_cgb else 0,
                                     mmu.joypad_buttons, mmu.serial_data, mmu.serial_control,
                                     mmu.vram_bank_select, mmu.key1, mmu.rp, mmu.svbk))
            parts.append(struct.pack('<iiiHBBBBBB',
                                     int(mmu.hdma_remaining), int(mmu.hdma_src), int(mmu.hdma_dst),
                                     int(mmu.dma_remaining) & 0xFFFF,
                                     mmu.dma_src & 0xFF, mmu.dma_index & 0xFF,
                                     mmu.dma_cycle_acc & 0xFF,
                                     1 if mmu.hdma_active else 0,
                                     mmu.mbc1_upper_bank & 0x03,
                                     1 if ppu.window_active else 0))
            # RTC
            parts.append(struct.pack('<BBBBBB d',
                                     mmu.rtc_s & 0xFF, mmu.rtc_m & 0xFF, mmu.rtc_h & 0xFF,
                                     mmu.rtc_dl & 0xFF, mmu.rtc_dh & 0xFF,
                                     mmu.rtc_latch_state & 0xFF, mmu.rtc_last_time))
            parts.append(struct.pack('<BBBBB',
                                     mmu.rtc_latch_s & 0xFF, mmu.rtc_latch_m & 0xFF,
                                     mmu.rtc_latch_h & 0xFF, mmu.rtc_latch_dl & 0xFF,
                                     mmu.rtc_latch_dh & 0xFF))
            parts.append(mmu.vram_bank1)
            # WRAM bank snapshots
            for i in range(7):
                parts.append(mmu.wram_banks[i])
            # PPU (reserved bytes carry APU frame-sequencer timing when magic is set)
            apu = self.apu
            fs_div_lo = apu.fs_div & 0xFF
            fs_remain_lo = apu._fs_remain & 0xFF
            fs_remain_hi = (apu._fs_remain >> 8) & 0xFF
            parts.append(struct.pack('<BBB BBBBB BBBBB BBBB BB',
                                     ppu.mode, ppu.scanline_dot & 0xFF,
                                     (ppu.scanline_dot >> 8) & 0xFF,
                                     ppu.mode3_duration & 0xFF, (ppu.mode3_duration >> 8) & 0xFF,
                                     fs_div_lo, self._STATE_TIMING_MAGIC, fs_remain_lo,
                                     ppu.bg_palette_addr, ppu.obj_palette_addr, ppu.cgb_opri,
                                     1 if ppu.lcd_was_on else 0, 1 if ppu.is_cgb else 0,
                                     ppu.window_line_counter & 0xFF,
                                     (ppu.window_line_counter >> 8) & 0xFF,
                                     1 if ppu.window_active else 0, fs_remain_hi, 0,
                                     1 if ppu.prev_stat_irq else 0))
            parts.append(ppu.bg_palette_data)
            parts.append(ppu.obj_palette_data)
            # APU: pack a flat list of 1-byte values for each boolean / small int.
            apu_values = [
                1 if apu.power else 0,
                1 if apu.is_cgb else 0,
                apu.frame_seq_step & 0x07,
                (apu.fs_div >> 8) & 0xFF,
                apu.vol_left & 0x07, apu.vol_right & 0x07,
                apu.pan_left, apu.pan_right,
                apu.sample_accum & 0xFF, (apu.sample_accum >> 8) & 0xFF,
                apu.sample_num & 0xFF, (apu.sample_num >> 8) & 0xFF,
                apu.sample_den & 0xFF, (apu.sample_den >> 8) & 0xFF,
                # Channel 1 (16 bytes)
                1 if apu.ch1_enabled else 0, 1 if apu.ch1_dac else 0,
                apu.ch1_freq & 0xFF, (apu.ch1_freq >> 8) & 0x0F,
                apu.ch1_freq_timer & 0xFF, (apu.ch1_freq_timer >> 8) & 0xFF,
                apu.ch1_duty & 0x03, apu.ch1_duty_step & 0x07,
                1 if apu.ch1_length_enabled else 0, apu.ch1_length & 0x3F,
                apu.ch1_volume & 0x0F, apu.ch1_env_initial & 0x0F,
                apu.ch1_env_direction & 0x01, apu.ch1_env_period & 0x07,
                apu.ch1_env_timer & 0x07, apu.ch1_sweep_period & 0x07,
                # Channel 2 (12 bytes)
                1 if apu.ch2_enabled else 0, 1 if apu.ch2_dac else 0,
                apu.ch2_freq & 0xFF, (apu.ch2_freq >> 8) & 0x0F,
                apu.ch2_freq_timer & 0xFF, (apu.ch2_freq_timer >> 8) & 0xFF,
                apu.ch2_duty & 0x03, apu.ch2_duty_step & 0x07,
                1 if apu.ch2_length_enabled else 0, apu.ch2_length & 0x3F,
                apu.ch2_volume & 0x0F, apu.ch2_env_period & 0x07,
                # Channel 3 (10 bytes)
                1 if apu.ch3_enabled else 0, 1 if apu.ch3_dac else 0,
                apu.ch3_freq & 0xFF, (apu.ch3_freq >> 8) & 0x0F,
                apu.ch3_freq_timer & 0xFF, (apu.ch3_freq_timer >> 8) & 0xFF,
                1 if apu.ch3_length_enabled else 0, apu.ch3_length & 0xFF,
                apu.ch3_vol_shift & 0x03, apu.ch3_wave_pos & 0x1F,
                # Channel 4 (12 bytes)
                1 if apu.ch4_enabled else 0, 1 if apu.ch4_dac else 0,
                apu.ch4_freq_timer & 0xFF, (apu.ch4_freq_timer >> 8) & 0xFF,
                1 if apu.ch4_length_enabled else 0, apu.ch4_length & 0x3F,
                apu.ch4_volume & 0x0F, apu.ch4_env_period & 0x07,
                apu.ch4_lfsr & 0xFF, (apu.ch4_lfsr >> 8) & 0x7F,
                apu.ch4_shift & 0x0F, apu.ch4_width_mode & 0x01,
                # Sweep + env state not in main channels
                apu.ch1_sweep_direction & 0x01, apu.ch1_sweep_shift & 0x07,
                apu.ch1_sweep_timer & 0x07, apu.ch1_sweep_shadow & 0xFF,
                (apu.ch1_sweep_shadow >> 8) & 0x0F,
                1 if apu.ch1_sweep_enabled else 0,
                apu.ch1_env_timer & 0x07,
                apu.ch2_env_initial & 0x0F, apu.ch2_env_direction & 0x01,
                apu.ch2_env_timer & 0x07,
                apu.ch4_env_initial & 0x0F, apu.ch4_env_direction & 0x01,
                apu.ch4_env_timer & 0x07,
                apu.ch4_divisor_code & 0x07,
            ]
            parts.append(bytes(apu_values))
            parts.append(apu.wave_ram)
            # Timers
            parts.append(struct.pack('<II', timers.div_counter, timers.tima_accum))
            parts.append(struct.pack('<I', len(mmu.ram_data)))
            parts.append(mmu.ram_data)
            parts.append(struct.pack('<BHHBBBB',
                                     mmu.serial_bits_left & 0xFF,
                                     mmu.serial_cycle_accum & 0xFFFF,
                                     mmu.serial_incoming & 0xFF,
                                     1 if mmu.is_sgb else 0,
                                     mmu.sgb_mask & 0x03,
                                     mmu.sgb_player_count & 0x07,
                                     mmu.sgb_current_player & 0x03))
            parts.append(struct.pack('<16I', *([mmu.sgb_pal_rgb[i] & 0xFFFFFF for i in range(16)])))
            parts.append(mmu.sgb_attr)
            parts.append(struct.pack('<BBBBBBBB',
                                     mmu.mbc6_rom_bank_a & 0x7F, mmu.mbc6_rom_bank_b & 0x7F,
                                     mmu.mbc6_ram_bank_a & 7, mmu.mbc6_ram_bank_b & 7,
                                     1 if mmu.mbc6_flash_a else 0, 1 if mmu.mbc6_flash_b else 0,
                                     1 if mmu.mbc6_flash_enable else 0, 1 if mmu.mbc6_flash_we else 0))
            flash = mmu.flash_data if mmu.mbc_type == 0x20 else b''
            parts.append(struct.pack('<I', len(flash)))
            parts.append(flash)
            parts.append(struct.pack('<BBHHBBBB',
                                     1 if mmu.mbc7_ram_enable2 else 0,
                                     1 if mmu.mbc7_latch_ready else 0,
                                     mmu.mbc7_latch_x & 0xFFFF, mmu.mbc7_latch_y & 0xFFFF,
                                     mmu.eeprom_state & 0xFF, mmu.eeprom_do & 1,
                                     1 if mmu.eeprom_write_en else 0, mmu.eeprom_addr & 0x7F))
            parts.append(struct.pack(
                '<HBBBBBBBBBBBBH',
                mmu.gdma_stall & 0xFFFF,
                getattr(self, 'speed_remainder', 0) & 0xFF,
                1 if mmu.bootrom_enabled else 0,
                mmu._joy_src[0], mmu._joy_src[1], mmu._joy_src[2],
                mmu._joy_src[3], mmu._joy_src[4],
                mmu._dpad_last[0], mmu._dpad_last[1],
                1 if mmu.eeprom_cs else 0,
                1 if mmu.eeprom_clk else 0,
                mmu.eeprom_bits & 0xFF,
                mmu.eeprom_shift & 0xFFFF))
            sgb_cmd = bytes(mmu.sgb_cmd_data[:self._STATE_SGB_CMD_MAX])
            parts.append(struct.pack(
                '<BHBB', 1 if mmu.sgb_in_packet else 0,
                mmu.sgb_bit_count & 0xFFFF,
                mmu.sgb_cmd & 0xFF,
                mmu.sgb_packets_left & 0xFF))
            parts.append(bytes(mmu.sgb_packet))
            parts.append(struct.pack('<H', len(sgb_cmd)))
            parts.append(sgb_cmd)
            with open(path, 'wb') as f:
                for p in parts:
                    f.write(p)
            logging.info(f"Saved state to slot {slot}: {os.path.basename(path)}")
            return True
        except (OSError, struct.error, ValueError) as e:
            logging.warning(f"Could not save state: {e}")
            self._last_state_error = 'io'
            self._last_state_detail = str(e)[:60]
            return False

    def load_state(self, slot=0):
        self._last_state_error = None
        self._last_state_detail = None
        path = self._state_path(slot)
        if not path or not os.path.isfile(path):
            self._last_state_error = 'missing'
            return False
        snap = None
        try:
            with open(path, 'rb') as f:
                data = f.read()
            pos = 0
            if data[pos:pos+4] != self.SAVE_STATE_MAGIC:
                logging.warning("Save state: bad magic")
                self._last_state_error = 'corrupt'
                return False
            pos += 4
            ver = data[pos]
            if ver < self.SAVE_STATE_VERSION_MIN or ver > self.SAVE_STATE_VERSION:
                logging.warning(f"Save state: unsupported version {ver}")
                self._last_state_error = 'version'
                return False
            if len(data) < self._state_min_bytes(ver):
                raise ValueError("truncated save state (header)")
            pos += 4
            if ver >= 4:
                name_len = data[6]
                full_len = data[7]
                saved_name = bytes(data[8:8 + name_len])
                if not self._rom_identity_ok(name_len, full_len, saved_name):
                    self._last_state_error = 'wrong_rom'
                    return False
                if ver >= 8:
                    fp_off = 8 + self._STATE_ROM_ID_BYTES
                    rom_size, rom_crc = struct.unpack_from('<II', data, fp_off)
                    if not self._rom_fingerprint_ok(rom_size, rom_crc):
                        self._last_state_error = 'wrong_rom'
                        return False
                pos = self._state_header_bytes(ver)
            mmu = self.mmu
            snap = self._state_backup()
            cpu = self.cpu
            ppu = self.ppu
            apu = self.apu
            timers = self.timers

            def _need(n, label):
                if pos + n > len(data):
                    raise ValueError(f"truncated save state ({label})")

            # CPU
            cpu_fmt = self._STATE_CPU_FMT
            _need(struct.calcsize(cpu_fmt) + 4, "cpu")
            (a, f_, b, c, d, e, h, l, sp, pc, _, _) = struct.unpack_from(cpu_fmt, data, pos)
            pos += struct.calcsize(cpu_fmt)
            cpu.reg.a = a; cpu.reg.f = f_ & 0xF0
            cpu.reg.b = b; cpu.reg.c = c
            cpu.reg.d = d; cpu.reg.e = e
            cpu.reg.h = h; cpu.reg.l = l
            cpu.reg.sp = sp; cpu.reg.pc = pc
            (halted, ime, ime_pending, halt_bug) = struct.unpack_from('<BBBB', data, pos)
            cpu.halted = bool(halted)
            cpu.interrupts_master_enabled = bool(ime)
            cpu.ime_pending = bool(ime_pending)
            cpu.halt_bug_pending = bool(halt_bug)
            pos += 4
            # MMU memory — require an exact 64KB slice so a truncated file
            # cannot shrink the live bytearray.
            _need(0x10000, "memory")
            mem_chunk = data[pos:pos + 0x10000]
            if len(mmu.memory) != 0x10000:
                mmu.memory = bytearray(0x10000)
            mmu.memory[:] = mem_chunk
            pos += 0x10000
            # MMU control
            fmt = '<BBBBBBBBBBBBBBBBBBB'
            _need(struct.calcsize(fmt), "mmu")
            unpacked = struct.unpack_from(fmt, data, pos)
            pos += struct.calcsize(fmt)
            (mbc_type, ram_enabled, rom_bank_lo, rom_bank_hi, ram_bank, mbc1_mode,
             num_rom_banks, num_ram_banks, has_ram, has_battery, has_rtc, is_cgb,
             joypad, serial_data, serial_control,
             vram_bank_select, key1, rp, svbk) = unpacked
            mmu.mbc_type = mbc_type
            mmu.ram_enabled = bool(ram_enabled)
            mmu.rom_bank = rom_bank_lo | ((rom_bank_hi & 1) << 8)
            mmu.ram_bank = ram_bank
            mmu.mbc1_mode = mbc1_mode
            mmu.num_rom_banks = num_rom_banks
            mmu.num_ram_banks = num_ram_banks
            mmu.has_ram = bool(has_ram)
            mmu.has_battery = bool(has_battery)
            mmu.has_rtc = bool(has_rtc)
            self._has_rtc = mmu.has_rtc
            mmu.is_cgb = bool(is_cgb)
            saved_joypad = joypad
            mmu.serial_data = serial_data
            mmu.serial_control = serial_control
            mmu.vram_bank_select = vram_bank_select & 1
            mmu.key1 = key1
            mmu.rp = rp
            mmu.svbk = svbk if 1 <= svbk <= 7 else 1
            # HDMA / DMA
            if ver >= 2:
                fmt = '<iiiHBBBBBB'
                _need(struct.calcsize(fmt), "hdma")
                unpacked = struct.unpack_from(fmt, data, pos)
                pos += struct.calcsize(fmt)
                (hdma_remaining, hdma_src, hdma_dst, dma_remaining,
                 dma_src, dma_index, dma_cycle_acc, hdma_active,
                 mbc1_upper, window_active_mmu) = unpacked
                mmu.hdma_remaining = hdma_remaining
                mmu.hdma_src = hdma_src & 0xFFFF
                mmu.hdma_dst = hdma_dst & 0xFFFF
                mmu.hdma_active = bool(hdma_active)
                mmu.dma_src = dma_src
                mmu.dma_remaining = dma_remaining
                mmu.dma_index = min(dma_index, 160)
                mmu.dma_cycle_acc = dma_cycle_acc
                mmu.mbc1_upper_bank = mbc1_upper & 0x03
                ppu.window_active = bool(window_active_mmu)
            else:
                fmt = '<iiBBBBBBBBBB'
                _need(struct.calcsize(fmt), "hdma")
                unpacked = struct.unpack_from(fmt, data, pos)
                pos += struct.calcsize(fmt)
                (hdma_remaining, hdma_src, hdma_dst, hdma_active, dma_src,
                 _d0, _d1, _d2, _d3, _d4, _d5, _d6) = unpacked
                mmu.hdma_remaining = hdma_remaining
                mmu.hdma_src = hdma_src
                mmu.hdma_dst = hdma_dst
                mmu.hdma_active = bool(hdma_active)
                mmu.dma_src = dma_src
                mmu.dma_remaining = 0
                mmu.dma_index = 0
                mmu.dma_cycle_acc = 0
            mmu.dma_buffer = bytearray()
            # RTC current
            fmt = '<BBBBBB d'
            unpacked = struct.unpack_from(fmt, data, pos)
            pos += struct.calcsize(fmt)
            (rtc_s, rtc_m, rtc_h, rtc_dl, rtc_dh, rtc_latch_state, rtc_last_time) = unpacked
            mmu.rtc_s = rtc_s & 0x3F
            mmu.rtc_m = rtc_m & 0x3F
            mmu.rtc_h = rtc_h & 0x1F
            mmu.rtc_dl = rtc_dl
            mmu.rtc_dh = rtc_dh & 0xC1
            mmu.rtc_latch_state = rtc_latch_state & 0xFF
            mmu.rtc_last_time = rtc_last_time
            # RTC latched
            fmt = '<BBBBB'
            unpacked = struct.unpack_from(fmt, data, pos)
            pos += struct.calcsize(fmt)
            (rtc_ls, rtc_lm, rtc_lh, rtc_ldl, rtc_ldh) = unpacked
            mmu.rtc_latch_s = rtc_ls & 0x3F
            mmu.rtc_latch_m = rtc_lm & 0x3F
            mmu.rtc_latch_h = rtc_lh & 0x1F
            mmu.rtc_latch_dl = rtc_ldl
            mmu.rtc_latch_dh = rtc_ldh & 0xC1
            # VRAM bank 1
            _need(0x2000, "vram1")
            mmu.vram_bank1[:] = data[pos:pos+0x2000]
            pos += 0x2000
            # WRAM banks
            for i in range(7):
                _need(0x1000, "wram")
                mmu.wram_banks[i][:] = data[pos:pos+0x1000]
                pos += 0x1000
            # PPU
            fmt = '<BBB BBBBB BBBBB BBBB BB'
            unpacked = struct.unpack_from(fmt, data, pos)
            pos += struct.calcsize(fmt)
            (mode, scan_lo, scan_hi, m3_lo, m3_hi, _r0, _r1, _r2,
             bg_pal_addr, obj_pal_addr, cgb_opri, lcd_was_on, is_cgb_ppu,
             win_lo, win_hi, _w0, _w1, _w2, _w3) = unpacked
            ppu.mode = mode
            ppu.scanline_dot = scan_lo | (scan_hi << 8)
            ppu.mode3_duration = m3_lo | (m3_hi << 8)
            ppu.bg_palette_addr = bg_pal_addr
            ppu.obj_palette_addr = obj_pal_addr
            ppu.cgb_opri = cgb_opri & 1
            ppu.lcd_was_on = bool(lcd_was_on)
            ppu.is_cgb = bool(is_cgb_ppu)
            ppu.prev_stat_irq = bool(_w3)
            ppu.window_line_counter = win_lo | (win_hi << 8)
            ppu.window_active = bool(_w0)
            timing_magic = (_r1 == self._STATE_TIMING_MAGIC)
            _need(64 + 64, "palettes")
            ppu.bg_palette_data[:] = data[pos:pos+64]
            pos += 64
            ppu.obj_palette_data[:] = data[pos:pos+64]
            pos += 64
            # Refresh derived CGB color tables so the PPU can use the loaded palettes.
            for i in range(32):
                ppu._update_cgb_bg_color(i)
                ppu._update_cgb_obj_color(i)
            # APU
            apu_size = 13 + 16 + 12 + 10 + 12 + 14 + 1  # +1 for is_cgb
            _need(apu_size + 16 + 8, "apu")
            apu_bytes = data[pos:pos+apu_size]
            pos += apu_size
            ap = apu_bytes
            ai = 0
            apu.power = bool(ap[ai]); ai += 1
            apu.is_cgb = bool(ap[ai]); ai += 1
            apu.frame_seq_step = ap[ai] & 0x07; ai += 1
            fs_div_hi = ap[ai] & 0xFF; ai += 1
            if timing_magic:
                apu.fs_div = _r0 | (fs_div_hi << 8)
                apu._fs_remain = _r2 | (_w1 << 8)
            else:
                apu.fs_div = fs_div_hi << 8
                apu._sync_fs_remain(apu.fs_div, bool(self.mmu.key1 & 0x80))
            apu.vol_left = ap[ai] & 0x07; apu.vol_right = ap[ai+1] & 0x07; ai += 2
            apu.pan_left = ap[ai]; apu.pan_right = ap[ai+1]; ai += 2
            apu.sample_accum = ap[ai] | (ap[ai+1] << 8); ai += 2
            ai += 4  # sample_num and sample_den (constants, skip)
            apu.sample_num = apu.CPU_CLOCK
            apu.sample_den = apu.SAMPLE_RATE
            # Channel 1
            apu.ch1_enabled = bool(ap[ai]); apu.ch1_dac = bool(ap[ai+1]); ai += 2
            apu.ch1_freq = ap[ai] | ((ap[ai+1] & 0x07) << 8); ai += 2
            apu.ch1_freq_timer = ap[ai] | (ap[ai+1] << 8); ai += 2
            apu.ch1_duty = ap[ai] & 0x03; apu.ch1_duty_step = ap[ai+1] & 0x07; ai += 2
            apu.ch1_length_enabled = bool(ap[ai]); apu.ch1_length = ap[ai+1] & 0x3F; ai += 2
            apu.ch1_volume = ap[ai] & 0x0F; apu.ch1_env_initial = ap[ai+1] & 0x0F; ai += 2
            apu.ch1_env_direction = ap[ai] & 0x01
            apu.ch1_env_period = ap[ai+1] & 0x07
            apu.ch1_env_timer = ap[ai+2] & 0x07
            apu.ch1_sweep_period = ap[ai+3] & 0x07; ai += 4
            # Channel 2
            apu.ch2_enabled = bool(ap[ai]); apu.ch2_dac = bool(ap[ai+1]); ai += 2
            apu.ch2_freq = ap[ai] | ((ap[ai+1] & 0x07) << 8); ai += 2
            apu.ch2_freq_timer = ap[ai] | (ap[ai+1] << 8); ai += 2
            apu.ch2_duty = ap[ai] & 0x03; apu.ch2_duty_step = ap[ai+1] & 0x07; ai += 2
            apu.ch2_length_enabled = bool(ap[ai]); apu.ch2_length = ap[ai+1] & 0x3F; ai += 2
            apu.ch2_volume = ap[ai] & 0x0F
            apu.ch2_env_period = ap[ai+1] & 0x07; ai += 2
            # Channel 3
            apu.ch3_enabled = bool(ap[ai]); apu.ch3_dac = bool(ap[ai+1]); ai += 2
            apu.ch3_freq = ap[ai] | ((ap[ai+1] & 0x07) << 8); ai += 2
            apu.ch3_freq_timer = ap[ai] | (ap[ai+1] << 8); ai += 2
            apu.ch3_length_enabled = bool(ap[ai]); apu.ch3_length = ap[ai+1] & 0xFF; ai += 2
            apu.ch3_vol_shift = ap[ai] & 0x03; apu.ch3_wave_pos = ap[ai+1] & 0x1F; ai += 2
            # Channel 4
            apu.ch4_enabled = bool(ap[ai]); apu.ch4_dac = bool(ap[ai+1]); ai += 2
            apu.ch4_freq_timer = ap[ai] | (ap[ai+1] << 8); ai += 2
            apu.ch4_length_enabled = bool(ap[ai]); apu.ch4_length = ap[ai+1] & 0x3F; ai += 2
            apu.ch4_volume = ap[ai] & 0x0F
            apu.ch4_env_period = ap[ai+1] & 0x07; ai += 2
            apu.ch4_lfsr = ap[ai] | ((ap[ai+1] & 0x7F) << 8); ai += 2
            apu.ch4_shift = ap[ai] & 0x0F
            apu.ch4_width_mode = ap[ai+1] & 0x01; ai += 2
            # Sweep + env state not in main channels
            apu.ch1_sweep_direction = ap[ai] & 0x01
            apu.ch1_sweep_shift = ap[ai+1] & 0x07
            apu.ch1_sweep_timer = ap[ai+2] & 0x07
            apu.ch1_sweep_shadow = ap[ai+3] | ((ap[ai+4] & 0x0F) << 8)
            apu.ch1_sweep_enabled = bool(ap[ai+5])
            apu.ch1_env_timer = ap[ai+6] & 0x07; ai += 7
            apu.ch2_env_initial = ap[ai] & 0x0F
            apu.ch2_env_direction = ap[ai+1] & 0x01
            apu.ch2_env_timer = ap[ai+2] & 0x07; ai += 3
            apu.ch4_env_initial = ap[ai] & 0x0F
            apu.ch4_env_direction = ap[ai+1] & 0x01
            apu.ch4_env_timer = ap[ai+2] & 0x07
            apu.ch4_divisor_code = ap[ai+3] & 0x07; ai += 4
            apu.wave_ram[:] = data[pos:pos+16]
            pos += 16
            apu._refresh_nr52()
            # Timers
            _need(8, "timers")
            (div_counter, tima_accum) = struct.unpack_from('<II', data, pos)
            pos += 8
            timers.div_counter = div_counter
            timers.tima_accum = tima_accum
            if ver >= 2:
                _need(4, "ram_len")
                (ram_len,) = struct.unpack_from('<I', data, pos)
                pos += 4
                if ram_len > 2 * 1024 * 1024:
                    raise ValueError("save state ram_data is implausibly large")
                _need(ram_len, "ram_data")
                ram_blob = data[pos:pos + ram_len]
                pos += ram_len
                if ram_len == len(mmu.ram_data):
                    mmu.ram_data[:] = ram_blob
                elif len(mmu.ram_data) > 0:
                    n = min(len(mmu.ram_data), ram_len)
                    mmu.ram_data[:n] = ram_blob[:n]
                    if ram_len < len(mmu.ram_data):
                        mmu.ram_data[ram_len:] = b'\x00' * (len(mmu.ram_data) - ram_len)
            if ver >= 3:
                fmt = '<BHHBBBB'
                _need(struct.calcsize(fmt), "serial_sgb")
                (sbits, sacc, sin, is_sgb, mask, pcount, pcur) = struct.unpack_from(fmt, data, pos)
                pos += struct.calcsize(fmt)
                mmu.serial_bits_left = sbits
                mmu.serial_cycle_accum = sacc
                mmu.serial_incoming = sin & 0xFF
                mmu.is_sgb = bool(is_sgb)
                ppu.is_sgb = mmu.is_sgb and not mmu.is_cgb
                mmu.sgb_mask = mask
                mmu.sgb_player_count = max(1, pcount)
                mmu.sgb_current_player = pcur
                _need(64, "sgb_pal")
                pals = struct.unpack_from('<16I', data, pos)
                pos += 64
                mmu.sgb_pal_rgb = [p & 0xFFFFFF for p in pals]
                _need(360, "sgb_attr")
                mmu.sgb_attr[:] = data[pos:pos + 360]
                pos += 360
                fmt = '<BBBBBBBB'
                _need(struct.calcsize(fmt), "mbc6")
                (ba, bb, ra, rb, fa, fb, fe, fw) = struct.unpack_from(fmt, data, pos)
                pos += struct.calcsize(fmt)
                mmu.mbc6_rom_bank_a, mmu.mbc6_rom_bank_b = ba, bb
                mmu.mbc6_ram_bank_a, mmu.mbc6_ram_bank_b = ra, rb
                mmu.mbc6_flash_a, mmu.mbc6_flash_b = bool(fa), bool(fb)
                mmu.mbc6_flash_enable, mmu.mbc6_flash_we = bool(fe), bool(fw)
                _need(4, "flash_len")
                (flash_len,) = struct.unpack_from('<I', data, pos)
                pos += 4
                if flash_len > MBC6_FLASH_SIZE:
                    raise ValueError("save state flash is implausibly large")
                _need(flash_len, "flash")
                if flash_len:
                    if len(mmu.flash_data) < flash_len:
                        mmu.flash_data = bytearray(flash_len)
                    mmu.flash_data[:flash_len] = data[pos:pos + flash_len]
                    if flash_len < len(mmu.flash_data):
                        mmu.flash_data[flash_len:] = b'\x00' * (len(mmu.flash_data) - flash_len)
                pos += flash_len
                fmt = '<BBHHBBBB'
                _need(struct.calcsize(fmt), "mbc7")
                (en2, lat, lx, ly, est, edo, ewe, eaddr) = struct.unpack_from(fmt, data, pos)
                pos += struct.calcsize(fmt)
                mmu.mbc7_ram_enable2 = bool(en2)
                mmu.mbc7_latch_ready = bool(lat)
                mmu.mbc7_latch_x, mmu.mbc7_latch_y = lx, ly
                mmu.eeprom_state = est
                mmu.eeprom_do = edo & 1
                mmu.eeprom_write_en = bool(ewe)
                mmu.eeprom_addr = eaddr
                if mmu.mbc_type == 0x20:
                    mmu._remap_rom_bank()
            if ver >= 6:
                _need(self._STATE_EXTRA_TAIL_V6, "extra")
                (gdma_stall, speed_rem, bootrom_on,
                 js0, js1, js2, js3, js4, dp0, dp1,
                 e_cs, e_clk, e_bits, e_shift) = struct.unpack_from(
                    '<HBBBBBBBBBBBBH', data, pos)
                pos += self._STATE_EXTRA_TAIL_V6
                mmu.gdma_stall = gdma_stall
                self.speed_remainder = speed_rem & 1
                mmu.bootrom_enabled = bool(bootrom_on)
                mmu._joy_src = [js0, js1, js2, js3, js4]
                mmu._dpad_last = [dp0, dp1]
                mmu.eeprom_cs = bool(e_cs)
                mmu.eeprom_clk = bool(e_clk)
                mmu.eeprom_bits = e_bits
                mmu.eeprom_shift = e_shift
                mmu._recompute_joypad()
                if ver >= 7:
                    _need(5 + 16 + 2, "sgb_fsm")
                    (sgb_pkt, sgb_bits, sgb_cmd, sgb_left) = struct.unpack_from(
                        '<BHBB', data, pos)
                    pos += 5
                    mmu.sgb_in_packet = bool(sgb_pkt)
                    mmu.sgb_bit_count = sgb_bits
                    mmu.sgb_cmd = sgb_cmd
                    mmu.sgb_packets_left = sgb_left
                    mmu.sgb_packet[:] = data[pos:pos + 16]
                    pos += 16
                    (cmd_len,) = struct.unpack_from('<H', data, pos)
                    pos += 2
                    if cmd_len > self._STATE_SGB_CMD_MAX:
                        raise ValueError("save state sgb_cmd_data is implausibly large")
                    _need(cmd_len, "sgb_cmd_data")
                    mmu.sgb_cmd_data = bytearray(data[pos:pos + cmd_len])
                    pos += cmd_len
            elif ver >= 5:
                _need(self._STATE_TIMING_TAIL_V5, "timing")
                gdma_stall, speed_rem = struct.unpack_from('<HB', data, pos)
                pos += self._STATE_TIMING_TAIL_V5
                mmu.gdma_stall = gdma_stall
                self.speed_remainder = speed_rem & 1
                mmu._joy_src = [saved_joypad, 0xFF, 0xFF, 0xFF, 0xFF]
                mmu._dpad_last = [0, 2]
                mmu._recompute_joypad()
            else:
                mmu.gdma_stall = 0
                self.speed_remainder = 0
                mmu._joy_src = [saved_joypad, 0xFF, 0xFF, 0xFF, 0xFF]
                mmu._dpad_last = [0, 2]
                mmu._recompute_joypad()
            self._restore_post_load(ver)
            apu.drain()
            if hasattr(self, '_audio_pending'):
                self._audio_pending.clear()
            self._av_start = time.perf_counter()
            self._sync_samples = 0
            self._sync_frames = 0
            logging.info(f"Loaded state from slot {slot}: {os.path.basename(path)}")
            return True
        except OSError as e:
            logging.warning(f"Could not load state: {e}")
            if snap is not None:
                self._state_restore(snap)
            self._last_state_error = 'io'
            self._last_state_detail = str(e)[:60]
            return False
        except (struct.error, ValueError, IndexError) as e:
            logging.warning(f"Could not load state: {e}")
            if snap is not None:
                self._state_restore(snap)
            self._last_state_error = 'corrupt'
            self._last_state_detail = str(e)[:60]
            return False

    def _pump_audio(self):
        """Feed the mixer from the pending-audio deque (channel + one queue slot).
        Drops stale queued buffers when the queue backs up to avoid drift."""
        if not self.audio_enabled:
            return
        ch = self.audio_channel
        # If the queue is backing up, drop the oldest buffers to stay in sync.
        while len(self._audio_pending) > 4:
            self._audio_pending.popleft()
        while self._audio_pending:
            if not ch.get_busy():
                ch.play(self._audio_pending.popleft())
            elif ch.get_queue() is None:
                ch.queue(self._audio_pending.popleft())
                break
            else:
                break

    def _flush_audio(self):
        """Push one frame's worth of PCM to the mixer; return stereo sample count."""
        data = self.apu.drain()
        n_samples = len(data) // APU_BYTES_PER_STEREO_SAMPLE
        if not self.audio_enabled:
            return n_samples
        if len(data) < APU_BYTES_PER_STEREO_SAMPLE or len(data) & 3:
            return 0
        try:
            sound = pygame.mixer.Sound(buffer=data)
        except (pygame.error, TypeError):
            return 0
        self._audio_pending.append(sound)
        self._audio_refs.append(sound)
        if len(self._audio_refs) > 8:
            self._audio_refs = self._audio_refs[-4:]
        # Warn if the audio queue is backing up (emulator running faster than mixer)
        if len(self._audio_pending) >= 12:
            logging.debug("Audio queue nearly full (%d entries); mixer may be behind",
                          len(self._audio_pending))
        self._pump_audio()
        return n_samples

    def _pace_frame(self, samples_this_frame):
        """Sleep so wall-clock time tracks emulated audio (or frame count when silent)."""
        if self.fps_limit <= 0:
            return
        now = time.perf_counter()
        if self.audio_enabled and samples_this_frame > 0:
            self._sync_samples += samples_this_frame
            target = self._av_start + self._sync_samples / self.apu.SAMPLE_RATE
        else:
            self._sync_frames += 1
            target = self._av_start + self._sync_frames / self.fps_limit
        delay = target - now
        if delay > 0:
            time.sleep(delay * 0.95)
        elif delay < -2.0:
            # Fallen >2 seconds behind — reset the sync clock to avoid a lurch.
            self._av_start = now
            self._sync_samples = 0
            self._sync_frames = 0

    def _halt_cpu_cycles(self):
        """How many CPU T-cycles a halted CPU can sleep before the next event.

        Stops at the next PPU mode/scanline boundary (STAT/VBlank sources) and
        at the next TIMA increment so interrupt wake-up is not delayed.
        """
        ppu = self.ppu
        mem = self.mmu.memory
        sd = ppu.scanline_dot
        remain = DOTS_PER_SCANLINE - sd
        if remain < 1:
            remain = 1
        if ppu.lcd_was_on and (mem[0xFF40] & 0x80):
            ly = mem[0xFF44]
            if ly < 144:
                if sd < MODE3_START_DOT:
                    remain = min(remain, MODE3_START_DOT - sd)
                else:
                    m3 = MODE3_START_DOT + ppu.mode3_duration
                    if sd < m3:
                        remain = min(remain, m3 - sd)
        if self.mmu.dma_remaining > 0:
            remain = min(remain, 4)
        if self.mmu.serial_bits_left > 0 and (self.mmu.serial_control & 0x81) == 0x81:
            until = self.mmu._serial_bit_period() - self.mmu.serial_cycle_accum
            if until < 1:
                until = 1
            remain = min(remain, until)
        tac = mem[0xFF07]
        if tac & 0x04:
            rate = self.timers._TIMA_RATES[tac & 0x03]
            until = rate - self.timers.tima_accum
            if until < 1:
                until = 1
            remain = min(remain, until)
        if self.mmu.key1 & 0x80:
            cpu_cycles = remain * 2 - self.speed_remainder
            if cpu_cycles < 1:
                cpu_cycles = 1
        else:
            cpu_cycles = remain
        return cpu_cycles

    def step_all(self):
        """Execute one CPU step and propagate cycles to PPU, timers, and APU."""
        mmu = self.mmu
        cpu = self.cpu
        mem = mmu.memory
        if (cpu.halted and mmu.gdma_stall == 0
                and not (mem[0xFFFF] & mem[0xFF0F])):
            cpu_cycles = self._halt_cpu_cycles()
        else:
            cpu_cycles = cpu.step()
            if mmu.gdma_stall > 0:
                cpu_cycles += mmu.gdma_stall
                mmu.gdma_stall = 0
        if mmu.dma_remaining > 0:
            mmu._dma_advance(cpu_cycles)
        # CGB double-speed (KEY1)
        if mmu.key1 & 0x80:
            self.speed_remainder += cpu_cycles
            dot_cycles = self.speed_remainder >> 1
            self.speed_remainder &= 1
        else:
            dot_cycles = cpu_cycles
        self.ppu.step(dot_cycles)
        # Timers: DIV update is inlined (always runs); TIMA only when TAC enabled
        timers = self.timers
        old_div = timers.div_counter
        div = old_div + cpu_cycles
        timers.div_counter = div
        mem[0xFF04] = (div >> 8) & 0xFF
        tac = mem[0xFF07]
        if tac & 0x04:
            timers._tima_step(cpu_cycles, tac)
        if mmu.serial_bits_left > 0:
            mmu._serial_step(cpu_cycles)
        self.apu.step(dot_cycles, div_old=old_div, div_new=div,
                      double_speed=bool(mmu.key1 & 0x80))
        if self._has_rtc:
            self._rtc_cycle_accum += cpu_cycles
            if self._rtc_cycle_accum >= CYCLES_PER_FRAME:
                self._rtc_cycle_accum -= CYCLES_PER_FRAME
                mmu._rtc_update()
        return dot_cycles

    def run(self):
        """Main execution loop.  Returns to caller when the user exits to the menu
        (via the pause menu) or closes the window."""
        self.running = True
        self.paused = False
        self._av_start = time.perf_counter()
        self._sync_samples = 0
        self._sync_frames = 0
        self._cycle_carry = 0
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0
        if pygame:
            pygame.key.set_repeat()  # disable key-repeat so held keys don't retrigger

        while self.running:
            cycles_this_frame = self._cycle_carry
            self._cycle_carry = 0
            while cycles_this_frame < CYCLES_PER_FRAME:
                cycles_this_frame += self.step_all()
            self._cycle_carry = cycles_this_frame - CYCLES_PER_FRAME
            self._frame_index += 1

            samples_this_frame = 0
            if pygame:
                self.handle_events()
                self._apply_turbo()
                skip_blit = self._fast_forward and (self._frame_index & 3)
                if not skip_blit:
                    self.render()
                samples_this_frame = self._flush_audio()

            if self.paused and pygame:
                self._pause_menu_loop()
                continue

            self._tick_fps()
            if self._status_ttl > 0:
                self._status_ttl -= 1
            if not self._fast_forward:
                self._pace_frame(samples_this_frame)

        self._save_sav()
        if self.mmu.link_cable is not None:
            self.mmu.link_cable.close()

    def handle_events(self):
        """Process window events and Joypad / Gamepad inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._open_pause_menu()
                    return
                elif event.key == pygame.K_F6:
                    self._state_toast('save', 0, self.save_state(0))
                elif event.key == pygame.K_F7:
                    ok = self.load_state(0)
                    self._state_toast('load', 0, ok)
                    if ok:
                        self.render()
                elif event.key == pygame.K_F8:
                    self._state_toast('save', 1, self.save_state(1))
                elif event.key == pygame.K_F9:
                    ok = self.load_state(1)
                    self._state_toast('load', 1, ok)
                    if ok:
                        self.render()
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_F3:
                    self._show_fps = not self._show_fps
                    self._status_msg = "FPS overlay on" if self._show_fps else "FPS overlay off"
                    self._status_ttl = 60
                elif event.key == pygame.K_F4:
                    self._show_input = not self._show_input
                    self._status_msg = "Input overlay on" if self._show_input else "Input overlay off"
                    self._status_ttl = 60
                elif event.key == pygame.K_TAB:
                    self._fast_forward = True
                elif event.key == pygame.K_r and (event.mod & pygame.KMOD_CTRL):
                    self.soft_reset()
                elif event.key in KEY_TO_JOYPAD_BIT:
                    self.mmu.set_joypad_button(
                        KEY_TO_JOYPAD_BIT[event.key], True, _JOY_SRC_KB)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_TAB:
                    self._fast_forward = False
                elif event.key in KEY_TO_JOYPAD_BIT:
                    self.mmu.set_joypad_button(
                        KEY_TO_JOYPAD_BIT[event.key], False, _JOY_SRC_KB)
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 9:
                    self._fast_forward = True
                    continue
                if event.button == 7 and not (self.mmu._joy_src[_JOY_SRC_BTN] & (1 << 6)):
                    # Select is already held: Select+Start opens the pause menu
                    # instead of injecting Start into the game.
                    self._open_pause_menu()
                    return
                bit = _GAMEPAD_BUTTON_MAP.get(event.button)
                if bit is not None:
                    self.mmu.set_joypad_button(bit, True, _JOY_SRC_BTN)
            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 9:
                    self._fast_forward = False
                bit = _GAMEPAD_BUTTON_MAP.get(event.button)
                if bit is not None:
                    self.mmu.set_joypad_button(bit, False, _JOY_SRC_BTN)
            elif event.type == pygame.JOYHATMOTION:
                for bit, pressed in _joystick_dpad_from_hat(event):
                    self.mmu.set_joypad_button(bit, pressed, _JOY_SRC_HAT)
            elif event.type == pygame.JOYAXISMOTION:
                for bit, pressed in _joystick_dpad_from_axis(event):
                    self.mmu.set_joypad_button(bit, pressed, _JOY_SRC_AXIS)
            elif event.type in (getattr(pygame, 'JOYDEVICEADDED', -1),
                                getattr(pygame, 'JOYDEVICEREMOVED', -2)):
                _init_joysticks()
                msg = _gamepad_hotplug_message(event)
                if msg:
                    self._status_msg = msg
                    self._status_ttl = 90

    def _sync_held_inputs(self):
        """Re-read currently held keyboard / gamepad state after the pause menu."""
        kb = 0xFF
        pressed = pygame.key.get_pressed()
        for key, bit in KEY_TO_JOYPAD_BIT.items():
            if key == pygame.K_ESCAPE:
                continue
            if bit == 7:  # don't stick Start from Enter used to confirm Resume
                continue
            if pressed[key]:
                kb &= ~(1 << bit)
        hat = 0xFF
        axis = 0xFF
        padbtn = 0xFF
        try:
            count = pygame.joystick.get_count()
        except pygame.error:
            count = 0
        T = _GAMEPAD_AXIS_THRESHOLD
        for i in range(count):
            try:
                js = pygame.joystick.Joystick(i)
                for b, bit in _GAMEPAD_BUTTON_MAP.items():
                    if b < js.get_numbuttons() and js.get_button(b):
                        padbtn &= ~(1 << bit)
                if js.get_numhats() > 0:
                    x, y = js.get_hat(0)
                    if x == 1:  hat &= ~0x01
                    if x == -1: hat &= ~0x02
                    if y == 1:  hat &= ~0x04
                    if y == -1: hat &= ~0x08
                naxes = js.get_numaxes()
                def _axis_pair(idx, pos_bit, neg_bit, mask):
                    if idx >= naxes:
                        return mask
                    val = js.get_axis(idx)
                    if val > T:
                        return mask & ~(1 << pos_bit)
                    if val < -T:
                        return mask & ~(1 << neg_bit)
                    return mask
                axis = _axis_pair(0, 0, 1, axis)
                axis = _axis_pair(1, 3, 2, axis)
                axis = _axis_pair(6, 0, 1, axis)
                axis = _axis_pair(7, 3, 2, axis)
            except pygame.error:
                continue
        self.mmu._joy_src = [kb, hat, axis, padbtn, 0xFF]
        self.mmu._recompute_joypad()

    def _apply_turbo(self):
        """Hold turbo A/B keys to auto-fire A/B at 30 Hz."""
        if not pygame:
            return
        pressed = pygame.key.get_pressed()
        turbo_a = turbo_b = False
        turbo_map = KEY_TO_TURBO_BIT or {
            pygame.K_q: 4, pygame.K_COMMA: 4,
            pygame.K_e: 5, pygame.K_PERIOD: 5,
        }
        for key, bit in turbo_map.items():
            if key in KEY_TO_JOYPAD_BIT:
                continue
            try:
                held = pressed[key]
            except IndexError:
                continue
            if held:
                if bit == 4:
                    turbo_a = True
                elif bit == 5:
                    turbo_b = True
        self._turbo_phase ^= 1
        fire = bool(self._turbo_phase)
        mask = 0xFF
        if turbo_a and fire:
            mask &= ~(1 << 4)
        if turbo_b and fire:
            mask &= ~(1 << 5)
        src = self.mmu._joy_src
        if len(src) < 5:
            src.extend([0xFF] * (5 - len(src)))
        if src[_JOY_SRC_TURBO] != mask:
            src[_JOY_SRC_TURBO] = mask
            self.mmu._recompute_joypad()

    def _tick_fps(self):
        self._fps_frames += 1
        now = time.perf_counter()
        dt = now - self._fps_t0
        if dt >= 0.4:
            self._fps_value = self._fps_frames / dt
            self._fps_frames = 0
            self._fps_t0 = now

    def soft_reset(self):
        """Reset CPU/PPU/APU to post-boot state without wiping cartridge RAM."""
        cpu = self.cpu
        cpu.halted = False
        cpu.interrupts_master_enabled = False
        cpu.ime_pending = False
        cpu.halt_bug_pending = False
        cpu._logged_opcodes.clear()
        cpu.reg.sp = 0xFFFE
        cpu.reg.pc = 0x0100
        if self.mmu.is_cgb:
            cpu.reg.a = 0x11
            cpu.reg.f = 0x80
            cpu.reg.b = 0x00
            cpu.reg.c = 0x00
            cpu.reg.d = 0xFF
            cpu.reg.e = 0x56
            cpu.reg.h = 0x00
            cpu.reg.l = 0x0D
        else:
            cpu.reg.a = 0x01
            cpu.reg.f = 0xB0
            cpu.reg.b = 0x00
            cpu.reg.c = 0x13
            cpu.reg.d = 0x00
            cpu.reg.e = 0xD8
            cpu.reg.h = 0x01
            cpu.reg.l = 0x4D
        self.mmu.bootrom_enabled = False
        self._apply_post_boot_io()
        ppu = self.ppu
        ppu.scanline_dot = 0
        ppu.mode = 2
        ppu.lcd_was_on = False
        ppu.window_line_counter = 0
        ppu.window_active = False
        ppu.prev_stat_irq = False
        self.timers.div_counter = 0
        self.timers.tima_accum = 0
        if hasattr(self, '_audio_pending'):
            self._audio_pending.clear()
        self.speed_remainder = 0
        self._cycle_carry = 0
        self._av_start = time.perf_counter()
        self._sync_samples = 0
        self._sync_frames = 0
        self._status_msg = "Reset"
        self._status_ttl = 90

    def _draw_hud(self):
        """FPS / input / fast-forward overlays drawn after the scaled frame."""
        if not pygame or getattr(self, 'screen', None) is None:
            return
        try:
            ox, oy, pw, ph = getattr(
                self, '_present_rect',
                (0, 0, self.screen.get_width(), self.screen.get_height()))
            f = get_font(18 if ph >= 400 else 14)
            right = ox + pw - 8
            badges = []
            lc = self.mmu.link_cable
            if lc is not None and lc.is_connected:
                badges.append(f.render("LINK", True, MENU_HI))
            if getattr(self, '_fast_forward', False):
                badges.append(f.render("FF", True, MENU_HI))
            if getattr(self, '_show_fps', False):
                badges.append(f.render(f"{getattr(self, '_fps_value', 0.0):.0f} fps", True, MENU_HI))
            if getattr(self, '_show_input', False):
                jp = self.mmu.joypad_buttons
                names = (('R', 0), ('L', 1), ('U', 2), ('D', 3),
                         ('A', 4), ('B', 5), ('Sel', 6), ('Sta', 7))
                parts = []
                for label, bit in names:
                    pressed = (jp & (1 << bit)) == 0
                    parts.append(f.render(label, True, MENU_HI if pressed else MENU_DIM))
                gap = 6
                pad_x = 8
                row_w = pad_x + sum(ps.get_width() for ps in parts) + gap * (len(parts) - 1) + pad_x
                row_h = f.get_height() + 6
                row = pygame.Surface((max(row_w, 12), row_h))
                row.fill(MENU_BG)
                rx = pad_x
                for ps in parts:
                    row.blit(ps, (rx, 3))
                    rx += ps.get_width() + gap
                badges.append(row)
            if not badges:
                return
            gap = 6
            pad = 6
            total_w = pad + sum(
                b.get_width() + pad for b in badges) + gap * (len(badges) - 1)
            bar_h = max(b.get_height() + 6 for b in badges)
            bar = pygame.Surface((total_w, bar_h))
            bar.fill(MENU_BG)
            bar.set_alpha(210)
            x = right - total_w
            y = oy + 8
            self.screen.blit(bar, (x, y))
            pygame.draw.rect(self.screen, MENU_DIM, (x, y, total_w, bar_h), 1)
            bx = x + pad
            for badge in badges:
                by = y + (bar_h - badge.get_height()) // 2
                self.screen.blit(badge, (bx, by))
                bx += badge.get_width() + gap
        except (pygame.error, AttributeError):
            pass

    # ── In-game pause menu ────────────────────────────────────────────
    PAUSE_ITEMS = ["Resume", "Save States...", "Settings", "Exit to Menu"]
    SAVE_STATE_ACTIONS = (('save', 0), ('load', 0), ('save', 1), ('load', 1))

    def _save_state_items(self):
        items = []
        for action, slot in self.SAVE_STATE_ACTIONS:
            stamp = _slot_status_suffix(self._state_path(slot))
            verb = "Save" if action == 'save' else "Load"
            items.append(f"{verb} Slot {slot}  ·  {stamp}")
        return items

    def _open_pause_menu(self):
        """Enter the paused state; the main loop hands control to _pause_menu_loop."""
        self.paused = True
        self.pause_cursor = 0
        self.pause_save_cursor = 0
        self.pause_settings_cursor = 0
        self.pause_exit_cursor = 0
        self.pause_controls_cursor = 0
        self.pause_controls_scroll = 0
        self.controls_capture = None
        self._fast_forward = False

    def _pause_status(self, msg):
        self._pause_msg = msg
        self._pause_msg_ttl = 120

    def _pause_settings_items(self):
        si = self._set_idx
        raw = [
            f"Window Scale: {WINDOW_SCALE_OPTIONS[si['scale']][0]}",
            f"Display: {'Fullscreen' if self.fullscreen else 'Window'}",
            f"Frame Rate: {FPS_LIMIT_OPTIONS[si['fps']][0]}",
            f"Audio: {AUDIO_OPTIONS[si['audio']][0]}",
            f"Volume: {VOLUME_OPTIONS[si['volume']][0]}",
            f"Palette: {PALETTE_LIST[si['palette']][0]}",
            f"Filter: {FILTER_OPTIONS[si['filter']][0]}",
            f"Shader: {SHADER_LIST[si['shader']][0]}",
            "Controls...",
        ]
        items = []
        for i, text in enumerate(raw):
            selected = i == self.pause_settings_cursor
            sid = SETTINGS_ROW_IDS[i] if i < len(SETTINGS_ROW_IDS) else None
            if sid == 'controls':
                items.append(_decorate_submenu_row(text, selected))
            else:
                items.append(_decorate_cyclic_setting(text, selected))
        return items

    def _cycle_pause_setting(self, cursor, direction=1):
        """Cycle the setting under the cursor forward (1) or backward (-1).
        Returns True if the window was resized (so the backdrop must be recaptured)."""
        si = self._set_idx
        resized = False
        sid = SETTINGS_ROW_IDS[cursor] if 0 <= cursor < len(SETTINGS_ROW_IDS) else None
        if sid == 'scale':
            n = len(WINDOW_SCALE_OPTIONS)
            si['scale'] = (si['scale'] + direction) % n
            self.window_scale = WINDOW_SCALE_OPTIONS[si['scale']][1]
            if self.fullscreen:
                self._pause_status("Scale applies in windowed mode")
            else:
                self._apply_game_display()
                resized = True
        elif sid == 'display':
            self.fullscreen = not self.fullscreen
            self._apply_game_display()
            resized = True
        elif sid == 'fps':
            n = len(FPS_LIMIT_OPTIONS)
            si['fps'] = (si['fps'] + direction) % n
            self.fps_limit = FPS_LIMIT_OPTIONS[si['fps']][1]
        elif sid == 'audio':
            n = len(AUDIO_OPTIONS)
            si['audio'] = (si['audio'] + direction) % n
            self._audio_on = AUDIO_OPTIONS[si['audio']][1]
            if self._audio_on:
                self._init_audio()
                self._set_volume(VOLUME_OPTIONS[si['volume']][1])
            else:
                if self.audio_channel is not None:
                    try:
                        self.audio_channel.stop()
                    except pygame.error:
                        pass
                self.audio_enabled = False
        elif sid == 'volume':
            n = len(VOLUME_OPTIONS)
            si['volume'] = (si['volume'] + direction) % n
            self._set_volume(VOLUME_OPTIONS[si['volume']][1])
        elif sid == 'palette':
            n = len(PALETTE_LIST)
            si['palette'] = (si['palette'] + direction) % n
            self.ppu.set_palette(PALETTE_LIST[si['palette']][1])
        elif sid == 'filter':
            n = len(FILTER_OPTIONS)
            si['filter'] = (si['filter'] + direction) % n
            self.smooth_scale = FILTER_OPTIONS[si['filter']][1]
        elif sid == 'shader':
            n = len(SHADER_LIST)
            si['shader'] = (si['shader'] + direction) % n
            self.shader = SHADER_LIST[si['shader']][1]
            self._prev_shader_frame = None
        elif sid == 'controls':
            return False
        else:
            return False
        if not _save_config(dict(
            window_scale=WINDOW_SCALE_OPTIONS[si['scale']][1],
            fullscreen=self.fullscreen,
            fps_limit=FPS_LIMIT_OPTIONS[si['fps']][1],
            audio_enabled=AUDIO_OPTIONS[si['audio']][1],
            volume=VOLUME_OPTIONS[si['volume']][1],
            palette=si['palette'],
            smooth_scale=FILTER_OPTIONS[si['filter']][1],
            shader=si['shader'],
            wasd_enabled=self.wasd_enabled,
            key_bindings=self.key_bindings,
            turbo_bindings=self.turbo_bindings,
        )):
            self._pause_status("Could not save settings")
        return resized

    def _capture_pause_backdrop(self):
        """Render the current frame, then return a dimmed copy to sit behind the menu."""
        self.render(overlays=False)
        backdrop = self.screen.copy()
        veil = pygame.Surface(backdrop.get_size())
        veil.fill((0, 0, 0))
        veil.set_alpha(160)
        backdrop.blit(veil, (0, 0))
        return backdrop

    def _pause_menu_loop(self):
        """Blocking loop that runs while the game is paused. Halts emulation and
        audio, shows the overlay menu, and returns once the player resumes or exits."""
        # Release every joypad button so the game doesn't see a stuck input.
        self.mmu.release_all_joypad()
        if self.audio_channel is not None:
            try:
                self.audio_channel.stop()
            except pygame.error:
                pass
        self._audio_pending.clear()
        self.apu.drain()
        pygame.event.clear()

        backdrop = self._capture_pause_backdrop()
        clock = pygame.time.Clock()
        page = "pause"
        self._stick_nav.reset()
        while self.paused and self.running:
            page, backdrop = self._handle_pause_events(page, backdrop)
            if not self.paused or not self.running:
                break
            if self._pause_msg_ttl > 0:
                self._pause_msg_ttl -= 1
            self._render_pause_page(page, backdrop)
            clock.tick(60)

        # Resume cleanly: re-sync held keys to the joypad and reset the A/V clock
        # so frame pacing doesn't try to "catch up" on the paused wall-clock time.
        if self.running:
            self._sync_held_inputs()
        pygame.event.clear()
        self._av_start = time.perf_counter()
        self._sync_samples = 0
        self._sync_frames = 0

    def _pause_quick_state(self, action, slot, backdrop):
        if action == 'save':
            ok = self.save_state(slot)
            detail = getattr(self, '_last_state_detail', None)
            self._pause_status(self._format_state_message(
                'save', slot, None if ok else getattr(self, '_last_state_error', 'io'), detail))
            return backdrop
        if self.load_state(slot):
            self._pause_status(self._format_state_message('load', slot, None))
            return self._capture_pause_backdrop()
        detail = getattr(self, '_last_state_detail', None)
        self._pause_status(self._format_state_message(
            'load', slot, getattr(self, '_last_state_error', 'corrupt'), detail))
        return backdrop

    def _pause_persist_controls(self):
        self.key_bindings, self.turbo_bindings = _rebuild_input_maps(
            self.key_bindings, self.wasd_enabled, self.turbo_bindings)
        if not _save_config({
            'key_bindings': self.key_bindings,
            'turbo_bindings': self.turbo_bindings,
            'wasd_enabled': self.wasd_enabled,
        }):
            self._pause_status("Could not save settings")
            return False
        return True

    def _overlay_host(self):
        r = getattr(self, '_present_rect', None)
        if r and r[2] >= 160 and r[3] >= 144:
            return r
        if getattr(self, 'screen', None) is None:
            return (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        return (0, 0, self.screen.get_width(), self.screen.get_height())

    def _handle_pause_events(self, page, backdrop):
        actions = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.paused = False
                return page, backdrop
            plug = _gamepad_hotplug_message(event)
            if event.type in (getattr(pygame, 'JOYDEVICEADDED', -1),
                              getattr(pygame, 'JOYDEVICEREMOVED', -2)):
                _init_joysticks()
                if plug:
                    self._pause_status(plug)
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                    backdrop = self._capture_pause_backdrop()
                    continue
                if event.key == pygame.K_r and (event.mod & pygame.KMOD_CTRL):
                    self.soft_reset()
                    backdrop = self._capture_pause_backdrop()
                    self._pause_status("Reset")
                    continue
                if event.key in (pygame.K_F6, pygame.K_F7, pygame.K_F8, pygame.K_F9):
                    quick = {
                        pygame.K_F6: ('save', 0), pygame.K_F7: ('load', 0),
                        pygame.K_F8: ('save', 1), pygame.K_F9: ('load', 1),
                    }
                    act, slot = quick[event.key]
                    backdrop = self._pause_quick_state(act, slot, backdrop)
                    continue
            if page == "controls" and self.controls_capture is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.controls_capture = None
                    else:
                        name = _normalize_key_name(pygame.key.name(event.key))
                        target = self.controls_capture
                        if target in TURBO_BUTTON_KEYS:
                            ok = _assign_binding_key(
                                self.turbo_bindings, target, name,
                                other_maps=[self.key_bindings])
                        else:
                            ok = _assign_binding_key(
                                self.key_bindings, target, name,
                                other_maps=[self.turbo_bindings])
                        if ok:
                            self._pause_persist_controls()
                            label = target.replace('_', ' ').title()
                            self._pause_status(f"{label} -> {_key_display_name(name)}")
                        else:
                            self._pause_status(_RESERVED_KEY_MSG)
                        self.controls_capture = None
                continue
            if event.type == pygame.KEYDOWN:
                actions.append(event.key)
            elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION):
                ga = _gamepad_menu_action(event)
                if ga:
                    actions.append(ga)
            elif event.type == pygame.MOUSEMOTION:
                idx = _hit_list_index(self._overlay_hits, event.pos)
                if idx is not None:
                    self._pause_hover(page, idx)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                idx = _hit_list_index(self._overlay_hits, event.pos)
                if idx is not None:
                    self._pause_hover(page, idx)
                    actions.append('select')
            elif event.type == getattr(pygame, 'MOUSEWHEEL', -3):
                if getattr(event, 'y', 0) > 0:
                    actions.append('up')
                elif getattr(event, 'y', 0) < 0:
                    actions.append('down')
        if self.controls_capture is None:
            stick = self._stick_nav.update(*_read_analog_menu_dirs(), pygame.time.get_ticks())
            if stick:
                actions.append(stick)
        for raw in actions:
            page, backdrop = self._dispatch_pause(page, _menu_nav_action(raw), backdrop)
        return page, backdrop

    def _pause_hover(self, page, idx):
        if page == "pause" and 0 <= idx < len(self.PAUSE_ITEMS):
            self.pause_cursor = idx
        elif page == "settings":
            items = self._pause_settings_items()
            if 0 <= idx < len(items):
                self.pause_settings_cursor = idx
        elif page == "save_states":
            items = self._save_state_items()
            if 0 <= idx < len(items):
                self.pause_save_cursor = idx
        elif page == "controls":
            items = self._pause_controls_items()
            if 0 <= idx < len(items):
                self.pause_controls_cursor = idx
        elif page == "confirm_exit" and idx in (0, 1):
            self.pause_exit_cursor = idx

    def _dispatch_pause(self, page, action, backdrop):
        if page == "pause":
            if action in (pygame.K_UP, 'up'):
                self.pause_cursor = (self.pause_cursor - 1) % len(self.PAUSE_ITEMS)
            elif action in (pygame.K_DOWN, 'down'):
                self.pause_cursor = (self.pause_cursor + 1) % len(self.PAUSE_ITEMS)
            elif action == pygame.K_RETURN or action == 'select':
                page, backdrop = self._activate_pause_item(page, backdrop)
            elif action == pygame.K_ESCAPE or action == 'back':
                self.paused = False
        elif page == "settings":
            items = self._pause_settings_items()
            sid = SETTINGS_ROW_IDS[self.pause_settings_cursor] if (
                0 <= self.pause_settings_cursor < len(SETTINGS_ROW_IDS)) else None
            if action in (pygame.K_UP, 'up'):
                self.pause_settings_cursor = (self.pause_settings_cursor - 1) % len(items)
            elif action in (pygame.K_DOWN, 'down'):
                self.pause_settings_cursor = (self.pause_settings_cursor + 1) % len(items)
            elif action == pygame.K_RETURN or action == 'select':
                if sid == 'controls':
                    self.pause_controls_cursor = 0
                    self.pause_controls_scroll = 0
                    self.controls_capture = None
                    page = "controls"
                elif self._cycle_pause_setting(self.pause_settings_cursor, 1):
                    backdrop = self._capture_pause_backdrop()
            elif action in (pygame.K_RIGHT, 'right'):
                if sid != 'controls':
                    if self._cycle_pause_setting(self.pause_settings_cursor, 1):
                        backdrop = self._capture_pause_backdrop()
            elif action in (pygame.K_LEFT, 'left'):
                if sid != 'controls':
                    if self._cycle_pause_setting(self.pause_settings_cursor, -1):
                        backdrop = self._capture_pause_backdrop()
            elif action == pygame.K_ESCAPE or action == 'back':
                page = "pause"
        elif page == "save_states":
            items = self._save_state_items()
            if action in (pygame.K_UP, 'up'):
                self.pause_save_cursor = (self.pause_save_cursor - 1) % len(items)
            elif action in (pygame.K_DOWN, 'down'):
                self.pause_save_cursor = (self.pause_save_cursor + 1) % len(items)
            elif action == pygame.K_RETURN or action == 'select':
                page, backdrop = self._activate_save_state_item(backdrop)
            elif action == pygame.K_ESCAPE or action == 'back':
                page = "pause"
        elif page == "controls":
            items = self._pause_controls_items()
            ox, oy, w, h = self._overlay_host()
            cap = _overlay_scroll_capacity(w, h, has_status=True)
            if action in (pygame.K_UP, 'up'):
                self.pause_controls_cursor = _advance_controls_cursor(
                    self.pause_controls_cursor, -1, len(items))
                self.pause_controls_scroll, _ = _sync_list_scroll(
                    self.pause_controls_cursor, self.pause_controls_scroll, cap, len(items))
            elif action in (pygame.K_DOWN, 'down'):
                self.pause_controls_cursor = _advance_controls_cursor(
                    self.pause_controls_cursor, 1, len(items))
                self.pause_controls_scroll, _ = _sync_list_scroll(
                    self.pause_controls_cursor, self.pause_controls_scroll, cap, len(items))
            elif action in (pygame.K_LEFT, pygame.K_RIGHT, 'left', 'right'):
                if self.pause_controls_cursor == CONTROLS_WASD_ROW:
                    self.wasd_enabled = not self.wasd_enabled
                    self.key_bindings = _sanitize_key_bindings(
                        self.key_bindings, self.wasd_enabled)
                    self._pause_persist_controls()
            elif action == pygame.K_RETURN or action == 'select':
                if self.pause_controls_cursor < CONTROLS_WASD_ROW:
                    self.controls_capture = JOYPAD_BUTTON_KEYS[self.pause_controls_cursor]
                elif self.pause_controls_cursor == CONTROLS_WASD_ROW:
                    self.wasd_enabled = not self.wasd_enabled
                    self.key_bindings = _sanitize_key_bindings(
                        self.key_bindings, self.wasd_enabled)
                    self._pause_persist_controls()
                elif self.pause_controls_cursor == CONTROLS_TURBO_A_ROW:
                    self.controls_capture = 'turbo_a'
                elif self.pause_controls_cursor == CONTROLS_TURBO_B_ROW:
                    self.controls_capture = 'turbo_b'
                elif self.pause_controls_cursor == CONTROLS_RESET_ROW:
                    self.wasd_enabled = True
                    self.key_bindings = _default_key_bindings(True)
                    self.turbo_bindings = {k: list(v) for k, v in DEFAULT_TURBO_BINDINGS.items()}
                    if self._pause_persist_controls():
                        self._pause_status("Controls reset to default")
                elif self.pause_controls_cursor == CONTROLS_GAMEPAD_ROW:
                    self._pause_status(_connected_gamepad_label())
            elif action == pygame.K_ESCAPE or action == 'back':
                self.controls_capture = None
                page = "settings"
        elif page == "confirm_exit":
            if action in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                          'up', 'down', 'left', 'right'):
                self.pause_exit_cursor ^= 1
            elif action == pygame.K_RETURN or action == 'select':
                if self.pause_exit_cursor == 1:
                    self.running = False
                    self.paused = False
                else:
                    page = "pause"
            elif action == pygame.K_ESCAPE or action == 'back':
                page = "pause"
        return page, backdrop

    def _activate_pause_item(self, page, backdrop):
        choice = self.PAUSE_ITEMS[self.pause_cursor]
        if choice == "Resume":
            self.paused = False
        elif choice == "Save States...":
            self.pause_save_cursor = 0
            page = "save_states"
        elif choice == "Settings":
            self.pause_settings_cursor = 0
            page = "settings"
        elif choice == "Exit to Menu":
            self.pause_exit_cursor = 0
            page = "confirm_exit"
        return page, backdrop

    def _activate_save_state_item(self, backdrop):
        actions = self.SAVE_STATE_ACTIONS
        if not actions:
            return "save_states", backdrop
        idx = self.pause_save_cursor % len(actions)
        action, slot = actions[idx]
        if action == 'save':
            ok = self.save_state(slot)
            detail = getattr(self, '_last_state_detail', None)
            self._pause_status(self._format_state_message(
                'save', slot, None if ok else getattr(self, '_last_state_error', 'io'), detail))
        else:
            if self.load_state(slot):
                self._pause_status(self._format_state_message('load', slot, None))
                backdrop = self._capture_pause_backdrop()
            else:
                detail = getattr(self, '_last_state_detail', None)
                self._pause_status(self._format_state_message(
                    'load', slot, getattr(self, '_last_state_error', 'corrupt'), detail))
        return "save_states", backdrop

    def _pause_controls_items(self):
        items = [
            f"{JOYPAD_BUTTON_LABELS[i]}: {_binding_label(self.key_bindings, key)}"
            for i, key in enumerate(JOYPAD_BUTTON_KEYS)
        ]
        if self.controls_capture in JOYPAD_BUTTON_KEYS:
            idx = JOYPAD_BUTTON_KEYS.index(self.controls_capture)
            items[idx] = f"{JOYPAD_BUTTON_LABELS[idx]}: Press a key..."
        items.append(f"WASD as D-Pad: {'On' if self.wasd_enabled else 'Off'}")
        ta = "Press a key..." if self.controls_capture == 'turbo_a' else _binding_label(
            self.turbo_bindings, 'turbo_a')
        tb = "Press a key..." if self.controls_capture == 'turbo_b' else _binding_label(
            self.turbo_bindings, 'turbo_b')
        items.append(f"Turbo A: {ta}")
        items.append(f"Turbo B: {tb}")
        items.append("Reset to Default")
        items.append(_connected_gamepad_label())
        return items

    def _draw_overlay_menu(self, backdrop, title, items, cursor, hint, hint_hi=False,
                           scroll=0, status=None, status_hi=False):
        self.screen.blit(backdrop, (0, 0))
        ox, oy, w, h = self._overlay_host()
        has_status = bool(status)
        cap = _overlay_scroll_capacity(w, h, has_hint=bool(hint), has_status=has_status)
        n_items = max(len(items), 1)
        scroll, cap = _sync_list_scroll(cursor, scroll, cap, n_items)
        visible = items[scroll:scroll + cap] if items else [""]
        display_cursor = cursor - scroll
        L = _overlay_layout(w, h, len(visible), has_hint=bool(hint), has_status=has_status)
        px, py = ox + L['px'], oy + L['py']
        panel_w, panel_h = L['panel_w'], L['panel_h']
        panel = pygame.Surface((panel_w, panel_h))
        panel.fill(MENU_BG)
        panel.set_alpha(240)
        self.screen.blit(panel, (px, py))
        pygame.draw.rect(self.screen, MENU_HI, (px, py, panel_w, panel_h), 2)

        tf = get_font(L['title_size'])
        ts = tf.render(_fit_text(tf, title, panel_w - 16), True, MENU_HI)
        self.screen.blit(ts, (px + (panel_w - ts.get_width()) // 2, py + 6))

        itf = get_font(L['item_size'])
        scroll_font = get_font(max(11, L['item_size'] - 4))
        max_item_w = panel_w - 40
        list_y = py + L['title_band']
        self._overlay_hits = []
        if scroll > 0:
            more = scroll_font.render("^ more", True, MENU_DIM)
            self.screen.blit(more, (px + (panel_w - more.get_width()) // 2, list_y - 2))
        for i, item in enumerate(visible):
            iy = list_y + i * L['item_h']
            bar_h = max(12, L['item_h'] - 2)
            self._overlay_hits.append((px + 6, iy - 1, panel_w - 12, bar_h, scroll + i))
            label = _fit_text(itf, item, max_item_w)
            if i == display_cursor:
                _blit_selection_bar(self.screen, px + 6, iy - 1, panel_w - 12, bar_h)
                colour = MENU_HI
            else:
                colour = MENU_FG
            isf = itf.render(label, True, colour)
            ix = px + max(16, (panel_w - isf.get_width()) // 2)
            if i == display_cursor:
                cursor_surf = itf.render(">", True, MENU_HI)
                self.screen.blit(
                    cursor_surf,
                    (max(px + 10, ix - cursor_surf.get_width() - 6), iy))
            self.screen.blit(isf, (ix, iy))
        if scroll + cap < n_items:
            more = scroll_font.render("v more", True, MENU_DIM)
            vy = list_y + len(visible) * L['item_h'] - 2
            self.screen.blit(more, (px + (panel_w - more.get_width()) // 2, vy))

        y_footer = py + panel_h
        if has_status:
            sf = get_font(L['hint_size'])
            ss = sf.render(_fit_text(sf, status, panel_w - 16), True, MENU_HI if status_hi else MENU_DIM)
            y_footer -= L['status_band']
            self.screen.blit(ss, (px + (panel_w - ss.get_width()) // 2, y_footer + 2))
        if hint:
            hf = get_font(L['hint_size'])
            hs = hf.render(
                _fit_text(hf, hint, panel_w - 16),
                True, MENU_HI if hint_hi else MENU_DIM)
            self.screen.blit(
                hs,
                (px + (panel_w - hs.get_width()) // 2,
                 py + panel_h - L['hint_band'] + 4))
        pygame.display.flip()
        return scroll

    def _pause_hint(self, wide, narrow):
        _ox, _oy, w, _h = self._overlay_host()
        return wide if w >= 400 else narrow

    def _pause_title(self, fallback="Paused"):
        name = os.path.basename(self.mmu.rom_path or '')
        if not name:
            return fallback
        return f"{fallback} — {name}"

    def _render_pause_page(self, page, backdrop):
        status = self._pause_msg if self._pause_msg_ttl > 0 else None
        if page == "settings":
            self._draw_overlay_menu(
                backdrop, "Settings", self._pause_settings_items(),
                self.pause_settings_cursor,
                self._pause_hint("Left/Right or Enter: Change   Esc: Back",
                                 "Left/Right: Change   Esc: Back"),
                status=status, status_hi=True)
        elif page == "controls":
            if self.controls_capture:
                hint = self._pause_hint("Press a new key   Esc: Cancel",
                                        "Press a key   Esc: Cancel")
            else:
                hint = self._pause_hint(
                    "Enter: Remap / Toggle   Esc: Back   F11 fullscreen",
                    "Enter: Remap   Esc: Back")
            self.pause_controls_scroll = self._draw_overlay_menu(
                backdrop, "Controls", self._pause_controls_items(),
                self.pause_controls_cursor, hint,
                scroll=self.pause_controls_scroll, status=status, status_hi=True)
        elif page == "save_states":
            self._draw_overlay_menu(
                backdrop, "Save States", self._save_state_items(),
                self.pause_save_cursor,
                self._pause_hint("Enter/click: Save/Load   Esc: Back   F6-F9: Quick keys",
                                 "Enter: Save/Load   Esc: Back   F6-F9"),
                status=status, status_hi=True)
        elif page == "confirm_exit":
            self._draw_overlay_menu(
                backdrop, "Exit to Menu?",
                ["Keep Playing", "Exit to Menu"], self.pause_exit_cursor,
                self._pause_hint("Battery save writes on exit. F6/F8 = quick-save slots.",
                                 "Battery save on exit"),
                status=status, status_hi=True)
        else:
            title = self._pause_title("Paused")
            self._draw_overlay_menu(
                backdrop, title, self.PAUSE_ITEMS,
                self.pause_cursor,
                self._pause_hint(
                    "Up/Down or click: Move   Enter: Select   Esc: Resume   F6-F9",
                    "Enter: Select   Esc: Resume   F6-F9"),
                status=status, status_hi=True)

    def _apply_display_shader(self, fnp):
        """Run the active post-process shader using pre-allocated scratch buffers."""
        shader = self.shader
        if shader is _shader_none:
            return fnp
        out = self._shader_out
        if shader is _shader_lcd_ghost:
            prev = self._prev_shader_frame
            if prev is None:
                np.copyto(self._ghost_store, fnp)
                self._prev_shader_frame = self._ghost_store
                return fnp
            arr = _shader_lcd_ghost(
                fnp, prev, out=out,
                scratch=self._shader_f32, scratch_b=self._shader_f32_b)
            np.copyto(prev, fnp)
            return arr
        if shader is _shader_crt_scanlines:
            return _shader_crt_scanlines(fnp, out)
        if shader is _shader_gamma_warm:
            return _shader_gamma_warm(fnp, out, self._shader_f32)
        if shader is _shader_pixel_bloom:
            return _shader_pixel_bloom(
                fnp, out=out, acc=self._shader_f32, blend=self._shader_f32_b)
        return shader(fnp)

    def render(self, overlays=True):
        """Draws the PPU framebuffer to the Pygame screen."""
        if np is not None:
            # framebuffer holds packed 24-bit colours; unpack the whole frame
            # into the persistent (H, W, 3) uint8 buffer with vectorised shifts.
            packed = np.asarray(self.ppu.framebuffer, dtype=np.uint32).reshape(
                SCREEN_HEIGHT, SCREEN_WIDTH)
            fnp = self._frame_np
            fnp[:, :, 0] = (packed >> 16) & 0xFF
            fnp[:, :, 1] = (packed >> 8) & 0xFF
            fnp[:, :, 2] = packed & 0xFF
            arr = self._apply_display_shader(fnp)
            np.copyto(self._blit_np, arr.transpose(1, 0, 2))
            surf = pygame.surfarray.make_surface(self._blit_np)
        else:
            if pygame and not self._numpy_slow_warned:
                self._numpy_slow_warned = True
                self._status_msg = "Install numpy for faster rendering (pip install numpy)"
                self._status_ttl = 180
            surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            pxa = pygame.PixelArray(surf)
            fb = self.ppu.framebuffer
            for i in range(SCREEN_HEIGHT * SCREEN_WIDTH):
                pxa[i % SCREEN_WIDTH, i // SCREEN_WIDTH] = fb[i]
            pxa.close()
        dw, dh = self.screen.get_size()
        scale = max(1, min(dw // SCREEN_WIDTH, dh // SCREEN_HEIGHT))
        tw, th = SCREEN_WIDTH * scale, SCREEN_HEIGHT * scale
        if (tw, th) == (SCREEN_WIDTH, SCREEN_HEIGHT):
            scaled = surf
        elif self.smooth_scale:
            scaled = pygame.transform.smoothscale(surf, (tw, th))
        else:
            scaled = pygame.transform.scale(surf, (tw, th))
        ox, oy = (dw - tw) // 2, (dh - th) // 2
        if (ox, oy) != (0, 0) or (dw, dh) != (tw, th):
            self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, (ox, oy))
        self._present_rect = (ox, oy, tw, th)
        if overlays:
            # Transient status overlay (save/load messages) — bottom-left of the picture
            if getattr(self, '_status_ttl', 0) > 0 and getattr(self, '_status_msg', ''):
                try:
                    f = get_font(20 if th >= 400 else 14)
                    s = f.render(self._status_msg, True, MENU_FG)
                    bg = pygame.Surface((s.get_width() + 16, s.get_height() + 8))
                    bg.fill(MENU_BG)
                    bg.set_alpha(210)
                    x, y = ox + 8, oy + th - bg.get_height() - 8
                    self.screen.blit(bg, (x, y))
                    pygame.draw.rect(self.screen, MENU_HI, (x, y, bg.get_width(), bg.get_height()), 1)
                    self.screen.blit(s, (x + 8, y + 4))
                except (pygame.error, AttributeError):
                    pass
            self._draw_hud()
        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gbc_emulator.py",
        description="Python Game Boy / Game Boy Color emulator")
    parser.add_argument("rom", nargs="?", help="Path to the .gb or .gbc ROM file")
    parser.add_argument("--nomenu", action="store_true",
                        help="Skip the menu and boot the ROM directly (implied when a ROM path is given)")
    parser.add_argument("--bootrom", help="Path to boot ROM (DMG 256B or CGB ~2304B)")
    parser.add_argument("--version", action="version",
                        version=f"Python GBC Emulator {__version__}")
    link_group = parser.add_argument_group("link cable (local multiplayer)")
    link_group.add_argument("--link-server", type=int, metavar="PORT",
                            help="Listen for a link cable connection on PORT")
    link_group.add_argument("--link-connect", metavar="HOST:PORT",
                            help="Connect to a link cable server at HOST:PORT")
    args = parser.parse_args()

    if args.nomenu and not args.rom:
        parser.error("--nomenu requires a ROM path")

    link_cable = None
    if args.link_server is not None and args.link_connect is not None:
        parser.error("Cannot use both --link-server and --link-connect")
    if args.link_server is not None:
        link_cable = LinkCable()
        print(f"Link cable: listening on port {args.link_server}...")
        link_cable.start_server(args.link_server)
        if link_cable.sock is None:
            print("Link cable: no connection (timeout). Running without link.")
            link_cable = None
        else:
            print("Link cable: connected!")
    elif args.link_connect is not None:
        host, _, port_str = args.link_connect.partition(":")
        try:
            port = int(port_str)
        except ValueError:
            parser.error("--link-connect must be HOST:PORT")
        link_cable = LinkCable()
        print(f"Link cable: connecting to {host}:{port}...")
        link_cable.connect(host, port)
        if link_cable.sock is None:
            print("Link cable: connection failed. Running without link.")
            link_cable = None
        else:
            print("Link cable: connected!")

    if args.rom:
        if not os.path.isfile(args.rom):
            print(f"error: ROM not found: {args.rom}", file=sys.stderr)
            sys.exit(1)
        try:
            emulator = GameBoy(args.rom, bootrom_path=args.bootrom, link_cable=link_cable)
            emulator.run()
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if link_cable is not None:
            link_cable.close()
            print("Note: link cable requires a ROM path. Example:")
            print("  python gbc_emulator.py game.gbc --link-server 12345")
        menu = EmulatorMenu(bootrom_path=args.bootrom)
        menu.run()