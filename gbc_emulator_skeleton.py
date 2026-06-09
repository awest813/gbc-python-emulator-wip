"""
Game Boy / Game Boy Color emulator written in Python.

Supports MBC1/MBC5 cartridges, BG / Window / Sprite rendering, and includes
a built-in menu with ROM browser. Performance-tuned to ~60-90 fps on
SUPERBAJTEK via precomputed tile / palette LUTs, unrolled scanline writers,
and a combined per-opcode dispatcher.

Usage:
    python gbc_emulator_skeleton.py                # launch the menu
    python gbc_emulator_skeleton.py rom.gb --nomenu  # boot a ROM directly

Requires: pygame >= 2.0, numpy >= 1.20
          (pip install pygame numpy)

SPDX-License-Identifier: MIT
Copyright (c) 2026 awest813
"""
import sys
import os
import time
import logging
import argparse
from collections import deque

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
FPS = 59.73
CYCLES_PER_FRAME = 70224  # 4.194304 MHz / 59.73 frames per second

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

def _shader_lcd_ghost(fb, prev=None):
    if prev is None:
        return fb
    return np.clip(fb * 0.80 + prev * 0.20, 0, 255).astype(np.uint8)

def _shader_crt_scanlines(fb):
    out = np.copy(fb)
    out[1::2, :, :] = np.clip(out[1::2, :, :] * 0.65, 0, 255).astype(np.uint8)
    return out

def _shader_gamma_warm(fb):
    return np.clip(np.power(fb.astype(np.float32) / 255.0, 1.2) * 255.0, 0, 255).astype(np.uint8)

def _shader_pixel_bloom(fb):
    blurred = np.zeros_like(fb)
    for c in range(3):
        blurred[:, :, c] = (np.roll(fb[:, :, c], 1, 0) + np.roll(fb[:, :, c], -1, 0) +
                            np.roll(fb[:, :, c], 1, 1) + np.roll(fb[:, :, c], -1, 1)) // 4
    return np.clip(fb * 0.70 + blurred * 0.30, 0, 255).astype(np.uint8)

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
WINDOW_SCALE_OPTIONS = [2, 3, 4, 5]


def _opt_index(options, value, default=0):
    """Return the index in a (label, value) option list whose value matches, else default."""
    for i, opt in enumerate(options):
        if opt[1] == value:
            return i
    return default

# Flag bit positions in the F register
FLAG_Z = 7  # Zero flag
FLAG_N = 6  # Subtract flag
FLAG_H = 5  # Half Carry flag
FLAG_C = 4  # Carry flag

# Key bindings: pygame key -> joypad bit (matches set_joypad_button bit order)
KEY_TO_JOYPAD_BIT = {
    pygame.K_RIGHT:  0,  # Right
    pygame.K_LEFT:   1,  # Left
    pygame.K_UP:     2,  # Up
    pygame.K_DOWN:   3,  # Down
    pygame.K_z:      4,  # A
    pygame.K_x:      5,  # B
    pygame.K_RSHIFT: 6,  # Select
    pygame.K_RETURN: 7,  # Start
} if pygame else {}


class Registers:
    """Manages the 8-bit and 16-bit paired registers of the LR35902 CPU."""
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
        self.trace_branch = _noop_trace

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
        carry = self.reg.get_flag(FLAG_C)
        if op_type == 0:
            result = a + operand
            self.reg.set_flag(FLAG_Z, (result & 0xFF) == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, (a & 0xF) + (operand & 0xF) > 0xF)
            self.reg.set_flag(FLAG_C, result > 0xFF)
            self.reg.a = result & 0xFF
        elif op_type == 1:
            result = a + operand + carry
            self.reg.set_flag(FLAG_Z, (result & 0xFF) == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, (a & 0xF) + (operand & 0xF) + carry > 0xF)
            self.reg.set_flag(FLAG_C, result > 0xFF)
            self.reg.a = result & 0xFF
        elif op_type == 2:
            result = a - operand
            self.reg.set_flag(FLAG_Z, (result & 0xFF) == 0)
            self.reg.set_flag(FLAG_N, 1)
            self.reg.set_flag(FLAG_H, (a & 0xF) < (operand & 0xF))
            self.reg.set_flag(FLAG_C, a < operand)
            self.reg.a = result & 0xFF
        elif op_type == 3:
            result = a - operand - carry
            self.reg.set_flag(FLAG_Z, (result & 0xFF) == 0)
            self.reg.set_flag(FLAG_N, 1)
            self.reg.set_flag(FLAG_H, (a & 0xF) < (operand & 0xF) + carry)
            self.reg.set_flag(FLAG_C, a < operand + carry)
            self.reg.a = result & 0xFF
        elif op_type == 4:
            result = a & operand
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 1)
            self.reg.set_flag(FLAG_C, 0)
            self.reg.a = result
        elif op_type == 5:
            result = a ^ operand
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, 0)
            self.reg.a = result
        elif op_type == 6:
            result = a | operand
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, 0)
            self.reg.a = result
        elif op_type == 7:
            result = a - operand
            self.reg.set_flag(FLAG_Z, (result & 0xFF) == 0)
            self.reg.set_flag(FLAG_N, 1)
            self.reg.set_flag(FLAG_H, (a & 0xF) < (operand & 0xF))
            self.reg.set_flag(FLAG_C, a < operand)

    def _inc_r8(self, val):
        result = (val + 1) & 0xFF
        self.reg.set_flag(FLAG_Z, result == 0)
        self.reg.set_flag(FLAG_N, 0)
        self.reg.set_flag(FLAG_H, (val & 0xF) == 0xF)
        return result

    def _dec_r8(self, val):
        result = (val - 1) & 0xFF
        self.reg.set_flag(FLAG_Z, result == 0)
        self.reg.set_flag(FLAG_N, 1)
        self.reg.set_flag(FLAG_H, (val & 0xF) == 0)
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
        val = self.mem[pc]
        self.reg.pc = (pc + 1) & 0xFFFF
        return val

    def fetch_word(self):
        """Fetches the next 16-bit word at PC and increments PC twice."""
        pc = self.reg.pc
        mem = self.mem
        val = mem[pc] | (mem[(pc + 1) & 0xFFFF] << 8)
        self.reg.pc = (pc + 2) & 0xFFFF
        return val

    def step(self):
        """Fetches, decodes, and executes a single instruction."""
        mem = self.mem
        if self.halted:
            if mem[0xFFFF] & mem[0xFF0F]:
                self.halted = False
                if self.interrupts_master_enabled:
                    return 4 + self._handle_interrupts()
            return 4

        if self.ime_pending:
            self.interrupts_master_enabled = True
            self.ime_pending = False

        pc = self.reg.pc
        if self.halt_bug_pending:
            self.halt_bug_pending = False
            opcode = mem[pc]
            # PC NOT advanced (HALT bug)
        else:
            opcode = mem[pc]
            self.reg.pc = pc + 1
        self.current_opcode_pc = pc
        cycles = self.execute(opcode)
        cycles += self._handle_interrupts()
        return cycles

    def _handle_interrupts(self):
        if not self.interrupts_master_enabled:
            return 0
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
            if self.reg.get_flag(FLAG_Z) == 0:
                if offset & 0x80: offset -= 256
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
            if self.reg.get_flag(FLAG_Z) == 1:
                if offset & 0x80: offset -= 256
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
        if opcode == 0x00:
            return 4
        if opcode == 0x01:
            self.reg.bc = self.fetch_word(); return 12
        if opcode == 0x02:
            self.mmu.write_byte(self.reg.bc, self.reg.a); return 8
        if opcode == 0x03:
            self.reg.bc = (self.reg.bc + 1) & 0xFFFF; return 8
        if opcode == 0x04:
            self.reg.b = self._inc_r8(self.reg.b); return 4
        if opcode == 0x05:
            self.reg.b = self._dec_r8(self.reg.b); return 4
        if opcode == 0x06:
            self.reg.b = self.fetch_byte(); return 8
        if opcode == 0x07:
            a = self.reg.a; carry = (a >> 7) & 1
            self.reg.a = ((a << 1) | carry) & 0xFF
            self.reg.set_flag(FLAG_Z, 0); self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0); self.reg.set_flag(FLAG_C, carry); return 4
        if opcode == 0x08:
            addr = self.fetch_word()
            self.mmu.write_word(addr, self.reg.sp); return 20
        if opcode == 0x09:
            self._add_hl_rr(self.reg.bc); return 8
        if opcode == 0x0A:
            self.reg.a = self.mmu.read_byte(self.reg.bc); return 8
        if opcode == 0x0B:
            self.reg.bc = (self.reg.bc - 1) & 0xFFFF; return 8
        if opcode == 0x0C:
            self.reg.c = self._inc_r8(self.reg.c); return 4
        if opcode == 0x0D:
            self.reg.c = self._dec_r8(self.reg.c); return 4
        if opcode == 0x0E:
            self.reg.c = self.fetch_byte(); return 8
        if opcode == 0x0F:
            a = self.reg.a; carry = a & 1
            self.reg.a = ((a >> 1) | (carry << 7)) & 0xFF
            self.reg.set_flag(FLAG_Z, 0); self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0); self.reg.set_flag(FLAG_C, carry); return 4
        if opcode == 0x10:
            self.fetch_byte()  # consume 0x00 padding
            if self.mmu.is_cgb and (self.mmu.key1 & 0x01):
                self.mmu.key1 ^= 0x80  # toggle double-speed
                self.mmu.key1 &= ~0x01  # clear prepare flag
            else:
                self.mmu.memory[0xFF40] &= 0x7F  # disable LCD
                self.halted = True  # wakes on any pending interrupt (joypad)
            return 4  # STOP
        if opcode == 0x11:
            self.reg.de = self.fetch_word(); return 12
        if opcode == 0x12:
            self.mmu.write_byte(self.reg.de, self.reg.a); return 8
        if opcode == 0x13:
            self.reg.de = (self.reg.de + 1) & 0xFFFF; return 8
        if opcode == 0x14:
            self.reg.d = self._inc_r8(self.reg.d); return 4
        if opcode == 0x15:
            self.reg.d = self._dec_r8(self.reg.d); return 4
        if opcode == 0x16:
            self.reg.d = self.fetch_byte(); return 8
        if opcode == 0x17:
            a = self.reg.a; old_c = self.reg.get_flag(FLAG_C)
            new_c = (a >> 7) & 1
            self.reg.a = ((a << 1) | old_c) & 0xFF
            self.reg.set_flag(FLAG_Z, 0); self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0); self.reg.set_flag(FLAG_C, new_c); return 4
        if opcode == 0x18:
            offset = self.fetch_byte()
            if offset & 0x80: offset -= 256
            new_pc = (self.reg.pc + offset) & 0xFFFF
            self.trace_branch("JR", self.current_opcode_pc, new_pc, opcode)
            self.reg.pc = new_pc; return 12
        if opcode == 0x19:
            self._add_hl_rr(self.reg.de); return 8
        if opcode == 0x1A:
            self.reg.a = self.mmu.read_byte(self.reg.de); return 8
        if opcode == 0x1B:
            self.reg.de = (self.reg.de - 1) & 0xFFFF; return 8
        if opcode == 0x1C:
            self.reg.e = self._inc_r8(self.reg.e); return 4
        if opcode == 0x1D:
            self.reg.e = self._dec_r8(self.reg.e); return 4
        if opcode == 0x1E:
            self.reg.e = self.fetch_byte(); return 8
        if opcode == 0x1F:
            a = self.reg.a; old_c = self.reg.get_flag(FLAG_C)
            new_c = a & 1
            self.reg.a = ((a >> 1) | (old_c << 7)) & 0xFF
            self.reg.set_flag(FLAG_Z, 0); self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0); self.reg.set_flag(FLAG_C, new_c); return 4
        if opcode == 0x20:
            offset = self.fetch_byte()
            if self._check_cond(0):
                if offset & 0x80: offset -= 256
                new_pc = (self.reg.pc + offset) & 0xFFFF
                self.trace_branch("JR NZ", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 12
            return 8
        if opcode == 0x21:
            self.reg.hl = self.fetch_word(); return 12
        if opcode == 0x22:
            self.mmu.write_byte(self.reg.hl, self.reg.a)
            self.reg.hl = (self.reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x23:
            self.reg.hl = (self.reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x24:
            self.reg.h = self._inc_r8(self.reg.h); return 4
        if opcode == 0x25:
            self.reg.h = self._dec_r8(self.reg.h); return 4
        if opcode == 0x26:
            self.reg.h = self.fetch_byte(); return 8
        if opcode == 0x27:
            self._daa(); return 4
        if opcode == 0x28:
            offset = self.fetch_byte()
            if self._check_cond(1):
                if offset & 0x80: offset -= 256
                new_pc = (self.reg.pc + offset) & 0xFFFF
                self.trace_branch("JR Z", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 12
            return 8
        if opcode == 0x29:
            self._add_hl_rr(self.reg.hl); return 8
        if opcode == 0x2A:
            self.reg.a = self.mmu.read_byte(self.reg.hl)
            self.reg.hl = (self.reg.hl + 1) & 0xFFFF; return 8
        if opcode == 0x2B:
            self.reg.hl = (self.reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x2C:
            self.reg.l = self._inc_r8(self.reg.l); return 4
        if opcode == 0x2D:
            self.reg.l = self._dec_r8(self.reg.l); return 4
        if opcode == 0x2E:
            self.reg.l = self.fetch_byte(); return 8
        if opcode == 0x2F:
            self.reg.a ^= 0xFF
            self.reg.set_flag(FLAG_N, 1); self.reg.set_flag(FLAG_H, 1); return 4
        if opcode == 0x30:
            offset = self.fetch_byte()
            if self._check_cond(2):
                if offset & 0x80: offset -= 256
                new_pc = (self.reg.pc + offset) & 0xFFFF
                self.trace_branch("JR NC", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 12
            return 8
        if opcode == 0x31:
            self.reg.sp = self.fetch_word(); return 12
        if opcode == 0x32:
            self.mmu.write_byte(self.reg.hl, self.reg.a)
            self.reg.hl = (self.reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x33:
            self.reg.sp = (self.reg.sp + 1) & 0xFFFF; return 8
        if opcode == 0x34:
            val = self.mmu.read_byte(self.reg.hl)
            self.mmu.write_byte(self.reg.hl, self._inc_r8(val)); return 12
        if opcode == 0x35:
            val = self.mmu.read_byte(self.reg.hl)
            self.mmu.write_byte(self.reg.hl, self._dec_r8(val)); return 12
        if opcode == 0x36:
            self.mmu.write_byte(self.reg.hl, self.fetch_byte()); return 12
        if opcode == 0x37:
            self.reg.set_flag(FLAG_N, 0); self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, 1); return 4
        if opcode == 0x38:
            offset = self.fetch_byte()
            if self._check_cond(3):
                if offset & 0x80: offset -= 256
                new_pc = (self.reg.pc + offset) & 0xFFFF
                self.trace_branch("JR C", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 12
            return 8
        if opcode == 0x39:
            self._add_hl_rr(self.reg.sp); return 8
        if opcode == 0x3A:
            self.reg.a = self.mmu.read_byte(self.reg.hl)
            self.reg.hl = (self.reg.hl - 1) & 0xFFFF; return 8
        if opcode == 0x3B:
            self.reg.sp = (self.reg.sp - 1) & 0xFFFF; return 8
        if opcode == 0x3C:
            self.reg.a = self._inc_r8(self.reg.a); return 4
        if opcode == 0x3D:
            self.reg.a = self._dec_r8(self.reg.a); return 4
        if opcode == 0x3E:
            self.reg.a = self.fetch_byte(); return 8
        # 0x3F
        carry = self.reg.get_flag(FLAG_C)
        self.reg.set_flag(FLAG_N, 0); self.reg.set_flag(FLAG_H, 0)
        self.reg.set_flag(FLAG_C, carry ^ 1)
        return 4

    def _exec_high(self, opcode):
        # 0xC0-0xFF: control flow and stack opcodes.
        if opcode == 0xC0:
            if self._check_cond(0):
                new_pc = self._pop()
                self.trace_branch("RET NZ", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 20
            return 8
        if opcode == 0xC1:
            self.reg.bc = self._pop(); return 12
        if opcode == 0xC2:
            addr = self.fetch_word()
            if self._check_cond(0):
                self.trace_branch("JP NZ", self.current_opcode_pc, addr, opcode)
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xC3:
            new_pc = self.fetch_word()
            self.trace_branch("JP", self.current_opcode_pc, new_pc, opcode)
            self.reg.pc = new_pc; return 16
        if opcode == 0xC4:
            addr = self.fetch_word()
            if self._check_cond(0):
                self.trace_branch("CALL NZ", self.current_opcode_pc, addr, opcode)
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xC5:
            self._push(self.reg.bc); return 16
        if opcode == 0xC6:
            self._alu_a_op(0, self.fetch_byte()); return 8
        if opcode == 0xC7:
            self.trace_branch("RST 00", self.current_opcode_pc, 0x00, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x00; return 16
        if opcode == 0xC8:
            if self._check_cond(1):
                new_pc = self._pop()
                self.trace_branch("RET Z", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 20
            return 8
        if opcode == 0xC9:
            new_pc = self._pop()
            self.trace_branch("RET", self.current_opcode_pc, new_pc, opcode)
            self.reg.pc = new_pc; return 16
        if opcode == 0xCA:
            addr = self.fetch_word()
            if self._check_cond(1):
                self.trace_branch("JP Z", self.current_opcode_pc, addr, opcode)
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xCC:
            addr = self.fetch_word()
            if self._check_cond(1):
                self.trace_branch("CALL Z", self.current_opcode_pc, addr, opcode)
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xCD:
            addr = self.fetch_word()
            self.trace_branch("CALL", self.current_opcode_pc, addr, opcode)
            self._push(self.reg.pc)
            self.reg.pc = addr; return 24
        if opcode == 0xCE:
            self._alu_a_op(1, self.fetch_byte()); return 8
        if opcode == 0xCF:
            self.trace_branch("RST 08", self.current_opcode_pc, 0x08, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x08; return 16
        if opcode == 0xD0:
            if self._check_cond(2):
                new_pc = self._pop()
                self.trace_branch("RET NC", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 20
            return 8
        if opcode == 0xD1:
            self.reg.de = self._pop(); return 12
        if opcode == 0xD2:
            addr = self.fetch_word()
            if self._check_cond(2):
                self.trace_branch("JP NC", self.current_opcode_pc, addr, opcode)
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xD4:
            addr = self.fetch_word()
            if self._check_cond(2):
                self.trace_branch("CALL NC", self.current_opcode_pc, addr, opcode)
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xD5:
            self._push(self.reg.de); return 16
        if opcode == 0xD6:
            self._alu_a_op(2, self.fetch_byte()); return 8
        if opcode == 0xD7:
            self.trace_branch("RST 10", self.current_opcode_pc, 0x10, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x10; return 16
        if opcode == 0xD8:
            if self._check_cond(3):
                new_pc = self._pop()
                self.trace_branch("RET C", self.current_opcode_pc, new_pc, opcode)
                self.reg.pc = new_pc; return 20
            return 8
        if opcode == 0xD9:
            new_pc = self._pop()
            self.trace_branch("RETI", self.current_opcode_pc, new_pc, opcode)
            self.reg.pc = new_pc
            self.interrupts_master_enabled = True; return 16
        if opcode == 0xDA:
            addr = self.fetch_word()
            if self._check_cond(3):
                self.trace_branch("JP C", self.current_opcode_pc, addr, opcode)
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xDC:
            addr = self.fetch_word()
            if self._check_cond(3):
                self.trace_branch("CALL C", self.current_opcode_pc, addr, opcode)
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xDE:
            self._alu_a_op(3, self.fetch_byte()); return 8
        if opcode == 0xDF:
            self.trace_branch("RST 18", self.current_opcode_pc, 0x18, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x18; return 16
        if opcode == 0xE0:
            self.mmu.write_byte(0xFF00 | self.fetch_byte(), self.reg.a); return 12
        if opcode == 0xE1:
            self.reg.hl = self._pop(); return 12
        if opcode == 0xE2:
            self.mmu.write_byte(0xFF00 | self.reg.c, self.reg.a); return 8
        if opcode == 0xE5:
            self._push(self.reg.hl); return 16
        if opcode == 0xE6:
            self._alu_a_op(4, self.fetch_byte()); return 8
        if opcode == 0xE7:
            self.trace_branch("RST 20", self.current_opcode_pc, 0x20, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x20; return 16
        if opcode == 0xE8:
            offset = self.fetch_byte()
            if offset & 0x80: offset -= 256
            result = self.reg.sp + offset
            self.reg.set_flag(FLAG_Z, 0); self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, (self.reg.sp & 0xF) + (offset & 0xF) > 0xF)
            self.reg.set_flag(FLAG_C, (self.reg.sp & 0xFF) + (offset & 0xFF) > 0xFF)
            self.reg.sp = result & 0xFFFF; return 16
        if opcode == 0xE9:
            self.trace_branch("JP HL", self.current_opcode_pc, self.reg.hl, opcode)
            self.reg.pc = self.reg.hl; return 4
        if opcode == 0xEA:
            self.mmu.write_byte(self.fetch_word(), self.reg.a); return 16
        if opcode == 0xEE:
            self._alu_a_op(5, self.fetch_byte()); return 8
        if opcode == 0xEF:
            self.trace_branch("RST 28", self.current_opcode_pc, 0x28, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x28; return 16
        if opcode == 0xF0:
            self.reg.a = self.mmu.read_byte(0xFF00 | self.fetch_byte()); return 12
        if opcode == 0xF1:
            self.reg.af = self._pop(); return 12
        if opcode == 0xF2:
            self.reg.a = self.mmu.read_byte(0xFF00 | self.reg.c); return 8
        if opcode == 0xF3:
            self.interrupts_master_enabled = False; return 4
        if opcode == 0xF5:
            self._push(self.reg.af); return 16
        if opcode == 0xF6:
            self._alu_a_op(6, self.fetch_byte()); return 8
        if opcode == 0xF7:
            self.trace_branch("RST 30", self.current_opcode_pc, 0x30, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x30; return 16
        if opcode == 0xF8:
            offset = self.fetch_byte()
            if offset & 0x80: offset -= 256
            result = self.reg.sp + offset
            self.reg.set_flag(FLAG_Z, 0); self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, (self.reg.sp & 0xF) + (offset & 0xF) > 0xF)
            self.reg.set_flag(FLAG_C, (self.reg.sp & 0xFF) + (offset & 0xFF) > 0xFF)
            self.reg.hl = result & 0xFFFF; return 12
        if opcode == 0xF9:
            self.reg.sp = self.reg.hl; return 8
        if opcode == 0xFA:
            self.reg.a = self.mmu.read_byte(self.fetch_word()); return 16
        if opcode == 0xFB:
            self.ime_pending = True; return 4
        if opcode == 0xFE:
            self._alu_a_op(7, self.fetch_byte()); return 8
        if opcode == 0xFF:
            self.trace_branch("RST 38", self.current_opcode_pc, 0x38, opcode)
            self._push(self.reg.pc); self.reg.pc = 0x38; return 16
        if opcode in self.INVALID_OPS:
            # Hardware-illegal opcode (locks up a real DMG/CGB); treat as NOP.
            self.invalid_opcode_count += 1
            return 4
        self.invalid_opcode_count += 1
        logging.error(f"Unimplemented Opcode: {opcode:02X} at PC: {self.current_opcode_pc:04X}")
        return 4

    def execute_cb(self, cb_opcode):
        """Executes prefixed CB instructions."""

        def _rlc(val):
            carry = (val >> 7) & 1
            result = ((val << 1) | carry) & 0xFF
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _rrc(val):
            carry = val & 1
            result = ((val >> 1) | (carry << 7)) & 0xFF
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _rl(val):
            old_c = self.reg.get_flag(FLAG_C)
            carry = (val >> 7) & 1
            result = ((val << 1) | old_c) & 0xFF
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _rr(val):
            old_c = self.reg.get_flag(FLAG_C)
            carry = val & 1
            result = ((val >> 1) | (old_c << 7)) & 0xFF
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _sla(val):
            carry = (val >> 7) & 1
            result = (val << 1) & 0xFF
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _sra(val):
            carry = val & 1
            result = (val >> 1) | (val & 0x80)
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _srl(val):
            carry = val & 1
            result = val >> 1
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, carry)
            return result

        def _swap(val):
            result = ((val & 0x0F) << 4) | ((val & 0xF0) >> 4)
            self.reg.set_flag(FLAG_Z, result == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 0)
            self.reg.set_flag(FLAG_C, 0)
            return result

        def _bit(val, bit):
            self.reg.set_flag(FLAG_Z, (val & (1 << bit)) == 0)
            self.reg.set_flag(FLAG_N, 0)
            self.reg.set_flag(FLAG_H, 1)

        def _res(val, bit):
            return val & ~(1 << bit)

        def _set(val, bit):
            return val | (1 << bit)

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
            ops = [_rlc, _rrc, _rl, _rr, _sla, _sra, _swap, _srl]
            result = ops[bit_pos](val)
            cycles = 16 if is_hl else 8
        elif op_group == 1:
            _bit(val, bit_pos)
            return 12 if is_hl else 8
        elif op_group == 2:
            result = _res(val, bit_pos)
            cycles = 16 if is_hl else 8
        else:
            result = _set(val, bit_pos)
            cycles = 16 if is_hl else 8

        if is_hl:
            self.mmu.write_byte(self.reg.hl, result)
        else:
            self._set_r8(reg_idx, result)
        return cycles


class MMU:
    """Memory Management Unit handling the 64KB address space with MBC1/MBC5."""
    def __init__(self):
        self.memory = bytearray(0x10000)
        self.rom_data = bytearray()
        self.ram_data = bytearray()
        self.mbc_type = 0x00
        self.ram_enabled = False
        self.rom_bank = 1
        self.ram_bank = 0
        self.mbc1_mode = 0
        self.has_ram = False
        self.has_battery = False
        self.num_rom_banks = 2
        self.num_ram_banks = 0
        self.joypad_buttons = 0xFF
        self.div_reset_callback = None
        self.apu = None
        self.ppu = None
        self.rom_path = None
        self.serial_data = 0x00
        self.serial_control = 0x00
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
        # OAM DMA (0xFF46): timed transfer over 640 dot-cycles
        self.dma_remaining = 0  # remaining CPU T-cycles of the transfer
        self.dma_src = 0       # source page high byte
        self.dma_buffer = bytearray()
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

        mbc_ram_types = {0x02, 0x03, 0x05, 0x06, 0x0F, 0x10, 0x12, 0x13, 0x1A, 0x1B, 0x1D, 0x1E}
        mbc_battery_types = {0x03, 0x06, 0x0F, 0x10, 0x13, 0x1B, 0x1E}
        self.has_ram = self.mbc_type in mbc_ram_types or self.num_ram_banks > 0
        self.has_battery = self.mbc_type in mbc_battery_types
        # MBC3 RTC is present on types 0x0F (timer+batt) and 0x10 (timer+ram+batt)
        self.has_rtc = self.mbc_type in (0x0F, 0x10)
        cgb_flag = self.rom_data[0x0143] if len(self.rom_data) > 0x0143 else 0x00
        self.is_cgb = bool(cgb_flag & 0x80)
        if self.mbc_type in (0x05, 0x06):
            # MBC2 has 512 nibbles = 256 bytes of 4-bit RAM
            self.ram_data = bytearray(256)
        elif self.has_ram and self.num_ram_banks > 0:
            self.ram_data = bytearray(self.num_ram_banks * 0x2000)

        self.memory[0:0x8000] = self.rom_data[0:min(0x8000, len(self.rom_data))]
        mbc_name = {0x00:"ROM ONLY", 0x01:"MBC1", 0x02:"MBC1+RAM", 0x03:"MBC1+RAM+BATT",
                    0x05:"MBC2", 0x06:"MBC2+BATT", 0x0F:"MBC3+TIMER+BATT",
                    0x10:"MBC3+TIMER+RAM+BATT", 0x11:"MBC3", 0x12:"MBC3+RAM",
                    0x13:"MBC3+RAM+BATT", 0x19:"MBC5", 0x1A:"MBC5+RAM",
                    0x1B:"MBC5+RAM+BATT", 0x1C:"MBC5+RUMBLE", 0x1D:"MBC5+RUMBLE+RAM",
                    0x1E:"MBC5+RUMBLE+RAM+BATT"}.get(self.mbc_type, f"UNKNOWN(0x{self.mbc_type:02X})")
        logging.info(f"Loaded ROM: {len(rom_data)} bytes [{mbc_name}, {self.num_rom_banks} ROM banks, {self.num_ram_banks} RAM banks{' CGB' if self.is_cgb else ''}]")

    def read_byte(self, address):
        # Boot ROM shadows cartridge ROM at 0x0000-N while enabled
        if self.bootrom_enabled and address < len(self.bootrom):
            return self.bootrom[address]
        # Fast path: most accesses hit WRAM/HRAM/VRAM/OAM (direct array)
        if address >= 0xC000:
            if address < 0xFE00:
                if address < 0xE000:
                    return self.memory[address]
                return self.memory[address - 0x2000]  # Echo RAM
            # OAM (0xFE00-0xFE9F): blocked during PPU modes 2 and 3, or during OAM DMA
            if address < 0xFEA0:
                if self.dma_remaining > 0:
                    return 0xFF
                if self.ppu is not None and self.ppu.mode >= 2:
                    return 0xFF
            if address == 0xFF00:
                return self._read_joypad()
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
            if self.has_ram and self.ram_enabled and len(self.ram_data) > 0:
                offset = self.ram_bank * 0x2000 + (address - 0xA000)
                return self.ram_data[offset] if offset < len(self.ram_data) else 0xFF
            return 0xFF
        # ROM (0x0000-0x7FFF)
        if address < 0x4000:
            return self.rom_data[address] if address < len(self.rom_data) else 0xFF
        if self.mbc_type == 0x00:
            return self.rom_data[address] if address < len(self.rom_data) else 0xFF
        offset = self.rom_bank * 0x4000 + (address - 0x4000)
        return self.rom_data[offset] if offset < len(self.rom_data) else 0xFF

    def _read_joypad(self):
        sel = self.memory[0xFF00] & 0x30
        line_dir = 0x0F
        line_act = 0x0F
        if not (self.joypad_buttons & 0x01): line_dir &= ~0x01
        if not (self.joypad_buttons & 0x02): line_dir &= ~0x02
        if not (self.joypad_buttons & 0x04): line_dir &= ~0x04
        if not (self.joypad_buttons & 0x08): line_dir &= ~0x08
        if not (self.joypad_buttons & 0x10): line_act &= ~0x01
        if not (self.joypad_buttons & 0x20): line_act &= ~0x02
        if not (self.joypad_buttons & 0x40): line_act &= ~0x04
        if not (self.joypad_buttons & 0x80): line_act &= ~0x08
        if not (sel & 0x10) and not (sel & 0x20):
            result = line_dir & line_act
        elif not (sel & 0x10):
            result = line_dir
        elif not (sel & 0x20):
            result = line_act
        else:
            result = 0x0F
        return 0xC0 | sel | result

    def set_joypad_button(self, bit, pressed):
        if pressed:
            new_state = self.joypad_buttons & ~(1 << bit)
        else:
            new_state = self.joypad_buttons | (1 << bit)
        if new_state != self.joypad_buttons and pressed:
            if_reg = self.memory[0xFF0F]
            self.memory[0xFF0F] = if_reg | 0x10
        self.joypad_buttons = new_state

    def read_word(self, address):
        lo = self.read_byte(address)
        hi = self.read_byte((address + 1) & 0xFFFF)
        return (hi << 8) | lo

    def write_byte(self, address, value):
        value &= 0xFF
        if address < 0x8000:
            self._handle_mbc_write(address, value)
        elif 0xA000 <= address <= 0xBFFF:
            if self.mbc_type in (0x05, 0x06):
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
            elif self.has_ram and self.ram_enabled and len(self.ram_data) > 0:
                offset = self.ram_bank * 0x2000 + (address - 0xA000)
                if offset < len(self.ram_data):
                    self.ram_data[offset] = value
        elif 0xE000 <= address <= 0xFDFF:
            self.memory[0xC000 + (address - 0xE000)] = value
        elif address == 0xFF00:
            self.memory[0xFF00] = (self.memory[0xFF00] & 0x0F) | (value & 0x30) | 0xC0
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
            if (value & 0x81) == 0x81:
                self._serial_transfer()
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

    def _serial_transfer(self):
        self.serial_control &= 0x7F
        self.memory[0xFF0F] |= 0x08

    def _hdma_transfer(self):
        src = ((self.memory[0xFF51] << 8) | self.memory[0xFF52]) & 0xFFF0
        dst = (((self.memory[0xFF53] << 8) | self.memory[0xFF54]) & 0x1FF0) | 0x8000
        length = ((self.memory[0xFF55] & 0x7F) + 1) * 16
        for i in range(length):
            val = self.read_byte(src + i)
            addr = 0x8000 + ((dst + i) & 0x1FFF)
            if self.vram_bank_select:
                self.vram_bank1[addr - 0x8000] = val
            else:
                self.memory[addr] = val
        self.memory[0xFF55] = 0xFF

    def _hdma_hblank_step(self):
        if not self.hdma_active:
            return
        chunk = min(16, self.hdma_remaining)
        for i in range(chunk):
            val = self.read_byte(self.hdma_src + i)
            addr = 0x8000 + ((self.hdma_dst + i) & 0x1FFF)
            if self.vram_bank_select:
                self.vram_bank1[addr - 0x8000] = val
            else:
                self.memory[addr] = val
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
        # Pre-cache source bytes; the actual OAM copy & timing happen in step_all
        src_base = value << 8
        self.dma_src = value
        self.dma_buffer = bytearray(160)
        for i in range(160):
            self.dma_buffer[i] = self.read_byte(src_base + i)
        self.dma_remaining = 640  # CPU T-cycles at single speed (640 dot-cycles)

    def _remap_rom_bank(self):
        bank = self.rom_bank % self.num_rom_banks
        src_offset = bank * 0x4000
        end = min(src_offset + 0x4000, len(self.rom_data))
        length = end - src_offset
        self.memory[0x4000:0x4000 + length] = self.rom_data[src_offset:end]
        if length < 0x4000:
            for i in range(length, 0x4000):
                self.memory[0x4000 + i] = 0xFF

    def _handle_mbc_write(self, address, value):
        mbc = self.mbc_type
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
            if self.mbc1_mode:
                self.ram_bank = value & 0x03
            else:
                self.rom_bank = (self.rom_bank & 0x1F) | ((value & 0x03) << 5)
                self._remap_rom_bank()
        elif 0x4000 <= address <= 0x5FFF and mbc in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
            self.ram_bank = value & 0x0F
        elif 0x4000 <= address <= 0x5FFF and mbc in (0x0F, 0x10, 0x11, 0x12, 0x13):
            # MBC3: 0x00-0x03 -> RAM bank 0-3, 0x08-0x0C -> RTC register
            v = value & 0x0F
            if v <= 0x03 or (0x08 <= v <= 0x0C):
                self.ram_bank = v
        elif 0x6000 <= address <= 0x7FFF and mbc in (0x01, 0x02, 0x03):
            self.mbc1_mode = value & 0x01
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
        self.rtc_last_time += delta
        if delta > 86400:
            delta = 86400  # cap at 1 day per call to avoid runaway after sleep
        for _ in range(delta):
            self._rtc_tick()

    def _rtc_tick(self):
        """Increment RTC by one second with proper carry/overflow."""
        if not self.has_rtc or (self.rtc_dh & 0x40):
            return
        self.rtc_s += 1
        if self.rtc_s < 60:
            return
        self.rtc_s = 0
        self.rtc_m += 1
        if self.rtc_m < 60:
            return
        self.rtc_m = 0
        self.rtc_h += 1
        if self.rtc_h < 24:
            return
        self.rtc_h = 0
        days = ((self.rtc_dh & 0x01) << 8) | self.rtc_dl
        days = (days + 1) & 0x1FF
        self.rtc_dl = days & 0xFF
        self.rtc_dh = (self.rtc_dh & 0xC0) | ((days >> 8) & 0x01)
        if days == 0:  # wrapped past 511 days
            self.rtc_dh |= 0x80

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
        self.bg_palette_data = bytearray(64)
        self.obj_palette_data = bytearray(64)
        self.bg_palette_addr = 0x00
        self.obj_palette_addr = 0x00
        self.cgb_opri = 0x00
        self.prev_stat_irq = False
        self.lcd_was_on = False
        self.window_line_counter = 0
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

        while dot >= 456:
            remaining = 456 - self.scanline_dot
            dot -= remaining
            self._advance_dots(456)
            self._finish_scanline()
            self.scanline_dot = 0
            if not (mem[0xFF40] & 0x80):
                return
            dot += self._begin_scanline()

        if dot > 0:
            self._advance_dots(dot)
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
                mem[0xFF0F] |= 0x01
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

    def _enter_mode3(self, ly, lcdc):
        mem = self.mmu.memory
        scx = mem[0xFF43]
        sprite_height = 16 if (lcdc & 0x04) else 8
        sprite_count = 0
        for i in range(40):
            oam_addr = 0xFE00 + i * 4
            y = mem[oam_addr]
            if y == 0 or y >= 160:
                continue
            spr_y = y - 16
            if spr_y > ly or spr_y + sprite_height <= ly:
                continue
            sprite_count += 1
            if sprite_count >= 10:
                break
        self.mode3_duration = 172 + (scx & 7) + sprite_count * 11
        if self.is_cgb or (lcdc & 0x01) or (lcdc & 0x20) or (lcdc & 0x02):
            self._render_scanline(ly, lcdc)
        self.mode = 3
        stat = (mem[0xFF41] & 0xFC) | 3
        mem[0xFF41] = stat

    def _finish_scanline(self):
        mem = self.mmu.memory
        ly = mem[0xFF44]
        new_ly = ly + 1
        if new_ly == 154:
            new_ly = 0
            self.window_line_counter = 0
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
            mem[0xFF0F] |= 0x02
        self.prev_stat_irq = current_irq

    def _render_scanline(self, ly, lcdc):
        mem = self.mmu.memory
        is_cgb = self.is_cgb
        # On CGB, LCDC bit 0 is BG/OBJ priority flag, not BG enable.
        # BG is always rendered in CGB mode regardless of bit 0.
        bg_enabled = (lcdc & 0x01) or is_cgb
        fb_row = ly * SCREEN_WIDTH

        if not bg_enabled:
            # DMG: BG/Window off - clear scanline to white, render sprites only.
            white = self.shades[0]
            fb = self.framebuffer
            bg_pri = self.bg_palette_idx
            for i in range(SCREEN_WIDTH):
                fb[fb_row + i] = white
            bg_pri[fb_row:fb_row + SCREEN_WIDTH] = self._zero_row
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

        if lcdc & 0x20:
            wy = mem[0xFF4A]
            wx_raw = mem[0xFF4B]
            if ly >= wy and (wx_raw - 7) < SCREEN_WIDTH:
                self._render_window(ly)
                self.window_line_counter += 1
        if lcdc & 0x02:
            self._render_sprites(ly)

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

    def _render_sprites(self, ly):
        mem = self.mmu.memory
        lcdc = mem[0xFF40]
        sprite_height = 16 if (lcdc & 0x04) else 8
        sprites = []
        for i in range(40):
            oam_addr = 0xFE00 + i * 4
            y = mem[oam_addr]
            x = mem[oam_addr + 1]
            if y == 0 or y >= 160:
                continue
            spr_y = y - 16
            spr_x = x - 8
            if spr_y > ly or spr_y + sprite_height <= ly:
                continue
            tile = mem[oam_addr + 2]
            flags = mem[oam_addr + 3]
            sprites.append((spr_x, spr_y, tile, flags))
            if len(sprites) >= 10:
                break
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
        self.div_counter = 0

    def step(self, cycles):
        mem = self.mmu.memory
        div_counter = self.div_counter + cycles
        self.div_counter = div_counter
        mem[0xFF04] = (div_counter >> 8) & 0xFF

        tac = mem[0xFF07]
        if not (tac & 0x04):
            return
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
                mem[0xFF0F] |= 0x04
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
    """DMG Audio Processing Unit: two square waves, wave channel, noise channel.

    Drives a 44.1 kHz signed-16-bit stereo PCM stream that the host audio
    backend consumes. The frame sequencer runs at 512 Hz (one tick every
    8192 CPU cycles) and clocks the length counters, envelopes, and the
    channel-1 sweep.
    """

    SAMPLE_RATE = 44100
    CPU_CLOCK = 4194304
    FRAME_SEQ_PERIOD = 8192  # CPU cycles per frame-sequencer tick (= 4194304 / 512)
    SOFT_BUFFER_CAP = 65536  # bytes (~370 ms stereo at 44.1 kHz); emergency drop threshold

    def __init__(self, mmu):
        self.mmu = mmu
        self.power = True
        self.buffer = bytearray()

        # Frame sequencer
        self.frame_seq_counter = self.FRAME_SEQ_PERIOD
        self.frame_seq_step = 0

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
                self.frame_seq_step = 0
                self.frame_seq_counter = self.FRAME_SEQ_PERIOD
            self.power = new_power
            self._refresh_nr52()
            return

        # When powered off, ignore writes to most registers (length-only writes
        # are allowed on DMG but we keep this simple and drop them).
        if not self.power and addr != 0xFF26 and not (0xFF30 <= addr <= 0xFF3F):
            return

        # Wave RAM
        if 0xFF30 <= addr <= 0xFF3F:
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
            if 0xFF10 <= addr <= 0xFF25:
                return 0xFF
            if addr == 0xFF26:
                return 0x70
            if 0xFF30 <= addr <= 0xFF3F:
                return 0xFF
            return 0xFF
        mem = self.mmu.memory
        if 0xFF30 <= addr <= 0xFF3F:
            return self.wave_ram[addr - 0xFF30]
        if addr == 0xFF12:
            return (self.ch1_volume << 4) | (mem[0xFF12] & 0x0F)
        if addr == 0xFF17:
            return (self.ch2_volume << 4) | (mem[0xFF17] & 0x0F)
        if addr == 0xFF21:
            return (self.ch4_volume << 4) | (mem[0xFF21] & 0x0F)
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
        self.frame_seq_counter = self.FRAME_SEQ_PERIOD
        self.frame_seq_step = 0

    # ── Channel triggers ─────────────────────────────────────────────

    def _trigger_ch1(self):
        if self.ch1_dac:
            self.ch1_enabled = True
        if self.ch1_length == 0:
            self.ch1_length = 64
        period = max((2048 - self.ch1_freq) * 4, 4)
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
            self.ch2_length = 64
        period = max((2048 - self.ch2_freq) * 4, 4)
        self.ch2_freq_timer = period
        self.ch2_duty_step = 0
        self.ch2_volume = self.ch2_env_initial
        self.ch2_env_timer = self.ch2_env_period if self.ch2_env_period else 8
        self._refresh_nr52()

    def _trigger_ch3(self):
        if self.ch3_dac:
            self.ch3_enabled = True
        if self.ch3_length == 0:
            self.ch3_length = 256
        period = max((2048 - self.ch3_freq) * 2, 2)
        self.ch3_freq_timer = period
        self.ch3_wave_pos = 0
        self._refresh_nr52()

    def _trigger_ch4(self):
        if self.ch4_dac:
            self.ch4_enabled = True
        if self.ch4_length == 0:
            self.ch4_length = 64
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
        if new_freq > 2047:
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
            if check2 > 2047:
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
        if self.ch1_env_period:
            self.ch1_env_timer -= 1
            if self.ch1_env_timer <= 0:
                self.ch1_env_timer = self.ch1_env_period
                v = self.ch1_volume
                if self.ch1_env_direction and v < 15:
                    self.ch1_volume = v + 1
                elif not self.ch1_env_direction and v > 0:
                    self.ch1_volume = v - 1
        if self.ch2_env_period:
            self.ch2_env_timer -= 1
            if self.ch2_env_timer <= 0:
                self.ch2_env_timer = self.ch2_env_period
                v = self.ch2_volume
                if self.ch2_env_direction and v < 15:
                    self.ch2_volume = v + 1
                elif not self.ch2_env_direction and v > 0:
                    self.ch2_volume = v - 1
        if self.ch4_env_period:
            self.ch4_env_timer -= 1
            if self.ch4_env_timer <= 0:
                self.ch4_env_timer = self.ch4_env_period
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
                period = max((2048 - self.ch1_freq) * 4, 4)
                advances = (-t) // period + 1
                self.ch1_duty_step = (self.ch1_duty_step + advances) & 7
                t = period - ((-t) % period)
            self.ch1_freq_timer = t
        # Channel 2
        if self.ch2_enabled:
            t = self.ch2_freq_timer - cycles
            if t <= 0:
                period = max((2048 - self.ch2_freq) * 4, 4)
                advances = (-t) // period + 1
                self.ch2_duty_step = (self.ch2_duty_step + advances) & 7
                t = period - ((-t) % period)
            self.ch2_freq_timer = t
        # Channel 3
        if self.ch3_enabled:
            t = self.ch3_freq_timer - cycles
            if t <= 0:
                period = max((2048 - self.ch3_freq) * 2, 2)
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
                    if width:
                        # 7-bit mode: shift within low 7 bits, new bit at 6
                        lfsr = ((lfsr >> 1) & 0x3F) | (bit << 6)
                    else:
                        lfsr = (lfsr >> 1) | (bit << 14)
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
        left = left * (self.vol_left + 1) * 70
        right = right * (self.vol_right + 1) * 70
        if left > 32767: left = 32767
        elif left < -32768: left = -32768
        if right > 32767: right = 32767
        elif right < -32768: right = -32768
        lv = left & 0xFFFF
        rv = right & 0xFFFF
        self.buffer.extend((lv & 0xFF, (lv >> 8) & 0xFF, rv & 0xFF, (rv >> 8) & 0xFF))

    # ── Main step ───────────────────────────────────────────────────

    def step(self, cycles):
        if not self.power:
            # APU off: still produce silence so the audio buffer keeps flowing.
            self.sample_accum += cycles * self.sample_den
            sn = self.sample_num
            while self.sample_accum >= sn:
                self.sample_accum -= sn
                self.buffer.extend(b'\x00\x00\x00\x00')
            if len(self.buffer) > self.SOFT_BUFFER_CAP:
                del self.buffer[:len(self.buffer) - self.SOFT_BUFFER_CAP]
            return

        # Frame sequencer (512 Hz)
        fsc = self.frame_seq_counter - cycles
        while fsc <= 0:
            fsc += self.FRAME_SEQ_PERIOD
            self._frame_seq_tick()
        self.frame_seq_counter = fsc

        # Channel waveform timers
        self._step_channels(cycles)

        # Sample emission
        self.sample_accum += cycles * self.sample_den
        sn = self.sample_num
        while self.sample_accum >= sn:
            self.sample_accum -= sn
            self._mix_sample()

        # Hard-cap the buffer so a stalled audio backend can't blow memory.
        if len(self.buffer) > self.SOFT_BUFFER_CAP:
            del self.buffer[:len(self.buffer) - self.SOFT_BUFFER_CAP]

    def drain(self):
        """Returns and clears the accumulated PCM bytes (signed-16 stereo LE)."""
        data = bytes(self.buffer)
        del self.buffer[:]
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
MENU_BG = (15, 25, 45)
MENU_FG = (210, 225, 245)
MENU_HI = (70, 200, 70)
MENU_DIM = (110, 130, 155)


class EmulatorMenu:
    def __init__(self):
        self.selected = 0
        self.main_items = ["Load ROM", "Settings", "Exit to OS"]
        self.roms = []
        self.rom_cursor = 0
        self.rom_scroll = 0
        self.max_visible = 12
        self.settings_items = ["Window Scale: 4x"]
        self.settings_cursor = 0
        self.exit_cursor = 0
        self.window_scale = 4
        self.fps_limit_idx = 0
        self.audio_idx = 0
        self.volume_idx = 4
        self.palette_idx = 0
        self.filter_idx = 0
        self.shader_idx = 0
        self.status_line = ""
        self.status_ttl = 0
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
        self.screen = pygame.display.set_mode((MENU_W, MENU_H))
        pygame.display.set_caption("Python GBC Emulator")
        self.logo = None
        self._load_logo()
        self._scan_roms()

    def _scan_roms(self):
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
                    self.roms.append(fp)
            for sub in sorted(os.listdir(base)):
                d1 = os.path.join(base, sub)
                if not os.path.isdir(d1):
                    continue
                try:
                    for f in sorted(os.listdir(d1)):
                        fp = os.path.join(d1, f)
                        if os.path.isfile(fp) and f.lower().endswith(('.gb', '.gbc')):
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
                                    self.roms.append(fp)
                        except OSError:
                            pass
                except OSError:
                    pass

    def _sync_settings_items(self):
        self.settings_items = [
            f"Window Scale: {self.window_scale}x",
            f"Frame Rate: {FPS_LIMIT_OPTIONS[self.fps_limit_idx][0]}",
            f"Audio: {AUDIO_OPTIONS[self.audio_idx][0]}",
            f"Volume: {VOLUME_OPTIONS[self.volume_idx][0]}",
            f"Palette: {PALETTE_LIST[self.palette_idx][0]}",
            f"Filter: {FILTER_OPTIONS[self.filter_idx][0]}",
            f"Shader: {SHADER_LIST[self.shader_idx][0]}",
        ]

    def _status(self, msg):
        self.status_line = msg
        self.status_ttl = 120

    def _centre_text(self, text, y, colour=MENU_FG, size=28):
        f = pygame.font.Font(None, size)
        s = f.render(text, True, colour)
        x = (MENU_W - s.get_width()) // 2
        self.screen.blit(s, (x, y))

    def _draw_menu(self, items, cursor, start_y, gap):
        for i, item in enumerate(items):
            y = start_y + i * gap
            colour = MENU_HI if i == cursor else MENU_FG
            self._centre_text(item, y, colour)
            if i == cursor:
                f = pygame.font.Font(None, 28)
                w = f.render(item, True, colour).get_width()
                left = (MENU_W - w) // 2
                pygame.draw.rect(self.screen, colour, (left, y + 22, w, 2))

    def run(self):
        clock = pygame.time.Clock()
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
            elif page == "confirm_exit":
                self._render_confirm_exit()
            if self.status_ttl:
                self._centre_text(self.status_line, MENU_H - 28, MENU_DIM, 18)
            pygame.display.flip()
            clock.tick(60)

    def _load_logo(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for candidate in ("gbclogo.png", os.path.join(here, "gbclogo.png")):
            if os.path.isfile(candidate):
                try:
                    raw = pygame.image.load(candidate).convert()
                except (OSError, pygame.error):
                    continue
                scaled = pygame.transform.smoothscale(raw, (160, 160))
                if np is not None:
                    arr = pygame.surfarray.array3d(scaled).transpose(1, 0, 2)
                    mask = (arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220)
                    arr[mask] = MENU_BG
                    new_surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
                else:
                    new_surf = scaled
                block = pygame.Surface((180, 180))
                block.fill(MENU_BG)
                block.blit(new_surf, (10, 10))
                self.logo = block
                try:
                    pygame.display.set_icon(pygame.transform.smoothscale(raw, (32, 32)))
                except pygame.error:
                    pass
                return
        self.logo = None

    def _handle(self, page):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type != pygame.KEYDOWN:
                continue
            if page == "main":
                if event.key == pygame.K_UP:
                    self.selected = (self.selected - 1) % len(self.main_items)
                elif event.key == pygame.K_DOWN:
                    self.selected = (self.selected + 1) % len(self.main_items)
                elif event.key == pygame.K_RETURN:
                    if self.selected == 0:
                        self._scan_roms()
                        self.rom_cursor = 0
                        self.rom_scroll = 0
                        return "load_rom"
                    elif self.selected == 1:
                        self._sync_settings_items()
                        self.settings_cursor = 0
                        return "settings"
                    elif self.selected == 2:
                        self.exit_cursor = 0
                        return "confirm_exit"
                elif event.key == pygame.K_ESCAPE:
                    self.exit_cursor = 0
                    return "confirm_exit"
            elif page == "load_rom":
                if event.key == pygame.K_UP:
                    if self.rom_cursor > 0:
                        self.rom_cursor -= 1
                        if self.rom_cursor < self.rom_scroll:
                            self.rom_scroll = self.rom_cursor
                elif event.key == pygame.K_DOWN:
                    if self.rom_cursor < len(self.roms) - 1:
                        self.rom_cursor += 1
                        if self.rom_cursor >= self.rom_scroll + self.max_visible:
                            self.rom_scroll = self.rom_cursor - self.max_visible + 1
                elif event.key == pygame.K_RETURN:
                    if self.roms:
                        path = self.roms[self.rom_cursor]
                        self._status(f"Loading: {os.path.basename(path)}")
                        self._render_load_rom()
                        pygame.display.flip()
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
                            )
                            gb.run()
                        except FileNotFoundError:
                            self._status(f"ROM not found: {os.path.basename(path)}")
                        except Exception as e:
                            self._status(f"Error loading ROM: {e}")
                        self.screen = pygame.display.set_mode((MENU_W, MENU_H))
                        pygame.event.clear()
                        return "main"
                elif event.key == pygame.K_ESCAPE:
                    self.selected = 0
                    return "main"
                elif event.key == pygame.K_F5:
                    self._scan_roms()
                    self.rom_cursor = 0
                    self.rom_scroll = 0
                    self._status("ROM list refreshed")
            elif page == "settings":
                if event.key == pygame.K_UP:
                    self.settings_cursor = (self.settings_cursor - 1) % len(self.settings_items)
                elif event.key == pygame.K_DOWN:
                    self.settings_cursor = (self.settings_cursor + 1) % len(self.settings_items)
                elif event.key == pygame.K_RETURN:
                    if self.settings_cursor == 0:
                        scales = [2, 3, 4, 5]
                        try:
                            idx = scales.index(self.window_scale)
                        except ValueError:
                            idx = 2
                        self.window_scale = scales[(idx + 1) % len(scales)]
                    elif self.settings_cursor == 1:
                        self.fps_limit_idx = (self.fps_limit_idx + 1) % len(FPS_LIMIT_OPTIONS)
                    elif self.settings_cursor == 2:
                        self.audio_idx = (self.audio_idx + 1) % len(AUDIO_OPTIONS)
                    elif self.settings_cursor == 3:
                        self.volume_idx = (self.volume_idx + 1) % len(VOLUME_OPTIONS)
                    elif self.settings_cursor == 4:
                        self.palette_idx = (self.palette_idx + 1) % len(PALETTE_LIST)
                    elif self.settings_cursor == 5:
                        self.filter_idx = (self.filter_idx + 1) % len(FILTER_OPTIONS)
                    elif self.settings_cursor == 6:
                        self.shader_idx = (self.shader_idx + 1) % len(SHADER_LIST)
                    self._sync_settings_items()
                elif event.key == pygame.K_ESCAPE:
                    self.selected = 1
                    return "main"
            elif page == "confirm_exit":
                if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    self.exit_cursor ^= 1
                elif event.key == pygame.K_RETURN:
                    if self.exit_cursor == 1:
                        pygame.quit()
                        sys.exit()
                    self.selected = 0
                    return "main"
                elif event.key == pygame.K_ESCAPE:
                    self.selected = 0
                    return "main"
        return page

    def _render_main(self):
        if self.logo is not None:
            lx = (MENU_W - self.logo.get_width()) // 2
            self.screen.blit(self.logo, (lx, 20))
            title_y = 200
            subtitle_y = 235
            menu_y = 290
        else:
            self._centre_text("Python GBC Emulator", 60, MENU_HI, 48)
            title_y = 105
            subtitle_y = 130
            menu_y = 200
        self._centre_text("Python GBC Emulator", title_y, MENU_HI, 44)
        self._centre_text("v1.0", subtitle_y, MENU_DIM, 20)
        self._draw_menu(self.main_items, self.selected, menu_y, 50)
        self._centre_text("Arrow Keys: Navigate  |  Enter: Select  |  Esc: Quit", MENU_H - 30, MENU_DIM, 18)

    def _render_load_rom(self):
        self._centre_text("Select ROM", 40, MENU_HI, 36)
        if not self.roms:
            self._centre_text("No .gb/.gbc files found", 180, MENU_DIM)
            self._centre_text("Place your ROM files in the  roms  folder alongside this program", 225, MENU_DIM)
            self._centre_text("or anywhere in this directory or its parent.", 260, MENU_DIM)
            self._centre_text("Press F5 to rescan for ROMs.", 310, MENU_HI, 22)
        else:
            visible = self.roms[self.rom_scroll:self.rom_scroll + self.max_visible]
            for i, rom_path in enumerate(visible):
                y = 90 + i * 30
                idx = self.rom_scroll + i
                name = os.path.basename(rom_path)
                if len(name) > 44:
                    name = name[:41] + "..."
                colour = MENU_HI if idx == self.rom_cursor else MENU_FG
                f = pygame.font.Font(None, 24)
                s = f.render(f"  {name}  ({os.path.dirname(rom_path) or '.'})", True, colour)
                self.screen.blit(s, (30, y))
                if idx == self.rom_cursor:
                    pygame.draw.rect(self.screen, colour, (28, y + 20, 580, 1))
        self._centre_text("Enter: Load  |  F5: Refresh  |  Esc: Back", MENU_H - 30, MENU_DIM, 18)

    def _render_settings(self):
        self._centre_text("Settings", 40, MENU_HI, 36)
        self._draw_menu(self.settings_items, self.settings_cursor, 120, 45)
        self._centre_text("Enter: Cycle  |  Esc: Back", MENU_H - 30, MENU_DIM, 18)

    def _render_confirm_exit(self):
        self._centre_text("Exit Emulator?", 130, MENU_HI, 40)
        self._centre_text("Are you sure you want to quit to the OS?", 195, MENU_DIM, 22)
        self._draw_menu(["Keep Playing", "Exit to OS"], self.exit_cursor, 270, 50)
        self._centre_text("Arrow Keys: Choose  |  Enter: Select  |  Esc: Cancel",
                          MENU_H - 30, MENU_DIM, 18)


class GameBoy:
    """The main emulator orchestrator class."""
    def __init__(self, rom_path=None, window_scale=4, fps_limit=59.73,
                 audio_enabled=True, volume=1.0, palette=PALETTE_DMG,
                 smooth_scale=False, shader=None, bootrom_path=None):
        self.mmu = MMU()
        self.cpu = CPU(self.mmu)
        self.ppu = PPU(self.mmu)
        self.timers = Timers(self.mmu)
        self.apu = APU(self.mmu)
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
            with open(rom_path, 'rb') as f:
                self.mmu.load_rom(f.read())
            self.mmu.rom_path = rom_path
            if self.mmu.is_cgb:
                self.ppu.is_cgb = True
                self.ppu._cgb_init_palettes()
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
            self._load_sav()
        else:
            logging.info("No ROM provided. Running dummy infinite loop.")
            self.mmu.memory[0x0100] = 0x00
            self.mmu.memory[0x0101] = 0xC3
            self.mmu.memory[0x0102] = 0x00
            self.mmu.memory[0x0103] = 0x01

        self.speed_remainder = 0  # carries the odd base-clock dot in double-speed mode
        self._rtc_cycle_accum = 0  # throttle MBC3 RTC wall-clock updates to once per frame
        self._status_msg = ''
        self._status_ttl = 0
        self.fps_limit = fps_limit
        self.smooth_scale = smooth_scale
        self.shader = shader if shader is not None else _shader_none
        self._prev_shader_frame = None
        self.ppu.set_palette(palette)
        # Reused display buffers avoid per-frame numpy allocations in render().
        if np is not None:
            self._frame_np = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            self._blit_np = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)
        else:
            self._frame_np = None
            self._blit_np = None

        if pygame:
            self.window_scale = window_scale
            self.screen = pygame.display.set_mode((SCREEN_WIDTH * self.window_scale, SCREEN_HEIGHT * self.window_scale))
            pygame.display.set_caption(f"Python GBC Emulator - {os.path.basename(rom_path) if rom_path else 'No ROM'}")
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
        self._pause_msg = ""
        self._pause_msg_ttl = 0
        self._set_idx = {
            'scale':   _opt_index([(s, s) for s in WINDOW_SCALE_OPTIONS], window_scale, 2),
            'fps':     _opt_index(FPS_LIMIT_OPTIONS, fps_limit, 0),
            'audio':   0 if audio_enabled else 1,
            'volume':  _opt_index(VOLUME_OPTIONS, volume, 4),
            'palette': _opt_index(PALETTE_LIST, palette, 0),
            'filter':  _opt_index(FILTER_OPTIONS, smooth_scale, 0),
            'shader':  _opt_index(SHADER_LIST, self.shader, 0),
        }

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

    def _sav_path(self):
        if not self.mmu.rom_path:
            return None
        base = os.path.splitext(self.mmu.rom_path)[0]
        return base + ".sav"

    def _load_sav(self):
        if not self.mmu.has_battery or not self.mmu.has_ram:
            return
        path = self._sav_path()
        if path and os.path.isfile(path):
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                if len(data) <= len(self.mmu.ram_data):
                    self.mmu.ram_data[:len(data)] = data
                else:
                    self.mmu.ram_data[:] = data[:len(self.mmu.ram_data)]
                # MBC3 RTC: extra 48 bytes (VBA format) with 5 RTC regs at offset 4
                if self.mmu.has_rtc and len(data) >= len(self.mmu.ram_data) + 8:
                    rtc_blob = data[len(self.mmu.ram_data):len(self.mmu.ram_data) + 48]
                    self.mmu.rtc_s = rtc_blob[4] & 0x3F
                    self.mmu.rtc_m = rtc_blob[5] & 0x3F
                    self.mmu.rtc_h = rtc_blob[6] & 0x1F
                    self.mmu.rtc_dl = rtc_blob[7]
                    self.mmu.rtc_dh = rtc_blob[8] & 0xC1
                    self.mmu.rtc_last_time = time.time()
                logging.info(f"Loaded save: {os.path.basename(path)} ({len(data)} bytes)")
            except OSError as e:
                logging.warning(f"Could not load save: {e}")

    def _save_sav(self):
        if not self.mmu.has_battery or not self.mmu.has_ram:
            return
        path = self._sav_path()
        if not path:
            return
        try:
            with open(path, 'wb') as f:
                f.write(self.mmu.ram_data)
                if self.mmu.has_rtc:
                    # VBA-style 48-byte RTC block: 4 zero bytes + 5 RTC bytes + 39 zeros
                    rtc_blob = bytearray(48)
                    self.mmu._rtc_update()
                    rtc_blob[4] = self.mmu.rtc_s & 0x3F
                    rtc_blob[5] = self.mmu.rtc_m & 0x3F
                    rtc_blob[6] = self.mmu.rtc_h & 0x1F
                    rtc_blob[7] = self.mmu.rtc_dl
                    rtc_blob[8] = self.mmu.rtc_dh & 0xC1
                    f.write(rtc_blob)
            logging.info(f"Saved: {os.path.basename(path)} ({len(self.mmu.ram_data) + 48 if self.mmu.has_rtc else len(self.mmu.ram_data)} bytes)")
        except OSError as e:
            logging.warning(f"Could not save: {e}")


    # ── Save state support ─────────────────────────────────────────
    SAVE_STATE_MAGIC = b'GBST'
    SAVE_STATE_VERSION = 1

    def _state_path(self, slot):
        if not self.mmu.rom_path:
            return None
        base = os.path.splitext(self.mmu.rom_path)[0]
        return f"{base}.ss{slot}"

    def save_state(self, slot=0):
        path = self._state_path(slot)
        if not path:
            return False
        try:
            apu = self.apu
            mmu = self.mmu
            ppu = self.ppu
            cpu = self.cpu
            timers = self.timers
            # Cancel any in-flight OAM DMA so it doesn't run during load.
            mmu.dma_remaining = 0
            mmu.dma_buffer = bytearray()
            # Sync CGB WRAM bank back to main memory so it round-trips.
            mmu.wram_banks[mmu.svbk - 1][:] = mmu.memory[0xD000:0xE000]
            # Flush APU to silence (state captures the buffer's tail).
            apu.drain()
            import struct
            parts = []
            parts.append(self.SAVE_STATE_MAGIC)
            parts.append(struct.pack('<BBBB', self.SAVE_STATE_VERSION, slot, 0, 0))
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
            parts.append(struct.pack('<iiBBBBBBBBBB',
                                     mmu.hdma_remaining, mmu.hdma_src, mmu.hdma_dst,
                                     1 if mmu.hdma_active else 0, mmu.dma_src,
                                     0, 0, 0, 0, 0, 0, 0))
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
            # PPU
            parts.append(struct.pack('<BBB BBBBB BBBBB BBBB BB',
                                     ppu.mode, ppu.scanline_dot & 0xFF,
                                     (ppu.scanline_dot >> 8) & 0xFF,
                                     ppu.mode3_duration & 0xFF, (ppu.mode3_duration >> 8) & 0xFF,
                                     0, 0, 0,  # reserved
                                     ppu.bg_palette_addr, ppu.obj_palette_addr, ppu.cgb_opri,
                                     1 if ppu.lcd_was_on else 0, 1 if ppu.is_cgb else 0,
                                     ppu.window_line_counter & 0xFF,
                                     (ppu.window_line_counter >> 8) & 0xFF,
                                     0, 0, 0, 0))
            parts.append(ppu.bg_palette_data)
            parts.append(ppu.obj_palette_data)
            # APU: pack a flat list of 1-byte values for each boolean / small int.
            apu_values = [
                1 if apu.power else 0,
                apu.frame_seq_step & 0x07,
                (apu.frame_seq_counter >> 8) & 0xFF,
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
            with open(path, 'wb') as f:
                for p in parts:
                    f.write(p)
            logging.info(f"Saved state to slot {slot}: {os.path.basename(path)}")
            return True
        except OSError as e:
            logging.warning(f"Could not save state: {e}")
            return False

    def load_state(self, slot=0):
        path = self._state_path(slot)
        if not path or not os.path.isfile(path):
            return False
        try:
            import struct
            with open(path, 'rb') as f:
                data = f.read()
            pos = 0
            if data[pos:pos+4] != self.SAVE_STATE_MAGIC:
                logging.warning("Save state: bad magic")
                return False
            pos += 4
            ver = data[pos]
            if ver != self.SAVE_STATE_VERSION:
                logging.warning(f"Save state: unsupported version {ver}")
                return False
            pos += 4
            mmu = self.mmu
            cpu = self.cpu
            ppu = self.ppu
            apu = self.apu
            timers = self.timers
            # CPU
            (a, f_, b, c, d, e, h, l, sp, pc, _, _) = struct.unpack_from('<BBBB BB BB HHHH', data, pos)
            pos += struct.calcsize('<BBBB BB BB HHHH')
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
            # MMU memory
            mmu.memory[:] = data[pos:pos+0x10000]
            pos += 0x10000
            # MMU control
            fmt = '<BBBBBBBBBBBBBBBBBBB'
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
            mmu.is_cgb = bool(is_cgb)
            mmu.joypad_buttons = joypad
            mmu.serial_data = serial_data
            mmu.serial_control = serial_control
            mmu.vram_bank_select = vram_bank_select & 1
            mmu.key1 = key1
            mmu.rp = rp
            mmu.svbk = svbk
            # HDMA / DMA
            fmt = '<iiBBBBBBBBBB'
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
            mmu.vram_bank1[:] = data[pos:pos+0x2000]
            pos += 0x2000
            # WRAM banks
            for i in range(7):
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
            ppu.window_line_counter = win_lo | (win_hi << 8)
            ppu.bg_palette_data[:] = data[pos:pos+64]
            pos += 64
            ppu.obj_palette_data[:] = data[pos:pos+64]
            pos += 64
            # Refresh derived CGB color tables so the PPU can use the loaded palettes.
            for i in range(32):
                ppu._update_cgb_bg_color(i)
                ppu._update_cgb_obj_color(i)
            # APU
            apu_size = 13 + 16 + 12 + 10 + 12 + 14
            apu_bytes = data[pos:pos+apu_size]
            pos += apu_size
            ap = apu_bytes
            ai = 0
            apu.power = bool(ap[ai]); ai += 1
            apu.frame_seq_step = ap[ai] & 0x07; ai += 1
            apu.frame_seq_counter = (ap[ai] & 0xFF) << 8; ai += 1
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
            (div_counter, tima_accum) = struct.unpack_from('<II', data, pos)
            pos += 8
            timers.div_counter = div_counter
            timers.tima_accum = tima_accum
            apu.drain()
            if hasattr(self, '_audio_pending'):
                self._audio_pending.clear()
            self._av_start = time.perf_counter()
            self._sync_samples = 0
            self._sync_frames = 0
            logging.info(f"Loaded state from slot {slot}: {os.path.basename(path)}")
            return True
        except (OSError, struct.error) as e:
            logging.warning(f"Could not load state: {e}")
            return False

    def _pump_audio(self):
        """Feed the mixer from the pending-audio deque (channel + one queue slot)."""
        if not self.audio_enabled:
            return
        ch = self.audio_channel
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
            time.sleep(min(delay, 1.0 / 30))
        elif delay < -0.5:
            self._av_start = now - (self._sync_frames / self.fps_limit if self.fps_limit > 0 else 0)
            self._sync_samples = 0

    def step_all(self):
        """Execute one CPU step and propagate cycles to PPU, timers, and APU in a
        single Python function call.  Avoids extra function-call dispatches per
        opcode, which is a measurable win when the per-opcode path is otherwise tight."""
        # OAM DMA: consumes CPU T-cycles without executing instructions
        if self.mmu.dma_remaining > 0:
            chunk = min(self.mmu.dma_remaining, 4)
            self.mmu.dma_remaining -= chunk
            cpu_cycles = chunk
            if self.mmu.dma_remaining == 0:
                self.mmu.memory[0xFE00:0xFEA0] = self.mmu.dma_buffer
                self.mmu.dma_buffer = bytearray()
        else:
            cpu_cycles = self.cpu.step()
        # CGB double-speed (KEY1): the CPU and the DIV/TIMA timer run at 2x the
        # base clock, but the PPU and APU stay on the base clock. Feed the latter
        # half the CPU cycles, carrying the odd cycle across calls so no dot is
        # lost. In normal speed this is a straight pass-through (dot == cpu).
        if self.mmu.key1 & 0x80:
            self.speed_remainder += cpu_cycles
            dot_cycles = self.speed_remainder >> 1
            self.speed_remainder &= 1
        else:
            dot_cycles = cpu_cycles
        self.ppu.step(dot_cycles)
        self.timers.step(cpu_cycles)
        self.apu.step(dot_cycles)
        if self.mmu.has_rtc:
            self._rtc_cycle_accum += cpu_cycles
            if self._rtc_cycle_accum >= CYCLES_PER_FRAME:
                self._rtc_cycle_accum -= CYCLES_PER_FRAME
                self.mmu._rtc_update()
        return dot_cycles

    def run(self):
        """Main execution loop.  Returns to caller when the user exits to the menu
        (via the pause menu) or closes the window."""
        self.running = True
        self.paused = False
        self._av_start = time.perf_counter()
        self._sync_samples = 0
        self._sync_frames = 0

        while self.running:
            cycles_this_frame = 0
            while cycles_this_frame < CYCLES_PER_FRAME:
                cycles_this_frame += self.step_all()

            samples_this_frame = 0
            if pygame:
                self.handle_events()
                self.render()
                samples_this_frame = self._flush_audio()

            if self.paused and pygame:
                self._pause_menu_loop()
                continue

            self._pace_frame(samples_this_frame)

        self._save_sav()

    def handle_events(self):
        """Process window events and Joypad inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._open_pause_menu()
                    return
                elif event.key == pygame.K_F6:
                    if self.save_state(0):
                        self._status_msg = "State saved to slot 0"
                    else:
                        self._status_msg = "Failed to save state"
                    self._status_ttl = 90
                elif event.key == pygame.K_F7:
                    if self.load_state(0):
                        self._status_msg = "State loaded from slot 0"
                    else:
                        self._status_msg = "No save state in slot 0"
                    self._status_ttl = 90
                elif event.key == pygame.K_F8:
                    if self.save_state(1):
                        self._status_msg = "State saved to slot 1"
                    else:
                        self._status_msg = "Failed to save state"
                    self._status_ttl = 90
                elif event.key == pygame.K_F9:
                    if self.load_state(1):
                        self._status_msg = "State loaded from slot 1"
                    else:
                        self._status_msg = "No save state in slot 1"
                    self._status_ttl = 90
                elif event.key in KEY_TO_JOYPAD_BIT:
                    self.mmu.set_joypad_button(KEY_TO_JOYPAD_BIT[event.key], True)
            elif event.type == pygame.KEYUP:
                if event.key in KEY_TO_JOYPAD_BIT:
                    self.mmu.set_joypad_button(KEY_TO_JOYPAD_BIT[event.key], False)

    # ── In-game pause menu ────────────────────────────────────────────
    PAUSE_ITEMS = ["Resume", "Save State", "Load State", "Settings", "Exit to Menu"]

    def _open_pause_menu(self):
        """Enter the paused state; the main loop hands control to _pause_menu_loop."""
        self.paused = True
        self.pause_cursor = 0
        self.pause_settings_cursor = 0
        self.pause_exit_cursor = 0

    def _pause_status(self, msg):
        self._pause_msg = msg
        self._pause_msg_ttl = 120

    def _pause_settings_items(self):
        si = self._set_idx
        return [
            f"Window Scale: {WINDOW_SCALE_OPTIONS[si['scale']]}x",
            f"Frame Rate: {FPS_LIMIT_OPTIONS[si['fps']][0]}",
            f"Audio: {AUDIO_OPTIONS[si['audio']][0]}",
            f"Volume: {VOLUME_OPTIONS[si['volume']][0]}",
            f"Palette: {PALETTE_LIST[si['palette']][0]}",
            f"Filter: {FILTER_OPTIONS[si['filter']][0]}",
            f"Shader: {SHADER_LIST[si['shader']][0]}",
        ]

    def _cycle_pause_setting(self, cursor):
        """Advance the setting under the cursor and apply it to the live emulator.
        Returns True if the window was resized (so the backdrop must be recaptured)."""
        si = self._set_idx
        resized = False
        if cursor == 0:
            si['scale'] = (si['scale'] + 1) % len(WINDOW_SCALE_OPTIONS)
            self.window_scale = WINDOW_SCALE_OPTIONS[si['scale']]
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH * self.window_scale, SCREEN_HEIGHT * self.window_scale))
            resized = True
        elif cursor == 1:
            si['fps'] = (si['fps'] + 1) % len(FPS_LIMIT_OPTIONS)
            self.fps_limit = FPS_LIMIT_OPTIONS[si['fps']][1]
        elif cursor == 2:
            si['audio'] = (si['audio'] + 1) % len(AUDIO_OPTIONS)
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
        elif cursor == 3:
            si['volume'] = (si['volume'] + 1) % len(VOLUME_OPTIONS)
            self._set_volume(VOLUME_OPTIONS[si['volume']][1])
        elif cursor == 4:
            si['palette'] = (si['palette'] + 1) % len(PALETTE_LIST)
            self.ppu.set_palette(PALETTE_LIST[si['palette']][1])
        elif cursor == 5:
            si['filter'] = (si['filter'] + 1) % len(FILTER_OPTIONS)
            self.smooth_scale = FILTER_OPTIONS[si['filter']][1]
        elif cursor == 6:
            si['shader'] = (si['shader'] + 1) % len(SHADER_LIST)
            self.shader = SHADER_LIST[si['shader']][1]
            self._prev_shader_frame = None
        return resized

    def _capture_pause_backdrop(self):
        """Render the current frame, then return a dimmed copy to sit behind the menu."""
        self.render()
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
        self.mmu.joypad_buttons = 0xFF
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
        while self.paused and self.running:
            page, backdrop = self._handle_pause_events(page, backdrop)
            if not self.paused or not self.running:
                break
            if self._pause_msg_ttl > 0:
                self._pause_msg_ttl -= 1
            self._render_pause_page(page, backdrop)
            clock.tick(30)

        # Resume cleanly: re-sync held keys to the joypad and reset the A/V clock
        # so frame pacing doesn't try to "catch up" on the paused wall-clock time.
        if self.running:
            pressed = pygame.key.get_pressed()
            for key, bit in KEY_TO_JOYPAD_BIT.items():
                self.mmu.set_joypad_button(bit, bool(pressed[key]))
        pygame.event.clear()
        self._av_start = time.perf_counter()
        self._sync_samples = 0
        self._sync_frames = 0

    def _handle_pause_events(self, page, backdrop):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.paused = False
                return page, backdrop
            if event.type != pygame.KEYDOWN:
                continue
            if page == "pause":
                if event.key == pygame.K_UP:
                    self.pause_cursor = (self.pause_cursor - 1) % len(self.PAUSE_ITEMS)
                elif event.key == pygame.K_DOWN:
                    self.pause_cursor = (self.pause_cursor + 1) % len(self.PAUSE_ITEMS)
                elif event.key == pygame.K_RETURN:
                    page, backdrop = self._activate_pause_item(page, backdrop)
                elif event.key == pygame.K_ESCAPE:
                    self.paused = False
            elif page == "settings":
                items = self._pause_settings_items()
                if event.key == pygame.K_UP:
                    self.pause_settings_cursor = (self.pause_settings_cursor - 1) % len(items)
                elif event.key == pygame.K_DOWN:
                    self.pause_settings_cursor = (self.pause_settings_cursor + 1) % len(items)
                elif event.key in (pygame.K_RETURN, pygame.K_LEFT, pygame.K_RIGHT):
                    if self._cycle_pause_setting(self.pause_settings_cursor):
                        backdrop = self._capture_pause_backdrop()
                elif event.key == pygame.K_ESCAPE:
                    page = "pause"
            elif page == "confirm_exit":
                if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    self.pause_exit_cursor ^= 1
                elif event.key == pygame.K_RETURN:
                    if self.pause_exit_cursor == 1:
                        self.running = False
                        self.paused = False
                    else:
                        page = "pause"
                elif event.key == pygame.K_ESCAPE:
                    page = "pause"
        return page, backdrop

    def _activate_pause_item(self, page, backdrop):
        choice = self.PAUSE_ITEMS[self.pause_cursor]
        if choice == "Resume":
            self.paused = False
        elif choice == "Save State":
            ok = self.save_state(0)
            self._pause_status("State saved to slot 0" if ok else "Failed to save state")
        elif choice == "Load State":
            if self.load_state(0):
                self._pause_status("State loaded from slot 0")
                backdrop = self._capture_pause_backdrop()
            else:
                self._pause_status("No save state in slot 0")
        elif choice == "Settings":
            self.pause_settings_cursor = 0
            page = "settings"
        elif choice == "Exit to Menu":
            self.pause_exit_cursor = 0
            page = "confirm_exit"
        return page, backdrop

    def _draw_overlay_menu(self, backdrop, title, items, cursor, hint):
        self.screen.blit(backdrop, (0, 0))
        w, h = self.screen.get_size()
        panel_w = min(w - 40, 380)
        panel_h = 70 + len(items) * 42
        px = (w - panel_w) // 2
        py = (h - panel_h) // 2
        panel = pygame.Surface((panel_w, panel_h))
        panel.fill(MENU_BG)
        panel.set_alpha(238)
        self.screen.blit(panel, (px, py))
        pygame.draw.rect(self.screen, MENU_HI, (px, py, panel_w, panel_h), 2)

        tf = pygame.font.Font(None, 38)
        ts = tf.render(title, True, MENU_HI)
        self.screen.blit(ts, (px + (panel_w - ts.get_width()) // 2, py + 16))

        itf = pygame.font.Font(None, 30)
        for i, item in enumerate(items):
            colour = MENU_HI if i == cursor else MENU_FG
            isf = itf.render(item, True, colour)
            iy = py + 62 + i * 42
            ix = px + (panel_w - isf.get_width()) // 2
            self.screen.blit(isf, (ix, iy))
            if i == cursor:
                pygame.draw.rect(self.screen, colour, (ix, iy + 26, isf.get_width(), 2))

        if hint:
            hf = pygame.font.Font(None, 22)
            hs = hf.render(hint, True, MENU_DIM)
            self.screen.blit(hs, ((w - hs.get_width()) // 2, h - 30))
        pygame.display.flip()

    def _render_pause_page(self, page, backdrop):
        if page == "settings":
            self._draw_overlay_menu(backdrop, "Settings", self._pause_settings_items(),
                                    self.pause_settings_cursor,
                                    "Enter: Cycle  |  Esc: Back")
        elif page == "confirm_exit":
            self._draw_overlay_menu(backdrop, "Exit to Menu?",
                                    ["Keep Playing", "Exit to Menu"], self.pause_exit_cursor,
                                    "Tip: Save State (F6) keeps your progress")
        else:
            hint = self._pause_msg if self._pause_msg_ttl > 0 else \
                "Up/Down: Move  |  Enter: Select  |  Esc: Resume"
            self._draw_overlay_menu(backdrop, "Paused", self.PAUSE_ITEMS,
                                    self.pause_cursor, hint)

    def render(self):
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
            arr = fnp
            if self.shader is _shader_lcd_ghost:
                arr = self.shader(arr, prev=self._prev_shader_frame)
                self._prev_shader_frame = arr.copy()
            else:
                arr = self.shader(arr)
            np.copyto(self._blit_np, arr.transpose(1, 0, 2))
            surf = pygame.surfarray.make_surface(self._blit_np)
        else:
            surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            pxa = pygame.PixelArray(surf)
            fb = self.ppu.framebuffer
            for i in range(SCREEN_HEIGHT * SCREEN_WIDTH):
                pxa[i % SCREEN_WIDTH, i // SCREEN_WIDTH] = fb[i]
            pxa.close()
        target_size = self.screen.get_size()
        if target_size == (SCREEN_WIDTH, SCREEN_HEIGHT):
            scaled = surf
        elif self.smooth_scale:
            scaled = pygame.transform.smoothscale(surf, target_size)
        else:
            scaled = pygame.transform.scale(surf, target_size)
        self.screen.blit(scaled, (0, 0))
        # Transient status overlay (save/load messages)
        if getattr(self, '_status_ttl', 0) > 0 and getattr(self, '_status_msg', ''):
            try:
                f = pygame.font.Font(None, 22)
                s = f.render(self._status_msg, True, (255, 255, 200))
                bg = pygame.Surface((s.get_width() + 16, s.get_height() + 8))
                bg.fill((0, 0, 0))
                bg.set_alpha(180)
                self.screen.blit(bg, (8, 8))
                self.screen.blit(s, (16, 12))
            except (pygame.error, AttributeError):
                pass
            self._status_ttl -= 1
        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Game Boy Emulator")
    parser.add_argument("rom", nargs="?", help="Path to the .gb or .gbc ROM file")
    parser.add_argument("--nomenu", action="store_true", help="Skip menu, boot directly into ROM")
    parser.add_argument("--bootrom", help="Path to boot ROM (DMG 256B or CGB ~2304B)")
    args = parser.parse_args()

    if args.nomenu and args.rom:
        emulator = GameBoy(args.rom, bootrom_path=args.bootrom)
        emulator.run()
    else:
        menu = EmulatorMenu()
        menu.run()