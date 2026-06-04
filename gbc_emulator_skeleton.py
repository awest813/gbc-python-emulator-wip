"""
Game Boy / Game Boy Color emulator written in Python.
Supports MBC1/MBC5 cartridges, BG/Window/Sprite rendering, and
includes a built-in menu with ROM browser.

Requires: pygame, numpy  (pip install pygame numpy)
"""
import sys
import os
import time
import logging
import argparse

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

# --- CONSTANTS ---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
FPS = 59.73
CYCLES_PER_FRAME = 70224  # 4.194304 MHz / 59.73 frames per second

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

    def load_rom(self, rom_data):
        self.rom_data = bytearray(rom_data)
        self.mbc_type = self.rom_data[0x0147] if len(self.rom_data) > 0x0147 else 0x00

        rom_size_code = self.rom_data[0x0148] if len(self.rom_data) > 0x0148 else 0
        ram_size_code = self.rom_data[0x0149] if len(self.rom_data) > 0x0149 else 0
        rom_size_map = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128, 7: 256, 8: 512}
        ram_size_map = {0: 0, 1: 1, 2: 1, 3: 4, 4: 16, 5: 8}
        self.num_rom_banks = rom_size_map.get(rom_size_code, 2)
        self.num_ram_banks = ram_size_map.get(ram_size_code, 0)

        mbc_ram_types = {0x02, 0x03, 0x0F, 0x10, 0x12, 0x13, 0x1A, 0x1B, 0x1D, 0x1E}
        mbc_battery_types = {0x03, 0x06, 0x0F, 0x10, 0x13, 0x1B, 0x1E}
        self.has_ram = self.mbc_type in mbc_ram_types or self.num_ram_banks > 0
        self.has_battery = self.mbc_type in mbc_battery_types
        if self.has_ram and self.num_ram_banks > 0:
            self.ram_data = bytearray(self.num_ram_banks * 0x2000)

        self.memory[0:0x8000] = self.rom_data[0:min(0x8000, len(self.rom_data))]
        mbc_name = {0x00:"ROM ONLY", 0x01:"MBC1", 0x02:"MBC1+RAM", 0x03:"MBC1+RAM+BATT",
                    0x05:"MBC2", 0x06:"MBC2+BATT", 0x0F:"MBC3+TIMER+BATT",
                    0x10:"MBC3+TIMER+RAM+BATT", 0x11:"MBC3", 0x12:"MBC3+RAM",
                    0x13:"MBC3+RAM+BATT", 0x19:"MBC5", 0x1A:"MBC5+RAM",
                    0x1B:"MBC5+RAM+BATT", 0x1C:"MBC5+RUMBLE", 0x1D:"MBC5+RUMBLE+RAM",
                    0x1E:"MBC5+RUMBLE+RAM+BATT"}.get(self.mbc_type, f"UNKNOWN(0x{self.mbc_type:02X})")
        logging.info(f"Loaded ROM: {len(rom_data)} bytes [{mbc_name}, {self.num_rom_banks} ROM banks, {self.num_ram_banks} RAM banks]")

    def read_byte(self, address):
        # Fast path: most accesses hit WRAM/HRAM/VRAM/OAM (direct array)
        if address >= 0xC000:
            if address < 0xFE00:
                if address < 0xE000:
                    return self.memory[address]
                return self.memory[address - 0x2000]  # Echo RAM
            if address == 0xFF00:
                return self._read_joypad()
            return self.memory[address]
        # VRAM (0x8000-0x9FFF)
        if address >= 0x8000:
            return self.memory[address]
        # Cartridge RAM (0xA000-0xBFFF) - slow path
        if address >= 0xA000:
            if self.has_ram and self.ram_enabled and len(self.ram_data) > 0:
                offset = self.ram_bank * 0x2000 + (address - 0xA000)
                return self.ram_data[offset] if offset < len(self.ram_data) else 0xFF
            return 0xFF
        # ROM (0x0000-0x7FFF) - slow path
        if address < 0x4000:
            return self.rom_data[address] if address < len(self.rom_data) else 0xFF
        if self.mbc_type == 0x00:
            return self.rom_data[address] if address < len(self.rom_data) else 0xFF
        offset = self.rom_bank * 0x4000 + (address - 0x4000)
        return self.rom_data[offset] if offset < len(self.rom_data) else 0xFF

    def _read_joypad(self):
        sel = self.memory[0xFF00] & 0x30
        line_dir = 0x0F
        if not (self.joypad_buttons & 0x01): line_dir &= ~0x01
        if not (self.joypad_buttons & 0x02): line_dir &= ~0x02
        if not (self.joypad_buttons & 0x04): line_dir &= ~0x04
        if not (self.joypad_buttons & 0x08): line_dir &= ~0x08
        line_act = 0x0F
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
        hi = self.read_byte(address + 1)
        return (hi << 8) | lo

    def write_byte(self, address, value):
        value &= 0xFF
        if address < 0x8000:
            self._handle_mbc_write(address, value)
        elif 0xA000 <= address <= 0xBFFF:
            if self.has_ram and self.ram_enabled and len(self.ram_data) > 0:
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
        elif address == 0xFF46:
            self._dma_transfer(value)
        else:
            self.memory[address] = value

    def _dma_transfer(self, value):
        src_base = value << 8
        for i in range(160):
            self.memory[0xFE00 + i] = self.read_byte(src_base + i)

    def write_word(self, address, value):
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)

    def _handle_mbc_write(self, address, value):
        if self.mbc_type == 0x00:
            return
        if 0x0000 <= address <= 0x1FFF:
            self.ram_enabled = (value & 0x0F) == 0x0A
        elif self.mbc_type in (0x01, 0x02, 0x03):
            self._mbc1_write(address, value)
        elif self.mbc_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
            self._mbc5_write(address, value)

    def _mbc1_write(self, address, value):
        if 0x2000 <= address <= 0x3FFF:
            bank = value & 0x1F
            if bank == 0: bank = 1
            self.rom_bank = (self.rom_bank & 0x60) | bank
            self._update_bank_mirror()
        elif 0x4000 <= address <= 0x5FFF:
            self.ram_bank = value & 0x03
            self.rom_bank = (self.rom_bank & 0x1F) | ((value & 0x03) << 5)
            self._update_bank_mirror()
        elif 0x6000 <= address <= 0x7FFF:
            self.mbc1_mode = value & 0x01
            if self.mbc1_mode == 1:
                self.rom_bank &= 0x1F
                self._update_bank_mirror()

    def _mbc5_write(self, address, value):
        if 0x2000 <= address <= 0x2FFF:
            self.rom_bank = (self.rom_bank & 0x100) | value
            self._update_bank_mirror()
        elif 0x3000 <= address <= 0x3FFF:
            self.rom_bank = (self.rom_bank & 0xFF) | ((value & 0x01) << 8)
            self._update_bank_mirror()
        elif 0x4000 <= address <= 0x5FFF:
            self.ram_bank = value & 0x0F

    def _update_bank_mirror(self):
        bank = self.rom_bank
        offset = bank * 0x4000
        if offset < len(self.rom_data):
            end = min(offset + 0x4000, len(self.rom_data))
            self.memory[0x4000:0x4000 + (end - offset)] = self.rom_data[offset:end]


class CPU:
    """The LR35902 CPU."""
    def __init__(self, mmu):
        self.mmu = mmu
        self.reg = Registers()
        self.halted = False
        self.interrupts_master_enabled = False
        self.trace_enabled = False

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
        val = self.mmu.memory[pc]
        self.reg.pc = (pc + 1) & 0xFFFF
        return val

    def fetch_word(self):
        """Fetches the next 16-bit word at PC and increments PC twice."""
        pc = self.reg.pc
        mem = self.mmu.memory
        val = mem[pc] | (mem[(pc + 1) & 0xFFFF] << 8)
        self.reg.pc = (pc + 2) & 0xFFFF
        return val

    def step(self):
        """Fetches, decodes, and executes a single instruction."""
        mem = self.mmu.memory
        if self.halted:
            if mem[0xFFFF] & mem[0xFF0F]:
                self.halted = False
            return 4

        if self.trace_enabled:
            a = self.reg.a; f = self.reg.f; b = self.reg.b; c = self.reg.c
            d = self.reg.d; e = self.reg.e; h = self.reg.h; l = self.reg.l
            pc = self.reg.pc; sp = self.reg.sp
            op = self.mmu.read_byte(pc)
            print(f"A:{a:02X} F:{f:02X} B:{b:02X} C:{c:02X} D:{d:02X} E:{e:02X} H:{h:02X} L:{l:02X} SP:{sp:04X} PCMEM:{op:02X}")

        pc = self.reg.pc
        opcode = mem[pc]
        self.reg.pc = pc + 1
        cycles = self.execute(opcode)
        cycles += self._handle_interrupts()
        return cycles

    def _handle_interrupts(self):
        if not self.interrupts_master_enabled:
            return 0
        pending = self.mmu.memory[0xFFFF] & self.mmu.memory[0xFF0F]
        if pending == 0:
            return 0
        self.interrupts_master_enabled = False
        for bit in range(5):
            if pending & (1 << bit):
                self.mmu.memory[0xFF0F] &= ~(1 << bit)
                self._push(self.reg.pc)
                self.reg.pc = 0x0040 + (bit * 8)
                return 20
        return 0

    def execute(self, opcode):
        # Hot-path opcodes first: NOP, LD r,r (0x40-0x7F), ALU A,r (0x80-0xBF).
        # These cover ~80% of DMG instruction mix and short-circuit the long
        # if/elif chain below.
        if opcode < 0x40:
            return self._exec_low(opcode)
        if opcode < 0x80:
            if opcode == 0x76:
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
            self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
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
                self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
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
                self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
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
                self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
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
                self.reg.pc = (self.reg.pc + offset) & 0xFFFF; return 12
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
                self.reg.pc = self._pop(); return 20
            return 8
        if opcode == 0xC1:
            self.reg.bc = self._pop(); return 12
        if opcode == 0xC2:
            addr = self.fetch_word()
            if self._check_cond(0):
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xC3:
            self.reg.pc = self.fetch_word(); return 16
        if opcode == 0xC4:
            addr = self.fetch_word()
            if self._check_cond(0):
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xC5:
            self._push(self.reg.bc); return 16
        if opcode == 0xC6:
            self._alu_a_op(0, self.fetch_byte()); return 8
        if opcode == 0xC7:
            self._push(self.reg.pc); self.reg.pc = 0x00; return 16
        if opcode == 0xC8:
            if self._check_cond(1):
                self.reg.pc = self._pop(); return 20
            return 8
        if opcode == 0xC9:
            self.reg.pc = self._pop(); return 16
        if opcode == 0xCA:
            addr = self.fetch_word()
            if self._check_cond(1):
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xCB:
            cb_opcode = self.fetch_byte(); return self.execute_cb(cb_opcode)
        if opcode == 0xCC:
            addr = self.fetch_word()
            if self._check_cond(1):
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xCD:
            addr = self.fetch_word()
            self._push(self.reg.pc)
            self.reg.pc = addr; return 24
        if opcode == 0xCE:
            self._alu_a_op(1, self.fetch_byte()); return 8
        if opcode == 0xCF:
            self._push(self.reg.pc); self.reg.pc = 0x08; return 16
        if opcode == 0xD0:
            if self._check_cond(2):
                self.reg.pc = self._pop(); return 20
            return 8
        if opcode == 0xD1:
            self.reg.de = self._pop(); return 12
        if opcode == 0xD2:
            addr = self.fetch_word()
            if self._check_cond(2):
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xD4:
            addr = self.fetch_word()
            if self._check_cond(2):
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xD5:
            self._push(self.reg.de); return 16
        if opcode == 0xD6:
            self._alu_a_op(2, self.fetch_byte()); return 8
        if opcode == 0xD7:
            self._push(self.reg.pc); self.reg.pc = 0x10; return 16
        if opcode == 0xD8:
            if self._check_cond(3):
                self.reg.pc = self._pop(); return 20
            return 8
        if opcode == 0xD9:
            self.reg.pc = self._pop()
            self.interrupts_master_enabled = True; return 16
        if opcode == 0xDA:
            addr = self.fetch_word()
            if self._check_cond(3):
                self.reg.pc = addr; return 16
            return 12
        if opcode == 0xDC:
            addr = self.fetch_word()
            if self._check_cond(3):
                self._push(self.reg.pc)
                self.reg.pc = addr; return 24
            return 12
        if opcode == 0xDE:
            self._alu_a_op(3, self.fetch_byte()); return 8
        if opcode == 0xDF:
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
            self.reg.pc = self.reg.hl; return 4
        if opcode == 0xEA:
            self.mmu.write_byte(self.fetch_word(), self.reg.a); return 16
        if opcode == 0xEE:
            self._alu_a_op(5, self.fetch_byte()); return 8
        if opcode == 0xEF:
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
            self.interrupts_master_enabled = True; return 4
        if opcode == 0xFE:
            self._alu_a_op(7, self.fetch_byte()); return 8
        if opcode == 0xFF:
            self._push(self.reg.pc); self.reg.pc = 0x38; return 16
        logging.error(f"Unimplemented Opcode: {hex(opcode)} at PC: {hex(self.reg.pc - 1)}")
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


class PPU:
    """Picture Processing Unit with background rendering."""
    _SHADES = [
        (224, 248, 208),
        (136, 192, 112),
        (52, 104, 86),
        (8, 24, 32),
    ]

    # Precomputed tables (built once at class definition time):
    # _TILE_COLORS[k] where k = (hi<<8)|lo -> tuple of 8 color_idx values
    # _PALETTE_SHADES[bgp] -> tuple of 4 shade_idx values (one per color_idx)
    _TILE_COLORS = tuple(
        tuple((((hi >> p) & 1) << 1 | ((lo >> p) & 1) for p in range(7, -1, -1)))
        for lo in range(256) for hi in range(256)
    )
    _PALETTE_SHADES = tuple(
        tuple((bgp >> (c << 1)) & 0x03 for c in range(4))
        for bgp in range(256)
    )

    def __init__(self, mmu):
        self.mmu = mmu
        self.cycles = 0
        self.mode = 2
        self.framebuffer = [(255, 255, 255)] * (SCREEN_WIDTH * SCREEN_HEIGHT)
        self.bg_palette_idx = bytearray(SCREEN_WIDTH * SCREEN_HEIGHT)

    def step(self, cycles_passed):
        cycles = self.cycles + cycles_passed
        mem = self.mmu.memory
        lcdc = mem[0xFF40]
        if not (lcdc & 0x80):
            self.cycles = cycles
            return

        while cycles >= 456:
            cycles -= 456
            ly = mem[0xFF44]

            if ly < 144:
                self._update_stat_mode(2)
                self._update_stat_mode(3)
                if lcdc & 0x01:
                    self._render_scanline(ly)
                self._update_stat_mode(0)
            else:
                self._update_stat_mode(1)

            new_ly = ly + 1
            if new_ly == 154:
                new_ly = 0
            mem[0xFF44] = new_ly

            if new_ly == 144:
                mem[0xFF0F] |= 0x01

            self._check_lyc(new_ly)

        self.cycles = cycles

    def _update_stat_mode(self, mode):
        self.mode = mode
        mem = self.mmu.memory
        stat = (mem[0xFF41] & 0xFC) | mode
        mem[0xFF41] = stat
        if mode == 0 and (stat & 0x08):
            mem[0xFF0F] |= 0x02
        elif mode == 1 and (stat & 0x10):
            mem[0xFF0F] |= 0x02
        elif mode == 2 and (stat & 0x20):
            mem[0xFF0F] |= 0x02

    def _check_lyc(self, ly):
        mem = self.mmu.memory
        lyc = mem[0xFF45]
        stat = mem[0xFF41]
        if ly == lyc:
            stat |= 0x04
            mem[0xFF41] = stat
            if stat & 0x40:
                mem[0xFF0F] |= 0x02
        else:
            mem[0xFF41] = stat & ~0x04

    def _render_scanline(self, ly):
        mem = self.mmu.memory
        lcdc = mem[0xFF40]
        bg_map_base = 0x9C00 if (lcdc & 0x08) else 0x9800
        signed_tiles = not (lcdc & 0x10)
        tile_base = 0x8000 if (lcdc & 0x10) else 0x8800

        scy = mem[0xFF42]
        scx = mem[0xFF43]
        bgp = mem[0xFF47]

        bg_y = (scy + ly) & 0xFF
        tile_row = bg_y >> 3
        pixel_row = bg_y & 7

        fb_row = ly * SCREEN_WIDTH
        shades = self._SHADES
        fb = self.framebuffer
        bg_pri = self.bg_palette_idx
        tile_colors = self._TILE_COLORS
        palette_shades = self._PALETTE_SHADES[bgp]
        scx_mod = scx & 7
        first_tile_col = scx >> 3
        # Fast path: when scx is 8-aligned, every tile column is fully on-screen,
        # so the inner 8-pixel loop has no bounds checks.
        if scx_mod == 0:
            for tile_col_offset in range(20):
                tile_col = (first_tile_col + tile_col_offset) & 0x1F
                map_addr = bg_map_base + (tile_row << 5) + tile_col
                tile_idx = mem[map_addr]
                if signed_tiles:
                    addr = 0x9000 + ((tile_idx if tile_idx < 128 else tile_idx - 256) << 4)
                else:
                    addr = tile_base + (tile_idx << 4)
                addr += pixel_row << 1
                lo = mem[addr]
                hi = mem[addr + 1]
                c0, c1, c2, c3, c4, c5, c6, c7 = tile_colors[(hi << 8) | lo]
                base = fb_row + tile_col_offset * 8
                fb[base]     = shades[palette_shades[c0]]
                fb[base + 1] = shades[palette_shades[c1]]
                fb[base + 2] = shades[palette_shades[c2]]
                fb[base + 3] = shades[palette_shades[c3]]
                fb[base + 4] = shades[palette_shades[c4]]
                fb[base + 5] = shades[palette_shades[c5]]
                fb[base + 6] = shades[palette_shades[c6]]
                fb[base + 7] = shades[palette_shades[c7]]
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
            for tile_col_offset in range(21):
                tile_col = (first_tile_col + tile_col_offset) & 0x1F
                map_addr = bg_map_base + (tile_row << 5) + tile_col
                tile_idx = mem[map_addr]
                if signed_tiles:
                    addr = 0x9000 + ((tile_idx if tile_idx < 128 else tile_idx - 256) << 4)
                else:
                    addr = tile_base + (tile_idx << 4)
                addr += pixel_row << 1
                lo = mem[addr]
                hi = mem[addr + 1]
                colors = tile_colors[(hi << 8) | lo]
                tile_x_start = tile_col_offset * 8 - scx_mod
                for p in range(8):
                    x = tile_x_start + p
                    if x < 0 or x >= SCREEN_WIDTH:
                        continue
                    c = colors[p]
                    fb[fb_row + x] = shades[palette_shades[c]]
                    bg_pri[fb_row + x] = c

        if lcdc & 0x20:
            self._render_window(ly)
        if lcdc & 0x02:
            self._render_sprites(ly)

    def _render_window(self, ly):
        mem = self.mmu.memory
        lcdc = mem[0xFF40]
        wy = mem[0xFF4A]
        wx_raw = mem[0xFF4B]
        if ly < wy:
            return
        win_x_offset = wx_raw - 7
        if win_x_offset >= SCREEN_WIDTH:
            return
        win_map_base = 0x9C00 if (lcdc & 0x40) else 0x9800
        signed_tiles = not (lcdc & 0x10)
        tile_base = 0x8000 if (lcdc & 0x10) else 0x8800
        bgp = mem[0xFF47]
        win_y = ly - wy
        tile_row = win_y >> 3
        pixel_row = win_y & 7
        fb_row = ly * SCREEN_WIDTH
        shades = self._SHADES
        fb = self.framebuffer
        bg_pri = self.bg_palette_idx
        tile_colors = self._TILE_COLORS
        palette_shades = self._PALETTE_SHADES[bgp]
        for tile_col in range(21):
            x = tile_col * 8 + win_x_offset
            if x >= SCREEN_WIDTH:
                break
            if x + 8 <= 0:
                continue
            map_addr = win_map_base + (tile_row << 5) + tile_col
            tile_idx = mem[map_addr]
            if signed_tiles:
                addr = 0x9000 + ((tile_idx if tile_idx < 128 else tile_idx - 256) << 4)
            else:
                addr = tile_base + (tile_idx << 4)
            addr += pixel_row << 1
            lo = mem[addr]
            hi = mem[addr + 1]
            colors = tile_colors[(hi << 8) | lo]
            for p in range(8):
                px = x + p
                if px < 0 or px >= SCREEN_WIDTH:
                    continue
                c = colors[p]
                fb[fb_row + px] = shades[palette_shades[c]]
                bg_pri[fb_row + px] = c

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
        sprites.sort(key=lambda s: s[0])
        obp0 = mem[0xFF48]
        obp1 = mem[0xFF49]
        fb_row = ly * SCREEN_WIDTH
        bg_pri = self.bg_palette_idx
        fb = self.framebuffer
        shades = self._SHADES
        for spr_x, spr_y, tile, flags in sprites:
            sprite_pixel_y = ly - spr_y
            if flags & 0x40:
                sprite_pixel_y = sprite_height - 1 - sprite_pixel_y
            if sprite_height == 16:
                tile_row_offset = sprite_pixel_y // 8
                tile_idx_used = (tile & 0xFE) + tile_row_offset
            else:
                tile_idx_used = tile
            tile_addr = 0x8000 + tile_idx_used * 16 + (sprite_pixel_y % 8) * 2
            lo = mem[tile_addr]
            hi = mem[tile_addr + 1]
            for sx in range(8):
                pixel_x = spr_x + sx
                if pixel_x < 0 or pixel_x >= SCREEN_WIDTH:
                    continue
                pixel_col = 7 - sx if not (flags & 0x20) else sx
                color_idx = ((hi >> pixel_col) & 1) << 1 | ((lo >> pixel_col) & 1)
                if color_idx == 0:
                    continue
                if flags & 0x80 and bg_pri[fb_row + pixel_x] != 0:
                    continue
                palette = obp1 if (flags & 0x10) else obp0
                shade = (palette >> (color_idx * 2)) & 0x03
                fb[fb_row + pixel_x] = shades[shade]


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
        tima = mem[0xFF05] + overflows
        if tima > 0xFF:
            mem[0xFF05] = mem[0xFF06]
            mem[0xFF0F] |= 0x04
        else:
            mem[0xFF05] = tima


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
        self.window_scale = 4
        self.status_line = ""
        self.status_ttl = 0
        if pygame is None:
            print("=" * 55)
            print("  ERROR: pygame is required for the emulator menu.")
            print("  Install it with:  pip install pygame numpy")
            print("=" * 55)
            sys.exit(1)
        pygame.init()
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
                        self.settings_items[0] = f"Window Scale: {self.window_scale}x"
                        self.settings_cursor = 0
                        return "settings"
                    elif self.selected == 2:
                        pygame.quit()
                        sys.exit()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
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
                            gb = GameBoy(path, self.window_scale)
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
                        self.settings_items[0] = f"Window Scale: {self.window_scale}x"
                elif event.key == pygame.K_ESCAPE:
                    self.selected = 1
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
        self._draw_menu(self.settings_items, self.settings_cursor, 180, 50)
        self._centre_text("Enter: Change  |  Esc: Back", MENU_H - 30, MENU_DIM, 18)


class GameBoy:
    """The main emulator orchestrator class."""
    def __init__(self, rom_path=None, window_scale=4):
        self.mmu = MMU()
        self.cpu = CPU(self.mmu)
        self.ppu = PPU(self.mmu)
        self.timers = Timers(self.mmu)
        self.mmu.div_reset_callback = self.timers.reset_div
        self.mmu.memory[0xFF04] = 0x00
        self.mmu.memory[0xFF05] = 0x00
        self.mmu.memory[0xFF06] = 0x00
        self.mmu.memory[0xFF07] = 0xF8
        
        if rom_path:
            with open(rom_path, 'rb') as f:
                self.mmu.load_rom(f.read())
        else:
            logging.info("No ROM provided. Running dummy infinite loop.")
            self.mmu.memory[0x0100] = 0x00
            self.mmu.memory[0x0101] = 0xC3
            self.mmu.memory[0x0102] = 0x00
            self.mmu.memory[0x0103] = 0x01

        if pygame:
            self.window_scale = window_scale
            self.screen = pygame.display.set_mode((SCREEN_WIDTH * self.window_scale, SCREEN_HEIGHT * self.window_scale))
            pygame.display.set_caption(f"Python GBC Emulator - {os.path.basename(rom_path) if rom_path else 'No ROM'}")
        elif not rom_path:
            logging.warning("pygame not available — running headless is not useful without a ROM.")

    def run(self):
        """Main execution loop.  Returns to caller when user presses Escape or closes window."""
        self.running = True
        clock = time.time()
        
        while self.running:
            cycles_this_frame = 0
            while cycles_this_frame < CYCLES_PER_FRAME:
                cycles = self.cpu.step()
                self.ppu.step(cycles)
                self.timers.step(cycles)
                cycles_this_frame += cycles

            if pygame:
                self.handle_events()
                self.render()

            elapsed = time.time() - clock
            target_time = 1.0 / FPS
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
            clock = time.time()

            if not pygame and int(time.time() * 10) % 10 == 0:
                print(f"Running... PC: {hex(self.cpu.reg.pc)}")

    def handle_events(self):
        """Process window events and Joypad inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in KEY_TO_JOYPAD_BIT:
                    self.mmu.set_joypad_button(KEY_TO_JOYPAD_BIT[event.key], True)
            elif event.type == pygame.KEYUP:
                if event.key in KEY_TO_JOYPAD_BIT:
                    self.mmu.set_joypad_button(KEY_TO_JOYPAD_BIT[event.key], False)

    def render(self):
        """Draws the PPU framebuffer to the Pygame screen."""
        if np is not None:
            arr = np.array(self.ppu.framebuffer, dtype=np.uint8).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 3)
            arr = np.ascontiguousarray(arr.transpose(1, 0, 2))
            surf = pygame.surfarray.make_surface(arr)
        else:
            surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            pxa = pygame.PixelArray(surf)
            fb = self.ppu.framebuffer
            for i in range(SCREEN_HEIGHT * SCREEN_WIDTH):
                r, g, b = fb[i]
                pxa[i % SCREEN_WIDTH, i // SCREEN_WIDTH] = (r << 16) | (g << 8) | b
            pxa.close()
        target_size = self.screen.get_size()
        if target_size == (SCREEN_WIDTH, SCREEN_HEIGHT):
            scaled = surf
        else:
            scaled = pygame.transform.scale(surf, target_size)
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Game Boy Emulator")
    parser.add_argument("rom", nargs="?", help="Path to the .gb or .gbc ROM file")
    parser.add_argument("--nomenu", action="store_true", help="Skip menu, boot directly into ROM")
    args = parser.parse_args()

    if args.nomenu and args.rom:
        emulator = GameBoy(args.rom)
        emulator.run()
    else:
        menu = EmulatorMenu()
        menu.run()