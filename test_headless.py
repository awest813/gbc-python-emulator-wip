"""
Portable headless smoke test for the GBC emulator.

Unlike the other test_*.py dev scripts (which point at hard-coded local ROM
paths), this one is fully self-contained: it builds a tiny synthetic ROM in
memory and exercises the CPU, MMU, PPU, timers and APU with no external files
and no display. It runs anywhere Python + numpy + pygame are installed:

    python test_headless.py

Exits 0 if every check passes, 1 otherwise.

SPDX-License-Identifier: MIT
"""
import os
import sys
import tempfile

# Force SDL into dummy mode so the test runs without a display or audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    """Exec the emulator source (minus its __main__ block) into a namespace."""
    path = os.path.join(HERE, "gbc_emulator_skeleton.py")
    src = open(path, encoding="utf-8").read().split("if __name__")[0]
    ns = {}
    exec(compile(src, "gbc_emulator_skeleton.py", "exec"), ns)
    return ns


def build_rom(cgb=True):
    """A minimal 32 KB ROM-only cartridge running a tight loop at 0x0100."""
    rom = bytearray(0x8000)
    rom[0x0143] = 0x80 if cgb else 0x00  # CGB flag
    rom[0x0147] = 0x00                   # ROM ONLY
    # 0x0100: enable the LCD, then spin forever.
    prog = [
        0x3E, 0x91,   # LD A,0x91
        0xE0, 0x40,   # LDH (FF40),A   ; LCDC on
        0x00,         # NOP
        0x18, 0xFD,   # JR -3          ; loop on the NOP
    ]
    rom[0x0100:0x0100 + len(prog)] = prog
    return bytes(rom)


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def test_opcodes(ns):
    CPU, MMU = ns["CPU"], ns["MMU"]
    FLAG_Z, FLAG_C, FLAG_H = ns["FLAG_Z"], ns["FLAG_C"], ns["FLAG_H"]
    m = MMU(); m.load_rom(build_rom())
    c = CPU(m)

    # ADD A,n with carry + half-carry + zero result.
    c.reg.a = 0x3A; c._alu_a_op(0, 0xC6)
    check("ADD wraps to 0 with Z/C set",
          c.reg.a == 0 and c.reg.get_flag(FLAG_Z) and c.reg.get_flag(FLAG_C))

    # SUB producing a borrow.
    c.reg.a = 0x10; c._alu_a_op(2, 0x20)
    check("SUB sets N + borrow C",
          c.reg.a == 0xF0 and c.reg.get_flag(FLAG_C))

    # DAA after a BCD addition: 0x09 + 0x08 = 0x11 in BCD.
    c.reg.a = 0x09; c._alu_a_op(0, 0x08); c._daa()
    check("DAA corrects BCD add", c.reg.a == 0x17)

    # CB SWAP (bit field 6) and SRL (bit field 7) must not be transposed.
    c.reg.a = 0xAB; c.execute_cb(0x37)
    check("CB SWAP A", c.reg.a == 0xBA)
    c.reg.a = 0x03; c.execute_cb(0x3F)
    check("CB SRL A", c.reg.a == 0x01 and c.reg.get_flag(FLAG_C))

    # CB BIT sets Z from the inverted tested bit and forces H.
    c.reg.a = 0x01; c.execute_cb(0x47)
    check("CB BIT 0 of set bit -> Z clear", not c.reg.get_flag(FLAG_Z) and c.reg.get_flag(FLAG_H))
    c.reg.a = 0x00; c.execute_cb(0x47)
    check("CB BIT 0 of clear bit -> Z set", c.reg.get_flag(FLAG_Z))

    # Illegal opcodes are tolerated and counted.
    before = c.invalid_opcode_count
    c.reg.pc = 0x4000
    cycles = c.execute(0xD3)
    check("illegal opcode is a 4-cycle no-op",
          cycles == 4 and c.invalid_opcode_count == before + 1)


def test_run(ns):
    """Spin the full machine for a while; nothing should raise."""
    CPU, MMU, PPU, APU, Timers = (ns["CPU"], ns["MMU"], ns["PPU"], ns["APU"], ns["Timers"])
    m = MMU(); m.load_rom(build_rom())
    cpu, ppu, apu, timers = CPU(m), PPU(m), APU(m), Timers(m)
    m.ppu, m.apu = ppu, apu
    m.div_reset_callback = timers.reset_div
    ppu.is_cgb = m.is_cgb
    if m.is_cgb:
        ppu._cgb_init_palettes()
    cpu.reg.pc = 0x0100
    for _ in range(300_000):
        c = cpu.step(); ppu.step(c); timers.step(c); apu.step(c)
    check("valid program runs with no illegal opcodes", cpu.invalid_opcode_count == 0)
    check("LCD is on after boot", m.memory[0xFF40] & 0x80)


def test_double_speed(ns):
    """KEY1 double-speed must halve the base-clock dots fed to the PPU/APU."""
    GameBoy = ns["GameBoy"]
    rom = build_rom()
    with tempfile.NamedTemporaryFile(suffix=".gbc", delete=False) as tf:
        tf.write(rom)
        path = tf.name
    try:
        normal = GameBoy(path, fps_limit=0); normal.cpu.reg.pc = 0x0100
        normal.mmu.key1 = 0x00
        d_norm = sum(normal.step_all() for _ in range(2000))

        fast = GameBoy(path, fps_limit=0); fast.cpu.reg.pc = 0x0100
        fast.mmu.key1 = 0x80  # double-speed active
        d_fast = sum(fast.step_all() for _ in range(2000))
    finally:
        os.unlink(path)
    ratio = d_fast / d_norm
    check(f"double-speed halves PPU dots (ratio {ratio:.3f})", 0.45 < ratio < 0.55)


def main():
    ns = load_module()
    print("opcode checks:");        test_opcodes(ns)
    print("full-machine run:");     test_run(ns)
    print("double-speed timing:");  test_double_speed(ns)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
