"""Timed smoke run over 2M CPU steps (portable)."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

src = open(os.path.join(HERE, "gbc_emulator_skeleton.py"), encoding="utf-8").read()
src = src.split("if __name__")[0]
exec(compile(src, "gbc_emulator_skeleton.py", "exec"))

rom = bytearray(0x8000)
rom[0x0143] = 0x80
rom[0x0147] = 0x00
prog = [0x3E, 0x91, 0xE0, 0x40, 0x00, 0x18, 0xFD]
rom[0x0100:0x0100 + len(prog)] = prog

mmu = MMU()
mmu.load_rom(bytes(rom))
cpu = CPU(mmu)
ppu = PPU(mmu)
mmu.ppu = ppu
apu = APU(mmu)
mmu.apu = apu
timers = Timers(mmu)
mmu.div_reset_callback = timers.reset_div
ppu.is_cgb = mmu.is_cgb
ppu._cgb_init_palettes()

cpu.reg.a = 0x11
cpu.reg.f = 0x80
cpu.reg.d = 0xFF
cpu.reg.e = 0x56
cpu.reg.l = 0x0D
cpu.reg.sp = 0xFFFE
cpu.reg.pc = 0x0100

start = time.perf_counter()
invalid_opcodes = 0
for i in range(2_000_000):
    c = cpu.step()
    if cpu.invalid_opcode_count > invalid_opcodes:
        invalid_opcodes = cpu.invalid_opcode_count
    ppu.step(c)
    timers.step(c)
    apu.step(c)
    if i and i % 500_000 == 0:
        print(f"  i={i}, PC=0x{cpu.reg.pc:04X}, LY={mmu.memory[0xFF44]:3d}")

elapsed = time.perf_counter() - start
print(f"Total time: {elapsed:.2f}s ({2_000_000 / elapsed:.0f} steps/s)")
print(f"Final PC: 0x{cpu.reg.pc:04X}, invalid opcodes: {invalid_opcodes}")
fb = ppu.framebuffer
print(f"Unique framebuffer colors: {len({fb[i] for i in range(160 * 144)})}")
