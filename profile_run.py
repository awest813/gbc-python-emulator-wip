"""Profile CPU/PPU/timer/APU stepping with cProfile (portable)."""
import os
import sys
import cProfile
import pstats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

src = open(os.path.join(HERE, "gbc_emulator_skeleton.py"), encoding="utf-8").read()
src = src.split("if __name__")[0]
exec(compile(src, "gbc_emulator_skeleton.py", "exec"))

# Minimal synthetic ROM: enable LCD, then tight JR loop.
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

for _ in range(100_000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

profiler = cProfile.Profile()
profiler.enable()
for _ in range(2_000_000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)
profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats(30)
print("---")
stats.sort_stats("tottime")
stats.print_stats(20)
