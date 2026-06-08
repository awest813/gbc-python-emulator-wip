"""Count opcode frequencies over 2M CPU steps (portable)."""
import os
import sys
from collections import Counter

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

for _ in range(100_000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

counter = Counter()
for _ in range(2_000_000):
    op = cpu.mem[cpu.reg.pc]
    counter[op] += 1
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

print("Top 30 opcodes:")
for op, count in counter.most_common(30):
    pct = count * 100 / 2_000_000
    print("  0x%02X: %7d (%.1f%%)" % (op, count, pct))

low = sum(c for op, c in counter.items() if op < 0x40)
mid_ld = sum(c for op, c in counter.items() if 0x40 <= op < 0x80)
mid_alu = sum(c for op, c in counter.items() if 0x80 <= op < 0xC0)
high = sum(c for op, c in counter.items() if op >= 0xC0)
print("\nBy range:")
print("  0x00-0x3F: %d (%.1f%%)" % (low, low * 100 / 2_000_000))
print("  0x40-0x7F: %d (%.1f%%)" % (mid_ld, mid_ld * 100 / 2_000_000))
print("  0x80-0xBF: %d (%.1f%%)" % (mid_alu, mid_alu * 100 / 2_000_000))
print("  0xC0-0xFF: %d (%.1f%%)" % (high, high * 100 / 2_000_000))
