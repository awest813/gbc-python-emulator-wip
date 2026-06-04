import sys
from collections import Counter
sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')
src = open(r'C:\Users\allen\Downloads\GBC\gbc_emulator_skeleton.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(compile(src, 'gbc_emulator_skeleton.py', 'exec'))

rom_path = r'C:\Users\allen\Downloads\rom (2)\extracted\Dragon Warrior III (USA).gbc'
with open(rom_path, 'rb') as f:
    rom = f.read()
mmu = MMU()
mmu.load_rom(rom)
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

# Warm up
for i in range(100000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

# Count opcodes
counter = Counter()
for i in range(2000000):
    op = cpu.mem[cpu.reg.pc]
    counter[op] += 1
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

# Print top 30 opcodes
print('Top 30 opcodes:')
for op, count in counter.most_common(30):
    pct = count * 100 / 2000000
    print('  0x%02X: %7d (%.1f%%)' % (op, count, pct))

# Group by range
low = sum(c for op, c in counter.items() if op < 0x40)
mid_ld = sum(c for op, c in counter.items() if 0x40 <= op < 0x80)
mid_alu = sum(c for op, c in counter.items() if 0x80 <= op < 0xC0)
high = sum(c for op, c in counter.items() if op >= 0xC0)
print('\nBy range:')
print('  0x00-0x3F: %d (%.1f%%)' % (low, low*100/2000000))
print('  0x40-0x7F: %d (%.1f%%)' % (mid_ld, mid_ld*100/2000000))
print('  0x80-0xBF: %d (%.1f%%)' % (mid_alu, mid_alu*100/2000000))
print('  0xC0-0xFF: %d (%.1f%%)' % (high, high*100/2000000))
