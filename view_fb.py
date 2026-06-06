import sys
import time

sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')
src = open(r'C:\Users\allen\Downloads\GBC\gbc_emulator_skeleton.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(compile(src, 'gbc_emulator_skeleton.py', 'exec'))

import os
rom_path = os.path.join('roms', 'game.gb')
print(f'=== {os.path.basename(rom_path)} ===')
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
if mmu.is_cgb:
    ppu._cgb_init_palettes()
cpu.reg.a = 0x11
cpu.reg.f = 0x80
cpu.reg.d = 0xFF
cpu.reg.e = 0x56
cpu.reg.l = 0x0D
cpu.reg.sp = 0xFFFE
cpu.reg.pc = 0x0100

# Run for a while
for i in range(5000000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

# Print framebuffer as ASCII art
fb = ppu.framebuffer
print(f'Final PC: 0x{cpu.reg.pc:04X}, LY: {mmu.memory[0xFF44]}, LCDC: 0x{mmu.memory[0xFF40]:02X}')

# Compute unique colors and counts
colors = {}
for idx in range(160*144):
    c = fb[idx]
    colors[c] = colors.get(c, 0) + 1
top = sorted(colors.items(), key=lambda x: -x[1])[:8]
print('Top colors:')
for c, count in top:
    print(f'  RGB{c}: {count} pixels')

# Map each color to a character
char_map = {}
chars = ' .:-=+*#%@'
for i, (c, _) in enumerate(top):
    char_map[c] = chars[i] if i < len(chars) else '?'

# Print every 4th row and 2nd column for ASCII view
for y in range(0, 144, 4):
    row = ''
    for x in range(0, 160, 2):
        c = fb[y*160+x]
        row += char_map.get(c, '?')
    print(row)
