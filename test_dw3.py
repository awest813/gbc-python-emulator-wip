import sys
import time
import os

sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')
src = open(r'C:\Users\allen\Downloads\GBC\gbc_emulator_skeleton.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(compile(src, 'gbc_emulator_skeleton.py', 'exec'))

import numpy as np
import imageio.v3 as iio

rom_path = os.path.join('roms', 'Dragon Warrior III (USA).gbc')
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
for i in range(8000000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

# Save framebuffer
fb = ppu.framebuffer
arr = np.array(fb, dtype=np.uint8).reshape(144, 160, 3)
big = arr.repeat(4, axis=0).repeat(4, axis=1)
iio.imwrite('dw3.png', big)
print(f'Final PC: 0x{cpu.reg.pc:04X}, LY: {mmu.memory[0xFF44]}, LCDC: 0x{mmu.memory[0xFF40]:02X}')
print(f'Unique colors: {len(np.unique(arr.reshape(-1, 3), axis=0))}')

# Also check colors
colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
order = np.argsort(-counts)
for i in range(min(8, len(order))):
    idx = order[i]
    print(f'  RGB{tuple(colors[idx])}: {counts[idx]} pixels')
