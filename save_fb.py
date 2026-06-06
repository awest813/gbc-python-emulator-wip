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

# Run for a while then save the framebuffer to a file
for i in range(5000000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)
    if i % 1000000 == 0:
        print(f'  i={i}, PC=0x{cpu.reg.pc:04X}, LY={mmu.memory[0xFF44]:3d}')

# Print framebuffer
fb = ppu.framebuffer
print(f'Final PC: 0x{cpu.reg.pc:04X}, LY: {mmu.memory[0xFF44]}, LCDC: 0x{mmu.memory[0xFF40]:02X}')

# Save framebuffer to PPM
with open('fb.ppm', 'wb') as f:
    f.write(b'P6\n160 144\n255\n')
    for y in range(144):
        for x in range(160):
            r, g, b = fb[y*160+x]
            f.write(bytes([r, g, b]))
print('Wrote fb.ppm')
