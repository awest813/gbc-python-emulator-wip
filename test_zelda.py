"""Run a ROM headless and save a PNG of the framebuffer.
Usage: python test_zelda.py <path-to-rom>"""
import os
import sys
import time
import imageio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "gbc_emulator_skeleton.py"), encoding="utf-8").read()
src = src.split('if __name__')[0]
exec(compile(src, 'gbc_emulator_skeleton.py', 'exec'))

if len(sys.argv) < 2:
    print("Usage: python test_zelda.py <path-to-rom>")
    sys.exit(1)
rom_path = sys.argv[1]
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

# Post-boot CGB state
cpu.reg.a = 0x11
cpu.reg.f = 0x80
cpu.reg.b = 0x00
cpu.reg.c = 0x00
cpu.reg.d = 0xFF
cpu.reg.e = 0x56
cpu.reg.h = 0x00
cpu.reg.l = 0x0D
cpu.reg.sp = 0xFFFE
cpu.reg.pc = 0x0100

start = time.time()
steps = 6000000

print(f"Running {steps} steps...")
for i in range(steps):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)
    if i % 1000000 == 0:
        print(f"  i={i:7d}, PC=0x{cpu.reg.pc:04X}, LY={mmu.memory[0xFF44]:3d}, LCDC=0x{mmu.memory[0xFF40]:02X}")

elapsed = time.time() - start
print(f"Total time: {elapsed:.2f}s")
print(f"Final PC: 0x{cpu.reg.pc:04X}")
print(f"Final LY: {mmu.memory[0xFF44]}")

# Save framebuffer
import numpy as np
fb = ppu.framebuffer
colors = set()
for idx in range(160*144):
    colors.add(fb[idx])
print(f"Unique colors in framebuffer: {len(colors)}")

packed = np.asarray(fb, dtype=np.uint32).reshape(144, 160)
arr = np.empty((144, 160, 3), np.uint8)
arr[:, :, 0] = (packed >> 16) & 0xFF
arr[:, :, 1] = (packed >> 8) & 0xFF
arr[:, :, 2] = packed & 0xFF
imageio.imwrite('zelda.png', arr)
print("Saved frame to zelda.png")
