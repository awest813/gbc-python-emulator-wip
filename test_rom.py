import sys
sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')
exec(open(r'C:\Users\allen\Downloads\GBC\gbc_emulator_skeleton.py', encoding='utf-8').read().split('if __name__')[0])

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

total_cycles = 0
for i in range(50000000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)
    total_cycles += c
    if i % 10000000 == 0:
        fb = ppu.framebuffer
        colors = set()
        for idx in range(0, 160*144, 17):
            colors.add(fb[idx])
        print('Step %d, PC=%04X, colors=%d' % (i, cpu.reg.pc, len(colors)))

print('Total cycles:', total_cycles)
print('PC:', hex(cpu.reg.pc))
print('LY:', mmu.memory[0xFF44], 'LCDC: 0x%02X' % mmu.memory[0xFF40])

fb = ppu.framebuffer
colors = {}
for y in range(SCREEN_HEIGHT):
    for x in range(SCREEN_WIDTH):
        idx = y * SCREEN_WIDTH + x
        px = fb[idx]
        key = ((px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)
        colors[key] = colors.get(key, 0) + 1

print('Unique colors:', len(colors))
top5 = sorted(colors.items(), key=lambda x: -x[1])[:5]
for c, count in top5:
    print('  %3d %3d %3d : %5d pixels' % (c[0], c[1], c[2], count))
