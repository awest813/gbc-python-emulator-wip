import sys
import cProfile
import pstats
sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')

# Load the emulator code
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

# Warm up: run a bit so JIT-like effects and caches are warm
for i in range(100000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

# Profile
profiler = cProfile.Profile()
profiler.enable()
for i in range(2000000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)
profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(30)
print('---')
stats.sort_stats('tottime')
stats.print_stats(20)
