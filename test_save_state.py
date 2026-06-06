import sys
import time
import os

sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')
src = open(r'C:\Users\allen\Downloads\GBC\gbc_emulator_skeleton.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(compile(src, 'gbc_emulator_skeleton.py', 'exec'))

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

# Build a GameBoy instance
gb = GameBoy.__new__(GameBoy)
gb.mmu = mmu
gb.cpu = cpu
gb.ppu = ppu
gb.apu = apu
gb.timers = timers
gb._prev_double_speed = False
gb.fps_limit = 60
gb.smooth_scale = False
gb.shader = _shader_none
gb._prev_shader_frame = None
gb.ppu.set_palette(PALETTE_DMG)
gb._status_msg = ''
gb._status_ttl = 0
gb.window_scale = 2
gb._audio_refs = []
gb.audio_enabled = False
gb.audio_channel = None
gb._volume = 1.0
gb.running = True

# Run for a while
print('Running 2M steps...')
for i in range(2000000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)

print(f'After 2M: PC=0x{cpu.reg.pc:04X}, LY={mmu.memory[0xFF44]:3d}, A=0x{cpu.reg.a:02X}, SP=0x{cpu.reg.sp:04X}')

# Snapshot the state
saved = {
    'pc': cpu.reg.pc, 'sp': cpu.reg.sp, 'a': cpu.reg.a, 'f': cpu.reg.f,
    'b': cpu.reg.b, 'c': cpu.reg.c, 'd': cpu.reg.d, 'e': cpu.reg.e,
    'h': cpu.reg.h, 'l': cpu.reg.l,
    'ly': mmu.memory[0xFF44], 'ime': cpu.interrupts_master_enabled,
    'halted': cpu.halted,
    'mem_pc_byte': mmu.memory[cpu.reg.pc],
    'mem_bc_byte': mmu.memory[cpu.reg.bc],
}
print(f'Snapshot: {saved}')

# Test save/load
import os.path
mmu.rom_path = rom_path
print('Saving state...')
ok = gb.save_state(0)
print(f'  Result: {ok}')

# Modify state to verify load works
cpu.reg.a = 0xFF
mmu.memory[0xFF44] = 0
cpu.reg.pc = 0x1234
print(f'After modification: PC=0x{cpu.reg.pc:04X}, A=0x{cpu.reg.a:02X}, LY={mmu.memory[0xFF44]}')

print('Loading state...')
ok = gb.load_state(0)
print(f'  Result: {ok}')

print(f'After load: PC=0x{cpu.reg.pc:04X}, A=0x{cpu.reg.a:02X}, LY={mmu.memory[0xFF44]}')
print(f'Expected:   PC=0x{saved["pc"]:04X}, A=0x{saved["a"]:02X}, LY={saved["ly"]}')

# Verify
ok = (cpu.reg.pc == saved['pc'] and cpu.reg.a == saved['a']
      and mmu.memory[0xFF44] == saved['ly'])
print(f'Match: {ok}')

# Continue running to verify it works
for i in range(100000):
    c = cpu.step()
    ppu.step(c)
    timers.step(c)
    apu.step(c)
print(f'After more steps: PC=0x{cpu.reg.pc:04X}, LY={mmu.memory[0xFF44]:3d}')
