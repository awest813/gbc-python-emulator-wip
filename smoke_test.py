import sys
import time

sys.path.insert(0, r'C:\Users\allen\Downloads\GBC')
src = open(r'C:\Users\allen\Downloads\GBC\gbc_emulator_skeleton.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(compile(src, 'gbc_emulator_skeleton.py', 'exec'))

import os
for rom_name in ['game.gb', 'Dragon Warrior III (USA).gbc']:
    rom_path = os.path.join('roms', rom_name)
    print(f'=== {rom_name} ===')
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
    # Post-boot CGB register state
    cpu.reg.a = 0x11
    cpu.reg.f = 0x80
    cpu.reg.d = 0xFF
    cpu.reg.e = 0x56
    cpu.reg.l = 0x0D
    cpu.reg.sp = 0xFFFE
    cpu.reg.pc = 0x0100

    start = time.time()
    invalid_opcodes = 0
    invalid_log = []
    for i in range(2000000):
        try:
            c = cpu.step()
        except Exception as e:
            print(f'CPU exception at i={i}: {e}')
            break
        if i > 100000:  # Skip initial warmup
            pass
        # Check for unhandled invalid opcodes
        if cpu.invalid_opcode_count > invalid_opcodes:
            invalid_opcodes = cpu.invalid_opcode_count
        ppu.step(c)
        timers.step(c)
        apu.step(c)
        if i % 500000 == 0:
            print(f'  i={i}, PC=0x{cpu.reg.pc:04X}, LY={mmu.memory[0xFF44]:3d}, LCDC=0x{mmu.memory[0xFF40]:02X}, IME={cpu.interrupts_master_enabled}')

    elapsed = time.time() - start
    print(f'  Total time: {elapsed:.2f}s')
    print(f'  Final PC: 0x{cpu.reg.pc:04X}')
    print(f'  Final LY: {mmu.memory[0xFF44]}, LCDC: 0x{mmu.memory[0xFF40]:02X}')
    print(f'  IME: {cpu.interrupts_master_enabled}, IE: 0x{mmu.memory[0xFFFF]:02X}, IF: 0x{mmu.memory[0xFF0F]:02X}')
    # Count unique framebuffer colors
    fb = ppu.framebuffer
    colors = set()
    for idx in range(160*144):
        colors.add(fb[idx])
    print(f'  Unique framebuffer colors: {len(colors)}')
    print()
