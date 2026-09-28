"""Portable regression tests for bugs found in the audit pass.

Covers CPU HALT/interrupt dispatch, boot ROM execution, CGB GDMA/HDMA,
MBC1/2/3/5 register decoding and RAM mirroring, RTC drift, save states for
4 MB+ carts, the CLI direct-boot pygame init, atomic file writes, and
ROM-browser robustness. No display and no local ROM files required.

Exits non-zero on any failure.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    path = os.path.join(HERE, "gbc_emulator.py")
    src = open(path, encoding="utf-8").read().split("if __name__")[0]
    ns = {}
    exec(compile(src, "gbc_emulator.py", "exec"), ns)
    # Keep test runs from overwriting the developer's real settings.
    ns["_CONFIG_PATH"] = os.path.join(tempfile.mkdtemp(), "gbc_config.json")
    return ns


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def build_rom(cart=0x00, cgb=False, size=0x8000, ram_code=0x00):
    rom = bytearray(size)
    rom[0x0143] = 0x80 if cgb else 0x00
    rom[0x0147] = cart
    code, banks = 0, 2
    while banks < size // 0x4000:
        banks <<= 1
        code += 1
    rom[0x0148] = code
    rom[0x0149] = ram_code
    rom[0x0100:0x0104] = bytes([0x00, 0x18, 0xFD, 0x00])  # NOP / JR -3
    return rom


def write_rom(rom, suffix=".gb"):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "test" + suffix)
    with open(path, "wb") as f:
        f.write(bytes(rom))
    return path


def test_halt_with_pending_irq(ns):
    """EI; HALT with an interrupt already pending must run the ISR now."""
    MMU, CPU = ns["MMU"], ns["CPU"]
    rom = build_rom()
    rom[0x0040] = 0xD9          # RETI at the VBlank vector
    rom[0x0100:0x0103] = bytes([0xFB, 0x76, 0x00])  # EI; HALT; NOP
    m = MMU(); m.load_rom(bytes(rom))
    c = CPU(m)
    m.memory[0xFFFF] = 0x01
    m.memory[0xFF0F] = 0x01
    c.step()   # EI
    c.step()   # HALT -> dispatch
    check("interrupt dispatched to 0x0040", c.reg.pc == 0x0040)
    check("CPU is not left halted after dispatch", not c.halted)
    c.step()   # RETI
    check("RETI returns past HALT", c.reg.pc == 0x0102)
    c.set_branch_trace(True)
    c._record_branch("JR", 0x100, 0x200, 0x18)
    c.dump_branch_trace()  # used to raise TypeError (slicing a deque)
    check("dump_branch_trace works", True)


def test_boot_rom_executes(ns):
    GameBoy = ns["GameBoy"]
    rom = build_rom()
    rom[0x0000] = 0x11          # distinctive cart byte under the boot ROM
    rom[0x0104] = 0x22          # header byte visible through a CGB boot ROM
    rom_path = write_rom(rom)
    boot = bytearray(0x900)
    # LD A,0x42 ; LD B,A ; JP 0x00FC ... 0x00FC: LD A,1 ; LDH (50),A -> 0x0100
    boot[0:6] = bytes([0x3E, 0x42, 0x47, 0xC3, 0xFC, 0x00])
    boot[0xFC:0x100] = bytes([0x3E, 0x01, 0xE0, 0x50])
    boot[0x104] = 0x99          # must NOT shadow the cartridge header
    boot[0x200] = 0x77
    boot_path = os.path.join(os.path.dirname(rom_path), "boot.bin")
    with open(boot_path, "wb") as f:
        f.write(boot)
    gb = GameBoy(rom_path, bootrom_path=boot_path, audio_enabled=False)
    mmu = gb.mmu
    check("CGB boot ROM leaves the cart header visible", mmu.read_byte(0x0104) == 0x22)
    check("CGB boot ROM maps 0x200-0x8FF", mmu.read_byte(0x0200) == 0x77)
    for _ in range(8):
        gb.step_all()
    check("boot ROM code actually ran", gb.cpu.reg.b == 0x42)
    check("boot ROM unmapped via FF50", not mmu.bootrom_enabled)
    check("execution reached the cartridge entry point", 0x0100 <= gb.cpu.reg.pc <= 0x0103)
    check("cart bytes restored after unmap", mmu.memory[0x0000] == 0x11
          and mmu.read_byte(0x0000) == 0x11)


def test_gdma_hdma(ns):
    MMU = ns["MMU"]
    m = MMU(); m.load_rom(bytes(build_rom(cgb=True)))
    for i in range(0x800):
        m.memory[0xC000 + i] = (i & 0xFF) or 1

    def regs(src, dst):
        m.write_byte(0xFF51, src >> 8); m.write_byte(0xFF52, src & 0xFF)
        m.write_byte(0xFF53, (dst >> 8) & 0x1F); m.write_byte(0xFF54, dst & 0xFF)

    def vram_used():
        return sum(1 for i in range(0x1000) if m.memory[0x8000 + i])

    regs(0xC000, 0x8000)
    m.write_byte(0xFF55, 0x03)           # GDMA, 4 blocks
    check("first GDMA copies the requested 64 bytes", vram_used() == 64)
    check("GDMA completion reads 0xFF", m.memory[0xFF55] == 0xFF)
    m.memory[0x8000:0x9000] = bytes(0x1000)
    regs(0xC000, 0x8000)
    m.write_byte(0xFF55, 0x01)           # GDMA, 2 blocks
    check("later GDMA copies 32 bytes (not 2048)", vram_used() == 32)

    m.memory[0x8000:0x9000] = bytes(0x1000)
    regs(0xC000, 0x8000)
    m.write_byte(0xFF55, 0x83)           # H-Blank DMA, 4 blocks
    m._hdma_hblank_step()
    check("HDMA FF55 reads blocks-left minus one", m.memory[0xFF55] == 0x02)
    m.write_byte(0xFF55, 0x00)           # bit 7 clear -> stop, not GDMA
    check("HDMA stop halts the transfer", not m.hdma_active)
    check("HDMA stop does not start a GDMA", vram_used() == 16)
    check("stopped HDMA reads remaining with bit 7", m.memory[0xFF55] == 0x82)

    dmg = MMU(); dmg.load_rom(bytes(build_rom(cgb=False)))
    dmg.write_byte(0xFF55, 0x03)
    check("DMG ignores FF55", dmg.gdma_stall == 0 and not dmg.hdma_active)


def test_mbc_registers(ns):
    MMU = ns["MMU"]
    # MBC1, 1 MB: BANK2 applies to 4000-7FFF in mode 1 too.
    rom = build_rom(cart=0x01, size=0x100000)
    for bank in range(64):
        rom[bank * 0x4000 + 0x10] = bank
    m = MMU(); m.load_rom(bytes(rom))
    m.write_byte(0x3FFF, 0x05)           # bank register spans 2000-3FFF
    check("MBC1 accepts bank writes at 0x3FFF", m.read_byte(0x4010) == 5)
    m.write_byte(0x6000, 0x01)           # mode 1
    m.write_byte(0x4000, 0x01)           # BANK2 = 1
    check("MBC1 mode 1 keeps BANK2 for 4000-7FFF", m.read_byte(0x4010) == 0x25)
    check("MBC1 mode 1 low window fetches bank 0x20", m.memory[0x0010] == 0x20)
    m.write_byte(0x6000, 0x00)
    check("MBC1 mode 0 restores bank 0 at 0000", m.memory[0x0010] == 0)

    # MBC3: 3000-3FFF selects the ROM bank as well.
    rom = build_rom(cart=0x13, size=0x40000, ram_code=0x02)
    for bank in range(16):
        rom[bank * 0x4000 + 0x10] = bank
    m = MMU(); m.load_rom(bytes(rom))
    m.write_byte(0x3000, 0x07)
    check("MBC3 accepts bank writes at 0x3000", m.read_byte(0x4010) == 7)
    # 8 KB RAM: banks 1-3 mirror bank 0 instead of reading 0xFF.
    m.write_byte(0x0000, 0x0A)
    m.write_byte(0xA000, 0x5A)
    m.write_byte(0x4000, 0x02)
    check("MBC3 small RAM mirrors across banks", m.read_byte(0xA000) == 0x5A)

    # MBC5 rumble: bit 3 drives the motor, not the RAM bank.
    m = MMU(); m.load_rom(bytes(build_rom(cart=0x1E, size=0x40000, ram_code=0x03)))
    m.write_byte(0x0000, 0x0A)
    m.write_byte(0x4000, 0x01)
    m.write_byte(0xA000, 0x33)
    m.write_byte(0x4000, 0x09)           # bank 1 + rumble on
    check("MBC5 rumble bit keeps RAM bank 1", m.read_byte(0xA000) == 0x33)

    # MBC2 registers live only in 0000-3FFF.
    rom = build_rom(cart=0x05, size=0x40000)
    for bank in range(16):
        rom[bank * 0x4000 + 0x10] = bank
    m = MMU(); m.load_rom(bytes(rom))
    m.write_byte(0x2100, 0x03)
    m.write_byte(0x4100, 0x09)
    check("MBC2 ignores writes above 0x3FFF", m.read_byte(0x4010) == 3)


def test_rtc_no_drift(ns):
    MMU = ns["MMU"]
    m = MMU(); m.load_rom(bytes(build_rom(cart=0x10, ram_code=0x03)))
    start = time.time() - 10.75
    m.rtc_last_time = start
    m._rtc_update()
    check("RTC advanced 10 whole seconds", m.rtc_s == 10)
    check("RTC keeps the fractional second", abs(m.rtc_last_time - (start + 10)) < 1e-6)


def test_big_rom_save_state(ns):
    GameBoy = ns["GameBoy"]
    rom = build_rom(cart=0x19, size=0x400000)   # 4 MB MBC5 = 256 banks
    path = write_rom(rom)
    gb = GameBoy(path, audio_enabled=False)
    check("4 MB ROM has 256 banks", gb.mmu.num_rom_banks == 256)
    gb.mmu.write_byte(0x2000, 0xF0)
    ok = gb.save_state(0)
    check("save state works for 256-bank ROMs", ok)
    gb.mmu.write_byte(0x2000, 0x01)
    check("load state works for 256-bank ROMs", gb.load_state(0))
    check("bank count survives the load", gb.mmu.num_rom_banks == 256)
    check("ROM bank restored", gb.mmu.rom_bank == 0xF0)


def test_direct_boot_init(ns):
    """The CLI direct-boot path must initialise pygame (fonts, joysticks)."""
    pygame = ns["pygame"]
    ns["_save_config"]({"palette": 1, "volume": 0.5})
    ns["_init_pygame"]()
    check("pygame initialised for direct boot", pygame.get_init() and pygame.font.get_init())
    settings = ns["_game_settings_from_config"]()
    check("saved palette is applied on direct boot", settings["palette"] == ns["PALETTE_GRAY"])
    check("saved volume is applied on direct boot", settings["volume"] == 0.5)
    gb = ns["GameBoy"](write_rom(build_rom()), **settings)
    backdrop = gb._capture_pause_backdrop()
    gb._render_pause_page("pause", backdrop)   # used to raise "font not initialized"
    check("pause menu renders after direct boot", True)
    check("window close is not flagged by default", gb.quit_requested is False)


def test_atomic_write_and_sav(ns):
    d = tempfile.mkdtemp()
    target = os.path.join(d, "x.sav")
    ns["_atomic_write"](target, [b"abc", b"def"])
    check("atomic write stores all chunks", open(target, "rb").read() == b"abcdef")
    check("atomic write leaves no temp file", not os.path.exists(target + ".tmp"))

    class Boom(Exception):
        pass

    rom = build_rom(cart=0x03, ram_code=0x02)   # MBC1+RAM+BATT
    gb = ns["GameBoy"](write_rom(rom), audio_enabled=False)
    gb.mmu.ram_data[0] = 0xAB

    def explode():
        raise Boom()
    gb._run_frame = explode
    try:
        gb.run()
    except Boom:
        pass
    sav = gb._sav_path()
    check("battery save written even when emulation raises",
          os.path.isfile(sav) and open(sav, "rb").read(1) == b"\xAB")


def test_menu_robustness(ns):
    d = tempfile.mkdtemp()
    short = os.path.join(d, "short.gb")
    with open(short, "wb") as f:
        f.write(b"\x00" * 16)
    check("truncated ROM header parses to None", ns["_parse_rom_header"](short) is None)
    check("truncated ROM tag is ???", ns["_read_rom_system_tag"](short) == "???")

    menu = ns["EmulatorMenu"].__new__(ns["EmulatorMenu"])
    menu.rom_cursor = 50
    cwd = os.getcwd()
    try:
        os.chdir(d)
        menu._scan_roms()
    finally:
        os.chdir(cwd)
    check("rescan clamps the ROM cursor", menu.rom_cursor < max(len(menu.roms), 1))


def test_div_timer_phase(ns):
    MMU, Timers = ns["MMU"], ns["Timers"]
    m = MMU(); m.load_rom(bytes(build_rom()))
    t = Timers(m)
    m.div_reset_callback = t.reset_div
    m.write_byte(0xFF07, 0x05)           # timer on, 16-cycle rate
    m.memory[0xFF05] = 0
    t.step(12)                           # DIV bit 3 is now high
    m.write_byte(0xFF04, 0)              # falling edge -> one TIMA tick
    check("DIV write with the rate bit high ticks TIMA", m.memory[0xFF05] == 1)
    t.step(15)
    check("TIMA phase restarts from the DIV reset", m.memory[0xFF05] == 1)
    t.step(1)
    check("next TIMA tick after a full period", m.memory[0xFF05] == 2)

    gb = ns["GameBoy"](write_rom(build_rom()), audio_enabled=False)
    gb.timers.div_counter = 0xFFF0
    for _ in range(100):
        gb.step_all()
    check("DIV counter wraps at 16 bits", gb.timers.div_counter < 0x10000)
    check("save state works after DIV wraps (~17 min of play)", gb.save_state(0))


def test_dmg_length_survives_power_off(ns):
    MMU, APU = ns["MMU"], ns["APU"]
    for cgb, want in ((False, 20), (True, 0)):
        m = MMU(); m.load_rom(bytes(build_rom(cgb=cgb)))
        apu = APU(m, is_cgb=cgb)
        apu.write_register(0xFF11, 64 - 20)
        apu.write_register(0xFF26, 0x00)
        label = "CGB clears" if cgb else "DMG keeps"
        check(f"{label} length counters on APU power-off", apu.ch1_length == want)


def test_sprite_bg_priority_masking(ns):
    """A top sprite hidden behind BG must not reveal a lower sprite beneath it."""
    MMU, PPU = ns["MMU"], ns["PPU"]
    m = MMU(); m.load_rom(bytes(build_rom()))
    p = PPU(m); m.ppu = p
    mem = m.memory
    mem[0x8010:0x8020] = bytes([0xFF, 0x00] * 8)   # tile 1: colour 1 everywhere
    mem[0x8020:0x8030] = bytes([0x00, 0xFF] * 8)   # tile 2: colour 2 everywhere
    mem[0x9800] = 1                                 # BG tile 0 is non-zero colour
    mem[0xFF47] = 0xE4
    mem[0xFF48] = 0xE4
    mem[0xFF40] = 0x93                              # LCD, BG, OBJ, unsigned tiles
    # OAM 0 (higher priority, same X): BG-priority flag set; OAM 1 lower priority.
    mem[0xFE00:0xFE08] = bytes([16, 8, 1, 0x80, 16, 8, 2, 0x00])
    p._render_scanline(0, mem[0xFF40])
    bg_colour = p._dmg_bgp_rgb[0xE4][1]
    check("BG shows where the top sprite yields to it",
          p.framebuffer[0] == bg_colour)


def main():
    ns = load_module()
    print("halt + pending irq:");     test_halt_with_pending_irq(ns)
    print("boot rom:");               test_boot_rom_executes(ns)
    print("gdma / hdma:");            test_gdma_hdma(ns)
    print("mbc registers:");          test_mbc_registers(ns)
    print("rtc drift:");              test_rtc_no_drift(ns)
    print("4 MB save state:");        test_big_rom_save_state(ns)
    print("direct boot init:");       test_direct_boot_init(ns)
    print("atomic write / .sav:");    test_atomic_write_and_sav(ns)
    print("menu robustness:");        test_menu_robustness(ns)
    print("div / timer phase:");      test_div_timer_phase(ns)
    print("dmg apu power-off:");      test_dmg_length_survives_power_off(ns)
    print("sprite bg priority:");     test_sprite_bg_priority_masking(ns)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
