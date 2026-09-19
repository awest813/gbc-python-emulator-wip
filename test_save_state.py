"""Portable save-state round-trip test (synthetic ROM, no display, no local files).

Builds a tiny CGB ROM in memory, runs it, snapshots the emulator with
``save_state``, mutates live state, reloads with ``load_state`` and verifies the
machine is restored exactly.  Also checks that the derived CGB colour tables
(``_bg_rgb`` / ``_obj_rgb``) are rebuilt from the saved palette bytes after a
load, which guards the packed-24-bit framebuffer path.

Exits non-zero on any failure.
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

src = open(os.path.join(HERE, "gbc_emulator.py"), encoding="utf-8").read()
src = src.split("if __name__")[0]
_ns = {}
exec(compile(src, "gbc_emulator.py", "exec"), _ns)
MMU, CPU, PPU, APU, Timers = (_ns["MMU"], _ns["CPU"], _ns["PPU"], _ns["APU"], _ns["Timers"])
GameBoy = _ns["GameBoy"]
PALETTE_DMG = _ns["PALETTE_DMG"]
_shader_none = _ns["_shader_none"]
_JOY_SRC_KB = _ns["_JOY_SRC_KB"]
_JOY_SRC_AXIS = _ns["_JOY_SRC_AXIS"]

_failures = 0


def check(name, ok):
    global _failures
    print(f"  {'ok' if ok else 'FAIL'}: {name}")
    if not ok:
        _failures += 1


def build_machine(rom_path, cart=0x00):
    """Wire a headless GameBoy around a freshly loaded synthetic CGB ROM."""
    rom = bytearray(0x8000)
    rom[0x0143] = 0x80  # CGB compatible
    rom[0x0147] = cart
    # Enable the LCD, set a BG palette colour, then spin in a tight JR loop so
    # the PPU/APU keep advancing while we accumulate some non-trivial state.
    prog = [0x3E, 0x91, 0xE0, 0x40, 0x00, 0x18, 0xFD]
    rom[0x0100:0x0100 + len(prog)] = prog
    with open(rom_path, "wb") as f:
        f.write(bytes(rom))

    mmu = MMU()
    mmu.load_rom(bytes(rom))
    mmu.rom_path = rom_path
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

    gb = GameBoy.__new__(GameBoy)
    gb.mmu, gb.cpu, gb.ppu, gb.apu, gb.timers = mmu, cpu, ppu, apu, timers
    gb.fps_limit = 60
    gb.smooth_scale = False
    gb.shader = _shader_none
    gb._prev_shader_frame = None
    gb._status_msg = ""
    gb._status_ttl = 0
    gb.window_scale = 2
    gb._audio_refs = []
    gb.audio_enabled = False
    gb.audio_channel = None
    gb._volume = 1.0
    gb.running = True
    return gb


def run_steps(gb, n):
    cpu, ppu, timers, apu = gb.cpu, gb.ppu, gb.timers, gb.apu
    for _ in range(n):
        c = cpu.step()
        ppu.step(c)
        timers.step(c)
        apu.step(c)


def main():
    print("save-state round trip:")
    with tempfile.TemporaryDirectory() as tmp:
        rom_path = os.path.join(tmp, "synthetic.gbc")
        gb = build_machine(rom_path)

        # Give a CGB BG palette colour a distinctive value so we can confirm the
        # derived colour table is rebuilt on load.
        gb.ppu.write_cgb_register(0xFF68, 0x01)      # BG palette index 1, no auto-inc
        gb.ppu.write_cgb_register(0xFF69, 0x7C)      # high byte of colour 0 -> blue-ish
        gb.ppu.write_cgb_register(0xFF68, 0x00)
        gb.ppu.write_cgb_register(0xFF69, 0x1F)      # low byte -> red bits set

        run_steps(gb, 500_000)

        snapshot = {
            "pc": gb.cpu.reg.pc, "sp": gb.cpu.reg.sp, "a": gb.cpu.reg.a,
            "f": gb.cpu.reg.f, "b": gb.cpu.reg.b, "c": gb.cpu.reg.c,
            "d": gb.cpu.reg.d, "e": gb.cpu.reg.e, "h": gb.cpu.reg.h,
            "l": gb.cpu.reg.l, "ly": gb.mmu.memory[0xFF44],
            "div": gb.timers.div_counter,
            "bg_rgb0": gb.ppu._bg_rgb[0],
            "bg_pal0": bytes(gb.ppu.bg_palette_data[0:2]),
        }

        check("save_state succeeds", gb.save_state(0) is True)
        check("state file written", os.path.isfile(gb._state_path(0)))

        # Corrupt live state, including the derived colour table, so a successful
        # load has to actively restore everything.
        gb.cpu.reg.a = 0xFF
        gb.cpu.reg.pc = 0x1234
        gb.mmu.memory[0xFF44] = 0
        gb.ppu.bg_palette_data[0:2] = b"\x00\x00"
        gb.ppu._bg_rgb[0] = 0xDEAD

        check("load_state succeeds", gb.load_state(0) is True)

        check("PC restored", gb.cpu.reg.pc == snapshot["pc"])
        check("A restored", gb.cpu.reg.a == snapshot["a"])
        check("SP restored", gb.cpu.reg.sp == snapshot["sp"])
        check("LY restored", gb.mmu.memory[0xFF44] == snapshot["ly"])
        check("DIV counter restored", gb.timers.div_counter == snapshot["div"])
        check("BG palette bytes restored",
              bytes(gb.ppu.bg_palette_data[0:2]) == snapshot["bg_pal0"])
        check("derived CGB BG colour rebuilt (packed int)",
              gb.ppu._bg_rgb[0] == snapshot["bg_rgb0"])
        check("derived CGB colour is a packed int",
              isinstance(gb.ppu._bg_rgb[0], int) and 0 <= gb.ppu._bg_rgb[0] <= 0xFFFFFF)

        # Machine must keep running cleanly after the load.
        run_steps(gb, 50_000)
        check("runs after load (LY in range)", 0 <= gb.mmu.memory[0xFF44] <= 153)

        # Loading a missing slot must fail gracefully, not raise.
        check("missing slot load returns False", gb.load_state(3) is False)

        # Cartridge SRAM must round-trip (battery games keep party/inventory).
        gb.mmu.ram_data = bytearray(0x2000)
        gb.mmu.has_ram = True
        gb.mmu.ram_data[0] = 0x42
        gb.mmu.ram_data[0x1FFF] = 0x99
        check("save_state with SRAM succeeds", gb.save_state(0) is True)
        gb.mmu.ram_data[0] = 0x00
        gb.mmu.ram_data[0x1FFF] = 0x00
        check("load_state restores SRAM", gb.load_state(0) is True)
        check("SRAM byte 0 restored", gb.mmu.ram_data[0] == 0x42)
        check("SRAM last byte restored", gb.mmu.ram_data[0x1FFF] == 0x99)

        # HDMA destination is a full address; saving mid-transfer must not crash.
        gb.mmu.hdma_src = 0xC000
        gb.mmu.hdma_dst = 0x8000
        gb.mmu.hdma_remaining = 32
        gb.mmu.hdma_active = True
        dma_left = 40
        gb.mmu.dma_remaining = dma_left
        gb.mmu.dma_index = 10
        check("save_state during HDMA succeeds", gb.save_state(1) is True)
        check("save_state does not cancel in-flight OAM DMA",
              gb.mmu.dma_remaining == dma_left)
        gb.mmu.hdma_dst = 0
        gb.mmu.hdma_active = False
        gb.mmu.dma_remaining = 0
        check("load_state restores HDMA destination", gb.load_state(1) is True)
        check("HDMA dst restored", gb.mmu.hdma_dst == 0x8000)
        check("HDMA remaining restored", gb.mmu.hdma_remaining == 32)
        check("OAM DMA remaining restored", gb.mmu.dma_remaining == dma_left)

        # A truncated / corrupt file must fail without shrinking live memory.
        corrupt = gb._state_path(0)
        with open(corrupt, "wb") as f:
            f.write(b"GBST" + bytes([3, 0, 0, 0]) + b"\x00" * 200)
        mem_len = len(gb.mmu.memory)
        pc_before = gb.cpu.reg.pc
        gb.mmu.rom_bank = 99
        gb.ppu.mode = 5
        rom_bank_before = 99
        ppu_mode_before = 5
        check("truncated load returns False", gb.load_state(0) is False)
        check("truncated load leaves 64KB memory", len(gb.mmu.memory) == mem_len == 0x10000)
        check("truncated load rolls back CPU state", gb.cpu.reg.pc == pc_before)
        check("truncated load rolls back MMU banking", gb.mmu.rom_bank == rom_bank_before)
        check("truncated load rolls back PPU mode", gb.ppu.mode == ppu_mode_before)
        check("truncated load reports corrupt file", gb._last_state_error == 'corrupt')

        # v6 saves record APU timing, gdma stall, joypad sources, and ROM identity.
        apu_snap = (gb.apu.fs_div, gb.apu._fs_remain, gb.apu.frame_seq_step)
        gb.mmu.gdma_stall = 500
        gb.speed_remainder = 1
        gb.mmu.set_joypad_button(1, True, _JOY_SRC_KB)   # Left
        gb.mmu.set_joypad_button(2, True, _JOY_SRC_AXIS)  # Up
        joy_src_snap = list(gb.mmu._joy_src)
        check("save_state v6 succeeds", gb.save_state(0) is True)
        gb.apu.fs_div = 0
        gb.apu._fs_remain = 0
        gb.apu.frame_seq_step = 0
        gb.mmu.gdma_stall = 0
        gb.speed_remainder = 0
        gb.mmu.release_all_joypad()
        check("load_state restores APU timing", gb.load_state(0) is True)
        check("APU fs_div restored", gb.apu.fs_div == apu_snap[0])
        check("APU fs_remain restored", gb.apu._fs_remain == apu_snap[1])
        check("APU frame_seq restored", gb.apu.frame_seq_step == apu_snap[2])
        check("GDMA stall restored", gb.mmu.gdma_stall == 500)
        check("speed remainder restored", gb.speed_remainder == 1)
        check("joypad sources restored", gb.mmu._joy_src == joy_src_snap)
        check("Left still held after load", (gb.mmu.joypad_buttons & 0x02) == 0)
        check("Up still held after load", (gb.mmu.joypad_buttons & 0x04) == 0)
        check("scanline sprite cache cleared", gb.ppu._scanline_sprites is None)

        # bootrom_enabled round-trips separately from the 64KB memory blob.
        gb.mmu.bootrom = bytearray([0xBE] * 256)
        gb.mmu.bootrom_enabled = False
        check("bootrom flag save succeeds", gb.save_state(0) is True)
        gb.mmu.bootrom_enabled = True
        check("bootrom flag load succeeds", gb.load_state(0) is True)
        check("bootrom disabled after load", gb.mmu.bootrom_enabled is False)

        # Mid-transfer serial is cleared when no link partner is connected.
        gb.mmu.serial_data = 0xAB
        gb.mmu.serial_control = 0x81
        gb.mmu.serial_bits_left = 7
        check("mid-serial save succeeds", gb.save_state(0) is True)
        gb.mmu.serial_bits_left = 0
        check("mid-serial load succeeds", gb.load_state(0) is True)
        check("mid-serial shift cleared on load", gb.mmu.serial_bits_left == 0)
        check("serial SC cleared on load", (gb.mmu.serial_control & 0x80) == 0)

        # v6 saves record the ROM basename; loading a mismatched snapshot fails.
        other_path = os.path.join(tmp, "other.gbc")
        shutil.copy(rom_path, other_path)
        shutil.copy(gb._state_path(0), os.path.splitext(other_path)[0] + ".ss0")
        gb.mmu.rom_path = other_path
        check("wrong ROM load returns False", gb.load_state(0) is False)
        check("wrong ROM reports mismatch", gb._last_state_error == 'wrong_rom')
        gb.mmu.rom_path = rom_path
        check("matching ROM load succeeds", gb.load_state(0) is True)

        # v4 full-length guard rejects basenames that share a prefix but differ in length.
        check("save_state for prefix guard", gb.save_state(0) is True)
        longer = os.path.join(tmp, "synthetic.gbc.backup")
        shutil.copy(rom_path, longer)
        shutil.copy(gb._state_path(0), os.path.splitext(longer)[0] + ".ss0")
        gb.mmu.rom_path = longer
        check("same-prefix longer basename rejected", gb.load_state(0) is False)
        check("prefix length mismatch reports wrong_rom", gb._last_state_error == 'wrong_rom')
        gb.mmu.rom_path = rom_path

        # v3 saves without a ROM id block still load on v6 builds.
        legacy = gb._state_path(1)
        raw = open(gb._state_path(0), "rb").read()
        with open(legacy, "wb") as f:
            f.write(raw[:4] + bytes([3, 1, 0, 0]) + raw[40:])
        gb.mmu.gdma_stall = 500
        gb.speed_remainder = 1
        check("legacy v3 save loads", gb.load_state(1) is True)
        check("legacy v3 load clears gdma stall", gb.mmu.gdma_stall == 0)
        check("legacy v3 load clears speed remainder", gb.speed_remainder == 0)

        # MBC3 RTC fields round-trip through save states.
        rtc_path = os.path.join(tmp, "rtc.gbc")
        gb_rtc = build_machine(rtc_path, cart=0x0F)
        gb_rtc._has_rtc = True
        gb_rtc.mmu.rtc_s = 25
        gb_rtc.mmu.rtc_m = 11
        gb_rtc.mmu.rtc_h = 3
        gb_rtc.mmu.rtc_dl = 4
        check("MBC3 RTC save succeeds", gb_rtc.save_state(0) is True)
        gb_rtc.mmu.rtc_s = 0
        gb_rtc.mmu.rtc_m = 0
        check("MBC3 RTC load succeeds", gb_rtc.load_state(0) is True)
        check("MBC3 RTC seconds restored", gb_rtc.mmu.rtc_s == 25)
        check("MBC3 RTC minutes restored", gb_rtc.mmu.rtc_m == 11)

        # MBC6 flash round-trips through save states.
        mbc6_path = os.path.join(tmp, "mbc6.gbc")
        gb6 = build_machine(mbc6_path, cart=0x20)
        if len(gb6.mmu.flash_data) < 0x2000:
            gb6.mmu.flash_data = bytearray(0x20000)
        gb6.mmu.flash_data[0x1234] = 0xBE
        check("MBC6 flash save succeeds", gb6.save_state(0) is True)
        gb6.mmu.flash_data[0x1234] = 0
        check("MBC6 flash load succeeds", gb6.load_state(0) is True)
        check("MBC6 flash byte restored", gb6.mmu.flash_data[0x1234] == 0xBE)

        # v7 saves mid-packet Super Game Boy FSM state.
        sgb_snap = (
            True, 42, bytes(range(16)), 0x0A, 3, bytearray([0x01, 0x02, 0x03, 0x04]),
        )
        gb.mmu.sgb_in_packet = sgb_snap[0]
        gb.mmu.sgb_bit_count = sgb_snap[1]
        gb.mmu.sgb_packet[:] = sgb_snap[2]
        gb.mmu.sgb_cmd = sgb_snap[3]
        gb.mmu.sgb_packets_left = sgb_snap[4]
        gb.mmu.sgb_cmd_data = bytearray(sgb_snap[5])
        check("SGB FSM save succeeds", gb.save_state(0) is True)
        gb.mmu.sgb_in_packet = False
        gb.mmu.sgb_bit_count = 0
        gb.mmu.sgb_packet[:] = bytes(16)
        gb.mmu.sgb_cmd = 0
        gb.mmu.sgb_packets_left = 0
        gb.mmu.sgb_cmd_data = bytearray()
        check("SGB FSM load succeeds", gb.load_state(0) is True)
        check("sgb_in_packet restored", gb.mmu.sgb_in_packet == sgb_snap[0])
        check("sgb_bit_count restored", gb.mmu.sgb_bit_count == sgb_snap[1])
        check("sgb_packet restored", bytes(gb.mmu.sgb_packet) == sgb_snap[2])
        check("sgb_cmd restored", gb.mmu.sgb_cmd == sgb_snap[3])
        check("sgb_packets_left restored", gb.mmu.sgb_packets_left == sgb_snap[4])
        check("sgb_cmd_data restored", bytes(gb.mmu.sgb_cmd_data) == sgb_snap[5])

        # v6 saves without an SGB tail clear in-progress packet assembly on load.
        v6_path = gb._state_path(1)
        v7_raw = open(gb._state_path(0), "rb").read()
        sgb_tail = 5 + 16 + 2 + len(sgb_snap[5])
        v6_raw = v7_raw[:4] + bytes([6]) + v7_raw[5:-sgb_tail]
        with open(v6_path, "wb") as f:
            f.write(v6_raw)
        gb.mmu.sgb_in_packet = True
        gb.mmu.sgb_bit_count = 99
        gb.mmu.sgb_cmd = 0xFF
        check("legacy v6 save loads", gb.load_state(1) is True)
        check("legacy v6 load clears sgb_in_packet", gb.mmu.sgb_in_packet is False)
        check("legacy v6 load clears sgb_bit_count", gb.mmu.sgb_bit_count == 0)
        check("legacy v6 load clears sgb_cmd", gb.mmu.sgb_cmd == 0)

        # Basenames longer than 255 UTF-8 bytes must not false-reject on load.
        long_base = ('x' * 256) + '.gbc'
        long_name = long_base.encode('utf-8', 'replace')[:31]
        gb.mmu.rom_path = long_base
        check("long basename identity accepted",
              gb._rom_identity_ok(len(long_name), 255, bytes(long_name)))
        gb.mmu.rom_path = rom_path

        # Expanded live SRAM buffer has stale tail cleared on load.
        ram_path = os.path.join(tmp, "sram.gbc")
        gb_ram = build_machine(ram_path, cart=0x13)
        if len(gb_ram.mmu.ram_data) < 0x2000:
            gb_ram.mmu.ram_data = bytearray(0x2000)
        gb_ram.mmu.ram_data[0x100] = 0xBE
        saved_len = len(gb_ram.mmu.ram_data)
        check("SRAM tail save succeeds", gb_ram.save_state(0) is True)
        gb_ram.mmu.ram_data.extend(b'\xFF' * 512)
        gb_ram.mmu.ram_data[saved_len + 10] = 0xFF
        gb_ram.mmu.ram_data[0x100] = 0
        check("SRAM tail load succeeds", gb_ram.load_state(0) is True)
        check("SRAM prefix restored", gb_ram.mmu.ram_data[0x100] == 0xBE)
        check("SRAM stale tail zeroed", gb_ram.mmu.ram_data[saved_len + 10] == 0)

    if _failures:
        print(f"\n{_failures} CHECK(S) FAILED")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
