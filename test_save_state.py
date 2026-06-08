"""Portable save-state round-trip test (synthetic ROM, no display, no local files).

Builds a tiny CGB ROM in memory, runs it, snapshots the emulator with
``save_state``, mutates live state, reloads with ``load_state`` and verifies the
machine is restored exactly.  Also checks that the derived CGB colour tables
(``_bg_rgb`` / ``_obj_rgb``) are rebuilt from the saved palette bytes after a
load, which guards the packed-24-bit framebuffer path.

Exits non-zero on any failure.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

src = open(os.path.join(HERE, "gbc_emulator_skeleton.py"), encoding="utf-8").read()
src = src.split("if __name__")[0]
_ns = {}
exec(compile(src, "gbc_emulator_skeleton.py", "exec"), _ns)
MMU, CPU, PPU, APU, Timers = (_ns["MMU"], _ns["CPU"], _ns["PPU"], _ns["APU"], _ns["Timers"])
GameBoy = _ns["GameBoy"]
PALETTE_DMG = _ns["PALETTE_DMG"]
_shader_none = _ns["_shader_none"]

_failures = 0


def check(name, ok):
    global _failures
    print(f"  {'ok' if ok else 'FAIL'}: {name}")
    if not ok:
        _failures += 1


def build_machine(rom_path):
    """Wire a headless GameBoy around a freshly loaded synthetic CGB ROM."""
    rom = bytearray(0x8000)
    rom[0x0143] = 0x80  # CGB compatible
    rom[0x0147] = 0x00  # ROM ONLY
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

    if _failures:
        print(f"\n{_failures} CHECK(S) FAILED")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
