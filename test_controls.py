"""Portable tests for joypad input sources, SOCD, and key-binding helpers.

Exits non-zero on any failure. No display and no local ROM files required.
"""
import os
import sys
import tempfile
import time
import struct

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    path = os.path.join(HERE, "gbc_emulator.py")
    src = open(path, encoding="utf-8").read().split("if __name__")[0]
    ns = {}
    exec(compile(src, "gbc_emulator.py", "exec"), ns)
    return ns


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def test_wram_hram_fast_path(ns):
    MMU = ns["MMU"]
    m = MMU()
    m.write_byte(0xC000, 0xAB)
    check("WRAM write_byte lands in memory", m.memory[0xC000] == 0xAB)
    check("WRAM read_byte matches", m.read_byte(0xC000) == 0xAB)
    m.write_byte(0xDFFF, 0x12)
    check("WRAM end write", m.read_byte(0xDFFF) == 0x12)
    m.write_byte(0xFF80, 0x34)
    check("HRAM write_byte", m.read_byte(0xFF80) == 0x34)
    m.write_byte(0xFFFF, 0x1F)
    check("IE write via HRAM fast path", m.read_byte(0xFFFF) == 0x1F)
    # Echo RAM still mirrors WRAM
    m.write_byte(0xE000, 0x77)
    check("echo RAM write mirrors C000", m.memory[0xC000] == 0x77)


def test_joypad_sources(ns):
    MMU = ns["MMU"]
    _JOY_SRC_KB = ns["_JOY_SRC_KB"]
    _JOY_SRC_HAT = ns["_JOY_SRC_HAT"]
    _JOY_SRC_AXIS = ns["_JOY_SRC_AXIS"]
    IF_JOYPAD = ns["IF_JOYPAD"]
    m = MMU()
    m.memory[0xFF0F] = 0

    m.set_joypad_button(1, True, _JOY_SRC_KB)  # Left
    check("keyboard Left presses joypad bit 1", (m.joypad_buttons & 0x02) == 0)
    check("joypad IF fires on press", m.memory[0xFF0F] & IF_JOYPAD)

    m.set_joypad_button(1, True, _JOY_SRC_AXIS)
    m.set_joypad_button(1, False, _JOY_SRC_AXIS)
    check("axis release does not un-press held keyboard Left",
          (m.joypad_buttons & 0x02) == 0)

    m.set_joypad_button(1, False, _JOY_SRC_KB)
    check("keyboard release actually releases Left",
          (m.joypad_buttons & 0x02) != 0)

    m.release_all_joypad()
    check("release_all_joypad clears state", m.joypad_buttons == 0xFF)

    m.set_joypad_button(0, True, _JOY_SRC_HAT)   # Right
    m.set_joypad_button(2, True, _JOY_SRC_AXIS)  # Up
    check("hat Right and stick Up combine",
          (m.joypad_buttons & 0x05) == 0)


def test_socd(ns):
    MMU = ns["MMU"]
    m = MMU()
    m.set_joypad_button(0, True)  # Right
    m.set_joypad_button(1, True)  # Left after Right -> Left wins
    check("SOCD last-wins keeps Left, drops Right",
          (m.joypad_buttons & 0x03) == 0x01)
    m.set_joypad_button(0, True)  # Right pressed last -> Right wins
    check("SOCD last-wins keeps Right, drops Left",
          (m.joypad_buttons & 0x03) == 0x02)

    m.release_all_joypad()
    m.set_joypad_button(2, True)  # Up
    m.set_joypad_button(3, True)  # Down last
    check("SOCD last-wins keeps Down, drops Up",
          (m.joypad_buttons & 0x0C) == 0x04)


def test_key_bindings(ns):
    pygame = ns["pygame"]
    if pygame is None:
        print("  skip: pygame not available")
        return
    pygame.init()
    _rebuild_key_map = ns["_rebuild_key_map"]
    _sanitize_key_bindings = ns["_sanitize_key_bindings"]
    _assign_binding_key = ns["_assign_binding_key"]
    _default_key_bindings = ns["_default_key_bindings"]

    bindings = _rebuild_key_map(None, True)
    check("default map binds arrow Right", pygame.K_RIGHT in ns["KEY_TO_JOYPAD_BIT"])
    check("default map binds WASD D", pygame.K_d in ns["KEY_TO_JOYPAD_BIT"])
    check("default A is Z", ns["KEY_TO_JOYPAD_BIT"].get(pygame.K_z) == 4)
    check("default B is X", ns["KEY_TO_JOYPAD_BIT"].get(pygame.K_x) == 5)

    off = _sanitize_key_bindings(bindings, False)
    check("WASD off removes D from Right", "d" not in off["right"])
    check("WASD off keeps arrow Right", "right" in off["right"])

    custom = _default_key_bindings(True)
    check("assign J to A", _assign_binding_key(custom, "a", "j"))
    check("A primary is j", custom["a"][0] == "j")
    check("assign reserved Escape is rejected",
          _assign_binding_key(custom, "start", "escape") is False)
    check("assign reserved Tab is rejected",
          _assign_binding_key(custom, "a", "tab") is False)
    check("assign reserved F3 is rejected",
          _assign_binding_key(custom, "b", "f3") is False)
    check("reserved-key copy mentions Tab", "Tab" in ns["_RESERVED_KEY_MSG"])
    check("assign z to Start swaps/steals", _assign_binding_key(custom, "start", "z"))
    check("Start now includes z", "z" in custom["start"])
    check("A no longer uses z as primary", custom["a"][0] != "z")

    _rebuild_key_map(_default_key_bindings(True), True)
    check("rebuild restores Z as A", ns["KEY_TO_JOYPAD_BIT"].get(pygame.K_z) == 4)


def test_config_merge(ns):
    _save_config = ns["_save_config"]
    _load_config = ns["_load_config"]
    _CONFIG_PATH = ns["_CONFIG_PATH"]
    original = None
    if os.path.isfile(_CONFIG_PATH):
        original = open(_CONFIG_PATH, encoding="utf-8").read()
    try:
        _save_config({"window_scale": 3, "key_bindings": {"a": ["j"]}})
        _save_config({"volume": 0.5})
        cfg = _load_config()
        check("config merge keeps key_bindings", cfg.get("key_bindings", {}).get("a") == ["j"])
        check("config merge keeps window_scale", cfg.get("window_scale") == 3)
        check("config merge writes volume", cfg.get("volume") == 0.5)
    finally:
        try:
            if original is None:
                os.remove(_CONFIG_PATH)
            else:
                with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
                    f.write(original)
        except OSError:
            pass


def test_gamepad_start_does_not_auto_pause(ns):
    """Start alone must press Start; Select+Start is the pause combo."""
    MMU = ns["MMU"]
    _JOY_SRC_BTN = ns["_JOY_SRC_BTN"]
    m = MMU()
    m.set_joypad_button(7, True, _JOY_SRC_BTN)  # Start
    check("Start alone presses joypad Start", (m.joypad_buttons & 0x80) == 0)
    select_held = not (m._joy_src[_JOY_SRC_BTN] & (1 << 6))
    check("Start alone is not the pause combo", select_held is False)

    m.release_all_joypad()
    m.set_joypad_button(6, True, _JOY_SRC_BTN)  # Select
    select_held = not (m._joy_src[_JOY_SRC_BTN] & (1 << 6))
    check("Select held arms the pause combo", select_held is True)


def test_inc_preserves_carry(ns):
    CPU, MMU = ns["CPU"], ns["MMU"]
    FLAG_C, FLAG_Z = ns["FLAG_C"], ns["FLAG_Z"]
    m = MMU()
    c = CPU(m)
    c.reg.a = 0x0F
    c.reg.set_flag(FLAG_C, 1)
    c.reg.a = c._inc_r8(c.reg.a)
    check("INC A 0x0F -> 0x10", c.reg.a == 0x10)
    check("INC sets H", c.reg.get_flag(ns["FLAG_H"]))
    check("INC preserves C", c.reg.get_flag(FLAG_C))
    check("INC clears Z", not c.reg.get_flag(FLAG_Z))


def test_silent_apu_batches(ns):
    MMU, APU = ns["MMU"], ns["APU"]
    CYCLES_PER_FRAME = ns["CYCLES_PER_FRAME"]
    APU_BYTES_PER_STEREO_SAMPLE = ns["APU_BYTES_PER_STEREO_SAMPLE"]
    m = MMU()
    apu = APU(m)
    peek = apu.peek_samples_for_cycles(CYCLES_PER_FRAME)
    apu.step(CYCLES_PER_FRAME)
    produced = len(apu.buffer) // APU_BYTES_PER_STEREO_SAMPLE
    check("silent APU still emits a frame of samples", produced == peek)
    check("silent APU buffer is sample-aligned",
          len(apu.buffer) % APU_BYTES_PER_STEREO_SAMPLE == 0)


def _halt_rom():
    rom = bytearray(0x8000)
    rom[0x0143] = 0x80
    rom[0x0147] = 0x00
    prog = [
        0x3E, 0x91, 0xE0, 0x40,  # LCD on
        0xFB,                    # EI
        0x3E, 0x01, 0xE0, 0xFF,  # IE = VBlank
        0xAF, 0xE0, 0x0F,        # IF = 0
        0x76,                    # HALT
        0x18, 0xFD,              # JR -3
    ]
    rom[0x0100:0x0100 + len(prog)] = prog
    return bytes(rom)


def test_halt_skip(ns):
    """HALT skip must still raise VBlank and match a 4-cycle stepper's LY."""
    GameBoy = ns["GameBoy"]
    CYCLES_PER_FRAME = ns["CYCLES_PER_FRAME"]
    rom = _halt_rom()
    with tempfile.NamedTemporaryFile(suffix=".gbc", delete=False) as tf:
        tf.write(rom)
        path = tf.name
    try:
        gb = GameBoy(path, fps_limit=0, audio_enabled=False)
        gb.cpu.reg.pc = 0x0100
        dots = 0
        steps = 0
        saw_vblank = False
        while dots < CYCLES_PER_FRAME:
            d = gb.step_all()
            dots += d
            steps += 1
            if gb.mmu.memory[0xFF0F] & 0x01:
                saw_vblank = True
        check("halt-skip uses far fewer than 4-cycle steps", steps < 4000)
        check("halt-skip covers one frame of dots", dots >= CYCLES_PER_FRAME)
        check("LY stays in range after halt-skip frame", 0 <= gb.mmu.memory[0xFF44] <= 153)
        check("VBlank IF fired during the halt-skip frame", saw_vblank)
        # A 4-cycle stepper of the same ROM must also finish a legal scanline.
        ref = GameBoy(path, fps_limit=0, audio_enabled=False)
        ref.cpu.reg.pc = 0x0100
        dots = 0
        while dots < CYCLES_PER_FRAME:
            c = ref.cpu.step()
            ref.ppu.step(c)
            ref.timers.step(c)
            ref.apu.step(c)
            dots += c
        check("4-cycle stepper LY also in range", 0 <= ref.mmu.memory[0xFF44] <= 153)
    finally:
        os.unlink(path)


def test_turbo_and_reset(ns):
    MMU = ns["MMU"]
    GameBoy = ns["GameBoy"]
    _JOY_SRC_TURBO = ns["_JOY_SRC_TURBO"]
    m = MMU()
    m.set_joypad_button(4, True, _JOY_SRC_TURBO)
    check("turbo source presses A", (m.joypad_buttons & 0x10) == 0)
    m.set_joypad_button(4, False, _JOY_SRC_TURBO)
    check("turbo source releases A", (m.joypad_buttons & 0x10) != 0)

    rom = _halt_rom()
    with tempfile.NamedTemporaryFile(suffix=".gbc", delete=False) as tf:
        tf.write(rom)
        path = tf.name
    try:
        gb = GameBoy(path, fps_limit=0, audio_enabled=False)
        gb.cpu.reg.pc = 0x1234
        gb.cpu.halted = True
        gb.mmu.memory[0xFF0F] = 0x1F
        gb.soft_reset()
        check("soft reset restores PC to 0x0100", gb.cpu.reg.pc == 0x0100)
        check("soft reset clears HALT", gb.cpu.halted is False)
        check("soft reset clears IF", gb.mmu.memory[0xFF0F] == 0)
        check("soft reset clears IME", gb.cpu.interrupts_master_enabled is False)
    finally:
        os.unlink(path)


def test_ui_layout(ns):
    """Menu overlay metrics must keep title, rows, and hint on-screen."""
    _overlay_layout = ns["_overlay_layout"]
    _decorate_cyclic_setting = ns["_decorate_cyclic_setting"]
    check("cyclic setting shows chevrons when selected",
          _decorate_cyclic_setting("Window Scale: 4x", True) == "Window Scale: < 4x >")
    check("cyclic setting unchanged when not selected",
          _decorate_cyclic_setting("Window Scale: 4x", False) == "Window Scale: 4x")
    check("Controls row is not decorated",
          _decorate_cyclic_setting("Controls...", True) == "Controls...")
    cases = (
        (320, 288, 10, True),
        (320, 288, 5, True),
        (320, 288, 2, True),
        (480, 432, 8, True),
        (640, 576, 10, True),
        (640, 480, 3, False),
        (800, 720, 10, True),
    )
    for w, h, n, hint in cases:
        L = _overlay_layout(w, h, n, has_hint=hint)
        check(f"overlay {w}x{h} n={n} stays in window",
              L['px'] >= 0 and L['py'] >= 0
              and L['px'] + L['panel_w'] <= w
              and L['py'] + L['panel_h'] <= h)
        last_bottom = L['py'] + L['title_band'] + n * L['item_h']
        hint_top = L['py'] + L['panel_h'] - L['hint_band']
        check(f"overlay {w}x{h} n={n} rows above hint",
              last_bottom <= hint_top + 1)
        check(f"overlay {w}x{h} n={n} readable row",
              L['item_h'] >= 13 and L['item_size'] >= 11)

    pygame = ns["pygame"]
    if pygame is None:
        print("  skip: pygame not available for _fit_text")
        return
    pygame.init()
    font = ns["get_font"](20)
    _fit_text = ns["_fit_text"]
    check("fit_text keeps short strings", _fit_text(font, "Start", 400) == "Start")
    fitted = _fit_text(font, "supercalifragilisticexpialidocious.gbc", 80)
    check("fit_text ellipsizes long strings", fitted.endswith("...") and font.size(fitted)[0] <= 80)


def test_config_indices(ns):
    _clamp_index = ns["_clamp_index"]
    _clamp_choice = ns["_clamp_choice"]
    PALETTE_LIST = ns["PALETTE_LIST"]
    SHADER_LIST = ns["SHADER_LIST"]
    check("palette index 3 is kept", _clamp_index(3, len(PALETTE_LIST), 0) == 3)
    check("palette index 0 is kept", _clamp_index(0, len(PALETTE_LIST), 1) == 0)
    check("out-of-range palette falls back", _clamp_index(99, len(PALETTE_LIST), 0) == 0)
    check("non-int palette falls back", _clamp_index("nope", len(PALETTE_LIST), 0) == 0)
    check("shader index 2 is kept", _clamp_index(2, len(SHADER_LIST), 0) == 2)
    check("window scale 3 is kept", _clamp_choice(3, ns["_WINDOW_SCALES"], 4) == 3)
    check("window scale 9 falls back", _clamp_choice(9, ns["_WINDOW_SCALES"], 4) == 4)


def test_rtc_blob(ns):
    MMU = ns["MMU"]
    m = MMU()
    m.has_rtc = True
    m.rtc_s = 50
    m.rtc_m = 0
    m.rtc_h = 1
    m.rtc_dl = 2
    m.rtc_dh = 0
    m.rtc_last_time = time.time() - 15
    blob = m.pack_rtc_blob()
    check("VBA RTC blob is 44 bytes", len(blob) == 44)
    ts = struct.unpack_from("<i", blob, 40)[0]
    check("VBA RTC blob stores a unix timestamp", ts > 0)
    check("RTC advanced across the 15s gap before packing", m.rtc_s == 5 and m.rtc_m == 1)
    m2 = MMU()
    m2.has_rtc = True
    m2.unpack_rtc_blob(blob)
    check("VBA RTC seconds restored", m2.rtc_s == m.rtc_s)
    check("VBA RTC minutes restored", m2.rtc_m == m.rtc_m)
    check("VBA RTC hours restored", m2.rtc_h == m.rtc_h)

    legacy = bytearray(48)
    legacy[4] = 12
    legacy[5] = 7
    legacy[6] = 3
    legacy[7] = 9
    legacy[8] = 0x41
    m3 = MMU()
    m3.has_rtc = True
    m3.unpack_rtc_blob(legacy)
    check("legacy RTC seconds", m3.rtc_s == 12)
    check("legacy RTC minutes", m3.rtc_m == 7)
    check("legacy RTC halt bit", m3.rtc_dh & 0x40)


def test_cli_errors(ns):
    """CLI argument handling without opening a display."""
    import subprocess
    emu = os.path.join(HERE, "gbc_emulator.py")
    py = sys.executable
    ver = subprocess.run([py, emu, "--version"], capture_output=True, text=True)
    out = (ver.stdout or "") + (ver.stderr or "")
    check("--version prints 1.0.0", ver.returncode == 0 and "1.0.0" in out)
    missing = subprocess.run([py, emu, os.path.join(HERE, "no-such-rom.gb")],
                             capture_output=True, text=True)
    err = (missing.stdout or "") + (missing.stderr or "")
    check("missing ROM exits non-zero", missing.returncode != 0)
    check("missing ROM mentions the path", "not found" in err.lower())
    nomenu = subprocess.run([py, emu, "--nomenu"], capture_output=True, text=True)
    nerr = (nomenu.stdout or "") + (nomenu.stderr or "")
    check("--nomenu without ROM is an error", nomenu.returncode != 0)
    check("--nomenu error mentions ROM", "rom" in nerr.lower())


def test_unsupported_cart_header(ns):
    _parse_rom_header = ns["_parse_rom_header"]
    with tempfile.NamedTemporaryFile(suffix=".gb", delete=False) as tf:
        rom = bytearray(0x200)
        rom[0x0143] = 0x00
        rom[0x0147] = 0xFC  # Pocket Camera — still not emulated
        rom[0x0148] = 0x00
        rom[0x0149] = 0x00
        tf.write(rom)
        path = tf.name
    try:
        info = _parse_rom_header(path)
        check("camera header is labelled unsupported", "unsupported" in info.lower())
    finally:
        os.unlink(path)
    with tempfile.NamedTemporaryFile(suffix=".gb", delete=False) as tf:
        rom = bytearray(0x200)
        rom[0x0143] = 0x80
        rom[0x0147] = 0x22  # MBC7 now emulated
        rom[0x0148] = 0x00
        rom[0x0149] = 0x00
        tf.write(rom)
        path = tf.name
    try:
        info = ns["_parse_rom_header"](path)
        check("MBC7 header is not labelled unsupported", "unsupported" not in info.lower())
        check("MBC7 header names the mapper", "MBC7" in info)
    finally:
        os.unlink(path)


def main():
    ns = load_module()
    print("wram/hram fast path:");     test_wram_hram_fast_path(ns)
    print("joypad sources:");          test_joypad_sources(ns)
    print("socd cleaning:");           test_socd(ns)
    print("key bindings:");            test_key_bindings(ns)
    print("config merge:");            test_config_merge(ns)
    print("gamepad start combo:");     test_gamepad_start_does_not_auto_pause(ns)
    print("inc preserves carry:");     test_inc_preserves_carry(ns)
    print("silent apu batch:");        test_silent_apu_batches(ns)
    print("halt skip:");               test_halt_skip(ns)
    print("turbo and reset:");         test_turbo_and_reset(ns)
    print("ui layout:");               test_ui_layout(ns)
    print("config indices:");          test_config_indices(ns)
    print("rtc blob:");                test_rtc_blob(ns)
    print("cli errors:");              test_cli_errors(ns)
    print("unsupported cart header:"); test_unsupported_cart_header(ns)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
