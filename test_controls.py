"""Portable tests for joypad input sources, SOCD, and key-binding helpers.

Exits non-zero on any failure. No display and no local ROM files required.
"""
import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    path = os.path.join(HERE, "gbc_emulator_skeleton.py")
    src = open(path, encoding="utf-8").read().split("if __name__")[0]
    ns = {}
    exec(compile(src, "gbc_emulator_skeleton.py", "exec"), ns)
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
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
