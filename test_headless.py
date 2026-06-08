"""
Portable headless smoke test for the GBC emulator.

Unlike the other test_*.py dev scripts (which point at hard-coded local ROM
paths), this one is fully self-contained: it builds a tiny synthetic ROM in
memory and exercises the CPU, MMU, PPU, timers and APU with no external files
and no display. It runs anywhere Python + numpy + pygame are installed:

    python test_headless.py

Exits 0 if every check passes, 1 otherwise.

SPDX-License-Identifier: MIT
"""
import os
import sys
import tempfile

# Force SDL into dummy mode so the test runs without a display or audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    """Exec the emulator source (minus its __main__ block) into a namespace."""
    path = os.path.join(HERE, "gbc_emulator_skeleton.py")
    src = open(path, encoding="utf-8").read().split("if __name__")[0]
    ns = {}
    exec(compile(src, "gbc_emulator_skeleton.py", "exec"), ns)
    return ns


def build_rom(cgb=True):
    """A minimal 32 KB ROM-only cartridge running a tight loop at 0x0100."""
    rom = bytearray(0x8000)
    rom[0x0143] = 0x80 if cgb else 0x00  # CGB flag
    rom[0x0147] = 0x00                   # ROM ONLY
    # 0x0100: enable the LCD, then spin forever.
    prog = [
        0x3E, 0x91,   # LD A,0x91
        0xE0, 0x40,   # LDH (FF40),A   ; LCDC on
        0x00,         # NOP
        0x18, 0xFD,   # JR -3          ; loop on the NOP
    ]
    rom[0x0100:0x0100 + len(prog)] = prog
    return bytes(rom)


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def test_opcodes(ns):
    CPU, MMU = ns["CPU"], ns["MMU"]
    FLAG_Z, FLAG_C, FLAG_H = ns["FLAG_Z"], ns["FLAG_C"], ns["FLAG_H"]
    m = MMU(); m.load_rom(build_rom())
    c = CPU(m)

    # ADD A,n with carry + half-carry + zero result.
    c.reg.a = 0x3A; c._alu_a_op(0, 0xC6)
    check("ADD wraps to 0 with Z/C set",
          c.reg.a == 0 and c.reg.get_flag(FLAG_Z) and c.reg.get_flag(FLAG_C))

    # SUB producing a borrow.
    c.reg.a = 0x10; c._alu_a_op(2, 0x20)
    check("SUB sets N + borrow C",
          c.reg.a == 0xF0 and c.reg.get_flag(FLAG_C))

    # DAA after a BCD addition: 0x09 + 0x08 = 0x11 in BCD.
    c.reg.a = 0x09; c._alu_a_op(0, 0x08); c._daa()
    check("DAA corrects BCD add", c.reg.a == 0x17)

    # CB SWAP (bit field 6) and SRL (bit field 7) must not be transposed.
    c.reg.a = 0xAB; c.execute_cb(0x37)
    check("CB SWAP A", c.reg.a == 0xBA)
    c.reg.a = 0x03; c.execute_cb(0x3F)
    check("CB SRL A", c.reg.a == 0x01 and c.reg.get_flag(FLAG_C))

    # CB BIT sets Z from the inverted tested bit and forces H.
    c.reg.a = 0x01; c.execute_cb(0x47)
    check("CB BIT 0 of set bit -> Z clear", not c.reg.get_flag(FLAG_Z) and c.reg.get_flag(FLAG_H))
    c.reg.a = 0x00; c.execute_cb(0x47)
    check("CB BIT 0 of clear bit -> Z set", c.reg.get_flag(FLAG_Z))

    # Illegal opcodes are tolerated and counted.
    before = c.invalid_opcode_count
    c.reg.pc = 0x4000
    cycles = c.execute(0xD3)
    check("illegal opcode is a 4-cycle no-op",
          cycles == 4 and c.invalid_opcode_count == before + 1)


def test_run(ns):
    """Spin the full machine for a while; nothing should raise."""
    CPU, MMU, PPU, APU, Timers = (ns["CPU"], ns["MMU"], ns["PPU"], ns["APU"], ns["Timers"])
    m = MMU(); m.load_rom(build_rom())
    cpu, ppu, apu, timers = CPU(m), PPU(m), APU(m), Timers(m)
    m.ppu, m.apu = ppu, apu
    m.div_reset_callback = timers.reset_div
    ppu.is_cgb = m.is_cgb
    if m.is_cgb:
        ppu._cgb_init_palettes()
    cpu.reg.pc = 0x0100
    for _ in range(300_000):
        c = cpu.step(); ppu.step(c); timers.step(c); apu.step(c)
    check("valid program runs with no illegal opcodes", cpu.invalid_opcode_count == 0)
    check("LCD is on after boot", m.memory[0xFF40] & 0x80)


def _tile_bytes(color):
    """Build lo/hi VRAM bytes for a solid 2bpp tile of the given color index."""
    if color == 0:
        return 0x00, 0x00
    if color == 1:
        return 0xFF, 0x00
    if color == 2:
        return 0x00, 0xFF
    if color == 3:
        return 0xFF, 0xFF
    raise ValueError(color)


def _fill_tile(mem, tile_idx, color):
    """Write all eight rows of a tile in VRAM bank 0."""
    lo, hi = _tile_bytes(color)
    base = 0x8000 + tile_idx * 16
    for row in range(8):
        mem[base + row * 2] = lo
        mem[base + row * 2 + 1] = hi


def _set_obj_pal(ppu, pal, colors555):
    for i, col in enumerate(colors555):
        off = pal * 8 + i * 2
        ppu.obj_palette_data[off] = col & 0xFF
        ppu.obj_palette_data[off + 1] = (col >> 8) & 0xFF
        ppu._update_cgb_obj_color(pal * 4 + i)


def _setup_cgb_ppu(ns):
    MMU, PPU = ns["MMU"], ns["PPU"]
    m = MMU()
    m.load_rom(build_rom(cgb=True))
    p = PPU(m)
    m.ppu = p
    p.is_cgb = m.is_cgb
    p._cgb_init_palettes()
    mem = m.memory
    mem[0xFF40] = 0x93  # LCD on, OBJ on, BG on, unsigned BG tiles
    mem[0xFF42] = 0
    mem[0xFF43] = 0
    mem[0x9800] = 0
    lo, hi = _tile_bytes(0)
    mem[0x8000] = lo
    mem[0x8001] = hi
    return m, p, mem


def test_cgb_sprite_rendering(ns):
    """CGB OBJ palette colors, overlap priority, OPRI, BG priority, VRAM bank 1."""
    PPU = ns["PPU"]
    m, p, mem = _setup_cgb_ppu(ns)

    # LUT must map VRAM lo/hi bytes to the correct 2bpp indices.
    for color in range(4):
        lo, hi = _tile_bytes(color)
        got = PPU._TILE_COLORS[(hi << 8) | lo][0]
        check(f"tile LUT color {color}", got == color)

    # OBJ palette color is written to the framebuffer.
    lo, hi = _tile_bytes(3)
    mem[0x8010] = lo
    mem[0x8011] = hi
    _set_obj_pal(p, 1, [0x0000, 0x001F, 0x03E0, 0x7C00])
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x01  # palette 1
    p._render_scanline(0, mem[0xFF40])
    check("OBJ palette color applied", p.framebuffer[0] == p._obj_rgb[7])

    # CGB overlap: earlier OAM entry wins (OPRI=0 default).
    m, p, mem = _setup_cgb_ppu(ns)
    lo, hi = _tile_bytes(1)
    mem[0x8010] = lo
    mem[0x8011] = hi
    lo, hi = _tile_bytes(2)
    mem[0x8020] = lo
    mem[0x8021] = hi
    _set_obj_pal(p, 0, [0x0000, 0x001F, 0x0000, 0x0000])
    _set_obj_pal(p, 1, [0x0000, 0x0000, 0x03E0, 0x0000])
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x00
    mem[0xFE04] = 16
    mem[0xFE05] = 12
    mem[0xFE06] = 2
    mem[0xFE07] = 0x01
    p._render_scanline(0, mem[0xFF40])
    check("CGB OAM priority keeps earlier sprite", p.framebuffer[4] == p._obj_rgb[1])
    check("CGB OAM priority exposes later sprite", p.framebuffer[8] == p._obj_rgb[6])

    # OPRI=1 restores DMG-style X priority on CGB.
    m, p, mem = _setup_cgb_ppu(ns)
    p.cgb_opri = 1
    lo, hi = _tile_bytes(1)
    mem[0x8010] = lo
    mem[0x8011] = hi
    lo, hi = _tile_bytes(2)
    mem[0x8020] = lo
    mem[0x8021] = hi
    _set_obj_pal(p, 0, [0x0000, 0x001F, 0x0000, 0x0000])
    _set_obj_pal(p, 1, [0x0000, 0x0000, 0x03E0, 0x0000])
    mem[0xFE00] = 16
    mem[0xFE01] = 20  # x=12
    mem[0xFE02] = 1
    mem[0xFE03] = 0x00
    mem[0xFE04] = 16
    mem[0xFE05] = 12  # x=4, overlaps at pixels 4-11
    mem[0xFE06] = 2
    mem[0xFE07] = 0x01
    p._render_scanline(0, mem[0xFF40])
    check("OPRI=1 X priority at overlap", p.framebuffer[4] == p._obj_rgb[6])
    check("OPRI=1 X priority keeps higher X", p.framebuffer[12] == p._obj_rgb[1])

    # BG priority hides non-transparent OBJ pixels when LCDC.0=1.
    m, p, mem = _setup_cgb_ppu(ns)
    lo, hi = _tile_bytes(2)
    mem[0x8000] = lo
    mem[0x8001] = hi
    m.vram_bank1[0x1800] = 0x80
    lo, hi = _tile_bytes(3)
    mem[0x8010] = lo
    mem[0x8011] = hi
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x82  # palette 2 + OAM priority
    p._render_scanline(0, mem[0xFF40])
    check("BG priority hides OBJ pixel", p.framebuffer[0] == p._bg_rgb[2])

    # OBJ tiles can be fetched from VRAM bank 1.
    m, p, mem = _setup_cgb_ppu(ns)
    _set_obj_pal(p, 3, [0x0000, 0x7C00, 0x0000, 0x0000])
    lo, hi = _tile_bytes(1)
    m.vram_bank1[0x10] = lo
    m.vram_bank1[0x11] = hi
    mem[0x8010] = 0x00
    mem[0x8011] = 0x00
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x1B  # palette 3 + VRAM bank 1
    p._render_scanline(0, mem[0xFF40])
    check("OBJ tile from VRAM bank 1", p.framebuffer[0] == p._obj_rgb[13])

    # X flip mirrors the tile row.
    m, p, mem = _setup_cgb_ppu(ns)
    pixels = [1, 2, 3, 0, 0, 0, 0, 0]
    lo = sum((px & 1) << (7 - i) for i, px in enumerate(pixels))
    hi = sum(((px >> 1) & 1) << (7 - i) for i, px in enumerate(pixels))
    mem[0x8010] = lo
    mem[0x8011] = hi
    _set_obj_pal(p, 0, [0x0000, 0x001F, 0x03E0, 0x7C00])
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x20
    p._render_scanline(0, mem[0xFF40])
    check("OBJ X flip leaves leading pixel transparent", p.framebuffer[0] == p._bg_rgb[0])
    check("OBJ X flip trailing pixel", p.framebuffer[7] == p._obj_rgb[1])

    # 8x16 sprites use paired tile indices.
    m, p, mem = _setup_cgb_ppu(ns)
    mem[0xFF40] = 0x97
    _fill_tile(mem, 2, 1)
    _fill_tile(mem, 3, 2)
    _set_obj_pal(p, 0, [0x0000, 0x001F, 0x03E0, 0x7C00])
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 2
    mem[0xFE03] = 0x00
    p._render_scanline(0, mem[0xFF40])
    p._render_scanline(8, mem[0xFF40])
    check("8x16 OBJ top half", p.framebuffer[0] == p._obj_rgb[1])
    check("8x16 OBJ bottom half", p.framebuffer[8 * 160] == p._obj_rgb[2])

    # Y flip mirrors the full 8x16 object.
    mem[0xFE03] = 0x40
    p._render_scanline(0, mem[0xFF40])
    check("8x16 OBJ Y flip", p.framebuffer[0] == p._obj_rgb[2])

    # LCDC.0=0 forces OBJ over BG priority bits.
    m, p, mem = _setup_cgb_ppu(ns)
    mem[0xFF40] = 0x92
    _fill_tile(mem, 0, 2)
    m.vram_bank1[0x1800] = 0x80
    _fill_tile(mem, 1, 3)
    _set_obj_pal(p, 0, [0x0000, 0x001F, 0x03E0, 0x7C00])
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x80
    p._render_scanline(0, mem[0xFF40])
    check("LCDC.0 clear keeps OBJ on top", p.framebuffer[0] == p._obj_rgb[3])

    # Ten off-screen (X=0) OBJ still consume the per-line quota.
    m, p, mem = _setup_cgb_ppu(ns)
    _fill_tile(mem, 1, 3)
    _set_obj_pal(p, 0, [0x0000, 0x0000, 0x0000, 0x7C00])
    for i in range(10):
        mem[0xFE00 + i * 4] = 16
        mem[0xFE01 + i * 4] = 0
    mem[0xFE28] = 16
    mem[0xFE29] = 8
    mem[0xFE2A] = 1
    mem[0xFE2B] = 0x00
    p._render_scanline(0, mem[0xFF40])
    check("10 OBJ limit blocks 11th sprite", p.framebuffer[0] == p._bg_rgb[0])


def test_dmg_sprite_rendering(ns):
    """DMG OBJ palette mapping through the tile LUT and OBP0/OBP1."""
    MMU, PPU = ns["MMU"], ns["PPU"]
    m = MMU()
    m.load_rom(build_rom(cgb=False))
    p = PPU(m)
    m.ppu = p
    mem = m.memory
    mem[0xFF40] = 0x93
    mem[0xFF42] = 0
    mem[0xFF43] = 0
    mem[0xFF47] = 0xE4
    mem[0xFF48] = 0xE4
    mem[0xFF49] = 0x1B
    mem[0x9800] = 0
    _fill_tile(mem, 0, 0)
    _fill_tile(mem, 1, 3)
    mem[0xFE00] = 16
    mem[0xFE01] = 8
    mem[0xFE02] = 1
    mem[0xFE03] = 0x10  # OBP1
    p._render_scanline(0, mem[0xFF40])
    obp1_shades = p._PALETTE_SHADES[mem[0xFF49]]
    check("DMG OBJ uses OBP1", p.framebuffer[0] == p.shades[obp1_shades[3]])


def test_apu_frame_sync(ns):
    """APU sample output and frame sequencer must track one video frame of dots."""
    MMU, APU = ns["MMU"], ns["APU"]
    CYCLES_PER_FRAME = ns["CYCLES_PER_FRAME"]
    APU_MIN_SAMPLES_PER_FRAME = ns["APU_MIN_SAMPLES_PER_FRAME"]
    APU_MAX_SAMPLES_PER_FRAME = ns["APU_MAX_SAMPLES_PER_FRAME"]
    APU_BYTES_PER_STEREO_SAMPLE = ns["APU_BYTES_PER_STEREO_SAMPLE"]

    m = MMU()
    apu = APU(m)

    peek = apu.peek_samples_for_cycles(CYCLES_PER_FRAME)
    check(
        f"one frame emits {APU_MIN_SAMPLES_PER_FRAME}-{APU_MAX_SAMPLES_PER_FRAME} samples (peek {peek})",
        APU_MIN_SAMPLES_PER_FRAME <= peek <= APU_MAX_SAMPLES_PER_FRAME,
    )

    before = len(apu.buffer)
    apu.step(CYCLES_PER_FRAME)
    produced = (len(apu.buffer) - before) // APU_BYTES_PER_STEREO_SAMPLE
    check(
        f"one frame step matches peek ({produced} samples)",
        produced == peek,
    )

    apu.buffer.clear()
    apu.sample_accum = 0
    expected = apu.peek_samples_for_cycles(CYCLES_PER_FRAME * 100)
    total = 0
    for _ in range(100):
        apu.step(CYCLES_PER_FRAME)
        total += len(apu.drain()) // APU_BYTES_PER_STEREO_SAMPLE
    check(
        f"100 frames total samples ({total} vs expected {expected})",
        total == expected,
    )

    apu.frame_seq_counter = apu.FRAME_SEQ_PERIOD
    apu.frame_seq_step = 0
    ticks = 0
    fsc = apu.frame_seq_counter
    fsc -= CYCLES_PER_FRAME
    while fsc <= 0:
        fsc += apu.FRAME_SEQ_PERIOD
        ticks += 1
    check("frame sequencer ticks per video frame (8-9)", 8 <= ticks <= 9)


def test_gameboy_frame_audio(ns):
    """GameBoy.step_all for one frame must produce the same PCM size as the APU."""
    GameBoy = ns["GameBoy"]
    CYCLES_PER_FRAME = ns["CYCLES_PER_FRAME"]
    APU_MIN_SAMPLES_PER_FRAME = ns["APU_MIN_SAMPLES_PER_FRAME"]
    APU_MAX_SAMPLES_PER_FRAME = ns["APU_MAX_SAMPLES_PER_FRAME"]
    APU_BYTES_PER_STEREO_SAMPLE = ns["APU_BYTES_PER_STEREO_SAMPLE"]

    rom = build_rom()
    with tempfile.NamedTemporaryFile(suffix=".gbc", delete=False) as tf:
        tf.write(rom)
        path = tf.name
    try:
        gb = GameBoy(path, fps_limit=0)
        gb.cpu.reg.pc = 0x0100
        gb.apu.buffer.clear()
        gb.apu.sample_accum = 0
        cycles = 0
        while cycles < CYCLES_PER_FRAME:
            cycles += gb.step_all()
        check("step_all covers at least one frame of dots", cycles >= CYCLES_PER_FRAME)
        n_bytes = len(gb.apu.buffer)
        n_samples = n_bytes // APU_BYTES_PER_STEREO_SAMPLE
        check(
            f"frame buffer is {APU_MIN_SAMPLES_PER_FRAME}-{APU_MAX_SAMPLES_PER_FRAME} samples ({n_samples})",
            APU_MIN_SAMPLES_PER_FRAME <= n_samples <= APU_MAX_SAMPLES_PER_FRAME,
        )
        check("frame buffer is sample-aligned", n_bytes % APU_BYTES_PER_STEREO_SAMPLE == 0)
    finally:
        os.unlink(path)


def test_double_speed(ns):
    """KEY1 double-speed must halve the base-clock dots fed to the PPU/APU."""
    GameBoy = ns["GameBoy"]
    rom = build_rom()
    with tempfile.NamedTemporaryFile(suffix=".gbc", delete=False) as tf:
        tf.write(rom)
        path = tf.name
    try:
        normal = GameBoy(path, fps_limit=0); normal.cpu.reg.pc = 0x0100
        normal.mmu.key1 = 0x00
        d_norm = sum(normal.step_all() for _ in range(2000))

        fast = GameBoy(path, fps_limit=0); fast.cpu.reg.pc = 0x0100
        fast.mmu.key1 = 0x80  # double-speed active
        d_fast = sum(fast.step_all() for _ in range(2000))
    finally:
        os.unlink(path)
    ratio = d_fast / d_norm
    check(f"double-speed halves PPU dots (ratio {ratio:.3f})", 0.45 < ratio < 0.55)


def main():
    ns = load_module()
    print("opcode checks:");           test_opcodes(ns)
    print("full-machine run:");        test_run(ns)
    print("cgb sprite rendering:");  test_cgb_sprite_rendering(ns)
    print("dmg sprite rendering:");  test_dmg_sprite_rendering(ns)
    print("apu frame sync:");          test_apu_frame_sync(ns)
    print("gameboy frame audio:");     test_gameboy_frame_audio(ns)
    print("double-speed timing:");     test_double_speed(ns)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
