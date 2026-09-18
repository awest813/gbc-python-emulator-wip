"""Portable tests for SGB, MBC6/7, serial bit-clocking, and CGB cart wait-states.

Exits non-zero on any failure. No display and no local ROM files required.
"""
import os
import sys

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


def build_rom(cart=0x00, cgb=True, sgb=False, size=0x8000):
    rom = bytearray(size)
    rom[0x0143] = 0x80 if cgb else 0x00
    rom[0x0146] = 0x03 if sgb else 0x00
    rom[0x0147] = cart
    # size code: 0=32KB, 1=64KB, 2=128KB, ...
    nbank_16k = max(2, size // 0x4000)
    code = 0
    banks = 2
    while banks < nbank_16k and code < 8:
        banks <<= 1
        code += 1
    rom[0x0148] = code
    rom[0x0100:0x0105] = bytes([0x00, 0x18, 0xFD, 0x00, 0x00])  # NOP / JR -3
    return bytes(rom)


def _send_sgb_packet(mmu, payload):
    data = bytearray(16)
    data[:len(payload)] = payload
    mmu.write_byte(0xFF00, 0x00)
    mmu.write_byte(0xFF00, 0x30)
    for byte in data:
        for i in range(8):
            mmu.write_byte(0xFF00, 0x30)
            mmu.write_byte(0xFF00, 0x20 if (byte >> i) & 1 else 0x10)
            mmu.write_byte(0xFF00, 0x30)


def test_serial_bit_clock(ns):
    GameBoy = ns["GameBoy"]
    IF_SERIAL = ns["IF_SERIAL"]
    gb = GameBoy(rom_path=None, audio_enabled=False)
    m = gb.mmu
    m.memory[0xFF0F] = 0
    m.serial_data = 0x00
    m.write_byte(0xFF02, 0x81)
    check("SC bit 7 stays set after starting a transfer", m.read_byte(0xFF02) & 0x80)
    check("eight bits remain to shift", m.serial_bits_left == 8)
    m._serial_step(511)
    check("no bit clocks before 512 T-cycles", m.serial_bits_left == 8)
    m._serial_step(1)
    check("first bit after 512 T-cycles", m.serial_bits_left == 7)
    check("disconnected input shifts a 1 into SB", m.serial_data == 0x01)
    m._serial_step(512 * 7)
    check("SC bit 7 clears after 8 bits", (m.serial_control & 0x80) == 0)
    check("disconnected transfer leaves SB = 0xFF", m.serial_data == 0xFF)
    check("serial IF is raised on completion", m.memory[0xFF0F] & IF_SERIAL)

    m.is_cgb = True
    m.memory[0xFF0F] = 0
    m.serial_data = 0x00
    m.write_byte(0xFF02, 0x83)
    m._serial_step(16)
    check("CGB fast clock is 16 T-cycles/bit", m.serial_bits_left == 7)


def test_cart_wait_states(ns):
    CPU, MMU = ns["CPU"], ns["MMU"]
    m = MMU()
    m.load_rom(build_rom(cgb=True))
    m.is_cgb = True
    c = CPU(m)
    m.memory[0x0150] = 0x00
    c.reg.pc = 0x0150
    m.key1 = 0x00
    check("NOP from ROM is 4 T-cycles at normal speed", c.step() == 4)

    m.key1 = 0x80
    c.reg.pc = 0x0150
    check("NOP from ROM is 5 T-cycles in double-speed", c.step() == 5)

    m.memory[0xC000] = 0x00
    c.reg.pc = 0xC000
    check("NOP from WRAM stays 4 T-cycles in double-speed", c.step() == 4)

    m.memory[0x0150] = 0x3A  # LD A,(HL-)
    c.reg.pc = 0x0150
    c.reg.hl = 0x0100
    check("LD A,(HL-) from ROM is 10 T-cycles in double-speed", c.step() == 10)


def test_mbc6_banking(ns):
    MMU = ns["MMU"]
    rom = bytearray(build_rom(cart=0x20, cgb=True, size=0x10000))
    rom[0x4000] = 0x42  # 8KB bank 2
    rom[0x6000] = 0x43  # 8KB bank 3
    m = MMU()
    m.load_rom(bytes(rom))
    check("MBC6 is a supported mapper", m.mbc_type == 0x20)
    m.write_byte(0x2000, 2)
    m.write_byte(0x3000, 3)
    check("MBC6 window A maps 8KB bank 2", m.read_byte(0x4000) == 0x42)
    check("MBC6 window B maps 8KB bank 3", m.read_byte(0x6000) == 0x43)
    m.write_byte(0x0000, 0x0A)
    m.write_byte(0x0400, 1)
    m.write_byte(0xA000, 0x5A)
    m.write_byte(0x0400, 0)
    m.write_byte(0x0400, 1)
    check("MBC6 RAM bank A round-trips", m.read_byte(0xA000) == 0x5A)


def test_mbc7_accel_and_eeprom(ns):
    MMU = ns["MMU"]
    MBC7_ACCEL_CENTER = ns["MBC7_ACCEL_CENTER"]
    MBC7_ACCEL_G = ns["MBC7_ACCEL_G"]
    m = MMU()
    m.load_rom(build_rom(cart=0x22, cgb=True))
    check("MBC7 EEPROM is 256 bytes", len(m.ram_data) == 256)
    m.write_byte(0x0000, 0x0A)
    m.write_byte(0x4000, 0x40)
    m.write_byte(0xA000, 0x55)
    m.write_byte(0xA010, 0xAA)
    x = m.read_byte(0xA020) | (m.read_byte(0xA030) << 8)
    y = m.read_byte(0xA040) | (m.read_byte(0xA050) << 8)
    check("latched X is centred", x == MBC7_ACCEL_CENTER)
    check("latched Y is centred", y == MBC7_ACCEL_CENTER)

    m.joypad_buttons = 0xFE  # Right pressed
    m.write_byte(0xA000, 0x55)
    m.write_byte(0xA010, 0xAA)
    x = m.read_byte(0xA020) | (m.read_byte(0xA030) << 8)
    check("tilting Right lowers X", x == (MBC7_ACCEL_CENTER - MBC7_ACCEL_G) & 0xFFFF)

    # EWEN then WRITE word 0xBEEF to address 0, then READ it back.
    def clock(cs_di_clk):
        m.write_byte(0xA080, cs_di_clk)

    def shift_bit(di):
        base = 0x80 | (0x02 if di else 0)
        clock(base)
        clock(base | 0x40)

    def shift_bits(value, n):
        for i in range(n - 1, -1, -1):
            shift_bit((value >> i) & 1)

    clock(0x00)
    clock(0x80)
    shift_bit(1)          # start
    shift_bits(0b0011000000, 10)  # EWEN
    clock(0x00)
    clock(0x80)
    shift_bit(1)
    shift_bits(0b0100000000, 10)  # WRITE addr 0
    shift_bits(0xBEEF, 16)
    clock(0x00)
    clock(0x80)
    shift_bit(1)
    shift_bits(0b1000000000, 10)  # READ addr 0
    clock(0x80)           # dummy 0
    clock(0xC0)
    word = 0
    for _ in range(16):
        clock(0x80)
        clock(0xC0)
        word = (word << 1) | (m.read_byte(0xA080) & 1)
    check("MBC7 EEPROM READ returns the programmed word", word == 0xBEEF)


def test_sgb_palettes_and_mlt(ns):
    MMU = ns["MMU"]
    m = MMU()
    m.load_rom(build_rom(cgb=False, sgb=True))
    check("SGB flag selects SGB mode for DMG carts", m.is_sgb)

    # PAL01: header $01, colour 0 = white (0x7FFF), pal0 col1 = red (0x001F)
    payload = bytearray(16)
    payload[0] = 0x01
    payload[1] = 0xFF
    payload[2] = 0x7F
    payload[3] = 0x1F
    payload[4] = 0x00
    _send_sgb_packet(m, payload)
    check("PAL01 sets shared colour 0 to white", m.sgb_pal_rgb[0] == 0xFFFFFF)
    check("PAL01 sets palette 0 colour 1", m.sgb_pal_rgb[1] != m.sgb_pal_rgb[0])

    payload = bytearray(16)
    payload[0] = 0x89  # MLT_REQ, length 1
    payload[1] = 0x01  # two players
    _send_sgb_packet(m, payload)
    check("MLT_REQ enables two players", m.sgb_player_count == 2)
    m.write_byte(0xFF00, 0x20)
    m.write_byte(0xFF00, 0x30)
    check("player id advances to player 2", (m.read_byte(0xFF00) & 0x0F) == 0x0E)


def main():
    ns = load_module()
    print("serial bit-clock:");      test_serial_bit_clock(ns)
    print("CGB cart wait-states:");  test_cart_wait_states(ns)
    print("MBC6 banking:");          test_mbc6_banking(ns)
    print("MBC7 accel/EEPROM:");     test_mbc7_accel_and_eeprom(ns)
    print("SGB palettes/MLT_REQ:");  test_sgb_palettes_and_mlt(ns)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
