"""Print Game Boy cartridge header info for one or more ROMs.
Usage: python check_roms.py <rom> [<rom> ...]   (defaults to roms/game.gb)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
paths = sys.argv[1:] or [os.path.join(HERE, 'roms', 'game.gb')]
for path in paths:
    with open(path, 'rb') as f:
        data = f.read()
    title = data[0x134:0x144].decode('latin-1', errors='replace').rstrip(chr(0))
    print(f'File: {os.path.basename(path)}')
    print(f'  Size: {len(data)} bytes')
    print(f'  Title: {title!r}')
    print(f'  CGB flag (0x143): 0x{data[0x143]:02X}')
    print(f'  MBC type (0x147): 0x{data[0x147]:02X}')
    print(f'  ROM size (0x148): 0x{data[0x148]:02X}')
    print(f'  RAM size (0x149): 0x{data[0x149]:02X}')
    # Logo check: 48 06 0E 76 FE B3 1A 0F CE 6B B3 83 2D C1 E5 D6 C9 19 7D 07 4F 1B 7E 33 9D BE 9C D3 09 6C D2 A1 4A 9F 53 1A 5C 1B 78 20 86 E0 49 38 84 B3 1C
    expected_logo = bytes([0x48,0x06,0x0E,0x76,0xFE,0xB3,0x1A,0x0F,0xCE,0x6B,0xB3,0x83,0x2D,0xC1,0xE5,0xD6,0xC9,0x19,0x7D,0x07,0x4F,0x1B,0x7E,0x33,0x9D,0xBE,0x9C,0xD3,0x09,0x6C,0xD2,0xA1,0x4A,0x9F,0x53,0x1A,0x5C,0x1B,0x78,0x20,0x86,0xE0,0x49,0x38,0x84,0xB3,0x1C])
    print(f'  Logo check passed: {data[0x104:0x134] == expected_logo}')
    # Header checksum
    x = 0
    for b in data[0x134:0x14D]:
        x = x - b - 1
    print(f'  Header checksum: 0x{data[0x14D]:02X} (computed 0x{x & 0xFF:02X})')
    print()
