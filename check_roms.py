"""Print Game Boy cartridge header info for one or more ROMs.
Usage: python check_roms.py <rom> [<rom> ...]   (defaults to roms/game.gb)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
paths = sys.argv[1:] or [os.path.join(HERE, 'roms', 'game.gb')]
_NINTENDO_LOGO = bytes([
    0x48, 0x06, 0x0E, 0x76, 0xFE, 0xB3, 0x1A, 0x0F, 0xCE, 0x6B, 0xB3, 0x83,
    0x2D, 0xC1, 0xE5, 0xD6, 0xC9, 0x19, 0x7D, 0x07, 0x4F, 0x1B, 0x7E, 0x33,
    0x9D, 0xBE, 0x9C, 0xD3, 0x09, 0x6C, 0xD2, 0xA1, 0x4A, 0x9F, 0x53, 0x1A,
    0x5C, 0x1B, 0x78, 0x20, 0x86, 0xE0, 0x49, 0x38, 0x84, 0xB3, 0x1C,
])
for path in paths:
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError as e:
        print(f'Error reading {path}: {e}', file=sys.stderr)
        sys.exit(1)
    title = data[0x134:0x144].decode('latin-1', errors='replace').rstrip(chr(0))
    print(f'File: {os.path.basename(path)}')
    print(f'  Size: {len(data)} bytes')
    print(f'  Title: {title!r}')
    print(f'  CGB flag (0x143): 0x{data[0x143]:02X}')
    print(f'  MBC type (0x147): 0x{data[0x147]:02X}')
    print(f'  ROM size (0x148): 0x{data[0x148]:02X}')
    print(f'  RAM size (0x149): 0x{data[0x149]:02X}')
    print(f'  Logo check passed: {data[0x104:0x104 + len(_NINTENDO_LOGO)] == _NINTENDO_LOGO}')
    if len(data) >= 0x150:
        chk = 0
        for b in data[0x134:0x14D]:
            chk = (chk - b - 1) & 0xFF
        print(f'  Header checksum: 0x{data[0x14D]:02X} (computed 0x{chk:02X})')
    print()
