"""Benchmark the per-frame framebuffer -> host-surface conversion.

Compares the old list-of-RGB-tuples representation against the current
packed-24-bit-integer framebuffer.  The packed framebuffer keeps the PPU
scanline writers at one assignment per pixel while letting the display path
unpack the whole frame with a vectorised numpy shift instead of having numpy
iterate 23 040 Python tuples every frame.

Run:  python profile_display.py
"""
import timeit

import numpy as np

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
N_PIXELS = SCREEN_WIDTH * SCREEN_HEIGHT

# A representative frame: a handful of distinct colours spread across pixels.
_palette = [
    (224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32),
    (200, 230, 255), (40, 80, 160), (255, 245, 200), (60, 40, 10),
]
fb_tuples = [_palette[(i * 7 + (i >> 4)) % len(_palette)] for i in range(N_PIXELS)]
fb_packed = [(r << 16) | (g << 8) | b for (r, g, b) in fb_tuples]

# Persistent destination buffer, matching GameBoy.render().
frame_np = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)


def convert_tuples():
    """Old path: numpy iterates a list of (r, g, b) tuples."""
    np.copyto(frame_np, np.asarray(fb_tuples, dtype=np.uint8).reshape(
        SCREEN_HEIGHT, SCREEN_WIDTH, 3))
    return frame_np


def convert_packed():
    """Current path: vectorised unpack of one packed int per pixel."""
    packed = np.asarray(fb_packed, dtype=np.uint32).reshape(
        SCREEN_HEIGHT, SCREEN_WIDTH)
    frame_np[:, :, 0] = (packed >> 16) & 0xFF
    frame_np[:, :, 1] = (packed >> 8) & 0xFF
    frame_np[:, :, 2] = packed & 0xFF
    return frame_np


def _check_equivalent():
    a = convert_tuples().copy()
    b = convert_packed().copy()
    assert np.array_equal(a, b), "packed unpack does not match tuple conversion"


if __name__ == "__main__":
    _check_equivalent()
    runs = 2000
    t_tuples = timeit.timeit(convert_tuples, number=runs) / runs * 1000
    t_packed = timeit.timeit(convert_packed, number=runs) / runs * 1000
    print(f"list-of-tuples conversion : {t_tuples:.3f} ms/frame")
    print(f"packed-int unpack         : {t_packed:.3f} ms/frame")
    print(f"speedup                   : {t_tuples / t_packed:.2f}x")
    print(f"saved per second @60fps   : {(t_tuples - t_packed) * 60:.1f} ms/s")
