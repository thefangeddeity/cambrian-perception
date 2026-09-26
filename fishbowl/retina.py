from __future__ import annotations

"""
The organism's own "eye" -- turns one raw frame into a small, fixed-
length numeric vector the block-tree genome can actually use. Not a
detail this repo hides: a real eye doesn't start with millions of raw
pixels either, it starts with a coarse grid of receptors and the
downstream machinery has to make sense of THAT. A 12x12 luminance grid
is the equivalent starting point here -- few enough "variables" that
genome.py's tree search stays tractable, large enough that real
spatial structure (a region growing, a region changing) survives the
downsample.

This module never runs a self-modifying program -- it's fixed, human-
written, and stays that way. The genome's OWN machinery starts here
and has to build everything else (motion sensitivity, depth-adjacent
cues, whatever it finds) on top of what this hands it; this module
does not pre-solve any of that for it.
"""

from functools import lru_cache

import numpy as np

GRID = (12, 12)
N_CELLS = GRID[0] * GRID[1]


def frame_to_vector(gray_frame: np.ndarray) -> np.ndarray:
    """
    gray_frame: 2-D array, any real resolution, grayscale, values in
    [0, 255] or [0.0, 1.0]. Returns a flat (N_CELLS,) array in [0, 1]
    -- mean luminance of each of GRID's cells, nearest-neighbor block
    reduction (not interpolated -- cheap, and precision beyond a 12x12
    grid isn't the point here).
    """
    frame = gray_frame.astype(np.float64)
    if frame.max() > 1.5:
        frame = frame / 255.0

    h, w = frame.shape
    row_starts, col_starts, counts = _cell_layout(h, w)
    sums = np.add.reduceat(np.add.reduceat(frame, row_starts, axis=0), col_starts, axis=1)
    return (sums / counts).reshape(-1)


@lru_cache(maxsize=256)
def _cell_layout(h: int, w: int):
    # Same cell boundaries np.array_split would produce, so reduceat
    # sums give results identical to per-cell mean() -- just without
    # 144 separate calls (this runs several times per frame per genome).
    row_sizes = np.array([len(a) for a in np.array_split(np.arange(h), GRID[0])])
    col_sizes = np.array([len(a) for a in np.array_split(np.arange(w), GRID[1])])
    row_starts = np.concatenate([[0], np.cumsum(row_sizes)[:-1]])
    col_starts = np.concatenate([[0], np.cumsum(col_sizes)[:-1]])
    return row_starts, col_starts, np.outer(row_sizes, col_sizes)


def opponent_planes(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Colour-opponent channels, the way real colour vision encodes colour
    after the photoreceptors: red-green (R - G) and blue-yellow
    (B - (R + G) / 2), each mapped to [0, 1] with 0.5 = neutral. Only the
    gaze gets these -- like a jumping spider, whose principal eyes see
    colour while its wide-field secondary eyes are monochrome.
    """
    f = bgr.astype(np.float64) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    rg = 0.5 + 0.5 * (r - g)
    by = 0.5 + 0.5 * (b - 0.5 * (r + g))
    return rg, by
