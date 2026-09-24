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
    rows = np.array_split(np.arange(h), GRID[0])
    cols = np.array_split(np.arange(w), GRID[1])

    cells = np.empty(GRID, dtype=np.float64)
    for i, row_idx in enumerate(rows):
        for j, col_idx in enumerate(cols):
            cells[i, j] = frame[np.ix_(row_idx, col_idx)].mean()

    return cells.reshape(-1)
