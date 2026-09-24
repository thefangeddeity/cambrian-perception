from __future__ import annotations

"""
Digital pan/tilt -- the user's own solution to there being no real
motorized camera: "we take a small frame of Tanzania's feed, a fovea,
and that's what it sees, so we replicate pan/tilt digitally. So the
reflexes can train." Same shape as foveated/active-vision models in
both biology (the eye doesn't process its whole visual field at high
resolution at once) and machine learning (Mnih et al.'s "Recurrent
Models of Visual Attention" -- a small glimpse window, and CHOOSING
where to look next is itself a real, trainable action). The genome's
own "pan"/"tilt" output trees (see genome.py's DEFAULT_CHANNELS)
control that choice; this module only ever applies a bounded step and
clamps the result inside the real frame -- it never decides WHERE to
look, only enforces that wherever the genome decides, it can't step
further than max_step per frame or off the edge of the real image.
"""

from dataclasses import dataclass

import numpy as np

from .retina import GRID, frame_to_vector

FOVEA_FRACTION = 0.35  # fovea window is this fraction of the full
# frame's width/height -- big enough to hold real content, small
# enough that covering the whole scene actually requires moving.
MAX_STEP = 0.12  # max fraction-of-frame the center can move per frame


@dataclass
class FoveaState:
    cx: float = 0.5  # center, normalized [0, 1] within the full frame
    cy: float = 0.5


def extract(full_frame_gray: np.ndarray, state: FoveaState) -> np.ndarray:
    """Crops the current fovea window out of the real full frame and returns its retina.py-style flat grid vector."""
    h, w = full_frame_gray.shape
    half_w = int(w * FOVEA_FRACTION / 2)
    half_h = int(h * FOVEA_FRACTION / 2)
    px = int(state.cx * w)
    py = int(state.cy * h)

    x0 = np.clip(px - half_w, 0, w - 2 * half_w)
    y0 = np.clip(py - half_h, 0, h - 2 * half_h)
    window = full_frame_gray[y0:y0 + 2 * half_h, x0:x0 + 2 * half_w]
    return frame_to_vector(window)


def step(state: FoveaState, pan_output: float, tilt_output: float) -> FoveaState:
    """
    Applies the genome's own raw pan/tilt tree output as a bounded
    move. tanh squashes an unbounded tree output (blocks.py's Node.
    evaluate can return values up to +-MAX_CONST*1e3 in extreme cases)
    into [-1, 1] BEFORE scaling by MAX_STEP -- so a wildly-valued
    output saturates to "move as far as allowed this frame," never
    further, rather than being clamped in a way that makes large and
    huge outputs indistinguishable in some other, less predictable way.
    """
    dx = float(np.tanh(pan_output)) * MAX_STEP
    dy = float(np.tanh(tilt_output)) * MAX_STEP
    half = FOVEA_FRACTION / 2.0
    new_cx = float(np.clip(state.cx + dx, half, 1.0 - half))
    new_cy = float(np.clip(state.cy + dy, half, 1.0 - half))
    return FoveaState(cx=new_cx, cy=new_cy)
