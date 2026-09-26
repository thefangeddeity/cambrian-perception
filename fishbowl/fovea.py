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

FOVEA_FRACTION = 0.35  # DEFAULT look size (fraction of the full
# frame's width/height) for a newborn genome. The live value is a
# heritable trait, genome.fovea_fraction, evolved within the bounds
# below and priced by real compute scarcity (see run_vision.py's
# field cost) -- User: "grow its visual field as curiosity wants and
# resources allow, but shrink as resource hunger limits it."
MIN_FRACTION = 0.10  # 2*int(179*0.10/2)=16 rows, still >= retina's 12
MAX_FRACTION = 0.90  # below 1.0 so the look can still move at all

# Real correction, User: "Curiosity and large saccades should evolve,
# not be forced." MAX_STEP used to be 0.12 -- small enough that NO
# possible pan/tilt tree output could ever produce a real saccade (a
# large, fast, ballistic jump, as opposed to smooth pursuit -- both
# real eye-movement modes, which one gets used is real behavior, not
# something to hand-pick). Because step() below already squashes the
# raw tree output through tanh before scaling by MAX_STEP, the OLD
# small constant meant every possible genome, no matter how it
# evolved, was structurally capped at the same tiny step -- the
# genome had no path to ever discover large movements, regardless of
# whether that would have been fitness-beneficial. Raised to exceed
# the full reachable range in one step (reachable width is 1 -
# FOVEA_FRACTION); the REAL constraint that remains is step()'s own
# final clip to stay inside the frame -- a genuine physical
# necessity, not a behavioral-style choice. Whether movement ends up
# smooth-small or saccade-large is now something the pan/tilt trees'
# OWN evolved output magnitude actually determines.
MAX_STEP = 1.0


@dataclass
class FoveaState:
    cx: float = 0.5  # center, normalized [0, 1] within the full frame
    cy: float = 0.5
    fraction: float = FOVEA_FRACTION  # fixed for a genome's lifetime, set from genome.fovea_fraction


def extract(full_frame_gray: np.ndarray, state: FoveaState) -> np.ndarray:
    """Crops the current fovea window out of the real full frame and returns its retina.py-style flat grid vector."""
    h, w = full_frame_gray.shape
    half_w = int(w * state.fraction / 2)
    half_h = int(h * state.fraction / 2)
    px = int(state.cx * w)
    py = int(state.cy * h)

    x0 = np.clip(px - half_w, 0, w - 2 * half_w)
    y0 = np.clip(py - half_h, 0, h - 2 * half_h)
    window = full_frame_gray[y0:y0 + 2 * half_h, x0:x0 + 2 * half_w]
    return frame_to_vector(window)


def step(state: FoveaState, pan_output: float, tilt_output: float) -> tuple[FoveaState, float, float]:
    """
    Applies the genome's own raw pan/tilt tree output as a bounded
    move. tanh squashes an unbounded tree output (blocks.py's Node.
    evaluate can return values up to +-MAX_CONST*1e3 in extreme cases)
    into [-1, 1] BEFORE scaling by MAX_STEP -- so a wildly-valued
    output saturates to "move as far as allowed this frame," never
    further, rather than being clamped in a way that makes large and
    huge outputs indistinguishable in some other, less predictable way.

    Returns (new_state, intended_dx, intended_dy) -- intended_dx/dy are
    the PRE-CLAMP step (what the tree actually tried to do), for
    run_vision.py's movement-cost penalty. Real motor effort isn't
    zero just because a wall stopped the actual displacement -- an
    isometric push against a stop still costs something -- so cost is
    measured on INTENT, computed once here (the one place this math
    already lives), not re-derived from the realized post-clamp
    position change tracked separately for the pursuit reward and
    motor-efference feedback.
    """
    dx = float(np.tanh(pan_output)) * MAX_STEP
    dy = float(np.tanh(tilt_output)) * MAX_STEP
    half = state.fraction / 2.0
    new_cx = float(np.clip(state.cx + dx, half, 1.0 - half))
    new_cy = float(np.clip(state.cy + dy, half, 1.0 - half))
    return FoveaState(cx=new_cx, cy=new_cy, fraction=state.fraction), dx, dy
