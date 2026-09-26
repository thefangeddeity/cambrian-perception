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
# Gemini's aperture range and per-frame zoom rate: the brain can widen
# or narrow the look every frame (a zoom motor), within these bounds.
# genome.fovea_fraction is only the aperture at birth.
MIN_FRACTION = 0.15
MAX_FRACTION = 0.60  # Gemini's bound, kept for a measured reason: at 0.9 the gaze snapped wide open,
# could barely move (0.1 of travel left) and fitness collapsed 1.31 -> 0.00 -- past ~0.6 it stops being a
# gaze (a part of the field it moves around) and becomes the field itself.
ZOOM_STEP = 0.05

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
MAX_STEP = 1.0  # legacy (pre-physics); kept for reference in older docs

# Eye physics: the brain applies a FORCE; the look has velocity, with
# damping -- an eyeball (or a jumping spider's retinal tube) on muscles.
# A small steady push gives a smooth glide; a hard push reaches
# terminal speed FORCE_GAIN / (1 - DAMPING) = 0.4 of the frame per
# frame within 2-3 frames, i.e. a real saccade (~150 ms at 15 frames/s).
# Smoothness comes from the body, not from forbidding jumps.
DAMPING = 0.7
FORCE_GAIN = 0.12


@dataclass
class FoveaState:
    cx: float = 0.5  # center, normalized [0, 1] within the full frame
    cy: float = 0.5
    fraction: float = FOVEA_FRACTION  # current aperture; starts at genome.fovea_fraction, then the zoom motor moves it
    vx: float = 0.0  # look velocity (fraction of frame per frame)
    vy: float = 0.0


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


def step(state: FoveaState, pan_output: float, tilt_output: float, zoom_output: float = 0.0) -> tuple[FoveaState, float, float, float]:
    """
    Applies the brain's pan/tilt as a FORCE on a damped eye (see
    DAMPING/FORCE_GAIN above), and zoom as a bounded aperture change.
    tanh bounds the raw outputs to [-1, 1] first.

    Returns (new_state, force_x, force_y, intended_dz). Motor energy is
    charged on force (run_vision.py: force squared -- muscle cost grows
    faster than force), whether or not a wall stopped the movement; an
    isometric push against a stop still costs something.
    """
    fx = float(np.tanh(pan_output))
    fy = float(np.tanh(tilt_output))
    dz = float(np.tanh(zoom_output)) * ZOOM_STEP
    new_frac = float(np.clip(state.fraction + dz, MIN_FRACTION, MAX_FRACTION))
    half = new_frac / 2.0
    vx = DAMPING * state.vx + FORCE_GAIN * fx
    vy = DAMPING * state.vy + FORCE_GAIN * fy
    raw_cx, raw_cy = state.cx + vx, state.cy + vy
    new_cx = float(np.clip(raw_cx, half, 1.0 - half))
    new_cy = float(np.clip(raw_cy, half, 1.0 - half))
    # Hitting the edge of the frame stops motion on that axis.
    if new_cx != raw_cx:
        vx = 0.0
    if new_cy != raw_cy:
        vy = 0.0
    return FoveaState(cx=new_cx, cy=new_cy, fraction=new_frac, vx=vx, vy=vy), fx, fy, dz
