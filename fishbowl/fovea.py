from __future__ import annotations

"""
Digital pan/tilt -- there is no motorized camera, so a small window
of the camera frame (the fovea / gaze) is what it sees, and moving that
window replicates pan/tilt digitally. Same shape as foveated/active-vision models in
both biology (the eye doesn't process its whole visual field at high
resolution at once) and machine learning (Mnih et al.'s "Recurrent
Models of Visual Attention" -- a small glimpse window, and CHOOSING
where to look next is itself a real, trainable action). The brain
(controller.py) chooses; this module only applies its force to a damped
eye -- it never decides WHERE to look.

The gaze's CENTER can reach any point of the frame, edges and corners
included, like an eye that can point its fovea at anything in view: a cat
walking along the edge of the room can be followed and eaten. Whatever
part of the gaze hangs past the frame sees nothing (black; neutral for
colour) -- no light arrives from outside the world, so it feeds nothing
and still costs its aperture.
"""

from dataclasses import dataclass

import numpy as np

from .retina import GRID, frame_to_vector, opponent_planes

FOVEA_FRACTION = 0.35  # DEFAULT look size (fraction of the full
# frame's width/height) for a newborn genome. The live value is a
# heritable trait, genome.fovea_fraction, evolved within the bounds
# below and priced by real compute scarcity (see run_vision.py's
# field cost): it can grow when seeing more pays off and resources
# allow, and shrinks when resources are scarce.
# Gemini's aperture range and per-frame zoom rate: the brain can widen
# or narrow the look every frame (a zoom motor), within these bounds.
# genome.fovea_fraction is only the aperture at birth.
MIN_FRACTION = 0.15
MAX_FRACTION = 0.60  # Gemini's bound, kept for a measured reason: at 0.9 the gaze snapped wide open,
# could barely move (0.1 of travel left) and fitness collapsed 1.31 -> 0.00 -- past ~0.6 it stops being a
# gaze (a part of the field it moves around) and becomes the field itself.
ZOOM_STEP = 0.05

# Eye physics: the brain applies a FORCE; the look has velocity, with
# damping -- an eyeball (or a jumping spider's retinal tube) on muscles.
# A small steady push gives a smooth glide; a hard push reaches
# terminal speed FORCE_GAIN / (1 - DAMPING) = 0.4 of the frame per
# frame within 2-3 frames, i.e. a real saccade (~150 ms at 15 frames/s).
# Smoothness comes from the body, not from forbidding jumps.
DAMPING = 0.5
FORCE_GAIN = 0.2
# The eye sits in elastic tissue that pulls it back toward straight ahead
# (the oculomotor plant, Robinson): relaxed, the gaze returns to the centre;
# holding it off-centre takes sustained force, which the body pays for
# (force squared). Without this the eye integrated force, so any steady
# push, however small, ran it into a wall -- a fresh brain's default was to
# hug an edge. The plant is overdamped, like a real eye's (tissue viscosity):
# released, it creeps back to centre in ~0.7 s without overshooting. An
# earlier underdamped tuning rang like a bell (5 overshoots), and a fresh
# brain learned to drive that resonance into free circling -- a physics
# quirk, not a strategy a real body could use; scanning that pays has to be
# driven, and paid for. Balance (design panel): terminal saccade speed 0.4
# of the frame per frame as before; holding the edge takes 1/5 of full force
# (a few % of waking burn -- cheap next to food, but a real choice).
SPRING = 0.08


@dataclass
class FoveaState:
    cx: float = 0.5  # center, normalized [0, 1] within the full frame
    cy: float = 0.5
    fraction: float = FOVEA_FRACTION  # current aperture; starts at genome.fovea_fraction, then the zoom motor moves it
    vx: float = 0.0  # look velocity (fraction of frame per frame)
    vy: float = 0.0


def _window(frame: np.ndarray, state: FoveaState) -> np.ndarray:
    """The gaze window, centered on (cx, cy) at its real pixel size; any
    part past the frame's edge is zero (black: no light from there)."""
    h, w = frame.shape[:2]
    half_w = max(1, int(w * state.fraction / 2))
    half_h = max(1, int(h * state.fraction / 2))
    x0, y0 = int(state.cx * w) - half_w, int(state.cy * h) - half_h
    out = np.zeros((2 * half_h, 2 * half_w) + frame.shape[2:], dtype=frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + 2 * half_w), min(h, y0 + 2 * half_h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def extract(full_frame_gray: np.ndarray, state: FoveaState) -> np.ndarray:
    """Crops the current gaze window out of the real full frame and returns its retina.py-style flat grid vector."""
    return frame_to_vector(_window(full_frame_gray, state))


def extract_colour(full_frame_bgr: np.ndarray, state: FoveaState, channels: int) -> np.ndarray:
    """The gaze's colour receptors: `channels` opponent grids (0, 1 = red-green,
    2 = + blue-yellow) over the same crop as extract(), flattened; empty if 0."""
    if channels <= 0 or full_frame_bgr is None:
        return np.zeros(0)
    rg, by = opponent_planes(_window(full_frame_bgr, state))  # black -> neutral (0.5)
    planes = [rg, by][:channels]
    return np.concatenate([frame_to_vector(pl) for pl in planes])


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
    # The brain's outputs are already tanh-bounded; clip only (audit: a
    # second tanh here capped force at tanh(1) = 0.76).
    fx = float(np.clip(pan_output, -1.0, 1.0))
    fy = float(np.clip(tilt_output, -1.0, 1.0))
    dz = float(np.clip(zoom_output, -1.0, 1.0)) * ZOOM_STEP
    new_frac = float(np.clip(state.fraction + dz, MIN_FRACTION, MAX_FRACTION))
    vx = DAMPING * state.vx + FORCE_GAIN * fx - SPRING * (state.cx - 0.5)
    vy = DAMPING * state.vy + FORCE_GAIN * fy - SPRING * (state.cy - 0.5)
    raw_cx, raw_cy = state.cx + vx, state.cy + vy
    new_cx = float(np.clip(raw_cx, 0.0, 1.0))
    new_cy = float(np.clip(raw_cy, 0.0, 1.0))
    # The center reaching the edge of the frame stops motion on that axis.
    if new_cx != raw_cx:
        vx = 0.0
    if new_cy != raw_cy:
        vy = 0.0
    return FoveaState(cx=new_cx, cy=new_cy, fraction=new_frac, vx=vx, vy=vy), fx, fy, dz
