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
and its receptors still cost their upkeep.
"""

from dataclasses import dataclass

import numpy as np

from .retina import frame_to_vector, opponent_planes

# The gaze is a square patch of n x n square receptors of a FIXED size (after
# a design panel: Land, Nilsson -- receptor mosaics are fixed; what evolves is
# how many receptors there are). n is inherited (genome.receptors) and grows
# or shrinks a ring at a time, so the centre stays the centre and every
# receptor keeps its position relative to it (the perception tree reads them
# by that position). Growing the eye is real growth: more receptors, more
# detail over more area, each one paid for (organism.RECEPTOR_COST). There is
# no zoom: nothing stretches a fixed set of receptors over a bigger patch (the
# brain's old zoom output is unused -- the slot a pupil could take later).
#
# Receptor size: 1/64 of the frame's height (square in pixels). Chosen when the
# zoom eye was retired so that no living lineage lost a receptor across its
# gaze (the narrowest, 0.185 of the frame, keeps its 12), and still ~2.8 of
# the processed frame's pixels per receptor (video_source keeps 320 px wide,
# 180 rows at 16:9).
RECEPTOR_PITCH = 1.0 / 64
DEFAULT_RECEPTORS = 22   # a newborn's eye: 22 x 22 (0.34 of the frame's height; the old default gaze was 0.35)
MIN_RECEPTORS = 4        # the smallest eye with a centre and a ring around it
MAX_RECEPTORS = 38       # 0.59 of the frame: past ~0.6 the gaze stops being a gaze and becomes the field
# (measured under the zoom eye: at 0.9 it snapped wide open, could barely move and fitness collapsed)


def even_receptors(n: float) -> int:
    """The nearest allowed receptor count: even (growth is by whole rings), within bounds."""
    return int(min(MAX_RECEPTORS, max(MIN_RECEPTORS, 2 * round(n / 2.0))))


def extent(n: int) -> float:
    """The gaze's side as a fraction of the frame's height."""
    return n * RECEPTOR_PITCH


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
    n: int = DEFAULT_RECEPTORS  # receptors per side (its genome's; fixed for a life)
    vx: float = 0.0  # look velocity (fraction of frame per frame)
    vy: float = 0.0

    @property
    def extent(self) -> float:
        """The gaze's side, as a fraction of the frame's height."""
        return extent(self.n)

    def half_extents(self, aspect: float) -> tuple[float, float]:
        """Half the gaze's width and height as fractions of the frame's width
        and height (aspect = frame width / height): square in pixels."""
        e = self.extent / 2.0
        return e / aspect, e


def _window(frame: np.ndarray, state: FoveaState) -> np.ndarray:
    """The gaze window, centered on (cx, cy): a square of n receptors of
    RECEPTOR_PITCH of the frame's height (at least a pixel each); any part
    past the frame's edge is zero (black: no light from there)."""
    h, w = frame.shape[:2]
    side = max(state.n, int(round(state.extent * h)))
    x0, y0 = int(state.cx * w) - side // 2, int(state.cy * h) - side // 2
    out = np.zeros((side, side) + frame.shape[2:], dtype=frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def extract(full_frame_gray: np.ndarray, state: FoveaState) -> np.ndarray:
    """The gaze's light receptors on the real full frame: a flat (n * n,) vector."""
    return frame_to_vector(_window(full_frame_gray, state), (state.n, state.n))


def cone_mask(n: int, cones: int) -> np.ndarray:
    """Which of the n x n receptors are cones: the central cones x cones (the
    rest are rods)."""
    m = np.zeros((n, n), dtype=bool)
    lo = (n - min(cones, n)) // 2
    m[lo:lo + min(cones, n), lo:lo + min(cones, n)] = True
    return m


def extract_colour(full_frame_bgr: np.ndarray, state: FoveaState, channels: int, cones: int | None = None) -> np.ndarray:
    """The gaze's colour receptors: `channels` opponent grids (0, 1 = red-green,
    2 = + blue-yellow) over the same receptors as extract(), flattened; empty if
    0. Only cones see colour: outside the central cones x cones patch (rods)
    the colour planes read 0, as a receptor it doesn't have."""
    if channels <= 0 or full_frame_bgr is None:
        return np.zeros(0)
    rg, by = opponent_planes(_window(full_frame_bgr, state))  # black -> neutral (0.5)
    planes = [rg, by][:channels]
    mask = cone_mask(state.n, state.n if cones is None else cones).reshape(-1)
    return np.concatenate([np.where(mask, frame_to_vector(pl, (state.n, state.n)), 0.0) for pl in planes])


def step(state: FoveaState, pan_output: float, tilt_output: float) -> tuple[FoveaState, float, float]:
    """
    Applies the brain's pan/tilt as a FORCE on a damped eye (see
    DAMPING/FORCE_GAIN above). The outputs are clipped to [-1, 1].

    Returns (new_state, force_x, force_y). Motor energy is
    charged on force (run_vision.py: force squared -- muscle cost grows
    faster than force), whether or not a wall stopped the movement; an
    isometric push against a stop still costs something.
    """
    # The brain's outputs are already tanh-bounded; clip only (audit: a
    # second tanh here capped force at tanh(1) = 0.76).
    fx = float(np.clip(pan_output, -1.0, 1.0))
    fy = float(np.clip(tilt_output, -1.0, 1.0))
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
    return FoveaState(cx=new_cx, cy=new_cy, n=state.n, vx=vx, vy=vy), fx, fy
