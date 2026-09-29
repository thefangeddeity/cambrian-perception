from __future__ import annotations

"""
The whole visual field's signals, frame by frame -- for a host that lives
live (a livecam's CV loop) instead of scoring a recorded snapshot. The same
numbers the evolution loop computes for a whole snapshot at once
(run_vision.World: reflexes.expansion_score, reflexes.mismatch_score, reflexes.motion_energy_score,
organism.peripheral_motion_centroid, the field's mean light and
organism.global_shift), computed incrementally, so a live organism feels its
world exactly as the one that evolved did (tested equal, frame for frame).
"""

import cv2
import numpy as np

from . import organism as org
from . import reflexes
from .retina import field_shape, frame_to_vector

EXPANSION_THRESHOLD, EXPANSION_ADAPT = 0.08, 0.05  # reflexes.expansion_score's defaults


class FieldSignals:
    """Feed it each grey frame (the organism's resolution) with step(); it
    returns that frame's whole-field signals (organism.SIGNAL_KEYS) and the
    frame's global shift since the previous one (for the stabilizer)."""

    def __init__(self):
        self.prev = None           # previous frame's whole-field receptors
        self.background = None     # the looming detector's slow background
        self.prev_area = self.prev_growth = 0.0
        self.t = 0
        self.prev_small = None
        self.expansion = True
        self._last_scale = 0.0
        self._last_roll = 0.0  # its expansion measurement (shed first among its senses when short of oxygen)
        self.window = None
        self.last_vector = None
        self.fps = 15.0            # the host sets its frame rate (the mismatch background adapts in seconds)
        self.structure = None      # the mismatch detector's slow model of the field's structure (mean, variance)

    def step(self, grey: np.ndarray) -> tuple[dict, tuple[float, float]]:
        shape = field_shape(*grey.shape[:2])
        v = frame_to_vector(grey, shape)
        self.last_vector = v
        # Looming (reflexes.expansion_score, streamed).
        if self.background is None:
            self.background = v.astype(np.float64).copy()
        area = float(((self.background - v) > EXPANSION_THRESHOLD).mean())
        growth = area - self.prev_area
        expansion = growth if (self.t >= 2 and growth > 0 and self.prev_growth > 0) else 0.0
        self.prev_area, self.prev_growth = area, growth
        self.background += EXPANSION_ADAPT * (v - self.background)
        # Movement and where it is (motion_energy_score, peripheral_motion_centroid).
        motion, mcx, mcy = 0.0, 0.5, 0.5
        motion_map = np.zeros(v.shape)
        if self.prev is not None:
            d = np.abs(v - self.prev)
            motion_map = np.clip(d * org.PERIPH_MOTION_GAIN, 0.0, 1.0)
            motion = float(d.mean())
            grid = d.reshape(shape)
            tot = grid.sum()
            if tot > 1e-9:
                rows, cols = shape
                yy, xx = np.mgrid[0:rows, 0:cols]
                mcx = float(((xx + 0.5) * grid).sum() / tot / cols)
                mcy = float(((yy + 0.5) * grid).sum() / tot / rows)
        self.prev = v
        # Mismatch with its slow model of the room (reflexes.mismatch_step).
        alpha = 1.0 - float(np.exp(-1.0 / max(1e-6, self.fps) / org.MISMATCH_TAU_S))
        mismatch, mmx, mmy, self.structure, mismatch_map = reflexes.mismatch_step(self.structure, v, shape, alpha, org.SURPRISE_SIGMAS, org.NOISE_FLOOR)
        # The frame's global shift (organism.global_shifts, streamed).
        size = org.shift_size(grey.shape)
        small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        shift = (0.0, 0.0)
        votes = []
        if self.prev_small is not None and self.prev_small.shape == small.shape:
            if self.window is None or self.window.shape != small.shape:
                self.window = cv2.createHanningWindow(size, cv2.CV_32F)
            shift = org.global_shift(self.prev_small.copy(), small.copy(), self.window)  # copies: phase correlation windows its inputs in place
            rl = []
            raw = org.global_scale(self.prev_small, small, None, rl, votes) if self.expansion else 0.0
            roll_raw = rl[0] if rl else 0.0
            shift = (shift[0], shift[1], org.replicated(raw, self._last_scale), org.replicated(roll_raw, self._last_roll))
            self._last_scale, self._last_roll = raw, roll_raw
        parallax = org.parallax_map(self.prev_small, small, shift, shape)
        self.prev_small = small
        self.t += 1
        sig = {"expansion": expansion, "motion_energy": motion, "motion_cx": mcx, "motion_cy": mcy,
               "field_light": float(v.mean()), "mismatch": mismatch, "mismatch_cx": mmx, "mismatch_cy": mmy,
               "structure": v - float(v.mean()), "parallax": parallax, "motion_map": motion_map, "mismatch_map": mismatch_map,
               "frame_votes": votes[0] if votes else None}
        return sig, shift
