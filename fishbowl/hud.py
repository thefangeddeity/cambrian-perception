from __future__ import annotations

"""
The organism's target lock, burned into a host's picture (the viewer's
client HUD, in Python): corner brackets on its gaze -- cyan SCAN, yellow
TRACK (prey in its gaze), red LOCK (eating: prey held in its gaze centre,
the snapshot moment) -- that glide between its looks; pink corners on the
living things the host's detector found; the mode and what it is on at the
top right. When it locks on, the brackets snap in from the picture's edges
onto its gaze with a soft flash, at most once a second.

Text height follows the fleet's HUD rule where it is drawn (the host passes
its cap height; by default e% of the picture's height, the livecam's rule).
"""

import math

import cv2
import numpy as np

COLOURS = {"SCAN": (127, 212, 255), "TRACK": (255, 221, 68), "LOCK": (255, 77, 109)}  # RGB
PREY_COLOUR = (255, 95, 162)
SNAP_S, SNAP_EVERY_S, FLASH_ALPHA = 0.28, 1.0, 0.12  # the viewer's client snap, in seconds
GLIDE_PER_S = 0.0005  # the viewer's glide: the share of the gap left after a second


def _ink(rgb, order):
    return tuple(int(c) for c in (rgb if order == "rgb" else rgb[::-1]))


def _corners(img, x0, y0, x1, y1, arm, colour, thickness):
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    arm = max(2, int(round(arm)))
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), colour, thickness, cv2.LINE_AA)


def _text(img, text, x_right, y, scale, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(x_right - tw)
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return th


class LockHUD:
    """draw(canvas, state) each displayed frame; state from fishbowl.live.LiveActor."""

    def __init__(self, order: str = "rgb", cap_fraction: float = math.e / 100.0):
        self.order, self.cap_fraction = order, cap_fraction
        self.rx = self.ry = self.rs = None
        self.last_t = None
        self.mode = None
        self.snap_t = -1e9

    def draw(self, canvas: np.ndarray, state: dict | None, labels: dict | None = None) -> np.ndarray:
        if not state:
            return canvas
        now = cv2.getTickCount() / cv2.getTickFrequency()
        dt = min(1.0, now - self.last_t) if self.last_t is not None else 0.0
        self.last_t = now
        h, w = canvas.shape[:2]
        mode = state["mode"]
        colour = _ink(COLOURS[mode], self.order)
        # Glide between its looks (its eye moves between them too).
        k = 1.0 - GLIDE_PER_S ** dt if dt > 0 else 1.0
        self.rx = state["cx"] if self.rx is None else self.rx + (state["cx"] - self.rx) * k
        self.ry = state["cy"] if self.ry is None else self.ry + (state["cy"] - self.ry) * k
        self.rs = state["extent"] if self.rs is None else self.rs + (state["extent"] - self.rs) * k
        side = self.rs * h  # square: its side is a fraction of the picture's height
        x, y = self.rx * w, self.ry * h
        thin = max(1, int(round(h / 480)))
        for b in state.get("boxes") or []:
            _corners(canvas, b[2] * w, b[3] * h, b[4] * w, b[5] * h, min((b[4] - b[2]) * w, (b[5] - b[3]) * h) * 0.2,
                     _ink(PREY_COLOUR, self.order), thin)
        _corners(canvas, x - side / 2, y - side / 2, x + side / 2, y + side / 2, side * 0.16, colour, 2 * thin)
        # The snap: when it locks on, brackets close in from the picture's edges.
        if mode == "LOCK" and self.mode != "LOCK" and now - self.snap_t > SNAP_EVERY_S:
            self.snap_t = now
        age = now - self.snap_t
        if age < SNAP_S:
            e = 1.0 - (1.0 - age / SNAP_S) ** 3
            X0, Y0 = (x - side / 2) * e, (y - side / 2) * e
            X1, Y1 = w + (x + side / 2 - w) * e, h + (y + side / 2 - h) * e
            _corners(canvas, X0, Y0, X1, Y1, side * 0.2, colour, 3 * thin)
            if age < SNAP_S / 2:
                flash = np.full_like(canvas, 255)
                a = FLASH_ALPHA * (1.0 - age / (SNAP_S / 2))
                cv2.addWeighted(flash, a, canvas, 1.0 - a, 0.0, dst=canvas)
        self.mode = mode
        # Readout, top right: mode + what it is on, then its meals.
        cap = self.cap_fraction * h
        scale = cap / 27.0  # FONT_HERSHEY_SIMPLEX's cap height at scale 1 (measured)
        t = state.get("target")
        name = (labels or {}).get(t[0], t[0]) if t else None
        line1 = mode + (f" {str(name).upper()} {t[1] * 100:.0f}%" if t else "")
        crit = state.get("criterion_s")
        line2 = f"meals {state.get('meals', 0)}" if crit is not None else "meals calibrating"
        top = int(round(2.0 * cap))
        right = w - int(round(1.64 * cap))
        _text(canvas, line1, right, top, scale, colour)
        _text(canvas, line2, right, top + int(round(1.8 * cap)), scale * 0.85, _ink((255, 159, 184), self.order))
        return canvas
