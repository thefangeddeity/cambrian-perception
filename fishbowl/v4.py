"""
Its V4 (a 2026-09-29 panel: Zeki, Conway, Roe, Pasupathy & Connor, Freeman &
Simoncelli, Gibson, Land, Nilsson, Dawkins, Sterling & Laughlin): the ventral
stream's middle stage, between its edges (receptors, pools, oriented pools)
and its objects (cortex.py). Functions, not primate anatomy -- bees keep
colour constant and an insect's lobula builds shape -- each an evolvable,
priced trait:

  - colour constancy (V4's colour domains): each cone channel's gain adapts to
    what its own cones have seen (von Kries 1902; a receptor's gain set by its
    time-averaged input), over its inherited adaptation time (genome
    colour_constancy, seconds; 0 = none; seeded at Fairchild & Reniff 1995's
    human time course: ~95% adapted in 60 s, tau = 20 s). Luminance is kept:
    each channel's gain is the illuminant's mean over that channel's. Its wide
    field is monochrome, so its surround is what its cones saw, in time.
  - texture statistics (V4's texture sensitivity; Freeman & Simoncelli): per
    whole-field cell, contrast (the intensity's standard deviation), fineness
    (the dominant spatial frequency, from the spectrum's second moment:
    f^2 = E|grad I|^2 / (4 pi^2 Var I), Rice 1944; cycles per frame height)
    and anisotropy (the gradient structure tensor's coherence: 0 isotropic,
    1 one orientation). A cell under the sensor-noise floor has no texture.
  - the texture gradient (Gibson 1950): on ground of one texture, what is
    finer is farther -- fineness grows as depth (1 - e) / (y - horizon) -- so
    each ground cell reads how its ground is raised, against the median of
    the frame's ground cells through its map (terrain_lessons).

Curvature (V4's curvature domains; Pasupathy & Connor) lives in the
perception tree's leaves (blocks.py: an oriented pool can bend).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .retina import frame_to_vector

COLOUR_ADAPT_SEED_S = 20.0  # Fairchild & Reniff 1995: ~95% adapted in 60 s (3 tau)
TEXTURE_FEATURES = ("contrast", "fineness", "anisotropy")
MIN_GROUND_CELLS = 8        # as its ego-motion's fit: fewer and a median says little


class ColourAdaptation:
    """Its cones' von Kries gains, adapting to what they have seen."""

    def __init__(self, tau_s: float):
        self.tau_s = float(tau_s)
        self.illuminant = None  # B, G, R: the time-averaged mean its cones saw

    def see(self, cone_bgr_mean: np.ndarray, seconds: float) -> None:
        m = np.asarray(cone_bgr_mean, dtype=float)
        if not np.all(np.isfinite(m)) or m.sum() <= 0.0:
            return  # nothing lit its cones (black, or past the frame): nothing to adapt to
        if self.illuminant is None:
            self.illuminant = m.copy()  # its first look: adapted to it
            return
        a = 1.0 - math.exp(-max(0.0, seconds) / max(1e-6, self.tau_s))
        self.illuminant += a * (m - self.illuminant)

    def gains(self) -> np.ndarray | None:
        """Per channel (B, G, R): the illuminant's mean over that channel's; None before it has seen anything."""
        if self.illuminant is None or np.any(self.illuminant <= 1e-6):
            return None
        return float(self.illuminant.mean()) / self.illuminant


def cone_mean(window_bgr: np.ndarray, n: int, cones: int) -> np.ndarray:
    """The mean B, G, R its cones see: each channel pooled into its n x n
    receptors (as its eye pools light), then averaged over the central cones x cones."""
    lo, c = (n - min(cones, n)) // 2, min(cones, n)
    if c <= 0:
        return np.zeros(3)
    grid = cv2.resize(window_bgr.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)  # all three channels on one scale
    return grid[lo:lo + c, lo:lo + c].reshape(-1, 3).mean(axis=0).astype(float)


def grey_world(bgr: np.ndarray) -> np.ndarray:
    """A frame with its colour cast removed (Buchsbaum 1980's grey world: each
    channel scaled so the frame's mean is grey, its luminance kept) -- for the
    visual cortex, which sees the whole frame."""
    f = bgr.astype(np.float32)
    m = f.reshape(-1, 3).mean(axis=0)
    if np.any(m <= 1e-6):
        return bgr
    return np.clip(f * (float(m.mean()) / m), 0, 255).astype(np.uint8)


def texture(grey: np.ndarray, shape: tuple[int, int], noise_floor: float) -> np.ndarray:
    """Its texture statistics per whole-field cell: (4, rows * cols) -- contrast,
    fineness (cycles per frame height), anisotropy, and fineness across
    (horizontal only: on the ground it grows as depth, where vertical
    fineness grows as depth squared -- Gibson's density vs compression
    gradients); 0 where the cell's contrast is under the sensor-noise floor
    (no texture to read); across is 0 where it is too fine to measure."""
    rows, cols = shape
    f = grey.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    h = f.shape[0]
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8.0  # per pixel
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    cell = lambda img: cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)  # noqa: E731
    m, m2 = cell(f), cell(f * f)
    jxx, jyy, jxy = cell(gx * gx), cell(gy * gy), cell(gx * gy)
    var = np.maximum(m2 - m * m, 0.0)
    contrast = np.sqrt(var)
    # the derivative filter reads a frequency f as sin(2 pi f) / (2 pi), not f
    # (the central difference's transfer function): undone exactly, up to its
    # limit of a quarter cycle a pixel
    raw = np.sqrt((jxx + jyy) / np.maximum(var, 1e-12))
    fineness = np.arcsin(np.clip(raw, 0.0, 1.0)) / (2.0 * math.pi) * h  # cycles per pixel -> per frame height
    tr = jxx + jyy
    anisotropy = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(tr, 1e-12)
    raw_x = np.sqrt(jxx / np.maximum(var, 1e-12))
    across = np.where(raw_x < 1.0, np.arcsin(np.clip(raw_x, 0.0, 1.0)) / (2.0 * math.pi) * h, 0.0)
    blank = contrast < noise_floor
    return np.stack([np.where(blank, 0.0, contrast), np.where(blank, 0.0, fineness),
                     np.where(blank, 0.0, np.clip(anisotropy, 0.0, 1.0)), np.where(blank, 0.0, across)]).reshape(4, -1)


def terrain_lessons(tex: np.ndarray, shape: tuple[int, int], horizon: float, e_map: np.ndarray,
                    exclude: np.ndarray | None = None) -> list[tuple[int, float]]:
    """The texture gradient's lessons: (cell, how its ground is raised, in
    camera heights) for each ground cell with texture. Fineness across f ~ k (1 - e) /
    (y - horizon) on ground of one texture; k is the median of its ROW's
    ground cells' readings through its map (e_map, per cell), at least
    MIN_GROUND_CELLS of them: along a row the depth on flat ground, and the
    camera's own resolution limit, are the same, so only what differs across
    the row teaches (pooled over rows, a texture reaching the pixel scale
    near the horizon read as a rise toward it: tested on rendered ground).
    It says what is raised or sunk beside its row, never how the whole ground
    tilts -- its flow and its detector teach that. Each lesson is e = 1 - f
    (y - horizon) / k, clipped to one camera height. exclude: cells not to
    read (its local frame)."""
    rows, cols = shape
    fine = tex[3]  # across: grows as depth on the ground
    y = (np.arange(rows * cols) // cols + 0.5) / rows
    ok = (fine > 0.0) & (y > horizon)
    if exclude is not None and len(exclude) == len(ok):
        ok &= ~np.asarray(exclude, dtype=bool)
    out = []
    for r in range(rows):
        idx = np.nonzero(ok[r * cols:(r + 1) * cols])[0] + r * cols
        if len(idx) < MIN_GROUND_CELLS:
            continue
        k0 = float(np.median(fine[idx] * (y[idx] - horizon) / (1.0 - np.clip(e_map[idx], -0.9, 0.9))))
        if k0 > 0.0:
            out += [(int(i), float(np.clip(1.0 - fine[i] * (y[i] - horizon) / k0, -1.0, 1.0))) for i in idx]
    return out
