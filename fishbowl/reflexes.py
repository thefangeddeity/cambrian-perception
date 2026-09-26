from __future__ import annotations

"""
Innate, FIXED salience signals -- never evolved, never fed to the
genome as an input. See the README's reflex-curriculum section: these
exist purely to grade fitness (did the organism's own output track a
biologically-relevant event), the same way a real nervous system's
low-level reflex circuits are largely hardwired rather than learned.
The genome never sees these numbers as input; it only ever gets
scored against them.

Four signals, staged easiest-to-hardest by how evolutionarily ancient
the real reflex is:

  luminance_change -- global brightness change. The most primitive
    orienting response that exists; organisms with no image-forming
    eye at all still have this.

  motion_energy -- coherent whole-field CHANGE, direction-blind (mean
    |change|, frame to frame) -- a real, primitive precursor signal,
    but NOT what it was called until 2026-09-24 ("optomotor"): a real
    optomotor/optokinetic response is direction-SELECTIVE (it drives a
    compensating turn, which requires knowing which way something
    moved), and this can't tell "moving right" from "moving left", or
    from "flickering in place." Renamed after an external audit caught
    the overclaim. See directional_motion below for the real thing.

  directional_motion -- a real, simplified Hassenstein-Reichardt
    correlator: the classic, decades-studied model of how insect
    (and more broadly, motion-sensitive retinal) circuits compute
    WHICH WAY something moved, not just that something changed. Two
    signed components (motion_x, motion_y): multiply a cell's CURRENT
    value against its neighbor's value one frame ago, subtract the
    mirrored (opposite-direction) product -- the antisymmetric
    subtraction is what cancels ordinary flicker and leaves only real,
    signed, directional motion. This is what run_vision.py's new
    optokinetic-pursuit reward is graded against -- added specifically
    because nothing previously rewarded the pan/tilt actuator for
    actually tracking real motion (the user, watching a real deployed kitten
    cam: "this cat's been there the whole time, but the fovea's too
    primitive to evolve to lock on it").

  loom -- local expansion of a change-region, a real approximation of
    tau (time-to-contact from expansion rate) -- the reflex behind
    startle/flinch responses to an approaching object, studied down to
    the single-neuron level in locusts (LGMD/DCMD). This is a
    deliberately SIMPLE approximation (see loom_score's own docstring
    for exactly what it does and doesn't capture), not a claim to
    replicate real neuroscience precisely.
"""

import numpy as np

from .retina import GRID


def luminance_change(vectors: np.ndarray) -> np.ndarray:
    """vectors: (T, N_CELLS). Returns (T,), 0 for the first frame."""
    mean_lum = vectors.mean(axis=1)
    change = np.zeros(len(mean_lum))
    change[1:] = np.abs(np.diff(mean_lum))
    return change


def motion_energy_score(vectors: np.ndarray) -> np.ndarray:
    """
    Whole-field coherent CHANGE -- mean absolute per-cell change,
    frame to frame. Doesn't distinguish "everything changed a little"
    from "one thing changed a lot" (that's what loom_score is for),
    and doesn't know WHICH WAY anything moved (see directional_motion
    for that) -- deliberately the crudest, most primitive signal here.
    """
    diffs = np.zeros(len(vectors))
    diffs[1:] = np.abs(np.diff(vectors, axis=0)).mean(axis=1)
    return diffs


def directional_motion(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A real, simplified Hassenstein-Reichardt correlator -- see the
    module docstring for why this exists and what it fixes. For each
    pair of horizontally (then vertically) adjacent grid cells: the
    "rightward" term is the LEFT cell's value one frame ago times the
    RIGHT cell's value now (a feature seen at left, then at right,
    shortly after = it moved right); the mirrored "leftward" term is
    the opposite pairing. Their difference is a real signed motion
    estimate that cancels out uniform flicker (which contributes
    equally to both terms) and responds only to genuine directional
    movement. Averaged over all adjacent-cell pairs in the grid for
    one global (motion_x, motion_y) estimate per frame -- coarse (this
    is a 12x12 grid, not a dense retina), but real: it is the actual
    correlator model, not a metaphor for it.

    Returns (motion_x, motion_y), each (T,) -- positive x = rightward,
    positive y = downward, 0 for the first frame (no prior frame to
    correlate against yet).
    """
    n = len(vectors)
    rows, cols = GRID
    grids = vectors.reshape(n, rows, cols)
    motion_x = np.zeros(n)
    motion_y = np.zeros(n)
    for t in range(1, n):
        cur, prev = grids[t], grids[t - 1]
        rightward = cur[:, 1:] * prev[:, :-1] - cur[:, :-1] * prev[:, 1:]
        motion_x[t] = rightward.mean()
        downward = cur[1:, :] * prev[:-1, :] - cur[:-1, :] * prev[1:, :]
        motion_y[t] = downward.mean()
    return motion_x, motion_y


def loom_score(vectors: np.ndarray) -> np.ndarray:
    """
    A real, simplified expansion-rate proxy: at each frame, compute
    the per-cell |change|, then the CHANGE-WEIGHTED spatial spread
    (mean distance of changed cells from the grid center). A real
    looming stimulus grows in extent AND in magnitude together over a
    short window -- this returns high only when BOTH the spread and
    the magnitude are increasing at once, zero otherwise (including
    when something is shrinking, or when magnitude grows without
    spatial spread -- e.g. a flicker at one fixed point, which isn't
    looming).

    What this does NOT capture: true tau requires knowing the change
    is a single coherent object's edge, not scattered independent
    motion -- this has no object model at all, it's a cheap global
    proxy. Real, stated plainly, not oversold.
    """
    n = len(vectors)
    rows, cols = GRID
    yy, xx = np.mgrid[0:rows, 0:cols]
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).reshape(-1)
    max_radius = radius.max() if radius.max() > 0 else 1.0
    radius = radius / max_radius

    spread = np.zeros(n)
    magnitude = np.zeros(n)
    for t in range(1, n):
        cell_diff = np.abs(vectors[t] - vectors[t - 1])
        total = cell_diff.sum()
        magnitude[t] = cell_diff.mean()
        if total > 1e-9:
            spread[t] = float((cell_diff * radius).sum() / total)

    d_spread = np.zeros(n)
    d_magnitude = np.zeros(n)
    d_spread[1:] = np.diff(spread)
    d_magnitude[1:] = np.diff(magnitude)

    growing_spread = np.clip(d_spread, 0.0, None)
    growing_magnitude = np.clip(d_magnitude, 0.0, None)
    return growing_spread * growing_magnitude


def all_signals(vectors: np.ndarray) -> dict[str, np.ndarray]:
    motion_x, motion_y = directional_motion(vectors)
    return {
        "luminance_change": luminance_change(vectors),
        "motion_energy": motion_energy_score(vectors),
        "motion_x": motion_x,
        "motion_y": motion_y,
        # Magnitude of the signed (motion_x, motion_y) pair -- the
        # combined signal SIGNAL_WEIGHTS correlates response against
        # (direction-aware, unlike motion_energy); motion_x/motion_y
        # stay available separately for the optokinetic pursuit reward,
        # which needs the actual SIGNED direction, not just magnitude.
        "directional_motion": np.hypot(motion_x, motion_y),
        "loom": loom_score(vectors),
    }


def expansion_score(vectors: np.ndarray, threshold: float = 0.08, adapt: float = 0.05) -> np.ndarray:
    """
    A looming detector modeled on the locust LGMD/DCMD: responds to a
    silhouette GROWING, not to something sliding past. Each frame, the
    area (fraction of cells) DARKER than a slow-adapting background by
    more than threshold -- an approaching object blocks light, and real
    looming-escape circuits (LGMD, a tubeworm's shadow reflex) respond
    to an expanding shadow, not to brightening (on the real camera, a
    large dark shape approaching fired this correctly, while the same
    shape leaving -- a brightening -- also fired before this was
    dark-only); loom = that area's growth, counted only while growth
    is sustained over consecutive frames. A crossing object keeps
    roughly constant area, so it scores ~0 (the translation-rejecting
    role lateral inhibition plays in the real neuron); an approaching
    one keeps getting bigger.

    Written because loom_score() above, measured 2026-09-26, scored a
    blob CROSSING the frame ~10x higher than a disc actually
    approaching it.
    """
    n = len(vectors)
    out = np.zeros(n)
    if n == 0:
        return out
    background = vectors[0].astype(np.float64).copy()
    prev_area = prev_growth = 0.0
    for t in range(n):
        area = float(((background - vectors[t]) > threshold).mean())
        growth = area - prev_area
        if t >= 2 and growth > 0 and prev_growth > 0:
            out[t] = growth
        prev_area, prev_growth = area, growth
        background += adapt * (vectors[t] - background)
    return out
