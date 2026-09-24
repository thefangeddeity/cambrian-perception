from __future__ import annotations

"""
Innate, FIXED salience signals -- never evolved, never fed to the
genome as an input. See the README's reflex-curriculum section: these
exist purely to grade fitness (did the organism's own output track a
biologically-relevant event), the same way a real nervous system's
low-level reflex circuits are largely hardwired rather than learned.
The genome never sees these numbers as input; it only ever gets
scored against them.

Three signals, staged easiest-to-hardest by how evolutionarily ancient
the real reflex is:

  luminance_change -- global brightness change. The most primitive
    orienting response that exists; organisms with no image-forming
    eye at all still have this.

  optomotor -- coherent whole-field motion (a real, well-established
    reflex present in almost every visual animal, from insects to
    scallops).

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


def optomotor_score(vectors: np.ndarray) -> np.ndarray:
    """
    Whole-field coherent motion -- mean absolute per-cell change,
    frame to frame. Doesn't distinguish "everything changed a little"
    from "one thing changed a lot" (that's what loom_score is for);
    this is deliberately the cruder, more primitive signal.
    """
    diffs = np.zeros(len(vectors))
    diffs[1:] = np.abs(np.diff(vectors, axis=0)).mean(axis=1)
    return diffs


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
    return {
        "luminance_change": luminance_change(vectors),
        "optomotor": optomotor_score(vectors),
        "loom": loom_score(vectors),
    }
