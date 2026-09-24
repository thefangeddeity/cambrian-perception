from __future__ import annotations

"""
CONSPEC -- a drive, not a reflex. Named directly after Morton &
Johnson's (1991) two-process account of newborn face perception:
CONSPEC is the crude, innate, subcortical template that biases a
newborn to orient toward face-like configurations before any learning
has happened at all (and works despite very poor newborn visual
acuity, precisely because it only needs a coarse configural pattern,
not fine detail -- the same reason a 12x12 retina grid is enough for
this). CONLERN is the second process: the part that later LEARNS to
refine that crude bias into real recognition through experience.

That split is exactly this project's own architecture: this module is
CONSPEC (fixed, human-written, never evolved -- same contract as
reflexes.py). The genome's own evolved response is standing in for
CONLERN -- whatever it learns to do with a sustained conspec signal is
its own discovery, not something handed to it.

Different in KIND from reflexes.py's three signals, which are all
reactive (score a single event as it happens). This is a DRIVE: it
rewards SUSTAINED engagement across a window, not a one-shot spike --
"stay oriented toward this" rather than "notice this happened." the user's
own framing: an innate pressure to seek out other beings in the
apartment for safety, the same real evolutionary logic behind
attachment behavior in altricial animals generally, not unique to
human infants.

Honest limitation: there is no pan/tilt/zoom actuator on the camera
this runs against, so "seek" here means "the genome's own output stays
elevated and correlated for as long as a being-like configuration is
present," not literal camera movement toward it. If the hardware ever
supports real orienting, that's a real future upgrade to this same
drive, not a different one.
"""

import numpy as np

from .retina import GRID

# A crude, fixed "two eyes + one mouth" template -- an inverted
# triangle of high local contrast, scanned over the grid at every
# plausible "head-sized" window position. Deliberately simple: real
# CONSPEC research (and the classic "two dots + a dash" moving-face
# preference studies) shows this level of crudeness is what a newborn
# visual system actually works with, not a stand-in for something more
# sophisticated that was too much effort to build.
#
# Derived from GRID as a fixed PROPORTION (roughly a third of the
# grid's own width), not a hardcoded cell count -- User: "change it so
# it can remain valid as [the retina] evolves." If GRID ever grows
# (manually, or via a future evolvable-resolution trait), "head-sized"
# stays calibrated to the same real fraction of the field of view
# instead of silently shrinking. This is a MECHANICAL fix only, not an
# evolvable one: _HEAD_SIZE stays entirely outside the genome's reach,
# same as every other fixed reflex/grading constant in this module --
# see conspec_signal's own module docstring, and the real reason this
# matters: letting the organism tune its own detection criteria (even
# with a fitness penalty attached) would just price a reward-hacking
# exploit, not remove it. Recomputed once at import time against
# whatever GRID currently is; if resolution ever becomes genuinely
# per-instance (not a single shared global, which it isn't today),
# this would need to become a function taking grid size as an
# argument instead of a module-level constant -- noted, not built,
# since that's the separate evolvable-resolution project, not this fix.
_HEAD_SIZE = max(2, GRID[0] // 3)  # window height/width, in retina cells
_EYE_ROW = max(0, _HEAD_SIZE // 4)              # ~1/4 down the window
_EYE_COLS = (max(0, _HEAD_SIZE // 4), min(_HEAD_SIZE - 1, 3 * _HEAD_SIZE // 4))  # ~1/4 and ~3/4 across
_MOUTH_ROW = min(_HEAD_SIZE - 1, 3 * _HEAD_SIZE // 4)  # ~3/4 down the window
_MOUTH_COL = _HEAD_SIZE // 2                     # centered


def _template_match(cells: np.ndarray) -> tuple[float, float, float]:
    """
    cells: (GRID[0], GRID[1]) frame, already reshaped from retina.py's
    flat vector. Returns (strength, peak_cx, peak_cy) -- strength is
    how strongly the strongest head-sized window in this frame matches
    the crude eyes+mouth contrast pattern (darker at the three feature
    points than the window's own local surround); peak_cx/peak_cy are
    that window's own CENTER, normalized to [0, 1] using fovea.py's
    own cx/cy convention (cx = horizontal fraction, cy = vertical
    fraction). The location used to be thrown away (only the max
    strength was kept) -- added 2026-09-24 so the "seek" drive below
    can reward the fovea for moving TOWARD a detected being, not just
    correlating its response with detection. 0/(0.5, 0.5) if the grid
    is too small for even one window.
    """
    h, w = cells.shape
    best = 0.0
    best_row, best_col = h / 2.0, w / 2.0
    for top in range(0, h - _HEAD_SIZE + 1):
        for left in range(0, w - _HEAD_SIZE + 1):
            window = cells[top:top + _HEAD_SIZE, left:left + _HEAD_SIZE]
            surround_mean = window.mean()
            eye_l = window[_EYE_ROW, _EYE_COLS[0]]
            eye_r = window[_EYE_ROW, _EYE_COLS[1]]
            mouth = window[_MOUTH_ROW, _MOUTH_COL]
            # Real faces are usually darker at eyes/mouth than the
            # surrounding skin/local average under typical indoor
            # lighting -- contrast, not absolute brightness (so this
            # doesn't just fire on "anything dark").
            contrast = surround_mean - np.array([eye_l, eye_r, mouth])
            score = float(np.clip(contrast, 0.0, None).mean())
            if score > best:
                best = score
                best_row = top + _HEAD_SIZE / 2.0
                best_col = left + _HEAD_SIZE / 2.0
    return best, best_col / w, best_row / h


def conspec_signal(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    vectors: (T, N_CELLS). Returns (strength, peak_cx, peak_cy), each
    (T,) -- strength is NOT yet a fitness score (see drive_fitness);
    peak_cx/peak_cy are where the detection is, for the real seek
    reward (see run_vision.py's _seek_reward).
    """
    rows, cols = GRID
    results = [_template_match(v.reshape(rows, cols)) for v in vectors]
    strength = np.array([r[0] for r in results])
    peak_cx = np.array([r[1] for r in results])
    peak_cy = np.array([r[2] for r in results])
    return strength, peak_cx, peak_cy


def drive_fitness(conspec: np.ndarray, organism_output: np.ndarray, window: int = 8) -> float:
    """
    The sustained-engagement reward: correlation between the conspec
    signal and the genome's own output, computed over ROLLING windows
    and averaged -- rewards staying correlated across a stretch of
    frames, not just matching a single instant. Returns 0.0 if there's
    no variance to correlate against (e.g. a being never appears, or
    the organism's output is constant) rather than a misleading NaN.
    """
    n = len(conspec)
    if n < window:
        return 0.0
    scores = []
    for start in range(0, n - window + 1):
        c = conspec[start:start + window]
        o = organism_output[start:start + window]
        if c.std() < 1e-9 or o.std() < 1e-9:
            continue
        corr = float(np.corrcoef(c, o)[0, 1])
        if np.isfinite(corr):
            scores.append(corr)
    return float(np.mean(scores)) if scores else 0.0


class Habituation:
    """
    "Humans and cats are harmless noise... commensal beings" -- the user's
    own framing, made concrete. A being detected repeatedly with no
    accompanying loom (nothing bad ever happens alongside it) should
    cost progressively less to ignore; a being detected alongside real
    threat-level events should never habituate the same way. This is
    one of the most ancient, well-documented forms of learning that
    exists -- present even in organisms with no real nervous network
    (Aplysia's gill-withdrawal habituation is the classic case) -- not
    something invented for this project.

    Persists ACROSS an entire run (frames, not generations) via an
    exponential moving average of "conspec present, no loom" exposure
    -- real accumulated history, not something reset per mutation.
    """

    def __init__(self, decay: float = 0.98, loom_threshold: float = 0.02):
        self.exposure = 0.0  # EMA of harmless-presence frequency, [0, 1]
        self.decay = decay
        self.loom_threshold = loom_threshold

    def observe(self, conspec_present: bool, loom_value: float) -> None:
        harmless_presence = 1.0 if (conspec_present and loom_value < self.loom_threshold) else 0.0
        self.exposure = self.decay * self.exposure + (1.0 - self.decay) * harmless_presence
        if loom_value >= self.loom_threshold:
            # A real threat-paired event resensitizes rather than just
            # failing to further habituate -- the asymmetry is the
            # actual point (see reflexes.py's own module docstring on
            # threat vs. food asymmetry).
            self.exposure *= 0.5

    @property
    def discount(self) -> float:
        """Reward multiplier for conspec-drive fitness -- 1.0 = full reward (novel/rare), down toward a floor as exposure accumulates. Never reaches exactly 0: even a fully habituated commensal is still worth SOME baseline attention."""
        floor = 0.15
        return floor + (1.0 - floor) * (1.0 - self.exposure)
