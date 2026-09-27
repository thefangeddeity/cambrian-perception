from __future__ import annotations

"""
What counts as one MEAL (or one snack), measured from the organism's own
feeding record instead of a hand-set "a second without eating" (after a
design panel; the behavioural-ecology yardstick):

  Feeding comes in bouts. The gaps between feeding acts form two
  populations -- short gaps within a bout, long gaps between bouts -- and
  their survivorship (how many gaps are longer than t) is the sum of two
  exponentials (Sibly, Nott & Fletcher 1990), fitted by maximum likelihood
  (Langton, Collett & Sibly 1995):

      density(t) = p_f * r_f * exp(-r_f t) + p_s * r_s * exp(-r_s t)

  The BOUT CRITERION is the gap at which a gap is equally likely to belong
  to either population -- the one that misassigns the fewest gaps:

      t_c = ln(p_f r_f / (p_s r_s)) / (r_f - r_s)

  A gap longer than t_c starts a new meal.

It stays "calibrating" (None) until the two-population model actually
explains its gaps better than one population, by the Bayesian information
criterion (Schwarz 1978) -- until then there is no bout structure to speak
of, and no number is invented.

A feeding act is one look (gaze) that caught prey (a meal act) or any
surprise (a snack act). Acts are timed by the feed's own frame index, and
a live organism re-lives overlapping snapshots, so FeedingRecord only adds
acts at frames it has not recorded yet: the record is its life, not the
sum of its rehearsals.
"""

import math

import numpy as np

MAX_GAPS = 5000        # the record keeps its most recent gaps (a rolling life history)
EM_ITERATIONS = 200
EM_TOLERANCE = 1e-9


def _loglik_one(t: np.ndarray) -> float:
    r = 1.0 / max(1e-12, float(t.mean()))
    return float(np.sum(np.log(r) - r * t))


def fit_bout_criterion(gaps) -> dict | None:
    """The bout criterion (seconds) from feeding gaps (seconds), with the
    fitted processes; None while there is no bout structure (see above)."""
    t = np.asarray([g for g in gaps if g > 0], dtype=float)
    n = len(t)
    if n <= 3 or np.ptp(t) <= 0:  # no more gaps than the model has parameters: nothing to fit
        return None
    # Start: the fast process from the shorter half of the gaps, the slow one from the rest.
    order = np.sort(t)
    r_f, r_s = 1.0 / max(1e-12, order[: n // 2].mean()), 1.0 / max(1e-12, order[n // 2:].mean())
    if r_f <= r_s:
        return None
    p_f = 0.5
    prev = -math.inf
    for _ in range(EM_ITERATIONS):
        a = p_f * r_f * np.exp(-r_f * t)
        b = (1.0 - p_f) * r_s * np.exp(-r_s * t)
        tot = np.maximum(a + b, 1e-300)
        loglik = float(np.sum(np.log(tot)))
        w = a / tot
        p_f = float(np.clip(w.mean(), 1e-6, 1 - 1e-6))
        r_f = float(w.sum() / max(1e-12, np.sum(w * t)))
        r_s = float((1 - w).sum() / max(1e-12, np.sum((1 - w) * t)))
        if abs(loglik - prev) < EM_TOLERANCE * max(1.0, abs(loglik)):
            break
        prev = loglik
    if r_f < r_s:  # keep "fast" the fast one
        r_f, r_s, p_f = r_s, r_f, 1.0 - p_f
    a = p_f * r_f * np.exp(-r_f * t)
    b = (1.0 - p_f) * r_s * np.exp(-r_s * t)
    loglik2 = float(np.sum(np.log(np.maximum(a + b, 1e-300))))
    bic1 = -2.0 * _loglik_one(t) + 1 * math.log(n)
    bic2 = -2.0 * loglik2 + 3 * math.log(n)
    if not (bic2 < bic1 and r_f > r_s):
        return None
    tc = math.log(p_f * r_f / ((1.0 - p_f) * r_s)) / (r_f - r_s)
    if not (math.isfinite(tc) and tc > 0):
        return None
    return {"criterion_s": round(tc, 3), "gaps": n,
            "within_bout_mean_s": round(1.0 / r_f, 3), "between_bouts_mean_s": round(1.0 / r_s, 3),
            "within_bout_share": round(p_f, 3)}


class FeedingRecord:
    """Its feeding acts over its life (one kind: meals or snacks), as gaps in
    seconds. add() takes a run's acts (the feed's frame indices) and the
    newest frame that run covered; only frames newer than any recorded
    before are added. A new feed (a restart, another video) starts a new
    stretch: no gap is counted across it."""

    def __init__(self, gaps=None):
        self.gaps = list(gaps or [])[-MAX_GAPS:]
        self.covered = None   # newest frame index already recorded (this feed)
        self.last_act = None  # frame index of the latest act (this feed)
        self.epoch = None

    def add(self, act_frames, newest_frame: int, fps: float, epoch=None) -> None:
        if epoch != self.epoch:
            self.epoch, self.covered, self.last_act = epoch, None, None
        fps = max(1e-6, float(fps))
        for k in sorted(act_frames):
            if self.covered is not None and k <= self.covered:
                continue
            if self.last_act is not None and k > self.last_act:
                self.gaps.append(round((k - self.last_act) / fps, 3))
            self.last_act = k
        self.covered = max(newest_frame, self.covered if self.covered is not None else newest_frame)
        del self.gaps[:-MAX_GAPS]
