from __future__ import annotations

"""
Lifetime reward learning: a mushroom body (after a design panel; the insect
circuit where a fly learns which odours mean food, e.g. Aso et al. 2014,
Hige et al. 2015).

  Kenyon cells: each samples KC_INPUTS random receptors of its gaze (Caron
    et al. 2013: ~7 inputs each, randomly wired), centre-surround (each
    receptor minus its eye's mean, like the projection neurons' lateral
    inhibition); only the most driven KC_ACTIVE share fire (Turner et al.
    2008, Honegger et al. 2011: ~5% of Kenyon cells respond) -- a sparse,
    high-dimensional code of what it is looking at.
  A food-value output neuron reads them through weights that LEARN during
    its life, by the three-factor rule: when a look is followed by eating,
    the weights from the cells that were active move by the learning rate x
    the reward prediction error (reward - predicted value) -- the dopamine
    teaching signal (Schultz et al. 1997). The reward is the energy it
    actually ate, in meals (one full look at prey = 1), so no scale is set
    by hand.
  The learned value is one of the brain's inputs ("food value"): the brain
    can come to steer toward whatever has meant food to THIS individual.

What evolves: how many Kenyon cells (genome.kc, grown by mutation and
priced like the brain's arithmetic), their random wiring (genome.kc_seed)
and the learning rate (genome.learning_rate). Both start at 0 -- a newborn
has no mushroom body, and a first one changes nothing until the brain
wires its output in. What is LEARNED is memory, not genes: the weights are
carried across generations like the surprise memory (run_vision), and a
live host's organism keeps learning in its own room (fishbowl/live.py).

Each Kenyon cell's inputs are drawn at positions relative to the eye's
size, so they keep sampling the same parts of the gaze as the eye grows.
"""

import numpy as np

KC_INPUTS = 7        # inputs per Kenyon cell (Caron et al. 2013)
KC_ACTIVE = 0.05     # share of Kenyon cells active on a look (Turner et al. 2008)
MAX_KC = 4096        # a safety bound only; the price limits it
_WIRING: dict[int, np.ndarray] = {}


def _wiring(seed: int) -> np.ndarray:
    """(MAX_KC, KC_INPUTS, 2) input positions in [-0.5, 0.5) of the eye's
    side, fixed for a seed -- growing adds cells, never rewires old ones."""
    if seed not in _WIRING:
        _WIRING[seed] = np.random.default_rng(seed).random((MAX_KC, KC_INPUTS, 2)) - 0.5
    return _WIRING[seed]


def macs(n_kc: int) -> int:
    """Multiply-adds per look: each cell's inputs, plus the output neuron."""
    return n_kc * (KC_INPUTS + 1)


class MushroomBody:
    def __init__(self, n_kc: int, seed: int, weights: np.ndarray | None = None):
        self.n_kc = int(n_kc)
        self.pos = _wiring(int(seed))[:self.n_kc]
        self.weights = resize(weights, self.n_kc)
        self.k = max(1, int(round(KC_ACTIVE * self.n_kc))) if self.n_kc else 0

    def active(self, look: np.ndarray, n: int) -> np.ndarray:
        """Indices of the Kenyon cells firing for this look (an n x n grid, flat)."""
        if not self.n_kc:
            return np.zeros(0, dtype=int)
        grid = look.reshape(n, n)
        grid = grid - grid.mean()
        idx = np.clip(np.floor((self.pos + 0.5) * n).astype(int), 0, n - 1)
        drive = grid[idx[..., 1], idx[..., 0]].sum(axis=1)
        return np.argpartition(-drive, self.k - 1)[:self.k]

    def value(self, active: np.ndarray) -> float:
        return float(self.weights[active].mean()) if len(active) else 0.0

    def learn(self, active: np.ndarray, reward: float, rate: float) -> float:
        """The three-factor update; returns the prediction error."""
        if not len(active) or rate <= 0.0:
            return 0.0
        error = reward - self.value(active)
        self.weights[active] += rate * error
        return error


def resize(weights: np.ndarray | None, n_kc: int) -> np.ndarray:
    """Learned weights for n_kc cells: kept where they exist, 0 for new cells."""
    out = np.zeros(int(n_kc))
    if weights is not None:
        w = np.asarray(weights, dtype=float)[:n_kc]
        out[:len(w)] = w
    return out
