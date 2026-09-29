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
PROTO_SIDE = 16      # its reconstructions' grid, relative to its eye (a storage and display bound)
MAX_KC = 16384       # a safety bound only; the price limits it (4096 until a lineage pressed it, 2026-09-28)
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


MAX_HEADS = 4  # archetype heads at most (their inputs are fixed slots): a structural bound
TERRAIN_ROW = MAX_HEADS  # the terrain head: one more readout, after them (organism.py)


class MushroomBody:
    def __init__(self, n_kc: int, seed: int, weights: np.ndarray | None = None, danger: np.ndarray | None = None,
                 proto: np.ndarray | None = None, heads: np.ndarray | None = None):
        self.n_kc = int(n_kc)
        self.pos = _wiring(int(seed))[:self.n_kc]
        self.weights = resize(weights, self.n_kc)
        # An aversive compartment (like the fly's PPL1 punishment compartments,
        # Aso et al. 2014): a second output neuron reading the same Kenyon
        # cells, learning what came before a swat. Its learning rate is its own
        # trait (genome.aversive_rate, born 0: off until evolution turns it on).
        self.danger_weights = resize(danger, self.n_kc)
        # Imagery (genome.imagery, born off; a 2026-09-28 panel): each Kenyon
        # cell's prototype -- the average of what its eye saw (PROTO_SIDE^2,
        # relative to the eye) when the cell fired -- a way back from memory to
        # the eye, like feedback connections carrying predictions. Summed over a
        # replayed code, it is its own reconstruction of that memory.
        self.proto = resize_proto(proto, self.n_kc)
        # Archetype heads (a 2026-09-29 panel; Tooby & Cosmides, Menzel; the
        # self-driving "shared backbone, many heads"): up to MAX_HEADS more
        # output neurons on the same Kenyon cells, each taught by the
        # three-factor rule to predict one detector class in its gaze -- which
        # class is inherited and evolves. Categories grounded by a teacher.
        h = np.asarray(heads, dtype=float) if heads is not None and len(heads) else np.zeros((0, 0))
        self.heads = np.zeros((MAX_HEADS + 1, self.n_kc))  # + its terrain head
        if h.ndim == 2 and h.size:
            m = min(self.n_kc, h.shape[1])
            self.heads[:min(MAX_HEADS + 1, h.shape[0]), :m] = h[:MAX_HEADS + 1, :m]
        self.k = max(1, int(round(KC_ACTIVE * self.n_kc))) if self.n_kc else 0

    def active(self, look: np.ndarray, n: int, live: int | None = None) -> np.ndarray:
        """Indices of the Kenyon cells firing for this look (an n x n grid,
        flat). live: how many cells are still alive (a wasting body loses the
        newest first -- fishbowl/organism.py); None = all."""
        n_live = self.n_kc if live is None else max(0, min(self.n_kc, int(live)))
        if not n_live:
            return np.zeros(0, dtype=int)
        k = max(1, int(round(KC_ACTIVE * n_live)))
        grid = look.reshape(n, n)
        grid = grid - grid.mean()
        idx = np.clip(np.floor((self.pos[:n_live] + 0.5) * n).astype(int), 0, n - 1)
        drive = grid[idx[..., 1], idx[..., 0]].sum(axis=1)
        return np.argpartition(-drive, k - 1)[:k]

    def value(self, active: np.ndarray) -> float:
        return float(self.weights[active].mean()) if len(active) else 0.0

    def danger(self, active: np.ndarray) -> float:
        return float(self.danger_weights[active].mean()) if len(active) else 0.0

    def learn_danger(self, active: np.ndarray, punishment: float, rate: float) -> float:
        """The same three-factor rule, with punishment (blood lost to a swat,
        in meals) as the teaching signal; returns the prediction error."""
        if not len(active) or rate <= 0.0:
            return 0.0
        error = punishment - self.danger(active)
        self.danger_weights[active] += rate * error
        return error

    def learn_proto(self, active: np.ndarray, seen: np.ndarray, rate: float) -> None:
        if len(active) and rate > 0.0:
            self.proto[active] += rate * (seen - self.proto[active])

    def reconstruct(self, active: np.ndarray) -> np.ndarray | None:
        return self.proto[active].mean(axis=0) if len(active) else None

    def head_values(self, active: np.ndarray) -> np.ndarray:
        """Its archetype heads' values (the first MAX_HEADS rows)."""
        return self.heads[:MAX_HEADS, active].mean(axis=1) if len(active) else np.zeros(MAX_HEADS)

    def terrain_value(self, active: np.ndarray) -> float:
        """Its terrain head: how near it feels what it looks at is, from its own eye's code."""
        return float(self.heads[TERRAIN_ROW, active].mean()) if len(active) else 0.0

    def learn_terrain(self, active: np.ndarray, target: float, rate: float) -> None:
        """The three-factor rule toward the teacher's nearness (rate already weighted by the teacher's confidence)."""
        if len(active) and rate > 0.0:
            self.heads[TERRAIN_ROW, active] += rate * (target - self.terrain_value(active))

    def learn_heads(self, active: np.ndarray, targets: np.ndarray, rate: float, n: int) -> None:
        """The three-factor rule for the first n heads, toward their targets."""
        if len(active) and rate > 0.0 and n > 0:
            err = np.asarray(targets[:n]) - self.head_values(active)[:n]
            self.heads[:n, active] += rate * err[:, None]

    def learn(self, active: np.ndarray, reward: float, rate: float) -> float:
        """The three-factor update; returns the prediction error."""
        if not len(active) or rate <= 0.0:
            return 0.0
        error = reward - self.value(active)
        self.weights[active] += rate * error
        return error


def resize_proto(proto: np.ndarray | None, n_kc: int) -> np.ndarray:
    """Prototypes for n_kc cells: kept where they exist; mid-grey for new ones."""
    out = np.full((int(n_kc), PROTO_SIDE * PROTO_SIDE), 0.5, dtype=np.float32)
    if proto is not None and len(proto):
        p = np.asarray(proto, dtype=np.float32)
        if p.ndim == 2 and p.shape[1] == PROTO_SIDE * PROTO_SIDE:
            m = min(int(n_kc), len(p))
            out[:m] = p[:m]
    return out


def resize(weights: np.ndarray | None, n_kc: int) -> np.ndarray:
    """Learned weights for n_kc cells: kept where they exist, 0 for new cells."""
    out = np.zeros(int(n_kc))
    if weights is not None:
        w = np.asarray(weights, dtype=float)[:n_kc]
        out[:len(w)] = w
    return out
