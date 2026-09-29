"""
Its medial entorhinal cortex and hippocampal map (a 2026-09-29 panel: O'Keefe,
the Mosers, McNaughton, Burgess, Jeffery, Taube; Kropff's speed cells,
Stensola's grid modules, Solstad's grid-to-place model, Foster, Morris & Dayan's
place-cell value learning).

  - speed cells: its forward speed, in eye-heights per second, from the whole
    field's expansion rate x its ground's typical depth (its ground model);
    signed; and its change per look (an otolith's sense: a start, a stop)
  - path integration: its position on its ground (eye-heights), speed x its
    compass's heading, frame by frame; re-anchored when it returns to a scene
  - grid cells: GRID_MODULES modules, spacings from its own unit (one
    eye-height) in Stensola's sqrt(2) steps, GRID_PER_MODULE cells each with a
    random phase; each the classic sum of three cosines 60 degrees apart
  - place cells: sparse random conjunctions of grid cells (Solstad), a fresh
    random map per scene (global remapping, its scene library the context),
    PLACE_ACTIVE of them firing, as its Kenyon cells do
  - a value map on its place cells: each learns what happened where it fires
    (its reward, at its learning rate) -- what this place has been worth

It is a passenger: it can't steer its body. Its map tells it where on its
route it is, and what has happened there before.
"""
from __future__ import annotations

import math

import numpy as np

GRID_MODULES = 4
GRID_PER_MODULE = 16
GRID_RATIO = math.sqrt(2.0)   # Stensola et al. 2012: successive modules ~1.42x apart
PLACE_CELLS = 128
PLACE_INPUTS_EACH = 7         # grid cells summed by each place cell (as a Kenyon cell's 7 receptor inputs)
PLACE_ACTIVE = 0.05           # the share firing (as its Kenyon cells: ~5%)
_AXES = np.array([[math.cos(a), math.sin(a)] for a in (0.0, math.pi / 3, 2 * math.pi / 3)])


class Entorhinal:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.spacings = np.array([GRID_RATIO ** m for m in range(GRID_MODULES)])  # eye-heights
        self.phases = rng.uniform(0.0, 1.0, (GRID_MODULES, GRID_PER_MODULE, 2)) * self.spacings[:, None, None]
        self.position = np.zeros(2)       # eye-heights on its ground (x right, y ahead at heading 0)
        self.speed = 0.0                  # eye-heights per second, signed
        self._speed_look = 0.0
        self.acceleration = 0.0           # change in speed per look (eye-heights per second, per look)
        self.place_values = np.zeros(PLACE_CELLS)
        self._maps: dict = {}             # scene -> its place cells' grid inputs (global remapping)
        self.active = np.zeros(0, dtype=int)

    def step(self, expansion_per_frame: float, fps: float, horizon: float | None, heading: float) -> None:
        """One frame: its speed from the field's expansion and its ground's
        typical depth (the ground half-way from the horizon to the frame's
        bottom: 2 / (1 - horizon) eye-heights, focal length ~ frame height), and
        its position moved along its heading."""
        if horizon is None or horizon >= 1.0 or fps <= 0:
            self.speed = 0.0
            return
        depth = 2.0 / (1.0 - horizon)
        self.speed = expansion_per_frame * fps * depth
        d = self.speed / fps
        self.position += d * np.array([math.sin(heading), math.cos(heading)])

    def look(self) -> tuple[float, float]:
        """Its speed and acceleration senses at a look: (speed, change since its
        last look), in eye-heights per second, clipped to [-1, 1] after tanh."""
        acc = self.speed - self._speed_look
        self._speed_look, self.acceleration = self.speed, acc
        return float(np.tanh(self.speed)), float(np.tanh(acc))

    def grid(self) -> np.ndarray:
        """All its grid cells' activity (0..1): for each cell, the three
        cosines of its position against its phase, at its module's spacing."""
        k = 4 * math.pi / (math.sqrt(3) * self.spacings)  # the hexagonal lattice's wave number
        rel = self.position[None, None, :] - self.phases                     # modules x cells x 2
        proj = np.einsum("mcd,ad->mca", rel, _AXES)                          # modules x cells x 3 axes
        g = np.cos(k[:, None, None] * proj).sum(axis=2)                       # -1.5 .. 3
        return ((g + 1.5) / 4.5).ravel()

    def places(self, scene: int) -> np.ndarray:
        """Its place cells firing now: each sums PLACE_INPUTS_EACH random grid
        cells (a random map per scene), the top PLACE_ACTIVE share fire."""
        m = self._maps.get(scene)
        if m is None:
            rng = np.random.default_rng((self.seed, int(scene)))
            m = rng.integers(0, GRID_MODULES * GRID_PER_MODULE, (PLACE_CELLS, PLACE_INPUTS_EACH))
            self._maps[scene] = m
        drive = self.grid()[m].sum(axis=1)
        k = max(1, int(round(PLACE_ACTIVE * PLACE_CELLS)))
        self.active = np.argpartition(-drive, k - 1)[:k]
        return self.active

    def value(self) -> float:
        """What this place has been worth (its active place cells' mean value)."""
        return float(self.place_values[self.active].mean()) if len(self.active) else 0.0

    def learn(self, reward: float, rate: float) -> None:
        if len(self.active) and rate > 0.0:
            self.place_values[self.active] += rate * (reward - self.place_values[self.active])

    def macs(self) -> int:
        """Its cost per look: grid cells (three cosines each) and place cells (their sums)."""
        return GRID_MODULES * GRID_PER_MODULE * 6 + PLACE_CELLS * PLACE_INPUTS_EACH
