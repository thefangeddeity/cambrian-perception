from __future__ import annotations

"""
The one thing the brain is trying to get better at -- a fixed,
synthetic, purely internal function. No real-world data, no external
dependency of any kind: even if every other fishbowl constraint
somehow failed, there is nothing "real" here for the brain to affect.

Deliberately arbitrary (not modeling anything real) -- the point of
this repo is watching HOW a genome self-improves and how its own
mutation strategy evolves, not producing a useful regression fit.
"""

import numpy as np

N_VARS = 3
SEED = 20260924  # fixed, so every run is scored against the exact
# same synthetic function and the same evaluation set -- otherwise
# "did it get better" has no stable meaning across generations.


def _hidden_function(x: np.ndarray) -> np.ndarray:
    """
    x: (n_samples, N_VARS). An arbitrary, fixed nonlinear combination
    -- picked for having curvature and cross-term interaction (so a
    trivial single-variable linear tree scores poorly), not for
    meaning anything.
    """
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    return np.sin(a) * b + 0.3 * c**2 - 0.5 * a * c


def make_eval_set(n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-3.0, 3.0, size=(n_samples, N_VARS))
    y = _hidden_function(x)
    return x, y


def fitness(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Lower is better -- plain MSE. Returns +inf for a non-finite
    prediction rather than letting NaN/inf silently compare as
    "better" than a real score (the exact class of bug this project's
    sibling, orbital-organism, found live in its own accept/reject
    gate: a NaN score can slip past a plain `<` comparison under
    IEEE754 semantics unless checked explicitly).
    """
    if not np.all(np.isfinite(predicted)):
        return float("inf")
    return float(np.mean((predicted - target) ** 2))
