from __future__ import annotations

"""
How fast this host does the brain's arithmetic -- measured, not assumed.
Speed is TIME, not energy (after a panel, revised: a snail's neurons aren't
dearer than a fly's, its world is slower -- Healy et al. 2013, temporal
resolution scales with metabolic rate): it sets the brain's deadline
(organism.py), while the energy price follows the CPU share granted.

sec_per_mac() times a fixed single-threaded matrix workload (the kind of
multiply-add the brain does) and returns seconds per multiply-add; the best
of several tries, so a background hiccup doesn't count as slowness.

Contention (a busy livecam encoder, the lowest priority) can only slow a
reading, so the evolution loop keeps the best rate measured in its run.

REFERENCE_SEC_PER_MAC is Tanzania's rate, for comparing hosts in the log.
"""

import time

import numpy as np

REFERENCE_SEC_PER_MAC = 5.8e-11  # Tanzania (Core i5-10210U), measured 2026-09-26; Tina (Core i3-2330M): ~4.0e-10
_SIZE, _REPEATS, _TRIES = 96, 60, 5


def sec_per_mac() -> float:
    rng = np.random.default_rng(0)
    a, b = rng.random((_SIZE, _SIZE)), rng.random((_SIZE, _SIZE))
    best = float("inf")
    for _ in range(_TRIES):
        t0 = time.perf_counter()
        for _ in range(_REPEATS):
            a @ b
        best = min(best, time.perf_counter() - t0)
    return best / (_REPEATS * _SIZE ** 3)


def speed_factor(measured: float | None = None) -> float:
    """This host's arithmetic speed relative to the reference host (1 = as fast;
    0.5 = half as fast)."""
    m = sec_per_mac() if measured is None else measured
    return REFERENCE_SEC_PER_MAC / max(1e-15, m)
