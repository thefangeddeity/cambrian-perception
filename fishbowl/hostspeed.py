from __future__ import annotations

"""
How fast this host does the brain's arithmetic -- measured, not assumed
(after a panel vote on pricing thought by the real machine: energy and
time are a brain's two currencies, and a price that treats every core as
equally fast is a hidden handwritten assumption).

sec_per_mac() times a fixed single-threaded matrix workload (the kind of
multiply-add the brain does) and returns seconds per multiply-add; the best
of several tries, so a background hiccup doesn't count as slowness. The
evolution loop re-measures it with the quota, so load changes are felt.

REFERENCE_SEC_PER_MAC is Tanzania's rate, measured when this was written:
the host the prices were anchored on. On a host k times slower, thinking
costs k times more of the same granted CPU share (run_vision's price
quota), and a brain that could not finish before its next look misses it
(organism.py).
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
