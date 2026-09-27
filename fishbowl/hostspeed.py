from __future__ import annotations

"""
How fast this host does the brain's arithmetic -- measured, not assumed
(after a panel vote on pricing thought by the real machine: energy and
time are a brain's two currencies, and a price that treats every core as
equally fast is a hidden handwritten assumption).

sec_per_mac() times a fixed single-threaded matrix workload (the kind of
multiply-add the brain does) and returns seconds per multiply-add; the best
of several tries, so a background hiccup doesn't count as slowness.

Speed is the HARDWARE's: host_sec_per_mac() keeps the best rate this host
has ever measured (state/host_speed.json). How much of the machine it gets
-- contention, a busy livecam encoder, running at the lowest priority --
is the granted quota's job (tools/resource_handler.py); measuring it here
too counted it twice (on a laptop running a livecam server, a contended
benchmark read 33x slower than the chip is and made every neuron 65x dear).

REFERENCE_SEC_PER_MAC is Tanzania's rate, measured when this was written:
the host the prices were anchored on. On a host k times slower, thinking
costs k times more of the same granted CPU share (run_vision's price
quota), and a brain that could not finish before its next look misses it
(organism.py).
"""

import json
import time
from pathlib import Path

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


def host_sec_per_mac(record: Path) -> float:
    """This host's hardware rate: the best of a fresh measurement and every
    earlier one kept in `record`."""
    now = sec_per_mac()
    try:
        best = float(json.loads(record.read_text()).get("best_sec_per_mac", now))
    except (OSError, ValueError):
        best = now
    best = min(best, now)
    try:
        record.write_text(json.dumps({"best_sec_per_mac": best, "last_measured": now, "at": time.time()}))
    except OSError:
        pass
    return best


def speed_factor(measured: float | None = None) -> float:
    """This host's arithmetic speed relative to the reference host (1 = as fast;
    0.5 = half as fast)."""
    m = sec_per_mac() if measured is None else measured
    return REFERENCE_SEC_PER_MAC / max(1e-15, m)
