#!/usr/bin/env python3
from __future__ import annotations

"""
A lesion experiment: gives the organism's brain a "stroke". A random,
contiguous region of hidden units (a quarter of them, by default) has every
weight into, out of and within it scrambled among themselves -- the weights
survive, their wiring doesn't, like tissue that is still there but no
longer connected the way it was. Everything else is left as it was.

Then watch what evolution does: which behaviours break, which survive, and
how fast (and how) it recovers. The checkpoint records where and when
("stroke": units and generation) for comparing recovery later.

Run only while the organism is stopped. The pre-stroke checkpoint is kept
as checkpoint.json.bak-stroke-<time>; restoring it is a copy.
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fishbowl import controller, genome as G, sandbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=controller.HIDDEN // 4, help="how many hidden units the stroke hits")
    args = parser.parse_args()
    ck = sandbox.load_checkpoint()
    if ck is None:
        print("No checkpoint -- nothing to lesion.")
        return 1
    g = G.Genome.from_dict(ck["genome"])
    b, rng = g.brain, random.Random()
    n = max(1, min(controller.HIDDEN, args.units))
    start = rng.randrange(controller.HIDDEN - n + 1)
    region = list(range(start, start + n))

    # Every weight touching the region: into it (inputs, recurrent from
    # anywhere), out of it (recurrent to everywhere, motor readout), its biases.
    slots = [(b.weights_ih, h, j) for h in region for j in range(len(b.weights_ih[h]))]
    slots += [(b.weights_hh, h, j) for h in region for j in range(controller.HIDDEN)]
    slots += [(b.weights_hh, i, h) for i in range(controller.HIDDEN) if i not in region for h in region]
    slots += [(b.weights_ho, o, h) for o in range(len(b.weights_ho)) for h in region]
    values = [m[r][c] for m, r, c in slots]
    rng.shuffle(values)
    for (m, r, c), v in zip(slots, values):
        m[r][c] = v
    biases = [b.bias_h[h] for h in region]
    rng.shuffle(biases)
    for h, v in zip(region, biases):
        b.bias_h[h] = v

    backup = sandbox.CHECKPOINT_PATH.with_name(f"checkpoint.json.bak-stroke-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(sandbox.CHECKPOINT_PATH, backup)
    ck["genome"] = g.to_dict()
    ck["stroke"] = {"units": region, "generation": ck.get("total_generation"), "at": time.time()}
    sandbox.save_checkpoint(ck)
    print(f"Stroke at generation {ck.get('total_generation')}: hidden units {region[0]}-{region[-1]} "
          f"({len(slots)} weights and {n} biases scrambled among themselves). Pre-stroke brain: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
