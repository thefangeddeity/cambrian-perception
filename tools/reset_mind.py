#!/usr/bin/env python3
from __future__ import annotations

"""
Give the organism a fresh mind: a new random brain and perception tree, and
the search's learned mutation habits back to uniform. Keeps its body -- the
current body state and its inherited traits (gaze size, tempo, colour
vision, stabilizer) -- and its memory of the room.

For when the body or senses changed enough that the old mind is adapted to
a world that no longer exists (e.g. Stage A's gut, reserve and sleep): a
fresh mind learns senses, body and sleep together, instead of treating them
as zero-weight add-ons to old habits.

Run only while the organism is stopped (it rewrites state/checkpoint.json;
CAMBRIAN state layout as in fishbowl/sandbox.py). The old checkpoint is kept
next to it as checkpoint.json.bak-mind-<time>, and restoring it is a copy.
"""

import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fishbowl import genome as G, sandbox  # noqa: E402


def main() -> int:
    ck = sandbox.load_checkpoint()
    if ck is None:
        print("No checkpoint -- nothing to reset (a first run starts fresh anyway).")
        return 1
    old = G.Genome.from_dict(ck["genome"])
    fresh = G.random_genome(random.Random(), n_vars=int(ck.get("n_vars", old.n_vars)))
    fresh.fovea_fraction, fresh.pace = old.fovea_fraction, old.pace
    fresh.colour_channels, fresh.stabilizer = old.colour_channels, old.stabilizer

    backup = sandbox.CHECKPOINT_PATH.with_name(f"checkpoint.json.bak-mind-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(sandbox.CHECKPOINT_PATH, backup)
    ck["genome"] = fresh.to_dict()
    ck["margin"] = 0.05
    ck["best_fitness"] = 0.0         # re-scored on the first frames of the next run
    ck["peak_fitness_seen"] = -1e9   # a new mind's own record starts now
    ck["mind_reset_at_generation"] = ck.get("total_generation")
    sandbox.save_checkpoint(ck)
    print(f"Fresh mind at generation {ck.get('total_generation')}: new brain "
          f"({len(fresh.brain.weights_ih[0])} inputs, {len(fresh.brain.weights_ho)} outputs, no channels) and "
          f"perception tree; body and traits kept (gaze {fresh.fovea_fraction:.2f}, pace {fresh.pace}, "
          f"colour {fresh.colour_channels}, stabilizer {fresh.stabilizer:.2f}). Old mind: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
