"""
Seed a lineage's born-off capacities once, and let evolution prune them
(a 2026-09-29 panel, 12-2: brains overproduce and prune -- Changeux's
selective stabilization, Edelman -- and Dennett's conditions: every seed is
priced, so what doesn't pay gets pruned, and every seed value comes from the
trait's own birth or mutation distribution a few steps along, never tuned).

    python tools/seed.py <state dir>     # with its organism stopped; backs up first

It edits state/checkpoint.json's genome once (marked by state/seeded.txt); the same
deterministic draws on every host, so the race's replicates stay comparable.

Seeded (born-off traits switched on, each at a value its own mutations reach):
  - pyramidal apical gain 0.3 (three trait steps) -- Friston's coincidence units
  - sleep distillation: plasticity at its birth distribution's median
  - imagery and recall on; a 3-scene library; a 32-look sleep test set
  - prey sense 3 (velocity), plant sense 1
  - 2 archetype heads, taught by classes drawn at random
  - the newer senses' input weights (mismatch, recall, protein, velocity,
    own speed, uncertainty, ground, parallax, archetypes): each unwired
    column gets small weights at the brain's own mutation step (0.05)
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20260929
TAG = "2026-09-29"


def seed_genome(g, rng: random.Random) -> list[str]:
    from fishbowl import genome as G, prey as prey_lib
    from fishbowl.controller import (ARCHETYPE_INPUTS, GROUND_INPUTS, MISMATCH_INPUTS, PARALLAX_INPUTS, PROTEIN_INPUT,
                                     RECALL_INPUTS, SPEED_INPUTS, UNCERTAINTY_INPUT, VELOCITY_INPUTS)
    done = []
    if g.brain.apical == 0.0:
        g.brain.apical = 3 * G.TRAIT_SIGMA
        done.append(f"apical {g.brain.apical:.2f}")
    if g.plasticity == 0.0:
        g.plasticity = math.sqrt(G.LEARNING_MIN * G.LEARNING_MAX)
        done.append(f"plasticity {g.plasticity:.3f}")
    for name, value in (("imagery", 1), ("recall", 1)):
        if not getattr(g, name):
            setattr(g, name, value)
            done.append(f"{name} on")
    if g.scenes < 3:
        g.scenes = 3
        done.append("scenes 3")
    if g.sleep_set < 32:
        g.sleep_set = 32
        done.append("sleep set 32")
    if g.prey_sense < 3:
        g.prey_sense = 3
        done.append("prey sense 3")
    if g.plant_sense < 1:
        g.plant_sense = 1
        done.append("plant sense 1")
    if g.archetypes < 2:
        classes = sorted(prey_lib.PREY_CLASSES) + [prey_lib.PLANT_CLASS]
        for k in range(g.archetypes, 2):
            g.archetype_classes[k] = rng.choice(classes)
        g.archetypes = 2
        done.append(f"archetypes {g.archetype_classes[:2]}")
    cols = (list(MISMATCH_INPUTS) + list(RECALL_INPUTS) + [PROTEIN_INPUT] + list(VELOCITY_INPUTS) + list(SPEED_INPUTS)
            + [UNCERTAINTY_INPUT] + list(GROUND_INPUTS) + list(PARALLAX_INPUTS) + list(ARCHETYPE_INPUTS))
    w = g.brain.weights_ih
    wired = 0
    for c in cols:
        if c < w.shape[1] and not w[:, c].any():
            w[:, c] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
            wired += 1
    if wired:
        done.append(f"{wired} sense inputs wired")
    return done


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    state = Path(sys.argv[1])
    path = state / "checkpoint.json"
    mark = state / "seeded.txt"  # its own file: a checkpoint save would drop a mark inside it
    if mark.exists():
        print(f"already seeded ({mark.read_text().strip()}): nothing to do -- evolution prunes from here")
        return 0
    c = json.loads(path.read_text(encoding="utf-8-sig"))
    from fishbowl.genome import Genome
    backup = state / f"backup-{time.strftime('%Y%m%d-%H%M%S')}-before-seed"
    backup.mkdir()
    for f in ("checkpoint.json", "checkpoint.prev.json"):
        if (state / f).exists():
            shutil.copy2(state / f, backup / f)
    g = Genome.from_dict(c["genome"])
    done = seed_genome(g, random.Random(SEED))
    c["genome"] = g.to_dict()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(c), encoding="utf-8")
    tmp.replace(path)
    (state / "checkpoint.prev.json").unlink(missing_ok=True)
    mark.write_text(f"{TAG}: " + "; ".join(done) + "\n", encoding="utf-8")
    print("seeded:", "; ".join(done) or "nothing needed", f"(backup: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
