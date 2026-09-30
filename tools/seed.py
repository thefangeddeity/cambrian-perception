"""
Seed a lineage's born-off capacities once, and let evolution prune them
(a 2026-09-29 panel, 12-2: brains overproduce and prune -- Changeux's
selective stabilization, Edelman -- and Dennett's conditions: every seed is
priced, so what doesn't pay gets pruned, and every seed value comes from the
trait's own birth or mutation distribution a few steps along, never tuned).

    python tools/seed.py <state dir>     # with its organism stopped; backs up first
    python tools/seed.py <state dir> --random-draws   # also redraw the seeded values at random

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


def seed_colliculus(g, rng: random.Random) -> list[str]:
    """The collicular map (2026-09-29): its six weights at draws a few trait
    steps wide, and its three inputs wired at the brain's mutation step."""
    from fishbowl import genome as G
    from fishbowl.controller import COLLICULUS_INPUTS
    done = []
    if not any(g.colliculus):
        g.colliculus = [rng.gauss(0.0, 3 * G.TRAIT_SIGMA) for _ in range(6)]
        done.append("colliculus " + ", ".join(f"{w:+.2f}" for w in g.colliculus))
    w = g.brain.weights_ih
    wired = 0
    for c in COLLICULUS_INPUTS:
        if c < w.shape[1] and not w[:, c].any():
            w[:, c] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
            wired += 1
    return done + ([f"{wired} collicular inputs wired"] if wired else [])  # (it used to say so even when none were)


def seed_terrain(g, rng: random.Random) -> list[str]:
    """Its terrain input (2026-09-29): wired at the brain's mutation step."""
    from fishbowl.controller import TERRAIN_INPUT
    w = g.brain.weights_ih
    if TERRAIN_INPUT < w.shape[1] and not w[:, TERRAIN_INPUT].any():
        w[:, TERRAIN_INPUT] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
        return ["terrain input wired"]
    return []


def seed_nearness(g, rng: random.Random) -> list[str]:
    """How near what it looks at is (2026-09-29): wired at the brain's mutation step."""
    from fishbowl.controller import NEARNESS_INPUT
    w = g.brain.weights_ih
    if NEARNESS_INPUT < w.shape[1] and not w[:, NEARNESS_INPUT].any():
        w[:, NEARNESS_INPUT] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
        return ["nearness input wired"]
    return []


def seed_everything(g, rng: random.Random) -> list[str]:
    """Everything else that can evolve and is still off (2026-09-29; the
    panel, with Dennett's compromise: seeded, but at random): each switched on
    by a random walk of three of its own mutation steps from off. What a
    lineage had seeded and evolution since pruned stays pruned (seed sets
    apply once); all of it is priced, so what doesn't pay is pruned again."""
    walk = lambda steps: [rng.choice((-1, 1)) for _ in range(steps)]  # noqa: E731
    from fishbowl import genome as G
    done = []
    if g.replay_backup == 0.0:
        g.replay_backup = float(min(1.0, abs(sum(rng.gauss(0.0, G.TRAIT_SIGMA) for _ in range(3)))))
        done.append(f"replay backup {g.replay_backup:.2f}")
    if g.dream_steps == 0:
        g.dream_steps = max(1, abs(sum(walk(3))))  # a walk of three +-1 steps, reflected at off
        done.append(f"dream steps {g.dream_steps}")
    if g.colour_channels == 0:
        g.colour_channels = rng.choice((1, 2))
        done.append(f"colour channels {g.colour_channels}")
    # its brain: any input still unwired, wired at the mutation step
    w = g.brain.weights_ih
    cols = [c for c in range(w.shape[1]) if not w[:, c].any()]
    for c in cols:
        w[:, c] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
    if cols:
        done.append(f"{len(cols)} more inputs wired")
    # its perception trees: constant leaves become receptors at random (each
    # with even odds, at least one per tree) -- a tree reading its eye can
    # grow pools, and pools lines; one that reads only constants never can
    n_side = g.receptors
    for name, tree in g.trees.items():
        leaves, stack = [], [tree]
        while stack:
            node = stack.pop()
            if node.kind == "const":
                leaves.append(node)
            stack.extend(node.children)
        if not leaves:
            continue
        pick = [n for n in leaves if rng.random() < 0.5] or [rng.choice(leaves)]
        for node in pick:
            node.kind, node.index, node.value = "cell", rng.randrange(1 + g.colour_channels), 0.0
            node.kx, node.ky = rng.randrange(n_side) - n_side // 2, rng.randrange(n_side) - n_side // 2
        done.append(f"tree {name}: {len(pick)} receptor leaves")
    return done


def random_draws(g, rng: random.Random) -> list[str]:
    """What the seed sets set to one value (so the fleet's replicate lineages
    stay comparable), drawn at random from each trait's own distribution
    instead -- as a young brain overproduces (Huttenlocher; Changeux) and its
    prices prune: apical gain a three-step walk, plasticity its birth
    distribution, 1-4 archetype heads of random classes (the heads' structural
    range), scenes a three-step walk from 1. A fresh install's founder gets these
    (run_vision.py); `--random-draws` gives them to a living lineage."""
    from fishbowl import genome as G, prey as prey_lib
    from fishbowl.mushroom import MAX_HEADS
    g.brain.apical = abs(sum(rng.gauss(0.0, G.TRAIT_SIGMA) for _ in range(3)))
    g.plasticity = float(G.LEARNING_MIN * (G.LEARNING_MAX / G.LEARNING_MIN) ** rng.random())
    classes = sorted(prey_lib.PREY_CLASSES) + [prey_lib.PLANT_CLASS]
    g.archetypes = rng.randint(1, MAX_HEADS)
    g.archetype_classes = [rng.choice(classes) for _ in range(MAX_HEADS)]
    g.scenes = 1 + sum(rng.random() < 0.5 for _ in range(3))  # three of its +1 / -1 mutation steps from 1, reflected at 1
    return [f"apical {g.brain.apical:.2f}", f"plasticity {g.plasticity:.3f}",
            f"{g.archetypes} archetype heads {g.archetype_classes[:g.archetypes]}", f"{g.scenes} scenes"]


def seed_maturation(g, rng: random.Random) -> list[str]:
    """Maturation (2026-09-29): switched on at a random draw from its birth
    distribution (log-uniform, 1 - 30 nights)."""
    from fishbowl import genome as G
    if g.maturation == 0.0:
        g.maturation = float(G.MATURATION_MIN * (G.MATURATION_BIRTH_MAX / G.MATURATION_MIN) ** rng.random())
        return [f"maturation {g.maturation:.1f} nights"]
    return []


def seed_maturation_nights(g, rng: random.Random) -> list[str]:
    """Maturation's unit became nights (2026-09-29 panel): a lineage seeded in
    the old unit (lessons) draws again, at random, in nights."""
    from fishbowl import genome as G
    if g.maturation > G.MATURATION_BIRTH_MAX:
        g.maturation = float(G.MATURATION_MIN * (G.MATURATION_BIRTH_MAX / G.MATURATION_MIN) ** rng.random())
        return [f"maturation {g.maturation:.1f} nights"]
    return []


def seed_felt_terrain(g, rng: random.Random) -> list[str]:
    """Its terrain head (2026-09-29): switched on; its input wired at the brain's mutation step."""
    from fishbowl.controller import FELT_NEARNESS_INPUT
    done = []
    if not g.felt_terrain:
        g.felt_terrain = 1
        done.append("felt terrain on")
    w = g.brain.weights_ih
    if FELT_NEARNESS_INPUT < w.shape[1] and not w[:, FELT_NEARNESS_INPUT].any():
        w[:, FELT_NEARNESS_INPUT] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
        done.append("felt nearness input wired")
    return done


def seed_kenyon_cells(g, rng: random.Random) -> list[str]:
    """Its mushroom body, if it has none (2026-09-29: random founders were born
    with none, and the "everything" set missed it): a random walk of three of
    its own growth steps from one step, 64 - 256 Kenyon cells; priced as ever."""
    from fishbowl import genome as G
    if g.kc == 0:
        g.kc = G.KC_STEP * (1 + sum(rng.random() < 0.5 for _ in range(3)))
        return [f"{g.kc} Kenyon cells"]
    return []


def seed_looking_ahead(g, rng: random.Random) -> list[str]:
    """Seeing the present despite its lag (2026-09-29): its terrain head taught
    toward the next look, an extrapolation drawn near exact compensation (1 +
    three of its own steps), and its tau input wired -- for evolution to prune."""
    from fishbowl import genome as G
    from fishbowl.controller import CONTACT_INPUT
    done = []
    if not g.lookahead:
        g.lookahead = 1
        done.append("lookahead on")
    if g.extrapolation == 0.0:
        g.extrapolation = float(min(G.EXTRAPOLATION_MAX, max(0.0, 1.0 + sum(rng.gauss(0.0, G.TRAIT_SIGMA) for _ in range(3)))))
        done.append(f"extrapolation {g.extrapolation:.2f}")
    w = g.brain.weights_ih
    if CONTACT_INPUT < w.shape[1] and not w[:, CONTACT_INPUT].any():
        w[:, CONTACT_INPUT] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
        done.append("tau input wired")
    return done


def seed_rotation(g, rng: random.Random) -> list[str]:
    """Feeling its rotation (2026-09-29): its compass on; its turning, tilting
    and heading inputs wired at the brain's mutation step."""
    from fishbowl.controller import HEADING_INPUTS, TILT_INPUT, TURN_INPUT
    done = []
    if not g.compass:
        g.compass = 1
        done.append("compass on")
    w = g.brain.weights_ih
    wired = 0
    for c in (TURN_INPUT, TILT_INPUT) + HEADING_INPUTS:
        if c < w.shape[1] and not w[:, c].any():
            w[:, c] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
            wired += 1
    if wired:
        done.append(f"{wired} rotation inputs wired")
    return done


def seed_entorhinal(g, rng: random.Random) -> list[str]:
    """Its entorhinal map (2026-09-29): on; its speed, acceleration and place-value inputs wired."""
    from fishbowl.controller import ACCELERATION_INPUT, EGO_SPEED_INPUT, PLACE_VALUE_INPUT
    done = []
    if not g.entorhinal:
        g.entorhinal = 1
        done.append("entorhinal map on")
    w = g.brain.weights_ih
    wired = 0
    for c in (EGO_SPEED_INPUT, ACCELERATION_INPUT, PLACE_VALUE_INPUT):
        if c < w.shape[1] and not w[:, c].any():
            w[:, c] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
            wired += 1
    if wired:
        done.append(f"{wired} map inputs wired")
    return done


def seed_frames(g, rng: random.Random) -> list[str]:
    """Its frames of reference (2026-09-29): its riding input wired (the map
    itself is early vision, always on)."""
    from fishbowl.controller import RIDING_INPUT
    w = g.brain.weights_ih
    if RIDING_INPUT < w.shape[1] and not w[:, RIDING_INPUT].any():
        w[:, RIDING_INPUT] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
        return ["riding input wired"]
    return []


def seed_flow_teacher(g, rng: random.Random) -> list[str]:
    """Flow-taught terrain (2026-09-29): its two teachers weighed equally (the
    ignorance prior), then a few mutation steps along the trait at random."""
    from fishbowl import genome as G
    if g.flow_teacher > 0.0:
        return []
    g.flow_teacher = float(min(1.0, max(0.0, 0.5 + sum(rng.gauss(0.0, G.TRAIT_SIGMA) for _ in range(3)))))
    return [f"flow teacher {g.flow_teacher:.2f}"]


def seed_v4(g, rng: random.Random) -> list[str]:
    """Its V4 (2026-09-29): colour constancy gained at the human time course
    (Fairchild & Reniff) a few steps along, its texture statistics on, the
    texture gradient teaching at the ignorance prior a few steps along, and its
    texture inputs wired."""
    import math
    from fishbowl import genome as G
    from fishbowl.controller import TEXTURE_INPUTS
    done = []
    if g.colour_constancy <= 0.0:
        g.colour_constancy = float(min(G.COLOUR_ADAPT_MAX_S, G.COLOUR_ADAPT_SEED_S * math.exp(sum(rng.gauss(0.0, G.LEARNING_SIGMA) for _ in range(3)))))
        done.append(f"colour constancy {g.colour_constancy:.1f} s")
    if not g.texture:
        g.texture = 1
        done.append("texture on")
    if g.texture_teacher <= 0.0:
        g.texture_teacher = float(min(1.0, max(0.0, 0.5 + sum(rng.gauss(0.0, G.TRAIT_SIGMA) for _ in range(3)))))
        done.append(f"texture teacher {g.texture_teacher:.2f}")
    w = g.brain.weights_ih
    wired = 0
    for c in TEXTURE_INPUTS:
        if c < w.shape[1] and not w[:, c].any():
            w[:, c] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
            wired += 1
    if wired:
        done.append(f"{wired} texture inputs wired")
    return done


def seed_strangeness(g, rng: random.Random) -> list[str]:
    """Its strangeness sense (2026-09-29): wired at the brain's mutation step
    (the gate that holds its learning outside its model is always on)."""
    from fishbowl.controller import STRANGENESS_INPUT
    w = g.brain.weights_ih
    if STRANGENESS_INPUT < w.shape[1] and not w[:, STRANGENESS_INPUT].any():
        w[:, STRANGENESS_INPUT] = [rng.gauss(0.0, 0.05) for _ in range(w.shape[0])]
        return ["strangeness input wired"]
    return []


def seed_reproduction(g, rng: random.Random) -> list[str]:
    """Reproduction (2026-09-29): kappa at the Add-my-Pet collection's median
    across ~3000 species, 0.9 (Marques et al. 2018, PLOS Comput Biol; 0.8 is
    only DEB's prior). Exactly, with no random steps: steps clipped at 1 would
    seed about a quarter of lineages that can never lay (and kappa changes only
    at laying). Its variation begins with its first egg."""
    if g.kappa < 1.0:
        return []
    g.kappa = 0.9
    return ["kappa 0.90"]


# Each seed set applies once per lineage (state/seeded.txt lists those applied).
SEED_SETS = {"2026-09-29": seed_genome, "2026-09-29 colliculus": seed_colliculus, "2026-09-29 terrain": seed_terrain,
             "2026-09-29 nearness": seed_nearness, "2026-09-29 everything": seed_everything,
             "2026-09-29 maturation": seed_maturation, "2026-09-29 maturation in nights": seed_maturation_nights,
             "2026-09-29 kenyon cells": seed_kenyon_cells, "2026-09-29 felt terrain": seed_felt_terrain,
             "2026-09-29 looking ahead": seed_looking_ahead, "2026-09-29 rotation": seed_rotation,
             "2026-09-29 entorhinal": seed_entorhinal, "2026-09-29 frames of reference": seed_frames,
             "2026-09-29 flow-taught terrain": seed_flow_teacher, "2026-09-29 V4": seed_v4,
             "2026-09-29 strangeness": seed_strangeness, "2026-09-29 reproduction": seed_reproduction}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    draws = "--random-draws" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    state = Path(args[0])
    path = state / "checkpoint.json"
    mark = state / "seeded.txt"  # its own file: a checkpoint save would drop a mark inside it
    applied = {line.split(":")[0] for line in mark.read_text(encoding="utf-8").splitlines() if line.strip()} if mark.exists() else set()
    todo = [name for name in SEED_SETS if name not in applied]
    if not todo and not draws:
        print("every seed set already applied: nothing to do -- evolution prunes from here")
        return 0
    c = json.loads(path.read_text(encoding="utf-8-sig"))
    from fishbowl.genome import Genome
    backup = state / f"backup-{time.strftime('%Y%m%d-%H%M%S')}-before-seed"
    backup.mkdir()
    for f in ("checkpoint.json", "checkpoint.prev.json"):
        if (state / f).exists():
            shutil.copy2(state / f, backup / f)
    g = Genome.from_dict(c["genome"])
    lines = []
    for name in todo:
        done = SEED_SETS[name](g, random.Random(f"{SEED}:{name}"))
        lines.append(f"{name}: " + "; ".join(done or ["nothing needed"]))
    if draws:
        lines.append(f"random draws {time.strftime('%Y-%m-%d %H:%M')}: " + "; ".join(random_draws(g, random.Random(time.time_ns()))))
    c["genome"] = g.to_dict()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(c), encoding="utf-8")
    tmp.replace(path)
    (state / "checkpoint.prev.json").unlink(missing_ok=True)
    with open(mark, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("seeded:", " | ".join(lines), f"(backup: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
