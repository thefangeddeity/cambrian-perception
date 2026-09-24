#!/usr/bin/env python3
from __future__ import annotations

"""
Entry point: evolves a genome against a real (curated clip or live
device) video source, using the innate reflex/conspec signals purely
to grade fitness -- never as genome input (see README's fishbowl
boundary). Frames are loaded into memory once per run (never written
to disk -- see video_source.py), and every generation replays the
SAME in-memory clip from the SAME starting fovea position, so accept/
reject comparisons are fair.

Usage:
    python3 run_vision.py <video_path_or_device> [--generations N] [--seconds S]
"""

import argparse
import math
import random
import sys

import numpy as np

from fishbowl import conspec, fovea, genome as G, reflexes, sandbox, video_source
from fishbowl.retina import N_CELLS

# How much each reflex/drive contributes to total fitness -- loom
# weighted heaviest, matching the real threat/food asymmetry discussed
# while designing this (missing a threat costs more than missing an
# opportunity). Not tuned against real data yet; a real, named,
# starting guess, not a claim of correctness.
SIGNAL_WEIGHTS = {"luminance_change": 0.5, "optomotor": 0.75, "loom": 2.0}
CONSPEC_WEIGHT = 1.5


def _correlate(signal: np.ndarray, response: np.ndarray) -> float:
    if signal.std() < 1e-9 or response.std() < 1e-9:
        return 0.0
    corr = float(np.corrcoef(signal, response)[0, 1])
    return corr if math.isfinite(corr) else 0.0


def evaluate_genome(g: G.Genome, frames: list[np.ndarray], habituation_discount: float) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """
    Returns (fitness, breakdown, conspec_signal, loom_signal) -- the
    caller needs both again after an accept, to actually advance
    habituation state over the trajectory the organism just really
    took (see run()'s own comment on why that has to happen AFTER the
    accept/reject decision, not before it). habituation_discount is
    read-only here (this generation's current discount, applied to
    the reward), never mutated by this function.
    """
    state = fovea.FoveaState()
    retina_vectors = []
    responses = []

    for frame in frames:
        v = fovea.extract(frame, state)
        retina_vectors.append(v)
        vb = v[None, :]
        response = float(g.evaluate("response", vb)[0])
        pan = float(g.evaluate("pan", vb)[0])
        tilt = float(g.evaluate("tilt", vb)[0])
        responses.append(response)
        state = fovea.step(state, pan, tilt)

    vectors = np.array(retina_vectors)
    responses = np.array(responses)

    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(responses)):
        return float("-inf"), {}, np.zeros(len(frames)), np.zeros(len(frames))

    signals = reflexes.all_signals(vectors)
    cs = conspec.conspec_signal(vectors)

    fitness = 0.0
    breakdown = {}
    for name, weight in SIGNAL_WEIGHTS.items():
        corr = _correlate(signals[name], responses)
        fitness += weight * corr
        breakdown[name] = corr

    drive = conspec.drive_fitness(cs, responses)
    discounted_drive = drive * habituation_discount
    fitness += CONSPEC_WEIGHT * discounted_drive
    breakdown["conspec_drive"] = discounted_drive
    breakdown["loom_max"] = float(signals["loom"].max())

    return fitness, breakdown, cs, signals["loom"]


def run(source: str, limits: sandbox.Limits, n_vars: int = N_CELLS) -> None:
    print(f"Loading real frames from {source!r} into memory (never written to disk)...")
    frames = list(video_source.read_frames(source, stride=2, max_frames=600))
    print(f"  {len(frames)} frames loaded.")
    if len(frames) < 10:
        print("Not enough real frames to evolve against -- aborting.")
        return

    rng = random.Random(20260924)
    box = sandbox.Sandbox(limits)
    habituation = conspec.Habituation()

    genome = G.random_genome(rng, n_vars=n_vars)
    best_fitness, _, best_conspec, best_loom = evaluate_genome(genome, frames, habituation.discount)
    print(f"Initial fitness: {best_fitness:.4f}")

    # The genome's very first real trajectory also counts as real
    # exposure -- habituation starts accumulating from generation
    # zero, not only after the first accepted mutation.
    for c, l in zip(best_conspec, best_loom):
        habituation.observe(conspec_present=c > 0.05, loom_value=l)

    # Ratcheting margin -- the user's own "arms race" framing: success
    # should raise the bar, not just clear a fixed one. This is a
    # real, honest v1 (a slowly-shrinking required margin), not the
    # full escalating-precision/timing vision discussed -- said
    # plainly, a real first step to build on, not the whole thing.
    margin = 0.05

    while box.should_continue():
        box.generation += 1
        candidate = genome.clone()
        ceiling_reason = None

        if rng.random() < 0.15:
            candidate.mutate_weights(rng)
        else:
            channel, applied = candidate.mutate_task(
                rng, max_nodes=limits.max_tree_nodes, max_depth=limits.max_tree_depth,
            )
            if applied == "noop":
                ceiling_reason = "tree_size_or_depth"

        box.note_ceiling(ceiling_reason)

        candidate_fitness, breakdown, candidate_conspec, candidate_loom = evaluate_genome(
            candidate, frames, habituation.discount,
        )

        accepted = math.isfinite(candidate_fitness) and candidate_fitness > best_fitness + margin
        if accepted:
            genome = candidate
            best_fitness = candidate_fitness
            margin = max(0.005, margin * 0.995)
            # Habituation only ever advances from the trajectory the
            # organism actually, really took (the ACCEPTED genome's
            # own real path through the clip) -- not from every
            # rejected candidate's hypothetical path, which never
            # happened and shouldn't count as real exposure.
            for c, l in zip(candidate_conspec, candidate_loom):
                habituation.observe(conspec_present=c > 0.05, loom_value=l)

        box.log_generation({
            "accepted": accepted,
            "fitness": candidate_fitness if math.isfinite(candidate_fitness) else None,
            "best_fitness": best_fitness,
            "breakdown": breakdown,
            "mutation_weights": dict(genome.mutation_weights),
            "meta_mutation_rate": genome.meta_mutation_rate,
            "margin": margin,
            "habituation_exposure": round(habituation.exposure, 4),
            "habituation_discount": round(habituation.discount, 4),
        })

        if box.generation % 25 == 0:
            print(
                f"  gen {box.generation:>5} best_fitness={best_fitness:.4f} "
                f"margin={margin:.4f} habituation={habituation.exposure:.3f}"
            )

    print(f"Stopped after {box.generation} generations, {round(__import__('time').perf_counter() - box.start_time, 1)}s.")
    print(f"Final best_fitness: {best_fitness:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file path or device (e.g. /dev/video0)")
    parser.add_argument("--generations", type=int, default=5000)
    parser.add_argument("--seconds", type=float, default=1800.0)
    args = parser.parse_args()

    limits = sandbox.Limits(max_generations=args.generations, max_wallclock_seconds=args.seconds)
    run(args.source, limits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
