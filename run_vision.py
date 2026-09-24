#!/usr/bin/env python3
from __future__ import annotations

"""
Entry point: evolves a genome against real curated clips, using the
innate reflex/conspec signals purely to grade fitness -- never as
genome input (see README's fishbowl boundary). Frames are loaded into
memory once per run (never written to disk -- see video_source.py),
and every generation replays the SAME in-memory clip from the SAME
starting fovea position, so accept/reject comparisons are fair.

Meant to run for YEARS (the user's own framing), not one sitting -- so this
is intentionally still a BOUNDED process (sandbox.Limits, checked
before each generation, same "never trust an unbounded loop"
discipline as everything else here), meant to be restarted repeatedly
by an outer supervisor (systemd, Restart=always -- see deploy/
cambrian-perception.service). Each restart:
  - resumes the genome/habituation/fitness state from state/
    checkpoint.json rather than starting from a fresh random genome
    (fishbowl/sandbox.py's save/load_checkpoint -- real long-term
    memory, leveraging Tanzania's disk instead of trying to keep years
    of state in RAM or re-discovering everything each time).
  - picks the NEXT source in rotation (persisted in the checkpoint
    too) -- one clip per bounded run keeps each run's own accept/
    reject comparisons fair, while rotating across runs gives real
    variety over a long lifetime instead of plateauing against one
    90-second loop forever. In "live" mode (source == "live") this
    rotates across LIVE_SOURCES and RE-RESOLVES the real live stream
    fresh each run (see _resolve_live_url) -- genuinely new real
    content every restart, never a cached/downloaded file, matching
    the user's own "watch actual live feed, not local cache" correction.

Usage:
    python3 run_vision.py <media_dir_or_single_file> [--generations N] [--seconds S]
"""

import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from fishbowl import conspec, fovea, genome as G, reflexes, sandbox, video_source
from fishbowl.retina import GRID, N_CELLS

# How much each reflex/drive contributes to total fitness -- loom
# weighted heaviest, matching the real threat/food asymmetry discussed
# while designing this (missing a threat costs more than missing an
# opportunity). Not tuned against real data yet; a real, named,
# starting guess, not a claim of correctness.
SIGNAL_WEIGHTS = {"luminance_change": 0.5, "optomotor": 0.75, "loom": 2.0}
CONSPEC_WEIGHT = 1.5

# The other boundary of the corridor -- the user's own framing: a deep-sea
# vent shrimp doesn't just flee scalding water, it also has to avoid
# drifting into the freezing water behind it. loom is the "scalding"
# side (big, sudden, real threat -- weighted heaviest above,
# deliberately never fully avoidable by just sitting still). This is
# the "freezing" side: a real penalty for the fovea settling somewhere
# nothing happens for a sustained stretch ("dead field... dead cold"),
# pushing the pan/tilt channels to actively seek information-rich
# regions rather than just passively dodge loom by staring at a wall.
DEAD_FIELD_ACTIVITY_THRESHOLD = 0.01
DEAD_FIELD_WINDOW = 10
DEAD_FIELD_PENALTY_WEIGHT = 1.0


# Curiosity -- User: "I don't see it 'looking' sideways, no curiosity
# yet?" Confirmed by real data (evolution_log.json's own breakdown
# keys): nothing in the fitness function ever rewarded WHERE the
# fovea goes, only how well the response correlates with reflex
# signals wherever it already happened to be sitting. This is
# deliberately separate from the earlier-discussed resolution-growth
# "curiosity" (a different, still-queued idea) -- this one is real,
# simple, count-based exploration: reward for covering distinct
# regions of the reachable pan/tilt space over a run, not just for
# reacting well to whatever's already in view. Same general shape as
# count-based exploration bonuses in real intrinsic-motivation RL
# literature, kept deliberately simple here (coverage fraction of a
# coarse grid, not a full novelty model).
CURIOSITY_GRID = 5
CURIOSITY_WEIGHT = 0.75


def _curiosity_score(positions: list[tuple[float, float]]) -> float:
    if not positions:
        return 0.0
    visited = set()
    for cx, cy in positions:
        cell = (min(CURIOSITY_GRID - 1, int(cx * CURIOSITY_GRID)),
                min(CURIOSITY_GRID - 1, int(cy * CURIOSITY_GRID)))
        visited.add(cell)
    return len(visited) / float(CURIOSITY_GRID * CURIOSITY_GRID)


def _dead_field_penalty(signals: dict[str, np.ndarray]) -> float:
    activity = signals["optomotor"] + signals["luminance_change"]
    dead = activity < DEAD_FIELD_ACTIVITY_THRESHOLD
    if len(dead) < DEAD_FIELD_WINDOW:
        return 0.0
    # Rolling count of consecutive dead frames -- a brief lull isn't
    # penalized (real scenes have quiet moments), only a SUSTAINED
    # stretch is, mirroring loom's own "this has to build over time"
    # shape rather than firing on every single quiet frame.
    run_length = 0
    worst_run = 0
    for is_dead in dead:
        run_length = run_length + 1 if is_dead else 0
        worst_run = max(worst_run, run_length)
    if worst_run < DEAD_FIELD_WINDOW:
        return 0.0
    return float(worst_run - DEAD_FIELD_WINDOW + 1) / len(dead)

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")

# User: "switch it to watch actual live feed, not local cache; I want a
# stimulus-rich feed." The same real, curated, long-running public
# streams tools/fetch_curriculum_videos.py already vetted -- but the
# stage3_loom pair specifically (already the "more action" tier in
# that curriculum staging), not the calmer stage1 feeder-cam, since
# "stimulus-rich" is the explicit ask here. Watched LIVE (resolved
# fresh every run via _resolve_live_url below), never downloaded --
# genuinely different real content every restart instead of the same
# cached 90s loop replaying forever.
LIVE_SOURCES = [
    ("cat_livestream", "https://www.youtube.com/watch?v=08_mWdxig2A"),
    ("pixcams_wildlife", "https://www.youtube.com/watch?v=XfNhPa26fP8"),
]


def _clip_display_name(source: str, clip_path: str) -> str:
    """
    A short, human-readable label for whatever's actually being
    watched right now -- User: "make sure display page shows what it's
    watching." For live mode this is the curated name from
    LIVE_SOURCES (e.g. "cat_livestream"), not the raw URL; for a local
    file it's the filename without extension.
    """
    if source == "live":
        for name, url in LIVE_SOURCES:
            if url == clip_path:
                return name
        return clip_path
    return Path(clip_path).stem


def _resolve_live_url(watch_url: str) -> str:
    """
    Resolves a live stream's watch page to the real, currently-
    playable direct URL via yt-dlp's `-g` (print URL, never
    downloads -- no file ever touches disk, unlike
    fetch_curriculum_videos.py's own use of yt-dlp). Deliberately
    lives HERE, in run_vision.py, not inside fishbowl/ -- the exact
    same boundary fetch_curriculum_videos.py already draws for
    itself: a fixed, human-configured piece of infrastructure (which
    stream to watch), not the organism's own evolved logic, so the
    README's "no subprocess" rule (which is scoped to fishbowl/
    specifically) doesn't apply to this call.

    Re-resolved fresh at the start of every bounded run, never
    cached -- a resolved CDN URL expires after a while regardless, and
    the whole point of watching live is a genuinely fresh window every
    run rather than the same footage replayed.

    Uses the yt-dlp installed in the SAME venv as this process
    (Path(sys.executable).parent / "yt-dlp") rather than relying on
    PATH -- systemd's ExecStart invokes the venv's python directly
    without activating the venv, so plain "yt-dlp" would not resolve
    under systemd even though it works fine from an interactive shell.
    """
    yt_dlp = Path(sys.executable).parent / "yt-dlp"
    if not yt_dlp.exists():
        yt_dlp = Path("yt-dlp")  # fall back to PATH (e.g. local dev run)
    # Video-only, not "best" -- confirmed live via --list-formats: these
    # streams serve video and audio as SEPARATE adaptive formats, no
    # muxed format exists at all, so a combined "best[height<=480]"
    # selector matches nothing and fails outright. We only ever read
    # frames (see video_source.py), so there's no reason to resolve
    # audio in the first place.
    result = subprocess.run(
        [str(yt_dlp), "-g", "-f", "bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/bestvideo", watch_url],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not resolve live stream {watch_url!r}: {result.stderr.strip()[-500:]}")
    direct_url = result.stdout.strip().splitlines()[0]
    if not direct_url:
        raise RuntimeError(f"yt-dlp returned no stream URL for {watch_url!r}")
    return direct_url


def _correlate(signal: np.ndarray, response: np.ndarray) -> float:
    if signal.std() < 1e-9 or response.std() < 1e-9:
        return 0.0
    corr = float(np.corrcoef(signal, response)[0, 1])
    return corr if math.isfinite(corr) else 0.0


def _list_clips(source: str) -> list[str]:
    if source == "live":
        # Watch-page URLs, not yet resolved -- run() resolves whichever
        # one is up next to its real direct stream URL right before
        # loading frames (see _resolve_live_url), never here.
        return [url for _, url in LIVE_SOURCES]
    path = Path(source)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        clips = sorted(
            str(p) for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not clips:
            raise RuntimeError(f"No video clips found under {source!r} (run tools/fetch_curriculum_videos.py first?)")
        return clips
    # Not a real local path -- a live device (e.g. /dev/video0).
    return [source]


def evaluate_genome(g: G.Genome, frames: list[np.ndarray], habituation_discount: float) -> tuple[float, dict, np.ndarray, np.ndarray, dict]:
    """
    Returns (fitness, breakdown, conspec_signal, loom_signal,
    live_info) -- live_info is the fovea's own final box position/size
    and final response value, for sandbox.save_live_status() (see
    run()); everything else the caller needs again after an accept, to
    actually advance habituation state over the trajectory the
    organism just really took (see run()'s own comment on why that has
    to happen AFTER the accept/reject decision, not before it).
    habituation_discount is read-only here (this generation's current
    discount, applied to the reward), never mutated by this function.
    """
    state = fovea.FoveaState()
    retina_vectors = []
    responses = []
    positions = []

    for frame in frames:
        v = fovea.extract(frame, state)
        retina_vectors.append(v)
        positions.append((state.cx, state.cy))
        vb = v[None, :]
        response = float(g.evaluate("response", vb)[0])
        pan = float(g.evaluate("pan", vb)[0])
        tilt = float(g.evaluate("tilt", vb)[0])
        responses.append(response)
        state = fovea.step(state, pan, tilt)

    live_info = {
        "fovea_cx": state.cx, "fovea_cy": state.cy,
        "fovea_fraction": fovea.FOVEA_FRACTION,
        "last_response": responses[-1] if responses else 0.0,
        # The actual 144-value grid the organism just processed, for
        # the viewer -- this is genuinely "what it's seeing", already
        # reduced far past anything reconstructable into real footage
        # (a blocky 12x12 luminance grid, not an image), so including
        # it here doesn't touch the no-raw-frames rule at all.
        "grid": retina_vectors[-1].tolist() if retina_vectors else [],
        "grid_shape": list(GRID),
    }

    vectors = np.array(retina_vectors)
    responses = np.array(responses)

    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(responses)):
        return float("-inf"), {}, np.zeros(len(frames)), np.zeros(len(frames)), live_info

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

    curiosity = _curiosity_score(positions)
    fitness += CURIOSITY_WEIGHT * curiosity
    breakdown["curiosity"] = curiosity

    dead_field = _dead_field_penalty(signals)
    fitness -= DEAD_FIELD_PENALTY_WEIGHT * dead_field
    breakdown["dead_field_penalty"] = dead_field

    return fitness, breakdown, cs, signals["loom"], live_info


def run(source: str, limits: sandbox.Limits, n_vars: int = N_CELLS) -> None:
    clips = _list_clips(source)

    checkpoint = sandbox.load_checkpoint()
    rng = random.Random()

    clip_index = 0
    if checkpoint is not None:
        clip_index = int(checkpoint.get("clip_index", 0)) % len(clips)
    clip_index = clip_index % len(clips)
    clip_path = clips[clip_index]

    # A "live" source list holds watch-page URLs, not playable ones --
    # resolve to the real, currently-live direct stream right before
    # loading, fresh every run (see _resolve_live_url's own docstring
    # on why this can't be done once and cached).
    load_source = clip_path
    if source == "live":
        print(f"Resolving live stream: {clip_path}")
        load_source = _resolve_live_url(clip_path)

    clip_name = _clip_display_name(source, clip_path)

    print(f"Loading real frames from {clip_path!r} into memory (never written to disk)...")
    frames = list(video_source.read_frames(load_source, stride=2, max_frames=600))
    print(f"  {len(frames)} frames loaded (clip {clip_index + 1}/{len(clips)}).")
    if len(frames) < 10:
        print("Not enough real frames to evolve against -- aborting.")
        return

    box = sandbox.Sandbox(limits)
    habituation = conspec.Habituation()
    margin = 0.05

    if checkpoint is not None and checkpoint.get("n_vars") == n_vars:
        print(f"Resuming from checkpoint (previous best_fitness={checkpoint['best_fitness']:.4f}).")
        genome = G.Genome.from_dict(checkpoint["genome"])
        best_fitness = float(checkpoint["best_fitness"])
        margin = float(checkpoint.get("margin", margin))
        habituation.exposure = float(checkpoint.get("habituation_exposure", 0.0))
    else:
        if checkpoint is not None:
            print("Checkpoint found but n_vars mismatch (retina/fovea shape changed) -- starting fresh.")
        genome = G.random_genome(rng, n_vars=n_vars)
        best_fitness, _, best_conspec, best_loom, _ = evaluate_genome(genome, frames, habituation.discount)
        print(f"Initial fitness: {best_fitness:.4f}")
        for c, l in zip(best_conspec, best_loom):
            habituation.observe(conspec_present=c > 0.05, loom_value=l)

    def _save():
        sandbox.save_checkpoint({
            "genome": genome.to_dict(),
            "best_fitness": best_fitness,
            "margin": margin,
            "habituation_exposure": habituation.exposure,
            "n_vars": n_vars,
            "clip_index": (clip_index + 1) % len(clips),
            "total_generation": checkpoint.get("total_generation", 0) if checkpoint else 0,
        })

    while box.should_continue():
        box.generation += 1
        candidate = genome.clone()

        # Every generation now really tries a tree mutation -- the
        # previous 15%-of-generations branch that mutated weights
        # INSTEAD of a tree was a wasted generation twice over (see
        # Genome.update_mutation_weights's own docstring): weight-only
        # candidates can never pass the tree accept/reject gate, so
        # that branch's change was always discarded, and no tree
        # mutation was even attempted on that generation either.
        channel, applied = candidate.mutate_task(
            rng, max_nodes=limits.max_tree_nodes, max_depth=limits.max_tree_depth,
        )
        # Only a REAL ceiling hit is worth surfacing as a request --
        # "noop_inapplicable" (e.g. mutate_const picked on a tree with
        # no consts yet) is normal and expected on a small tree, not
        # something a bigger ceiling would fix at all (see genome.py's
        # mutate_task docstring for the real bug this used to be:
        # every noop got blamed on the size ceiling, which made a
        # healthy small tree look artificially stuck).
        ceiling_reason = "tree_size_or_depth" if applied == "noop_ceiling" else None
        box.note_ceiling(ceiling_reason)

        candidate_fitness, breakdown, candidate_conspec, candidate_loom, live_info = evaluate_genome(
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

        # The real, continuous meta-mutation step (see
        # Genome.update_mutation_weights) -- applied to whichever
        # genome persists (the just-accepted candidate, or the
        # unchanged parent on a reject), using the real evidence from
        # THIS generation's real attempt. Never gated on fitness --
        # there's no fitness for a weights-only change to be gated on.
        genome.update_mutation_weights(applied, accepted)

        # Margin eases a little every generation, not only on accept,
        # so a long dry spell can't permanently freeze the acceptance
        # threshold above what real single-mutation deltas can clear
        # once the genome's converged past its early easy wins (the
        # deadlock the plateau was actually stuck in: margin can only
        # shrink on an accept, but accepts stopped clearing it well
        # before margin had shrunk much past its 0.05 starting point).
        margin = max(0.005, margin * 0.9995)

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
            "clip": clip_path,
        })

        # Real-time-ish snapshot for anything polling from outside
        # (see HLSLS's own broadcast-api /api/cv-state precedent) --
        # every generation, not just every 100th checkpoint save; this
        # is small and cheap, unlike the full checkpoint.
        sandbox.save_live_status({
            "generation": box.generation,
            "best_fitness": round(best_fitness, 4),
            "fovea_cx": round(live_info["fovea_cx"], 4),
            "fovea_cy": round(live_info["fovea_cy"], 4),
            "fovea_fraction": live_info["fovea_fraction"],
            "response": round(live_info["last_response"], 4),
            "habituation_exposure": round(habituation.exposure, 4),
            "clip": clip_path,
            # Human-readable label + whether it's a real live stream
            # right now, for the viewer -- User: "make sure display
            # page shows what it's watching."
            "clip_name": clip_name,
            "is_live": source == "live",
            "grid": [round(x, 4) for x in live_info["grid"]],
            "grid_shape": live_info["grid_shape"],
            # The CURRENT ACCEPTED genome's own tree structure (not
            # the just-tried candidate's, even on a rejected
            # generation) -- `genome` only ever changes on an accept,
            # so this is always "its real brain right now." Each
            # channel is capped at max_tree_nodes (limits.max_tree_nodes,
            # currently 1000), cheap enough to include every generation
            # rather than gating it.
            "trees": genome.to_dict()["trees"],
            # Real node_count()/depth() per channel plus the real
            # ceiling, alongside the tree itself -- User: "make display
            # an accurate representation of growth." A small tree
            # drawn next to its real budget (e.g. "12 / 1000 nodes")
            # is honest about how much headroom is actually left,
            # instead of just looking small with no context for why.
            "tree_stats": {
                name: {"nodes": tree.node_count(), "depth": tree.depth()}
                for name, tree in genome.trees.items()
            },
            "tree_limits": {"max_nodes": limits.max_tree_nodes, "max_depth": limits.max_tree_depth},
        })

        if box.generation % 25 == 0:
            print(
                f"  gen {box.generation:>5} best_fitness={best_fitness:.4f} "
                f"margin={margin:.4f} habituation={habituation.exposure:.3f}"
            )
        if box.generation % 100 == 0:
            _save()

    _save()
    print(f"Stopped after {box.generation} generations, {round(__import__('time').perf_counter() - box.start_time, 1)}s.")
    print(f"Final best_fitness: {best_fitness:.4f} -- checkpoint saved, next restart resumes from here.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="'live' for real live streams (see LIVE_SOURCES), a media/ directory of clips, a single file, or a live device (e.g. /dev/video0)")
    parser.add_argument("--generations", type=int, default=200000)
    parser.add_argument("--seconds", type=float, default=3600.0)
    args = parser.parse_args()

    limits = sandbox.Limits(max_generations=args.generations, max_wallclock_seconds=args.seconds)
    run(args.source, limits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
