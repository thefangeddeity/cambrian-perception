#!/usr/bin/env python3
from __future__ import annotations

"""
Entry point: evolves a genome against real curated clips, using the
innate reflex/conspec signals purely to grade fitness -- never as
genome input (see README's fishbowl boundary). Frames are loaded into
memory once per run (never written to disk -- see video_source.py),
and every generation replays the SAME in-memory clip from the SAME
starting fovea position, so accept/reject comparisons are fair.

Rewritten 2026-09-24 after an independent external audit found the
~18k-generation plateau wasn't a mutation-strategy problem at all --
three structural bugs made real progress nearly impossible regardless
of how mutation was tuned:
  - `best_fitness` used to be carried across restarts (different
    clips) and across habituation-discount drift within a run, so
    candidates were compared against a STALE, incomparable number.
    Fixed: the parent is re-evaluated fresh, on the SAME frames and
    SAME habituation_discount as the candidate, every single
    generation -- see the main loop below.
  - The response tree only ever saw the CURRENT frame, but is graded
    against frame-DIFFERENCE reflex signals -- structurally incapable
    of representing what it's scored on. Fixed: it now also gets the
    PREVIOUS frame's fovea vector as input (n_vars doubled), giving it
    the raw material to compute a difference itself if that helps.
  - Reflex/conspec signals used to be computed on the fovea's OWN
    cropped view, which means simply PANNING the fovea across a static
    scene manufactured apparent luminance-change/motion/loom (the same
    way a real eye's saccades look like the world moved without
    corollary-discharge correction) -- a self-stimulation loophole,
    confirmed empirically (a static scene with the fovea held still
    scored fitness -0.91). Fixed: all grading signals are now computed
    ONCE per run on the FULL, un-foveated frame -- independent of
    whatever the organism's own pan/tilt choices were.

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
import time
import sys
from pathlib import Path

import numpy as np

from fishbowl import conspec, fovea, genome as G, reflexes, sandbox, video_source
from fishbowl.state import MosquitoState
from fishbowl.controller import HIDDEN as BRAIN_HIDDEN
from fishbowl.retina import GRID, N_CELLS, frame_to_vector

# How much each reflex/drive contributes to total fitness -- loom
# weighted heaviest, matching the real threat/food asymmetry discussed
# while designing this (missing a threat costs more than missing an
# opportunity). Not tuned against real data yet; a real, named,
# starting guess, not a claim of correctness. "optomotor" renamed to
# "motion_energy" (honest -- it's direction-blind, see reflexes.py's
# module docstring); "directional_motion" is the real thing, added the
# same day, correlated against response the same way as the others.
SIGNAL_WEIGHTS = {"luminance_change": 0.5, "motion_energy": 0.75, "directional_motion": 0.75, "loom": 2.0}
CONSPEC_WEIGHT = 1.5

# Real orienting pressure, added 2026-09-24 -- the user, watching a real
# deployed kitten cam: "this cat's been there the whole time, but the
# fovea's too primitive to evolve to lock on it." Confirmed by design
# review: nothing previously rewarded pan/tilt for actually moving
# TOWARD anything -- curiosity rewards coverage, dead_field penalizes
# staying put, but neither one cares whether the fovea is near a real
# subject. Two real, named reflexes close this gap, both still purely
# REWARDS (never force a specific movement, same discipline as every
# other drive here):
#   optokinetic pursuit -- does pan/tilt's own movement direction
#   correlate with real detected motion direction (reflexes.py's
#   directional_motion)? This is what a real optokinetic/smooth-
#   pursuit reflex does: track whole-field or object motion to
#   stabilize gaze, found across nearly all motile visual animals.
#   seek -- when CONSPEC detects a being-like pattern strongly enough,
#   is the fovea actually near where it is? Makes conspec.py's own
#   "seek" drive literal instead of a documented "honest limitation"
#   (there IS a real pan/tilt actuator now) -- and matches the real
#   CONSPEC/CONLERN literature more closely too: real newborns
#   orient head/eyes toward face-like stimuli, not just look longer.
PURSUIT_WEIGHT = 1.0
SEEK_WEIGHT = 1.0
SEEK_STRENGTH_THRESHOLD = 0.05  # only scored when something's actually there -- never penalizes absence

# Real movement-cost PENALTY, added 2026-09-24 after the user watched a
# real deployed genome corner-pin the fovea (land on one saturated
# jump, then stay there for the rest of the run) instead of gliding.
# Real animals pay genuine metabolic cost for large/fast eye or head
# movements -- nothing here cost anything before, so a big, mostly-
# unjustified jump was exactly as "free" as a small graded one, no
# counterweight to the tanh/MAX_STEP saturation bias measured
# empirically in fovea.py's own docstring (39% of random genomes
# produce a near-maximal step vs. 7% a small one). This does NOT cap
# or forbid large jumps -- PURSUIT_WEIGHT/SEEK_WEIGHT above can still
# justify a big move when it's genuinely worth it; it just means an
# UNJUSTIFIED one no longer costs nothing. Costed on INTENT (the
# pre-clamp tanh output, see fovea.step's own docstring), not net
# realized displacement -- a tree that keeps outputting a maximal
# pan/tilt while already pinned against a wall is still "trying" every
# frame, real motor effort even though the wall zeroes out its net
# movement.
MOVEMENT_COST_WEIGHT = 0.5

# Corner penalty, User: "It's obsessed with corners... A corner
# penalty, of sorts." Soft, not a hard constraint -- real reward can
# still outweigh it if a corner is ever genuinely the right place to
# be. Fixed constant, outside the genome's reach (same reasoning as
# _HEAD_SIZE: this grades behavior, it isn't a perception trait, so
# it doesn't evolve).
CORNER_PENALTY_WEIGHT = 1.0  # doubled 2026-09-24, User: "Double the edge and corner penalties please"


def _corner_penalty(positions: list[tuple[float, float]], fracs: list[float]) -> float:
    """
    "Cornerness" = product of how off-center each axis is (normalized
    to [-1, 1] over the reachable range) -- zero along either center
    line, maximal only where BOTH axes are extreme at once (a real
    corner, not just one edge). Averaged over the run.
    """
    if not positions:
        return 0.0
    scores = []
    for (cx, cy), frac in zip(positions, fracs):
        half_range = 0.5 - frac / 2.0
        scores.append(abs((cx - 0.5) / half_range * (cy - 0.5) / half_range) if half_range > 1e-9 else 0.0)
    return float(np.mean(scores))


# User: "I want it to shun edges unless they're worth it." Real,
# separate, LESSER cost than corner_penalty -- a single edge (one axis
# extreme, the other centered) is genuinely less wasteful than a true
# corner, so it costs less, not nothing. Still soft: seek/pursuit can
# outweigh it when an edge really is where the subject is.
EDGE_PENALTY_WEIGHT = 0.5  # doubled 2026-09-24, User: "Double the edge and corner penalties please"


def _edge_penalty(positions: list[tuple[float, float]], fracs: list[float]) -> float:
    """Max (not product) of how off-center each axis is -- unlike cornerness, this alone is already high for EITHER a corner or a single edge; corner_penalty's own weight is what makes a true corner cost more overall."""
    if not positions:
        return 0.0
    scores = []
    for (cx, cy), frac in zip(positions, fracs):
        half_range = 0.5 - frac / 2.0
        scores.append(max(abs((cx - 0.5) / half_range), abs((cy - 0.5) / half_range)) if half_range > 1e-9 else 0.0)
    return float(np.mean(scores))


# --- Homeostasis (Gemini's plan: fishbowl/state.py + controller.py) ---
#
# The body pays per frame: basal metabolism, motor effort, and the
# look's aperture. Aperture cost = area x scarcity (REFERENCE_QUOTA_PCT
# / the real CPU quota resource_handler.py has granted) -- User: "grow
# its visual field as curiosity wants and resources allow, but shrink
# as resource hunger limits it."
REFERENCE_QUOTA_PCT = 150.0
APERTURE_COST = 0.01


def _aperture_cost(frac: float, quota_pct: float) -> float:
    return APERTURE_COST * (frac * frac) * (REFERENCE_QUOTA_PCT / max(1.0, quota_pct))


# Food = genuinely new visual structure (Gemini: "appetite for novel
# visual structure"), measured against a spatial memory of what the
# look has already seen at each WORLD location. Panning across a static
# scene is only news the first time; after that, only real change in
# the world feeds it -- so moving the eye can't manufacture food (the
# self-stimulation loophole the audit found, kept closed). A bigger
# look takes in more at once, which is what pays for its aperture cost.
MEM_H, MEM_W = 24, 32
UNSEEN_NOVELTY = 0.25
FOOD_GAIN = 400.0


def _feed_on_novelty(memory: np.ndarray, look: np.ndarray, st: "fovea.FoveaState") -> float:
    x0 = int(round((st.cx - st.fraction / 2) * MEM_W)); x1 = max(x0 + 1, int(round((st.cx + st.fraction / 2) * MEM_W)))
    y0 = int(round((st.cy - st.fraction / 2) * MEM_H)); y1 = max(y0 + 1, int(round((st.cy + st.fraction / 2) * MEM_H)))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(MEM_W, x1), min(MEM_H, y1)
    grid = look.reshape(GRID)
    patch = grid[np.ix_(np.arange(y1 - y0) * GRID[0] // (y1 - y0), np.arange(x1 - x0) * GRID[1] // (x1 - x0))]
    region = memory[y0:y1, x0:x1]
    seen = ~np.isnan(region)
    novelty = np.where(seen, np.abs(patch - np.nan_to_num(region)), UNSEEN_NOVELTY)
    memory[y0:y1, x0:x1] = np.where(seen, 0.7 * np.nan_to_num(region) + 0.3 * patch, patch)
    return float(min(1.0, novelty.sum() / (MEM_H * MEM_W) * FOOD_GAIN))


# Scale of the look's own motion/flow readings into the [0, 1]
# range Gemini's state and reflex thresholds assume (calibrated on real
# camera + synthetic looming frames -- see the commit message).
MOTION_GAIN = 10.0
FLOW_GAIN = 20.0

# Fitness from staying viable: mean homeostatic drive over the run
# (energy deficit, threat, fatigue -- MosquitoState.drive()). Mean, not
# summed drive_reduction(): a sum of per-frame reductions telescopes to
# D(start) - D(end) and ignores everything in between.
# The WHOLE visual field (the fixed camera's coarse 12x12 view -- a
# jumping spider's wide-field secondary eyes, or a locust's LGMD/DCMD
# looming neurons) is what detects threat and where something moved;
# the look (the spider's movable principal retinae) is for detail and
# food. Whole-field gains calibrated like the look's.
PERIPH_MOTION_GAIN = 300.0   # whole-field motion_energy: still room ~0.001 -> ~0.3, real movement saturates
EXPANSION_GAIN = 10.0        # reflexes.expansion_score: approaching disc 0.083 -> 0.83, crossing blob 0.014 -> 0.14


def _peripheral_motion_centroid(world_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where in the whole field things changed, per frame (0.5, 0.5 when nothing did)."""
    n = len(world_vectors)
    cx, cy = np.full(n, 0.5), np.full(n, 0.5)
    rows, cols = GRID
    yy, xx = np.mgrid[0:rows, 0:cols]
    for t in range(1, n):
        d = np.abs(world_vectors[t] - world_vectors[t - 1]).reshape(GRID)
        tot = d.sum()
        if tot > 1e-9:
            cx[t] = ((xx + 0.5) * d).sum() / tot / cols
            cy[t] = ((yy + 0.5) * d).sum() / tot / rows
    return cx, cy


HOMEOSTASIS_WEIGHT = 3.0
# Gemini's brain has an "alarm" output; it earns fitness by tracking
# real world loom (never forced to).
ALARM_WEIGHT = 1.0

# Per-look compute cost (the brain and tree running once), priced like
# the aperture: x CPU scarcity. With basal rate scaling by pace (see
# MosquitoState.update), this is what makes a fast pace of life expensive.
THINK_COST = 0.001

# The flinch, evolved rather than wired: at each onset of a real
# approach (whole-field dark expansion crossing FLINCH_THRESHOLD), reward
# reacting within FLINCH_WINDOW frames (~200 ms at 15 frames/s) by
# widening the look or making a saccade -- more for a faster reaction,
# nothing for none. No approach in a run means no reward and no penalty.
FLINCH_WEIGHT = 1.0
FLINCH_THRESHOLD = 0.18
FLINCH_WINDOW = 3


def _flinch(looms: list[float], speeds: np.ndarray, fracs: list[float], pace: int = 1) -> dict:
    onsets = [t for t in range(1, len(looms)) if looms[t] > FLINCH_THRESHOLD >= looms[t - 1]]
    latencies = []
    for t in onsets:
        lat = None
        # Looks are `pace` base frames apart, and on average an approach
        # starts (pace-1)/2 frames before the look that notices it -- so
        # latency is counted in real frames, and a slow pace pays for it.
        k = 0
        while t + k < len(speeds):
            real = k * pace + (pace - 1) / 2.0
            if real > FLINCH_WINDOW:
                break
            i = t + k
            widened = i + 1 < len(fracs) and fracs[i + 1] - fracs[i] > 0.01
            if widened or speeds[i] >= 0.05:
                lat = real
                break
            k += 1
        latencies.append(lat)
    reacted = [l for l in latencies if l is not None]
    score = float(np.mean([1.0 - l / (FLINCH_WINDOW + 1) if l is not None else 0.0 for l in latencies])) if latencies else 0.0
    return {"events": len(onsets), "reacted": len(reacted),
            "mean_latency_frames": round(float(np.mean(reacted)), 2) if reacted else None, "score": round(score, 3)}

# The other boundary of the corridor -- the user's own framing: a deep-sea
# vent shrimp doesn't just flee scalding water, it also has to avoid
# drifting into the freezing water behind it. loom is the "scalding"
# side (big, sudden, real threat -- weighted heaviest above,
# deliberately never fully avoidable by just sitting still). This is
# the "freezing" side: a real penalty for a sustained stretch with
# nothing happening ("dead field... dead cold"). NOTE this now grades
# WORLD activity (see evaluate_genome's world_signals), not "is my own
# current view dead" -- the audit fix that closed the self-stimulation
# loophole (panning the fovea used to manufacture fake activity) also
# means this term can no longer be dodged by just moving the eye. It's
# a slightly blunter pressure now (organism can't fix a genuinely dead
# WORLD by looking elsewhere), but curiosity below still rewards real
# fovea coverage, so exploration pressure isn't lost, just no longer
# exploitable.
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
# Rescaled 2026-09-24 alongside the _curiosity_score fix -- the new
# per-frame novelty-rate formula's natural max is CELLS/N_FRAMES
# (~25/600 = 0.042 for a typical run), not 1.0 like the old fraction
# was. Rescaled so the max possible contribution to fitness stays
# roughly comparable to before (0.75 * 1.0 -> ~18 * 0.042 = ~0.75).
CURIOSITY_WEIGHT = 18.0


def _curiosity_score(positions: list[tuple[float, float]]) -> float:
    """
    Real bug fix, User: "I think definition of curiosity is making
    fovea hopscotch instead of saccading; check definition." Verified
    empirically before fixing: the OLD formula (final fraction of
    distinct cells ever visited) let a genome hop through all 25 cells
    in the first 25 of 600 frames, then coast idle for the remaining
    575, and still collect the FULL, PERMANENT reward -- a real timing
    mismatch against movement_cost, which is averaged over the whole
    run. A burst-then-coast strategy paid a heavily diluted average
    cost for a one-time, forever-kept reward.

    Fixed to a per-frame NOVELTY RATE instead -- credits a frame only
    if it discovered a cell not yet seen this run, averaged over ALL
    frames. Coasting after exploring now correctly contributes 0, not
    the max. This is also more faithful to the real count-based
    exploration bonuses in the RL literature this was already citing
    as its inspiration, which reward the MOMENT of discovery, not a
    final tally. Max possible score is now CELLS/N_FRAMES (~0.042 for
    25 cells over 600 frames), not 1.0 -- CURIOSITY_WEIGHT rescaled to
    match (see its own comment).
    """
    if not positions:
        return 0.0
    visited = set()
    novel_frames = 0
    for cx, cy in positions:
        cell = (min(CURIOSITY_GRID - 1, int(cx * CURIOSITY_GRID)),
                min(CURIOSITY_GRID - 1, int(cy * CURIOSITY_GRID)))
        if cell not in visited:
            visited.add(cell)
            novel_frames += 1
    return novel_frames / float(len(positions))


def _dead_field_penalty(signals: dict[str, np.ndarray]) -> float:
    activity = signals["motion_energy"] + signals["luminance_change"]
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


def _seek_reward(
    positions: list[tuple[float, float]],
    peak_cx: np.ndarray,
    peak_cy: np.ndarray,
    strength: np.ndarray,
) -> float:
    """
    Real orienting reward -- see PURSUIT_WEIGHT/SEEK_WEIGHT's own
    comment for why this exists. Only scored on frames where CONSPEC
    actually detected something (strength > threshold) -- this can
    only ever reward genuine proximity to a real detection, never
    penalize the fovea for where it is when nothing's there, same
    reward-not-force discipline as every other drive here.
    """
    scores = []
    for (cx, cy), pcx, pcy, s in zip(positions, peak_cx, peak_cy, strength):
        if s <= SEEK_STRENGTH_THRESHOLD:
            continue
        dist = math.hypot(cx - pcx, cy - pcy)
        # Normalized: 1.0 = fovea centered exactly on the detection,
        # 0.0 = as far as possible (opposite corners of the frame).
        scores.append(max(0.0, 1.0 - dist / math.sqrt(2.0)))
    return float(np.mean(scores)) if scores else 0.0

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")

# User: "Change training feed to this actual live feed with no people
# or picture-in-picture" (2026-09-24, replacing the earlier
# cat_livestream/pixcams_wildlife pair). Confirmed live via yt-dlp
# metadata before wiring in (title: "Kitten Rescue Cat Cam powered by
# EXPLORE.org", is_live=True) -- a real, established source, no
# on-screen people and no compositing overlay to confuse the world-
# level reflex signals with something that isn't real scene content.
# Watched LIVE (resolved fresh every run via _resolve_live_url below),
# never downloaded -- genuinely different real content every restart.
#
# User: "Switch to this training feed" -- kitten_rescue_baby_kittens_cam
# now the sole source (previously the backup entry, added earlier the
# same day; already confirmed live via yt-dlp metadata then).
LIVE_SOURCES = [
    ("kitten_rescue_baby_kittens_cam", "https://www.youtube.com/watch?v=gBdqOuhj2P4"),
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
    # Video-only, preferring H.264 (avc1) -- YouTube serves both AV1 (av01)
    # and H.264 in MP4 containers; without vcodec^=avc1, yt-dlp selects AV1,
    # which fails pixel-format decode in OpenCV/FFmpeg on Tanzania.
    result = subprocess.run(
        [str(yt_dlp), "-g", "-f", "bestvideo[height<=480][vcodec^=avc1]/bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/bestvideo", watch_url],
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


def _world_vectors(frames: list[np.ndarray]) -> np.ndarray:
    """
    The FULL, un-foveated frame at every timestep, reduced to the same
    12x12 grid shape retina.py already uses -- computed ONCE per run,
    the same for every genome/generation, since it depends only on the
    real clip, never on any genome's pan/tilt choices. This is what
    closes the self-stimulation loophole an external audit found: when
    grading used to run on the fovea's OWN cropped/panned view, moving
    the fovea across a perfectly static scene manufactured apparent
    luminance-change/motion/loom out of nothing (confirmed empirically:
    fovea held still on a static scene scored fitness -0.91, almost
    all of it curiosity/dead-field pressure to just move the eye).
    Grading against the world's own real signal means the only way to
    earn reward is to produce a response that actually tracks what's
    really happening, not to manufacture apparent motion by panning.
    """
    return np.array([frame_to_vector(f) for f in frames])


def evaluate_genome(
    g: G.Genome,
    frames: list[np.ndarray],
    world_signals: dict[str, np.ndarray],
    world_conspec: tuple[np.ndarray, np.ndarray, np.ndarray],
    habituation_discount: float,
    quota_pct: float = REFERENCE_QUOTA_PCT,
) -> tuple[float, dict, dict]:
    """
    Returns (fitness, breakdown, live_info). world_signals/world_conspec
    are precomputed ONCE per run by the caller (see run()) -- see
    _world_vectors's own docstring for why grading is independent of
    this genome's own fovea path. world_conspec is (strength, peak_cx,
    peak_cy) -- see conspec.conspec_signal. habituation_discount is
    read-only here, never mutated by this function (habituation itself
    is now observed once per run, from the real world signal, not from
    any genome's path -- see run()).
    """
    state = fovea.FoveaState(fraction=float(np.clip(g.fovea_fraction, fovea.MIN_FRACTION, fovea.MAX_FRACTION)))
    body = MosquitoState()
    # Pace of life: the caller hands over every pace-th frame (and the
    # world signals at that rate); each look spans `pace` base frames of
    # real time (1/15 s each).
    pace = max(1, int(getattr(g, "pace", 1)))
    brain = g.brain
    brain.reset_hidden()
    memory = np.full((MEM_H, MEM_W), np.nan)  # what the look has seen, per world location
    scarcity_cost = lambda frac: _aperture_cost(frac, quota_pct)
    scarcity = REFERENCE_QUOTA_PCT / max(1.0, quota_pct)

    responses, alarms = [], []
    positions, fracs = [], []
    dxs, dys = [], []  # real (post-clamp) look movement per frame
    movement_costs = []  # pre-clamp motor intent per frame
    energies, drives, foods = [], [], []
    periph_active = []
    field_events = []  # per frame: where the whole field saw motion, how much, loom, reflex
    reflex_frames = 0
    last_grid = None
    prev_v = np.zeros(N_CELLS)
    # Response tree's motor-efference input (its old pan/tilt feedback):
    # now the brain's real applied movement.
    prev_dx, prev_dy = 0.0, 0.0
    prev_frame = None

    for frame in frames:
        v = fovea.extract(frame, state)
        positions.append((state.cx, state.cy))
        fracs.append(state.fraction)

        # What the look sees change, with its own eye movement cancelled
        # out (corollary discharge): the previous frame sampled at
        # the look's CURRENT position, so a saccade across a still scene
        # doesn't register as motion, threat, or loom.
        h1 = fovea.extract(prev_frame, state) if prev_frame is not None else v
        hist = np.array([h1, v])
        lum = float(v.mean())
        # Only real history counts: on the first frame there's no older
        # sample to compare against.
        motion = flow_x = flow_y = 0.0
        if prev_frame is not None:
            motion = min(1.0, float(np.abs(v - h1).mean()) * MOTION_GAIN)
            mx, my = reflexes.directional_motion(hist)
            flow_x = float(np.clip(mx[-1] * FLOW_GAIN, -1.0, 1.0))
            flow_y = float(np.clip(my[-1] * FLOW_GAIN, -1.0, 1.0))
        # Threat and arousal come from the WHOLE visual field (spider's
        # secondary eyes / locust LGMD), plus where in it something moved,
        # relative to the look -- so the brain can learn to swing its look
        # toward movement. The look's own readings above are for detail.
        t_idx = len(positions) - 1
        loom = min(1.0, float(world_signals["expansion"][t_idx]) * EXPANSION_GAIN)
        periph_motion = min(1.0, float(world_signals["motion_energy"][t_idx]) * PERIPH_MOTION_GAIN)
        periph_dx = float(world_signals["motion_cx"][t_idx]) - state.cx
        periph_dy = float(world_signals["motion_cy"][t_idx]) - state.cy

        pan, tilt, zoom, alarm, is_reflex = brain.step(
            lum, motion, flow_x, flow_y, loom, state.cx, state.cy, state.fraction, body,
            periph_dx, periph_dy, state.vx, state.vy,
        )
        # The brain's recurrent memory (its 16 hidden units, updated just
        # above) feeds the perception tree as extra inputs x290-x305, after
        # the look (x0-143), previous look (x144-287) and own movement.
        vb = np.concatenate([v, prev_v, [prev_dx, prev_dy], brain.hidden])[None, :]
        response = float(g.evaluate("response", vb)[0])
        reflex_frames += int(is_reflex)
        field_events.append([
            round(float(world_signals["motion_cx"][t_idx]), 3), round(float(world_signals["motion_cy"][t_idx]), 3),
            round(periph_motion, 3), round(loom, 3), int(is_reflex),
        ])
        responses.append(response)
        alarms.append(alarm)
        last_grid = v
        prev_v = v

        prev_cx, prev_cy = state.cx, state.cy
        state, force_x, force_y, intended_dz = fovea.step(state, pan, tilt, zoom)
        prev_dx, prev_dy = state.cx - prev_cx, state.cy - prev_cy
        dxs.append(prev_dx)
        dys.append(prev_dy)
        # Muscle energy grows with force squared: many gentle pushes are
        # cheaper than one violent one covering the same ground.
        effort = force_x * force_x + force_y * force_y + abs(intended_dz) / fovea.ZOOM_STEP * 0.1
        movement_costs.append(math.hypot(force_x, force_y))
        periph_active.append(periph_motion)

        body.update(periph_motion, loom, effort, scarcity_cost(state.fraction) + THINK_COST * scarcity, dt=pace, pace=pace)
        food = _feed_on_novelty(memory, fovea.extract(frame, state), state)
        body.feed_visual_sustenance(food)
        energies.append(body.energy)
        drives.append(body.drive())
        foods.append(food)

        prev_frame = frame

    step_n = max(1, len(energies) // 60)

    # Movement style, MEASURED not rewarded -- the yardstick from Land
    # (1969) on jumping-spider retinae: fixating (still), gliding (slow,
    # smooth), saccades (fast jumps), and tracking (following where the
    # whole field says something is moving). "Lifelike" as numbers
    # comparable to a real animal, not an impression.
    # Per base frame (1/15 s), so a slow pace can't look "smooth" just by
    # moving the same distance in fewer, bigger steps.
    speeds = (np.hypot(np.array(dxs), np.array(dys)) / pace) if dxs else np.zeros(1)
    active = np.array(periph_active) > 0.2 if periph_active else np.zeros(1, bool)
    tracking = None
    if active.sum() >= 10 and len(dxs) > 2:
        mcx = np.diff(np.asarray(world_signals["motion_cx"][:len(dxs)]), prepend=0.5)
        mcy = np.diff(np.asarray(world_signals["motion_cy"][:len(dys)]), prepend=0.5)
        tracking = round((_correlate(mcx[active], np.array(dxs)[active]) + _correlate(mcy[active], np.array(dys)[active])) / 2.0, 3)
    movement = {
        "fixate": round(float(np.mean(speeds < 0.005)), 3),
        "glide": round(float(np.mean((speeds >= 0.005) & (speeds < 0.05))), 3),
        "saccade": round(float(np.mean(speeds >= 0.05)), 3),
        "scan_while_still": round(float(np.mean(((speeds >= 0.005) & (speeds < 0.05))[~active])) if (~active).any() else 0.0, 3),
        "tracking": tracking,
        "world_active": round(float(np.mean(active)), 3),
    }
    live_info = {
        "fovea_cx": state.cx, "fovea_cy": state.cy,
        "fovea_fraction": state.fraction,
        "last_response": responses[-1] if responses else 0.0,
        # The actual 144-value grid the organism just processed -- a
        # blocky 12x12 luminance grid, not an image, so it doesn't touch
        # the no-raw-frames rule.
        "grid": last_grid.tolist() if last_grid is not None else [],
        "grid_shape": list(GRID),
        "body": body.to_dict(),
        "energy_series": [round(e, 4) for e in energies[::step_n]],
        # What it's eating: genuinely new visual structure per frame
        # (see _feed_on_novelty), sampled like energy.
        "food_series": [round(float(np.mean(foods[k:k + step_n])), 4) for k in range(0, len(foods), step_n)],
        "mean_food": round(float(np.mean(foods)), 4) if foods else 0.0,
        "movement": movement,
        "field_events": field_events,
        "flinch": _flinch([e[3] for e in field_events], speeds, fracs, pace),
        "pace": pace,
        "brain_hidden": [round(h, 3) for h in brain.hidden],
        "reflex_frames": reflex_frames,
        "trajectory": [[round(x, 4), round(y, 4), round(f, 4)] for (x, y), f in zip(positions, fracs)],
    }

    responses = np.array(responses)

    if not np.all(np.isfinite(responses)) or not np.all(np.isfinite(alarms)):
        return float("-inf"), {}, live_info

    signals = world_signals
    cs, conspec_peak_cx, conspec_peak_cy = world_conspec

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

    curiosity = _curiosity_score(positions) / pace  # per base frame of real time
    fitness += CURIOSITY_WEIGHT * curiosity
    breakdown["curiosity"] = curiosity

    dead_field = _dead_field_penalty(signals)
    fitness -= DEAD_FIELD_PENALTY_WEIGHT * dead_field
    breakdown["dead_field_penalty"] = dead_field

    # Real orienting pressure (see PURSUIT_WEIGHT/SEEK_WEIGHT's own
    # comment) -- does pan/tilt's real movement track real motion
    # direction, and does the fovea end up near a real detection?
    pursuit = (_correlate(signals["motion_x"], np.array(dxs)) + _correlate(signals["motion_y"], np.array(dys))) / 2.0
    fitness += PURSUIT_WEIGHT * pursuit
    breakdown["optokinetic_pursuit"] = pursuit

    seek = _seek_reward(positions, conspec_peak_cx, conspec_peak_cy, cs)
    fitness += SEEK_WEIGHT * seek
    breakdown["seek_reward"] = seek

    movement_cost = float(np.mean(movement_costs)) if movement_costs else 0.0
    fitness -= MOVEMENT_COST_WEIGHT * movement_cost
    breakdown["movement_cost"] = movement_cost

    corner = _corner_penalty(positions, fracs)
    fitness -= CORNER_PENALTY_WEIGHT * corner
    breakdown["corner_penalty"] = corner

    edge = _edge_penalty(positions, fracs)
    fitness -= EDGE_PENALTY_WEIGHT * edge
    breakdown["edge_penalty"] = edge

    mean_drive = float(np.mean(drives)) if drives else 0.0
    fitness -= HOMEOSTASIS_WEIGHT * mean_drive
    breakdown["mean_drive"] = mean_drive
    breakdown["mean_energy"] = float(np.mean(energies)) if energies else 0.0
    breakdown["final_energy"] = energies[-1] if energies else 0.0
    breakdown["mean_food"] = float(np.mean(foods)) if foods else 0.0
    breakdown["mean_aperture"] = float(np.mean(fracs)) if fracs else 0.0
    breakdown["reflex_frames"] = reflex_frames
    breakdown["movement"] = live_info["movement"]

    flinch = live_info["flinch"]
    fitness += FLINCH_WEIGHT * flinch["score"]
    breakdown["flinch"] = flinch["score"]
    breakdown["loom_events"] = flinch["events"]

    alarm_corr = _correlate(signals["expansion"], np.array(alarms))
    fitness += ALARM_WEIGHT * alarm_corr
    breakdown["alarm_loom"] = alarm_corr

    return fitness, breakdown, live_info


# Small, fixed tolerance/probability for accepting a roughly-tied
# candidate (never a worse one) -- real neutral drift, not forced
# exploration: the fitness landscape is what decides whether a neutral
# move is even available at a given moment, this only decides whether
# one gets taken when it IS available. External audit's recommended
# fix for a pure greedy (1+1) hill-climb never being able to cross a
# flat plateau (equal-fitness moves were always rejected before).
NEUTRAL_EPSILON = 0.001
NEUTRAL_ACCEPT_PROB = 0.1


WORLD_REFRESH_GENERATIONS = 5


class World:
    """
    One snapshot of the world (frames + their retina vectors), with the
    world signals every genome is graded on computed on demand at each
    pace of life actually in use, then cached. Parent and candidate are
    always scored on the same World within a generation.
    """

    def __init__(self, frames: list[np.ndarray], vectors: np.ndarray):
        self.frames, self.vectors = frames, vectors
        self._cache: dict[int, tuple] = {}

    def at_pace(self, pace: int):
        pace = max(1, int(pace))
        if pace not in self._cache:
            wv = self.vectors[::pace]
            ws = reflexes.all_signals(wv)
            ws["expansion"] = reflexes.expansion_score(wv)
            ws["motion_cx"], ws["motion_cy"] = _peripheral_motion_centroid(wv)
            self._cache[pace] = (self.frames[::pace], ws, conspec.conspec_signal(wv))
        return self._cache[pace]


def _is_device(source: str) -> bool:
    return source.startswith("/dev/video") or source.isdigit()


def _dessert() -> dict | None:
    """
    User: "submit a live YT video it can watch overnight when everything
    is sleeping... like feeding it dessert." A human-chosen live stream
    with a deadline ("until", unix seconds), written by the viewer.
    Active while the deadline is in the future; None otherwise.
    """
    sel = sandbox.load_selected_source()
    if not sel or not sel.get("url"):
        return None
    until = sel.get("until")
    if until is not None and time.time() >= float(until):
        return None
    return sel


def run(source: str, limits: sandbox.Limits, n_vars: int = N_CELLS * 2 + 2 + BRAIN_HIDDEN) -> None:
    # Dessert overrides the camera until its deadline; then this run
    # exits and systemd restarts it back on the camera.
    home_source = source
    dessert = _dessert() if _is_device(source) else None
    if dessert is not None:
        print(f"Dessert: watching {dessert['url']} until {time.ctime(float(dessert['until'])) if dessert.get('until') else 'cleared'}")
        source = "live"
    clips = _list_clips(source)

    checkpoint = sandbox.load_checkpoint()
    rng = random.Random()

    clip_index = 0
    if checkpoint is not None:
        clip_index = int(checkpoint.get("clip_index", 0)) % len(clips)
    clip_index = clip_index % len(clips)
    clip_path = clips[clip_index]

    # Human-only override, User: "put a list of training videos I can
    # pick from the viewer" / "add a field I can input the video to be
    # watched." Written by tools/viewer.py, never by the organism --
    # only takes effect for live-mode sources. A "name" must match
    # LIVE_SOURCES; a free-text "url" is only ever accepted by the
    # viewer after it's already confirmed live via a real yt-dlp
    # metadata check (see viewer.py's _check_live_url) -- this side
    # just trusts that already happened, same as it already trusts
    # LIVE_SOURCES itself was vetted before being written into the
    # code. Sticky: stays selected across restarts until cleared back
    # to "auto" in the viewer, which resumes normal round-robin.
    if source == "live":
        selection = sandbox.load_selected_source()
        selected_name = selection.get("name") if selection else None
        selected_url = selection.get("url") if selection else None
        if selected_name:
            for i, (name, url) in enumerate(LIVE_SOURCES):
                if name == selected_name:
                    clip_index, clip_path = i, url
                    break
        elif selected_url:
            clip_index, clip_path = 0, selected_url

    # A "live" source list holds watch-page URLs, not playable ones --
    # resolve to the real, currently-live direct stream right before
    # loading, fresh every run (see _resolve_live_url's own docstring
    # on why this can't be done once and cached).
    load_source = clip_path
    if source == "live":
        print(f"Resolving live stream: {clip_path}")
        load_source = _resolve_live_url(clip_path)

    clip_name = _clip_display_name(source, clip_path)

    # A live camera is read continuously (LiveFeed) and every generation
    # is scored on the newest ~40 s -- the world it is living in now.
    # Files and YouTube streams keep one fixed clip per run.
    feed = None
    if _is_device(source):
        print(f"Opening live feed {source!r} (frames kept in memory only, never written to disk)...")
        feed = video_source.LiveFeed(int(source) if source.isdigit() else source)
        if not feed.wait_for(600):
            print("Live feed never filled its window -- aborting.")
            feed.close()
            return
        frames, vectors, seen_total = feed.snapshot()
    else:
        print(f"Loading real frames from {clip_path!r} into memory (never written to disk)...")
        frames = list(video_source.read_frames(load_source, stride=2, max_frames=600))
        print(f"  {len(frames)} frames loaded (clip {clip_index + 1}/{len(clips)}).")
        if len(frames) < 10:
            print("Not enough real frames to evolve against -- aborting.")
            return
        vectors = _world_vectors(frames)
        seen_total = len(frames)
    world = World(frames, vectors)

    # The real source frame's shape -- User: "make foveal rectangle honest."
    frame_h, frame_w = frames[0].shape[:2]

    box = sandbox.Sandbox(limits)
    if checkpoint is not None:
        box.generation = int(checkpoint.get("total_generation", 0))
    habituation = conspec.Habituation()
    margin = 0.05

    # Inputs are only ever APPENDED (e.g. the brain's hidden units), so a
    # checkpoint with fewer inputs is prefix-compatible: every existing
    # tree index keeps its meaning.
    if checkpoint is not None and 0 < int(checkpoint.get("n_vars", 0)) <= n_vars:
        print(f"Resuming from checkpoint (previous best_fitness={checkpoint['best_fitness']:.4f}).")
        genome = G.Genome.from_dict(checkpoint["genome"])
        genome.n_vars = n_vars
        margin = float(checkpoint.get("margin", margin))
        habituation.exposure = float(checkpoint.get("habituation_exposure", 0.0))
    else:
        if checkpoint is not None:
            print("Checkpoint found but n_vars mismatch (retina/fovea shape changed) -- starting fresh.")
        genome = G.random_genome(rng, n_vars=n_vars)

    # Habituation is now observed ONCE per run, straight from the
    # real world signal -- it no longer depends on any genome's path
    # (see module docstring). This also fixes a real related bug: it
    # used to only advance on an ACCEPTED generation's real trajectory,
    # so a long dry spell (exactly what B1 was causing) meant
    # habituation silently stopped updating for thousands of
    # generations even while real exposure was happening on screen.
    def _observe(new_count: int) -> None:
        _, ws1, wc1 = world.at_pace(1)
        n = min(new_count, len(ws1["loom"]))
        for c, l in zip(wc1[0][-n:], ws1["loom"][-n:]):
            habituation.observe(conspec_present=c > 0.05, loom_value=l)

    _observe(len(frames))

    # NEVER trust a best_fitness carried over from a different clip or
    # a different habituation state (external audit finding: this was
    # the actual root cause of the plateau, not mutation strategy) --
    # always re-derive it fresh, on THIS run's real frames, before
    # anything is compared against it. peak_fitness_seen is a separate,
    # purely informational running max (never used for accept/reject),
    # so the "how good has this lineage ever been" number isn't lost
    # now that best_fitness itself is an honest, re-scored-every-
    # generation live value rather than a one-way ratchet.
    # Real current CPU quota (resource_handler.py's own record), prices
    # the look's size -- refreshed periodically below, never inferred.
    quota_pct = sandbox.load_quota_pct(REFERENCE_QUOTA_PCT)
    best_fitness, _, _ = evaluate_genome(genome, *world.at_pace(genome.pace), habituation.discount, quota_pct)
    peak_fitness_seen = checkpoint.get("peak_fitness_seen", best_fitness) if checkpoint is not None else best_fitness
    peak_fitness_seen = max(peak_fitness_seen, best_fitness) if math.isfinite(best_fitness) else peak_fitness_seen
    print(f"Parent's real fitness on this run's frames: {best_fitness:.4f} (peak ever: {peak_fitness_seen:.4f})")

    def _save():
        sandbox.save_checkpoint({
            "genome": genome.to_dict(),
            "best_fitness": best_fitness,
            "peak_fitness_seen": peak_fitness_seen,
            "margin": margin,
            "habituation_exposure": habituation.exposure,
            "n_vars": n_vars,
            "clip_index": (clip_index + 1) % len(clips),
            "total_generation": box.generation,
        })

    while box.should_continue():
        # Dessert over (deadline passed, or cleared in the viewer): stop
        # this run so systemd brings it back on its home camera.
        if dessert is not None and box.generation % 25 == 0 and _dessert() is None:
            print(f"Dessert over -- returning to {home_source}.")
            break
        box.generation += 1
        if box.generation % 50 == 0:
            quota_pct = sandbox.load_quota_pct(REFERENCE_QUOTA_PCT)
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

        # Re-evaluate the PARENT fresh, right here, right now -- never
        # compare against a stored best_fitness that might be stale
        # (external audit finding B1, the actual root cause of the
        # plateau). The only thing that can legitimately drift between
        # generations is habituation.discount; re-scoring both parent
        # and candidate under the SAME current discount on the SAME
        # frames keeps every single accept/reject decision honest.
        # Live camera: refresh to the newest window every generation. Fair
        # by construction -- parent and candidate are both scored on this
        # same snapshot just below.
        # Every WORLD_REFRESH_GENERATIONS generations (a few seconds):
        # rebuilding the world's signals costs ~0.65 s, which every
        # generation would nearly double generation time.
        if feed is not None and box.generation % WORLD_REFRESH_GENERATIONS == 0:
            frames, vectors, total = feed.snapshot()
            world = World(frames, vectors)
            _observe(total - seen_total)
            seen_total = total
        parent_fitness, _, _ = evaluate_genome(genome, *world.at_pace(genome.pace), habituation.discount, quota_pct)
        candidate_fitness, breakdown, live_info = evaluate_genome(
            candidate, *world.at_pace(candidate.pace), habituation.discount, quota_pct,
        )

        both_finite = math.isfinite(candidate_fitness) and math.isfinite(parent_fitness)
        accepted = both_finite and candidate_fitness > parent_fitness + margin
        if not accepted and both_finite and candidate_fitness >= parent_fitness - NEUTRAL_EPSILON:
            # Real neutral drift (external audit's recommended fix for
            # a pure greedy hill-climb that can never cross a flat
            # plateau -- equal-fitness moves used to always be
            # rejected). A small, fixed chance to take a roughly-tied
            # step sideways; never a worse one.
            accepted = rng.random() < NEUTRAL_ACCEPT_PROB

        if accepted:
            # Evolution just chose to pay for a bigger look -- a real,
            # organism-derived demand for resources, surfaced as a request
            # resource_handler.py can weigh (with real fitness gain) to
            # grant more quota. Never granted by the organism itself.
            if applied == "mutate_fovea" and candidate.fovea_fraction > genome.fovea_fraction:
                box.log_request(
                    f"look grew {genome.fovea_fraction:.3f} -> {candidate.fovea_fraction:.3f} "
                    f"at quota {quota_pct:.0f}%"
                )
            genome = candidate
            best_fitness = candidate_fitness
            margin = max(0.005, margin * 0.995)
        elif both_finite:
            # Keep best_fitness in sync with reality even on a reject
            # -- it's the PARENT's own freshly-scored real fitness now,
            # never a stale ratchet (see B1 fix above).
            best_fitness = parent_fitness
        if math.isfinite(best_fitness):
            peak_fitness_seen = max(peak_fitness_seen, best_fitness)

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
            "parent_fitness": parent_fitness if math.isfinite(parent_fitness) else None,
            # best_fitness is now the CURRENT genome's real, freshly-
            # scored fitness every generation -- not a one-way ratchet
            # (see the B1 fix above). peak_fitness_seen is the
            # separate, purely informational running max.
            "best_fitness": best_fitness,
            "peak_fitness_seen": peak_fitness_seen,
            "breakdown": breakdown,
            "mutation_weights": dict(genome.mutation_weights),
            "meta_mutation_rate": genome.meta_mutation_rate,
            "margin": margin,
            "habituation_exposure": round(habituation.exposure, 4),
            "habituation_discount": round(habituation.discount, 4),
            "clip": clip_path,
            # Structural growth telemetry, added 2026-09-24 -- User:
            # "plot fitness, total nodes, nodes per channel, tree
            # depth, accepted mutation type, fitness delta... show
            # exactly when complexity increases and whether it earns
            # its structural cost." channel/applied were already
            # computed above for this generation's real mutation
            # attempt; tree_stats reflects the CURRENT (persisting)
            # genome, same as live_status.json's own tree_stats.
            "channel": channel,
            "mutation_type": applied,
            "fitness_delta": (candidate_fitness - parent_fitness) if both_finite else None,
            "fovea_fraction": genome.fovea_fraction,
            "quota_pct": quota_pct,
            "pace": genome.pace,
            "tree_stats": {
                name: {"nodes": tree.node_count(), "depth": tree.depth()}
                for name, tree in genome.trees.items()
            },
        })

        # Real-time-ish snapshot for anything polling from outside
        # (see HLSLS's own broadcast-api /api/cv-state precedent) --
        # every generation, not just every 100th checkpoint save; this
        # is small and cheap, unlike the full checkpoint.
        sandbox.save_live_status({
            "generation": box.generation,
            # No longer a ratchet -- see the B1 fix above. This is the
            # current genome's real fitness, re-scored fresh every
            # generation; peak_fitness_seen is the separate, purely
            # informational running max.
            "best_fitness": round(best_fitness, 4),
            "peak_fitness_seen": round(peak_fitness_seen, 4),
            "fovea_cx": round(live_info["fovea_cx"], 4),
            "fovea_cy": round(live_info["fovea_cy"], 4),
            "fovea_fraction": live_info["fovea_fraction"],
            "fovea_fraction_accepted": round(genome.fovea_fraction, 4),
            "quota_pct": quota_pct,
            # Gemini's homeostasis: the candidate's body at the end of this
            # generation's run, its energy over time, how many frames the
            # giant-fiber reflex took over, and the look's real path
            # (cx, cy, aperture) -- so the viewer can replay movement
            # instead of showing one end-point per generation.
            "body": live_info.get("body"),
            "energy_series": live_info.get("energy_series"),
            "reflex_frames": live_info.get("reflex_frames"),
            "trajectory": live_info.get("trajectory"),
            "food_series": live_info.get("food_series"),
            "mean_food": live_info.get("mean_food"),
            "movement": live_info.get("movement"),
            "field_events": live_info.get("field_events"),
            "flinch": live_info.get("flinch"),
            "max_fraction": fovea.MAX_FRACTION,
            # Pace of life: the accepted genome's, and the one the replay
            # below was recorded at (the candidate's) -- replay speed
            # depends on it (15 / pace looks per second).
            "pace_accepted": genome.pace,
            "pace": live_info.get("pace"),
            # Real analyzed frames per second (a live camera's rate varies
            # with light; files are treated as 15). The viewer derives
            # gazes/s, replay speed and real times from this.
            "frames_per_second": round(feed.frames_per_second(), 2) if feed is not None else 15.0,
            # The accepted genome's whole recurrent brain (Gemini's
            # MosquitoBrain) for the viewer's brain diagram, plus the
            # candidate's hidden state at the end of its run.
            "brain": genome.brain.to_dict(),
            "brain_hidden": live_info.get("brain_hidden"),
            # Real source frame shape -- User: "make foveal rectangle
            # honest." Lets the viewer draw the box at the REAL aspect
            # ratio instead of a hardcoded one.
            "frame_w": frame_w,
            "frame_h": frame_h,
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
            # User: "Why can't fovea rectangle display what the PROGRAM
            # is seeing. Screw the video. What is a pixel dump of what
            # it's seeing?" The WORLD's own 12x12 grid (same reduction
            # already used for reflex grading, never transmitted
            # before) -- same no-raw-frames justification the fovea
            # grid above already has (already reduced far past
            # anything resembling real footage), just for the FULL
            # frame instead of the fovea's own crop. Lets the viewer
            # draw the fovea box on a REAL pixel dump instead of a
            # third-party video embed -- no YouTube dependency, no
            # embedding restrictions, no autoplay/play-state games.
            "world_grid": [round(x, 4) for x in world.vectors[-1]],
            "world_grid_shape": list(GRID),
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
    if feed is not None:
        feed.close()
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
