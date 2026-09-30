#!/usr/bin/env python3
from __future__ import annotations

"""
Entry point: evolves an organism (fishbowl/) that lives on a real camera
or live stream. Each generation, the current genome (parent) and one
mutated child are scored on the SAME world snapshot, starting from the
SAME body and memory; the child replaces the parent only if it does
strictly better (plus a small chance of accepting a tie).

What it is scored on (evaluate_genome):
  - its body (fishbowl/state.py): homeostatic drive and drive reduction
    on a real clock -- energy spent on basal metabolism, muscle force,
    thinking and gaze size, restored only by eating;
  - food: prey (people/animals found by fishbowl/prey.py) held in the
    center of its gaze, plus small surprise snacks (habituating memory);
  - a few remaining hand-written pressures (curiosity, dead field,
    movement cost, corner/edge, flinch), to be retired over time.
Older correlation scores are still measured and logged, not scored.

The world (World): a live camera or live stream is read continuously
(video_source.LiveFeed) and each generation is scored on the newest
~600 frames, refreshed every few generations; a local file is one fixed
clip. Raw frames never leave memory -- only receptor grids and prey boxes.
On a live feed it also keeps a short ring of small JPEG frames in RAM
for the viewer's replay of its latest run (CAMBRIAN_CAMERA_PREVIEW=0
turns it off).

Meant to run for years as a BOUNDED process restarted by systemd
(Restart=always): each run resumes the genome, body and memory from
state/checkpoint.json (fishbowl/sandbox.py), and time spent down is
charged to the body.

Usage:
    python3 run_vision.py <camera device | "live" | video file> [--generations N] [--seconds S]
"""

import argparse
import contextlib
import math
import os

# One math thread per process (set before numpy loads): parallel work goes
# through worker processes (see _Workers), and a thread per core spinning in
# the math library only fights the host -- on a busy laptop it took 4 cores.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import random
import signal
import socket
import threading
import subprocess
import time
import sys
from pathlib import Path

from fishbowl import video_source as _quiet  # noqa: F401,E402  (sets OpenCV's log levels before cv2 loads)
import cv2
import numpy as np

from fishbowl import fovea, genome as G, hostspeed, reflexes, sandbox, video_source
from fishbowl.state import MosquitoState
from fishbowl import prey as prey_lib
from fishbowl.bouts import FeedingRecord, fit_bout_criterion
from fishbowl.metrics import HourlyMetrics
from fishbowl.livelife import LiveLife
from fishbowl.live import _b64, _learned_parts, apply_learned  # a body's frozen learned parts (model card, eggs)
from fishbowl.controller import TREE_HIDDEN
# The organism itself and the world's fixed physics it lives by -- one
# implementation, shared with any live host (fishbowl/organism.py).
from fishbowl.organism import (  # noqa: E402
    CHANNEL_COST, CONE_COST, CONSOLIDATE_RATE, EXPANSION_GAIN, FLOW_GAIN, FOOD_GAIN, MAX_INTERVAL,
    MEAN_RATE, MEM_H, MEM_W, MISMATCH_TAU_S, MOTION_GAIN, NOISE_FLOOR, PERIPH_MOTION_GAIN, PREY_SENSE_COST, REFERENCE_QUOTA_PCT,
    SHIFT_MAX, SHIFT_MIN_RESPONSE, SHIFT_WIDTH, STABILIZER_COST, SURPRISE_SIGMAS, TEMPO_RANGE, THINK_COST,
    RECEPTOR_COST, UNSEEN_NOVELTY, VAR_RATE, Organism, _receptor_cost,
    feed_on_novelty as _feed_on_novelty, global_shifts as _global_shifts,
    peripheral_motion_centroid as _peripheral_motion_centroid, prey_sense as _prey_sense, parallax_series,
)
from fishbowl.retina import field_shape, frame_to_vector

# How much each reflex/drive contributes to total fitness -- loom
# weighted heaviest, matching the real threat/food asymmetry discussed
# while designing this (missing a threat costs more than missing an
# opportunity). Not tuned against real data yet; a real, named,
# starting guess, not a claim of correctness. "optomotor" renamed to
# "motion_energy" (honest -- it's direction-blind, see reflexes.py's
# module docstring); "directional_motion" is the real thing, added the
# same day, correlated against response the same way as the others.
SIGNAL_WEIGHTS = {"luminance_change": 0.5, "motion_energy": 0.75, "directional_motion": 0.75, "loom": 2.0}

# CONSPEC (a hand-built face-template detector, after Morton & Johnson's
# newborn face-orienting theory) and its "seek" reward are RETIRED: on
# real frames the template fired on almost everything, so seek just
# pulled the gaze toward any dark triangle. Finding living things is now
# prey.py's job -- and YOLO there is itself a SHORTCUT: the goal is for
# the organism to grow its own prey detector (planned: its own perception
# learns "prey or not" from its receptors with YOLO as the teacher, then
# YOLO is weaned off). It only ever needs people and animals, not pencils.

# Real orienting pressure, added 2026-09-24 -- on a live kitten cam a
# cat was in view the whole time and the fovea never locked on. Confirmed by design
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
#   (seek, toward CONSPEC detections, is retired -- see above.)
PURSUIT_WEIGHT = 1.0  # retired from fitness with the other correlation scores; kept for the record

# Real movement-cost PENALTY, added 2026-09-24 after a real deployed
# genome was seen to corner-pin the fovea (land on one saturated
# jump, then stay there for the rest of the run) instead of gliding.
# Real animals pay genuine metabolic cost for large/fast eye or head
# movements -- nothing here cost anything before, so a big, mostly-
# unjustified jump was exactly as "free" as a small graded one (then,
# 39% of random genomes produced a near-maximal step vs. 7% a small one).
# This does NOT cap
# or forbid large jumps -- real food at the gaze center can still
# justify a big move when it's genuinely worth it; it just means an
# UNJUSTIFIED one no longer costs nothing. Costed on INTENT (the
# pre-clamp tanh output, see fovea.step's own docstring), not net
# realized displacement -- a tree that keeps outputting a maximal
# pan/tilt while already pinned against a wall is still "trying" every
# frame, real motor effort even though the wall zeroes out its net
# movement.
# Retired from fitness 2026-09-26, measured only: the body now pays for
# muscle force itself (force squared) and the eye has a spring back to
# centre, so this charged twice for exactly the sustained hold needed to
# watch someone off-centre (a design-panel call).
MOVEMENT_COST_WEIGHT = 0.5  # retired, kept for the record

# Corner penalty: genomes kept hiding in corners. Soft, not a hard constraint -- real reward can
# still outweigh it if a corner is ever genuinely the right place to
# be. Fixed constant, outside the genome's reach (same reasoning as
# _HEAD_SIZE: this grades behavior, it isn't a perception trait, so
# it doesn't evolve).
# Both corner and edge are measured on the gaze's CENTER over the whole
# frame (fixed 2026-09-26). They used to be normalized over the range the
# center could reach, which shrinks as the gaze widens: at aperture 0.6 the
# center could only travel 0.3..0.7, so almost any position scored as "on
# an edge" (the live organism read ~0.95) -- a penalty on a wide gaze, not
# on where it looked. The center can now reach the frame's edges
# (fovea.py), so following a cat along a wall is possible; the penalty
# stays soft, so food there can outweigh it.
CORNER_PENALTY_WEIGHT = 1.0  # doubled 2026-09-24


def _corner_penalty(positions: list[tuple[float, float]], fracs: list[float]) -> float:
    """
    "Cornerness" = product of how off-center the gaze's center is on each
    axis (normalized to [-1, 1] over the frame) -- zero along either center
    line, maximal only where BOTH axes are extreme at once (a real corner,
    not just one edge). Averaged over the run.
    """
    if not positions:
        return 0.0
    return float(np.mean([abs((cx - 0.5) / 0.5 * (cy - 0.5) / 0.5) for cx, cy in positions]))


# Shun edges unless they're worth it. Real,
# separate, LESSER cost than corner_penalty -- a single edge (one axis
# extreme, the other centered) is genuinely less wasteful than a true
# corner, so it costs less, not nothing. Still soft: seek/pursuit can
# outweigh it when an edge really is where the subject is.
EDGE_PENALTY_WEIGHT = 0.5  # doubled 2026-09-24


def _edge_penalty(positions: list[tuple[float, float]], fracs: list[float]) -> float:
    """Max (not product) of how off-center each axis is -- unlike cornerness, this alone is already high for EITHER a corner or a single edge; corner_penalty's own weight is what makes a true corner cost more overall."""
    if not positions:
        return 0.0
    return float(np.mean([max(abs(cx - 0.5), abs(cy - 0.5)) / 0.5 for cx, cy in positions]))


# Fitness from staying viable: mean homeostatic drive over the run
# (energy deficit, threat, fatigue -- MosquitoState.drive()). Mean, not
# a summed per-frame drive reduction: such a sum telescopes to
# D(start) - D(end) and ignores everything in between.
HOMEOSTASIS_WEIGHT = 3.0
DRIVE_REDUCTION_WEIGHT = 1.0
# Gemini's brain has an "alarm" output; it earns fitness by tracking
# real world loom (never forced to).
ALARM_WEIGHT = 1.0

# The perception tree's teacher (the teacher-student plan: YOLO is the
# shortcut, the tree is to become its own detector): the tree is graded on
# predicting, from its own pixels, how much prey fills its gaze window --
# confidence-weighted YOLO boxes as soft labels, error balanced between
# frames with and without prey so "never prey" can't score well. Its
# prediction also reaches the brain (the "tree" input), so a tree that sees
# food lets the brain steer to it.
TEACHER_WEIGHT = 3.0


def _round2(v):
    return None if v is None else round(float(v), 2)


# The flinch, evolved rather than wired: at each onset of a real
# approach (whole-field dark expansion crossing FLINCH_THRESHOLD), reward
# reacting within FLINCH_WINDOW frames (~200 ms at 15 frames/s) with a
# saccade -- more for a faster reaction, nothing for none. No approach in a
# run means no reward and no penalty. (Widening the gaze also counted while
# the eye could zoom; its eye is a fixed mosaic now -- fovea.py.)
FLINCH_WEIGHT = 1.0
FLINCH_THRESHOLD = 0.18
FLINCH_WINDOW = 3

def _flinch(looms: list[float], speeds: np.ndarray, intervals: list[int]) -> dict:
    """Onsets of real approach, and whether it made a saccade within
    FLINCH_WINDOW real frames -- gazes are unevenly spaced,
    so latency is counted in real frames (plus, on average, half the
    interval before the gaze that noticed it). A slow tempo pays for it."""
    onsets = [t for t in range(1, len(looms)) if looms[t] > FLINCH_THRESHOLD >= looms[t - 1]]
    latencies = []
    for t in onsets:
        lat, elapsed = None, (intervals[t - 1] - 1) / 2.0 if t - 1 < len(intervals) else 0.0
        i = t
        while i < len(speeds) and elapsed <= FLINCH_WINDOW:
            if speeds[i] >= 0.05:
                lat = elapsed
                break
            elapsed += intervals[i] if i < len(intervals) else 1
            i += 1
        latencies.append(lat)
    reacted = [l for l in latencies if l is not None]
    score = float(np.mean([1.0 - l / (FLINCH_WINDOW + 1) if l is not None else 0.0 for l in latencies])) if latencies else 0.0
    return {"events": len(onsets), "reacted": len(reacted),
            "mean_latency_frames": round(float(np.mean(reacted)), 2) if reacted else None, "score": round(score, 3)}


# The other boundary of the corridor -- like a deep-sea
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


# Curiosity -- it wasn't looking around at all. Confirmed by real data (evolution_log.json's own breakdown
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
    Real bug fix: the old definition made the fovea hopscotch instead of
    saccading. Verified
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


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")

# Training feed chosen for no people or picture-in-picture (2026-09-24,
# replacing the earlier
# cat_livestream/pixcams_wildlife pair). Confirmed live via yt-dlp
# metadata before wiring in (title: "Kitten Rescue Cat Cam powered by
# EXPLORE.org", is_live=True) -- a real, established source, no
# on-screen people and no compositing overlay to confuse the world-
# level reflex signals with something that isn't real scene content.
# Watched LIVE (resolved fresh every run via _resolve_live_url below),
# never downloaded -- genuinely different real content every restart.
#
# kitten_rescue_baby_kittens_cam
# now the sole source (previously the backup entry, added earlier the
# same day; already confirmed live via yt-dlp metadata then).
LIVE_SOURCES = [
    ("kitten_rescue_baby_kittens_cam", "https://www.youtube.com/watch?v=gBdqOuhj2P4"),
]


def _clip_display_name(source: str, clip_path: str) -> str:
    """
    A short, human-readable label for whatever's actually being
    watched right now, for the viewer. For live mode this is the curated name from
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
    yt_dlp = Path(sys.executable).parent / ("yt-dlp.exe" if os.name == "nt" else "yt-dlp")
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
    The FULL, un-foveated frame at every timestep, reduced to the whole
    field's square receptors (retina.field_shape) -- computed ONCE per run,
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


class _SignalsAt:
    """One frame's whole-field signals, read from a snapshot's arrays."""
    __slots__ = ("ws", "t")

    def __init__(self, ws: dict):
        self.ws, self.t = ws, 0

    def __getitem__(self, key: str):
        return self.ws[key][self.t]


def evaluate_genome(
    g: G.Genome,
    frames: list[np.ndarray],
    world_signals: dict[str, np.ndarray],
    quota_pct: float = REFERENCE_QUOTA_PCT,
    start_body: dict | None = None,
    fps: float = 15.0,
    world_prey: list | None = None,
    start_memory: tuple[np.ndarray, np.ndarray] | None = None,
    world_colour: list | None = None,
    sec_per_mac: float = 0.0,
) -> tuple[float, dict, dict]:
    """
    Returns (fitness, breakdown, live_info). world_signals are the
    world's own signals (whole field, computed per snapshot by World),
    independent of this genome's gaze. world_prey holds prey boxes per
    frame; start_body / start_memory are the organism's current body and
    surprise memory (shared by parent and candidate).
    """
    # One organism lives through the snapshot, frame by frame (fishbowl/
    # organism.py -- the same code a live host runs).
    org = Organism(g, start_body, start_memory, quota_pct, fps,
                   colour=world_colour is not None, prey=world_prey is not None, record=True, sec_per_mac=sec_per_mac)
    sig = _SignalsAt(world_signals)
    shift_x, shift_y = world_signals["shift_x"], world_signals["shift_y"]
    shift_s = world_signals.get("shift_s")
    if shift_s is None:
        shift_s = np.zeros(len(shift_x))
    shift_r = world_signals.get("shift_r")
    if shift_r is None:
        shift_r = np.zeros(len(shift_x))
    for t in range(len(frames)):
        sig.t = t
        org.frame(frames[t], sig, world_prey[t] if world_prey is not None and t < len(world_prey) else [],
                  world_colour[t] if world_colour is not None else None, (shift_x[t], shift_y[t], shift_s[t], shift_r[t]))
    org.finish()
    (positions, fracs, frame_path, responses, alarms, teacher_p, teacher_y, idxs, intervals, dxs, dys,
     movement_costs, periph_active, prey_eaten, foods, energies, drives, asleeps) = (org.rec[n] for n in Organism.RECORDS)
    state, body, brain = org.state, org.body, org.brain
    memory, variance, last_grid, last_colour = org.memory, org.variance, org.last_grid, org.last_colour
    drive_start, stab, prey_level = org.drive_start, org.stab, org.prey_level

    step_n = max(1, len(energies) // 60)
    sel = {k: np.asarray(v)[idxs] for k, v in world_signals.items()}
    iv = np.array(intervals, dtype=float) if intervals else np.ones(1)

    # Movement style, MEASURED not rewarded -- the yardstick from Land
    # (1969) on jumping-spider retinae: fixating (still), gliding (slow,
    # smooth), saccades (fast jumps), and tracking (following where the
    # whole field says something is moving). "Lifelike" as numbers
    # comparable to a real animal, not an impression.
    # Per base frame (1/15 s), so a slow pace can't look "smooth" just by
    # moving the same distance in fewer, bigger steps.
    # Everything about MOVEMENT is measured on the per-frame path over the
    # full world timeline (audit: sampling only at gaze frames let a slow
    # tempo dodge world-graded terms).
    fp = np.array(frame_path) if frame_path else np.array([[state.cx, state.cy, state.extent]])
    nf = len(fp)
    fvx = np.diff(fp[:, 0], prepend=fp[0, 0])
    fvy = np.diff(fp[:, 1], prepend=fp[0, 1])
    speeds = np.hypot(fvx, fvy)
    wmot = np.asarray(world_signals["motion_energy"][:nf]) * PERIPH_MOTION_GAIN
    active = wmot > 0.2
    tracking = None
    if active.sum() >= 10 and nf > 2:
        mcx = np.diff(np.asarray(world_signals["motion_cx"][:nf]), prepend=0.5)
        mcy = np.diff(np.asarray(world_signals["motion_cy"][:nf]), prepend=0.5)
        tracking = round((_correlate(mcx[active], fvx[active]) + _correlate(mcy[active], fvy[active])) / 2.0, 3)
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
        "fovea_fraction": state.extent,
        "receptors": state.n,
        "cones": org.cones,
        "last_response": responses[-1] if responses else 0.0,
        # The actual receptor values the organism just processed -- a
        # blocky n x n luminance grid, not an image, so it doesn't touch
        # the no-raw-frames rule.
        "grid": last_grid.tolist() if last_grid is not None else [],
        "grid_shape": [state.n, state.n],
        # the gaze's colour receptors at the end of the run (red-green,
        # then blue-yellow), for the viewer; empty without colour vision
        "colour_grid": [round(float(x), 4) for x in last_colour] if last_colour is not None else [],
        "body": body.to_dict(),
        "energy_series": [round(e, 4) for e in energies[::step_n]],
        "sleep_series": [round(float(a), 2) for a in asleeps[::step_n]],
        # What it's eating: genuinely new visual structure per frame
        # (see _feed_on_novelty), sampled like energy.
        "food_series": [round(float(np.mean(foods[k:k + step_n])), 4) for k in range(0, len(foods), step_n)],
        "mean_food": round(float(np.mean(foods)), 4) if foods else 0.0,
        "mean_prey": round(float(np.mean(prey_eaten)), 4) if prey_eaten else 0.0,
        "prey_series": [round(float(np.mean(prey_eaten[k:k + step_n])), 4) for k in range(0, len(prey_eaten), step_n)],
        # prey boxes per frame + whether it was eating at that frame (held
        # from the last gaze), for the viewer
        "prey_boxes": [prey_lib.hosts_only(world_prey[k]) if world_prey is not None and k < len(world_prey) else [] for k in range(nf)],
        "plant_boxes": [prey_lib.plants_only(world_prey[k]) if world_prey is not None and k < len(world_prey) else [] for k in range(nf)],
        "thing_boxes": [prey_lib.things_only(world_prey[k]) if world_prey is not None and k < len(world_prey) else [] for k in range(nf)],
        "eating": [round(float(prey_eaten[max(0, int(np.searchsorted(idxs, k, side='right')) - 1)]), 3) if prey_eaten else 0.0 for k in range(nf)],
        # Per frame: its perception tree's own guess at how much prey fills
        # its gaze, next to the teacher's (YOLO's) label -- for the viewer.
        "snacks": [round(float(foods[max(0, int(np.searchsorted(idxs, k, side='right')) - 1)]), 3) if foods else 0.0 for k in range(nf)],
        "tree_guess": [_round2(teacher_p[max(0, int(np.searchsorted(idxs, k, side='right')) - 1)]) if teacher_p else None for k in range(nf)],
        "teacher_label": [_round2(teacher_y[max(0, int(np.searchsorted(idxs, k, side='right')) - 1)]) if teacher_y else None for k in range(nf)],
        # Per frame: asleep (1), eyes shut -- for the viewer's SLEEP.
        "asleep": [int(asleeps[max(0, int(np.searchsorted(idxs, k, side='right')) - 1)] >= 0.5) if asleeps else 0 for k in range(nf)],
        # Memory carried to the next generation: surprise, the mushroom body's
        # learning, the place map and where people are expected (day, night).
        "_memory": (memory, variance, org.mb.weights, org.place, org.people_day, org.people_night, org.mb.danger_weights,
                    org.value_map, dict(org.nectar), org.mb.proto, org.library(), org.ground, org.mb.heads, org.terrain,
                    org.frame_map),
        # Per frame: its alarm (the warning) and the intruder sense, for the viewer.
        "alarm": [int(alarms[max(0, int(np.searchsorted(idxs, k, side='right')) - 1)] > 0.0) if alarms else 0 for k in range(nf)],
        "replays": dict(org.replays),
        "intruder": round(org.intruder, 3),
        "food_value": round(org.food_value, 3),
        # Host defense: the frames of the looks it was swatted on, its learned danger now.
        "swat_acts": [int(k) for k in org.swat_frames],
        "danger_value": round(org.danger_value, 3),
        # How often a look was missed because its brain hadn't finished (its real reaction time).
        "missed_share": round(org.missed / max(1, len(idxs)), 3),
        "movement": movement,
        "field_events": [[round(float(world_signals["motion_cx"][k]), 3), round(float(world_signals["motion_cy"][k]), 3),
                          round(float(min(1.0, world_signals["motion_energy"][k] * PERIPH_MOTION_GAIN)), 3),
                          round(float(min(1.0, world_signals["expansion"][k] * EXPANSION_GAIN)), 3), 0] for k in range(nf)],
        "flinch": _flinch(list(np.minimum(1.0, np.asarray(world_signals["expansion"][:nf]) * EXPANSION_GAIN)), speeds, [1] * nf),
        "pace": round(float(iv.mean()), 2),  # mean gaze interval actually used this run
        "tempo_series": [round(float(np.mean(intervals[k:k + step_n])), 2) for k in range(0, len(intervals), step_n)],
        "brain_hidden": [round(h, 3) for h in brain.hidden],
        # (cx, cy, gaze side as a fraction of the frame's height, frame index): gazes are unevenly spaced now,
        # so the viewer replays each at its real moment.
        "trajectory": [[round(float(x), 4), round(float(y), 4), round(float(f), 4), i] for i, (x, y, f) in enumerate(fp)],
        # Its feeding acts: the frames of the looks that caught prey / any
        # surprise (fishbowl/bouts.py groups them into meals and snacks).
        "meal_acts": [int(k) for k, p in zip(idxs, prey_eaten) if p > 0],
        "snack_acts": [int(k) for k, s in zip(idxs, foods) if s > 0],
    }

    responses = np.array(responses)

    if not np.all(np.isfinite(responses)) or not np.all(np.isfinite(alarms)):
        return float("-inf"), {}, live_info

    signals = sel

    fitness = 0.0
    breakdown = {}
    # RETIRED from fitness (after two independent audits: the correlation block
    # carried 80-95% of selection while moving nothing in the body, and
    # its heaviest term graded the discredited loom detector). Still
    # measured, for the record.
    for name in SIGNAL_WEIGHTS:
        breakdown[name] = _correlate(signals[name], responses)

    breakdown["loom_max"] = float(signals["loom"].max())

    curiosity = _curiosity_score([(q[0], q[1]) for q in fp])  # per real frame, on the per-frame path
    fitness += CURIOSITY_WEIGHT * curiosity
    breakdown["curiosity"] = curiosity

    dead_field = _dead_field_penalty({k: np.asarray(v)[:nf] for k, v in world_signals.items()})  # full timeline
    fitness -= DEAD_FIELD_PENALTY_WEIGHT * dead_field
    breakdown["dead_field_penalty"] = dead_field

    # Does pan/tilt's real movement track real motion direction? (retired
    # from fitness, measured only)
    pursuit = (_correlate(signals["motion_x"], np.array(dxs)) + _correlate(signals["motion_y"], np.array(dys))) / 2.0
    breakdown["optokinetic_pursuit"] = pursuit  # retired from fitness (correlation score); measured only

    movement_cost = float(np.mean(movement_costs)) if movement_costs else 0.0
    breakdown["movement_cost"] = movement_cost

    corner = _corner_penalty([(q[0], q[1]) for q in fp], list(fp[:, 2]))
    fitness -= CORNER_PENALTY_WEIGHT * corner
    breakdown["corner_penalty"] = corner

    edge = _edge_penalty([(q[0], q[1]) for q in fp], list(fp[:, 2]))
    fitness -= EDGE_PENALTY_WEIGHT * edge
    breakdown["edge_penalty"] = edge

    mean_drive = float(np.mean(drives)) if drives else 0.0
    fitness -= HOMEOSTASIS_WEIGHT * mean_drive
    # Keramati & Gutkin (2014) homeostatic reward: drive reduction --
    # did this window leave its body better or worse off? Parent and
    # candidate start from the same body, so this is a fair comparison.
    # (Constants audit: it used to be extrapolated x 1200 s / the window's
    # length, which blew a few seconds' noise up into the largest term.)
    drive_reduction = drive_start - body.drive()
    fitness += DRIVE_REDUCTION_WEIGHT * drive_reduction

    # The perception tree's teacher (see TEACHER_WEIGHT).
    teacher_error = None
    graded = [(p, y) for p, y in zip(teacher_p, teacher_y) if y is not None]
    if graded:
        tp, ty = np.array([p for p, _ in graded]), np.array([y for _, y in graded])
        with_prey = ty > 0.05
        parts = [float(np.mean((tp[m] - ty[m]) ** 2)) for m in (with_prey, ~with_prey) if m.any()]
        teacher_error = float(np.mean(parts))
        fitness -= TEACHER_WEIGHT * teacher_error
    breakdown["teacher_error"] = teacher_error
    breakdown["prey_sense"] = prey_level
    breakdown["drive_reduction"] = drive_reduction
    breakdown["mean_drive"] = mean_drive
    breakdown["mean_energy"] = float(np.mean(energies)) if energies else 0.0
    breakdown["asleep_share"] = float(np.mean(asleeps)) if asleeps else 0.0
    breakdown["loop_weight"] = brain.loop_synapses()
    breakdown["stabilizer"] = stab
    breakdown["shake"] = float(np.mean(np.hypot(world_signals["shift_x"], world_signals["shift_y"])))
    breakdown["final_energy"] = energies[-1] if energies else 0.0
    breakdown["mean_food"] = float(np.mean(foods)) if foods else 0.0
    breakdown["mean_prey"] = float(np.mean(prey_eaten)) if prey_eaten else 0.0
    breakdown["mean_aperture"] = float(np.mean(fracs)) if fracs else 0.0
    breakdown["movement"] = live_info["movement"]
    breakdown["missed_looks"] = org.missed
    breakdown["kc"] = org.mb.n_kc
    breakdown["value_error"] = float(np.mean(org.value_errors)) if org.value_errors else None

    flinch = live_info["flinch"]
    fitness += FLINCH_WEIGHT * flinch["score"]
    breakdown["flinch"] = flinch["score"]
    breakdown["loom_events"] = flinch["events"]

    breakdown["alarm_loom"] = _correlate(signals["expansion"], np.array(alarms))  # retired from fitness; measured only

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


# ---- Parallel evaluation: several children per generation -----------------
# A (1+lambda) search: each generation mutates lambda children from the
# parent and scores them at once in worker processes, on the same world
# snapshot, body and memory as the parent (scored meanwhile in the main
# process). The world's frames go into shared memory once per snapshot, never
# copied per child. lambda follows the CPU it is granted -- the resource
# handler's quota, which already tracks what is idle: one core per 100%, less
# one for the main process (capture, detection, the parent), at most the
# host's cores minus one. CAMBRIAN_WORKERS=<n> overrides; 1 = serial.
# The best child must pass the usual acceptance, then a re-check on the
# previous snapshot (RECHECK): picking the best of several children favours
# one that merely exploits this particular window.
_WORKER: dict = {"id": None, "shm": [], "frames": None, "colour": None}


def _n_children(quota_pct: float) -> int:
    cap = max(1, (os.cpu_count() or 2) - 1)
    env = os.environ.get("CAMBRIAN_WORKERS")
    if env is not None and env.strip().isdigit():
        return max(1, min(cap, int(env)))
    return max(1, min(cap, int(quota_pct // 100) - 1))


def _worker_attach(meta: dict) -> None:
    """In a worker: its snapshot's frames, mapped from shared memory (once per snapshot)."""
    from multiprocessing import shared_memory
    if _WORKER["id"] != meta["id"]:
        for old in _WORKER["shm"]:
            old.close()
        _WORKER["shm"] = []
        arrays = {}
        for key in ("grey", "colour"):
            spec = meta.get(key)
            if spec is None:
                arrays[key] = None
                continue
            name, shape, dtype = spec
            block = shared_memory.SharedMemory(name=name, track=False)
            _WORKER["shm"].append(block)
            arrays[key] = np.ndarray(shape, dtype=dtype, buffer=block.buf)
        _WORKER["frames"] = [arrays["grey"][k] for k in range(arrays["grey"].shape[0])]
        _WORKER["colour"] = [arrays["colour"][k] for k in range(arrays["colour"].shape[0])] if arrays["colour"] is not None else None
        _WORKER["id"] = meta["id"]


def _worker_signals(meta: dict, vectors, fps: float) -> dict:
    """Runs in a worker process: this snapshot's world signals (World.at_pace),
    built here so the organism's own process -- its live body's -- never
    spends its time on them (a 2026-09-30 measurement: 7 s a snapshot)."""
    _worker_attach(meta)
    return World(_WORKER["frames"], vectors, fps, meta["prey"], _WORKER["colour"]).at_pace(1)[1]


def _worker_evaluate(meta: dict, genome_dict: dict, quota_pct: float, body: dict, fps: float, memory, sec_per_mac: float = 0.0):
    """Runs in a worker process: attaches to the snapshot's frames (once per
    snapshot) and scores one child."""
    _worker_attach(meta)
    g = G.Genome.from_dict(genome_dict)
    return evaluate_genome(g, _WORKER["frames"], meta["ws"], quota_pct, body, fps, meta["prey"], memory, _WORKER["colour"], sec_per_mac)


def _kernel32():
    """kernel32 with the handle types declared: undeclared, ctypes passes the
    process's pseudo-handle as a 32-bit int on 64-bit Windows -- an invalid
    handle, and the call fails silently."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    return k32


def _cgroup_base():
    """This service's own cgroup (v2), when systemd delegated it (Delegate=yes)
    and it can write there; its body's cgroup's parent, once split."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        line = next(l for l in open("/proc/self/cgroup", encoding="utf-8").read().splitlines() if l.startswith("0::"))
    except (OSError, StopIteration):
        return None
    base = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    if base.name in ("body", "workers"):
        base = base.parent
    return base if os.access(base / "cgroup.subtree_control", os.W_OK) else None


def _split_body_and_workers() -> None:
    """Linux (a 2026-09-29 panel; Poettering): the kernel, not niceness, keeps
    evolution out of its body's way. Its service's delegated cgroup splits in
    two: body/ (this process, and what it starts) with its memory protected
    (memory.low = max: under pressure the workers' pages are reclaimed first),
    and workers/ at the idle scheduling class (cpu.idle: they run only on time
    nothing else wants). The service's own caps (CPUQuota, MemoryMax) still
    bound the two together. Without delegation (or cgroup v2), nothing changes."""
    base = _cgroup_base()
    if base is None:
        return
    try:
        for d in ("body", "workers"):
            (base / d).mkdir(exist_ok=True)
        (base / "body" / "cgroup.procs").write_text(str(os.getpid()))
        (base / "cgroup.subtree_control").write_text("+cpu +memory")
        try:
            (base / "workers" / "cpu.idle").write_text("1")
        except OSError:
            (base / "workers" / "cpu.weight").write_text("1")  # kernels before 5.15: the lowest weight
        (base / "body" / "memory.low").write_text("max")
        print("Its body and its evolution in their own cgroups (workers idle; its memory protected).")
    except OSError as e:
        print(f"Its cgroups couldn't be split ({e}); niceness only.")


def _worker_below_the_body() -> None:
    """A worker runs one step below the organism's own process: the living
    organism (its live actor, in the main process) is real time, evolution is
    background -- on a busy host the scheduler serves the body first, and
    evolution gets what is left (Windows: background mode -- the documented
    way to lower a process's CPU, I/O and memory priority together, so under
    memory pressure its pages go before the body's (Russinovich); Linux: the
    idle cgroup (Poettering, _split_body_and_workers); macOS: the Darwin
    background band (Oakley); and everywhere but Windows, 5 more niceness).
    And OpenCV runs single-threaded in it (one thread a worker, as its BLAS):
    a worker forked from a process whose OpenCV had started its thread pool
    inherits a broken pool and crashes in it (macOS, 2026-09-30: a segfault
    in calcOpticalFlowPyrLK's parallel_for)."""
    try:
        cv2.setNumThreads(1)
    except Exception:  # an OpenCV without the call: its default
        pass
    try:
        if os.name == "nt":
            k32 = _kernel32()
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00100000)  # PROCESS_MODE_BACKGROUND_BEGIN (I/O, memory)
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000040)  # then IDLE_PRIORITY_CLASS (CPU): in this order both hold
        else:
            os.nice(5)
            base = _cgroup_base()
            if base is not None and (base / "workers" / "cgroup.procs").exists():
                (base / "workers" / "cgroup.procs").write_text(str(os.getpid()))  # Linux: into the idle cgroup
            if sys.platform == "darwin":
                # macOS (Oakley, Levin): the Darwin background band -- efficiency
                # cores on Apple Silicon, throttled disk -- PRIO_DARWIN_PROCESS, PRIO_DARWIN_BG
                import ctypes
                ctypes.CDLL(None, use_errno=True).setpriority(4, 0, 0x1000)
    except (OSError, AttributeError):
        pass


class _Workers:
    """The worker pool and the snapshot it is sharing."""

    def __init__(self, n: int):
        import concurrent.futures
        import multiprocessing
        # One thread each: the workers already use every core they are given.
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        # forkserver where the platform has it (Linux, macOS); spawn on Windows.
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context(method),
                                                           initializer=_worker_below_the_body)
        self.world, self.meta, self.shm = None, None, []
        self.prev_world, self.prev_meta, self.prev_shm = None, None, []  # the snapshot before, kept for the re-check
        import atexit
        atexit.register(self.close)  # however the run ends, its shared frames are released

    def publish(self, world: "World") -> bool:
        """Shares this snapshot's frames (once). False if they can't be shared."""
        if world is self.world:
            return self.meta is not None
        from multiprocessing import shared_memory
        meta, blocks = None, []
        try:
            arrays = {"grey": np.stack(world.frames)}
            if world.colour is not None:
                arrays["colour"] = np.stack(world.colour)
            meta = {"id": f"{os.getpid()}-{time.time_ns()}", "prey": world.prey, "colour": None}
            for key, arr in arrays.items():
                block = shared_memory.SharedMemory(create=True, size=max(1, arr.nbytes))
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=block.buf)[...] = arr
                blocks.append(block)
                meta[key] = (block.name, arr.shape, arr.dtype.str)
            # its signals, built by a worker (this process only waits: its live body's thread runs on);
            # kept as the snapshot's own, so nothing here computes them again
            if 1 in world._cache:
                meta["ws"] = world._cache[1][1]
            else:
                try:
                    ws = self.pool.submit(_worker_signals, meta, world.vectors, world.fps).result(timeout=600)
                except Exception as e:  # a lost worker: built here instead
                    print(f"Worker failed building the snapshot's signals ({type(e).__name__}); building them here.")
                    ws = world.at_pace(1)[1]
                world._cache[1] = (world.frames[::1], ws)
                meta["ws"] = ws
        except (ValueError, OSError) as e:  # frames of different sizes, or no shared memory
            print(f"Parallel evaluation off for this snapshot ({e}).")
            for block in blocks:
                block.close()
                block.unlink()
            meta, blocks = None, []
        # the snapshot before stays shared (its re-check runs there); the one before that is released
        older = self.prev_shm
        self.prev_world, self.prev_meta, self.prev_shm = self.world, self.meta, self.shm
        self.world, self.meta, self.shm = world, meta, blocks
        for block in older:  # workers still mapping these keep them until they move on
            block.close()
            block.unlink()
        return meta is not None

    def meta_for(self, world: "World"):
        """The shared snapshot for this world (the current one or the one before), or None."""
        return self.meta if world is self.world else self.prev_meta if world is self.prev_world else None

    def submit(self, genome, quota_pct: float, body: dict, fps: float, memory, sec_per_mac: float = 0.0, meta: dict | None = None):
        return self.pool.submit(_worker_evaluate, meta or self.meta, genome.to_dict(), quota_pct, body, fps, memory, sec_per_mac)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        for block in self.shm + self.prev_shm:
            try:
                block.close()
                block.unlink()
            except (FileNotFoundError, OSError):
                pass
        self.shm, self.prev_shm = [], []


WORLD_REFRESH_GENERATIONS = 5
# A feed has stalled when no new frame has come for this many of its own
# longest gaps (its cadence, bursts included) -- the HLS convention of giving
# up after a few missed reload periods, applied to whatever the source is.
STALL_CADENCES = 3
# A web stream arrives in segments a few seconds apart (YouTube live: ~2-5 s),
# so it gets at least a few missed reloads' worth before it counts as stalled.
STREAM_STALL_MIN_S = 30.0
CHECKPOINT_EVERY_S = 600.0
# Its watchdog (a 2026-09-29 panel; Poettering): alive only while (a) its body
# lives a frame within the feed's own stall line whenever frames are arriving,
# and (b) its main loop progresses within its own longest wait (a worker's
# result, the feed's fill) plus the last generation's length. Else it exits and
# its supervisor starts it again; under systemd the same verdict feeds WatchdogSec.
WATCHDOG_MAIN_S = 600.0  # as its episodes (livelife.EPISODES_EVERY_S): a crash loses at most this much
# ...or sooner, once the snapshot is older than this or three generations,
# whichever is longer: on a slow host its gaze stays close to live (and a
# dead feed is noticed) without spending most of its time refreshing.
WORLD_REFRESH_MAX_S = 10.0


class World:
    """
    One snapshot of the world (frames + their retina vectors), with the
    world signals every genome is graded on computed on demand at each
    pace of life actually in use, then cached. Parent and candidate are
    always scored on the same World within a generation.
    """

    def __init__(self, frames: list[np.ndarray], vectors: np.ndarray, fps: float = 15.0, prey: list | None = None,
                 colour: list | None = None):
        self.frames, self.vectors = frames, vectors
        self.colour = colour  # colour frames (memory only), for the gaze's colour receptors
        self.t_newest = None  # arrival time of its newest frame (live feeds only)
        self.first_index = None  # the feed's index of its first frame (live feeds only)
        self.prey = prey if prey is not None else [[] for _ in frames]  # prey boxes per frame (fishbowl/prey.py)
        # Frozen with the snapshot: parent and candidate must be scored at
        # the SAME rate (audit: reading the live rate per evaluation could
        # differ by 0.1 fps between them -- enough to flip a decision).
        self.fps = fps if fps and fps > 0 else 15.0
        self._cache: dict[int, tuple] = {}

    @property
    def field_shape(self) -> tuple[int, int]:
        """The whole field's receptors (rows, cols) on this snapshot's frames."""
        return field_shape(*self.frames[0].shape[:2])

    def at_pace(self, pace: int):
        pace = max(1, int(pace))
        if pace not in self._cache:
            wv = self.vectors[::pace]
            ws = reflexes.all_signals(wv, self.field_shape)
            ws["expansion"] = reflexes.expansion_score(wv)
            ws["motion_cx"], ws["motion_cy"] = _peripheral_motion_centroid(wv, self.field_shape)
            ws["field_light"] = np.asarray(wv).mean(axis=1)
            ws["structure"] = np.asarray(wv, dtype=float) - np.asarray(wv, dtype=float).mean(axis=1, keepdims=True)
            ws["mismatch"], ws["mismatch_cx"], ws["mismatch_cy"], ws["mismatch_map"] = reflexes.mismatch_score(
                wv, self.field_shape, pace / self.fps, MISMATCH_TAU_S, SURPRISE_SIGMAS, NOISE_FLOOR)
            ws["shift_r"] = np.zeros(len(self.frames[::pace]))
            votes = []
            ws["shift_x"], ws["shift_y"], ws["shift_s"] = _global_shifts(self.frames[::pace], None, ws["shift_r"], votes)
            ws["frame_votes"] = np.empty(len(votes), dtype=object)  # ragged: each frame's own corners
            ws["frame_votes"][:] = votes
            dv = np.abs(np.diff(np.asarray(wv, dtype=float), axis=0))
            ws["motion_map"] = np.clip(np.vstack([np.zeros((1, dv.shape[1])), dv]) * PERIPH_MOTION_GAIN, 0.0, 1.0) if len(wv) > 1 \
                else np.zeros((len(wv), np.asarray(wv).shape[-1]))
            ws["parallax"] = parallax_series(self.frames[::pace], ws["shift_x"], ws["shift_y"], self.field_shape, ws["shift_s"])
            self._cache[pace] = (self.frames[::pace], ws)
        return self._cache[pace]


def _is_device(source: str) -> bool:
    """Its own camera: a local device, or a network camera's stream (e.g.
    rtsp://127.0.0.1:8554/cam, a livecam server's output on the same host)."""
    return source.startswith(("/dev/video", "/dev/v4l/", "rtsp://", "rtsps://", "srt://")) or source.isdigit()


def _dessert() -> dict | None:
    """
    A human-chosen live stream
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


# The perception tree's plain inputs: its own last movement (x, y) and the
# brain's units it reads; its receptors are read by position besides (blocks.py).
TREE_PLAIN_INPUTS = 2 + TREE_HIDDEN


def _pack_proto(proto) -> dict | None:
    """Its imagery prototypes, compact for the checkpoint: 8-bit, base64."""
    if proto is None or not len(proto):
        return None
    import base64
    q = np.clip(np.round(np.asarray(proto) * 255.0), 0, 255).astype(np.uint8)
    return {"shape": list(q.shape), "b64": base64.b64encode(q.tobytes()).decode("ascii")}


def _unpack_proto(packed):
    if not packed:
        return None
    import base64
    q = np.frombuffer(base64.b64decode(packed["b64"]), dtype=np.uint8).reshape(packed["shape"])
    return q.astype(np.float32) / 255.0


def _pack_scenes(lib) -> dict | None:
    """Its scene library for the checkpoint: each scene's layout and maps (NaN, never seen, as null)."""
    if not lib:
        return None
    def arr(a):
        return None if a is None else [[None if not np.isfinite(x) else round(float(x), 5) for x in row]
                                       for row in np.atleast_2d(np.asarray(a, dtype=float))]
    return {"current": int(lib["current"]),
            "library": [{**{k: arr(s.get(k)) for k in ("gist", "place", "people_day", "people_night", "value_map", "memory", "variance", "ground", "terrain")},
                         "gist_shape": list(np.shape(s["gist"])), "last": float(s.get("last", 0.0)),
                         "heading": None if s.get("heading") is None else round(float(s["heading"]), 5),
                         "position": None if s.get("position") is None else [round(float(v), 4) for v in s["position"]],
                         "nectar": {k: round(float(v), 4) for k, v in (s.get("nectar") or {}).items()}}
                        for s in lib["library"]]}


def _unpack_scenes(packed):
    if not packed:
        return None
    def arr(a):
        return None if a is None else np.array([[np.nan if x is None else x for x in row] for row in a], dtype=float)
    lib = []
    for s in packed["library"]:
        sc = {k: arr(s.get(k)) for k in ("place", "people_day", "people_night", "value_map", "memory", "variance", "ground", "terrain")}
        sc["gist"] = arr(s["gist"]).reshape(s.get("gist_shape") or -1)
        sc["last"], sc["nectar"] = float(s.get("last", 0.0)), dict(s.get("nectar") or {})
        if s.get("heading") is not None:
            sc["heading"] = float(s["heading"])
        if s.get("position") is not None:
            sc["position"] = [float(v) for v in s["position"]]
        lib.append(sc)
    return {"current": int(packed["current"]), "library": lib}


def _default_brain(genome) -> None:
    """A fresh install's founder (a 2026-09-29 panel): a random genome with
    every seed set applied (tools/seed.py) -- random, as Dennett's compromise
    asks, but with every capacity switched on for evolution to prune -- and
    recorded in state/seeded.txt, so none is applied twice."""
    import random as _random
    from tools import seed as seeds
    lines = []
    for name, fn in seeds.SEED_SETS.items():
        done = fn(genome, _random.Random(f"{seeds.SEED}:{name}:{time.time_ns()}"))
        lines.append(f"{name}: " + "; ".join(done or ["nothing needed"]))
    # and, as a young brain does, it overproduces at random (tools/seed.py)
    lines.append("founder, at random: " + "; ".join(seeds.random_draws(genome, _random.Random(time.time_ns()))))
    try:
        sandbox.STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(sandbox.STATE_DIR / "seeded.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
    print("A default brain: a random founder, every seed set applied.")


def memory_from_checkpoint(checkpoint: dict | None):
    """The memory tuple (evaluate_genome's order) saved in a checkpoint, or None."""
    if not checkpoint or not checkpoint.get("memory"):
        return None
    m = checkpoint["memory"]
    return (np.array([[np.nan if x is None else x for x in row] for row in m["mean"]], dtype=float),
            np.array(m["var"], dtype=float),
            np.array(m.get("learned") or [], dtype=float),
            *(np.array(m[k], dtype=float) if m.get(k) else None for k in ("place", "people_day", "people_night", "danger", "value")),
            dict(m.get("nectar") or {}),
            _unpack_proto(m.get("proto")),
            _unpack_scenes(m.get("scenes")),
            np.array(m["ground"], dtype=float) if m.get("ground") else None,
            np.array(m["heads"], dtype=float) if m.get("heads") else None,
            np.array(m["terrain"], dtype=float) if m.get("terrain") else None,
            np.array(m["frames"], dtype=float) if m.get("frames") else None)


def _for_evaluation(memory):
    """Memory as the children are scored from, without the imagery prototypes
    (megabytes each, and a child's run needs only their price, not them)."""
    if memory is None or len(memory) <= 9:
        return memory
    return tuple(memory[:9]) + (None,) + tuple(memory[10:])


def _stream_failed(dessert: dict | None) -> None:
    """The chosen stream failed this run. Its choice is dropped (back to the
    camera) only after it fails sandbox.SOURCE_FAILURES_TO_DROP runs in a
    row; one hiccup just ends this run, and the next one tries it again."""
    if dessert is not None and sandbox.source_failed(dessert["url"]):
        print(f"The chosen stream failed {sandbox.SOURCE_FAILURES_TO_DROP} runs in a row -- dropping it.")
        sandbox.clear_selected_source(dessert["url"])
        sandbox.source_opened()


def run(source: str, limits: sandbox.Limits, n_vars: int = TREE_PLAIN_INPUTS) -> None:
    # Dessert overrides the camera until its deadline; then this run
    # exits and systemd restarts it back on the camera.
    home_source = source
    watchdog = _Watchdog()
    # A stale stop request must not stop a new run (it is checked from here on,
    # while the feed fills and every generation).
    sandbox.STOP_REQUEST_PATH.unlink(missing_ok=True)
    dessert = _dessert() if _is_device(source) else None
    if dessert is not None:
        print(f"Dessert: watching {dessert['url']} until {time.ctime(float(dessert['until'])) if dessert.get('until') else 'cleared'}")
        source = "live"
    clips = _list_clips(source)

    checkpoint = sandbox.load_checkpoint()
    # Experiments done to this lineage (tools/reset_mind.py, tools/stroke.py):
    # carried through every save, and the stroke shown in the status.
    lineage_notes = {k: checkpoint[k] for k in ("mind_reset_at_generation", "stroke") if checkpoint and k in checkpoint}
    rng = random.Random()

    clip_index = 0
    if checkpoint is not None:
        clip_index = int(checkpoint.get("clip_index", 0)) % len(clips)
    clip_index = clip_index % len(clips)
    clip_path = clips[clip_index]

    # Human-only override (which video it watches). Written by tools/viewer.py, never by the organism --
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
        try:
            load_source = _resolve_live_url(clip_path)
        except Exception as e:
            # Audit: a chosen stream that ended used to crash here forever
            # (Restart=always). Drop the choice and go back to the camera --
            # but only once it has failed a few runs in a row: one timeout
            # is a network hiccup, not a stream that ended.
            print(f"Live stream unavailable ({e}).")
            _stream_failed(dessert)
            return

    clip_name = _clip_display_name(source, clip_path)

    # A live camera OR a live stream is read continuously (LiveFeed) and
    # every generation is scored on the newest ~600 frames -- the world it
    # is living in now (audit: a fixed stream clip reused for an hour let
    # the organism memorise it). Only local files keep one fixed clip.
    feed = None
    # This run's id: frame files carry it, so a restart never collides with
    # (or wipes) the frames the viewer is still playing from the last run.
    feed_epoch = int(time.time())
    if _is_device(source) or source == "live":
        print(f"Opening live feed {clip_path!r} (frames kept in memory only, never written to disk)...")
        detector = prey_lib.PreyDetector()
        print(f"Prey detector: {'yolov8n loaded' if detector.available else 'MODEL MISSING -- no prey, snacks only'} ({detector.model_path}); plants: {'Open Images V7 model loaded' if detector.flower_net is not None else 'potted plants only (no yolov8n-oiv7 model)'}")
        feed_src = (int(source) if source.isdigit() else source) if _is_device(source) else load_source
        # Its frames for the viewer's replay (small JPEGs, a short ring):
        # only when the runtime dir really is in RAM.
        frames_dir = sandbox.LIVE_STATUS_PATH.with_name("frames")
        in_ram = sandbox.LIVE_STATUS_PATH.parent != sandbox.STATE_DIR
        use_frames = in_ram and os.environ.get("CAMBRIAN_CAMERA_PREVIEW", "1") != "0"
        if in_ram:
            for old in ("camera.jpg", "camera.json"):  # the earlier single preview
                sandbox.LIVE_STATUS_PATH.with_name(old).unlink(missing_ok=True)
        if use_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
        elif os.environ.get("CAMBRIAN_CAMERA_PREVIEW", "1") != "0":
            # No RAM dir (Windows, macOS): the frames stay in this process's
            # memory and the viewer fetches them over localhost.
            frames_dir = video_source.FrameRing()
            frames_dir.serve(sandbox.LIVE_STATUS_PATH.with_name("frames.json"))
            use_frames = True
        feed = video_source.LiveFeed(feed_src, detector=detector if detector.available else None,
                                     frames_dir=frames_dir if use_frames else None, epoch=feed_epoch,
                                     resolve=(lambda: _resolve_live_url(clip_path)) if source == "live" else None)
        # A slow camera on a busy host (e.g. 8 frames/s on a laptop already
        # running a livecam server) takes minutes to fill the window.
        # A deliberate stop while it fills (the other half of the camera suite
        # starting, docs/suite.md) ends it at once: no generation has run yet,
        # so there is nothing new to save.
        fill_deadline = time.time() + 600.0
        while not feed.wait_for(600, timeout=2.0) and time.time() < fill_deadline:
            if sandbox.STOP_REQUEST_PATH.exists():
                sandbox.STOP_REQUEST_PATH.unlink(missing_ok=True)
                print("Stopped before its first generation -- nothing new to save.")
                feed.close()
                return
        if not feed.wait_for(600, timeout=0.5):
            print("Live feed never filled its window -- aborting.")
            feed.close()
            _stream_failed(dessert)
            return
        if dessert is not None:
            sandbox.source_opened()  # it delivered: its failure count starts over
        frames, vectors, seen_total, prey_boxes, colour_frames = feed.snapshot()
    else:
        print(f"Loading real frames from {clip_path!r} into memory (never written to disk)...")
        detector = prey_lib.PreyDetector()
        frames, prey_boxes, colour_frames = video_source.read_frames_with_prey(load_source, stride=2, max_frames=600,
                                                                 detector=detector if detector.available else None)
        print(f"  {len(frames)} frames loaded (clip {clip_index + 1}/{len(clips)}).")
        if len(frames) < 10:
            print("Not enough real frames to evolve against -- aborting.")
            return
        vectors = _world_vectors(frames)
        seen_total = len(frames)
    world = World(frames, vectors, feed.frames_per_second() if feed is not None else 15.0, prey_boxes, colour_frames)
    world.t_newest = feed.newest_time if feed is not None else None
    world.first_index = feed.snapshot_first if feed is not None else None

    def _fps() -> float:
        return world.fps

    # The real source frame's shape (the gaze box is drawn at its real aspect ratio).
    frame_h, frame_w = frames[0].shape[:2]

    box = sandbox.Sandbox(limits)
    if checkpoint is not None:
        box.generation = int(checkpoint.get("total_generation", 0))
        box.run_start_generation = box.generation
    margin = MARGIN_START

    # Plain inputs are only ever APPENDED (e.g. the brain's hidden units), so
    # a genome with fewer is prefix-compatible: every existing tree index
    # keeps its meaning. (Genome.from_dict migrates the older flat layout,
    # receptors included, to receptors read by position.)
    founded = False
    genome = G.Genome.from_dict(checkpoint["genome"]) if checkpoint is not None else None
    if genome is not None and 0 < genome.n_vars <= n_vars:
        print(f"Resuming from checkpoint (previous best_fitness={checkpoint['best_fitness']:.4f}; "
              f"eye {genome.receptors}x{genome.receptors} receptors).")
        genome.n_vars = n_vars
        genome.receptors = genome.receptors or fovea.DEFAULT_RECEPTORS
        margin = float(checkpoint.get("margin", margin))
    elif genome is not None:
        # Audit: this used to silently start a brand-new lineage and
        # overwrite the checkpoint. A checkpoint with MORE inputs than this
        # code expects means the code is older than the lineage -- stop.
        raise SystemExit(f"Checkpoint has n_vars={genome.n_vars} but this code expects {n_vars}; "
                         "refusing to overwrite the lineage with a fresh genome.")
    else:
        genome = G.random_genome(rng, n_vars=n_vars, receptors=fovea.DEFAULT_RECEPTORS)
        _default_brain(genome)
        hist = sandbox.life_history()
        if hist and hist[-1].get("event") == "extinct":  # its lineage died out: a migrant may come instead (in the hive)
            genome = _migrate(world, genome, n_vars)
        founded = True


    # NEVER trust a best_fitness carried over from a different world
    # snapshot or body state (external audit finding: this was
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
    space = sandbox.load_space(max(1, (os.cpu_count() or 2) - 1))  # worker bodies allowed (memory)
    # Energy is priced by the CPU share granted (capacity). The host's speed is
    # TIME, not price (a snail's neurons aren't dearer, its world is slower --
    # Healy et al. 2013): its brain's deadline (organism.py) uses this host's
    # time per multiply-add, the best measured this run (fishbowl/hostspeed.py).
    price_quota = quota_pct
    host_rate = hostspeed.sec_per_mac()
    print(f"Host speed: {host_rate:.2e} s per multiply-add ({hostspeed.speed_factor(host_rate):.2f}x the reference host).")
    # Its body persists across generations (and restarts): every window
    # starts from how it actually is now, and afterwards the lasting body
    # moves toward the survivor's end-of-window body in proportion to the
    # real time that passed -- so hours of stillness drain it at a real
    # rate, not 80 s of body-time per 1 s generation.
    body_now = (checkpoint or {}).get("body")
    body_clock = time.time()
    # Cryptobiosis (a 2026-09-30 panel; tardigrades, dried out): while its
    # process isn't running -- the machine off, its feed down -- it isn't
    # living, so no time passes for it: no burn, no ageing, no clock. (An
    # earlier audit charged downtime as idle, foodless time; once it could die
    # of starvation, an outage it never lived through could kill its lineage.)
    if body_now and checkpoint and checkpoint.get("saved_at"):
        away = max(0.0, time.time() - float(checkpoint["saved_at"]))
        print(f"Body: {away / 60:.1f} min since last save, in cryptobiosis (nothing charged; energy {body_now.get('energy', 0.0):.3f}).")
    # Its feeding record (gaps between feeding acts, over its life), from
    # which what counts as one meal / one snack is measured (fishbowl/bouts.py).
    feeding = {kind: FeedingRecord((checkpoint or {}).get("feeding_record", {}).get(kind)) for kind in ("meal", "snack")}
    bouts_cache: dict = {}

    def _bouts() -> dict:
        out = {}
        for kind, rec in feeding.items():
            key = (kind, len(rec.gaps), rec.gaps[-1] if rec.gaps else None)
            if key not in bouts_cache:  # refit only when the record changed
                if len(bouts_cache) > 8:
                    bouts_cache.clear()
                bouts_cache[key] = fit_bout_criterion(rec.gaps)
            out[kind] = {"fit": bouts_cache[key], "gaps": len(rec.gaps)}
        return out

    # Surprise memory persists too (NaN = never seen, stored as null).
    memory_now = memory_from_checkpoint(checkpoint)
    best_fitness, _, _ = evaluate_genome(genome, *world.at_pace(1), price_quota, body_now, _fps(), world.prey, memory_now, world.colour, host_rate)
    peak_fitness_seen = checkpoint.get("peak_fitness_seen", best_fitness) if checkpoint is not None else best_fitness
    peak_fitness_seen = max(peak_fitness_seen, best_fitness) if math.isfinite(best_fitness) else peak_fitness_seen
    print(f"Parent's real fitness on this run's frames: {best_fitness:.4f} (peak ever: {peak_fitness_seen:.4f})")

    # Its model card (a 2026-09-29 panel; Gelman): the champion with what it
    # learned, how well each teacher-free part agrees with its teacher out of
    # sample, and under which rules -- scores, not verdicts: each downstream
    # module sets its own bar. Out-of-sample fitness: the champion scored on
    # the snapshots after the one it won on.
    card = {"adopted_fitness": None, "adopted_world": None, "oos": [0, 0.0, 0.0]}  # n, mean, M2
    # Its development (a 2026-09-30 panel; Gelman): at each new world snapshot
    # its founder, as it was at birth (a newborn body, no memories), is scored
    # beside it on the same frames: the paired lead (it - its founder), n / mean
    # / M2. Developed = a lead beyond 1.96 standard errors, over 3 pairs at least
    # (as the ground fit's own minimum) -- only then may it encyst (state.py).
    card["dev"] = [float(v) for v in ((checkpoint or {}).get("development_leads") or [])]  # its leads over its founder, one a snapshot
    founder_genome = None
    if sandbox.FOUNDER_PATH.exists():
        try:
            import json
            founder_genome = G.Genome.from_dict(json.loads(sandbox.FOUNDER_PATH.read_text(encoding="utf-8"))["genome"])
            founder_genome.n_vars = n_vars
        except (OSError, ValueError, KeyError) as e:
            print(f"Its founder couldn't be read for its development test ({e}).")
    life = None  # its live body, once it starts (below)

    def _publish_champion() -> None:
        org = life.org if life is not None else None
        with (life.lock if life is not None else contextlib.nullcontext()):
            learned = _learned_parts(org)
            scores = {k: v.report() for k, v in org.scores.items()} if org is not None else {}
        n, mean, m2 = card["oos"]
        scores["perception_tree"] = None  # graded only in evolution's runs; its sleep test set chooses its edits: no held-out score yet
        sandbox.save_champion({"genome": genome.to_dict(), "bouts": _bouts(), "generation": box.generation,
                               "host": socket.gethostname(), "saved_at": time.time(),
                               "frames_per_second": round(world.fps, 2),
                               "rules_version": RULES_VERSION, "code_version": CODE_VERSION,
                               "learned": learned, "scores": scores,
                               "life_history": _life_card(life),
                               "development": _development_card(card.get("dev")),
                               "fitness": {"at_adoption": card["adopted_fitness"],
                                           "out_of_sample": None if n < 2 else {"n": n, "mean": round(mean, 4),
                                                                                "se": round(math.sqrt(m2 / (n - 1) / n), 4)}}})

    _publish_champion()

    saved_at = time.time()

    def _save():
        sandbox.save_checkpoint({
            "genome": genome.to_dict(),
            "best_fitness": best_fitness,
            "peak_fitness_seen": peak_fitness_seen,
            "margin": margin,
            "n_vars": n_vars,
            "clip_index": (clip_index + 1) % len(clips),
            "total_generation": box.generation,
            # Experiments on the lineage (tools/reset_mind.py, tools/stroke.py), kept across saves.
            **lineage_notes,
            "saved_at": time.time(),
            "body": body_now,
            "memory": {"mean": [[None if np.isnan(x) else round(float(x), 4) for x in row] for row in memory_now[0]],
                       "var": [[round(float(x), 6) for x in row] for row in memory_now[1]],
                       "learned": [round(float(x), 5) for x in memory_now[2]] if len(memory_now) > 2 else [],
                       **{k: np.round(memory_now[i], 5).tolist()
                          for i, k in ((3, "place"), (4, "people_day"), (5, "people_night"), (6, "danger"), (7, "value"))
                          if len(memory_now) > i and memory_now[i] is not None},
                       # the plants' standing crops, as it has lived them (item 9)
                       "nectar": {k: round(float(v), 4) for k, v in memory_now[8].items()} if len(memory_now) > 8 and memory_now[8] else {},
                       "proto": _pack_proto(memory_now[9]) if len(memory_now) > 9 else None,
                       "scenes": _pack_scenes(memory_now[10]) if len(memory_now) > 10 else None,
                       "ground": np.round(memory_now[11], 5).tolist() if len(memory_now) > 11 and memory_now[11] is not None else None,
                       "heads": np.round(memory_now[12], 5).tolist() if len(memory_now) > 12 and memory_now[12] is not None else None,
                       "terrain": np.round(memory_now[13], 5).tolist() if len(memory_now) > 13 and memory_now[13] is not None else None,
                       # its frames of reference: still and world votes per field cell (item 15)
                       "frames": np.round(memory_now[14], 3).tolist() if len(memory_now) > 14 and memory_now[14] is not None else None}
                      if memory_now is not None else None,
            "feeding_record": {kind: rec.gaps for kind, rec in feeding.items()},
            "development_leads": card.get("dev", []),
        })

    if founded:
        # A founder is saved the moment it is made: killed or crashed before
        # its first save, it used to be founded afresh at every restart (7elwe,
        # 2026-09-29: three founders in ten minutes), each with new random draws.
        _save()
        # and kept as it was at birth, for Reset to founder (with the seed sets its genome holds now)
        import shutil
        try:
            shutil.copy2(sandbox.CHECKPOINT_PATH, sandbox.FOUNDER_PATH)
            if (sandbox.STATE_DIR / "seeded.txt").exists():
                shutil.copy2(sandbox.STATE_DIR / "seeded.txt", sandbox.FOUNDER_SEEDED_PATH)
        except OSError as e:
            print(f"Its founder couldn't be kept for a reset ({e}).")
        print("Its founder is saved: a restart resumes it, and Reset to founder returns to it.")
        sandbox.record_life({"event": "founded", "kappa": round(genome.kappa, 4), "metabolism": round(genome.metabolism, 4)})
    # In the hive or solo, recorded as it changes, so hive and solo lineages can
    # be told apart later (the 2026-09-30 panel, Gelman).
    in_hive = sandbox.hive()
    if next((r["hive"] for r in reversed(sandbox.life_history()) if "hive" in r), None) != in_hive:
        sandbox.record_life({"event": "joined the hive" if in_hive else "solo", "hive": in_hive})
    print("In the hive: it may take a migrant from its peers after an extinction, and serves its own to them."
          if in_hive else "Solo (not in the hive): it neither takes migrants nor serves its organism to peers.")
    # What it eats here (prey.py: the defaults, or cambrian.json's "diet"),
    # recorded as it changes -- a lineage fed on teddy bears is a test.
    eats = prey_lib.diet()
    for problem in prey_lib.DIET_PROBLEMS:
        print(problem)
    print(f"Food (bites): {', '.join(eats['food'])}. Nectar (sips): {', '.join(eats['nectar'])}.")
    if next((r["diet"] for r in reversed(sandbox.life_history()) if "diet" in r), None) != eats:
        sandbox.record_life({"event": "diet", "diet": eats})

    # A stop request (systemctl restart/stop -> SIGTERM, e.g. every video
    # switch in the viewer) ends the loop cleanly so the checkpoint is
    # saved below -- it used to die mid-loop and lose up to 99 generations.
    stop = {"now": False}
    workers = None  # the worker pool, started when more than one child is scored (False = unavailable)
    world_prev = None  # the previous snapshot, for re-checking a winner
    world_time = time.time()  # when the current snapshot was taken
    gen_seconds, gen_started = 0.0, time.time()  # how long the last generation took
    _last_status = [0.0]
    last_new_frame = time.time()

    def _status_due() -> bool:
        now_s = time.time()
        if now_s - _last_status[0] < 1.0:
            return False
        _last_status[0] = now_s
        return True
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))

    metrics = HourlyMetrics()

    def _metrics_line() -> None:
        with (life.lock if life is not None else contextlib.nullcontext()):
            line = _metrics_flush()
        if line:
            sandbox.append_metrics(line)

    def _metrics_flush():
        return metrics.flush(world.fps, {
            "memory_mb": _memory_account(life.org if life is not None else None, feed),
            "generation": box.generation, "watching": clip_path, "rules_version": RULES_VERSION,
            "pump": round(genome.pump, 3), "metabolism": round(genome.metabolism, 3), "pace": genome.pace,
            "kc": genome.kc, "receptors": genome.receptors, "zoom": round(genome.zoom, 3),
            "vigilance": round(genome.vigilance, 3), "quota_pct": quota_pct,
            "replay": {"awake": genome.awake_replay, "sleep": genome.sleep_replay, "rem_share": round(genome.rem_share, 2),
                       "backup": round(genome.replay_backup, 3), "dream_steps": genome.dream_steps},
            "mobilize": round(genome.mobilize, 3), "store": round(genome.store, 3),
            "body": {k: round(float(body_now[k]), 3) for k in ("energy", "glycogen", "reserve", "ketone", "wasting")
                     if body_now and k in body_now},
        })

    # The organism acting live (fishbowl/livelife.py; a 2026-09-28 panel: act
    # live, learn offline). On a live feed, one living body lives every frame
    # as it arrives; its body and memory are the ones of record -- each
    # generation's children start from a copy -- and a winning child's genome
    # is adopted by it (a brain transplant). A recorded file has no live body:
    # there the survivor's window carries the body on, as before.
    life = None
    if feed is not None:
        life = LiveLife(feed, genome, body_now, memory_now, price_quota, host_rate, feed_epoch,
                        sandbox.LIVE_STATUS_PATH.with_name("live_actor.json"), metrics, _fps())
        life.torpor_after_s = STREAM_STALL_MIN_S  # torpor: no world for longer than its feed's own stall line
        life.developed = _developed(card["dev"])  # its development test, as saved
        if checkpoint and checkpoint.get("hatched") and checkpoint.get("upbringing") and not checkpoint.get("raised"):
            # a hatchling's upbringing: what its mother's body learned (an egg carries it), given once
            with life.lock:
                took = apply_learned(life.org, checkpoint["upbringing"])
            print(f"Its upbringing: {', '.join(took) if took else 'nothing that fits'} from its mother.")

    phase: dict = {}
    phase_worst: dict = {}
    while box.should_continue() and not stop["now"]:
        if sandbox.STOP_REQUEST_PATH.exists():  # a deliberate stop (docs/packaging.md)
            print("Stop requested -- saving and exiting.")
            sandbox.STOP_REQUEST_PATH.unlink(missing_ok=True)
            break
        if sandbox.RESET_FOUNDER_REQUEST_PATH.exists():
            # The owner's Reset to founder: saved as at any stop, then (main)
            # its state moves aside and its founder, as it was at birth, returns.
            sandbox.RESET_FOUNDER_REQUEST_PATH.unlink(missing_ok=True)
            if sandbox.FOUNDER_PATH.exists():
                print("Reset to founder requested -- saving, then its founder as it was at birth.")
                _AMNESIA["founder"] = True
                break
            print("Reset to founder: no founder was kept for this lineage (born before founders were kept).")
        if sandbox.AMNESIA_REQUEST_PATH.exists():
            # The owner's Amnesia: it saves as at any stop, then (main) its
            # state moves aside as tools/reset_founder.py moves it, and it exits
            # to be started again as a fresh install's founder.
            sandbox.AMNESIA_REQUEST_PATH.unlink(missing_ok=True)
            print("Amnesia requested -- saving, then a fresh install's founder.")
            _AMNESIA["now"] = True
            break
        if sandbox.RANDOMIZE_REQUEST_PATH.exists():
            # The owner's Randomize (the viewer's brain card): its brain drawn
            # afresh as a founder's is (random weights over every input); its
            # eye, body, memories and other traits stay. Its checkpoint is
            # backed up first, as a seeding is.
            sandbox.RANDOMIZE_REQUEST_PATH.unlink(missing_ok=True)
            import shutil
            backup = sandbox.STATE_DIR / f"backup-{time.strftime('%Y%m%d-%H%M%S')}-before-randomize"
            try:
                backup.mkdir(parents=True, exist_ok=True)
                if sandbox.CHECKPOINT_PATH.exists():
                    shutil.copy2(sandbox.CHECKPOINT_PATH, backup / "checkpoint.json")
            except OSError as e:
                print(f"Randomize: couldn't back up its checkpoint ({e}).")
            from fishbowl.controller import MosquitoBrain
            # on a copy: the live body still runs on this genome's objects until it adopts
            # (swapped in place, its old brain met the new one mid-look: the live body failed)
            genome = genome.clone()
            genome.brain = MosquitoBrain.random(random.Random(time.time_ns()))
            if life is not None:
                life.adopt(genome)
            print(f"Randomized: a founder's brain, drawn afresh (backup: {backup.name}).")
            sandbox.log_event({"t": round(time.time(), 1), "event": "randomized", "backup": backup.name})
        # Dessert over (deadline passed, or cleared in the viewer): stop
        # this run so systemd brings it back on its home camera.
        # A chosen video (live or recorded) that has ended: back to its camera.
        if feed is not None and feed.ended:
            print(f"The video ended -- back to {home_source}.")
            if source == "live":
                _stream_failed(dessert)  # a live stream's "end" can be a hiccup
            elif dessert is not None:
                sandbox.clear_selected_source(dessert["url"])  # a recording that ended has ended
            break
        # The video choice, checked every generation (a small file read): any
        # change -- a video chosen, cleared, or swapped for another -- stops
        # this run so its supervisor (systemd, or a plain restart loop where
        # the viewer cannot restart it) brings it back on the new choice.
        chosen = _dessert()
        if dessert is not None and chosen is None:
            print(f"Dessert over -- returning to {home_source}.")
            break
        if dessert is None and chosen is not None and _is_device(home_source):
            print("Dessert chosen -- restarting onto it.")
            break
        if dessert is not None and chosen.get("url") == dessert.get("url") and chosen.get("until") != dessert.get("until"):
            dessert = chosen  # only its deadline changed
        elif dessert is not None and chosen.get("url") != dessert.get("url"):
            # Another stream: switched in place (a 2026-09-29 panel) -- the new
            # feed fills beside the old one, then its body moves onto it.
            switched = None
            try:
                print(f"Another stream chosen: {chosen['url']} -- switching in place.")
                new_url = chosen["url"]
                new_epoch = int(time.time())
                new_feed = video_source.LiveFeed(_resolve_live_url(new_url), detector=detector if detector.available else None,
                                                 frames_dir=frames_dir if use_frames else None, epoch=new_epoch,
                                                 resolve=lambda u=new_url: _resolve_live_url(u))
                deadline = time.time() + WATCHDOG_MAIN_S
                while not new_feed.wait_for(600, timeout=2.0) and time.time() < deadline and not stop["now"]:
                    watchdog.tick()
                if new_feed.wait_for(600, timeout=0.5):
                    switched = new_feed
                else:
                    new_feed.close()
            except Exception as e:
                print(f"The new stream couldn't be opened ({e}).")
            if switched is None:
                print("Restarting onto it instead.")
                break
            old, feed, feed_epoch, clip_path, dessert = feed, switched, new_epoch, new_url, chosen
            if life is not None:
                life.switch_feed(feed, feed_epoch)
            watchdog.feed = feed
            old.close()
            world_time, last_new_frame, seen_total = 0.0, time.time(), -1
            sandbox.source_opened()
            print(f"Now watching {clip_path}.")
        box.generation += 1
        gen_seconds, gen_started = time.time() - gen_started, time.time()
        watchdog.tick(gen_seconds)
        if not watchdog.armed:
            watchdog.life, watchdog.feed, watchdog.armed = life, feed, True
        if box.generation % 50 == 0:
            quota_pct = sandbox.load_quota_pct(REFERENCE_QUOTA_PCT)
            price_quota = quota_pct
            host_rate = min(host_rate, hostspeed.sec_per_mac())  # contention can only slow a reading
            new_space = sandbox.load_space(space)
            if new_space < space and workers:  # memory is short: fewer worker bodies (the pool is rebuilt smaller)
                workers.close()
                workers = None
            space = new_space
        n_children = min(_n_children(quota_pct), space)
        children = []
        for _ in range(n_children):
            child = genome.clone()
            # Every generation now really tries a tree mutation -- the
        # previous 15%-of-generations branch that mutated weights
        # INSTEAD of a tree was a wasted generation twice over (see
        # Genome.update_mutation_weights's own docstring): weight-only
        # candidates can never pass the tree accept/reject gate, so
        # that branch's change was always discarded, and no tree
        # mutation was even attempted on that generation either.
            ch, ap = child.mutate_task(
                rng, max_nodes=limits.max_tree_nodes, max_depth=limits.max_tree_depth,
            )
            children.append((child, ch, ap))
        # Only a REAL ceiling hit is worth surfacing as a request --
        # "noop_inapplicable" (e.g. mutate_const picked on a tree with
        # no consts yet) is normal and expected on a small tree, not
        # something a bigger ceiling would fix at all (see genome.py's
        # mutate_task docstring for the real bug this used to be:
        # every noop got blamed on the size ceiling, which made a
        # healthy small tree look artificially stuck).
        ceiling_reason = "tree_size_or_depth" if any(ap == "noop_ceiling" for _, _, ap in children) else None
        box.note_ceiling(ceiling_reason)

        # Re-evaluate the PARENT fresh, right here, right now -- never
        # compare against a stored best_fitness that might be stale
        # (external audit finding B1, the actual root cause of the
        # plateau). The world, body and memory drift between
        # generations; scoring parent and candidate from the SAME
        # snapshot, body and memory keeps every decision honest.
        if life is not None and (life.torpid or life.encysted):
            # torpid (its eyes get no world) or encysted: evolution waits --
            # nothing is scored on blank frames, and a cyst doesn't change. (Here,
            # after the owner's requests and the stream's switching: a dormant
            # organism can still be reset, and a new stream is what wakes a cyst.)
            watchdog.tick()
            time.sleep(1.0)
            continue
        # Every WORLD_REFRESH_GENERATIONS generations (a few seconds):
        # rebuilding the world's signals costs ~0.65 s, which every
        # generation would nearly double generation time.
        if feed is not None and (box.generation % WORLD_REFRESH_GENERATIONS == 0
                                 or time.time() - world_time > max(WORLD_REFRESH_MAX_S, 3.0 * gen_seconds)):
            world_time = time.time()
            phase["world"] = -time.perf_counter()
            frames, vectors, total, prey_boxes, colour_frames = feed.snapshot()
            world_prev = world
            world = World(frames, vectors, feed.frames_per_second(), prey_boxes, colour_frames)
            world.t_newest = feed.newest_time
            world.first_index = feed.snapshot_first
            phase["world"] += time.perf_counter()
            if total != seen_total:
                last_new_frame = time.time()
            elif time.time() - last_new_frame > max(STALL_CADENCES * max(feed.longest_gap(), 1.0 / max(1.0, _fps())),
                                                    STREAM_STALL_MIN_S if source == "live" else 0.0):
                # Audit: a stalled feed used to be scored forever on the same
                # frozen frames. A dead stream goes back to the camera; a
                # stalled camera restarts the process (reopening the device).
                print(f"No new frames for {time.time() - last_new_frame:.0f} s -- "
                      + ("stream ended, back to the camera." if source == "live" else "camera stalled, restarting."))
                if source == "live":
                    _stream_failed(dessert)
                break
            seen_total = total
        # Short of oxygen, its body sheds evolution first (fishbowl/livelife.py
        # SHED): the next generation waits until it breathes again -- at most
        # as long as the last generation took, so a host that is always short
        # still evolves, at half speed or less.
        if life is not None:
            waited = time.time()
            while life.stage >= 1 and life.error is None and not stop["now"] and time.time() - waited < gen_seconds:
                time.sleep(0.1)
        if life is not None and life.died:
            # It died (state.MosquitoState.death): recorded, saved as at any
            # stop, then (main) buried and its newest egg hatched -- or, with
            # none, its lineage is extinct and a new random founder starts.
            t = life.tally
            print(f"It died of {life.died}, {t.lived_s / 86400.0:.2f} days old, having laid {int(t.total['eggs'])} eggs.")
            sandbox.record_life({"event": "died", "cause": life.died, "lived_s": round(t.lived_s, 1), "born": round(t.born, 1),
                                 "eggs": int(t.total["eggs"]), "kappa": round(genome.kappa, 4), "metabolism": round(genome.metabolism, 4),
                                 "generation": box.generation})
            sandbox.log_event({"t": round(time.time(), 1), "event": "died", "cause": life.died, "lived_s": round(t.lived_s, 1),
                               "eggs": int(t.total["eggs"])})
            _AMNESIA["died"] = life.died
            break
        if life is not None:
            if life.error:
                # A living body is reborn from its last snapshot (its body and
                # memory as of the last generation), not left dead: runs are
                # unbounded now, and "the rest of this run" meant hours without one.
                print(f"Its live body failed ({life.error}); reborn from its last snapshot.\n{life.error_trace}")
                sandbox.log_event({"t": round(time.time(), 1), "event": "live body failed", "error": life.error, "trace": life.error_trace[-2000:]})
                life = LiveLife(feed, genome, body_now, memory_now, price_quota, host_rate, feed_epoch,
                                sandbox.LIVE_STATUS_PATH.with_name("live_actor.json"), metrics, _fps()) if feed is not None else None
            else:
                if box.generation % 50 == 0:
                    life.set_prices(price_quota, host_rate)
                body_now, memory_now = life.snapshot()  # the children start from the living body and memory
        memory_eval = _for_evaluation(memory_now)  # the children are scored without the imagery prototypes
        # Where the organism's own process spends its generations (a 2026-09-29
        # check: 7elwe's body starved while this process worked): the worst of
        # each phase since the last report, printed when its body is a second
        # behind and every 25 generations.
        behind = life._latency if life is not None else 0.0
        for k, v in phase.items():
            phase_worst[k] = max(phase_worst.get(k, 0.0), v)
        phase = {}
        if phase_worst and (behind > 1.0 or box.generation % 25 == 0):
            print("phases, worst (s): " + ", ".join(f"{k} {v:.2f}" for k, v in phase_worst.items()) + f"; its body {behind:.1f} s behind")
            phase_worst = {}
        # Every genome -- the parent too -- is scored in a worker process, even
        # a pool of one (a host short of memory): the organism's own process
        # only lives and keeps the books, so evolution never takes its body's
        # time (a 2026-09-29 panel: 7elwe, short of memory, scored parent and
        # child in its body's process every generation and fell minutes behind).
        futures, parent_future, founder_future = None, None, None
        if workers is not False:
            try:
                if workers is None:
                    workers = _Workers(max(1, min((os.cpu_count() or 2) - 1, space)))
                t_pub = time.perf_counter()
                published = workers.publish(world)  # builds this world's signals (World.at_pace) here, once
                phase["publish"] = time.perf_counter() - t_pub
                if published:
                    parent_future = workers.submit(genome, price_quota, body_now, _fps(), memory_eval, host_rate)
                    if founder_genome is not None and card.get("dev_world") is not world:  # its founder, newborn, on this snapshot
                        card["dev_world"] = world
                        founder_future = workers.submit(founder_genome, price_quota, MosquitoState().to_dict(), _fps(), None, host_rate)
                    futures = [workers.submit(c, price_quota, body_now, _fps(), memory_eval, host_rate) for c, _, _ in children]
            except Exception as e:  # no worker processes on this host: serial from here on
                print(f"Parallel evaluation unavailable ({e}); continuing serially.")
                workers, futures, parent_future = False, None, None
        parent_fitness = None
        if parent_future is not None:
            try:
                parent_fitness, _, parent_info = parent_future.result(timeout=600)
            except Exception as e:  # a lost worker: score the parent here instead
                if not stop["now"]:
                    print(f"Worker failed ({type(e).__name__}); scoring the parent here.")
        if parent_fitness is None:
            parent_fitness, _, parent_info = evaluate_genome(genome, *world.at_pace(1), price_quota, body_now, _fps(), world.prey, memory_eval, world.colour, host_rate)
        if founder_future is not None:
            try:
                f_fit = founder_future.result(timeout=600)[0]
                if math.isfinite(f_fit) and math.isfinite(parent_fitness):
                    card["dev"].append(round(parent_fitness - f_fit, 5))
            except Exception as e:  # a lost worker: this snapshot gives no pair
                if not stop["now"]:
                    print(f"Its founder's score failed ({type(e).__name__}).")
            if life is not None:
                life.developed = _developed(card["dev"])
        if card["adopted_world"] is not None and world is not card["adopted_world"] and math.isfinite(parent_fitness):
            n, mean, m2 = card["oos"]  # the champion on a snapshot it didn't win on
            n += 1; d = parent_fitness - mean; mean += d / n; m2 += d * (parent_fitness - mean)
            card["oos"] = [n, mean, m2]
        if futures is not None:
            results = []
            for (c, _, _), fut in zip(children, futures):
                try:
                    results.append(fut.result(timeout=600))
                except Exception as e:  # a lost worker: score that child here instead
                    if not stop["now"]:
                        print(f"Worker failed ({type(e).__name__}); scoring the child here.")
                    results.append(evaluate_genome(c, *world.at_pace(1), price_quota, body_now, _fps(), world.prey, memory_eval, world.colour, host_rate))
        else:
            results = [evaluate_genome(c, *world.at_pace(1), price_quota, body_now, _fps(), world.prey, memory_eval, world.colour, host_rate)
                       for c, _, _ in children]
        best = max(range(len(children)), key=lambda k: results[k][0] if math.isfinite(results[k][0]) else -math.inf)
        candidate, channel, applied = children[best]
        candidate_fitness, breakdown, live_info = results[best]

        both_finite = math.isfinite(candidate_fitness) and math.isfinite(parent_fitness)
        accepted = both_finite and candidate_fitness > parent_fitness + margin
        if not accepted and both_finite and candidate_fitness >= parent_fitness - NEUTRAL_EPSILON:
            # Real neutral drift (external audit's recommended fix for
            # a pure greedy hill-climb that can never cross a flat
            # plateau -- equal-fitness moves used to always be
            # rejected). A small, fixed chance to take a roughly-tied
            # step sideways; never a worse one.
            # A newborn brain channel changes nothing yet (controller.py):
            # kept on a tie, so it can drift until it is useful.
            accepted = rng.random() < (1.0 if applied in G.NEUTRAL_GROWTH_OPS else NEUTRAL_ACCEPT_PROB)
        rechecked_out = False
        if accepted and both_finite and candidate_fitness > parent_fitness + margin and world_prev is not None:
            # The best of several children re-checked on the previous
            # snapshot: it must not be worse there (see RECHECK above).
            t_re = time.perf_counter()
            prev_meta = workers.meta_for(world_prev) if workers not in (None, False) else None
            p2 = c2 = None
            if prev_meta is not None:  # in workers, on the snapshot before (kept shared for this)
                try:
                    f_p = workers.submit(genome, price_quota, body_now, _fps(), memory_eval, host_rate, meta=prev_meta)
                    f_c = workers.submit(candidate, price_quota, body_now, _fps(), memory_eval, host_rate, meta=prev_meta)
                    p2, c2 = f_p.result(timeout=600)[0], f_c.result(timeout=600)[0]
                except Exception as e:  # a lost worker: re-checked here instead
                    if not stop["now"]:
                        print(f"Worker failed re-checking ({type(e).__name__}); re-checking here.")
                    p2 = c2 = None
            if p2 is None or c2 is None:
                p2, _, _ = evaluate_genome(genome, *world_prev.at_pace(1), price_quota, body_now, _fps(), world_prev.prey, memory_eval, world_prev.colour, host_rate)
                c2, _, _ = evaluate_genome(candidate, *world_prev.at_pace(1), price_quota, body_now, _fps(), world_prev.prey, memory_eval, world_prev.colour, host_rate)
            phase["recheck"] = time.perf_counter() - t_re
            if not (math.isfinite(c2) and math.isfinite(p2) and c2 >= p2 - NEUTRAL_EPSILON):
                accepted, rechecked_out = False, True

        if accepted:
            # Evolution just chose to pay for a bigger eye -- a real,
            # organism-derived demand for resources, surfaced as a request
            # resource_handler.py can weigh (with real fitness gain) to
            # grant more quota. Never granted by the organism itself.
            if applied == "mutate_fovea" and candidate.receptors > genome.receptors:
                box.log_request(
                    f"eye grew {genome.receptors}x{genome.receptors} -> {candidate.receptors}x{candidate.receptors} receptors "
                    f"at quota {quota_pct:.0f}%"
                )
            genome = candidate
            best_fitness = candidate_fitness
            margin = max(0.005, margin * 0.995)
            if life is not None:
                life.adopt(genome)  # the living body takes the new genome: a brain transplant
            card["adopted_fitness"], card["adopted_world"], card["oos"] = round(candidate_fitness, 4), world, [0, 0.0, 0.0]
            _publish_champion()
        elif both_finite:
            # Keep best_fitness in sync with reality even on a reject
            # -- it's the PARENT's own freshly-scored real fitness now,
            # never a stale ratchet (see B1 fix above).
            best_fitness = parent_fitness
        if math.isfinite(best_fitness):
            peak_fitness_seen = max(peak_fitness_seen, best_fitness)

        if life is not None:
            # Its body and memory are the living ones (the next generation
            # takes a fresh copy); its feeding record and the hourly metrics
            # come from what it lived, each frame once.
            for kind, acts in life.drain_acts().items():
                if acts:
                    feeding[kind].add(acts, life.last, _fps(), feed_epoch)
            if metrics.due():
                _metrics_line()
        else:
            # Advance the lasting body toward the survivor's end-of-window body,
            # by the fraction of the window's real duration that actually passed.
            end_body = (live_info if accepted else parent_info).get("body")
            survivor = live_info if accepted else parent_info
            survivor_memory = survivor.get("_memory")
            if survivor_memory is not None:
                memory_now = survivor_memory
            # Its feeding record: the survivor's acts on frames not lived before
            # (a live feed only -- a file is re-watched, not lived).
            metrics.add(survivor, world.first_index, world.fps)  # Gelman's hourly metrics (observation only)
            if metrics.due():
                _metrics_line()
            if world.first_index is not None:
                for kind in ("meal", "snack"):
                    feeding[kind].add([world.first_index + k for k in survivor.get(kind + "_acts", [])],
                                      world.first_index + len(world.frames) - 1, world.fps, feed_epoch)
            if end_body:
                now = time.time()
                fps_real = feed.frames_per_second() if feed is not None else 15.0
                window_s = len(world.frames) / max(1.0, fps_real)
                f = min(1.0, (now - body_clock) / max(1.0, window_s))
                body_clock = now
                start = body_now or MosquitoState().to_dict()
                body_now = {k: start.get(k, v) + (v - start.get(k, v)) * f for k, v in end_body.items()}
                # Asleep or awake is a state, not a quantity: take where it ended.
                for k in ("asleep", "sleep_clock"):
                    if k in end_body:
                        body_now[k] = end_body[k]

        # The real, continuous meta-mutation step (see
        # Genome.update_mutation_weights) -- applied to whichever
        # genome persists (the just-accepted candidate, or the
        # unchanged parent on a reject), using the real evidence from
        # THIS generation's real attempt. Never gated on fitness --
        # there's no fitness for a weights-only change to be gated on.
        # Credit an operator only for a REAL improvement (audit: counting
        # neutral-drift accepts as successes let edits to unused branches of
        # the tree win the operator race and starve the brain).
        # Every child's operator is credited on its own result.
        for (_, _, ap), (fit, _, _) in zip(children, results):
            genome.update_mutation_weights(ap, math.isfinite(fit) and math.isfinite(parent_fitness) and fit > parent_fitness + 1e-9)

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
            "clip": clip_path,
            # Structural growth telemetry, added 2026-09-24: shows when
            # complexity increases and whether it earns its structural
            # cost. channel/applied were already
            # computed above for this generation's real mutation
            # attempt; tree_stats reflects the CURRENT (persisting)
            # genome, same as live_status.json's own tree_stats.
            "channel": channel,
            "mutation_type": applied,
            # (1+lambda): how many children this generation, what each tried,
            # and whether the best failed its re-check on the previous snapshot.
            "children": n_children,
            "children_ops": [ap for _, _, ap in children],
            "rechecked_out": rechecked_out,
            "fitness_delta": (candidate_fitness - parent_fitness) if both_finite else None,
            "fovea_fraction": genome.fovea_fraction,
            "receptors": genome.receptors,
            "cones": genome.cones,
            "kc": genome.kc,
            "learning_rate": genome.learning_rate,
            "quota_pct": quota_pct,
            "host_speed": round(hostspeed.speed_factor(host_rate), 3),
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
        # At most once a second (audit: every generation was ~25 GB/day of
        # writes; it now lives in RAM too -- see sandbox.LIVE_STATUS_PATH).
        if _status_due(): sandbox.save_live_status({
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
            # Its gaze above is on the newest frame of its current snapshot;
            # this is how long ago that frame arrived (the viewer's lock HUD).
            "world_age_s": round(time.time() - world.t_newest, 1) if world.t_newest else None,
            # Where its frames are, for the viewer's replay of this run.
            "world_first_index": world.first_index,
            "world_epoch": feed_epoch,
            "fovea_fraction_accepted": round(genome.fovea_fraction, 4),
            # The eye of the run shown (its grid and path): a child's, when it differs.
            "receptors": live_info.get("receptors", genome.receptors),
            "cones": live_info.get("cones", genome.cones),
            "kc": genome.kc,
            "learning_rate": round(genome.learning_rate, 5),
            "food_value": live_info.get("food_value"),
            "quota_pct": quota_pct,
            "host_cores": os.cpu_count(),  # the quota is in cores (100% = one); the viewer shows it of these
            # Gemini's homeostasis: the candidate's body at the end of this
            # generation's run, its energy over time, how many frames the
            # flinch, and the gaze's real path
            # (cx, cy, aperture) -- so the viewer can replay movement
            # instead of showing one end-point per generation.
            "body": live_info.get("body"),
            "energy_series": live_info.get("energy_series"),
            "sleep_series": live_info.get("sleep_series"),
            "trajectory": live_info.get("trajectory"),
            "food_series": live_info.get("food_series"),
            "mean_food": live_info.get("mean_food"),
            "movement": live_info.get("movement"),
            "field_events": live_info.get("field_events"),
            "flinch": live_info.get("flinch"),
            # Prey (YOLO boxes per frame, coordinates only) and how much it
            # was eating at each frame -- the "is YOLO firing" view.
            "prey_boxes": live_info.get("prey_boxes"),
            "eating": live_info.get("eating"),
            "tree_guess": live_info.get("tree_guess"),
            "snacks": live_info.get("snacks"),
            "stroke": lineage_notes.get("stroke"),
            # What counts as one meal / snack, measured from its own feeding
            # gaps (fishbowl/bouts.py): null fit = still calibrating.
            "bouts": _bouts(),
            "teacher_label": live_info.get("teacher_label"),
            "asleep_frames": live_info.get("asleep"),
            "alarm_frames": live_info.get("alarm"),
            "replays": live_info.get("replays"),
            "intruder": live_info.get("intruder"),
            "swats": len(live_info.get("swat_acts") or []),
            "danger_value": live_info.get("danger_value"),
            "aversive_rate": genome.aversive_rate,
            "receptor_slowness": genome.receptor_slowness,
            "plant_sense": genome.plant_sense,
            "plant_boxes": live_info.get("plant_boxes"),
            "setpoints": {"mobilize": round(genome.mobilize, 3), "store": round(genome.store, 3)},
            "mean_prey": live_info.get("mean_prey"),
            "prey_series": live_info.get("prey_series"),
            "max_fraction": fovea.extent(fovea.MAX_RECEPTORS),
            "colour_grid": live_info.get("colour_grid"),
            "colour_channels": genome.colour_channels,
            "stabilizer": genome.stabilizer,
            "zoom_gain": genome.zoom,
            "metabolism": genome.metabolism,
            "host_pref": {prey_lib.PREY_CLASSES[c]: round(w, 2) for c, w in genome.host_pref.items()},
            "replay_traits": {"awake": genome.awake_replay, "sleep": genome.sleep_replay, "rem_share": round(genome.rem_share, 2),
                              "backup": round(genome.replay_backup, 3), "dream_steps": genome.dream_steps},
            "vigilance": genome.vigilance,
            "pump": genome.pump,
            "prey_sense": genome.prey_sense,
            # Its lasting body right now (persists across generations and
            # restarts), vs "body" = the candidate's at the end of its window.
            "body_now": {k: round(v, 4) for k, v in body_now.items()} if body_now else None,
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
            # Real source frame shape. Lets the viewer draw the box at the REAL aspect
            # ratio instead of a hardcoded one.
            "frame_w": frame_w,
            "frame_h": frame_h,
            "response": round(live_info["last_response"], 4),
            "clip": clip_path,
            # Human-readable label + whether it's a real live stream
            # right now, for the viewer.
            "clip_name": clip_name,
            "is_live": source == "live",
            "grid": [round(x, 4) for x in live_info["grid"]],
            "grid_shape": live_info["grid_shape"],
            # The WORLD's own whole-field grid (same reduction
            # already used for reflex grading, never transmitted
            # before) -- same no-raw-frames justification the fovea
            # grid above already has (already reduced far past
            # anything resembling real footage), just for the FULL
            # frame instead of the fovea's own crop. Lets the viewer
            # draw the fovea box on a REAL pixel dump instead of a
            # third-party video embed -- no YouTube dependency, no
            # embedding restrictions, no autoplay/play-state games.
            "world_grid": [round(x, 4) for x in world.vectors[-1]],
            "world_grid_shape": list(world.field_shape),
            # The CURRENT ACCEPTED genome's own tree structure (not
            # the just-tried candidate's, even on a rejected
            # generation) -- `genome` only ever changes on an accept,
            # so this is always "its real brain right now." Each
            # channel is capped at max_tree_nodes (limits.max_tree_nodes,
            # currently 1000), cheap enough to include every generation
            # rather than gating it.
            "trees": genome.to_dict()["trees"],
            # Real node_count()/depth() per channel plus the real
            # ceiling, alongside the tree itself. A small tree
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
                f"margin={margin:.4f}"
            )
        if box.generation % 100 == 0 or time.time() - saved_at >= CHECKPOINT_EVERY_S:
            _save()
            if time.time() - saved_at >= CHECKPOINT_EVERY_S:
                _publish_champion()  # its card, with the scores so far
            saved_at = time.time()

    if life is not None:
        life.stop()
        body_now, memory_now = life.snapshot()
        if life.error:
            print(f"Its live body had failed: {life.error}")
    _save()
    _metrics_line()  # the hour so far
    if workers:
        workers.close()
    if feed is not None:
        feed.close()
    print(f"Stopped after {box.generation} generations, {round(__import__('time').perf_counter() - box.start_time, 1)}s.")
    print(f"Final best_fitness: {best_fitness:.4f} -- checkpoint saved, next restart resumes from here.")


def _resident_mb() -> float | None:
    """This process's resident memory (MB), where the OS says."""
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/self/status", encoding="utf-8"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                           [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            c = _Counters(); c.cb = ctypes.sizeof(c)
            k32 = _kernel32()
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.WorkingSetSize / 1048576.0
        import resource  # macOS: the peak (ru_maxrss is bytes there)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    except (OSError, ValueError, AttributeError, ImportError):
        return None


def _memory_account(org, feed) -> dict:
    """What the organism holds, by owner, against what the process holds (a
    2026-09-29 panel; Knuth: a growing brain looks like a leak from outside --
    account memory by owner, and a leak is resident memory growing faster than
    this account). MB, hourly in the metrics."""
    def nbytes(x, depth=0):
        if isinstance(x, np.ndarray):
            return x.nbytes
        if depth < 3 and isinstance(x, (list, tuple)):
            return sum(nbytes(v, depth + 1) for v in x)
        if depth < 3 and isinstance(x, dict):
            return sum(nbytes(v, depth + 1) for v in x.values())
        return 0
    out = {"resident": _resident_mb()}
    if org is not None:
        mb = org.mb
        out["brain"] = nbytes([org.brain.weights_ih, org.brain.weights_hh, org.brain.weights_ho, getattr(org.brain, "layers", [])]) / 1048576.0
        out["mushroom_body"] = nbytes(list(vars(mb).values())) / 1048576.0
        out["episodes"] = nbytes([e[0] for e in org.episodes] + list(getattr(org, "test_set", []))) / 1048576.0
        out["maps"] = nbytes([org.place, org.people_day, org.people_night, org.value_map, org.memory, org.variance,
                              org.ground, org.terrain, org.scenes]) / 1048576.0
    if feed is not None:
        with feed._lock:
            out["frames"] = nbytes(list(feed._buf)) / 1048576.0
    return {k: (None if v is None else round(v, 1)) for k, v in out.items()}


def _fingerprint(paths) -> str:
    """A short fingerprint of source files (their bytes, in path order)."""
    import hashlib
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


_HERE = Path(__file__).resolve().parent
# The rules version (a 2026-09-29 panel; the rule-version tag voted earlier):
# the organism's physics is fishbowl/'s source -- fitness and scores are only
# comparable under the same rules. The code version adds the search loop.
RULES_VERSION = _fingerprint(list((_HERE / "fishbowl").glob("*.py")))
CODE_VERSION = _fingerprint(list((_HERE / "fishbowl").glob("*.py")) + [_HERE / "run_vision.py"])


def _sd_notify(message: str) -> None:
    """systemd's notify protocol (a datagram to $NOTIFY_SOCKET); nothing elsewhere."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr or not hasattr(socket, "AF_UNIX"):
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect("\0" + addr[1:] if addr.startswith("@") else addr)
            s.sendall(message.encode("utf-8"))
    except OSError:
        pass


class _Watchdog:
    """Its own watchdog thread: see WATCHDOG_MAIN_S. Armed once it lives (the
    first generation); until then it only tells systemd the process is up."""

    def __init__(self):
        self.progress, self.gen_seconds, self.armed = time.time(), 0.0, False
        self.life, self.feed = None, None
        self._total, self._total_at = None, time.time()
        interval = float(os.environ.get("WATCHDOG_USEC", 0)) / 2e6 or STREAM_STALL_MIN_S / 2
        threading.Thread(target=self._run, args=(interval,), name="Watchdog", daemon=True).start()

    def tick(self, gen_seconds: float | None = None) -> None:
        self.progress = time.time()
        if gen_seconds is not None:
            self.gen_seconds = gen_seconds

    def _verdict(self) -> str | None:
        now = time.time()
        if now - self.progress > WATCHDOG_MAIN_S + self.gen_seconds:
            return f"its main loop made no progress for {now - self.progress:.0f} s"
        life, feed = self.life, self.feed
        if life is not None and life.error is None and not getattr(life, "died", None) and feed is not None:
            if feed.total != self._total:
                self._total, self._total_at = feed.total, now
            arriving = now - self._total_at < STREAM_STALL_MIN_S
            if arriving and now - life.lived_at > STREAM_STALL_MIN_S:
                return f"its body lived no frame for {now - life.lived_at:.0f} s while frames arrived"
        return None

    def _run(self, interval: float) -> None:
        while True:
            time.sleep(interval)
            why = self._verdict() if self.armed else None
            if why is None:
                _sd_notify("WATCHDOG=1")
                continue
            print(f"Watchdog: {why} -- ending, so its supervisor starts it again.", flush=True)
            os._exit(3)


def _body_at_full_speed() -> None:
    """Windows 11 may run a below-normal, windowless process as background
    work -- on the efficiency cores, at a low clock (EcoQoS). Its body is real
    time: this process opts out of that throttling (its priority stays below
    normal, so the host's user still comes first); its workers don't opt out,
    so evolution stays on the efficiency cores."""
    if os.name != "nt":
        return
    try:
        import ctypes

        class _Throttle(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
        state = _Throttle(1, 0x1, 0x0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED: controlled, and off
        k32 = _kernel32()
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))  # ProcessPowerThrottling
    except (OSError, AttributeError):
        pass


MARGIN_START = 0.05  # a new lineage's acceptance margin (it shrinks as it runs)
_AMNESIA = {"now": False, "founder": False, "died": None}


def _t_sf(t: float, df: float) -> float:
    """Student's t upper tail, P(T > t): the regularized incomplete beta by its
    continued fraction (Press et al., Numerical Recipes, betacf) -- no scipy."""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
            de = d * c
            h *= de
            if abs(de - 1.0) < 1e-12:
                break
        return h

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x) if 0 < x < 1 else None
    if lbeta is None:
        ib = 0.0 if x <= 0 else 1.0
    elif x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(lbeta) * cf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(lbeta) * cf(b, a, 1.0 - x) / b
    tail = 0.5 * ib  # P(|T| > |t|) / 2
    return tail if t >= 0 else 1.0 - tail


def _developed(leads) -> bool:
    """Its lead over its founder, paired snapshot by snapshot: a one-sided
    paired t-test at the standard 5% (is it better?), its effective pairs
    corrected for successive snapshots overlapping (AR(1): n (1 - r) / (1 + r),
    as its prequential scores are), from 2 pairs (the fewest with a spread)."""
    x = np.asarray(leads or [], dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return False
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd <= 0.0:
        return mean > 0.0
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else 0.0
    r = min(0.99, max(-0.99, r))
    n_eff = min(float(n), max(2.0, n * (1.0 - r) / (1.0 + r)))
    t = mean / (sd / math.sqrt(n_eff))
    return _t_sf(t, n_eff - 1.0) < 0.05


def _development_card(leads) -> dict:
    x = [v for v in (leads or []) if math.isfinite(v)]
    return {"pairs": len(x), "lead_over_founder": round(float(np.mean(x)), 4) if x else None,
            "sd": round(float(np.std(x, ddof=1)), 4) if len(x) > 1 else None, "developed": _developed(x)}


def _life_card(life) -> dict:
    """Its life history for the model card: eggs this life, and this host's
    record of lives -- lifespans of the dead, deaths by cause, eggs per lifetime."""
    hist = sandbox.life_history()
    deaths = [r for r in hist if r.get("event") == "died"]
    return {"eggs_this_life": int(life.tally.total["eggs"]) if life is not None else None,
            "lifespans_days": [round(r.get("lived_s", 0.0) / 86400.0, 3) for r in deaths],
            "deaths_by_cause": {c: sum(1 for r in deaths if r.get("cause") == c) for c in ("starvation", "age")},
            "eggs_per_lifetime": [r.get("eggs", 0) for r in deaths],
            "hatched": sum(1 for r in hist if r.get("event") == "hatched"),
            "extinctions": sum(1 for r in hist if r.get("event") == "extinct")}


def _migrate(world, founder, n_vars: int):
    """After its lineage went extinct (a 2026-09-30 panel -- Wright's island
    model, Gelman): every organism the fleet can reach (tools/fleet.py's
    discovery; pulled, never pushed) and a new random founder, each scored as it
    would arrive -- its genome in a newborn body, no memories -- on this host's
    own first snapshot at the reference prices (the fleet's fair tournament).
    The winner comes as an egg (kappa stepped, as at any laying); if the random
    founder wins, or no peer is reachable, or anything fails, the founder stays.
    Solo (out of the hive, sandbox.hive()), the founder stays without asking."""
    if not sandbox.hive():
        sandbox.record_life({"event": "founded after extinction", "solo": True})
        return founder
    try:
        from tools import fleet
        from fishbowl import hostspeed
        host_rate = hostspeed.sec_per_mac()
        newborn = MosquitoState().to_dict()

        def as_it_would_arrive(g) -> float:
            return float(evaluate_genome(g, *world.at_pace(1), REFERENCE_QUOTA_PCT, newborn, world.fps, world.prey,
                                         None, world.colour, host_rate)[0])

        field = [("a new random founder", founder, as_it_would_arrive(founder))]
        for h in fleet.discover(None):
            if h.get("local"):
                continue
            if not h.get("hive", True):  # (discover already skips solo organisms; belt and braces)
                continue
            fleet.fetch(h)
            c = h.get("checkpoint")
            if not c or not c.get("genome"):
                continue
            why = fleet.vet_genome(c["genome"])  # data from another machine: held to this host's own limits first
            if why:
                print(f"Migration: {h['host']}'s genome refused -- {why}.")
                sandbox.record_life({"event": "migrant refused", "from": h["host"], "why": why})
                continue
            g = G.Genome.from_dict(c["genome"])
            g.n_vars = n_vars
            field.append((h["host"], g, as_it_would_arrive(g)))
        who, winner, best = max(field, key=lambda f: f[2] if math.isfinite(f[2]) else -math.inf)
        scores = {w: round(sc, 4) for w, _, sc in field}
        print("Migration tournament (as each would arrive): " + ", ".join(f"{w} {sc:.4f}" for w, sc in scores.items()))
        if winner is founder:
            sandbox.record_life({"event": "founded after extinction", "scores": scores})
            return founder
        egg = winner.laid_egg(random.Random(time.time_ns()))
        egg.n_vars = n_vars
        sandbox.record_life({"event": "migrated", "from": who, "kappa": round(egg.kappa, 4), "scores": scores})
        print(f"A migrant from {who} arrives as an egg (kappa {egg.kappa:.3f}).")
        return egg
    except Exception as e:  # never let a migration keep a lineage from starting
        print(f"No migration ({type(e).__name__}: {e}); a new random founder.")
        return founder


def _bury_and_hatch(cause: str) -> int:
    """After a death (its state already saved): the dead body's state moves
    aside into backup-<time>-died (tools/reset_founder.reset). Its newest egg
    hatches -- that genome, a fresh body (MosquitoState's own defaults), no
    memories; the lineage's founder and seed record go on with it, the other
    eggs stay with their mother. With no egg, the lineage is extinct and the
    next start makes a new random founder. Then it exits to be restarted."""
    import json
    import shutil
    from tools.reset_founder import reset
    backup, moved = reset(sandbox.STATE_DIR, "died")
    eggs = sorted((backup / "eggs").glob("*.json")) if (backup / "eggs").is_dir() else []
    if not eggs:
        sandbox.record_life({"event": "extinct", "cause": cause, "backup": backup.name})
        print(f"Its lineage is extinct (it laid no egg); {len(moved)} files moved to {backup.name}. A new random founder starts.")
        return sandbox.EXIT_RESTART_ME
    egg = json.loads(eggs[-1].read_text(encoding="utf-8"))
    mother = json.loads((backup / "checkpoint.json").read_text(encoding="utf-8")) if (backup / "checkpoint.json").exists() else {}
    for name in ("founder.json", "founder_seeded.txt", "seeded.txt"):  # the lineage's, not the body's
        if (backup / name).exists():
            shutil.copy2(backup / name, sandbox.STATE_DIR / name)
    sandbox.save_checkpoint({
        "genome": egg["genome"],
        "best_fitness": mother.get("best_fitness", 0.0),  # re-derived on its first frames, as at any start
        "peak_fitness_seen": mother.get("peak_fitness_seen", 0.0),
        "margin": mother.get("margin", MARGIN_START),
        "n_vars": mother.get("n_vars", TREE_PLAIN_INPUTS),
        "clip_index": 0, "total_generation": 0, "saved_at": time.time(),
        "body": MosquitoState().to_dict(),  # a newborn's body
        "memory": None, "feeding_record": {},
        "hatched": {"egg": eggs[-1].name, "laid_at": egg.get("laid_at"), "mother_died_of": cause, "mother_backup": backup.name},
        "upbringing": egg.get("upbringing"),  # its mother's learned parts, given at its first start (not saved after)
    })
    sandbox.record_life({"event": "hatched", "egg": eggs[-1].name, "kappa": egg["genome"].get("kappa"),
                         "mother_died_of": cause, "eggs_left_with_mother": len(eggs) - 1})
    print(f"Its newest egg hatches ({eggs[-1].name}; kappa {egg['genome'].get('kappa', 1.0):.3f}); "
          f"its mother's {len(moved)} files and {len(eggs) - 1} other eggs are in {backup.name}.")
    return sandbox.EXIT_RESTART_ME


def main() -> int:
    _body_at_full_speed()
    _split_body_and_workers()
    # Line-buffered, so the journal shows each line as it happens (not all
    # at once when the run exits).
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="'live' for real live streams (see LIVE_SOURCES), a media/ directory of clips, a single file, or a live device (e.g. /dev/video0)")
    parser.add_argument("--generations", type=int, default=0, help="0 (the default): no limit")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 (the default): no limit")
    args = parser.parse_args()

    limits = sandbox.Limits(max_generations=args.generations, max_wallclock_seconds=args.seconds)
    run(args.source, limits)
    if _AMNESIA["died"]:
        return _bury_and_hatch(_AMNESIA["died"])
    if _AMNESIA["founder"]:
        import shutil
        from tools.reset_founder import reset
        backup, moved = reset(sandbox.STATE_DIR, "before-reset-to-founder")
        for name, into in (("founder.json", ("checkpoint.json", "founder.json")),
                           ("founder_seeded.txt", ("seeded.txt", "founder_seeded.txt"))):
            if (backup / name).exists():
                for target in into:
                    shutil.copy2(backup / name, sandbox.STATE_DIR / target)
        print(f"Reset to founder: {len(moved)} files moved to {backup.name}; its founder, as it was at birth, starts again.")
        return sandbox.EXIT_RESTART_ME
    if _AMNESIA["now"]:
        from tools.reset_founder import reset
        backup, moved = reset(sandbox.STATE_DIR)
        print(f"Amnesia: {len(moved)} files moved to {backup.name}; starting again as a founder.")
        return sandbox.EXIT_RESTART_ME
    return 0


if __name__ == "__main__":
    sys.exit(main())
