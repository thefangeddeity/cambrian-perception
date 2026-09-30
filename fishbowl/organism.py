from __future__ import annotations

"""
One organism living through frames -- the single implementation both of
its uses run:
  - evolution (run_vision.evaluate_genome) drives it through a snapshot of
    recorded frames to score a genome;
  - a live host (a livecam's CV loop) drives it frame by frame, live, with
    the current champion genome.
So what a livecam runs is literally the code that evolved.

Organism.frame() is one frame of its life: the eye takes one physics step
into this frame (with the stabilizer, which needs this frame's shift); if
that completes a gaze interval, the body is paid and fed for it; and on a
gaze frame it looks and decides (brain, perception tree, sleep, tempo,
eating). finish() completes the last interval when a recording ends.

Also here, the world's fixed physics the organism lives by (so a host needs
only fishbowl/): surprise memory (snacks), the prices of gazing, thinking,
colour, loops, stabilizing and sensing prey, the prey sense itself, and the
whole-field helpers (where motion is, the frame's global shift).
"""

import copy
import math
import random

import cv2
import numpy as np

from . import fovea, prey as prey_lib
from .controller import (ARCHETYPE_INPUTS, DANGER_INPUT, GROUND_INPUTS, INTRUDER_INPUT, MISMATCH_INPUTS, PARALLAX_INPUTS, PLACE_INPUTS,
                         PLANT_INPUTS, RECALL_INPUTS, REFERENCE_MACS, UNCERTAINTY_INPUT, COLLICULUS_INPUTS, TERRAIN_INPUT,
                         NEARNESS_INPUT, FELT_NEARNESS_INPUT, CONTACT_INPUT, TURN_INPUT, TILT_INPUT, HEADING_INPUTS,
                         EGO_SPEED_INPUT, ACCELERATION_INPUT, PLACE_VALUE_INPUT, RIDING_INPUT, TEXTURE_INPUTS, STRANGENESS_INPUT)
from .entorhinal import Entorhinal
from . import v4
from .retina import field_shape
from .genome import RETINA_PLANES
from .mushroom import PROTO_SIDE, MushroomBody, macs as kc_macs
from .state import (FOOD_PER_LOOK, GUT_CAP, LEGACY_UNIT, LIGHT_SLOW_S, PREY_FOOD_PER_LOOK, SLEEP_SETTLE_S, TEMPO_SHARE,
                    WAKE_FLOOR, WAKE_LOOM, MosquitoState)
from . import reflexes

# --- Prices (the body pays per gaze, x CPU scarcity) -----------------------
# Anchored on published shares of an animal's resting metabolism (constants
# audit, docs/constants-audit.md) instead of hand-set numbers: at the
# reference pace -- a look every frame at 15 frames/s, calm, awake (a resting
# burn of WAKE_FLOOR + TEMPO_SHARE B/s) --
#   - a newborn eye (DEFAULT_RECEPTORS x DEFAULT_RECEPTORS) costs 8% of it:
#     the blowfly's photoreceptors' share of its resting metabolic rate
#     (Laughlin, de Ruyter van Steveninck & Anderson 1998);
#   - a newborn brain's thinking costs 5%: the middle of the central nervous
#     system's 2-8% of body metabolism across vertebrates (humans ~20%;
#     Mink, Blumenschine & Adams 1981).
# Everything priced is x scarcity: REFERENCE_QUOTA_PCT / the CPU share it is
# granted, in reference-host cores (run_vision measures this host's speed,
# fishbowl/hostspeed.py) -- a slow or busy host makes neurons dear, a fast
# idle one cheap.
REFERENCE_QUOTA_PCT = 150.0
REFERENCE_GAZES_PER_S = 15.0
RESTING_BURN = WAKE_FLOOR + TEMPO_SHARE  # B/s, awake and calm at the reference pace
EYE_SHARE, BRAIN_SHARE = 0.08, 0.05
_PER_GAZE = RESTING_BURN / REFERENCE_GAZES_PER_S / LEGACY_UNIT  # 100% of resting burn, per gaze, in legacy units
RECEPTOR_COST = EYE_SHARE * _PER_GAZE / fovea.DEFAULT_RECEPTORS ** 2  # per receptor, per gaze, x scarcity


def _receptor_cost(n: int, quota_pct: float) -> float:
    return RECEPTOR_COST * n * n * (REFERENCE_QUOTA_PCT / max(1.0, quota_pct))


# Per-look compute cost (the brain and tree running once): x CPU scarcity,
# and x the brain's real arithmetic relative to the original 16-unit brain
# (controller.think_factor) -- a grown brain pays for what it computes. With
# the waking burn scaling with tempo (state.py), this is what makes a fast
# pace of life expensive.
THINK_COST = BRAIN_SHARE * _PER_GAZE  # per gaze, x scarcity
# Colour is seen by cones only (fovea.py: a central patch of the gaze, its
# size inherited, genome.cones; the rest are rods -- grey, and cheaper for
# lacking the colour circuitry). Each colour-opponent channel a cone serves
# is one more signal, priced like a receptor's (Laughlin: the cost is per
# signalling channel): colour vision, and how much of the eye has it, only
# evolve if seeing colour pays for itself.
CONE_COST = RECEPTOR_COST  # per cone, per channel, per gaze, x scarcity
# A brain channel's loop (controller.py) costs energy in proportion to the
# weight on its way back in: a loop that does nothing is free, one that
# matters has to pay for itself (a synaptic cost, like any real circuit).
CHANNEL_COST = THINK_COST  # per unit of loop weight, per gaze, x scarcity (relative to thinking: chosen, flagged in the audit)
# Sleep consolidates its habituation memory: clearing sleep pressure tightens
# each spot's remembered variation toward the sensor-noise floor, so after a
# night real change stands out more sharply (and "boring" stays learned).
CONSOLIDATE_RATE = 3.0
# Image stabilization (genome.stabilizer, 0..1, evolved; after a design
# panel): a reflex that moves the gaze with the WHOLE frame's shift from one
# frame to the next -- camera shake -- the way an eye's optokinetic reflex
# holds the image still between deliberate movements. Global only (see
# global_shift), so it can never follow an object -- pursuing prey stays the
# brain's job. Priced like any receptor, x gain.
STABILIZER_COST = 0.5 * THINK_COST  # per gaze at gain 1, x scarcity (relative to thinking: chosen, flagged in the audit)
# Prey sense (genome.prey_sense; after a design panel): 1 = scent, prey
# somewhere in its whole field and how much, without where -- "go look";
# 2 = + a coarse direction from its gaze to the strongest prey (left/right,
# up/down, or none within 5% of centre). It still has to centre prey with
# its eyes to eat. Like a grown brain channel, a new sense changes nothing
# until the brain wires it up, and its price grows with that wiring
# (synaptic cost), so it is kept on a tie and spreads only if it pays.
PREY_SENSE_COST = THINK_COST  # per unit of weight on its inputs, per gaze, x scarcity (relative to thinking: chosen, flagged)

# --- Tempo --------------------------------------------------------------------
TEMPO_RANGE = 3.0   # brain can speed up / slow down its gazing up to 3x around its resting pace
MAX_INTERVAL = 12   # slowest: one gaze every 12 frames

# --- Senses' gains --------------------------------------------------------------
# Scale of the look's own motion/flow readings into the [0, 1] range the
# body's and reflexes' thresholds assume (calibrated on a real camera and
# synthetic looming frames).
MOTION_GAIN = 10.0
FLOW_GAIN = 20.0
# The WHOLE visual field (the fixed camera's coarse view, retina.field_shape -- a jumping
# spider's wide-field secondary eyes, or a locust's LGMD/DCMD looming
# neurons) is what detects threat and where something moved; the look (the
# spider's movable principal retinae) is for detail and food.
PERIPH_MOTION_GAIN = 300.0   # whole-field motion_energy: still room ~0.001 -> ~0.3, real movement saturates
EXPANSION_GAIN = 10.0        # reflexes.expansion_score: approaching disc 0.083 -> 0.83, crossing blob 0.014 -> 0.14
# Field mismatch (Sokolov's orienting reflex; 2026-09-28 panel): how much of
# the field's structure differs from its slow model of the room
# (reflexes.mismatch_step) -- someone arriving, and staying, until it
# habituates to them. No new constants: the model (each cell's mean and
# usual variation) adapts over its sense of light's slow timescale (~20 min,
# state.LIGHT_SLOW_S); a cell mismatches beyond SURPRISE_SIGMAS x its own
# variation, floored at NOISE_FLOOR (the snack memory's surprise rule), so
# waving leaves and a codec's shimmer become expected; the area is read with
# the looming detector's gain (both are areas of the field that differ from
# what it expects), so it wakes at the looming line (state.WAKE_LOOM) x its
# vigilance. The brain gets it, and where it is, as a sense (priced).
MISMATCH_TAU_S = LIGHT_SLOW_S

# --- Snacks: surprise against a memory of the world ------------------------------
# Food = genuinely new visual structure, measured against a spatial memory
# of what the look has already seen at each WORLD location. Panning across
# a static scene is only news the first time; after that, only real change
# in the world feeds it -- so moving the eye can't manufacture food (the
# self-stimulation loophole an audit found, kept closed). A bigger look
# takes in more at once, which is what pays for its aperture cost.
MEM_H, MEM_W = 24, 32
UNSEEN_NOVELTY = 0.25
FOOD_GAIN = 400.0
# Tuned on synthetic scenes (noise / swinging fan / new object / something
# crossing): at 4 sigmas with these rates, noise feeds 0, a fan 0.20 while
# new and 0 once learned, a new object ~0.2 when it appears, something
# crossing to new places ~0.10 steadily.
SURPRISE_SIGMAS = 4.0      # change beyond ~4x a spot's usual variation counts as surprise
NOISE_FLOOR = 0.02         # smallest variation any spot is assumed to have (sensor noise)
MEAN_RATE, VAR_RATE = 0.1, 0.05

# --- Place map, replay and the intruder sense (2026-09-27 design panels) ---------
# Place map: over the whole field's grid, what each spot has fed it (learned
# with its mushroom body's learning rate, the same three-factor rule), read
# by the brain as the direction to the best spot and how good it was --
# where food recurs (bees learning a route, Lihoreau 2012). Priced by the
# weight on those inputs, like the prey sense.
# Replay (after panels on sleep: Foster & Wilson, Mattar & Daw, Tononi &
# Cirelli, Diekelmann & Born, Hoel): each waking look's (Kenyon-cell code,
# what it ate, where) is kept for this life. It re-learns from them --
#   awake, in quiet moments (nothing in view, no big change: a change cuts
#     it short), genome.awake_replay per look;
#   asleep (settled), genome.sleep_replay per look: NREM replays the biggest
#     surprises first (prioritized, Mattar & Daw), and the place map is
#     scaled back down as sleep clears pressure (synaptic homeostasis --
#     the same consolidation rate as the habituation memory); REM
#     (genome.rem_share of them) replays recombined halves of two
#     experiences at their mean reward (Hoel's corrupted replay: generalize).
# A replay costs what re-activating its Kenyon cells costs: thinking energy,
# and time against the look's deadline -- heavy replay misses looks. All
# replay starts off (0).
# Intruder: a person at a spot where, at this time of day, people haven't
# been over the last ~20 minutes (its day/night sense's slow average) -- its
# own surprise; the regulars' spots become expected. One of the brain's
# inputs; warning of it is its alarm output (nothing rewards it yet: the
# owner's feedback is to breed that).

# --- The frame's global shift (for the stabilizer) -------------------------------
SHIFT_WIDTH = 160
SHIFT_MIN_RESPONSE = 0.2
SHIFT_MAX = 0.08  # of the frame, per frame
LK_HALF_WINDOW = 10  # px: half of cv2.calcOpticalFlowPyrLK's default 21 px window, which every tracker here uses


def new_memory() -> tuple[np.ndarray, np.ndarray]:
    """A blank surprise memory: nothing seen yet, sensor-noise variation."""
    return np.full((MEM_H, MEM_W), np.nan), np.full((MEM_H, MEM_W), NOISE_FLOOR ** 2)


def feed_on_novelty(memory: np.ndarray, look: np.ndarray, st: fovea.FoveaState, variance: np.ndarray | None = None,
                    aspect: float = 16 / 9) -> float:
    """
    Food = SURPRISE at each spot the gaze covers: how far what it sees
    now is beyond that spot's usual variation (a running estimate of
    both its brightness and how much it varies) -- habituation. Sensor
    noise never feeds it; a swinging fan feeds it only until its swing
    becomes expected; something new in a still corner is a big meal.
    Spots never seen before count as UNSEEN_NOVELTY. Plain running
    statistics per spot, no learning model needed.
    """
    # The gaze may hang past the frame's edge (fovea.py); only its on-frame
    # part is remembered, each memory cell mapped to its own gaze cell.
    # aspect = the frame's width / height (the gaze is square in pixels).
    hx, hy = st.half_extents(aspect)
    gx0 = int(round((st.cx - hx) * MEM_W)); gx1 = max(gx0 + 1, int(round((st.cx + hx) * MEM_W)))
    gy0 = int(round((st.cy - hy) * MEM_H)); gy1 = max(gy0 + 1, int(round((st.cy + hy) * MEM_H)))
    x0, y0, x1, y1 = max(0, gx0), max(0, gy0), min(MEM_W, gx1), min(MEM_H, gy1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    n = st.n
    grid = look.reshape(n, n)
    rows = np.minimum(n - 1, (np.arange(y0, y1) - gy0) * n // (gy1 - gy0))
    cols = np.minimum(n - 1, (np.arange(x0, x1) - gx0) * n // (gx1 - gx0))
    patch = grid[np.ix_(rows, cols)]
    region = memory[y0:y1, x0:x1]
    seen = ~np.isnan(region)
    mean = np.nan_to_num(region)
    if variance is None:
        variance = np.full(memory.shape, NOISE_FLOOR ** 2)
    var = variance[y0:y1, x0:x1]
    sigma = np.sqrt(np.maximum(var, NOISE_FLOOR ** 2))
    dev = patch - mean
    surprise = np.where(seen, np.maximum(0.0, np.abs(dev) - SURPRISE_SIGMAS * sigma), UNSEEN_NOVELTY)
    # Weighted toward the gaze CENTER (its central half counts fully, the
    # rim a quarter): what it is actually looking at is what feeds it.
    ry = (np.arange(y0, y1) - gy0 + 0.5) / (gy1 - gy0) - 0.5
    rx = (np.arange(x0, x1) - gx0 + 0.5) / (gx1 - gx0) - 0.5
    center = (np.abs(ry)[:, None] <= 0.25) & (np.abs(rx)[None, :] <= 0.25)
    surprise = surprise * np.where(center, 1.0, 0.25)
    memory[y0:y1, x0:x1] = np.where(seen, mean + MEAN_RATE * dev, patch)
    variance[y0:y1, x0:x1] = np.where(seen, var + VAR_RATE * (dev * dev - var), NOISE_FLOOR ** 2)
    return float(min(1.0, surprise.sum() / (MEM_H * MEM_W) * FOOD_GAIN))


def prey_sense(boxes: list, cx: float, cy: float, level: int, host_pref: dict | None = None) -> tuple[float, float, float]:
    """(scent, direction x, direction y) for its prey-sense level, each prey
    weighted by its host preference for that class (genome.host_pref)."""
    if level <= 0 or not boxes:
        return 0.0, 0.0, 0.0
    w = (lambda c: host_pref.get(int(c), 1.0)) if host_pref else (lambda c: 1.0)
    scent = min(1.0, sum(w(c) * conf * min(1.0, (x1 - x0) * (y1 - y0) / 0.02) for c, conf, x0, y0, x1, y1 in boxes))
    if level < 2:
        return scent, 0.0, 0.0
    best = max(boxes, key=lambda b: w(b[0]) * b[1] * (b[4] - b[2]) * (b[5] - b[3]))
    coarse = lambda v: 0.0 if abs(v) < 0.05 else (1.0 if v > 0 else -1.0)
    return scent, coarse((best[2] + best[4]) / 2 - cx), coarse((best[3] + best[5]) / 2 - cy)


def shift_size(shape: tuple) -> tuple[int, int]:
    h, w = shape[:2]
    return SHIFT_WIDTH, max(8, int(round(h * SHIFT_WIDTH / w)))


def global_shift(prev_small: np.ndarray, cur_small: np.ndarray, window: np.ndarray) -> tuple[float, float]:
    """The whole frame's shift between two small float32 frames (fractions of
    width and height): phase correlation; only a small, confident shift
    counts -- a cut or a big moving object gives none."""
    (dx, dy), response = cv2.phaseCorrelate(prev_small, cur_small, window)
    fx, fy = dx / prev_small.shape[1], dy / prev_small.shape[0]
    if response >= SHIFT_MIN_RESPONSE and abs(fx) <= SHIFT_MAX and abs(fy) <= SHIFT_MAX:
        return fx, fy
    return 0.0, 0.0


def _similarity(a: np.ndarray, b: np.ndarray):
    """One similarity transform fitted to tracked points with RANSAC: (log scale,
    its standard error, the inliers, the transform), or None. The error comes
    from the inliers' residuals and their spread about their centre."""
    m, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if m is None or inl is None or inl.sum() < 8:
        return None
    inl = inl.ravel() == 1
    A, B = a[inl], b[inl]
    e = B - (A @ m[:, :2].T + m[:, 2])
    sigma2 = float((e ** 2).sum()) / max(1, 2 * len(A) - 4)
    spread = float(((A - A.mean(axis=0)) ** 2).sum())
    k = max(1e-6, math.hypot(m[0, 0], m[1, 0]))
    return float(np.log(k)), math.sqrt(sigma2 / max(spread, 1e-9)) / k, inl, m


def global_scale(prev_small: np.ndarray, cur_small: np.ndarray, keep: list | None = None, roll: list | None = None,
                 votes: list | None = None) -> float:
    """The whole frame's expansion between two small frames (log scale; + when
    the camera walks forward -- Gibson's outflow of forward locomotion): corner
    features tracked from one to the other (Lucas-Kanade), a similarity
    transform fitted to them with RANSAC (things moving on their own are
    outliers). Its own body in view (a 2026-09-29 panel; Gibson's nose, Neisser's
    ecological self): when most of what it tracks doesn't move at all -- a tram's
    cab, a car's bonnet -- that still part travels with the camera, and the
    world's motion is what the rest agree on. An expansion counts when it is
    beyond twice its standard error (the fit's own precision), and within
    SHIFT_MAX, as the shift is.

    votes (its frames of reference; a 2026-09-29 panel -- Galileo's ship,
    Jeffery, Burgess): each tracked corner where the world's fitted motion
    should have moved it at least a pixel says which frame it belongs to --
    it stayed put (its local frame: the cab rides with it), or it moved as the
    world did (within RANSAC's own pixel). What moved some other way (moving on
    its own) says nothing. Only when most of what it tracks flowed: feeling
    itself move takes wide-field flow (vection; Brandt, Dichgans & Koenig 1973);
    when most stood still it may be stopped among things moving on their own,
    or a big cab may fill its view -- it can't tell, so nothing votes. Appended
    as (x, y, +1 local / -1 world, x then, y then), frame fractions."""
    a, b = prev_small.astype(np.uint8), cur_small.astype(np.uint8)
    if votes is not None:
        votes.append(None)
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=100, qualityLevel=0.01, minDistance=5)
    if p0 is None or len(p0) < 8:
        if roll is not None:
            roll.append(0.0)
        return 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        if roll is not None:
            roll.append(0.0)
        return 0.0
    pa, pb = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
    if keep is not None:  # the tracked corners, as frame fractions (x0, y0, x1, y1), for tau at its gaze
        h, w = prev_small.shape[:2]
        keep.append((np.hstack([pa, pb]) / np.array([w, h, w, h], dtype=np.float32)).astype(np.float32))
    fit = _similarity(pa, pb)
    if fit is None:
        if roll is not None:
            roll.append(0.0)
        return 0.0
    s, se, inl, m = fit
    still = float(np.median(np.hypot(*(pb[inl] - pa[inl]).T))) < 1.0  # its inliers moved less than RANSAC's own pixel
    if still and (~inl).sum() >= 8:  # the still part is its body; the world is what the rest agree on
        world = _similarity(pa[~inl], pb[~inl])
        if world is not None:
            s, se, m = world[0], world[1], world[3]
    if votes is not None and not still:
        expect = pa @ m[:, :2].T + m[:, 2]
        should = np.hypot(*(expect - pa).T) >= 1.0
        moved = np.hypot(*(pb - pa).T)
        off = np.hypot(*(pb - expect).T)
        v = np.where(should & (moved < 1.0) & (off >= 1.0), 1.0, np.where(should & (off < 1.0) & (moved >= 1.0), -1.0, 0.0))
        h, w = prev_small.shape[:2]
        votes[-1] = np.column_stack([pa[:, 0] / w, pa[:, 1] / h, v, pb[:, 0] / w, pb[:, 1] / h])[v != 0.0].astype(np.float32)
    if roll is not None:
        # its roll: the camera's rotation about its view (+ clockwise), the
        # opposite of the world's on screen; for a similarity its error equals
        # the scale's, so the same test applies
        r = -math.atan2(m[1, 0], m[0, 0])
        roll.append(r if abs(r) > 2.0 * se and abs(r) <= SHIFT_MAX else 0.0)
    return s if abs(s) > 2.0 * se and abs(s) <= SHIFT_MAX else 0.0


def replicated(s: float, previous: float) -> float:
    """An expansion counts when the frame before showed one too, the same way
    (replication: real motion persists, a fit's noise flips; the fits' errors
    are correlated -- neighbouring corners share tracking windows -- so one
    significant frame alone isn't enough)."""
    return s if s and previous and (s > 0) == (previous > 0) else 0.0


def parallax_map(prev_small, small, shift, shape) -> np.ndarray:
    """Parallax (a 2026-09-29 panel; Land, Friston: depth from a moving
    camera): with the camera's own motion (the frame's global shift) undone,
    what still moved, per whole-field cell -- near things slide more than far
    ones. Zero when the camera doesn't move. The camera's motion is its
    shift and, when it walks forward or back, its expansion (shift[2], log
    scale about the frame's centre): both are undone."""
    rows, cols = shape
    if prev_small is None or small is None or prev_small.shape != small.shape or not camera_moves(shift, small.shape):
        return np.zeros(rows * cols)
    h, w = small.shape
    k = math.exp(shift[2]) if len(shift) > 2 else 1.0
    th = -shift[3] if len(shift) > 3 else 0.0  # its roll undone too (the fit's own angle)
    a, b = k * math.cos(th), k * math.sin(th)
    cx, cy = w / 2, h / 2
    moved = cv2.warpAffine(prev_small, np.float32([[a, -b, cx + shift[0] * w - (a * cx - b * cy)], [b, a, cy + shift[1] * h - (b * cx + a * cy)]]),
                           (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Cell by cell (as its motion sense compares receptors), how much of the
    # change is left once the camera's own motion is undone: what still moved,
    # over what moved at all (+ sensor noise) -- 0 for the scene sliding past as
    # a whole, towards 1 for what is nearer, or moving on its own. No gain.
    cell = lambda img: cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA) / 255.0  # noqa: E731
    now, was, before = cell(small), cell(moved), cell(prev_small)
    return np.clip(np.abs(now - was) / (np.abs(now - before) + NOISE_FLOOR), 0.0, 1.0).ravel()


def camera_moves(shift, small_shape) -> bool:
    """The camera itself moved: a whole-frame shift of at least one pixel of the
    small frame it is measured on (below that it's the measurement's noise), or
    a significant expansion (walking forward: the image flows out of its
    centre, with no shift at all)."""
    h, w = small_shape[:2]
    if abs(shift[0]) * w >= 1.0 or abs(shift[1]) * h >= 1.0:
        return True
    return (len(shift) > 2 and shift[2] != 0.0) or (len(shift) > 3 and shift[3] != 0.0)  # expansion or roll: only reported when significant


def parallax_series(frames: list[np.ndarray], sx: np.ndarray, sy: np.ndarray, shape, ss: np.ndarray | None = None) -> np.ndarray:
    """parallax_map over a recording (the batch twin of field.FieldSignals)."""
    out = np.zeros((len(frames), shape[0] * shape[1]))
    if len(frames) < 2:
        return out
    size = shift_size(frames[0].shape)
    prev = cv2.resize(frames[0], size, interpolation=cv2.INTER_AREA).astype(np.float32)
    for k in range(1, len(frames)):
        cur = cv2.resize(frames[k], size, interpolation=cv2.INTER_AREA).astype(np.float32)
        out[k] = parallax_map(prev, cur, (sx[k], sy[k], 0.0 if ss is None else ss[k]), shape)
        prev = cur
    return out


def global_shifts(frames: list[np.ndarray], tracks: list | None = None, rolls: np.ndarray | None = None,
                  votes: list | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole frame's shift (and expansion) from frame k-1 to frame k, for a recording."""
    n = len(frames)
    sx, sy, ss = np.zeros(n), np.zeros(n), np.zeros(n)
    if n < 2:
        return sx, sy, ss
    size = shift_size(frames[0].shape)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    prev = cv2.resize(frames[0], size, interpolation=cv2.INTER_AREA).astype(np.float32)
    if tracks is not None:
        tracks.append(None)
    if votes is not None:
        votes.append(None)
    for k in range(1, n):
        cur = cv2.resize(frames[k], size, interpolation=cv2.INTER_AREA).astype(np.float32)
        sx[k], sy[k] = global_shift(prev.copy(), cur.copy(), window)  # copies: phase correlation windows its inputs in place
        keep = [] if tracks is not None else None
        rl = [] if rolls is not None else None
        vt = [] if votes is not None else None
        ss[k] = global_scale(prev, cur, keep, rl, vt)
        if votes is not None:
            votes.append(vt[0] if vt else None)
        if rolls is not None:
            rolls[k] = rl[0] if rl else 0.0
        if tracks is not None:
            tracks.append(keep[0] if keep else None)
        prev = cur
    raw = ss.copy()
    for k in range(1, n):
        ss[k] = replicated(raw[k], raw[k - 1])
    if rolls is not None:
        rraw = rolls.copy()
        for k in range(1, n):
            rolls[k] = replicated(rraw[k], rraw[k - 1])
    return sx, sy, ss


def peripheral_motion_centroid(world_vectors: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Where in the whole field (receptors shape = (rows, cols)) things
    changed, per frame (0.5, 0.5 when nothing did)."""
    n = len(world_vectors)
    cx, cy = np.full(n, 0.5), np.full(n, 0.5)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    for t in range(1, n):
        d = np.abs(world_vectors[t] - world_vectors[t - 1]).reshape(shape)
        tot = d.sum()
        if tot > 1e-9:
            cx[t] = ((xx + 0.5) * d).sum() / tot / cols
            cy[t] = ((yy + 0.5) * d).sum() / tot / rows
    return cx, cy


# A plant's nectar (a 2026-09-28 panel): a full plant holds about one gut-full
# (a flower's standing crop is about one mosquito sugar meal), and refills over
# hours once drunk (nectar secretion: assumption, 3 h -- flagged in the audit).
NECTAR_CROP = 1.0         # a full plant, in gut-fulls
NECTAR_REFILL_S = 3 * 3600.0

# What a host passes each frame about the whole field (sig): the keys below.
SIGNAL_KEYS = ("expansion", "motion_energy", "motion_cx", "motion_cy", "field_light", "mismatch", "mismatch_cx", "mismatch_cy", "structure",
               "parallax", "motion_map", "mismatch_map")


# Its collicular priority map (a 2026-09-29 panel; the superior colliculus --
# Land, Nilsson, Tooby & Cosmides, Friston): over the whole field's cells, a
# weighted sum of these features; the strongest cell pulls its gaze. The
# weights are inherited (genome.colliculus); host size's weight is the cat's
# size gate, evolvable either way (a cat wants small and fast, a mosquito big).
COLLICULAR_FEATURES = ("motion", "mismatch", "parallax", "host", "host size", "plant")
# The classes whose sizes measure its ground plane: its hosts, and its plants
# (rooted: a plant's base is where the ground is). Plants vary in size far
# more than a host class does, so each class's horizon counts by how well its
# own line fits (inverse variance, horizon() below): a class that fits badly
# -- flowers up on a bush, pots on a sill -- counts for little, by evidence.
GROUND_CLASSES = (sorted(prey_lib.PREY_CLASSES) + [prey_lib.PLANT_CLASS]
                  + [c for c in range(prey_lib.COCO_CLASSES) if c not in prey_lib.PREY_CLASSES and c != prey_lib.PLANT_CLASS])
# (older memories' rows keep their places: hosts, then plants, then the rest)
MATURATION_KEEP = 3650  # nights of lessons kept at most (maturation's safety cap)
GROUND_COLS = 7  # n, sum y, sum h, sum y^2, sum y*h; then m, sum of squared residuals (against the fit then)


class _Running:
    """A running mean (Welford)."""

    def __init__(self):
        self.n, self.mean = 0, 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n


class _Prequential:
    """Predictions scored before they are learned from: mean absolute error, its
    standard error corrected for autocorrelation (AR(1): n_eff = n (1 - r) / (1 + r)),
    and the prediction-target correlation."""

    def __init__(self):
        self.n = 0
        self.sa = self.saa = 0.0           # absolute errors
        self.sp = self.st = self.spp = self.stt = self.spt = 0.0
        self.last_e = None
        self.se1 = self.se0 = 0.0          # lag-1 products and squares of centred errors (running)

    def add(self, pred: float, target: float) -> None:
        e = abs(pred - target)
        self.n += 1
        self.sa += e; self.saa += e * e
        self.sp += pred; self.st += target; self.spp += pred * pred; self.stt += target * target; self.spt += pred * target
        if self.last_e is not None:
            m = self.sa / self.n
            self.se1 += (e - m) * (self.last_e - m)
            self.se0 += (e - m) ** 2
        self.last_e = e

    def report(self) -> dict | None:
        if self.n < 3:
            return None
        n, mae = self.n, self.sa / self.n
        var = max(0.0, self.saa / n - mae * mae)
        r = max(-0.99, min(0.99, self.se1 / self.se0)) if self.se0 > 0 else 0.0
        n_eff = min(float(n), max(1.0, n * (1 - r) / (1 + r)))  # never more than it saw
        vp, vt = self.spp / n - (self.sp / n) ** 2, self.stt / n - (self.st / n) ** 2
        corr = (self.spt / n - (self.sp / n) * (self.st / n)) / math.sqrt(vp * vt) if vp > 1e-12 and vt > 1e-12 else None
        return {"n": n, "n_eff": round(n_eff, 1), "mae": round(mae, 4), "mae_se": round(math.sqrt(var / n_eff), 4),
                "corr": None if corr is None else round(corr, 3)}


def _ground_shape(a) -> np.ndarray:
    """Ground sums at the current shape (older memories had no plant row or residual columns)."""
    a = np.zeros((0, GROUND_COLS)) if a is None or len(a) == 0 else np.atleast_2d(np.array(a, dtype=float))
    if a.shape == (len(GROUND_CLASSES), GROUND_COLS):
        return a
    out = np.zeros((len(GROUND_CLASSES), GROUND_COLS))
    r, c = min(a.shape[0], out.shape[0]), min(a.shape[1], GROUND_COLS)
    out[:r, :c] = a[:r, :c]
    return out


class Organism:
    """
    One organism living through frames: its genome's brain and perception
    tree, its eye (fovea physics + stabilizer), its body and its surprise
    memory. frame() is one frame of its life; finish() completes the last
    gaze interval when a recording ends. record=True keeps the per-gaze
    history evolution scores it on (self.rec).
    """

    RECORDS = ("positions", "fracs", "frame_path", "responses", "alarms", "teacher_p", "teacher_y", "idxs",
               "intervals", "dxs", "dys", "movement_costs", "periph_active", "prey_eaten", "foods", "energies",
               "drives", "asleeps")

    def __init__(self, g, body: dict | None = None, memory: tuple | None = None,
                 quota_pct: float = REFERENCE_QUOTA_PCT, fps: float = 15.0,
                 colour: bool = False, prey: bool = False, record: bool = False, sec_per_mac: float = 0.0):
        self.g = g
        self.state = fovea.FoveaState(n=g.receptors or fovea.DEFAULT_RECEPTORS)
        self.n_cells = self.state.n * self.state.n
        self.aspect = 16 / 9  # the frame's width / height, from its first frame
        # Its body as it actually is right now (carried across generations
        # by run_vision.run()), not a fresh full-energy body each window.
        self.body = MosquitoState.from_dict(body) if body else MosquitoState()
        # Tempo: the genome's pace is its RESTING gaze interval (temperament);
        # the brain's tempo output speeds it up or slows it down up to
        # TEMPO_RANGE-fold either way, gaze by gaze -- a continuum. The body's
        # metabolic rate acclimatizes to it slowly (MosquitoState.metabolic_rate).
        self.pace = max(1, int(getattr(g, "pace", 1)))
        self.drive_start = self.body.drive()
        # Sleep distillation (a 2026-09-29 panel; complementary learning systems,
        # McClelland, McNaughton & O'Reilly 1995; skills consolidate in sleep,
        # Walker & Stickgold): with plasticity > 0 its brain's output weights
        # -- the plastic slice -- learn in NREM from replayed episodes that went
        # better than usual: they move toward what it did then (reward-
        # weighted, like habits forming). Its own copy of the brain, this life
        # only; the genome's brain never changes (Darwinian, Dawkins, Wagner).
        self.plasticity = float(getattr(g, "plasticity", 0.0))
        self.brain = g.brain.clone() if self.plasticity > 0.0 else g.brain
        self.episode_acts: list = []   # each episode's (what fed its outputs, what they were), for distillation
        self.mean_reward = 0.0
        self.distilled = 0
        self.distill_macs = 0
        # Maturation (a 2026-09-29 panel; Hensch's critical periods, Benna &
        # Fusi's synaptic cascades, Kirkpatrick's elastic consolidation): each
        # output synapse counts the good lessons that shaped it, and locks as
        # they add up -- lock = lessons / (lessons + maturation), its updates
        # scaled by 1 - lock. What proved itself holds; what hasn't stays
        # plastic, and so do its mushroom body (its fast learner, the
        # hippocampus's part) and all its maps and libraries. And it doesn't go
        # dumb: each night, locks loosen by how wrong its predictions have
        # been lately (Nader's reconsolidation -- a memory recalled into a
        # world that no longer fits it opens up again).
        self.maturation = float(getattr(g, "maturation", 0.0))
        # Its terrain head (a 2026-09-29 panel): taught at each look by its
        # ground model's nearness at its gaze -- the lesson and the look share
        # one gaze, so their locations match by construction -- each lesson
        # weighted by how sure the teacher is (its precision against its
        # running mean, at most 1). What it learns is a nearness it feels from
        # its own eye (input 62), there when the teacher is silent.
        self.felt_terrain = int(getattr(g, "felt_terrain", 0))
        self.horizon_precision = 0.0
        self._precision_mean = _Running()
        self.felt_nearness = 0.0
        # Its prequential scores (Dawid 1984; a 2026-09-29 panel): every head
        # scored on a look before it learns from it -- out-of-sample by
        # construction. For its model card (run_vision.py) and its export.
        self.scores = {"terrain": _Prequential(), "archetypes": _Prequential(),
                       "terrain_map_vs_detector": _Prequential(), "terrain_map_vs_flow": _Prequential(),
                       "terrain_map_vs_texture": _Prequential(),
                       "host_position": _Prequential(), "host_position_plain": _Prequential()}
        self.lookahead = int(getattr(g, "lookahead", 0))
        self.flow_teacher = float(getattr(g, "flow_teacher", 0.0))
        self.extrapolation = float(getattr(g, "extrapolation", 0.0))
        self._ahead = None            # (its code, the moment) of its last look, for a lesson from the next one
        self._last_local_s = 0.0      # tau: the expansion at its gaze on its last look (replication)
        self.contact = 0.0
        self._tau_prev = None         # (its last look's small frame, where it looked)
        self.last_ahead = None        # where it expects the host it follows (its extrapolation), for the viewer
        self._predicted = None        # (the followed box, where it said it would be, where it was) for the host-position score
        self.lessons = None     # per output synapse (weights, then biases), when it matures
        self._night = False
        self._tonight = 0         # lessons distilled this night
        self.night_lessons: list = []  # lessons per night, its last ones (as many as its maturation's nights)
        self.brain.reset_hidden()
        self.brain.live_units = None  # all alive until wasting says otherwise (set every look)
        # What the gaze has seen per world location, and how much each spot
        # usually varies -- carried across generations (a fresh memory every
        # window meant a fan was only "boring" for ~40 s).
        if memory is not None:
            self.memory, self.variance = memory[0].copy(), memory[1].copy()
        else:
            self.memory, self.variance = new_memory()
        # Its mushroom body (fishbowl/mushroom.py): what it has LEARNED is
        # memory, carried like the surprise memory (the third item of memory).
        learned = memory[2] if memory is not None and len(memory) > 2 else None
        danger = memory[6] if memory is not None and len(memory) > 6 else None  # its aversive memory (item 7)
        # Its dreamed value map (item 8): what each place leads to, learned in
        # dreams (_dream); sized with the place map on the first frame.
        self.value_map = None if memory is None or len(memory) <= 7 or memory[7] is None else np.array(memory[7], dtype=float)
        # The plants it has drunk from, by field cell: each one's standing crop
        # of nectar (0..1 of a full plant; item 9 of memory, carried like the
        # rest -- the world's state as it has lived it). Unknown plants are full.
        self.nectar = dict(memory[8]) if memory is not None and len(memory) > 8 and memory[8] else {}
        self.plant_level = int(getattr(g, "plant_sense", 0))
        self.sips = 0
        proto = memory[9] if memory is not None and len(memory) > 9 else None  # its imagery (item 10)
        heads = memory[12] if memory is not None and len(memory) > 12 else None  # its archetype heads (item 13)
        self.mb = MushroomBody(int(getattr(g, "kc", 0)), int(getattr(g, "kc_seed", 0)), learned, danger, proto, heads)
        self.n_heads = int(getattr(g, "archetypes", 0))
        self.head_classes = list(getattr(g, "archetype_classes", [0, 0, 0, 0]))
        # Its ground plane (a 2026-09-29 panel; Gibson, Land, the self-driving
        # bird's-eye view): every host it sees is a measuring stick -- under a
        # flat ground and a pinhole eye, a thing's height in the frame grows in
        # proportion to how far below the horizon its base stands. Per class
        # (people and geese differ in size), running sums of (base y, height)
        # give a line whose zero is the horizon. Kept per scene (item 12).
        g_mem = memory[11] if memory is not None and len(memory) > 11 else None
        self.ground = _ground_shape(g_mem)
        # Its terrain map (a 2026-09-29 panel: the surfaces mammals draw --
        # V2's border ownership, CIP's surface slant, the occipital place area's
        # layout, boundary vector cells): per field cell, where the ground is
        # under the things standing there, as it sees them. Each whole thing it
        # measures (host, plant or object) says: standing bigger than its kind's
        # line predicts for where its feet are, it is nearer than the plane
        # puts it -- the ground there is raised by camera height x (1 -
        # predicted / seen) (pinhole geometry; clipped to one camera height).
        # Running (weight, weighted elevation) sums per cell, like the ground
        # plane's own: early vision, no gene. Kept per scene (item 14).
        t_mem = memory[13] if memory is not None and len(memory) > 13 else None
        self.terrain = np.array(t_mem, dtype=float) if t_mem is not None and len(t_mem) else None
        # Its frames of reference (a 2026-09-29 panel -- Galileo's ship, Jeffery,
        # Burgess, Wolpert): per field cell, how often a corner there stayed put
        # while the world moved (its local frame: the cab, the bonnet, the
        # glass) and how often one moved with the world -- (2, cells) counts,
        # voted only while it moves (global_scale), and remembered while it is
        # still: it learns the frames moving and keeps them stopped. One map a
        # life, not per scene: its body is the same in every place. Forgotten
        # at its slow model of the room's rate (MISMATCH_TAU_S), in seconds
        # moving only. A cell is its local frame when its share of still votes
        # beats a half at the standard 5% test. Early vision, no gene (item 15).
        f_mem = memory[14] if memory is not None and len(memory) > 14 else None
        self.frame_map = np.array(f_mem, dtype=float) if f_mem is not None and len(f_mem) else None
        self.local = None      # its local-frame cells at its latest look (bool per cell)
        self.riding = 0.0      # the share of its view that rides with it
        self.things: list = []
        self.last_nearness = 0.0
        self.colliculus = np.array(getattr(g, "colliculus", [0.0] * 6), dtype=float)
        self.priority_map = None
        self.uncertainty = 0.0   # running mean of its mushroom body's prediction errors (at its own learning rate)
        self.just_missed = False
        self.cam_shift = (0.0, 0.0)
        self.cam_scale = 0.0
        self.compass = int(getattr(g, "compass", 0))
        self.ec = Entorhinal(int(getattr(g, "kc_seed", 0)) ^ 0x5EC) if int(getattr(g, "entorhinal", 0)) else None
        self._hz_cache = None
        self.heading = 0.0            # its compass (radians): where it faces, as it has integrated its turning
        self._yaw_look = self._roll_look = 0.0
        self.turning = self.tilting = 0.0
        self.cam_moving = False
        self.last_parallax = None
        self.imagery = bool(getattr(g, "imagery", 0))
        # Recall (pattern completion; a 2026-09-28 panel -- Marr's CA3, which at
        # low load retrieves like a Hopfield net: the stored pattern that best
        # overlaps the cue): what it sees now calls up the episode of this life
        # whose Kenyon-cell code overlaps it most, if more than chance; the
        # brain gets what that episode held and where it happened. An inverted
        # index (cell -> episodes), paid per stored cell compared.
        self.recall = bool(getattr(g, "recall", 0))
        # Its perception tree, its own for this life (sleep programming may
        # edit it; a 2026-09-28 panel -- Dennett's Popperian step, the Baldwin
        # effect: what it learns dies with it, lineages that can learn win).
        # The tree now pays for its arithmetic like the brain: a step per node
        # plus each receptor a pool averages (Sterling & Laughlin).
        self.tree = copy.deepcopy(g.trees["response"]) if "response" in getattr(g, "trees", {}) else None
        self.tree_macs = self.tree.macs() if self.tree is not None else 0
        self.sleep_set = int(getattr(g, "sleep_set", 0))
        self.test_set: list = []   # a uniform sample of its waking looks: (retina planes, plain inputs, teacher's label)
        self.test_seen = 0         # waking looks offered to the sample
        self.mean_miss = 0.0       # its tree's running mean disagreement with the teacher (the sample's weights)
        self.edits_tried = self.edits_kept = 0
        self.rng_edit = np.random.default_rng(int(getattr(g, "kc_seed", 0) or 0) + 7919)
        # Scenes (a 2026-09-28 panel; hippocampal remapping, O'Keefe & Moser):
        # a library of places it has lived in, each with its own maps (food,
        # values, where people are expected, the surprise memory, the plants'
        # crops) and its layout (the field's structure, adapting over
        # LIGHT_SLOW_S). Each look it correlates the field's structure with
        # the scene it is in; while that is significant (the standard 5% test
        # on the field's effective number of cells, Bretherton et al. 1999) it stays -- someone
        # walking in hardly moves it. When it isn't, it goes to the stored
        # scene that matches best, if one does significantly, or starts a new
        # one (the least recently visited is forgotten when the library is
        # full). Born 1: one scene, never swapped. Matching costs a multiply-
        # add per cell per stored scene per look.
        self.max_scenes = int(getattr(g, "scenes", 1))
        lib = memory[10] if memory is not None and len(memory) > 10 and memory[10] else None
        self.scenes = [dict(s) for s in lib["library"]] if lib else []
        self.scene = int(lib["current"]) if lib else 0
        self.scene_switches = 0
        self._index, self._indexed, self._index_of = {}, 0, None
        self._stale = 0        # episode slots replaced since the recall index was built
        self.recalled = None   # the episode recalled this look (index), or None
        self.recalls = 0       # looks that recalled something, this life
        self.feed_index = None      # the feed's index of the frame being lived (set by a live actor)
        self.episode_meta: list = []  # each episode's (feed index, gaze cx, cy, extent): what it saw then
        self.imagery_sums = [0.0] * 6  # n, sx, sy, sxx, syy, sxy: its reconstructions against what it saw
        self.learning_rate = float(getattr(g, "learning_rate", 0.0))
        self.aversive_rate = float(getattr(g, "aversive_rate", 0.0))
        # Photoreceptor speed (after a 2026-09-28 panel; Laughlin & Weckstrom
        # 1993): a slow receptor integrates light over longer -- it blurs motion
        # and sees change later -- and costs less, as a receptor's pumping cost
        # follows the membrane conductance that sets its speed. Only slower than
        # the camera is possible: the camera's frames are the fastest light there is.
        self.slowness = float(getattr(g, "receptor_slowness", 0.0))
        self.slow = None  # what its slow receptors hold (the gaze's image, low-passed)
        self.danger_value = 0.0
        # Host defense: the host at its mouth last look (a swat is that host
        # coming at it), and the frames of the looks it was swatted on.
        self.prev_mouth_box = None
        self.followed = None   # the host it follows: (its box, when the box last changed, velocity) -- prey sense 3
        self.swats = 0
        self.swat_frames: list = []
        self.food_value = 0.0
        self.value_errors = []
        # Colour vision: how many opponent channels this genome's gaze has
        # (0-2). A plane it doesn't have reads zero.
        self.colour_n = int(getattr(g, "colour_channels", 0)) if colour else 0
        self.cones = int(min(getattr(g, "cones", self.state.n), self.state.n))
        self.last_colour = None
        # Its V4 (v4.py; a 2026-09-29 panel): colour constancy (its cones' von
        # Kries gains, adapting over genome.colour_constancy seconds), texture
        # statistics over its field (genome.texture) and the texture gradient
        # as a teacher of its terrain (genome.texture_teacher).
        tau = float(getattr(g, "colour_constancy", 0.0))
        self.v4_colour = v4.ColourAdaptation(tau) if tau > 0.0 and self.colour_n else None
        self.texture = int(getattr(g, "texture", 0))
        self.texture_teacher = float(getattr(g, "texture_teacher", 0.0))
        self.texture_map = None          # (3, cells) at its latest look, for the viewer
        self.texture_here = (0.0, 0.0, 0.0)
        # Outside its model (a 2026-09-29 panel -- Friston, Wolpert, Gelman,
        # Dennett, Nesse; a shark's tonic immobility, turned upside down): each
        # look, its detections are tested against its ground model (_model_check).
        # Failing the standard 5% test, it holds its world model's learning
        # that look (never freezes); how strange it was is a sense (input 75).
        self.strangeness = 0.0
        self.out_of_model = False
        self.model_checks = self.model_held = 0
        self.quota_pct = quota_pct
        self.scarcity = REFERENCE_QUOTA_PCT / max(1.0, quota_pct)
        self.fps = fps
        # Its deadline (after a panel vote): the brain's step takes its
        # multiply-adds x this host's measured time per multiply-add
        # (fishbowl/hostspeed.py); one that can't finish before the next look
        # misses that look and keeps its last outputs. 0 = not measured.
        self.sec_per_mac = sec_per_mac
        self.last_out = None
        self.last_interval = max(1, int(getattr(g, "pace", 1)))
        self.missed = 0
        self.stab = float(getattr(g, "stabilizer", 0.0))
        self.zoom_gain = float(getattr(g, "zoom", 0.0))
        self.frame_h = 180  # the frame's height in pixels, from its first frame (for its lens's limit)
        self.prey_level = int(getattr(g, "prey_sense", 0)) if prey else 0
        # Metabolic strategy, host preference, replay and vigilance (genome).
        self.body.metabolism = float(getattr(g, "metabolism", 1.0))
        self.body.mobilize = float(getattr(g, "mobilize", self.body.mobilize))  # its inherited fuel set points
        self.body.store = float(getattr(g, "store", self.body.store))
        self.host_pref = dict(getattr(g, "host_pref", {}) or {})
        self.awake_replay = int(getattr(g, "awake_replay", 0))
        self.sleep_replay = int(getattr(g, "sleep_replay", 0))
        self.rem_share = float(getattr(g, "rem_share", 0.0))
        self.replay_backup = float(getattr(g, "replay_backup", 0.0))
        self.dream_steps = int(getattr(g, "dream_steps", 0))
        self.dreams = 0  # dream paths dreamt in this life
        self.chain = None  # the replayed path it is walking back along: (next episode index, sequence id)
        self.seq = 0
        self.vigilance = float(getattr(g, "vigilance", 1.0))
        self.body.pump = float(getattr(g, "pump", self.body.pump))
        self.body.bore = float(getattr(g, "bore", 1.0))
        # Place map and the people-expectation maps (day, night): memory,
        # carried like the rest (items 4-6 of memory); sized on the first frame.
        extra = list(memory[3:6]) if memory is not None and len(memory) > 5 else [None, None, None]
        self.place, self.people_day, self.people_night = (None if a is None else np.array(a, dtype=float) for a in extra)
        self.episodes: list = []       # this life's (Kenyon-cell code, reward, field cell), for replay
        self.priority: list = []       # each episode's last prediction error (replay order)
        self.replays = {"awake": 0, "nrem": 0, "rem": 0}
        self.last_kc = np.zeros(0, dtype=int)  # the Kenyon cells firing at its latest look (for the viewer)
        self.replay_log: list = []             # recent replays: (kind, field row, col, frame), for the viewer's dreams
        self.last_replay = None                # (kind, Kenyon-cell code, seconds lived, episode meta): its latest replay, for the viewer
        self.mismatch = 0.0                    # its field's mismatch with its slow model of the room, this look
        self.dreaming = False                  # its eye sees its reconstruction this look (asleep, with imagery)
        self.intruder = 0.0
        self.last_alarm = 0.0
        # Tissue still alive (a wasting body loses its costliest structure:
        # Kenyon cells, hidden units, receptors -- in proportion; see _gaze).
        self.live_kc, self.live_n = self.mb.n_kc, self.state.n
        self.stab_dx = self.stab_dy = 0.0  # how far the stabilizer moved the gaze since the last gaze
        self.prev_v = np.zeros(self.n_cells)
        # Response tree's motor-efference input: the brain's real applied movement.
        self.prev_dx, self.prev_dy = 0.0, 0.0
        self.prev_frame = None
        self.prev_response = 0.0
        self.last_grid = None
        self.k = 0             # frames lived
        self.lived_s = 0.0     # seconds lived (frames at their own rate, which varies)
        self.pending = None    # the current gaze interval: its decisions and progress
        self.eating = 0.0      # prey in its gaze centre at its latest gaze
        self.rec = {name: [] for name in self.RECORDS} if record else None

    # ---- one frame -------------------------------------------------------------
    def frame(self, frame: np.ndarray, sig, boxes: list | None = None, colour_frame: np.ndarray | None = None,
              shift: tuple = (0.0, 0.0)) -> dict:
        """
        One frame of its life. frame: grey (uint8); sig: this frame's
        whole-field signals (SIGNAL_KEYS); boxes: prey boxes in it (prey.py);
        colour_frame: BGR, if it has colour vision; shift: the whole frame's
        shift from the previous frame (for the stabilizer).
        """
        self.cam_shift = (float(shift[0]), float(shift[1]))
        self.cam_scale = float(shift[2]) if len(shift) > 2 else 0.0
        # its vestibular sense (a 2026-09-29 panel; Taube, Jayaraman): yaw from
        # the frame's shift (+ turning right: the world slides left; focal length
        # ~ the frame's height, as its ground model assumes) and roll from the
        # ego-motion fit, summed over each look; its compass integrates the yaw
        yaw = -math.atan(self.cam_shift[0] * self.aspect)
        roll = float(shift[3]) if len(shift) > 3 else 0.0
        self._yaw_look += yaw
        self._roll_look += roll
        if self.compass:
            self.heading = (self.heading + yaw) % (2 * math.pi)
        if self.ec is not None:  # its speed cells and path integrator, frame by frame
            self.ec.step(self.cam_scale, self.fps, self._hz_cache, self.heading)
        self.cam_moving = camera_moves(self.cam_shift + (self.cam_scale, roll), shift_size(frame.shape)[::-1])
        if self.k == 0:
            if len(self.scenes) > self.max_scenes:  # a smaller library now: keep the most recent
                keep = sorted(range(len(self.scenes)), key=lambda i: self.scenes[i].get("last", 0.0))[-self.max_scenes:]
                cur = self.scenes[self.scene] if self.scene < len(self.scenes) else None
                self.scenes = [self.scenes[i] for i in sorted(keep)]
                self.scene = next((i for i, s in enumerate(self.scenes) if s is cur), 0)
            self.aspect = frame.shape[1] / max(1, frame.shape[0])
            self.frame_h = frame.shape[0]
            self.field = field_shape(frame.shape[0], frame.shape[1])
            for name in ("place", "people_day", "people_night", "value_map"):
                a = getattr(self, name)
                if a is None or a.shape != self.field:
                    setattr(self, name, np.zeros(self.field))
        if self.cam_moving and self.field:
            self._vote_frames(sig)
            if self.flow_teacher > 0.0 and not self.out_of_model:
                self._learn_terrain_from_flow(sig, shift_size(frame.shape), roll)
        if self.slowness > 0.0:  # slow photoreceptors: its gaze sees the frames low-passed
            a = 1.0 - math.exp(-REFERENCE_GAZES_PER_S / (max(1.0, self.fps) * self.slowness))
            f = frame.astype(np.float32)
            self.slow = f if self.slow is None or self.slow.shape != f.shape else self.slow + a * (f - self.slow)
            frame = self.slow
        plants = prey_lib.plants_only(boxes)  # its nectar; everything else sees hosts only
        self.things = prey_lib.things_only(boxes)  # its early vision's measuring sticks only
        boxes = prey_lib.hosts_only(boxes)
        if self.pending is not None:
            self._substep(shift, in_world=True)
            if self.pending["done"] == self.pending["interval"]:
                self._close()
            elif self.rec is not None:
                self.rec["frame_path"].append((self.state.cx, self.state.cy, self.state.extent))
        gazed = self.pending is None
        if gazed:
            self._gaze(frame, sig, boxes or [], colour_frame, plants)
        self.k += 1
        self.lived_s += 1.0 / max(1.0, self.fps)
        return {"cx": self.state.cx, "cy": self.state.cy, "extent": self.state.extent, "receptors": self.state.n,
                "asleep": self.body.asleep >= 0.5, "eating": self.eating, "gazed": gazed}

    def finish(self) -> None:
        """A recording ended: complete the last gaze interval (the eye keeps
        moving and the body is paid for it, with no more frames to see)."""
        while self.pending is not None:
            self._substep((0.0, 0.0), in_world=False)
            if self.pending["done"] == self.pending["interval"]:
                self._close()

    # ---- internals ---------------------------------------------------------------
    def _gaze(self, frame: np.ndarray, sig, boxes: list, colour_frame, plants: list | None = None) -> None:
        g, brain, body, state, rec = self.g, self.brain, self.body, self.state, self.rec
        # Asleep, its eyes are shut: the gaze sees nothing. The whole field
        # still reaches it (light and movement through closed eyes), so a
        # big enough change can wake it.
        was_asleep = body.asleep >= 0.5
        n = state.n
        # Wasting (tissue burned for a brain with no sugar, state.py) takes the
        # costly structure with it, in proportion: Kenyon cells (the newest
        # first), hidden units (silenced), and the eye's outer rings (blind).
        # What is lost stops costing too (_close).
        w = body.wasting
        self.live_kc = int(self.mb.n_kc * (1.0 - w)) if w > 0.0 else self.mb.n_kc
        brain.live_units = max(1, int(round(brain.n_hidden * (1.0 - w)))) if w > 0.0 else None
        self.live_n = max(2, 2 * int(n * (1.0 - w) / 2)) if w > 0.0 else n
        v = np.zeros(self.n_cells) if was_asleep else self._blind(fovea.extract(frame, state), n)
        # Dreaming, it sees its dream (a 2026-09-28 panel): eyes shut, its eye
        # gets its own reconstruction of what it replayed last look -- never a
        # recorded frame -- so its brain and mushroom body run on the dream.
        dreaming = False
        if was_asleep and self.imagery and self.last_replay is not None and self.lived_s - self.last_replay[2] < 1.0:
            dream_img = self.mb.reconstruct(np.asarray(self.last_replay[1])[np.asarray(self.last_replay[1]) < self.mb.n_kc])
            if dream_img is not None:
                v = cv2.resize(dream_img.reshape(PROTO_SIDE, PROTO_SIDE).astype(np.float32), (n, n), interpolation=cv2.INTER_LINEAR).ravel()
                dreaming = True
        self.dreaming = dreaming
        if rec is not None:
            rec["positions"].append((state.cx, state.cy))
            rec["fracs"].append(state.extent)
            rec["frame_path"].append((state.cx, state.cy, state.extent))
        # What the look sees change, with its own eye movement cancelled out
        # (efference copy): the previous frame sampled where the gaze is now,
        # less what the stabilizer moved it -- so a saccade across a still
        # scene doesn't register as motion, and a shake it held still reads
        # as stillness.
        h1 = (fovea.extract(self.prev_frame, fovea.FoveaState(cx=state.cx - self.stab_dx, cy=state.cy - self.stab_dy, n=n, mag=state.mag))
              if self.prev_frame is not None and not was_asleep else v)
        h1 = self._blind(h1, n)
        hist = np.array([h1, v])
        lum = float(v.mean())
        # Only real history counts: on the first frame there's no older sample.
        motion = flow_x = flow_y = 0.0
        if self.prev_frame is not None and not was_asleep:
            motion = min(1.0, float(np.abs(v - h1).mean()) * MOTION_GAIN)
            mx, my = reflexes.directional_motion(hist, (n, n))
            flow_x = float(np.clip(mx[-1] * FLOW_GAIN, -1.0, 1.0))
            flow_y = float(np.clip(my[-1] * FLOW_GAIN, -1.0, 1.0))
        # Threat and arousal come from the WHOLE visual field, plus where in it
        # something moved, relative to the look -- so the brain can learn to
        # swing its look toward movement.
        loom = min(1.0, float(sig["expansion"]) * EXPANSION_GAIN)
        periph_motion = min(1.0, float(sig["motion_energy"]) * PERIPH_MOTION_GAIN)
        periph_dx = float(sig["motion_cx"]) - state.cx
        periph_dy = float(sig["motion_cy"]) - state.cy
        field_light = float(sig["field_light"])
        mismatch = min(1.0, float(sig["mismatch"]) * EXPANSION_GAIN)
        mismatch_dx = float(sig["mismatch_cx"]) - state.cx
        mismatch_dy = float(sig["mismatch_cy"]) - state.cy
        self.mismatch = mismatch
        host_vx, host_vy = self._host_velocity(boxes) if self.prey_level >= 3 else (0.0, 0.0)
        seen_boxes = self._extrapolate(boxes or [], host_vx, host_vy)
        scent, prey_dx, prey_dy = prey_sense(seen_boxes, state.cx, state.cy, self.prey_level, self.host_pref)
        plant_scent, plant_dx, plant_dy = prey_sense(plants or [], state.cx, state.cy, self.plant_level, None)
        # Its mushroom body's learned value of what the look shows (eyes shut: nothing).
        kc_active = self.mb.active(v, n, self.live_kc) if (not was_asleep or dreaming) else np.zeros(0, dtype=int)
        self.last_kc = kc_active
        self.food_value = float(np.clip(self.mb.value(kc_active), -1.0, 1.0))
        self.danger_value = float(np.clip(self.mb.danger(kc_active), -1.0, 1.0)) if self.aversive_rate > 0.0 else 0.0
        head_vals = np.clip(self.mb.head_values(kc_active), -1.0, 1.0) if self.n_heads else np.zeros(4)  # the archetype rows only
        head_vals[self.n_heads:] = 0.0
        head_macs = 2 * (self.n_heads + self.felt_terrain) * len(kc_active)  # reading them now, and teaching them below
        ground_near, horizon = self._ground_sense(state.cy)
        terrain = self.terrain_at(state.cx, state.cy) if not was_asleep else 0.0
        nearness = self._nearness(state.cx, state.cy, (boxes or []) + (plants or []) + self.things) if not was_asleep else 0.0
        felt = float(np.clip(self.mb.terrain_value(kc_active), 0.0, 1.0)) if self.felt_terrain else 0.0
        self.felt_nearness = felt
        self.local = self.local_frame()
        self.riding = float(self.local.mean()) if self.local is not None else 0.0
        if not was_asleep:
            head_macs += self._model_check((boxes or []) + (plants or []) + self.things)
        texture_in = (0.0, 0.0, 0.0)
        if self.texture and not was_asleep and self.field:
            texture_in = self._texture(frame, state)
            head_macs += 3 * self.field[0] * self.field[1]  # its V4 texture cells: three statistics a field cell, as its collicular map's
        contact, contact_macs = self._contact(frame, state) if not was_asleep else (0.0, 0)
        head_macs += contact_macs
        # what its body felt turning and tilting since its last look, in its half field of view
        half_fov = math.atan(0.5 * self.aspect)
        self.turning = float(np.clip(self._yaw_look / half_fov, -1.0, 1.0))
        self.tilting = float(np.clip(self._roll_look / half_fov, -1.0, 1.0))
        self._yaw_look = self._roll_look = 0.0
        heading = (math.sin(self.heading), math.cos(self.heading)) if self.compass else (0.0, 0.0)
        ego_speed = acceleration = map_value = 0.0
        if self.ec is not None:
            self._hz_cache = self.horizon()
            ego_speed, acceleration = self.ec.look()
            self.ec.places(self.scene)
            map_value = float(np.clip(self.ec.value(), -1.0, 1.0))
            head_macs += self.ec.macs()
        self.last_nearness = nearness
        cell_p = self._cell(state.cx, state.cy)
        par = sig["parallax"] if not was_asleep else None
        self.last_parallax = par
        parallax = float(par[cell_p[0] * self.place.shape[1] + cell_p[1]]) if par is not None and cell_p is not None and len(par) else 0.0
        camera_moving = float(min(1.0, max(math.hypot(*self.cam_shift), abs(self.cam_scale)) / SHIFT_MAX)) if self.cam_moving else 0.0
        coll, coll_macs = self._colliculus(sig, boxes, plants, state, was_asleep)
        own_pace = float(np.tanh(math.log2(max(1e-6, self.last_interval / max(1.0, self.fps) * REFERENCE_GAZES_PER_S))))
        cell = self._cell(state.cx, state.cy)
        place_dx, place_dy, place_value = self._place_sense(state.cx, state.cy)
        scene_macs = self._place_in(sig["structure"], self.last_interval / max(1.0, self.fps))
        self.intruder = self._intruder(boxes, body.daylight, self.last_interval / max(1.0, self.fps))
        recalled = recalled_dx = recalled_dy = 0.0
        recall_macs = 0
        self.recalled = None
        if self.recall and len(kc_active):
            self.recalled, recall_macs = self._recall(kc_active)
            if self.recalled is not None:
                _, r_reward, r_cell = self.episodes[self.recalled]
                recalled = float(np.clip(r_reward, -1.0, 1.0))
                if self._in_map(r_cell):
                    rows, cols = self.place.shape
                    recalled_dx, recalled_dy = (r_cell[1] + 0.5) / cols - state.cx, (r_cell[0] + 0.5) / rows - state.cy
                self.recalls += 1
        # Replay (awake in a quiet moment, or asleep once settled): it takes
        # time from this look's deadline and costs thinking energy.
        quiet = not was_asleep and not boxes and not body.big_change(loom, periph_motion, self.vigilance)
        replay_macs = self._replay(was_asleep and body.sleep_clock > SLEEP_SETTLE_S, quiet)
        if quiet:  # a quiet waking moment: it can try an edit to its tree as it does asleep (awake consolidation)
            replay_macs += self._sleep_program()
        if was_asleep and body.sleep_clock > SLEEP_SETTLE_S:
            replay_macs += self._dream()
            replay_macs += self._sleep_program()
        replay_macs += recall_macs + scene_macs + head_macs + coll_macs
        if (self.sec_per_mac and self.last_out is not None
                and self.sec_per_mac / max(1e-3, body.metabolism) * (brain.macs() + replay_macs)
                > self.last_interval / max(1.0, self.fps)):  # its brain runs at the pace its metabolism powers
            out = self.last_out  # still thinking: this look is missed
            self.missed += 1
            self.just_missed = True
        else:
            self.just_missed = False
            out = brain.step(
                lum, motion, flow_x, flow_y, loom, state.cx, state.cy, state.extent, body,
                periph_dx, periph_dy, state.vx, state.vy, self.prev_response, field_light, scent, prey_dx, prey_dy,
                self.food_value, place_dx, place_dy, place_value, self.intruder, self.danger_value,
                plant_scent, plant_dx, plant_dy, mismatch, mismatch_dx, mismatch_dy,
                recalled, recalled_dx, recalled_dy, host_vx, host_vy,
                own_pace, 1.0 if self.just_missed else 0.0, self.uncertainty, ground_near, horizon, parallax, camera_moving,
                tuple(head_vals), coll, terrain, nearness, felt, contact, self.turning, self.tilting, heading,
                ego_speed, acceleration, map_value, self.riding, texture_in, self.strangeness,
            )
            self.last_out = out
        pan, tilt, alarm, tempo = out.pan, out.tilt, out.alarm, out.tempo
        self.last_alarm = alarm
        # Sleep is its own choice (its sleep output); the body adds only the
        # physiological overrides -- collapse, hunger, a big change (state.py).
        body.set_sleep(out.sleep > 0.0, loom, periph_motion, self.vigilance, mismatch)
        asleep = body.asleep >= 0.5
        # An empty body runs on less (soft floor): colour off, slower gazing.
        # Asleep, the eye is shut: no colour either.
        colour_on = self.colour_n if not (asleep or body.degraded) else 0
        # Its lens for the next look (zoom in only, fovea.magnification);
        # asleep, the eye is shut and the lens relaxes.
        state.mag = 1.0 if asleep else fovea.magnification(out.zoom, self.zoom_gain, self.frame_h)
        # The perception tree reads its receptors by position (the look, the
        # previous look, its colour planes) and, as plain inputs, its own
        # last movement and the brain's recurrent memory.
        gains = self.v4_colour.gains() if (colour_on and self.v4_colour is not None) else None
        col = fovea.extract_colour(colour_frame, state, colour_on, self.cones, gains) if colour_on else np.zeros(0)
        if colour_on:
            self.last_colour = col
            if self.v4_colour is not None and colour_frame is not None:  # its cones adapt to what they saw (von Kries)
                self.v4_colour.see(v4.cone_mean(fovea._window(colour_frame, state), state.n, self.cones),
                                   self.last_interval / max(1.0, self.fps))
        planes = np.zeros((1, RETINA_PLANES, n, n))
        planes[0, 0], planes[0, 1] = v.reshape(n, n), self.prev_v.reshape(n, n)
        for c in range(len(col) // self.n_cells):
            planes[0, 2 + c] = col[c * self.n_cells:(c + 1) * self.n_cells].reshape(n, n)
        plain = np.concatenate([[self.prev_dx, self.prev_dy], brain.tree_view()])[None, :]
        response = float(self.tree.evaluate(plain, planes)[0]) if self.tree is not None else float(g.evaluate("response", plain, planes)[0])
        if self.sleep_set and not was_asleep:
            # Its sleep test set: a sample of its waking looks, weighted toward
            # where its tree disagreed with the teacher (the "data engine" of
            # self-driving research, 2026-09-29 panel 13-1): weighted reservoir
            # sampling (Efraimidis & Spirakis 2006, key = u^(1/w)); weight = its
            # disagreement + the running mean disagreement (self-scaling: an
            # average miss counts double a perfect look; none is excluded).
            label = float(prey_lib.prey_in_window(boxes, state.cx, state.cy, *state.half_extents(self.aspect)))
            guess = 0.5 * (1.0 + math.tanh(response)) if math.isfinite(response) else 0.5
            miss = abs(guess - label)
            self.test_seen += 1
            self.mean_miss += (miss - self.mean_miss) / min(self.test_seen, max(1, self.sleep_set))
            key = float(self.rng_edit.random()) ** (1.0 / max(1e-9, miss + self.mean_miss))
            item = (planes.copy(), plain.copy(), label, key)
            if len(self.test_set) < self.sleep_set:
                self.test_set.append(item)
            else:
                j = min(range(len(self.test_set)), key=lambda i: self.test_set[i][3] if len(self.test_set[i]) > 3 else 0.0)
                if key > (self.test_set[j][3] if len(self.test_set[j]) > 3 else 0.0):
                    self.test_set[j] = item
        # The perception tree's output reaches the brain next gaze; the tree
        # is also graded by its teacher (run_vision.TEACHER_WEIGHT).
        self.prev_response = float(np.tanh(response)) if math.isfinite(response) else 0.0
        if rec is not None:
            # One per gaze (None asleep): its tree's own guess at how much prey
            # fills its gaze, and the teacher's label for it.
            rec["teacher_p"].append(None if was_asleep else 0.5 * (1.0 + self.prev_response))
            rec["teacher_y"].append(None if was_asleep else prey_lib.prey_in_window(boxes, state.cx, state.cy, *state.half_extents(self.aspect)))
            rec["responses"].append(response)
            rec["alarms"].append(alarm)
        self.last_grid = v
        self.prev_v = v

        # Eating happens at the gaze: prey under its centre (its mouth) is a meal;
        # genuinely new structure there is a small snack.
        sip_key = None  # the plant it sips from this look, if any
        proto_macs = 0  # its imagery's learning this look
        if asleep:
            # Asleep: no eating, no gazing, slow coarse sampling of the field.
            pan = tilt = 0.0
            prey_now = snack = 0.0
            interval = MAX_INTERVAL
            self.prev_mouth_box = None
            body.bite_over()
        else:
            prey_now = prey_lib.prey_at_mouth(boxes, state.cx, state.cy, self.aspect)  # its mouth: the gaze centre
            # Host defense (a 2026-09-28 panel): a swat is the host it is biting
            # coming at it -- the same host, nearer than last look, while the
            # field looms past the line that already counts as a big change
            # (WAKE_LOOM). The swat takes back this bite's blood; whether to
            # stay is its own business.
            host = prey_lib.host_box_at_mouth(boxes, state.cx, state.cy, self.aspect)
            punishment = 0.0
            if host is not None and loom >= WAKE_LOOM and prey_lib.approached(self.prev_mouth_box, host):
                punishment = body.swat() / PREY_FOOD_PER_LOOK  # in meals, like the reward
                prey_now = 0.0  # swatted: no blood this interval
                self.swats += 1
                if rec is not None:
                    self.swat_frames.append(self.k)
            elif host is None:
                body.bite_over()
            self.prev_mouth_box = host
            if len(kc_active) and self.aversive_rate > 0.0:
                self.mb.learn_danger(kc_active, punishment, self.aversive_rate)
            snack = feed_on_novelty(self.memory, v, state, self.variance, self.aspect)
            # Nectar: a plant under its mouth, when no host is (blood first). It
            # flows as blood does -- at its pump's rate -- times how full the
            # plant still is; the plant's crop is drawn down by what it takes.
            plant = prey_lib.host_box_at_mouth(plants or [], state.cx, state.cy, self.aspect) if prey_now <= 0 else None
            sip_key = None
            if plant is not None:
                pcell = self._cell((plant[2] + plant[4]) / 2, (plant[3] + plant[5]) / 2)
                if pcell is not None:
                    sip_key = f"{pcell[0]},{pcell[1]}"
                    self.nectar.setdefault(sip_key, 1.0)
            crop = self.nectar.get(sip_key, 0.0) if sip_key else 0.0
            # Lifetime learning: what it ate this look (in meals: a full look at
            # prey = 1; a sip is worth as much as the plant is full) teaches its
            # mushroom body what the look showed.
            reward = (PREY_FOOD_PER_LOOK * prey_now + FOOD_PER_LOOK * snack) / PREY_FOOD_PER_LOOK + crop
            if self.ec is not None:
                self.ec.learn(reward, self.learning_rate)  # what this place has been worth
            if len(kc_active) and self.learning_rate > 0.0:
                err = self.mb.learn(kc_active, reward, self.learning_rate)
                self.value_errors.append(abs(err))
                self.uncertainty += self.learning_rate * (min(1.0, abs(err)) - self.uncertainty)  # its own timescale
                if self.n_heads:  # its archetype heads, each taught by its class's presence in the gaze
                    seen = (boxes or []) + (plants or [])
                    hx, hy = state.half_extents(self.aspect)
                    targets = np.array([prey_lib.prey_in_window([b for b in seen if int(b[0]) == c], state.cx, state.cy, hx, hy)
                                        for c in self.head_classes])
                    before = self.mb.head_values(kc_active)
                    for k in range(self.n_heads):  # scored before it learns (prequential)
                        self.scores["archetypes"].add(float(before[k]), float(targets[k]))
                    self.mb.learn_heads(kc_active, targets, self.learning_rate, self.n_heads)
                if self.felt_terrain and not self.out_of_model and self.horizon() is not None and self.horizon_precision > 0.0:
                    # its terrain head, taught by its ground model's nearness at this gaze (not when its model fails)
                    self._precision_mean.add(self.horizon_precision)
                    weight = min(1.0, self.horizon_precision / max(1e-12, self._precision_mean.mean))
                    if self.lookahead:
                        # taught toward the next look: what it will see by the time it has seen this one
                        if self._ahead is not None and len(self._ahead):
                            self.scores["terrain"].add(self.mb.terrain_value(self._ahead), nearness)
                            self.mb.learn_terrain(self._ahead, nearness, self.learning_rate * weight)
                        self._ahead = kc_active.copy()
                    else:
                        self.scores["terrain"].add(self.mb.terrain_value(kc_active), nearness)
                        self.mb.learn_terrain(kc_active, nearness, self.learning_rate * weight)
                self._remember((kc_active.copy(), reward, cell), abs(err),
                               (self.feed_index, state.cx, state.cy, state.extent))
            if boxes or plants or self.things:
                self._learn_ground((boxes or []) + (plants or []) + self.things)
            proto_macs = 0
            if self.imagery and len(kc_active) and self.learning_rate > 0.0:
                # its imagery: what the eye sees now, on the prototypes' grid;
                # first how well its memory predicts it (Gelman's measure), then learn
                seen = cv2.resize(v.reshape(n, n).astype(np.float32), (PROTO_SIDE, PROTO_SIDE),
                                  interpolation=cv2.INTER_AREA if n >= PROTO_SIDE else cv2.INTER_LINEAR).ravel()
                guess = self.mb.reconstruct(kc_active)
                sm = self.imagery_sums
                for idx, val in enumerate((1.0, guess.sum(), seen.sum(), (guess * guess).sum(), (seen * seen).sum(), (guess * seen).sum())):
                    sm[idx] += float(val) if idx else float(len(seen))
                self.mb.learn_proto(kc_active, seen, self.learning_rate)
                proto_macs = 2 * len(kc_active) * PROTO_SIDE * PROTO_SIDE
            if self.learning_rate > 0.0 and cell is not None:
                self.place[cell] += self.learning_rate * (reward - self.place[cell])
            interval = int(np.clip(round(self.pace * TEMPO_RANGE ** (-float(tempo))), 1, MAX_INTERVAL))
            if body.degraded:
                interval = min(MAX_INTERVAL, interval * 2)
        self.eating = prey_now
        self.last_interval = interval
        if rec is not None:
            rec["idxs"].append(self.k)
            rec["intervals"].append(interval)
        # Eye physics every FRAME until the next gaze: the brain's force is
        # held, and the damped eye keeps moving meanwhile.
        self.stab_dx = self.stab_dy = 0.0
        self.pending = {"pan": pan, "tilt": tilt, "asleep": asleep, "colour_on": colour_on, "kc_on": bool(len(kc_active)),
                        "replay_macs": replay_macs,
                        "interval": interval, "done": 0, "effort": 0.0, "force": (0.0, 0.0),
                        "alarm": max(0.0, float(alarm)),
                        "cx0": state.cx, "cy0": state.cy, "periph_motion": periph_motion, "loom": loom,
                        "field_light": field_light, "prey_now": prey_now, "snack": snack,
                        "sip": sip_key if not asleep else None,
                        "proto_macs": 0 if asleep else proto_macs}
        self.prev_frame = frame

    def _blind(self, look: np.ndarray, n: int) -> np.ndarray:
        """A wasting eye's outer rings, lost: they read black."""
        if self.live_n >= n:
            return look
        m = (n - self.live_n) // 2
        grid = look.reshape(n, n).copy()
        grid[:m, :] = grid[-m:, :] = grid[:, :m] = grid[:, -m:] = 0.0
        return grid.reshape(-1)

    def _sleep_program(self) -> int:
        """One trial edit to its own tree, tested on its sample of waking looks
        against the teacher's labels; kept if it predicts them better. Returns
        the multiply-adds spent (both trees over the sample)."""
        if not self.sleep_set or self.tree is None:
            return 0
        n = self.state.n
        items = [t for t in self.test_set if t[0].shape[-1] == n]
        if len(items) < 2:
            return 0
        from .genome import edit_tree
        from .sandbox import Limits
        lim = Limits()
        rnd = random.Random(int(self.rng_edit.integers(1 << 30)))
        cand = edit_tree(self.tree, rnd, getattr(self.g, "n_vars", 3), getattr(self.g, "receptors", n),
                         lim.max_tree_nodes, lim.max_tree_depth)
        if cand is None:
            return 0
        planes = np.concatenate([t[0] for t in items]); plain = np.concatenate([t[1] for t in items])
        y = np.array([t[2] for t in items])
        def err(tree):
            out = np.tanh(np.nan_to_num(tree.evaluate(plain, planes)))
            return float(np.mean((0.5 * (1.0 + out) - y) ** 2))
        cand_macs = cand.macs()
        self.edits_tried += 1
        if err(cand) < err(self.tree):
            self.tree, self.tree_macs = cand, cand_macs
            self.edits_kept += 1
        return (self.tree_macs + cand_macs) * len(items)

    SCENE_MAPS = ("place", "people_day", "people_night", "value_map", "memory", "variance", "ground", "terrain")

    def _place_in(self, structure, seconds: float) -> int:
        """Which scene it is in (see __init__); returns the multiply-adds spent."""
        if self.max_scenes <= 1:
            return 0
        s = np.asarray(structure, dtype=float)
        if float(s.std()) <= NOISE_FLOOR:  # no structure (a blank or black frame): nothing to place
            return len(s)
        if self.cam_moving:  # the camera itself is moving: the layout is sliding, not a new place (Gelman)
            return len(s)
        if not self.scenes:  # its first scene: where it is now
            self.scenes, self.scene = [{"gist": s.copy(), "last": self.lived_s}], 0
            return len(s)

        grid = self.field if self.field and self.field[0] * self.field[1] == len(s) else (1, len(s))

        def lag1(v):  # neighbour autocorrelation over the field's grid (rows and columns)
            m = v.reshape(grid) - v.mean()
            num = float((m[:, 1:] * m[:, :-1]).sum() + (m[1:, :] * m[:-1, :]).sum())
            den = float((m * m).sum()) * (m[:, 1:].size + m[1:, :].size) / m.size
            return num / den if den > 0 else 0.0
        rho_s = lag1(s)

        def match(g):
            """Its correlation with a scene's layout, and whether that beats chance: the
            standard 5% test on the effective number of cells (neighbours are alike:
            Bretherton et al. 1999, N_eff = N (1 - ra rb) / (1 + ra rb))."""
            if g.shape != s.shape:
                return 0.0, False
            a, b = s - s.mean(), g - g.mean()
            d = math.sqrt(float((a * a).sum() * (b * b).sum()))
            r = float((a * b).sum() / d) if d > 0 else 0.0
            rr = max(0.0, rho_s * lag1(g))
            n_eff = max(3.0, len(s) * (1.0 - rr) / (1.0 + rr))
            return r, r > 1.96 / math.sqrt(n_eff)

        def corr(g):
            r, sig = match(g)
            return r if sig else -1.0
        line = -1.0
        here = corr(self.scenes[self.scene]["gist"])
        macs = len(s)
        if here <= line:
            rs = [corr(sc["gist"]) if i != self.scene else -1.0 for i, sc in enumerate(self.scenes)]
            macs += len(s) * (len(self.scenes) - 1)
            best = int(np.argmax(rs)) if rs else 0
            self._stash()
            if rs and rs[best] > line:
                self.scene = best
                self._unstash()
            else:
                if len(self.scenes) >= self.max_scenes:  # forget the least recently visited
                    old = min(range(len(self.scenes)), key=lambda i: self.scenes[i].get("last", 0.0))
                    del self.scenes[old]
                self.scenes.append({"gist": s.copy(), "last": self.lived_s})
                self.scene = len(self.scenes) - 1
                self._fresh_maps()
            self.scene_switches += 1
        sc = self.scenes[self.scene]
        a = 1.0 - math.exp(-max(0.0, seconds) / LIGHT_SLOW_S)
        sc["gist"] = sc["gist"] + a * (s - sc["gist"]) if sc["gist"].shape == s.shape else s.copy()
        sc["last"] = self.lived_s
        return macs

    def _stash(self) -> None:
        """The scene it is leaving keeps its maps."""
        sc = self.scenes[self.scene]
        for name in self.SCENE_MAPS:
            a = getattr(self, name)
            sc[name] = None if a is None else np.array(a, copy=True)
        sc["nectar"] = dict(self.nectar)
        if self.compass:
            sc.setdefault("heading", self.heading)  # its landmark: the heading it had here
        if self.ec is not None:
            sc.setdefault("position", [float(v) for v in self.ec.position])  # and where it was

    def _unstash(self) -> None:
        """Back in a stored scene: its maps again."""
        sc = self.scenes[self.scene]
        for name in self.SCENE_MAPS:
            if sc.get(name) is not None:
                setattr(self, name, np.array(sc[name], copy=True))
        self.nectar = dict(sc.get("nectar") or {})
        if self.compass and sc.get("heading") is not None:
            self.heading = float(sc["heading"])  # back among its landmarks: its compass re-anchors to them
        if self.ec is not None and sc.get("position") is not None:
            self.ec.position = np.array(sc["position"], dtype=float)  # and its path integrator

    def _fresh_maps(self) -> None:
        """A new place: nothing learned about it yet."""
        for name in ("place", "people_day", "people_night", "value_map"):
            a = getattr(self, name)
            if a is not None:
                setattr(self, name, np.zeros_like(a))
        self.memory, self.variance = new_memory()
        self.nectar = {}
        self.ground = np.zeros_like(self.ground)
        self.terrain = None if self.terrain is None else np.zeros_like(self.terrain)

    def library(self) -> dict | None:
        """Its scene library, for memory (item 11): the current scene's maps stashed first."""
        if not self.scenes:
            return None
        self._stash()
        return {"current": self.scene, "library": [dict(s) for s in self.scenes]}

    def _colliculus(self, sig, boxes, plants, state, asleep: bool):
        """Its collicular priority map: per whole-field cell, the weighted sum
        of COLLICULAR_FEATURES; the brain gets the direction to the strongest
        cell and how strong it is. Off (no work) while every weight is 0."""
        if asleep or self.place is None or not self.colliculus.any():
            self.priority_map = None
            return (0.0, 0.0, 0.0), 0
        rows, cols = self.place.shape
        n = rows * cols
        def cells(key):
            try:
                a = np.asarray(sig[key], dtype=float).ravel()
            except (KeyError, IndexError, TypeError):
                return np.zeros(n)
            return a if a.shape[0] == n else np.zeros(n)
        host, size, plant = np.zeros(n), np.zeros(n), np.zeros(n)
        for grid, items, sized in ((host, boxes or [], True), (plant, plants or [], False)):
            for c, conf, x0, y0, x1, y1 in items:
                cell = self._cell((x0 + x1) / 2, (y0 + y1) / 2)
                if cell is None:
                    continue
                k = cell[0] * cols + cell[1]
                grid[k] = max(grid[k], float(conf))
                if sized:
                    size[k] = max(size[k], float(conf) * min(1.0, (x1 - x0) * (y1 - y0) * n))  # 1 = a box a cell big or more
        feats = np.stack([cells("motion_map"), cells("mismatch_map"), cells("parallax"), host, size, plant])
        s = self.colliculus @ feats
        if self.local is not None and len(self.local) == n:  # what rides with it doesn't pull its gaze
            s = np.where(self.local, np.minimum(s, 0.0), s)
        self.priority_map = s
        k = int(np.argmax(s))
        if s[k] <= 0.0:
            return (0.0, 0.0, 0.0), feats.size
        r, c = divmod(k, cols)
        return ((c + 0.5) / cols - state.cx, (r + 0.5) / rows - state.cy, float(np.clip(s[k], 0.0, 1.0))), feats.size

    def _learn_ground(self, boxes: list) -> None:
        """Each host or plant a measuring stick: its base's height in the frame
        and its own height, into its class's running sums (n, y, h, y^2, y*h),
        weighted by the detector's confidence; and how far it sat from its
        class's line as fitted then (the residual, for how much to trust it).
        Outside its model (_model_check), only the residuals learn: its lines
        and terrain hold, while how much it expects to be wrong keeps up with
        what it sees -- a long spell of strangeness widens what it accepts, as
        people adapt to inverting goggles (Stratton 1897) -- and censoring the
        data that sets its own test would close the gate ever tighter."""
        self.ground = _ground_shape(self.ground)
        for c, conf, x0, y0, x1, y1 in boxes:
            if any(prey_lib.cut_by_frame((c, conf, x0, y0, x1, y1), self.aspect)):
                continue  # the frame cuts its base or its top: where it stands, or its height, is unseen
            if int(c) in GROUND_CLASSES and y1 > y0:
                k, w = GROUND_CLASSES.index(int(c)), float(conf)
                fit = self._ground_fit(self.ground[k])
                if fit is not None:
                    a, b = fit
                    self.ground[k, 5:7] += w * np.array([1.0, ((y1 - y0) - (a * y1 + b)) ** 2])
                    pred = a * y1 + b
                    cell = self._cell((x0 + x1) / 2, min(y1, 1.0 - 1e-6))
                    if pred > 0 and self._terrain_map() is not None and self._in_map(cell) and self.flow_teacher < 1.0 and not self.out_of_model:
                        elev = float(np.clip(1.0 - pred / (y1 - y0), -1.0, 1.0))
                        self.scores["terrain_map_vs_detector"].add(self.terrain_at((x0 + x1) / 2, min(y1, 1.0 - 1e-6)), elev)
                        self.terrain[cell[0] * self.place.shape[1] + cell[1]] += w * (1.0 - self.flow_teacher) * np.array([1.0, elev])
                if not self.out_of_model:
                    self.ground[k, :5] += w * np.array([1.0, y1, y1 - y0, y1 * y1, y1 * (y1 - y0)])

    def _texture(self, frame, state) -> tuple[float, float, float]:
        """Its V4's texture statistics over its field (v4.texture), the texture
        gradient's lessons to its terrain (weighted texture_teacher x the
        cell's share of signal, 1 - noise floor / contrast; scored before it
        learns), and what the texture is at its gaze: contrast (x 2: the most a
        0-1 image's spread can be is 1/2), fineness over the most it can measure
        (a quarter cycle a pixel: v4.texture), anisotropy."""
        tex = v4.texture(frame, self.field, NOISE_FLOOR)
        self.texture_map = tex
        hz = self.horizon()
        if self.texture_teacher > 0.0 and not self.out_of_model and hz is not None and 0.0 < hz < 1.0 and self._terrain_map() is not None:
            e_now, _ = self._terrain_blend()
            for k, elev in v4.terrain_lessons(tex, self.field, hz, e_now.ravel(), self.local):
                self.scores["terrain_map_vs_texture"].add(float(e_now.ravel()[k]), elev)
                self.terrain[k] += self.texture_teacher * (1.0 - NOISE_FLOOR / max(float(tex[0, k]), NOISE_FLOOR)) * np.array([1.0, elev])
        cell = self._cell(state.cx, state.cy)
        if not self._in_map(cell):
            return 0.0, 0.0, 0.0
        k = cell[0] * self.field[1] + cell[1]
        most = 0.25 * frame.shape[0]
        self.texture_here = (float(min(1.0, 2.0 * tex[0, k])), float(min(1.0, tex[1, k] / most)), float(tex[2, k]))
        return self.texture_here

    def _model_check(self, items: list) -> int:
        """Is this look inside its model? Each whole detection's residual from
        its class's ground line, over that class's own residual variance (kept
        by _learn_ground), summed over the look's detections: chi-square, one
        degree of freedom a detection; p by Wilson & Hilferty's cube-root
        approximation (1931). Failing the standard 5% test, its world model's
        learning is held this look. Its strangeness sense is 1 - p (uniform
        while it sees what it knows, near 1 outside); with no detection to test,
        0 and the gate open. Returns its multiply-adds."""
        self.ground = _ground_shape(self.ground)
        x, k = 0.0, 0
        for c, conf, x0, y0, x1, y1 in items:
            if int(c) not in GROUND_CLASSES or y1 <= y0 or any(prey_lib.cut_by_frame((c, conf, x0, y0, x1, y1), self.aspect)):
                continue
            row = self.ground[GROUND_CLASSES.index(int(c))]
            fit = self._ground_fit(row)
            if fit is None or row[5] < 3 or row[6] <= 0.0:
                continue
            r = (y1 - y0) - (fit[0] * y1 + fit[1])
            x += r * r / (row[6] / row[5])
            k += 1
        if k == 0:
            self.strangeness, self.out_of_model = 0.0, False
            return 0
        v = 2.0 / (9.0 * k)
        z = ((x / k) ** (1.0 / 3.0) - (1.0 - v)) / math.sqrt(v)
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
        self.strangeness, self.out_of_model = float(1.0 - p), p < 0.05
        self.model_checks += 1
        self.model_held += int(self.out_of_model)
        return 4 * k

    def _learn_terrain_from_flow(self, sig, small: tuple, roll: float) -> None:
        """Its terrain taught by its own ground flow (a 2026-09-29 panel; Gibson,
        Longuet-Higgins & Prazdny 1980, Friston): on a frame where it moves
        straight ahead -- a significant expansion, no shift of a pixel, no
        significant roll -- a ground point at row y (depth (1 - e) / (y -
        horizon) eye-heights on ground raised e) flows out of the focus (the
        horizon's middle) by its speed / depth of its distance a frame. Its
        speed is read off the ground's own flow through its map (the median over
        the frame's ground corners, at least 8, ego-motion's own minimum: where
        most of its ground agrees with its map), not from the whole-frame fit,
        which on a ground plane locks onto the near rows and reads ~35% fast
        (tested on rendered ground). Each corner that moved with the
        world, below its horizon, where that is at least a pixel, is a lesson:
        flowing faster than its map's flat ground says, the ground there is
        nearer -- raised by 1 - predicted / measured camera heights, the
        detector's lesson's own formula. Weighted flow_teacher x its share of
        signal (1 - one pixel / how far it moved); scored before it learns."""
        w_px, h_px = small
        if (self.cam_scale == 0.0 or roll != 0.0
                or abs(self.cam_shift[0]) * w_px >= 1.0 or abs(self.cam_shift[1]) * h_px >= 1.0):
            return
        try:
            votes = sig["frame_votes"]
        except (KeyError, IndexError, TypeError):
            return
        hz = self.horizon()
        if votes is None or not len(votes) or hz is None or not 0.0 < hz < 1.0 or self._terrain_map() is None:
            return
        v = np.asarray(votes, dtype=float)
        if v.shape[1] < 5:
            return
        # its tracker's window (Lucas-Kanade's default, 21 px) must stay inside
        # the frame, then and now: past an edge it reads fast flow short
        mx, my = LK_HALF_WINDOW / w_px, LK_HALF_WINDOW / h_px
        inside = lambda x, y: (x > mx) & (x < 1.0 - mx) & (y < 1.0 - my)  # noqa: E731
        v = v[(v[:, 2] < 0) & (v[:, 1] > hz) & inside(v[:, 0], v[:, 1]) & inside(v[:, 3], v[:, 4])]
        if not len(v):
            return
        r0 = np.column_stack([(v[:, 0] - 0.5) * self.aspect, v[:, 1] - hz])   # from the focus, in frame heights
        r1 = np.column_stack([(v[:, 3] - 0.5) * self.aspect, v[:, 4] - hz])
        n0 = np.hypot(*r0.T)
        rho = (r1 * r0).sum(axis=1) / np.maximum(n0 ** 2, 1e-12) - 1.0             # its measured outflow a frame
        e_now, _ = self._terrain_blend()
        cols = self.place.shape[1]
        rr = np.clip((v[:, 1] * self.place.shape[0]).astype(int), 0, self.place.shape[0] - 1)
        cc = np.clip((v[:, 0] * cols).astype(int), 0, cols - 1)
        e_map = np.clip(e_now[rr, cc], -0.9, 0.9)
        k = rho * (1.0 - e_map) / (v[:, 1] - hz)       # each corner's reading of its speed (eye-heights a frame), through its map
        k = k[np.sign(k) == np.sign(self.cam_scale)]
        if len(k) < 8:
            return
        speed = float(np.median(k))
        rho0 = speed * (v[:, 1] - hz)                   # flat ground's outflow a frame at this speed
        moved = np.hypot(*(r1 - r0).T) * h_px
        ok = (np.abs(rho0) * n0 * h_px >= 1.0) & (rho * rho0 > 0.0) & (moved >= 1.0)
        if not ok.any():
            return
        for x, y, r, r_0, d in zip(v[ok, 0], v[ok, 1], rho[ok], rho0[ok], moved[ok]):
            cell = self._cell(x, y)
            if not self._in_map(cell):
                continue
            elev = float(np.clip(1.0 - r_0 / r, -1.0, 1.0))
            self.scores["terrain_map_vs_flow"].add(float(e_now[cell]), elev)
            self.terrain[cell[0] * cols + cell[1]] += self.flow_teacher * (1.0 - 1.0 / d) * np.array([1.0, elev])

    def _terrain_map(self):
        """Its terrain sums at the field's grid ((rows x cols, 2): weight, weighted elevation); None without a field."""
        if self.place is None:
            return None
        n = self.place.size
        if self.terrain is None or self.terrain.shape != (n, 2):
            self.terrain = np.zeros((n, 2))
        return self.terrain.reshape(self.place.shape + (2,))

    def _terrain_blend(self):
        """Its terrain as it sees it: the ground is continuous (Grimson's
        surface interpolation from sparse depth; Marr's 2.5D sketch), so each
        cell's estimate blends its neighbours' evidence (a Gaussian one cell
        wide: the map's own resolution), shrunk toward the flat plane where
        evidence is thin -- one measurement alone counts as it did in its cell.
        Evidence outside the field is none (zeros, not mirrored)."""
        t = self._terrain_map()
        if t is None:
            return None, None
        bw = cv2.GaussianBlur(t[..., 0].astype(np.float32), (0, 0), 1.0, borderType=cv2.BORDER_CONSTANT)
        bs = cv2.GaussianBlur(t[..., 1].astype(np.float32), (0, 0), 1.0, borderType=cv2.BORDER_CONSTANT)
        k0 = 1.0 / (2.0 * math.pi)  # the kernel's own centre: one measurement's weight in its cell
        return bs / np.maximum(k0, bw), bw

    def terrain_at(self, cx: float, cy: float) -> float:
        """The ground's rise (+) or fall (-) at a point, in camera heights: 0 where it has no evidence near."""
        e, _ = self._terrain_blend()
        cell = self._cell(cx, cy)
        if e is None or not self._in_map(cell):
            return 0.0
        return float(e[cell])

    def terrain_view(self):
        """Its terrain as it sees it, per cell (rise in camera heights, evidence nearby), for the viewer."""
        e, bw = self._terrain_blend()
        if e is None:
            return None
        return [[round(float(v), 3), round(float(w), 2)] if w > 1e-3 else None for v, w in zip(e.ravel(), bw.ravel())]

    def ground_fits(self) -> dict:
        """Each measuring class's line (height = a y + b), for the viewer."""
        self.ground = _ground_shape(self.ground)
        return {int(c): [round(float(v), 5) for v in f] for c, row in zip(GROUND_CLASSES, self.ground)
                if (f := self._ground_fit(row)) is not None}

    @staticmethod
    def _ground_fit(row):
        """A class's line height = a y + b, or None until it has three members spread in depth."""
        n, sy, sh, syy, syh = row[:5]
        if n < 3:
            return None
        vy = syy / n - (sy / n) ** 2
        if vy <= 1e-6:
            return None
        a = (syh / n - (sy / n) * (sh / n)) / vy
        return (a, sh / n - a * sy / n) if a > 0 else None

    def horizon(self) -> float | None:
        """Where its ground plane meets the sky (0 = top of the frame): each
        class's line height = a (y - horizon) gives a horizon, and they combine
        by inverse variance -- the textbook calibration variance of a line's
        zero, (mse / a^2)(1/n + (mean y - horizon)^2 / (n var y)), plus the
        classes' spread between them (random effects) -- so a class whose
        members fit their line badly counts for little, and a class that
        disagrees with the rest cannot outweigh them. None until a
        class has spread in where its members stand and a few residuals."""
        self.ground = _ground_shape(self.ground)
        est, var = [], []
        for row in self.ground:
            fit = self._ground_fit(row)
            m, rss = row[5], row[6]
            if fit is None or m < 3:
                continue
            a, b = fit
            n, sy, syy = row[0], row[1], row[3]
            vy = syy / n - (sy / n) ** 2
            hz = -b / a
            est.append(hz)
            var.append((max(rss / m, 1e-8) / (a * a)) * (1.0 / n + (sy / n - hz) ** 2 / (n * vy)))
        if not est:
            return None
        # Random effects (DerSimonian & Laird 1986): classes that disagree more
        # than their own noise explains (a class biased by where it stands --
        # trees on a far bank, signs on poles) share the spread between them,
        # so no one biased class outweighs the others that agree.
        th, w = np.array(est), 1.0 / np.array(var)
        fixed = float((w * th).sum() / w.sum())
        q, c = float((w * (th - fixed) ** 2).sum()), float(w.sum() - (w * w).sum() / w.sum())
        tau2 = max(0.0, (q - (len(th) - 1)) / c) if c > 0 else 0.0
        w = 1.0 / (np.array(var) + tau2)
        est, wsum = float((w * th).sum()), float(w.sum())
        self.horizon_precision = wsum  # how sure its ground model is (1 / the pooled horizon's variance)
        return float(np.clip(est / wsum, -1.0, 1.0)) if wsum else None

    def _ground_sense(self, cy: float) -> tuple[float, float]:
        """(how near the ground at its gaze is: 1 at the frame's bottom, 0 at
        the horizon and above; where the horizon is, relative to the middle)."""
        h = self.horizon()
        if h is None or h >= 1.0:
            return 0.0, 0.0
        return float(np.clip((cy - h) / (1.0 - h), 0.0, 1.0)), float(np.clip(h - 0.5, -1.0, 1.0))

    def _nearness(self, cx: float, cy: float, boxes: list) -> float:
        """How near what it looks at is (a 2026-09-29 panel: it knows where it
        looks -- its gaze is among its inputs, an efference copy -- but not how
        far). Relative, as a monocular eye's must be: 1 at the frame's bottom
        edge's ground, 0 at the horizon. What it looks at is the nearest thing
        whose box holds its gaze, read from its feet (where its base meets the
        ground), else the ground itself at its gaze; either way on its terrain
        (ground raised by e camera heights puts a base nearer: / (1 - e))."""
        h = self.horizon()
        if h is None or h >= 1.0:
            return 0.0
        held = [b for b in boxes if b[2] <= cx <= b[4] and b[3] <= cy <= b[5]]
        if held:
            b = max(held, key=lambda b: b[5])   # the nearest: its feet lowest in the frame
            x, y = (b[2] + b[4]) / 2, min(b[5], 1.0 - 1e-6)
        else:
            x, y = cx, cy
        if y <= h:
            return 0.0  # the sky, or a thing standing beyond its horizon's reach
        e = float(np.clip(self.terrain_at(x, y), -0.9, 0.9))
        return float(np.clip((y - h) / (1.0 - h) / (1.0 - e), 0.0, 1.0))

    def _extrapolate(self, boxes: list, vx: float, vy: float) -> list:
        """Where the host it follows will be by the time it acts (a 2026-09-29
        panel; Nijhawan): its box carried forward by its velocity x its own lag
        (its look interval) x its inherited extrapolation. And the host-position
        score: where it said the host would be, and where the host was left
        (the plain baseline), against where the next detection puts it."""
        followed = self.followed[0] if self.followed is not None else None
        if self._predicted is not None and followed is not None and followed != self._predicted[0]:
            old, said, was = self._predicted
            now = ((followed[0] + followed[2]) / 2, (followed[1] + followed[3]) / 2)
            if min(followed[2], old[2]) > max(followed[0], old[0]) and min(followed[3], old[3]) > max(followed[1], old[1]):
                self.scores["host_position_plain"].add(math.dist(was, now), 0.0)
                if self.extrapolation > 0.0:
                    self.scores["host_position"].add(math.dist(said, now), 0.0)
            self._predicted = None
        if followed is None or self.prey_level < 3:
            self.last_ahead = None
            return boxes
        lag = self.last_interval / max(1.0, self.fps)
        dx, dy = vx * lag * self.extrapolation, vy * lag * self.extrapolation
        centre = ((followed[0] + followed[2]) / 2, (followed[1] + followed[3]) / 2)
        if self._predicted is None:
            self._predicted = (followed, (centre[0] + dx, centre[1] + dy), centre)
        if not (dx or dy):
            self.last_ahead = None
            return boxes
        self.last_ahead = [round(followed[0] + dx, 4), round(followed[1] + dy, 4), round(followed[2] + dx, 4), round(followed[3] + dy, 4)]
        return [[b[0], b[1], b[2] + dx, b[3] + dy, b[4] + dx, b[5] + dy] if tuple(b[2:6]) == followed else b for b in boxes]

    def _contact(self, frame, state) -> tuple[float, int]:
        """How soon what it looks at will reach it (Lee's tau): corners tracked
        where it looks (attention: an eye tracks more where it looks), from its
        last look to this one, fitted on their own -- their expansion over one
        look is its look interval / tau, whoever is moving. Counted when
        significant (beyond twice its standard error) and replicated by the look
        before, as its ego-motion is; clipped to [0, 1] (1 = contact within one
        look). Returns (it, multiply-adds)."""
        small = cv2.resize(frame, shift_size(frame.shape), interpolation=cv2.INTER_AREA).astype(np.uint8)
        prev, self._tau_prev = self._tau_prev, (small, state.cx, state.cy)
        if prev is None or prev[0].shape != small.shape:
            return 0.0, 0
        h, w = small.shape
        hx, hy = state.half_extents(self.aspect)
        mask = np.zeros_like(small)
        mask[int(max(0.0, prev[2] - hy) * h):int(min(1.0, prev[2] + hy) * h) + 1,
             int(max(0.0, prev[1] - hx) * w):int(min(1.0, prev[1] + hx) * w) + 1] = 255
        if self.local is not None and self.local.any():  # what rides with it can't approach it
            mask[cv2.resize(self.local.reshape(self.field).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0] = 0
        p0 = cv2.goodFeaturesToTrack(prev[0], maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
        if p0 is None or len(p0) < 8:
            self._last_local_s = 0.0
            return 0.0, 0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev[0], small, p0, None)
        ok = st.ravel() == 1
        macs = 12 * int(ok.sum())  # a least-squares similarity: a dozen multiply-adds a point (tracking is its eye's own work)
        fit = _similarity(p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)) if ok.sum() >= 8 else None
        s = fit[0] if fit is not None and fit[0] > 2.0 * fit[1] else 0.0
        s_rep, self._last_local_s = replicated(s, self._last_local_s), s
        self.contact = float(np.clip(s_rep, 0.0, 1.0))
        return self.contact, macs

    def _host_velocity(self, boxes: list) -> tuple[float, float]:
        """The velocity of the host it follows (frame fractions per second,
        clipped to +-1): its preferred host now, matched to the one it followed
        by overlapping boxes; measured when the detector's box moves, held in
        between (detections come every fraction of a second to seconds)."""
        if not boxes:
            self.followed = None
            return 0.0, 0.0
        w = (lambda c: self.host_pref.get(int(c), 1.0)) if self.host_pref else (lambda c: 1.0)
        best = max(boxes, key=lambda b: w(b[0]) * b[1] * (b[4] - b[2]) * (b[5] - b[3]))
        box = tuple(best[2:6])
        if self.followed is not None:
            old, t, v = self.followed
            overlap = min(box[2], old[2]) > max(box[0], old[0]) and min(box[3], old[3]) > max(box[1], old[1])
            if overlap and box == old:
                return v
            if overlap and self.lived_s > t:
                dt = self.lived_s - t
                v = (float(np.clip(((box[0] + box[2]) - (old[0] + old[2])) / 2 / dt, -1, 1)),
                     float(np.clip(((box[1] + box[3]) - (old[1] + old[3])) / 2 / dt, -1, 1)))
                self.followed = (box, self.lived_s, v)
                return v
        self.followed = (box, self.lived_s, (0.0, 0.0))
        return 0.0, 0.0

    def _distill(self, i: int, value: float) -> None:
        """One replayed episode, better than its usual (advantage > 0), makes
        what it did then more decisive: each output is pulled toward the full
        commitment of what it did (its sign) -- habits forming (Graybiel). The
        delta rule on the output layer, scaled by plasticity x advantage; it
        stops by itself as an output saturates, so it can't run away."""
        act = self.episode_acts[i] if i < len(self.episode_acts) else None
        adv = value - self.mean_reward
        b = self.brain
        if act is None or adv <= 0.0:
            return
        top, did = act
        if top.shape[0] != b.weights_ho.shape[1] or did.shape[0] != b.weights_ho.shape[0]:
            return  # its brain has changed shape since (an adoption): that moment no longer fits
        now = np.tanh(b.bias_o + b.weights_ho @ top)
        delta = (np.sign(did) - now) * (1.0 - now * now) * self.plasticity * min(1.0, adv)
        dw = np.outer(delta, top)
        if self.maturation > 0.0:
            shape = (b.weights_ho.shape[0], b.weights_ho.shape[1] + 1)
            if self.lessons is None or self.lessons.shape != shape:
                self.lessons = np.zeros(shape)
            lock = self._lock()
            dw *= 1.0 - lock[:, :-1]
            delta = delta * (1.0 - lock[:, -1])
            # a good lesson, credited to the synapses it moved (in proportion)
            moved = np.abs(np.hstack([dw, delta[:, None]]))
            self.lessons += min(1.0, adv) * moved / max(1e-12, float(moved.max()))
        self._tonight += 1
        b.weights_ho += dw
        b.bias_o += delta
        self.distilled += 1
        self.distill_macs += 3 * b.weights_ho.size

    def lessons_per_night(self) -> float | None:
        """Its lessons per night, over its last `maturation` nights (the gene
        sets its own horizon); None before its first whole night."""
        if not self.night_lessons:
            return None
        return float(np.mean(self.night_lessons[-max(1, math.ceil(self.maturation)):]))

    def _lock(self):
        """Each output synapse's lock: lessons / (lessons + maturation's nights
        in lessons). Nothing locks before a night has been measured."""
        per_night = self.lessons_per_night()
        if self.lessons is None or self.maturation <= 0.0 or not per_night:
            return np.zeros_like(self.lessons) if self.lessons is not None else None
        k = self.maturation * per_night
        return self.lessons / (self.lessons + k)

    def locked_share(self) -> float:
        """How much of its distilled brain has matured (mean lock over its output synapses)."""
        lock = self._lock()
        return 0.0 if lock is None else float(np.mean(lock))

    def _act_now(self):
        b = self.brain
        return None if b.last_top is None else (np.array(b.last_top, copy=True), np.array(b.last_outputs, copy=True))

    def _remember(self, episode, priority: float, meta) -> None:
        """A new episode. Its memory holds about one episode per Kenyon cell (a
        sparse associative memory holds at least as many patterns as it has
        cells); when full, the least surprising one gives way (prioritized
        consolidation: Mattar & Daw 2018) -- the new one takes its slot.
        2026-09-28 audit: the list used to grow for the whole life."""
        cap = max(1, self.live_kc or self.mb.n_kc)
        while len(self.episode_meta) < len(self.episodes):
            self.episode_meta.append(None)
        while len(self.episode_acts) < len(self.episodes):
            self.episode_acts.append(None)
        act = self._act_now() if self.plasticity > 0.0 else None
        self.mean_reward += (episode[1] - self.mean_reward) / min(len(self.episodes) + 1, cap)
        if len(self.episodes) < cap:
            self.episodes.append(episode)
            self.priority.append(priority)
            self.episode_meta.append(meta)
            self.episode_acts.append(act)
            return
        j = int(np.argmin(self.priority))
        self.episodes[j], self.priority[j], self.episode_meta[j], self.episode_acts[j] = episode, priority, meta, act
        self._stale += 1
        if self._index_of is self.episodes:  # the recall index: the slot's new cells (its old ones go stale)
            for k in episode[0].tolist():
                self._index.setdefault(k, []).append(j)

    def _recall(self, code) -> tuple[int | None, int]:
        """Pattern completion: the episode whose code overlaps this one most,
        if more than two random codes of their sizes would (chance), the most
        recent of equals; and the multiply-adds it took (a stored cell
        compared, or indexed, is one)."""
        eps = self.episodes
        if self._index_of is not eps or self._indexed > len(eps) or self._stale > len(eps) // 8:
            self._stale = 0
            self._index, self._indexed, self._index_of = {}, 0, eps  # a new life's (or trimmed) episodes
        macs = 0
        for e in range(self._indexed, len(eps)):
            cells = eps[e][0].tolist()
            macs += len(cells)
            for k in cells:
                self._index.setdefault(k, []).append(e)
        self._indexed = len(eps)
        posts = [self._index[k] for k in code.tolist() if k in self._index]
        if not posts:
            return None, macs
        macs += sum(len(p) for p in posts)
        hits = np.bincount(np.concatenate(posts), minlength=len(eps))
        best = len(hits) - 1 - int(np.argmax(hits[::-1]))
        chance = len(code) * len(eps[best][0]) / max(1, self.mb.n_kc)
        return (best if hits[best] > chance else None), macs

    def _vote_frames(self, sig) -> None:
        """This frame's corners vote on its frames of reference (see __init__)."""
        try:
            votes = sig["frame_votes"]
        except (KeyError, IndexError, TypeError):
            return
        rows, cols = self.field
        n = rows * cols
        if self.frame_map is None or self.frame_map.shape != (2, n):
            self.frame_map = np.zeros((2, n))
        self.frame_map *= math.exp(-1.0 / max(1.0, self.fps) / MISMATCH_TAU_S)
        if votes is None or not len(votes):
            return
        v = np.asarray(votes, dtype=float)
        k = np.clip((v[:, 1] * rows).astype(int), 0, rows - 1) * cols + np.clip((v[:, 0] * cols).astype(int), 0, cols - 1)
        np.add.at(self.frame_map[0], k[v[:, 2] > 0], 1.0)
        np.add.at(self.frame_map[1], k[v[:, 2] < 0], 1.0)

    def local_frame(self):
        """Its local-frame cells (bool per field cell), or None before it has
        moved: a cell's still votes beat half of its votes at the standard 5%
        test (a binomial share against 1/2, normal approximation)."""
        if self.frame_map is None or not getattr(self, "field", None) or self.frame_map.shape[1] != self.field[0] * self.field[1]:
            return None
        still, world = self.frame_map
        return (still - world) > 1.96 * np.sqrt(still + world)

    def _cell(self, x: float, y: float):
        """The whole field's grid cell (row, col) at a point of the frame."""
        if self.place is None:
            return None
        rows, cols = self.place.shape
        return (min(rows - 1, max(0, int(y * rows))), min(cols - 1, max(0, int(x * cols))))

    def _place_sense(self, cx: float, cy: float) -> tuple[float, float, float]:
        """Direction from its gaze to the spot worth most, and how much: what
        has fed it there plus what its dreams learned it leads to (the value
        map; all zero until it dreams, so the sense is unchanged until then)."""
        if self.place is None:
            return 0.0, 0.0, 0.0
        worth = self.place + self.value_map if self.value_map is not None else self.place
        if not np.any(worth > 0.0):
            return 0.0, 0.0, 0.0
        rows, cols = worth.shape
        r, c = np.unravel_index(int(np.argmax(worth)), worth.shape)
        return (c + 0.5) / cols - cx, (r + 0.5) / rows - cy, float(np.clip(worth[r, c], -1.0, 1.0))

    def _dream(self) -> int:
        """Closed-loop dreaming (a 2026-09-28 panel; Hobson & Friston 2012;
        Pfeiffer & Foster 2013's preplay; Sutton's Dyna): eyes shut, its brain
        steers an imagined gaze through the real eye physics, its senses fed
        by its own maps -- its internal representation standing in for the
        world. Along the imagined path it learns a value map by the same
        backup as sequence replay: a place is worth its food plus backup x
        what the path reached next. Brain steps cost as thinking does and
        count against the look's deadline. Returns the multiply-adds spent."""
        steps = self.dream_steps
        if not steps or self.place is None or self.value_map is None or self.replay_backup <= 0.0 or self.learning_rate <= 0.0:
            return 0
        brain, body = self.brain, self.body
        st = fovea.FoveaState(cx=self.state.cx, cy=self.state.cy, n=self.state.n)
        cell = self._cell(st.cx, st.cy)
        self.seq += 1
        self.dreams += 1
        self._log_replay("dream", cell, self.seq)
        for _ in range(steps):
            pdx, pdy, pval = self._place_sense(st.cx, st.cy)
            here = float(self.place[cell] + self.value_map[cell]) if cell is not None else 0.0
            out = brain.step(0.0, 0.0, 0.0, 0.0, 0.0, st.cx, st.cy, st.extent, body, 0.0, 0.0, st.vx, st.vy,
                             self.prev_response, 0.0, 0.0, 0.0, 0.0, here, pdx, pdy, pval, 0.0, 0.0)
            for _ in range(max(1, self.pace)):
                st, _, _ = fovea.step(st, out.pan, out.tilt)
            nxt = self._cell(st.cx, st.cy)
            if cell is not None and nxt is not None:
                target = self.place[cell] + self.replay_backup * (self.place[nxt] + self.value_map[nxt])
                self.value_map[cell] += self.learning_rate * (target - self.place[cell] - self.value_map[cell])
                self._log_replay("dream", nxt, self.seq)
            cell = nxt
        return steps * brain.macs()

    def _intruder(self, boxes: list, daylight: float, seconds: float) -> float:
        """A person where, at this time of day, people haven't been lately:
        confidence x (1 - how expected people are there). Then learns what
        this look showed (per light phase, over ~LIGHT_SLOW_S)."""
        if self.people_day is None:
            return 0.0
        people = [b for b in boxes if int(b[0]) == prey_lib.PERSON_CLASS]
        expected = daylight * self.people_day + (1.0 - daylight) * self.people_night
        surprise = 0.0
        seen = np.zeros(self.people_day.shape)
        rows, cols = seen.shape
        for _, conf, x0, y0, x1, y1 in people:
            r, c = self._cell((x0 + x1) / 2, (y0 + y1) / 2)
            surprise = max(surprise, float(conf) * (1.0 - float(expected[r, c])))
            seen[int(y0 * rows):max(int(y0 * rows) + 1, int(np.ceil(y1 * rows))),
                 int(x0 * cols):max(int(x0 * cols) + 1, int(np.ceil(x1 * cols)))] = 1.0
        rate = 1.0 - math.exp(-max(0.0, seconds) / LIGHT_SLOW_S)
        self.people_day += rate * daylight * (seen - self.people_day)
        self.people_night += rate * (1.0 - daylight) * (seen - self.people_night)
        return float(np.clip(surprise, 0.0, 1.0))

    def _replay(self, asleep_settled: bool, quiet: bool) -> int:
        """Re-learning from this life's episodes; returns the multiply-adds spent."""
        count = self.sleep_replay if asleep_settled else self.awake_replay if quiet else 0
        self.distill_macs = 0
        if asleep_settled and not self._night:  # the night begins: reconsolidation loosens its locks
            self._night = True
            if self.lessons is not None:
                self.lessons *= 1.0 - float(np.clip(self.uncertainty, 0.0, 1.0))
        elif not asleep_settled and not quiet:
            if self._night:  # the night ends: what it distilled counts toward its nights
                self.night_lessons.append(self._tonight)
                del self.night_lessons[:-max(1, min(int(MATURATION_KEEP), math.ceil(max(1.0, self.maturation))))]
                self._tonight = 0
            self._night = False
        if not count or not self.episodes or self.learning_rate <= 0.0 or not self.mb.n_kc:
            return 0
        rng = np.random.default_rng(self.k)  # reproducible: the same life replays the same way
        for _ in range(count):
            if asleep_settled and rng.random() < self.rem_share:
                # REM: two experiences recombined, at their mean reward (generalize)
                i, j = rng.integers(len(self.episodes), size=2)
                (ci, ri, cell_i), (cj, rj, cell_j) = self.episodes[i], self.episodes[j]
                half = max(1, len(ci) // 2)
                code = np.unique(np.concatenate([ci[:half], cj[half:]]))
                self.last_replay = ("rem", code, self.lived_s, None)  # recombined: no one moment it saw
                self.mb.learn(code, 0.5 * (ri + rj), self.learning_rate)
                self.replays["rem"] += 1
                self.seq += 1
                self._log_replay("rem", cell_i, self.seq)
                self._log_replay("rem", cell_j, self.seq)
            else:
                # NREM, or awake: the biggest surprise first (prioritized replay).
                # With a backup (genome.replay_backup, born 0), a replay goes on
                # as a path: each next step is the moment BEFORE the last one,
                # which learns its own reward plus backup x the value of the
                # moment that followed it -- reverse replay after reward
                # (Foster & Wilson 2006) handing a meal's value back along the
                # route that led to it. At 0 it is exactly the old replay.
                if self.replay_backup > 0.0 and self.chain is not None and self.chain[0] >= 0:
                    i, seq = self.chain
                else:
                    i = int(np.argmax(self.priority))
                    self.seq += 1
                    seq = self.seq
                code, reward, cell = self.episodes[i]
                self.last_replay = ("nrem" if asleep_settled else "awake", code, self.lived_s,
                                    self.episode_meta[i] if i < len(self.episode_meta) else None)
                target = reward
                if self.replay_backup > 0.0 and i + 1 < len(self.episodes):
                    target = reward + self.replay_backup * self.mb.value(self.episodes[i + 1][0])
                self.mb.learn(code, target, self.learning_rate)
                self.priority[i] = abs(target - self.mb.value(code))
                if self._in_map(cell):  # (a memory from a differently shaped world touches no map)
                    self.place[cell] += self.learning_rate * (target - self.place[cell])
                self.chain = (i - 1, seq) if self.replay_backup > 0.0 else None
                if asleep_settled and self.plasticity > 0.0:
                    self._distill(i, target)
                self.replays["nrem" if asleep_settled else "awake"] += 1
                self._log_replay("nrem" if asleep_settled else "awake", cell, seq)
        return count * kc_macs(self.live_kc) + self.distill_macs

    def _in_map(self, cell) -> bool:
        """A field cell that exists on its maps now (memories formed on a
        stream of another shape carry cells that may not)."""
        return cell is not None and self.place is not None and 0 <= cell[0] < self.place.shape[0] and 0 <= cell[1] < self.place.shape[1]

    def _log_replay(self, kind: str, cell, seq: int = 0) -> None:
        if self._in_map(cell):
            self.replay_log.append((kind, int(cell[0]), int(cell[1]), self.lived_s, seq))
            del self.replay_log[:-64]

    def _substep(self, shift: tuple, in_world: bool) -> None:
        """The eye moves one frame on (muscle energy = force squared, per frame pushed)."""
        p = self.pending
        state, force_x, force_y = fovea.step(self.state, p["pan"], p["tilt"])
        if self.stab > 0.0 and not p["asleep"] and in_world:
            nx = float(np.clip(state.cx + self.stab * shift[0], 0.0, 1.0))
            ny = float(np.clip(state.cy + self.stab * shift[1], 0.0, 1.0))
            self.stab_dx, self.stab_dy = self.stab_dx + nx - state.cx, self.stab_dy + ny - state.cy
            state = fovea.FoveaState(cx=nx, cy=ny, n=state.n, vx=state.vx, vy=state.vy, mag=state.mag)
        self.state = state
        # Muscle, force squared per frame pushed -- and a warning is made like a
        # movement (a 2026-09-28 panel: every real signal costs its sender;
        # Sherman, Zahavi): its alarm output, while positive, costs the same.
        p["effort"] += force_x * force_x + force_y * force_y + p.get("alarm", 0.0) ** 2
        p["force"] = (force_x, force_y)
        p["done"] += 1

    def _close(self) -> None:
        """A gaze interval is over: the body pays for it and eats what it caught."""
        p, body, brain, rec = self.pending, self.body, self.brain, self.rec
        self.prev_dx, self.prev_dy = self.state.cx - p["cx0"], self.state.cy - p["cy0"]
        if rec is not None:
            rec["dxs"].append(self.prev_dx)
            rec["dys"].append(self.prev_dy)
            rec["movement_costs"].append(math.hypot(*p["force"]))
            rec["periph_active"].append(p["periph_motion"])
        asleep = p["asleep"]
        # only living receptors cost; a slow receptor costs less (1 / (1 + slowness))
        gaze_cost = 0.0 if asleep else _receptor_cost(self.live_n, self.quota_pct) / (1.0 + self.slowness)
        body.update(p["periph_motion"], p["loom"], p["effort"],
                    gaze_cost + (THINK_COST * brain.think_factor() + CONE_COST * self.cones * self.cones * p["colour_on"] * (2 if self.v4_colour is not None else 1)
                                 + CHANNEL_COST * brain.loop_synapses()
                                 + (0.0 if asleep else STABILIZER_COST * self.stab)
                                 + (THINK_COST * (kc_macs(self.live_kc) + (self.live_kc if self.aversive_rate > 0.0 else 0))
                                    / REFERENCE_MACS if p["kc_on"] else 0.0)  # + the aversive output neuron, once it exists
                                 + PREY_SENSE_COST * brain.prey_synapses(self.prey_level)
                                 + PREY_SENSE_COST * brain.sense_synapses(PLACE_INPUTS + (INTRUDER_INPUT, DANGER_INPUT) + MISMATCH_INPUTS + RECALL_INPUTS
                                                                         + (UNCERTAINTY_INPUT,) + GROUND_INPUTS + PARALLAX_INPUTS + ARCHETYPE_INPUTS
                                                                         + COLLICULUS_INPUTS + (TERRAIN_INPUT, NEARNESS_INPUT, FELT_NEARNESS_INPUT, CONTACT_INPUT, TURN_INPUT, TILT_INPUT) + HEADING_INPUTS
                                                                         + (EGO_SPEED_INPUT, ACCELERATION_INPUT, PLACE_VALUE_INPUT, RIDING_INPUT, STRANGENESS_INPUT) + TEXTURE_INPUTS)
                                 + (PREY_SENSE_COST * brain.sense_synapses(PLANT_INPUTS[:1 if self.plant_level == 1 else 3])
                                    if self.plant_level else 0.0)
                                 + THINK_COST * (p["replay_macs"] + p.get("proto_macs", 0)
                                                 + (0 if asleep else self.tree_macs)) / REFERENCE_MACS) * self.scarcity,
                    dt=p["interval"], pace=p["interval"], dt_seconds=p["interval"] / max(1.0, self.fps),
                    field_light=p["field_light"])
        cleared = body.take_cleared()
        if cleared > 0.0:
            floor = NOISE_FLOOR ** 2
            self.variance[...] = floor + (self.variance - floor) * math.exp(-CONSOLIDATE_RATE * cleared)
            if self.place is not None:  # NREM: the place map scaled back down (synaptic homeostasis)
                self.place *= math.exp(-CONSOLIDATE_RATE * cleared)
        body.feed_visual_sustenance(p["snack"])
        seconds = p["interval"] / max(1.0, self.fps)
        body.feed_host(p["prey_now"], seconds)  # a flow over the look's interval
        # Plants refill their nectar over hours; a sip draws its plant down.
        refill = 1.0 - math.exp(-seconds / NECTAR_REFILL_S)
        for k in self.nectar:
            self.nectar[k] += refill * (1.0 - self.nectar[k])
        if p.get("sip"):
            crop = self.nectar.get(p["sip"], 1.0)
            took = body.feed_nectar(body.flow * crop * seconds / LEGACY_UNIT)
            self.nectar[p["sip"]] = max(0.0, crop - took * LEGACY_UNIT / (NECTAR_CROP * GUT_CAP))
            if took > 0:
                self.sips += 1
        if rec is not None:
            rec["prey_eaten"].append(p["prey_now"])
            rec["foods"].append(p["snack"])
            rec["energies"].append(body.energy)
            rec["drives"].append(body.drive())
            rec["asleeps"].append(body.asleep)
        self.pending = None
