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

import math

import cv2
import numpy as np

from . import fovea, prey as prey_lib
from .controller import DANGER_INPUT, INTRUDER_INPUT, PLACE_INPUTS, REFERENCE_MACS
from .retina import field_shape
from .genome import RETINA_PLANES
from .mushroom import MushroomBody, macs as kc_macs
from .state import (FOOD_PER_LOOK, LEGACY_UNIT, LIGHT_SLOW_S, PREY_FOOD_PER_LOOK, SLEEP_SETTLE_S, TEMPO_SHARE,
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


def global_shifts(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The whole frame's shift from frame k-1 to frame k, for a recording."""
    n = len(frames)
    sx, sy = np.zeros(n), np.zeros(n)
    if n < 2:
        return sx, sy
    size = shift_size(frames[0].shape)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    prev = cv2.resize(frames[0], size, interpolation=cv2.INTER_AREA).astype(np.float32)
    for k in range(1, n):
        cur = cv2.resize(frames[k], size, interpolation=cv2.INTER_AREA).astype(np.float32)
        sx[k], sy[k] = global_shift(prev, cur, window)
        prev = cur
    return sx, sy


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


# What a host passes each frame about the whole field (sig): the keys below.
SIGNAL_KEYS = ("expansion", "motion_energy", "motion_cx", "motion_cy", "field_light")


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
        self.brain = g.brain
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
        self.mb = MushroomBody(int(getattr(g, "kc", 0)), int(getattr(g, "kc_seed", 0)), learned, danger)
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
        self.swats = 0
        self.swat_frames: list = []
        self.food_value = 0.0
        self.value_errors = []
        # Colour vision: how many opponent channels this genome's gaze has
        # (0-2). A plane it doesn't have reads zero.
        self.colour_n = int(getattr(g, "colour_channels", 0)) if colour else 0
        self.cones = int(min(getattr(g, "cones", self.state.n), self.state.n))
        self.last_colour = None
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
        # Place map and the people-expectation maps (day, night): memory,
        # carried like the rest (items 4-6 of memory); sized on the first frame.
        extra = list(memory[3:6]) if memory is not None and len(memory) > 5 else [None, None, None]
        self.place, self.people_day, self.people_night = (None if a is None else np.array(a, dtype=float) for a in extra)
        self.episodes: list = []       # this life's (Kenyon-cell code, reward, field cell), for replay
        self.priority: list = []       # each episode's last prediction error (replay order)
        self.replays = {"awake": 0, "nrem": 0, "rem": 0}
        self.last_kc = np.zeros(0, dtype=int)  # the Kenyon cells firing at its latest look (for the viewer)
        self.replay_log: list = []             # recent replays: (kind, field row, col, frame), for the viewer's dreams
        self.last_replay = None                # (kind, Kenyon-cell code, seconds lived): its latest replayed memory, for the viewer
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
        if self.k == 0:
            self.aspect = frame.shape[1] / max(1, frame.shape[0])
            self.frame_h = frame.shape[0]
            self.field = field_shape(frame.shape[0], frame.shape[1])
            for name in ("place", "people_day", "people_night", "value_map"):
                a = getattr(self, name)
                if a is None or a.shape != self.field:
                    setattr(self, name, np.zeros(self.field))
        if self.slowness > 0.0:  # slow photoreceptors: its gaze sees the frames low-passed
            a = 1.0 - math.exp(-REFERENCE_GAZES_PER_S / (max(1.0, self.fps) * self.slowness))
            f = frame.astype(np.float32)
            self.slow = f if self.slow is None or self.slow.shape != f.shape else self.slow + a * (f - self.slow)
            frame = self.slow
        if self.pending is not None:
            self._substep(shift, in_world=True)
            if self.pending["done"] == self.pending["interval"]:
                self._close()
            elif self.rec is not None:
                self.rec["frame_path"].append((self.state.cx, self.state.cy, self.state.extent))
        gazed = self.pending is None
        if gazed:
            self._gaze(frame, sig, boxes or [], colour_frame)
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
    def _gaze(self, frame: np.ndarray, sig, boxes: list, colour_frame) -> None:
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
        scent, prey_dx, prey_dy = prey_sense(boxes, state.cx, state.cy, self.prey_level, self.host_pref)
        # Its mushroom body's learned value of what the look shows (eyes shut: nothing).
        kc_active = self.mb.active(v, n, self.live_kc) if not was_asleep else np.zeros(0, dtype=int)
        self.last_kc = kc_active
        self.food_value = float(np.clip(self.mb.value(kc_active), -1.0, 1.0))
        self.danger_value = float(np.clip(self.mb.danger(kc_active), -1.0, 1.0)) if self.aversive_rate > 0.0 else 0.0
        cell = self._cell(state.cx, state.cy)
        place_dx, place_dy, place_value = self._place_sense(state.cx, state.cy)
        self.intruder = self._intruder(boxes, body.daylight, self.last_interval / max(1.0, self.fps))
        # Replay (awake in a quiet moment, or asleep once settled): it takes
        # time from this look's deadline and costs thinking energy.
        replay_macs = self._replay(was_asleep and body.sleep_clock > SLEEP_SETTLE_S,
                                   not was_asleep and not boxes and not body.big_change(loom, periph_motion, self.vigilance))
        if was_asleep and body.sleep_clock > SLEEP_SETTLE_S:
            replay_macs += self._dream()
        if (self.sec_per_mac and self.last_out is not None
                and self.sec_per_mac * (brain.macs() + replay_macs) > self.last_interval / max(1.0, self.fps)):
            out = self.last_out  # still thinking: this look is missed
            self.missed += 1
        else:
            out = brain.step(
                lum, motion, flow_x, flow_y, loom, state.cx, state.cy, state.extent, body,
                periph_dx, periph_dy, state.vx, state.vy, self.prev_response, field_light, scent, prey_dx, prey_dy,
                self.food_value, place_dx, place_dy, place_value, self.intruder, self.danger_value,
            )
            self.last_out = out
        pan, tilt, alarm, tempo = out.pan, out.tilt, out.alarm, out.tempo
        self.last_alarm = alarm
        # Sleep is its own choice (its sleep output); the body adds only the
        # physiological overrides -- collapse, hunger, a big change (state.py).
        body.set_sleep(out.sleep > 0.0, loom, periph_motion, self.vigilance)
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
        col = fovea.extract_colour(colour_frame, state, colour_on, self.cones) if colour_on else np.zeros(0)
        if colour_on:
            self.last_colour = col
        planes = np.zeros((1, RETINA_PLANES, n, n))
        planes[0, 0], planes[0, 1] = v.reshape(n, n), self.prev_v.reshape(n, n)
        for c in range(len(col) // self.n_cells):
            planes[0, 2 + c] = col[c * self.n_cells:(c + 1) * self.n_cells].reshape(n, n)
        plain = np.concatenate([[self.prev_dx, self.prev_dy], brain.tree_view()])[None, :]
        response = float(g.evaluate("response", plain, planes)[0])
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
            # Lifetime learning: what it ate this look (in meals: a full look at
            # prey = 1) teaches its mushroom body what the look showed.
            reward = (PREY_FOOD_PER_LOOK * prey_now + FOOD_PER_LOOK * snack) / PREY_FOOD_PER_LOOK
            if len(kc_active) and self.learning_rate > 0.0:
                err = self.mb.learn(kc_active, reward, self.learning_rate)
                self.value_errors.append(abs(err))
                self.episodes.append((kc_active.copy(), reward, cell))
                self.priority.append(abs(err))
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
                        "cx0": state.cx, "cy0": state.cy, "periph_motion": periph_motion, "loom": loom,
                        "field_light": field_light, "prey_now": prey_now, "snack": snack}
        self.prev_frame = frame

    def _blind(self, look: np.ndarray, n: int) -> np.ndarray:
        """A wasting eye's outer rings, lost: they read black."""
        if self.live_n >= n:
            return look
        m = (n - self.live_n) // 2
        grid = look.reshape(n, n).copy()
        grid[:m, :] = grid[-m:, :] = grid[:, :m] = grid[:, -m:] = 0.0
        return grid.reshape(-1)

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
                self.last_replay = ("rem", code, self.lived_s)
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
                self.last_replay = ("nrem" if asleep_settled else "awake", code, self.lived_s)
                target = reward
                if self.replay_backup > 0.0 and i + 1 < len(self.episodes):
                    target = reward + self.replay_backup * self.mb.value(self.episodes[i + 1][0])
                self.mb.learn(code, target, self.learning_rate)
                self.priority[i] = abs(target - self.mb.value(code))
                if cell is not None:
                    self.place[cell] += self.learning_rate * (target - self.place[cell])
                self.chain = (i - 1, seq) if self.replay_backup > 0.0 else None
                self.replays["nrem" if asleep_settled else "awake"] += 1
                self._log_replay("nrem" if asleep_settled else "awake", cell, seq)
        return count * kc_macs(self.live_kc)

    def _log_replay(self, kind: str, cell, seq: int = 0) -> None:
        if cell is not None:
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
        p["effort"] += force_x * force_x + force_y * force_y
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
                    gaze_cost + (THINK_COST * brain.think_factor() + CONE_COST * self.cones * self.cones * p["colour_on"]
                                 + CHANNEL_COST * brain.loop_synapses()
                                 + (0.0 if asleep else STABILIZER_COST * self.stab)
                                 + (THINK_COST * (kc_macs(self.live_kc) + (self.live_kc if self.aversive_rate > 0.0 else 0))
                                    / REFERENCE_MACS if p["kc_on"] else 0.0)  # + the aversive output neuron, once it exists
                                 + PREY_SENSE_COST * brain.prey_synapses(self.prey_level)
                                 + PREY_SENSE_COST * brain.sense_synapses(PLACE_INPUTS + (INTRUDER_INPUT, DANGER_INPUT))
                                 + THINK_COST * p["replay_macs"] / REFERENCE_MACS) * self.scarcity,
                    dt=p["interval"], pace=p["interval"], dt_seconds=p["interval"] / max(1.0, self.fps),
                    field_light=p["field_light"])
        cleared = body.take_cleared()
        if cleared > 0.0:
            floor = NOISE_FLOOR ** 2
            self.variance[...] = floor + (self.variance - floor) * math.exp(-CONSOLIDATE_RATE * cleared)
            if self.place is not None:  # NREM: the place map scaled back down (synaptic homeostasis)
                self.place *= math.exp(-CONSOLIDATE_RATE * cleared)
        body.feed_visual_sustenance(p["snack"])
        body.feed_host(p["prey_now"], p["interval"] / max(1.0, self.fps))  # a flow over the look's interval
        if rec is not None:
            rec["prey_eaten"].append(p["prey_now"])
            rec["foods"].append(p["snack"])
            rec["energies"].append(body.energy)
            rec["drives"].append(body.drive())
            rec["asleeps"].append(body.asleep)
        self.pending = None
