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
from .genome import RETINA_PLANES
from .state import MosquitoState
from . import reflexes

# --- Prices (the body pays per gaze, x CPU scarcity) -----------------------
# Every receptor of its gaze (and the wiring behind it) costs upkeep per gaze
# x scarcity (REFERENCE_QUOTA_PCT / the real CPU quota granted): the eye can
# grow when resources allow and shrinks when they are scarce (Sterling and
# Laughlin: neurons are priced per use). Re-anchored from the zoom eye's
# aperture cost (1e-4 x (fraction of the frame)^2), so an eye of the same
# extent costs the same as before.
REFERENCE_QUOTA_PCT = 150.0
APERTURE_COST = 1e-4  # the zoom eye's price, per gaze, x (extent)^2 -- the anchor
RECEPTOR_COST = APERTURE_COST * fovea.RECEPTOR_PITCH ** 2  # per receptor, per gaze, x scarcity


def _receptor_cost(n: int, quota_pct: float) -> float:
    return RECEPTOR_COST * n * n * (REFERENCE_QUOTA_PCT / max(1.0, quota_pct))


# Per-look compute cost (the brain and tree running once), priced like
# the aperture: x CPU scarcity, and x the brain's real arithmetic relative
# to the original 16-unit brain (controller.think_factor) -- a grown brain
# pays for what it computes. With the waking burn scaling with tempo
# (state.py), this is what makes a fast pace of life expensive.
THINK_COST = 2e-5  # per gaze, x scarcity
# Colour is seen by cones only (fovea.py: a central patch of the gaze, its
# size inherited, genome.cones; the rest are rods -- grey, and cheaper for
# lacking the colour circuitry). Each cone costs energy per gaze for each
# colour-opponent channel it serves: colour vision, and how much of the eye
# has it, only evolve if seeing colour pays for itself. Anchored on the old
# eye's price: 1e-5 per channel for its 144 receptors.
COLOUR_COST = 1e-5                 # per channel, for the old eye's 144 receptors -- the anchor
CONE_COST = COLOUR_COST / 144      # per cone, per channel, per gaze, x scarcity
# A brain channel's loop (controller.py) costs energy in proportion to the
# weight on its way back in: a loop that does nothing is free, one that
# matters has to pay for itself (a synaptic cost, like any real circuit).
CHANNEL_COST = 2e-5  # per unit of loop weight, per gaze, x scarcity
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
STABILIZER_COST = 1e-5  # per gaze at gain 1, x scarcity
# Prey sense (genome.prey_sense; after a design panel): 1 = scent, prey
# somewhere in its whole field and how much, without where -- "go look";
# 2 = + a coarse direction from its gaze to the strongest prey (left/right,
# up/down, or none within 5% of centre). It still has to centre prey with
# its eyes to eat. Like a grown brain channel, a new sense changes nothing
# until the brain wires it up, and its price grows with that wiring
# (synaptic cost), so it is kept on a tie and spreads only if it pays.
PREY_SENSE_COST = 2e-5  # per unit of weight on its inputs, per gaze, x scarcity

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


def prey_sense(boxes: list, cx: float, cy: float, level: int) -> tuple[float, float, float]:
    """(scent, direction x, direction y) for its prey-sense level."""
    if level <= 0 or not boxes:
        return 0.0, 0.0, 0.0
    scent = min(1.0, sum(conf * min(1.0, (x1 - x0) * (y1 - y0) / 0.02) for _, conf, x0, y0, x1, y1 in boxes))
    if level < 2:
        return scent, 0.0, 0.0
    best = max(boxes, key=lambda b: b[1] * (b[4] - b[2]) * (b[5] - b[3]))
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
                 colour: bool = False, prey: bool = False, record: bool = False):
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
        # What the gaze has seen per world location, and how much each spot
        # usually varies -- carried across generations (a fresh memory every
        # window meant a fan was only "boring" for ~40 s).
        if memory is not None:
            self.memory, self.variance = memory[0].copy(), memory[1].copy()
        else:
            self.memory, self.variance = new_memory()
        # Colour vision: how many opponent channels this genome's gaze has
        # (0-2). A plane it doesn't have reads zero.
        self.colour_n = int(getattr(g, "colour_channels", 0)) if colour else 0
        self.cones = int(min(getattr(g, "cones", self.state.n), self.state.n))
        self.last_colour = None
        self.quota_pct = quota_pct
        self.scarcity = REFERENCE_QUOTA_PCT / max(1.0, quota_pct)
        self.fps = fps
        self.stab = float(getattr(g, "stabilizer", 0.0))
        self.prey_level = int(getattr(g, "prey_sense", 0)) if prey else 0
        self.stab_dx = self.stab_dy = 0.0  # how far the stabilizer moved the gaze since the last gaze
        self.prev_v = np.zeros(self.n_cells)
        # Response tree's motor-efference input: the brain's real applied movement.
        self.prev_dx, self.prev_dy = 0.0, 0.0
        self.prev_frame = None
        self.prev_response = 0.0
        self.last_grid = None
        self.k = 0             # frames lived
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
        v = np.zeros(self.n_cells) if was_asleep else fovea.extract(frame, state)
        if rec is not None:
            rec["positions"].append((state.cx, state.cy))
            rec["fracs"].append(state.extent)
            rec["frame_path"].append((state.cx, state.cy, state.extent))
        # What the look sees change, with its own eye movement cancelled out
        # (efference copy): the previous frame sampled where the gaze is now,
        # less what the stabilizer moved it -- so a saccade across a still
        # scene doesn't register as motion, and a shake it held still reads
        # as stillness.
        h1 = (fovea.extract(self.prev_frame, fovea.FoveaState(cx=state.cx - self.stab_dx, cy=state.cy - self.stab_dy, n=n))
              if self.prev_frame is not None and not was_asleep else v)
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
        scent, prey_dx, prey_dy = prey_sense(boxes, state.cx, state.cy, self.prey_level)
        out = brain.step(
            lum, motion, flow_x, flow_y, loom, state.cx, state.cy, state.extent, body,
            periph_dx, periph_dy, state.vx, state.vy, self.prev_response, field_light, scent, prey_dx, prey_dy,
        )
        # (out.zoom is unused: its eye has no zoom -- fovea.py.)
        pan, tilt, alarm, tempo = out.pan, out.tilt, out.alarm, out.tempo
        # Sleep is its own choice (its sleep output); the body adds only the
        # physiological overrides -- collapse, hunger, a big change (state.py).
        body.set_sleep(out.sleep > 0.0, loom, periph_motion)
        asleep = body.asleep >= 0.5
        # An empty body runs on less (soft floor): colour off, slower gazing.
        # Asleep, the eye is shut: no colour either.
        colour_on = self.colour_n if not (asleep or body.degraded) else 0
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

        # Eating happens at the gaze: prey held in the gaze center is a meal;
        # genuinely new structure there is a small snack.
        if asleep:
            # Asleep: no eating, no gazing, slow coarse sampling of the field.
            pan = tilt = 0.0
            prey_now = snack = 0.0
            interval = MAX_INTERVAL
        else:
            prey_now = prey_lib.prey_in_gaze(boxes, state.cx, state.cy, *state.half_extents(self.aspect))
            snack = feed_on_novelty(self.memory, v, state, self.variance, self.aspect)
            interval = int(np.clip(round(self.pace * TEMPO_RANGE ** (-float(tempo))), 1, MAX_INTERVAL))
            if body.degraded:
                interval = min(MAX_INTERVAL, interval * 2)
        self.eating = prey_now
        if rec is not None:
            rec["idxs"].append(self.k)
            rec["intervals"].append(interval)
        # Eye physics every FRAME until the next gaze: the brain's force is
        # held, and the damped eye keeps moving meanwhile.
        self.stab_dx = self.stab_dy = 0.0
        self.pending = {"pan": pan, "tilt": tilt, "asleep": asleep, "colour_on": colour_on,
                        "interval": interval, "done": 0, "effort": 0.0, "force": (0.0, 0.0),
                        "cx0": state.cx, "cy0": state.cy, "periph_motion": periph_motion, "loom": loom,
                        "field_light": field_light, "prey_now": prey_now, "snack": snack}
        self.prev_frame = frame

    def _substep(self, shift: tuple, in_world: bool) -> None:
        """The eye moves one frame on (muscle energy = force squared, per frame pushed)."""
        p = self.pending
        state, force_x, force_y = fovea.step(self.state, p["pan"], p["tilt"])
        if self.stab > 0.0 and not p["asleep"] and in_world:
            nx = float(np.clip(state.cx + self.stab * shift[0], 0.0, 1.0))
            ny = float(np.clip(state.cy + self.stab * shift[1], 0.0, 1.0))
            self.stab_dx, self.stab_dy = self.stab_dx + nx - state.cx, self.stab_dy + ny - state.cy
            state = fovea.FoveaState(cx=nx, cy=ny, n=state.n, vx=state.vx, vy=state.vy)
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
        gaze_cost = 0.0 if asleep else _receptor_cost(self.state.n, self.quota_pct)
        body.update(p["periph_motion"], p["loom"], p["effort"],
                    gaze_cost + (THINK_COST * brain.think_factor() + CONE_COST * self.cones * self.cones * p["colour_on"]
                                 + CHANNEL_COST * brain.loop_synapses()
                                 + (0.0 if asleep else STABILIZER_COST * self.stab)
                                 + PREY_SENSE_COST * brain.prey_synapses(self.prey_level)) * self.scarcity,
                    dt=p["interval"], pace=p["interval"], dt_seconds=p["interval"] / max(1.0, self.fps),
                    field_light=p["field_light"])
        cleared = body.take_cleared()
        if cleared > 0.0:
            floor = NOISE_FLOOR ** 2
            self.variance[...] = floor + (self.variance - floor) * math.exp(-CONSOLIDATE_RATE * cleared)
        body.feed_visual_sustenance(p["snack"])
        body.feed_prey(p["prey_now"])
        if rec is not None:
            rec["prey_eaten"].append(p["prey_now"])
            rec["foods"].append(p["snack"])
            rec["energies"].append(body.energy)
            rec["drives"].append(body.drive())
            rec["asleeps"].append(body.asleep)
        self.pending = None
