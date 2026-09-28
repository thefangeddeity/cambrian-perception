from __future__ import annotations

"""
The organism living live inside a host -- a livecam's CV loop -- frame by
frame, with a frozen champion genome (no evolution here; that happens in
cambrian-perception). Host-agnostic: it needs only frames and the host's own
detections of living things, and returns where its gaze is and what it is
doing, for the host's HUD (fishbowl/hud.py).

It sees its world the way the champion evolved to: frames scaled to
video_source's size, every `stride`-th camera frame (the evolution feed's
rule), the same whole-field signals (fishbowl/field.py) and the same
Organism (fishbowl/organism.py).

Meals are grouped by its own bout criterion (fishbowl/bouts.py, measured
from its own feeding gaps here; the champion's until it has its own), and a
host can hang actions on a meal's start and end -- take a picture, start
and stop a recording. That is the "snapshot ready" hook: nothing is taken
unless the host registers an action and switches it on.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import genome as G
from .bouts import FeedingRecord, fit_bout_criterion
from .field import FieldSignals
from .organism import Organism
from .video_source import DEFAULT_MAX_DIM, SLOW_SOURCE_FPS

# Living things by the labels detectors use (COCO names, and "human").
LIVING = {"person", "human", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def load_champion(*paths) -> dict | None:
    """The first readable champion file: {"genome": ..., "bouts": ..., ...}."""
    for p in paths:
        if not p:
            continue
        try:
            data = json.loads(Path(p).read_text())
            if data.get("genome"):
                data["_path"] = str(p)
                return data
        except (OSError, ValueError):
            continue
    return None


def lock_state(cx: float, cy: float, hx: float, hy: float, eating: float, boxes: list,
               asleep: bool = False) -> tuple[str, list | None]:
    """SCAN / TRACK (prey in its gaze) / LOCK (eating: prey held in its gaze
    centre) / SLEEP (eyes shut: its gaze sees nothing), and the box it is
    on -- the viewer's lockState, in Python."""
    if asleep:
        return "SLEEP", None
    def overlap(b, k):
        return (max(0.0, min(b[4], cx + hx * k) - max(b[2], cx - hx * k))
                * max(0.0, min(b[5], cy + hy * k) - max(b[3], cy - hy * k)))
    best, target, in_gaze = 0.0, None, False
    for b in boxes:
        whole = overlap(b, 1.0)
        score = 4 * overlap(b, 0.5) + whole
        in_gaze = in_gaze or whole > 0
        if score > best:
            best, target = score, b
    return ("LOCK" if eating > 0 else "TRACK" if in_gaze else "SCAN"), target


class LiveActor:
    """champion: a champion dict (load_champion). order: the host's channel
    order ("rgb" or "bgr"). Call frame() with every camera frame and the
    host's detections [(label, confidence, x0, y0, x1, y1) normalised]."""

    def __init__(self, champion: dict, order: str = "bgr", colour: bool = True):
        self.genome = G.Genome.from_dict(champion["genome"])
        self.champion_bouts = (champion.get("bouts") or {}).get("meal", {}).get("fit")
        self.order = order
        self.colour = colour
        self.org = None
        self.field = FieldSignals()
        self.arrivals: list[float] = []
        self.n = 0
        self.state = None
        self.meals = FeedingRecord()
        self.kept = 0
        self.in_meal = False
        self.last_act = None
        self.meal_count = 0
        self._on_start, self._on_end = [], []

    def adopt(self, champion: dict) -> None:
        """A newer champion from cambrian-perception: its genome takes over
        this body -- the body, its surprise memory and its feeding record go on
        (a brain transplant, not a new animal)."""
        self.genome = G.Genome.from_dict(champion["genome"])
        self.champion_bouts = (champion.get("bouts") or {}).get("meal", {}).get("fit")
        if self.org is not None:
            old = self.org
            self.org = Organism(self.genome, body=old.body.to_dict(), memory=(old.memory, old.variance, old.mb.weights, old.place, old.people_day, old.people_night),
                                fps=old.fps, colour=self.colour, prey=True)

    # ---- actions on meals (the snapshot-ready hook) ---------------------------
    def on_meal_start(self, action) -> None:
        """action(state, frame) when a meal starts (e.g. take a picture, start recording)."""
        self._on_start.append(action)

    def on_meal_end(self, action) -> None:
        """action(state) when a meal has ended (e.g. stop recording)."""
        self._on_end.append(action)

    def criterion_s(self) -> float | None:
        """What separates two meals, in seconds: its own, else the champion's."""
        fit = fit_bout_criterion(self.meals.gaps)
        fit = fit or self.champion_bouts
        return fit["criterion_s"] if fit else None

    # ---- one camera frame ---------------------------------------------------------
    def _fps(self) -> float:
        a = self.arrivals
        return (len(a) - 1) / max(1e-6, a[-1] - a[0]) if len(a) > 1 else 15.0

    def frame(self, frame: np.ndarray, detections: list) -> dict | None:
        now = time.time()
        self.arrivals = (self.arrivals + [now])[-60:]
        camera_fps = self._fps()
        stride = 1 if 0 < camera_fps < SLOW_SOURCE_FPS else 2  # video_source.LiveFeed's rule
        self.n += 1
        if (self.n - 1) % stride:
            return self.state
        original = frame  # the host's full picture, for its actions (a snapshot)
        h, w = frame.shape[:2]
        if max(h, w) > DEFAULT_MAX_DIM:
            s = DEFAULT_MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        bgr = frame if self.order == "bgr" else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fps = camera_fps / stride
        if self.org is None:
            self.org = Organism(self.genome, fps=fps, colour=self.colour, prey=True)
        self.org.fps = fps
        self.field.fps = fps
        boxes = [[lbl, float(c), x0, y0, x1, y1] for lbl, c, x0, y0, x1, y1 in detections
                 if str(lbl).split()[0].lower() in LIVING]
        sig, shift = self.field.step(grey)
        out = self.org.frame(grey, sig, boxes, bgr if self.colour else None, shift)
        st = self.org.state
        hx, hy = st.half_extents(w / h)
        mode, target = lock_state(st.cx, st.cy, hx, hy, out["eating"], boxes, out["asleep"])
        # Meals: its feeding acts (looks that caught prey), grouped by its bout criterion.
        self.kept += 1
        if out["gazed"] and out["eating"] > 0:
            self.meals.add([self.kept], self.kept, fps)
            crit = self.criterion_s()
            if not self.in_meal and (self.last_act is None or crit is None or (self.kept - self.last_act) / fps > crit):
                self.in_meal = True
                self.meal_count += 1
                state = self._state(out, mode, target, boxes, fps)
                for act in self._on_start:
                    act(state, original)
            self.last_act = self.kept
        elif self.in_meal and self.last_act is not None:
            crit = self.criterion_s()
            if crit is not None and (self.kept - self.last_act) / fps > crit:
                self.in_meal = False
                for act in self._on_end:
                    act(self._state(out, mode, target, boxes, fps))
        self.state = self._state(out, mode, target, boxes, fps)
        return self.state

    def _state(self, out, mode, target, boxes, fps) -> dict:
        body = self.org.body
        return {"cx": out["cx"], "cy": out["cy"], "extent": out["extent"], "receptors": out["receptors"],
                "mode": mode, "target": target, "boxes": boxes, "eating": out["eating"], "asleep": out["asleep"],
                "meals": self.meal_count, "in_meal": self.in_meal, "energy": round(body.energy, 3),
                "criterion_s": self.criterion_s(), "fps": round(fps, 2)}
