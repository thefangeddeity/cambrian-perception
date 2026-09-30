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

# What has essence, by the labels detectors use: this host's food classes
# (prey.PREY_CLASSES: living beings and the things we lend a life to), by
# each label's first word as they are matched below, and "human".
from . import prey as _prey
LIVING = {n.split()[0] for n in _prey.PREY_CLASSES.values()} | {"human"}


def _unb64(d):
    import base64
    if not d:
        return None
    return np.frombuffer(base64.b64decode(d["f16"]), dtype=np.float16).astype(float).reshape(d["shape"])


def _b64(a) -> dict:
    """An array, compact: float16 bytes, base64, with its shape."""
    import base64
    a = np.asarray(a, dtype=np.float16)
    return {"shape": list(a.shape), "f16": base64.b64encode(a.tobytes()).decode("ascii")}


def _learned_parts(org) -> dict | None:
    """What its body has learned that runs from its own eye with no teacher (its
    model card's frozen parts): its mushroom body's code (size, seed), its food
    and danger readouts, its archetype heads (taught by the detector) and its
    terrain head (taught by its ground model), and its distilled output layer."""
    if org is None:
        return None
    mb = org.mb
    rows = list(range(org.n_heads)) + ([len(mb.heads) - 1] if org.felt_terrain else [])
    out = {"n_kc": mb.n_kc, "kc_seed": int(getattr(org.g, "kc_seed", 0)),
           "food": _b64(mb.weights), "heads": _b64(mb.heads[rows]) if rows else None,
           "head_classes": [int(c) for c in org.head_classes[:org.n_heads]], "terrain_head": bool(org.felt_terrain),
           "terrain_target": "the next look" if org.lookahead else "this look",
           "extrapolation": round(float(org.extrapolation), 3)}
    if getattr(mb, "danger_weights", None) is not None:
        out["danger"] = _b64(mb.danger_weights)
    if org.plasticity > 0.0 and org.brain is not org.g.brain:
        out["distilled"] = {"weights_ho": _b64(org.brain.weights_ho), "bias_o": _b64(org.brain.bias_o)}
    return out


def apply_learned(org, learned: dict | None) -> list[str]:
    """Its model card's frozen learned parts (run_vision._learned_parts) into a
    body built from the same genome: its mushroom body's readouts (when its
    code -- size and seed -- is the one they were learned on), its heads
    (archetypes, terrain) and its distilled output layer. Returns what it took."""
    took = []
    if not learned or org is None:
        return took
    mb = org.mb
    if learned.get("n_kc") == mb.n_kc and learned.get("kc_seed") == int(getattr(org.g, "kc_seed", 0)):
        food = _unb64(learned.get("food"))
        if food is not None and food.shape == mb.weights.shape:
            mb.weights[:] = food; took.append("food")
        danger = _unb64(learned.get("danger"))
        if danger is not None and getattr(mb, "danger_weights", None) is not None and danger.shape == mb.danger_weights.shape:
            mb.danger_weights[:] = danger; took.append("danger")
        heads = _unb64(learned.get("heads"))
        if heads is not None and heads.shape[1] == mb.n_kc:
            n = len(learned.get("head_classes") or [])
            mb.heads[:n] = heads[:n]
            if learned.get("terrain_head") and heads.shape[0] > n:
                mb.heads[-1] = heads[n]; took.append("terrain head")
            took.append(f"{n} archetype heads")
    d = learned.get("distilled")
    if d:
        w, b = _unb64(d["weights_ho"]), _unb64(d["bias_o"])
        if org.brain is org.g.brain:
            org.brain = org.g.brain.clone()
        if w.shape == org.brain.weights_ho.shape and b.shape == org.brain.bias_o.shape:
            org.brain.weights_ho, org.brain.bias_o = w, b; took.append("distilled habits")
    return took


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
        self.learned = champion.get("learned")  # its model card's frozen parts, applied to its body when it's built
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
        self.learned = champion.get("learned")
        if self.org is not None:
            old = self.org
            self.org = Organism(self.genome, body=old.body.to_dict(), memory=(old.memory, old.variance, old.mb.weights, old.place, old.people_day, old.people_night),
                                fps=old.fps, colour=self.colour, prey=True)
            apply_learned(self.org, self.learned)

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
            apply_learned(self.org, self.learned)
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
