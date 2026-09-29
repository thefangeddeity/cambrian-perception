"""
Its visual cortex (a 2026-09-29 panel): the models it builds of what it sees,
below its cognition -- transforms it has no access to, as ours are (Marr's
2.5D sketch; the ventral stream's who and the dorsal stream's where, Ungerleider
& Mishkin). Its ground plane and terrain map live in the organism (they are
senses too); this keeps the rest, for the live organism only:

  - tracks: each thing it sees, followed from frame to frame (SORT's rule,
    Bewley et al. 2016: same class, box overlap IoU >= 0.3)
  - on its ground: where each tracked thing stands (in camera heights, from the
    horizon), its true height, its speed across the ground
  - gait: the rhythm of the motion inside its box (Johansson's biological
    motion; Cutting & Kozlowski 1977 -- friends known by their walk alone):
    the cadence, from the autocorrelation of that motion over a few seconds
  - a library of individuals (who, not just what): each track's signature --
    true height, cadence, speed, colour -- matched to the individuals it has
    met of that class, or a new one

Matching is a chi-square test (95%, one degree of freedom per feature used)
of the track against each individual, each feature scaled by how much one
individual varies in it. That scale is learned the way an infant (or a rat)
could: a thing it keeps its eyes on is certainly one thing (spatiotemporal
continuity, Spelke), so the spread of each measurement within its tracks is
how much one individual varies. Identity is read from what it measures at
every detection -- true height and colour; step rate and speed are kept on
each individual but not matched on until their own spread is measured. Nothing here is evolved, fed to its brain, or charged for yet: the
panel's order was to let us watch it first.
Assumptions (listed for the constants audit): focal length ~ the frame's
height (ground distances); a track is lost after TRACK_LOST_S unseen; gait is
looked for between 0.5 and 4 Hz (walking to trotting, people and pets).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import prey as prey_lib

IOU_MATCH = 0.3        # SORT's association threshold
TRACK_LOST_S = 2.0     # unseen this long: the track ends
MATURE_S = 2.0         # a track signs (is matched to an individual) after this long
GAIT_WINDOW_S = 4.0    # the stretch of motion its gait is read from
GAIT_HZ = (0.5, 4.0)   # the cadences looked for
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}
LIBRARY_MAX = 64       # individuals remembered (the least seen, longest ago, go first)
FEATURES = ("height",)               # matched on (with colour); cadence and speed are recorded
RECORDED = ("height", "cadence", "speed")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _hist(colour, grey, box) -> tuple[str, np.ndarray] | None:
    """Its colour inside a box: hue x saturation (8 x 4) if it sees colour, else grey levels (16)."""
    src = colour if colour is not None else grey
    if src is None:
        return None
    h, w = src.shape[:2]
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = src[y0:y1, x0:x1]
    if colour is not None:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist, kind = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel(), "hs"
    else:
        hist, kind = cv2.calcHist([patch], [0], None, [16], [0, 256]).ravel(), "grey"
    s = float(hist.sum())
    return (kind, hist / s) if s > 0 else None


def _bhatt(p, q) -> float:
    return float(math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(p * q))))))


class _Welford:
    def __init__(self, d=None):
        self.n, self.mean, self.m2 = (d or {}).get("n", 0), (d or {}).get("mean", 0.0), (d or {}).get("m2", 0.0)

    def add(self, x: float) -> None:
        self.n += 1
        dx = x - self.mean
        self.mean += dx / self.n
        self.m2 += dx * (x - self.mean)

    @property
    def var(self) -> float | None:
        return self.m2 / (self.n - 1) if self.n > 1 else None

    def to_dict(self):
        return {"n": self.n, "mean": round(self.mean, 5), "m2": round(self.m2, 6)}


class Track:
    def __init__(self, tid: int, cls: int, box, t: float):
        self.id, self.cls, self.box, self.first, self.last = tid, cls, box, t, t
        self.heights: list = []       # true heights (camera heights), one per fresh detection
        self.ground: list = []        # (t, X, Z) where it stood
        self.motion: list = []        # (t, motion energy inside its box), every frame
        self.hist = None              # (kind, running mean histogram)
        self.n_hist = 0
        self.hist_d: list = []        # each detection's colour distance from the track's mean so far
        self.who = None               # the individual it was matched to
        self.signed = False

    def signature(self) -> dict:
        sig = {}
        if len(self.heights) >= 3:
            sig["height"] = float(np.median(self.heights))
        cad = self.cadence()
        if cad is not None:
            sig["cadence"] = cad  # its step rate: the motion in its box peaks once a step
        if len(self.ground) >= 2:
            (t0, x0, z0), (t1, x1, z1) = self.ground[0], self.ground[-1]
            if t1 - t0 > 0.5:
                sig["speed"] = math.hypot(x1 - x0, z1 - z0) / (t1 - t0)
        return sig

    def cadence(self) -> float | None:
        """Its gait's rhythm (Hz): the strongest autocorrelation peak of the motion in
        its box over the last few seconds, if it clears the white-noise band (2 / sqrt N)."""
        if len(self.motion) < 8:
            return None
        t = np.array([m[0] for m in self.motion])
        span = t[-1] - t[0]
        if span < 1.0 / GAIT_HZ[0]:  # one period of the slowest gait at least
            return None
        fps = (len(t) - 1) / span
        x = np.array([m[1] for m in self.motion], dtype=float)
        x = x - x.mean()
        if not np.any(x):
            return None
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac = ac / ac[0]
        lo, hi = max(1, int(fps / GAIT_HZ[1])), min(len(ac) - 1, int(fps / GAIT_HZ[0]))
        if hi <= lo:
            return None
        lag = lo + int(np.argmax(ac[lo:hi + 1]))
        if ac[lag] <= 2.0 / math.sqrt(len(x)):
            return None
        # the peak between frames: a parabola through it and its neighbours
        exact = float(lag)
        if 0 < lag < len(ac) - 1:
            den = ac[lag - 1] - 2 * ac[lag] + ac[lag + 1]
            if den < 0:
                exact += 0.5 * (ac[lag - 1] - ac[lag + 1]) / den
        return float(fps / exact)


class Cortex:
    def __init__(self, saved: dict | None = None):
        saved = saved or {}
        self.tracks: list[Track] = []
        self.next_track = 0
        self.library: list[dict] = saved.get("library", [])
        self.next_id = int(saved.get("next_id", len(self.library)))
        # per class, per feature: spread across the class (all tracks), and within one individual (re-met)
        self.pop = {k: {f: _Welford(v) for f, v in d.items()} for k, d in (saved.get("pop") or {}).items()}
        self.within = {k: {f: _Welford(v) for f, v in d.items()} for k, d in (saved.get("within") or {}).items()}
        self.prev_grey = None
        self.now = 0.0
        self.events: list = []   # (kind, data) for the organism's event log, drained by the caller

    # ---- each frame ------------------------------------------------------------
    def see(self, t: float, grey, colour, boxes: list, horizon: float | None, fresh: bool) -> None:
        """One frame: boxes are its detector's (held between detections; fresh = new this frame)."""
        self.now = t
        motion = None
        if grey is not None and self.prev_grey is not None and self.prev_grey.shape == grey.shape:
            motion = cv2.absdiff(grey, self.prev_grey)
        self.prev_grey = grey
        free = list(self.tracks)
        for b in boxes or []:
            cls, box = int(b[0]), tuple(float(v) for v in b[2:6])
            best = max((tr for tr in free if tr.cls == cls), key=lambda tr: _iou(tr.box, box), default=None)
            if best is None or _iou(best.box, box) < IOU_MATCH:
                best = Track(self.next_track, cls, box, t)
                self.next_track += 1
                self.tracks.append(best)
            else:
                free.remove(best)
            tr = best
            tr.box, tr.last = box, t
            if motion is not None:
                h, w = motion.shape[:2]
                x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), max(int(box[0] * w) + 1, int(box[2] * w)), max(int(box[1] * h) + 1, int(box[3] * h))
                tr.motion.append((t, float(motion[y0:y1, x0:x1].mean())))
                while tr.motion and t - tr.motion[0][0] > GAIT_WINDOW_S:
                    tr.motion.pop(0)
            if not fresh:
                continue
            cut_base, cut_top = prey_lib.cut_by_frame(b)
            if horizon is not None and not cut_base and not cut_top and box[3] - horizon > 0.02:
                depth = 1.0 / (box[3] - horizon)   # camera heights (focal length ~ frame height)
                tr.heights.append((box[3] - box[1]) * depth)
                tr.ground.append((t, ((box[0] + box[2]) / 2 - 0.5) * depth, depth))
                tr.ground = tr.ground[-50:]
                tr.heights = tr.heights[-50:]
            hs = _hist(colour, grey, box)
            if hs is not None:
                if tr.hist is None or tr.hist[0] != hs[0]:
                    tr.hist, tr.n_hist = (hs[0], hs[1].copy()), 1
                else:
                    tr.hist_d.append(_bhatt(hs[1], tr.hist[1]))
                    tr.n_hist += 1
                    tr.hist[1][:] += (hs[1] - tr.hist[1]) / tr.n_hist
            if not tr.signed and tr.cls in prey_lib.PREY_CLASSES and t - tr.first >= MATURE_S:  # individuals: living things
                self._sign(tr, t)
        for tr in [tr for tr in self.tracks if t - tr.last > TRACK_LOST_S]:
            self._learn_within(tr)
            if tr.signed:
                self._learn(tr, t)
            self.tracks.remove(tr)

    def _learn_within(self, tr: Track) -> None:
        """What one track -- certainly one thing -- says about how much one
        individual's measurements spread: every detection's deviation from
        its track's mean (height), and its colour's distance from the mean."""
        k = str(tr.cls)
        within = self.within.setdefault(k, {})
        if len(tr.heights) >= 3:
            m = float(np.mean(tr.heights))
            for h in tr.heights:
                within.setdefault("height", _Welford()).add(h - m)
        for d in tr.hist_d:
            within.setdefault("colour", _Welford()).add(d)

    # ---- who it is ---------------------------------------------------------------
    def _scale(self, cls: int, f: str) -> float | None:
        """How much one individual of this class varies in a feature: learned
        from re-met individuals, else the whole class's spread (lumping early)."""
        # colour's statistics are distances already: its scale is their mean square
        w = self.within.get(str(cls), {}).get(f)
        if w is None or w.var is None or w.n < 3:
            return None  # it doesn't know yet how much one of these varies: no judgement
        return w.mean ** 2 + w.var if f == "colour" else w.var

    def _distance(self, tr: Track, sig: dict, ind: dict) -> tuple[float, int]:
        d2, used = 0.0, 0
        for f in FEATURES:
            v = self._scale(tr.cls, f)
            if v is not None:
                v = max(v, 1e-9)  # a spread of exactly nothing is only a numerical accident
            if f in sig and f in ind["mean"] and v:
                d2 += (sig[f] - ind["mean"][f]) ** 2 / v
                used += 1
        if tr.hist is not None and ind.get("hist") and ind["hist"][0] == tr.hist[0]:
            v = self._scale(tr.cls, "colour")
            if v is not None:
                v = max(v, 1e-9)
                d2 += _bhatt(tr.hist[1], np.array(ind["hist"][1])) ** 2 / v
                used += 1
        return d2, used

    def _sign(self, tr: Track, t: float) -> None:
        sig = self._signature(tr)
        best, best_d = None, None
        for ind in self.library:
            if ind["cls"] != tr.cls:
                continue
            d2, used = self._distance(tr, sig, ind)
            if used and d2 <= CHI2_95[min(used, 4)] and (best_d is None or d2 < best_d):
                best, best_d = ind, d2
        if best is None and not any(self._scale(tr.cls, f) for f in FEATURES + ("colour",)):
            return  # no judgement possible yet (it hasn't learned how much one of these varies): later
        tr.signed = True
        if best is None:
            if not sig and tr.hist is None:
                tr.signed = False  # nothing to know it by yet
                return
            best = {"id": self.next_id, "cls": tr.cls, "seen": 0, "first": round(t, 1), "last": round(t, 1),
                    "mean": {}, "n": {}, "hist": None, "n_hist": 0}
            self.next_id += 1
            self.library.append(best)
            self.events.append(("met", {"who": best["id"], "cls": tr.cls, **{f: round(v, 2) for f, v in sig.items()}}))
            if len(self.library) > LIBRARY_MAX:
                self.library.remove(min(self.library, key=lambda i: (i["seen"], i["last"])))
        else:
            self.events.append(("met again", {"who": best["id"], "cls": tr.cls, "away_s": round(t - best["last"], 1),
                                              "distance": round(float(best_d), 2)}))
        tr.who = best["id"]

    def _signature(self, tr: Track) -> dict:
        return tr.signature()

    def _learn(self, tr: Track, t: float) -> None:
        """A signed track ended: its signature teaches its individual, and the
        class's spreads (across the class; within the individual, when re-met)."""
        ind = next((i for i in self.library if i["id"] == tr.who), None)
        if ind is None:
            return
        sig, k = tr.signature(), str(tr.cls)
        pop = self.pop.setdefault(k, {})
        for f, x in sig.items():
            pop.setdefault(f, _Welford()).add(x)
            n = ind["n"].get(f, 0) + 1
            ind["n"][f] = n
            ind["mean"][f] = ind["mean"].get(f, x) + (x - ind["mean"].get(f, x)) / n
        if tr.hist is not None:
            if ind["hist"] and ind["hist"][0] == tr.hist[0]:
                old = np.array(ind["hist"][1])
                ind["n_hist"] += 1
                ind["hist"] = [tr.hist[0], (old + (tr.hist[1] - old) / ind["n_hist"]).round(5).tolist()]
            else:
                ind["hist"], ind["n_hist"] = [tr.hist[0], tr.hist[1].round(5).tolist()], 1
            others = [np.array(i["hist"][1]) for i in self.library if i is not ind and i["cls"] == tr.cls
                      and i.get("hist") and i["hist"][0] == tr.hist[0]]
            for o in others[:8]:
                pop.setdefault("colour", _Welford()).add(_bhatt(tr.hist[1], o))
        ind["seen"] += 1
        ind["last"] = round(t, 1)
        self._merge(ind)

    def _merge(self, ind: dict) -> None:
        """Two it met as strangers, and now knows well enough to tell they are
        one (the same chi-square test, on what it knows of each): one individual."""
        probe = Track(-1, ind["cls"], (0, 0, 0, 0), 0.0)
        probe.hist = None if not ind.get("hist") else (ind["hist"][0], np.array(ind["hist"][1]))
        for other in [i for i in self.library if i is not ind and i["cls"] == ind["cls"]]:
            d2, used = self._distance(probe, dict(ind["mean"]), other)
            if used and d2 <= CHI2_95[min(used, 4)]:
                keep, gone = (ind, other) if ind["seen"] >= other["seen"] else (other, ind)
                for f, x in gone["mean"].items():
                    n0, n1 = keep["n"].get(f, 0), gone["n"].get(f, 0)
                    keep["mean"][f] = (keep["mean"].get(f, x) * n0 + x * n1) / max(1, n0 + n1)
                    keep["n"][f] = n0 + n1
                keep["seen"] += gone["seen"]
                keep["first"], keep["last"] = min(keep["first"], gone["first"]), max(keep["last"], gone["last"])
                self.library.remove(gone)
                for tr in self.tracks:
                    if tr.who == gone["id"]:
                        tr.who = keep["id"]
                self.events.append(("one and the same", {"kept": keep["id"], "merged": gone["id"], "cls": keep["cls"]}))
                return

    # ---- for the viewer and the disk ----------------------------------------------
    def view(self) -> dict:
        names = {**prey_lib.PREY_CLASSES, prey_lib.PLANT_CLASS: "plant"}
        return {
            "tracks": [{"id": tr.id, "cls": tr.cls, "box": [round(v, 4) for v in tr.box], "who": tr.who,
                        **{f: round(v, 3) for f, v in tr.signature().items()}} for tr in self.tracks],
            "known": [{"id": i["id"], "cls": i["cls"], "name": names.get(i["cls"], str(i["cls"])), "seen": i["seen"],
                       "first": i["first"], "last": i["last"], **{f: round(v, 3) for f, v in i["mean"].items()}}
                      for i in sorted(self.library, key=lambda i: -i["last"])[:24]],
            "individuals": len(self.library), "now": round(self.now, 1),
        }

    def to_dict(self) -> dict:
        return {"library": self.library, "next_id": self.next_id,
                "pop": {k: {f: w.to_dict() for f, w in d.items()} for k, d in self.pop.items()},
                "within": {k: {f: w.to_dict() for f, w in d.items()} for k, d in self.within.items()}}
