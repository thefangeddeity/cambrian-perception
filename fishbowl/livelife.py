"""
The organism acting live while evolution learns offline (after a
2026-09-28 panel; Friston, Dennett: animals act awake and consolidate from
replay).

A LiveLife is the lineage's one living body. On its own thread, it lives
every frame the live feed keeps, once, in order, as the frame arrives: the
same Organism a generation's run uses, with the accepted genome, its body
and its memory. Its body and memory are the body and memory of record:
each generation scores its children starting from a copy of them, and
when a child wins, the genome is adopted by this living body -- a brain
transplant (fishbowl/live.py's adopt): the body, memory and gaze go on.

What it lived is the viewer's picture (live_actor.json beside
live_status.json, the newest KEEP frames), its feeding record, and the
hourly metrics -- each frame exactly once, with no overlapping windows.
Its only delay is its own: the feed's, and its brain's reaction time.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import fovea
from .field import FieldSignals
from .metrics import HourlyMetrics
from .organism import EXPANSION_GAIN, PERIPH_MOTION_GAIN, Organism

KEEP = 150            # frames shown to the viewer (~20 s of kept frames): enough for its replay delay
WRITE_EVERY_S = 1.0   # how often the viewer's file is rewritten (it is in RAM where the host has /dev/shm)


def memory_of(org: Organism) -> tuple:
    """A copy of an organism's memory, in run_vision's memory-tuple order."""
    def c(a):
        return None if a is None else np.array(a, dtype=float, copy=True)
    return (c(org.memory), c(org.variance), c(org.mb.weights), c(org.place), c(org.people_day),
            c(org.people_night), c(org.mb.danger_weights))


class LiveLife:
    def __init__(self, feed, genome, body: dict | None, memory: tuple | None, quota_pct: float,
                 sec_per_mac: float, epoch, path: Path, metrics: HourlyMetrics, fps: float = 15.0):
        self.feed, self.epoch, self.path, self.metrics = feed, epoch, Path(path), metrics
        self.lock = threading.Lock()
        self.quota_pct, self.sec_per_mac = quota_pct, sec_per_mac
        self.org = self._organism(genome, body, memory, fps)
        self.field = FieldSignals()
        self.last = None          # the feed's index of the last frame it lived
        self.last_time = None     # when that frame arrived
        self.shown = collections.deque(maxlen=KEEP)
        self.acts = {"meal": [], "snack": []}
        self.adoptions = 0
        self.error = None
        self._stop = False
        self._written = 0.0
        self._thread = threading.Thread(target=self._run, name="LiveLife", daemon=True)
        self._thread.start()

    def _organism(self, genome, body, memory, fps) -> Organism:
        return Organism(genome, body, memory, self.quota_pct, fps, colour=True, prey=True, record=False,
                        sec_per_mac=self.sec_per_mac)

    # ---- the main loop's side (all under the lock) ------------------------------
    def snapshot(self) -> tuple[dict, tuple]:
        """Its body and memory now: where each generation's children start."""
        with self.lock:
            return self.org.body.to_dict(), memory_of(self.org)

    def adopt(self, genome) -> None:
        """A child won: its genome takes over this body (a brain transplant)."""
        with self.lock:
            old = self.org
            new = self._organism(genome, old.body.to_dict(), memory_of(old), old.fps)
            new.state = fovea.FoveaState(cx=old.state.cx, cy=old.state.cy, n=new.state.n)  # the gaze goes on
            self.org = new
            self.adoptions += 1

    def set_prices(self, quota_pct: float, sec_per_mac: float) -> None:
        with self.lock:
            self.quota_pct, self.sec_per_mac = quota_pct, sec_per_mac
            self.org.quota_pct, self.org.sec_per_mac = quota_pct, sec_per_mac
            self.org.scarcity = 150.0 / max(1.0, quota_pct)  # organism.REFERENCE_QUOTA_PCT / quota

    def drain_acts(self) -> dict:
        """Its feeding acts (the feed's frame indices) since last asked."""
        with self.lock:
            acts, self.acts = self.acts, {"meal": [], "snack": []}
            return acts

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=10.0)

    # ---- its life, frame by frame ----------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop:
                items = self.feed.since(self.last)
                if not items:
                    time.sleep(0.02)
                    continue
                self._live(items)
                if time.time() - self._written >= WRITE_EVERY_S:
                    self._write()
        except Exception as e:  # its death must not take evolution with it; the run's end reports it
            self.error = repr(e)

    def _live(self, items: list) -> None:
        fps = max(1.0, self.feed.frames_per_second() or 15.0)
        batch = {"eating": [], "asleep": [], "alarm": [], "trajectory": [], "swat_acts": []}
        looks = missed = 0
        with self.lock:
            org = self.org
            org.fps = fps
            missed0 = org.missed
            for j, (index, grey, boxes, colour, arrived) in enumerate(items):
                sig, shift = self.field.step(grey)
                swats0 = org.swats
                out = org.frame(grey, sig, boxes or [], colour, shift)
                snack = org.pending["snack"] if out["gazed"] and org.pending else 0.0
                if out["gazed"]:
                    looks += 1
                    if out["eating"] > 0:
                        self.acts["meal"].append(index)
                    if snack > 0:
                        self.acts["snack"].append(index)
                prev = self.shown[-1] if self.shown else None
                rec = {"i": index, "cx": round(out["cx"], 4), "cy": round(out["cy"], 4), "f": round(out["extent"], 4),
                       "eat": round(float(out["eating"]), 3),
                       "snack": round(float(snack), 3) if out["gazed"] else (prev["snack"] if prev else 0.0),
                       "asleep": int(out["asleep"]), "alarm": int(org.last_alarm > 0.0), "boxes": boxes or [],
                       "ev": [round(float(sig["motion_cx"]), 3), round(float(sig["motion_cy"]), 3),
                              round(float(min(1.0, sig["motion_energy"] * PERIPH_MOTION_GAIN)), 3),
                              round(float(min(1.0, sig["expansion"] * EXPANSION_GAIN)), 3), 0]}
                self.shown.append(rec)
                batch["eating"].append(rec["eat"])
                batch["asleep"].append(rec["asleep"])
                batch["alarm"].append(rec["alarm"])
                batch["trajectory"].append([rec["cx"], rec["cy"], rec["f"], j])
                if org.swats > swats0:
                    batch["swat_acts"].append(j)
                self.last, self.last_time = index, arrived
            missed = org.missed - missed0
            batch["pace"] = float(org.last_interval)
            batch["missed_share"] = missed / max(1, looks)
            self.metrics.add(batch, items[0][0], fps)

    def _write(self) -> None:
        """The newest frames it lived, for the viewer (merged over live_status.json)."""
        with self.lock:
            shown = list(self.shown)
            body = {k: round(float(v), 4) for k, v in self.org.body.to_dict().items() if isinstance(v, (int, float))}
            pace = self.org.last_interval
            fps = self.org.fps
        if not shown:
            return
        first = shown[0]["i"]
        data = {
            "live_actor": True, "live_adoptions": self.adoptions,
            "world_epoch": self.epoch, "world_first_index": first,
            "world_age_s": round(time.time() - self.last_time, 2) if self.last_time else None,
            "trajectory": [[r["cx"], r["cy"], r["f"], r["i"] - first] for r in shown],
            "eating": [r["eat"] for r in shown], "snacks": [r["snack"] for r in shown],
            "asleep_frames": [r["asleep"] for r in shown], "alarm_frames": [r["alarm"] for r in shown],
            "prey_boxes": [r["boxes"] for r in shown], "field_events": [r["ev"] for r in shown],
            "tree_guess": None, "teacher_label": None,  # graded only in evolution's runs
            "fovea_cx": shown[-1]["cx"], "fovea_cy": shown[-1]["cy"],
            "pace": pace, "frames_per_second": round(fps, 2), "body_now": body,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # the viewer keeps the last one; its life goes on
        self._written = time.time()
