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

import base64
import collections
import copy
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import fovea, prey as prey_lib
from .field import FieldSignals
from .metrics import HourlyMetrics
from .mushroom import PROTO_SIDE
from .organism import (EXPANSION_GAIN, NOISE_FLOOR, PERIPH_MOTION_GAIN, REFERENCE_MACS, RESTING_BURN, THINK_COST,
                       Organism, kc_macs)
from .state import EMPTY_G, LEGACY_UNIT

KEEP = 150            # frames shown to the viewer (~20 s of kept frames): enough for its replay delay
WRITE_EVERY_S = 1.0   # how often the viewer's file is rewritten (it is in RAM where the host has /dev/shm)


def memory_of(org: Organism) -> tuple:
    """A copy of an organism's memory, in run_vision's memory-tuple order."""
    def c(a):
        return None if a is None else np.array(a, dtype=float, copy=True)
    return (c(org.memory), c(org.variance), c(org.mb.weights), c(org.place), c(org.people_day),
            c(org.people_night), c(org.mb.danger_weights), c(org.value_map), dict(org.nectar),
            np.array(org.mb.proto, copy=True), copy.deepcopy(org.library()))


def _carry(old: Organism, new: Organism) -> None:
    """A transplant keeps the moment going wherever the new genome's shapes
    still fit: the gaze and its motion, the brain's recurrent state (most
    adoptions are neutral drift, many a minute -- a fresh brain state each
    time would wipe its short-term memory every few seconds), what the eye
    last saw, and the frame it has reached."""
    new.state = fovea.FoveaState(cx=old.state.cx, cy=old.state.cy, n=new.state.n,
                                 vx=old.state.vx, vy=old.state.vy)
    ob, nb = old.brain, new.brain
    # Its episodes of this life and their replay order go on too: replay
    # draws on them, and dozens of adoptions a minute used to empty them.
    new.episodes, new.priority, new.replays = old.episodes, old.priority, old.replays
    if new.mb.n_kc < old.mb.n_kc:
        # fewer Kenyon cells now (a shrink): an episode's code keeps only the
        # cells that still exist, or replay would index cells that are gone
        n = new.mb.n_kc
        new.episodes = [(code[code < n], reward, cell) for code, reward, cell in old.episodes]
    new.replay_log, new.value_errors, new.seq, new.dreams = old.replay_log, old.value_errors, old.seq, old.dreams
    new.last_replay, new.dreaming = old.last_replay, old.dreaming
    new.nectar, new.sips = old.nectar, old.sips
    new.episode_meta, new.imagery_sums = old.episode_meta, old.imagery_sums
    # what it taught itself asleep survives a transplant that left its tree alone
    if old.tree is not None and old.g.trees.get("response") is not None and new.g.trees.get("response") is not None \
            and old.g.trees["response"].to_dict() == new.g.trees["response"].to_dict():
        new.tree, new.tree_macs = old.tree, old.tree_macs
    new.test_set, new.test_seen = (old.test_set, old.test_seen) if new.sleep_set else ([], 0)
    if len(new.test_set) > new.sleep_set:
        new.test_set = new.test_set[:new.sleep_set]
    new.edits_tried, new.edits_kept = old.edits_tried, old.edits_kept
    if ob.hidden.shape == nb.hidden.shape:
        nb.hidden = ob.hidden.copy()
        if len(ob.layer_hidden) == len(nb.layer_hidden):
            nb.layer_hidden = [h.copy() for h in ob.layer_hidden]
    if len(ob.channels) == len(nb.channels):
        nb.loop_in, nb._pred = ob.loop_in.copy(), ob._pred.copy()
    new.k, new.lived_s, new.aspect, new.frame_h, new.field = old.k, old.lived_s, old.aspect, old.frame_h, old.field
    new.prev_frame, new.prev_response = old.prev_frame, old.prev_response
    new.prev_dx, new.prev_dy, new.last_interval = old.prev_dx, old.prev_dy, old.last_interval
    if new.state.n == old.state.n:
        new.prev_v = old.prev_v.copy()
    if new.slowness > 0.0 and old.slow is not None:
        new.slow = old.slow


def _int8(w: np.ndarray, live: int | None = None) -> tuple[str, float]:
    """Weights as base64 int8, scaled by the largest magnitude among the live
    cells (a lost cell's frozen weight must not wash out the rest)."""
    alive = w[:live] if live is not None else w
    scale = float(np.max(np.abs(alive))) if len(alive) else 0.0
    q = np.clip(np.round(w / scale * 127.0), -127, 127).astype(np.int8) if scale > 0 else np.zeros(len(w), dtype=np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii"), round(scale, 4)


def _circuits(org: Organism) -> dict:
    """What it has learned, for the owner's viewer (a 2026-09-28 panel):
    its maps over the whole field, its mushroom body's firing set and learned
    weights, what it has replayed lately, and what the mushroom body costs.
    Learned values, not thoughts."""
    out = {}
    if org.place is not None:
        rows, cols = org.place.shape
        dl = float(org.body.daylight)
        people = dl * org.people_day + (1.0 - dl) * org.people_night
        mean, var = org.memory, org.variance
        # familiar: 1 = as still as sensor noise (fully habituated), low = still varying; -1 = never seen
        fam = np.where(np.isnan(mean), -1.0, np.clip(NOISE_FLOOR ** 2 / np.maximum(var, 1e-12), 0.0, 1.0))
        out["maps"] = {"shape": [rows, cols], "place": np.round(org.place, 3).ravel().tolist(),
                       "value": np.round(org.value_map, 3).ravel().tolist() if org.value_map is not None else None,
                       "people": np.round(people, 3).ravel().tolist(),
                       "mem_shape": list(mean.shape), "familiar": np.round(fam, 2).ravel().tolist()}
    mb = org.mb
    if mb.n_kc:
        food, _ = _int8(mb.weights, org.live_kc)
        danger, _ = _int8(mb.danger_weights, org.live_kc)
        # its real price, as organism._close charges it: nothing while its eyes
        # are shut (no Kenyon cells fire), and less on a run-down body
        share = 0.0
        if len(org.last_kc):
            seconds = max(1e-6, org.last_interval / max(1.0, org.fps))
            macs = kc_macs(org.live_kc) + (org.live_kc if org.aversive_rate > 0.0 else 0)
            share = THINK_COST * macs / REFERENCE_MACS * org.scarcity * LEGACY_UNIT / seconds / RESTING_BURN
            if org.body.degraded:
                share *= 0.5 + 0.5 * org.body.energy / EMPTY_G
        out["mb"] = {"n": mb.n_kc, "live": org.live_kc, "active": [int(k) for k in org.last_kc],
                     "food": food, "danger": danger,
                     "cost_share": round(share, 4)}
    # its latest replayed memory on its eye: each reactivated Kenyon cell's 7
    # receptor positions, back-projected onto its n x n gaze (not a picture it
    # makes: which parts of its eye the memory is built from)
    lr = org.last_replay
    if lr is not None and org.lived_s - lr[2] < 10.0 and mb.n_kc and len(lr[1]):
        kind, code, t, meta = lr
        n = org.state.n
        pos = mb.pos[np.asarray(code)[np.asarray(code) < mb.n_kc]].reshape(-1, 2)
        idx = np.clip(np.floor((pos + 0.5) * n).astype(int), 0, n - 1)
        grid = np.zeros((n, n))
        np.add.at(grid, (idx[:, 1], idx[:, 0]), 1.0)
        out["replay_eye"] = {"kind": kind, "age": round(org.lived_s - t, 1), "n": n,
                             "grid": np.round(grid / max(1.0, grid.max()), 3).ravel().tolist()}
        if org.imagery:  # its own reconstruction of the memory (its prototypes, summed)
            rec = mb.reconstruct(np.asarray(code)[np.asarray(code) < mb.n_kc])
            if rec is not None:
                out["replay_eye"]["recon"] = np.round(rec, 3).tolist()
                out["replay_eye"]["recon_side"] = PROTO_SIDE
        if meta is not None and meta[0] is not None:  # what it saw then: the frame, if the ring still holds it
            out["replay_eye"]["seen"] = {"i": int(meta[0]), "cx": round(meta[1], 4), "cy": round(meta[2], 4), "f": round(meta[3], 4)}
    b = org.body
    out["sleep"] = {"asleep": bool(b.asleep >= 0.5), "for_s": round(b.sleep_clock, 0), "pressure": round(b.sleep_pressure, 3),
                    "dreaming": bool(org.dreaming), "imagery": bool(org.imagery),
                    "mismatch": round(float(org.mismatch), 3), "woke_by": b.woke_by,
                    "recall": bool(org.recall), "recalled": org.recalled,
                    "sleep_set": org.sleep_set, "edits": [org.edits_kept, org.edits_tried],
                    "scene": org.scene + 1 if org.scenes else 1, "scenes": max(1, len(org.scenes)), "max_scenes": org.max_scenes,
                    "traits": {"awake": org.awake_replay, "asleep": org.sleep_replay, "rem": round(org.rem_share, 2),
                               "backup": round(org.replay_backup, 2), "dream_steps": org.dream_steps}}
    fps = max(1.0, org.fps)
    out["dreams"] = [[kind, r, c, round(org.lived_s - t, 1), seq] for kind, r, c, t, seq in org.replay_log
                     if org.lived_s - t < 10.0]  # ages in the seconds it lived (the frame rate varies)
    return out


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
        self.field_motion = None  # per-cell change its motion sense is fed, smoothed over a few frames (the viewer's heat)
        self._pursuit = None  # (gaze, pursued box, last step) of the host it is following, for the metrics
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
            _carry(old, new)
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
        batch = {"eating": [], "asleep": [], "alarm": [], "trajectory": [], "swat_acts": [],
                 "sums": collections.Counter()}
        looks = missed = 0
        with self.lock:
            org = self.org
            org.fps = fps
            self.field.fps = fps
            missed0 = org.missed
            replays0, seq0, dreams0, sips0, recalls0 = dict(org.replays), org.seq, org.dreams, org.sips, org.recalls
            switches0 = org.scene_switches
            tried0, kept0 = org.edits_tried, org.edits_kept
            img0 = list(org.imagery_sums)
            for j, (index, grey, boxes, colour, arrived) in enumerate(items):
                prev_v = self.field.last_vector
                sig, shift = self.field.step(grey)
                if prev_v is not None and self.field.last_vector is not None and prev_v.shape == self.field.last_vector.shape:
                    change = np.abs(self.field.last_vector - prev_v)
                    self.field_motion = change if self.field_motion is None or self.field_motion.shape != change.shape \
                        else 0.7 * self.field_motion + 0.3 * change
                swats0 = org.swats
                was_asleep = org.body.asleep >= 0.5
                org.feed_index = index
                out = org.frame(grey, sig, boxes or [], colour, shift)
                hosts, plants = prey_lib.hosts_only(boxes), prey_lib.plants_only(boxes)
                if was_asleep and org.body.asleep < 0.5:
                    batch["sums"]["woke_" + (org.body.woke_by or "choice")] += 1
                snack = org.pending["snack"] if out["gazed"] and org.pending else 0.0
                if out["gazed"]:
                    looks += 1
                    if not out["asleep"]:
                        self._competences(org, hosts, batch["sums"])
                    if out["eating"] > 0:
                        self.acts["meal"].append(index)
                    if snack > 0:
                        self.acts["snack"].append(index)
                prev = self.shown[-1] if self.shown else None
                rec = {"i": index, "cx": round(out["cx"], 4), "cy": round(out["cy"], 4), "f": round(out["extent"], 4),
                       "eat": round(float(out["eating"]), 3),
                       "snack": round(float(snack), 3) if out["gazed"] else (prev["snack"] if prev else 0.0),
                       "asleep": int(out["asleep"]), "alarm": int(org.last_alarm > 0.0), "boxes": hosts, "plants": plants,
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
            batch["sums"]["sips"] += org.sips - sips0
            for key, now_v, then_v in zip(("img_n", "img_sx", "img_sy", "img_sxx", "img_syy", "img_sxy"), org.imagery_sums, img0):
                batch["sums"][key] += now_v - then_v
            # what it replayed and dreamt (for the hourly metrics)
            for kind in ("awake", "nrem", "rem"):
                batch["sums"]["replay_" + kind] += org.replays[kind] - replays0.get(kind, 0)
            batch["sums"]["replay_sequences"] += org.seq - seq0
            batch["sums"]["dreams"] += org.dreams - dreams0
            batch["sums"]["recalls"] += org.recalls - recalls0
            batch["sums"]["scene_switches"] += org.scene_switches - switches0
            batch["sums"]["edits_tried"] += org.edits_tried - tried0
            batch["sums"]["edits_kept"] += org.edits_kept - kept0
            batch["pace"] = float(org.last_interval)
            batch["missed_share"] = missed / max(1, looks)
            self.metrics.add(batch, items[0][0], fps)

    def _competences(self, org: Organism, boxes: list, sums) -> None:
        """What the panel asked the week to measure (observation only), per
        waking look: does its gaze lead or follow a moving host; does hunger
        widen its diet to faint hosts; does its perception tree see hosts
        the detector misses ("this one must be wearing repellent")."""
        st = org.state
        g = (st.cx, st.cy)
        # 1. Pursuit: the host nearest its gaze, the same one as last look
        # (its box overlaps the last one), and how both moved since.
        near = min(boxes, key=lambda b: ((b[2] + b[4]) / 2 - g[0]) ** 2 + ((b[3] + b[5]) / 2 - g[1]) ** 2) if boxes else None
        prev = self._pursuit
        step = None
        if near is not None and prev is not None and prev[1] is not None:
            pb = prev[1]
            same = min(pb[4], near[4]) > max(pb[2], near[2]) and min(pb[5], near[5]) > max(pb[3], near[3])
            if same:
                dg = (g[0] - prev[0][0], g[1] - prev[0][1])
                dh = ((near[2] + near[4] - pb[2] - pb[4]) / 2, (near[3] + near[5] - pb[3] - pb[5]) / 2)
                step = (dg, dh)
                sums["pursuit_steps"] += 1
                sums["pursuit_same"] += dg[0] * dh[0] + dg[1] * dh[1]
                sums["pursuit_gg"] += dg[0] ** 2 + dg[1] ** 2
                sums["pursuit_hh"] += dh[0] ** 2 + dh[1] ** 2
                if prev[2] is not None:  # the previous step of the same pursuit
                    (pg, ph) = prev[2]
                    sums["pursuit_lead"] += pg[0] * dh[0] + pg[1] * dh[1]    # its gaze moved where the host then went
                    sums["pursuit_follow"] += dg[0] * ph[0] + dg[1] * ph[1]  # its gaze went where the host had gone
        self._pursuit = (g, near, step) if near is not None else None
        # 2. Diet breadth: at a bite, how faint the host was (its scent: confidence
        # x apparent size, as the prey sense weighs it), against its hunger.
        host = prey_lib.host_box_at_mouth(boxes, st.cx, st.cy, org.aspect)
        if host is not None and org.eating > 0:
            s = float(host[1]) * min(1.0, (host[4] - host[2]) * (host[5] - host[3]) / 0.02)
            h = float(org.body.hunger)
            for k, v in (("diet_n", 1), ("diet_s", s), ("diet_h", h), ("diet_ss", s * s), ("diet_hh", h * h), ("diet_hs", h * s)):
                sums[k] += v
        # 3. Its perception tree's verdict on this look vs the detector's.
        tree_yes = org.prev_response > 0.0  # its guess, 0.5 * (1 + response), above even
        yolo_yes = prey_lib.prey_in_window(boxes, st.cx, st.cy, *st.half_extents(org.aspect)) > 0.0
        sums["tree_looks"] += 1
        sums["tree_yes_yolo_no"] += int(tree_yes and not yolo_yes)
        sums["tree_no_yolo_yes"] += int(yolo_yes and not tree_yes)

    def _write(self) -> None:
        """The newest frames it lived, for the viewer (merged over live_status.json)."""
        with self.lock:
            shown = list(self.shown)
            body = {k: round(float(v), 4) for k, v in self.org.body.to_dict().items() if isinstance(v, (int, float))}
            pace = self.org.last_interval
            fps = self.org.fps
            circuits = _circuits(self.org)
            fm = None if self.field_motion is None else np.round(self.field_motion, 4).tolist()
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
            "prey_boxes": [r["boxes"] for r in shown], "plant_boxes": [r["plants"] for r in shown],
            "field_events": [r["ev"] for r in shown],
            "tree_guess": None, "teacher_label": None,  # graded only in evolution's runs
            "fovea_cx": shown[-1]["cx"], "fovea_cy": shown[-1]["cy"],
            "pace": pace, "frames_per_second": round(fps, 2), "body_now": body,
            **circuits,
            # what its wide-field motion sense is fed, per cell (not a picture: it
            # never senses per-cell brightness, only where and how much things change)
            "field_motion": fm, "field_motion_gain": PERIPH_MOTION_GAIN,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # the viewer keeps the last one; its life goes on
        self._written = time.time()
