"""
Hourly metrics (Gelman: measure before changing the world). One line an
hour in state/metrics.jsonl, from the lineage's own lived frames:

- bites: runs of consecutive frames with a host at its mouth; how many,
  how long (median, mean, longest), and the share of waking time spent biting;
- loops: how much its gaze winds (net turning / 2 pi) per waking minute,
  the "spirals", and how far it travels per waking minute;
- sleep, warnings (share of waking frames), and the traits and stores it
  had at the end of the hour.

Generations re-score overlapping windows of the feed, so only frames not
counted before are taken (by the feed's absolute frame index), as the
feeding record does. Observation only: nothing here feeds back into
fitness or the body.
"""

import collections
import math
import statistics
import time


class HourlyMetrics:
    def __init__(self, period_s: float = 3600.0):
        self.period_s = period_s
        self._reset(time.time())
        self.last_index = None   # the feed's last frame index already counted
        self.bite_run = 0        # frames in the bite still going on
        self.last_xy = None
        self.last_heading = None

    def _reset(self, now: float) -> None:
        self.started = now
        self.frames = self.awake = self.asleep = self.warn = self.biting = self.swats = 0
        self.look_sum = self.missed_sum = 0.0  # look interval (frames) and missed share, weighted by new frames
        self.sums = collections.Counter()  # the live actor's per-look competence sums (fishbowl/livelife.py)
        self.bites: list[float] = []
        self.turning = 0.0
        self.path = 0.0

    def add(self, live: dict, first_index: int | None, fps: float) -> None:
        """Count the survivor's frames that are newer than the last counted."""
        if first_index is None or not live:
            return
        eating, asleep, alarm = live.get("eating") or [], live.get("asleep") or [], live.get("alarm") or []
        traj = live.get("trajectory") or []
        self.sums.update(live.get("sums") or {})
        swat_frames = set(live.get("swat_acts") or [])
        n = len(eating)
        start = 0 if self.last_index is None else max(0, self.last_index + 1 - first_index)
        if start > 0 and self.last_index + 1 < first_index:  # a gap in the feed: a bite can't span it
            self._end_bite(fps)
            self.last_xy = self.last_heading = None
        for k in range(start, n):
            self.frames += 1
            sleeping = k < len(asleep) and asleep[k]
            if sleeping:
                self.asleep += 1
                self._end_bite(fps)
                self.last_xy = self.last_heading = None
                continue
            self.awake += 1
            if k < len(alarm) and alarm[k]:
                self.warn += 1
            if k in swat_frames:
                self.swats += 1
            if eating[k] > 0:
                self.biting += 1
                self.bite_run += 1
            else:
                self._end_bite(fps)
            if k < len(traj):
                self._move(traj[k][0], traj[k][1])
        new = max(0, n - start)
        self.look_sum += float(live.get("pace") or 0.0) * new
        self.missed_sum += float(live.get("missed_share") or 0.0) * new
        if n:
            self.last_index = first_index + n - 1

    def _end_bite(self, fps: float) -> None:
        if self.bite_run:
            self.bites.append(self.bite_run / max(1.0, fps))
            self.bite_run = 0

    def _move(self, x: float, y: float) -> None:
        if self.last_xy is not None:
            dx, dy = x - self.last_xy[0], y - self.last_xy[1]
            step = math.hypot(dx, dy)
            if step > 1e-4:  # it moved: its heading is defined
                self.path += step
                heading = math.atan2(dy, dx)
                if self.last_heading is not None:
                    self.turning += (heading - self.last_heading + math.pi) % (2 * math.pi) - math.pi
                self.last_heading = heading
        self.last_xy = (x, y)

    def _competences(self) -> dict:
        """Pursuit: correlation of its gaze's moves with the host's, same look
        (tracking), its move then the host's next (leading) and the host's
        move then its next (following). Diet: correlation of hunger with how
        faint a bitten host was (negative = hungrier takes fainter hosts).
        Tree: share of looks its tree saw a host the detector didn't, and
        the reverse."""
        s = self.sums
        norm = math.sqrt(s["pursuit_gg"] * s["pursuit_hh"])
        r = (lambda x: round(x / norm, 3)) if norm > 0 else (lambda x: None)
        n = s["diet_n"]
        diet = None
        if n > 2:
            vs, vh = s["diet_ss"] - s["diet_s"] ** 2 / n, s["diet_hh"] - s["diet_h"] ** 2 / n
            if vs > 0 and vh > 0:
                diet = round((s["diet_hs"] - s["diet_h"] * s["diet_s"] / n) / math.sqrt(vs * vh), 3)
        looks = max(1, s["tree_looks"])
        return {
            "pursuit_steps": int(s["pursuit_steps"]), "pursuit_same": r(s["pursuit_same"]),
            "pursuit_lead": r(s["pursuit_lead"]), "pursuit_follow": r(s["pursuit_follow"]),
            "diet_bites": int(n), "diet_hunger_vs_scent": diet,
            "diet_scent_mean": round(s["diet_s"] / n, 3) if n else None,
            "tree_sees_unlabelled": round(s["tree_yes_yolo_no"] / looks, 3),
            "tree_misses_labelled": round(s["tree_no_yolo_yes"] / looks, 3),
        }

    def due(self) -> bool:
        return time.time() - self.started >= self.period_s

    def flush(self, fps: float, extra: dict) -> dict | None:
        """The hour's line (None if it lived no frames), and a fresh hour."""
        now = time.time()
        if not self.frames:
            self._reset(now)
            return None
        waking_min = self.awake / max(1.0, fps) / 60.0
        b = self.bites
        line = {
            "t": round(now), "seconds": round(now - self.started), "frames": self.frames,
            "asleep_share": round(self.asleep / self.frames, 3),
            "bites": len(b),
            "bite_s_median": round(statistics.median(b), 2) if b else None,
            "bite_s_mean": round(statistics.fmean(b), 2) if b else None,
            "bite_s_max": round(max(b), 2) if b else None,
            "biting_share_awake": round(self.biting / max(1, self.awake), 3),
            "loops_per_awake_min": round(abs(self.turning) / (2 * math.pi) / waking_min, 3) if waking_min else None,
            "path_per_awake_min": round(self.path / waking_min, 3) if waking_min else None,
            "warn_share_awake": round(self.warn / max(1, self.awake), 4),
            "swats": self.swats,
            # Its latency (Gelman's condition for photoreceptor speed): seconds between
            # looks, and the share of looks missed because its brain was still thinking.
            "look_s": round(self.look_sum / self.frames / max(1.0, fps), 3),
            "missed_share": round(self.missed_sum / self.frames, 3),
            **self._competences(),
            **extra,
        }
        self._reset(now)
        return line
