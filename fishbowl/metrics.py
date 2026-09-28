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
        self.bites: list[float] = []
        self.turning = 0.0
        self.path = 0.0

    def add(self, live: dict, first_index: int | None, fps: float) -> None:
        """Count the survivor's frames that are newer than the last counted."""
        if first_index is None or not live:
            return
        eating, asleep, alarm = live.get("eating") or [], live.get("asleep") or [], live.get("alarm") or []
        traj = live.get("trajectory") or []
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
            **extra,
        }
        self._reset(now)
        return line
