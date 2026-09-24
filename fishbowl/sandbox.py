from __future__ import annotations

"""
Real, hard limits -- checked BEFORE each generation starts, not
trusted to self-regulate. See README's "fishbowl boundary". This is
the one module every run loop is required to go through for state
writes and limit checks; nothing else in this repo touches the
filesystem directly.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.json"
REQUESTS_PATH = STATE_DIR / "requests.json"
CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"
LIVE_STATUS_PATH = STATE_DIR / "live_status.json"


def save_live_status(data: dict) -> None:
    """
    A small, frequently-updated snapshot for anything polling this
    project's real-time state from outside (e.g. HLSLS's broadcast-
    api, see its own /api/cv-state precedent -- a sibling /api/
    cambrian-state route reads this file, never reaches into this
    process). Only ever derived numbers (fovea box, response
    magnitude, current fitness) -- same no-raw-frames boundary as
    everything else here, and cambrian-perception itself never opens
    a socket to serve this; something else reads the file.
    """
    _write_json_atomic(LIVE_STATUS_PATH, data)


def save_checkpoint(data: dict) -> None:
    """
    Real long-term memory, leveraging disk (cheap and abundant on
    Tanzania -- 224GB free, checked live) to make up for what RAM and
    CPU can't hold across restarts. Without this, every process start
    threw away all accumulated evolution and began again from a fresh
    random genome -- the wrong default for a machine meant to run this
    for a long time. Only ever contains derived state (genome, fitness
    history, habituation) -- never raw frames, same boundary as
    everything else this module writes.
    """
    _write_json_atomic(CHECKPOINT_PATH, data)


def load_checkpoint() -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    return _read_json(CHECKPOINT_PATH, None)


@dataclass
class Limits:
    max_generations: int = 5000
    max_wallclock_seconds: float = 1800.0
    max_tree_nodes: int = 60
    max_tree_depth: int = 10


class Sandbox:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.start_time = time.perf_counter()
        self.generation = 0
        self._ceiling_hits: dict[str, int] = {}
        self._recent_window: list[str] = []  # last 10 generations' ceiling-hit reasons, "" if none

    def should_continue(self) -> bool:
        if self.generation >= self.limits.max_generations:
            return False
        if time.perf_counter() - self.start_time >= self.limits.max_wallclock_seconds:
            return False
        return True

    def note_ceiling(self, reason: str | None) -> None:
        """
        reason: e.g. "max_tree_nodes" when a grow mutation couldn't
        apply because the tree was already at the size ceiling, or
        None if this generation didn't hit anything. Tracked over a
        rolling 10-generation window; 8+ hits in that window triggers
        a real, structured request (see log_request) rather than
        silently discarding the signal.
        """
        self._recent_window.append(reason or "")
        if len(self._recent_window) > 10:
            self._recent_window.pop(0)
        if reason:
            self._ceiling_hits[reason] = self._ceiling_hits.get(reason, 0) + 1
            recent_count = sum(1 for r in self._recent_window if r == reason)
            if recent_count >= 8:
                self.log_request(
                    f"hit {reason} on {recent_count} of the last {len(self._recent_window)} generations"
                )

    def log_request(self, text: str) -> None:
        """
        Purely observational -- see README. Nothing reads this back
        into the running process; it exists only for a human (or the
        future observation tool) to review.
        """
        entries = _read_json(REQUESTS_PATH, [])
        entries.append({
            "generation": self.generation,
            "elapsed_seconds": round(time.perf_counter() - self.start_time, 1),
            "text": text,
        })
        _write_json_atomic(REQUESTS_PATH, entries)

    def log_generation(self, record: dict) -> None:
        record = dict(record)
        record["generation"] = self.generation
        record["elapsed_seconds"] = round(time.perf_counter() - self.start_time, 1)
        entries = _read_json(EVOLUTION_LOG_PATH, [])
        entries.append(record)
        # Append-only, but capped -- an unbounded log file is itself a
        # real resource the sandbox should limit, same reasoning as
        # every other ceiling here.
        if len(entries) > 20000:
            entries = entries[-20000:]
        _write_json_atomic(EVOLUTION_LOG_PATH, entries)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, data) -> None:
    # The ONE whitelisted place any brain-influenced value ever
    # reaches disk from, always inside STATE_DIR, always via a temp
    # file + atomic replace so a killed process can never leave a
    # truncated, unparseable state file behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)
