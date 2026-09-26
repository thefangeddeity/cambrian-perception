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
# .jsonl, not .json -- see log_generation's own comment for the real
# disk-write-volume bug this fixes (external audit, 2026-09-24). Any
# reader expecting a single JSON array at the old evolution_log.json
# path needs updating to read one JSON object per line instead.
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
MAX_LOG_LINES = 20000
_ROTATE_CHECK_INTERVAL = 500
REQUESTS_PATH = STATE_DIR / "requests.json"
CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"
LIVE_STATUS_PATH = STATE_DIR / "live_status.json"
SELECTED_SOURCE_PATH = STATE_DIR / "selected_source.json"
HANDLER_STATE_PATH = STATE_DIR / "handler_state.json"


def load_quota_pct(default: float) -> float:
    """The CPU quota resource_handler.py last actually set (its own
    written record) -- read-only, used to price the look's size."""
    data = _read_json(HANDLER_STATE_PATH, None) if HANDLER_STATE_PATH.exists() else None
    try:
        return float(data["last_quota_pct"])
    except (TypeError, KeyError, ValueError):
        return float(default)


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


def load_selected_source() -> dict | None:
    """
    Human-only control, User: "put a list of training videos I can pick
    from the viewer." Written by tools/viewer.py itself (its own
    independent writer, not through this module -- same file-is-the-
    contract pattern as live_status.json), read here. The organism
    never touches this; the viewer only ever offers a fixed whitelist.
    """
    if not SELECTED_SOURCE_PATH.exists():
        return None
    return _read_json(SELECTED_SOURCE_PATH, None)


@dataclass
class Limits:
    max_generations: int = 5000
    max_wallclock_seconds: float = 1800.0
    # Raised from 60/10 -- User: "with 5GB we can be generous with its
    # nodes and depth." Worth being precise about what this actually
    # trades off, though: a Node is a small Python object, so even a
    # tree of several hundred nodes is a few hundred KB at most --
    # memory was never really the constraint on tree SIZE (it was the
    # constraint on the earlier video-frame-storage bug, a completely
    # different part of the system, already fixed). The real cost of
    # a bigger ceiling is CPU time (more nodes evaluated per frame,
    # per channel, per generation) -- already governed separately by
    # resource_handler.py's own hunger/disgust CPUQuota adjustment, so
    # raising this is safe to do generously; the resource handler is
    # what actually keeps real runtime cost in check, not this number.
    #
    # Raised again 300/20 -> 1000/30, User: "please raise the tree
    # ceiling" (prompted by a real accepted genome still being tiny --
    # response was a single leaf node -- after ~18k generations). Worth
    # being honest here too: checked live (evolution_log.json), the
    # actual accepted trees were nowhere near 300 nodes when this was
    # raised -- the real bottleneck was a mislabeling bug (see
    # genome.py's mutate_task), not this ceiling. Raising it further
    # doesn't fix that, but it removes any doubt that a real ceiling
    # is what's limiting growth, and costs nothing extra (same
    # reasoning as above -- a Node is tiny, CPU is the real governed
    # cost, not node count).
    max_tree_nodes: int = 1000
    max_tree_depth: int = 30


class Sandbox:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.start_time = time.perf_counter()
        self.generation = 0
        self._ceiling_hits: dict[str, int] = {}
        self._recent_window: list[str] = []  # last 10 generations' ceiling-hit reasons, "" if none
        self._gens_since_rotate_check = 0

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
            "unix": time.time(),
            "text": text,
        })
        _write_json_atomic(REQUESTS_PATH, entries)

    def log_generation(self, record: dict) -> None:
        """
        Real bug fix (external audit, 2026-09-24): this used to read
        the ENTIRE log, append one record, and rewrite the whole thing
        -- every single generation. Measured live: at the 20000-entry
        cap that file was 15.6MB, rewritten on every generation, which
        at the deployed generation rate worked out to roughly 0.8TB/day
        of real disk writes -- enough to wear out a consumer SSD in
        months on a service meant to run for YEARS. Now an O(1) append
        of one JSON line; the only operation that still does a full
        rewrite is the rare rotation below, not every generation.
        """
        record = dict(record)
        record["generation"] = self.generation
        record["elapsed_seconds"] = round(time.perf_counter() - self.start_time, 1)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with EVOLUTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._gens_since_rotate_check += 1
        if self._gens_since_rotate_check >= _ROTATE_CHECK_INTERVAL:
            self._gens_since_rotate_check = 0
            self._rotate_log_if_needed()

    def _rotate_log_if_needed(self) -> None:
        # Checked only every _ROTATE_CHECK_INTERVAL generations, not
        # every one -- an unbounded log file is still a real resource
        # to cap (same reasoning as every other ceiling here), but the
        # cap doesn't need enforcing on every single append.
        if not EVOLUTION_LOG_PATH.exists():
            return
        lines = EVOLUTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
        if len(lines) <= MAX_LOG_LINES:
            return
        kept = lines[-MAX_LOG_LINES:]
        temp = EVOLUTION_LOG_PATH.with_suffix(".tmp")
        temp.write_text("\n".join(kept) + "\n", encoding="utf-8")
        temp.replace(EVOLUTION_LOG_PATH)


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
