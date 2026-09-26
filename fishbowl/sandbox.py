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
import os
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
# .jsonl, not .json -- see log_generation's own comment for the real
# disk-write-volume bug this fixes (external audit, 2026-09-24). Any
# reader expecting a single JSON array at the old evolution_log.json
# path needs updating to read one JSON object per line instead.
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
MAX_LOG_LINES = 20000
EVOLUTION_LOG_PREV_PATH = STATE_DIR / "evolution_log.1.jsonl"
REQUESTS_PATH = STATE_DIR / "requests.json"
CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"
# live_status.json is throwaway (rewritten constantly, only for the viewer):
# kept in RAM (/dev/shm) where available, so it costs no SSD writes (audit:
# ~25 GB/day when it was written to disk every generation).
_RUNTIME_DIR = Path(os.environ.get("CAMBRIAN_RUNTIME_DIR", "/dev/shm/cambrian-perception"))
LIVE_STATUS_PATH = (_RUNTIME_DIR if _RUNTIME_DIR.parent.is_dir() else STATE_DIR) / "live_status.json"
CHECKPOINT_PREV_PATH = STATE_DIR / "checkpoint.prev.json"
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
    _write_json_atomic(LIVE_STATUS_PATH, data, compact=True)


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
    # Keep the previous checkpoint as a fallback (audit: one unreadable
    # checkpoint used to silently start a brand-new lineage).
    if CHECKPOINT_PATH.exists():
        os.replace(CHECKPOINT_PATH, CHECKPOINT_PREV_PATH)
    _write_json_atomic(CHECKPOINT_PATH, data, durable=True)


class CheckpointUnreadable(RuntimeError):
    pass


def load_checkpoint() -> dict | None:
    """The checkpoint, else its previous copy. None only if neither file
    exists (a genuine first birth). If files exist but none can be read,
    raise -- never silently start a fresh lineage over a damaged one."""
    existing = [p for p in (CHECKPOINT_PATH, CHECKPOINT_PREV_PATH) if p.exists()]
    for p in existing:
        data = _read_json(p, None)
        if isinstance(data, dict) and "genome" in data:
            if p is CHECKPOINT_PREV_PATH:
                print("checkpoint.json unreadable -- resuming from checkpoint.prev.json")
            return data
    if existing:
        raise CheckpointUnreadable(f"checkpoint(s) exist but none are readable: {[str(p) for p in existing]}")
    return None


def clear_selected_source() -> None:
    """Drop the video choice (a chosen stream failed): back to the camera."""
    try:
        SELECTED_SOURCE_PATH.unlink()
    except FileNotFoundError:
        pass


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
        # Lines in the current log, counted once at start; rotation then
        # just renames the file (audit: the old rotation rewrote the whole
        # 34 MB log every ~75 s -- ~40 GB/day of SSD writes).
        try:
            with EVOLUTION_LOG_PATH.open("rb") as f:
                self._log_lines = sum(1 for _ in f)
        except OSError:
            self._log_lines = 0

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

        self._log_lines += 1
        if self._log_lines >= MAX_LOG_LINES:
            # Rotate by renaming: the current log becomes evolution_log.1
            # (replacing the older one) and a fresh one starts. No copy, no
            # rewrite; readers can stitch .1 + current for longer history.
            os.replace(EVOLUTION_LOG_PATH, EVOLUTION_LOG_PREV_PATH)
            self._log_lines = 0


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, data, compact: bool = False, durable: bool = False) -> None:
    # The ONE whitelisted place any brain-influenced value ever
    # reaches disk from (inside STATE_DIR, or the RAM runtime dir for the
    # live status), always via a temp file + atomic replace so a killed
    # process can never leave a truncated, unparseable state file behind.
    # durable=True (checkpoints) also fsyncs before the replace.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    text = json.dumps(data, separators=(",", ":")) if compact else json.dumps(data, indent=2)
    with open(temp, "w", encoding="utf-8") as f:
        f.write(text)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    temp.replace(path)
