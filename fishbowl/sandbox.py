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

# Which organism on this host. PARKED FEATURE (2026-09-30) -- a second
# organism per host; the full story, the steps to switch it on and the gaps
# still open are in docs/second-organism.md. Summary for a reader here:
#   - unset (every host today): the one organism; every path is what it
#     always was -- state/, /dev/shm/cambrian-perception/, port 8090,
#     cambrian-perception.service. Nothing in this file behaves differently.
#   - CAMBRIAN_INSTANCE=b (set only by the parked units in
#     deploy/second-organism/): a second organism with its OWN state-b/,
#     /dev/shm/cambrian-perception-b/, viewer on 8091 and
#     cambrian-perception-b.service; the code, venv and models are shared.
#   - only "b" is supported (tools/viewer.py DEFAULT_PORT, tools/fleet.py's
#     "siblings" probe); a third would need a port scheme first.
# Parked by a resource panel (Gregg, Poettering, Russinovich, Gelman): only
# one host of four had room, and there the two would share one USB disk.
# The derivations below are repeated in tools/viewer.py and
# tools/resource_handler.py (which don't import this module): keep them equal.
INSTANCE = os.environ.get("CAMBRIAN_INSTANCE", "").strip()
STATE_DIR = Path(__file__).resolve().parents[1] / ("state" if not INSTANCE else f"state-{INSTANCE}")
# .jsonl, not .json -- see log_generation's own comment for the real
# disk-write-volume bug this fixes (external audit, 2026-09-24). Any
# reader expecting a single JSON array at the old evolution_log.json
# path needs updating to read one JSON object per line instead.
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
MAX_LOG_LINES = 20000
EVOLUTION_LOG_PREV_PATH = STATE_DIR / "evolution_log.1.jsonl"
REQUESTS_PATH = STATE_DIR / "requests.json"
CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"
# A deliberate stop, portably (docs/packaging.md): the run finishes the
# generation it is on, saves its checkpoint and exits -- the file form of
# systemd's SIGTERM, for supervisors that can't signal (Task Scheduler).
STOP_REQUEST_PATH = STATE_DIR / "stop.request"
RANDOMIZE_REQUEST_PATH = STATE_DIR / "randomize.request"
AMNESIA_REQUEST_PATH = STATE_DIR / "amnesia.request"  # the owner's Amnesia: a fresh install's founder (tools/reset_founder.py)
RESET_FOUNDER_REQUEST_PATH = STATE_DIR / "reset_founder.request"  # the owner's Reset to founder
FOUNDER_PATH = STATE_DIR / "founder.json"                  # its founder's checkpoint, as it was at birth
FOUNDER_SEEDED_PATH = STATE_DIR / "founder_seeded.txt"     # and the seed sets its genome held then
EXIT_RESTART_ME = 75  # EX_TEMPFAIL: its supervisor (systemd, cambrian_service.py) starts it again  # the owner's Randomize (viewer's brain card): a founder's brain, drawn afresh
# live_status.json is throwaway (rewritten constantly, only for the viewer):
# kept in RAM (/dev/shm) where available, so it costs no SSD writes (audit:
# ~25 GB/day when it was written to disk every generation).
_RUNTIME_DIR = Path(os.environ.get("CAMBRIAN_RUNTIME_DIR", "/dev/shm/cambrian-perception" + (f"-{INSTANCE}" if INSTANCE else "")))
LIVE_STATUS_PATH = (_RUNTIME_DIR if _RUNTIME_DIR.parent.is_dir() else STATE_DIR) / "live_status.json"
CHECKPOINT_PREV_PATH = STATE_DIR / "checkpoint.prev.json"
SELECTED_SOURCE_PATH = STATE_DIR / "selected_source.json"
HANDLER_STATE_PATH = STATE_DIR / "handler_state.json"


METRICS_PATH = STATE_DIR / "metrics.jsonl"


def append_metrics(line: dict) -> None:
    """One hour's metrics (fishbowl/metrics.py): an O(1) append, ~24 lines a day."""
    try:
        with open(METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
    except OSError as e:
        print(f"metrics: could not append ({e})")


def load_quota_pct(default: float) -> float:
    """The CPU quota resource_handler.py last actually set (its own
    written record) -- read-only, used to price the look's size."""
    data = _read_json(HANDLER_STATE_PATH, None) if HANDLER_STATE_PATH.exists() else None
    try:
        return float(data["last_quota_pct"])
    except (TypeError, KeyError, ValueError):
        return float(default)


EPISODES_PATH = STATE_DIR / "episodes.npz"
CORTEX_PATH = STATE_DIR / "cortex.json"
EGGS_DIR = STATE_DIR / "eggs"  # the eggs its live body laid (livelife.LiveLife._lay): one genome each
LIFE_HISTORY_PATH = STATE_DIR / "life_history.json"  # this host's lives: founded, died (of what, how old, how many eggs), hatched, extinct
LIFE_HISTORY_PREV_PATH = STATE_DIR / "life_history.prev.json"  # its previous copy (a power cut mid-write loses one record, not the history)
TALLY_PATH = STATE_DIR / "tally.json"
EXPERIMENTS_PATH = STATE_DIR / "experiments.json"  # this host's trials, e.g. {"ram_feeding": true} (the host's, kept across lineages)  # its life's good and bad events since its birth (livelife.Tally)


def load_cortex() -> dict | None:
    """Its visual cortex's library of individuals (cortex.py), or None."""
    return _read_json(CORTEX_PATH, None) if CORTEX_PATH.exists() else None


def save_cortex(data: dict) -> None:
    """Numbers only (heights, cadences, speeds, colour histograms): temp file + atomic replace."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = CORTEX_PATH.with_name("cortex.tmp.json")
    temp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for attempt in range(20):
        try:
            temp.replace(CORTEX_PATH)
            return
        except PermissionError:
            time.sleep(0.05)


def save_episodes(arrays: dict) -> None:
    """The live organism's episodes and sleep test set (livelife.py): numbers
    only -- Kenyon-cell codes, rewards, field cells, and its sample of looks
    as 8-bit receptor levels (its own coarse retina, not camera frames). Temp
    file + atomic replace, like everything else written here."""
    import numpy as np
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = EPISODES_PATH.with_name("episodes.tmp.npz")
    with open(temp, "wb") as f:  # durable, as a checkpoint is: its memories survive a power cut too
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):
        try:
            temp.replace(EPISODES_PATH)
            _fsync_dir(EPISODES_PATH)
            return
        except PermissionError:
            time.sleep(0.05)


EVENTS_PATH = STATE_DIR / "events.jsonl"
EVENTS_MAX_BYTES = 5_000_000  # then it rotates to events.1.jsonl (one old file kept): a disk-wear bound


def experiment(name: str) -> bool:
    """Whether this host runs a trial (state/experiments.json), for its live
    body and its evaluations alike, so evolution scores what its body lives."""
    try:
        return bool((_read_json(EXPERIMENTS_PATH, {}) or {}).get(name, False)) if EXPERIMENTS_PATH.exists() else False
    except (OSError, ValueError, AttributeError):
        return False


SETTINGS_PATH = Path(__file__).resolve().parents[1] / "cambrian.json"  # the host's own settings, written by the installers


def hive() -> bool:
    """Whether this host's organisms are in the hive (a 2026-09-30 panel --
    Ostrom: joining a shared pool is the owner's choice, so it is opt-in;
    Schneier: out of it means it neither fetches genomes nor serves its own).
    cambrian.json's "hive", set by an installer's --hive; absent, it is solo."""
    try:
        return bool((_read_json(SETTINGS_PATH, {}) or {}).get("hive", False))
    except AttributeError:
        return False


def life_history() -> list:
    """This host's lives, oldest first (a list of records) -- from its previous
    copy if the file itself was damaged (a power cut mid-write)."""
    for p in (LIFE_HISTORY_PATH, LIFE_HISTORY_PREV_PATH):
        data = _read_json(p, None) if p.exists() else None
        if isinstance(data, list):
            return data
    return []


def record_life(entry: dict) -> None:
    """Appends one record to this host's life history (founded, died, hatched,
    extinct). Durable, with the previous copy kept, as a checkpoint is: its
    history is what migration and the development record stand on."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        hist = life_history()
        if LIFE_HISTORY_PATH.exists():
            os.replace(LIFE_HISTORY_PATH, LIFE_HISTORY_PREV_PATH)
        _write_json_atomic(LIFE_HISTORY_PATH, hist + [{"t": round(time.time(), 1), **entry}], durable=True)
    except OSError:
        pass


def log_event(entry: dict) -> None:
    """One rare event of the live organism (livelife.py), appended."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if EVENTS_PATH.exists() and EVENTS_PATH.stat().st_size > EVENTS_MAX_BYTES:
            EVENTS_PATH.replace(EVENTS_PATH.with_name("events.1.jsonl"))
        with open(EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # a log line is never worth its life


def load_space(default: int) -> int:
    """How many worker bodies resource_handler.py last allowed it (its
    "space": shrunk when memory is short) -- read-only."""
    data = _read_json(HANDLER_STATE_PATH, None) if HANDLER_STATE_PATH.exists() else None
    try:
        return max(1, int(data["last_space"]))
    except (TypeError, KeyError, ValueError):
        return int(default)


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


class CheckpointUnreadable(RuntimeError):  # kept for anything that imports it; no longer raised
    pass


def load_checkpoint() -> dict | None:
    """The checkpoint, else its previous copy. None only if no checkpoint
    exists anywhere (a genuine first birth).

    Reboot-hardened (2026-09-30: like a game that loses some play but never
    the save): if both are damaged -- a power cut mid-write on a disk that
    reordered it -- it neither crash-loops (it used to raise, and systemd
    restarted it into the same error forever) nor silently starts over. The
    damaged files are moved aside to state/damaged-<time>/ for a human, and
    its lineage restarts from its own founder (state/founder.json, saved at
    its birth); only if that is unreadable too does a new founder start.
    Which, and why, goes in its life history. (Two good copies failing at
    once needs the disk to lose a file it wasn't writing: rare.)"""
    existing = [p for p in (CHECKPOINT_PATH, CHECKPOINT_PREV_PATH) if p.exists()]
    for p in existing:
        data = _read_json(p, None)
        if isinstance(data, dict) and "genome" in data:
            if p is CHECKPOINT_PREV_PATH:
                print("checkpoint.json unreadable -- resuming from checkpoint.prev.json")
            return data
    if not existing:
        return None
    aside = STATE_DIR / time.strftime("damaged-%Y%m%d-%H%M%S")
    try:
        aside.mkdir(parents=True, exist_ok=True)
        for p in existing:
            os.replace(p, aside / p.name)
    except OSError as e:
        print(f"Its damaged checkpoints couldn't be moved aside ({e}).")
    # Only this lineage's own founder: state/backup-*/ hold the lineage a reset
    # or a fleet clone REPLACED, so resuming from one would bring back another organism.
    data = _read_json(FOUNDER_PATH, None) if FOUNDER_PATH.exists() else None
    if isinstance(data, dict) and "genome" in data:
        print(f"Both checkpoints were damaged (kept in {aside.name}/) -- its lineage restarts from its founder.")
        record_life({"event": "resumed after damage", "from": "founder", "damaged": aside.name})
        return data
    print(f"Both checkpoints were damaged (kept in {aside.name}/) and its founder is unreadable -- a new founder starts.")
    record_life({"event": "resumed after damage", "from": None, "damaged": aside.name})
    return None


SOURCE_FAILURES_PATH = STATE_DIR / "source_failures.json"
SOURCE_FAILURES_TO_DROP = 3  # a chosen stream is dropped after this many failed runs in a row


def _resolves(url: str) -> bool:
    """Whether the stream's host name resolves now (no name: assume it does)."""
    import socket
    from urllib.parse import urlparse
    host = urlparse(url).hostname
    if not host:
        return True
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


def source_failed(url: str) -> bool:
    """A chosen stream failed to open: count it, and say whether to give up
    on it. One timeout (a network hiccup) used to drop a human's choice for
    good; now only a stream that fails SOURCE_FAILURES_TO_DROP runs in a row
    (about a minute apart, as the supervisor restarts it) is dropped.
    And a failure while the machine has no network at all (its stream's host
    doesn't resolve: e.g. just after a reboot, before the network is up) is
    the machine's, not the stream's: not counted."""
    if not _resolves(url):
        return False
    seen =_read_json(SOURCE_FAILURES_PATH, {}) if SOURCE_FAILURES_PATH.exists() else {}
    count = (int(seen.get("count", 0)) if seen.get("url") == url else 0) + 1
    _write_json_atomic(SOURCE_FAILURES_PATH, {"url": url, "count": count})
    return count >= SOURCE_FAILURES_TO_DROP


def source_opened() -> None:
    """A chosen stream opened: its failure count starts over."""
    SOURCE_FAILURES_PATH.unlink(missing_ok=True)


def clear_selected_source(url: str | None = None) -> None:
    """Drop the video choice (a chosen stream failed): back to the camera.
    With url, only if that is still the choice -- a run must never wipe a
    newer choice made while it was watching the old one."""
    try:
        if url is not None:
            current = _read_json(SELECTED_SOURCE_PATH, None) if SELECTED_SOURCE_PATH.exists() else None
            if not current or current.get("url") != url:
                return
        SELECTED_SOURCE_PATH.unlink()
    except FileNotFoundError:
        pass


def load_selected_source() -> dict | None:
    """
    Human-only control: which video it watches. Written by tools/viewer.py itself (its own
    independent writer, not through this module -- same file-is-the-
    contract pattern as live_status.json), read here. The organism
    never touches this; the viewer only accepts verified-live YouTube links.
    """
    if not SELECTED_SOURCE_PATH.exists():
        return None
    return _read_json(SELECTED_SOURCE_PATH, None)


@dataclass
class Limits:
    max_generations: int = 5000
    max_wallclock_seconds: float = 1800.0
    # Raised from 60/10 (plenty of memory to be generous with nodes and
    # depth). Worth being precise about what this actually
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
    # Raised again 300/20 -> 1000/30 (prompted by a real accepted genome
    # still being tiny --
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
        self.run_start_generation = 0
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
        # max_generations is PER RUN: the generation counter itself is the
        # lifetime total restored from the checkpoint (it once hit 200000
        # and every restart stopped immediately).
        # 0 = no limit: it lives on, one long process (a 2026-09-29 panel -- the
        # old hourly bounded runs wiped everything it held in memory each hour)
        if self.limits.max_generations and self.generation - self.run_start_generation >= self.limits.max_generations:
            return False
        if self.limits.max_wallclock_seconds and time.perf_counter() - self.start_time >= self.limits.max_wallclock_seconds:
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


CHAMPION_PATH = LIVE_STATUS_PATH.with_name("champion.json")


def save_champion(data: dict) -> None:
    """Its current champion (genome, meal criterion, where it lives), for a
    host that runs the organism live (fishbowl/live.py, e.g. a livecam's CV):
    world-readable, next to the live status, rewritten when it changes."""
    _write_json_atomic(CHAMPION_PATH, data, compact=True)
    try:
        os.chmod(CHAMPION_PATH, 0o644)
    except OSError:
        pass


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
    _replace(temp, path, durable)


def _fsync_dir(path: Path) -> None:
    """After a rename, the folder's own entry to disk too (POSIX): without it
    a power cut can undo the rename, or keep the name with the old data."""
    if os.name != "posix":
        return
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _replace(temp: Path, path: Path, durable: bool) -> None:
    # Windows refuses to replace a file another process has open this instant
    # (the viewer or `cambrian --status` reading it): "Access is denied". It
    # crashed the organism on hera 22 times. Retry briefly; a live status
    # that still can't land is skipped (the next comes within a second), a
    # checkpoint is not.
    for attempt in range(20):
        try:
            temp.replace(path)
            if durable:
                _fsync_dir(path)
            return
        except PermissionError:
            if attempt == 19:
                if durable:
                    raise
                return
            time.sleep(0.05)
