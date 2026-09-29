#!/usr/bin/env python3
from __future__ import annotations

"""
The external "handler" (its feeding tube) that decides how
much real compute cambrian-perception.service gets, adjusted live via
systemd (`systemctl set-property ... CPUQuota=`). This is deliberately
OUTSIDE the sandboxed brain -- same status as any other tools/ script,
never something the organism itself can invoke or influence directly.
The organism can only ever leave a trail (evolution_log.json's real
fitness history, requests.json's real ceiling-hit requests); this
script is the only thing that ever turns that trail into a resource
decision.

Three real, distinct mechanisms, not one knob:

  HARD CEILING (never negotiable) -- MIN_QUOTA_PCT/MAX_QUOTA_PCT
    below. Nothing here can push past them regardless of any other
    signal: a permissive feeder that still cuts off food when it's bad
    for the host, however hungry the organism is.

  IDLE (use what nobody else is using) -- quota moves UP a core at a
    time while a whole core sits idle beyond what it already has, and
    DOWN a core when other work needs them back. Needs no request and
    no fitness gain: an idle core costs the host nothing.

  HUNGER (reward for real, demonstrated cognition) -- quota moves UP,
    within the ceiling, when evolution_log.json shows real recent
    fitness improvement. Never based on the organism merely asking.

  DISGUST / WASTE (the "own den" signal, unconditional) -- quota moves
    DOWN, regardless of hunger, when REAL system telemetry (load
    average, free memory) shows the shared machine is under genuine
    strain. Never based on anything the organism reports about
    itself -- unfakeable from inside the sandbox.

  HUNGER EXTINCTION -- not an eternally hungry organism: if it keeps
    requesting resources without real improvement, the handler's
    responsiveness decays (it gets less hungry the more it uselessly
    asks), and only real new "food" (fitness improvement) resets it.
    Habituation's shape, applied to requests: repeated
    ceiling-hit requests (state/requests.json) with NO accompanying
    real fitness gain progressively dampen how much this handler
    responds to further requests -- an unreinforced "give me more"
    fades on its own rather than needing to be externally suppressed
    every cycle. A genuine new fitness gain resets responsiveness
    close to full immediately.

Meant to run periodically (systemd timer, e.g. every 15 min), not as
a long-lived daemon -- each invocation reads real state, makes one
bounded adjustment, and exits.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
REQUESTS_PATH = STATE_DIR / "requests.json"
HANDLER_STATE_PATH = STATE_DIR / "handler_state.json"

SERVICE_NAME = "cambrian-perception.service"
# Windows (docs/packaging.md): no systemd; tools/cambrian_service.py applies
# the quota recorded in handler_state.json as a Job Object hard CPU cap, and
# the measurements come from the Win32 API instead of /proc.
WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
# Only Linux has systemd's quota; elsewhere the supervisor applies what this
# records (a Job Object hard cap on Windows; priority only on macOS).
SYSTEMD = sys.platform.startswith("linux")

MIN_QUOTA_PCT = 50    # hard floor -- never starve it completely
# Hard ceiling: every core but one, regardless of any signal below -- the
# host always keeps a core to itself. Raised from 3 of 8 cores (300%) on
# 2026-09-26 as a deliberate decision, together with run_vision.py scoring
# several children at once on the cores it is granted. Change this only as
# a deliberate, written decision, not a tuning knob to nudge casually.
MAX_QUOTA_PCT = max(100, ((os.cpu_count() or 2) - 1) * 100)
# A host may set a lower ceiling for itself, as its own written decision:
# state/host_limits.json {"max_quota_pct": ...} -- e.g. a laptop someone
# uses, which throttles when hot (2026-09-28 infrastructure review). It can
# only lower the ceiling, never raise it.
HOST_LIMITS_PATH = STATE_DIR / "host_limits.json"
IDLE_STEP_PCT = 100   # IDLE moves a whole core at a time
STEP_PCT = 25         # bounded step per invocation, up or down --
# no single run can swing the quota far, so a bad read of the signals
# costs at most one small step, not a lurch.

# Real, external, unfakeable-from-inside-the-sandbox strain thresholds.
LOAD_STRAIN_PER_CORE = 1.3   # loadavg / nproc above this = real strain
FREE_MEM_STRAIN_MB = 800     # system-wide free memory below this = real strain

HUNGER_DECAY = 0.85  # EMA decay for the extinction/satiation signal --
# tuned fast since this runs far
# less often (per-invocation of this script, not per-video-frame).


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


_DURATION_SUFFIXES = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0}


def _current_quota_pct() -> int:
    if not SYSTEMD:
        return int(_read_json(HANDLER_STATE_PATH, {}).get("last_quota_pct", 150))
    result = subprocess.run(
        ["systemctl", "show", SERVICE_NAME, "-p", "CPUQuotaPerSecUSec", "--value"],
        capture_output=True, text=True,
    )
    value = result.stdout.strip()
    if value in ("", "infinity"):
        return MAX_QUOTA_PCT  # unset means unlimited -- treat as already at ceiling

    # Real format, checked live (not assumed): "1.500000s" for a 150%
    # quota, not "150ms" -- the accounting period is 1 real second, so
    # the value IN SECONDS is directly the fractional quota (1.5s of
    # CPU time per 1s wall-clock = 150%). Suffix varies by magnitude
    # (systemd picks the largest convenient unit), so this converts
    # whatever real suffix comes back to seconds first rather than
    # assuming one specific format.
    for suffix, seconds_per_unit in sorted(_DURATION_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if value.endswith(suffix):
            number = float(value[: -len(suffix)])
            return int(round(number * seconds_per_unit * 100.0))
    return MAX_QUOTA_PCT


def _set_quota_pct(pct: int) -> None:
    if not SYSTEMD:
        return  # recorded in handler_state.json; the supervisor applies it
    # sudo, not a root-owned service -- this script runs as the same
    # unprivileged user as everything else here, using the same
    # already-verified passwordless sudo rule the initial deployment
    # used, scoped to exactly this one real command.
    subprocess.run(
        ["sudo", "systemctl", "set-property", SERVICE_NAME, f"CPUQuota={pct}%"],
        check=True,
    )


def _memory_limits() -> tuple[int, int] | None:
    """Linux (a 2026-09-29 panel; Poettering, Russinovich): its memory limits
    from what the host has, not static numbers. MemoryHigh (where the kernel
    starts reclaiming from it) = what the service holds now + what the host can
    still give before reaching this handler's own strain line; MemoryMax (the
    leak guard) = the host's RAM less that line. MB."""
    try:
        info = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines() if ":" in l}
        now = int(Path(f"/sys/fs/cgroup/system.slice/{SERVICE_NAME}/memory.current").read_text()) // 1048576
    except (OSError, ValueError, KeyError, IndexError):
        return None
    total, avail = info["MemTotal"] // 1024, info["MemAvailable"] // 1024
    ceiling = max(256, total - FREE_MEM_STRAIN_MB)
    return min(ceiling, now + max(0, avail - FREE_MEM_STRAIN_MB)), ceiling


def _set_memory_limits(high_mb: int, max_mb: int) -> None:
    subprocess.run(["sudo", "systemctl", "set-property", SERVICE_NAME, f"MemoryHigh={high_mb}M", f"MemoryMax={max_mb}M"], check=True)


def _tail_jsonl(path: Path, max_lines: int, max_bytes: int = 2_000_000) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode("utf-8", errors="replace").splitlines()[-max_lines:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial first line from the byte-offset seek
    return out


def _recent_fitness_improved(window: int = 200) -> bool:
    # Real bug fixed 2026-09-25: this read evolution_log.json, which
    # stopped existing when sandbox.py moved to evolution_log.jsonl --
    # so it returned False every run and the quota could never grow.
    # best_fitness is also no longer a ratchet (re-scored per clip), so
    # "last > first" was noise anyway. "New food" = a real accepted
    # candidate that actually beat its parent.
    for e in _tail_jsonl(EVOLUTION_LOG_PATH, window):
        f, p = e.get("fitness"), e.get("parent_fitness")
        if e.get("accepted") and f is not None and p is not None and f > p + 1e-6:
            return True
    return False


def _recent_request_count(since_seconds: float = 1800.0) -> int:
    requests = _read_json(REQUESTS_PATH, [])
    if not requests:
        return 0
    stamped = [r for r in requests if "unix" in r]
    if stamped:
        cutoff = time.time() - since_seconds
        return sum(1 for r in stamped if r["unix"] >= cutoff)
    # requests.json entries carry their own elapsed_seconds from THAT
    # run's own start, not a wall-clock timestamp -- real, honest
    # limitation: this counts requests in the tail of the log instead
    # of a true time window, since sandbox.py doesn't stamp wall-clock
    # time (see its own log_request()). Close enough for a coarse,
    # periodic handler decision, not pretended to be more precise.
    return len(requests[-20:])


def _idle_cores(current_pct: int) -> float:
    """Cores sitting idle right now beyond one kept for the host, measured
    from /proc/stat over a second (not the load average, which also counts
    tasks waiting on I/O -- a USB camera keeps it high on idle cores). The
    organism's own use is busy time, so this is what nobody is using."""
    if MACOS:  # top's second sample: "CPU usage: 4.1% user, 3.2% sys, 92.7% idle"
        try:
            out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"], capture_output=True, text=True).stdout
            idle_pct = float([l for l in out.splitlines() if l.startswith("CPU usage")][-1].split(",")[2].split("%")[0])
        except (OSError, ValueError, IndexError):
            return 0.0
        return idle_pct / 100.0 * (os.cpu_count() or 1) - 1.0

    def sample():
        if WINDOWS:  # GetSystemTimes: idle, kernel (which includes idle), user
            import ctypes
            idle, kernel, user = (ctypes.c_uint64() for _ in range(3))
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            return idle.value, kernel.value + user.value
        fields = [int(x) for x in Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]]
        return fields[3] + fields[4], sum(fields)  # idle + iowait, total
    try:
        i0, t0 = sample()
        time.sleep(1.0)
        i1, t1 = sample()
    except (OSError, ValueError, IndexError):
        return 0.0  # can't tell: change nothing
    nproc = os.cpu_count() or 1
    return (i1 - i0) / max(1, t1 - t0) * nproc - 1.0


def _load_average_strain() -> bool:
    if WINDOWS:
        return False  # no load average; the idle-core measure carries this
    load1, _, _ = os.getloadavg()
    nproc = os.cpu_count() or 1
    return (load1 / nproc) > LOAD_STRAIN_PER_CORE


def _free_memory_strain() -> bool:
    if MACOS:  # free + inactive pages (vm_stat), the memory macOS can hand out
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(out.split("page size of ")[1].split()[0])
            pages = {l.split(":")[0].strip(): int(l.split(":")[1].strip().rstrip(".")) for l in out.splitlines()[1:] if ":" in l}
            return (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)) * page / 1024 / 1024 < FREE_MEM_STRAIN_MB
        except (OSError, ValueError, IndexError):
            return False
    if WINDOWS:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + \
                       [(n, ctypes.c_uint64) for n in ("total", "avail", "totalPage", "availPage",
                                                       "totalVirtual", "availVirtual", "availExtended")]
        m = _Mem()
        m.dwLength = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return False
        return (m.avail / 1024 / 1024) < FREE_MEM_STRAIN_MB
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return False  # not on Linux, or unreadable -- don't fabricate strain
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            return (kb / 1024.0) < FREE_MEM_STRAIN_MB
    return False


def run(dry_run: bool = False) -> None:
    handler_state = _read_json(HANDLER_STATE_PATH, {"satiation": 0.0})
    satiation = float(handler_state.get("satiation", 0.0))

    fitness_improved = _recent_fitness_improved()
    request_count = _recent_request_count()
    cpu_strained, mem_strained = _load_average_strain(), _free_memory_strain()
    strained = cpu_strained

    # Hunger extinction. Real new food
    # resets it sharply; repeated unaccompanied requesting decays it
    # toward full satiation (low responsiveness) instead of staying
    # perpetually agitated.
    if fitness_improved:
        satiation *= 0.3
    elif request_count > 0:
        satiation = HUNGER_DECAY * satiation + (1.0 - HUNGER_DECAY) * 1.0
    responsiveness = 0.1 + 0.9 * (1.0 - satiation)

    current = _current_quota_pct()
    target = current
    reason = "no change"
    # Two kinds of strain, two levers (2026-09-28 infrastructure review): a
    # busy CPU thins its "oxygen" (the quota, below); short memory shrinks its
    # "space" -- how many worker bodies it may run at once -- and leaves the
    # quota alone (cutting CPU frees no memory: 7elwe sat at half a core for
    # hours with 7-10 cores idle while a browser used the RAM).
    space_cap = max(1, (os.cpu_count() or 2) - 1)
    state_now = _read_json(HANDLER_STATE_PATH, {}) or {}
    space = int(state_now.get("last_space", space_cap))
    space = max(1, space - 1) if mem_strained else min(space_cap, space + 1)  # a body at a time, like the quota's steps
    limits = _read_json(HOST_LIMITS_PATH, {}) or {}
    if isinstance(limits.get("max_quota_pct"), (int, float)):
        global MAX_QUOTA_PCT
        MAX_QUOTA_PCT = max(MIN_QUOTA_PCT, min(MAX_QUOTA_PCT, int(limits["max_quota_pct"])))
    idle = _idle_cores(current)

    if current > MAX_QUOTA_PCT:
        target = MAX_QUOTA_PCT
        reason = f"above this host's ceiling ({MAX_QUOTA_PCT}%) -- down to it"
    elif strained:
        # Unconditional -- overrides hunger/fitness entirely. The
        # "own den" signal, never negotiable regardless of how well
        # the organism is doing.
        target = max(MIN_QUOTA_PCT, current - STEP_PCT)
        reason = "real CPU strain (load) -- cutting back regardless of fitness"
    elif idle >= 1.0 and current < MAX_QUOTA_PCT:
        target = min(MAX_QUOTA_PCT, current + IDLE_STEP_PCT)
        reason = f"{idle:.1f} idle cores -- granting one more"
    elif idle < 0.0 and current > MIN_QUOTA_PCT:
        target = max(MIN_QUOTA_PCT, current - IDLE_STEP_PCT)
        reason = f"other work needs the cores ({-idle:.1f} short) -- giving one back"
    elif fitness_improved and request_count > 0:
        step = int(round(STEP_PCT * responsiveness))
        target = min(MAX_QUOTA_PCT, current + max(step, 1 if responsiveness > 0.15 else 0))
        reason = f"real fitness improvement + real request -- granting (responsiveness={responsiveness:.2f})"
    elif request_count > 0 and responsiveness < 0.2:
        reason = f"requesting but habituated (responsiveness={responsiveness:.2f}) -- not granting"
    elif request_count == 0 and not fitness_improved:
        reason = "quiet -- no request, no change"

    print(f"[resource_handler] current={current}% target={target}% satiation={satiation:.3f} "
          f"responsiveness={responsiveness:.3f} strained={strained} "
          f"fitness_improved={fitness_improved} requests={request_count} idle_cores={idle:.2f} ceiling={MAX_QUOTA_PCT}% "
          f"memory_short={mem_strained} space={space}/{space_cap} workers")
    print(f"[resource_handler] {reason}")

    memory = _memory_limits() if SYSTEMD else None
    if memory:
        print(f"[resource_handler] memory: high {memory[0]} MB, max {memory[1]} MB (from what the host has)")
    if not dry_run:
        if target != current:
            _set_quota_pct(target)
        if memory:
            _set_memory_limits(*memory)
        _write_json_atomic(HANDLER_STATE_PATH, {
            "satiation": satiation,
            "last_run_unix": time.time(),
            "last_quota_pct": target,
            "last_space": space,
            "last_reason": reason + (" (memory short: space cut to %d workers)" % space if mem_strained else ""),
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute and print, but don't apply or persist")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
