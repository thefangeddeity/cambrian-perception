#!/usr/bin/env python3
"""
The organism's supervisor where systemd isn't (Windows now, macOS next; see
docs/packaging.md). The boot task / login agent runs this; it:

  - keeps the viewer up (tools/viewer.py, port from cambrian.json);
  - runs the organism (run_vision.py) in a loop at low priority -- it exits
    every hour by design, and on a video switch; systemd restarted it on
    Linux, Task Scheduler and launchd only restart on failure;
  - runs the resource handler every 15 minutes and applies its quota as a
    hard CPU cap (Windows: a Job Object CPU-rate cap, the counterpart of
    Linux's cgroup CPUQuota);
  - ties every child to itself: if this process ends -- a stopped task,
    a crash -- they end too (Windows: a kill-on-close Job Object);
  - stops gracefully on request: `cambrian --stop` creates state/service.stop;
    the organism is asked to finish its generation and save (the
    state/stop.request it checks every generation), then everything exits.

Settings: cambrian.json next to this repo's run_vision.py, written by the
installer: {"source": "0" | "rtsp://127.0.0.1:8554/cam" | ..., "viewer_port": 8090}.
Logs: state/service.log, state/run.log, state/viewer.log.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
SERVICE_STOP = STATE / "service.stop"
STOP_REQUEST = STATE / "stop.request"   # read by run_vision.py (fishbowl/sandbox.py)
HANDLER_STATE = STATE / "handler_state.json"
HANDLER_EVERY_S = 15 * 60
RESTART_AFTER_S = 10                    # as the systemd unit's RestartSec
STOP_GRACE_S = 180                      # a generation can take a while on a small host
LOG_MAX = 5 * 1024 * 1024
WINDOWS = os.name == "nt"


def log(msg: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    with open(STATE / "service.log", "a", encoding="utf-8") as f:
        f.write(line)


def log_file(name: str):
    path = STATE / name
    if path.exists() and path.stat().st_size > LOG_MAX:
        path.replace(path.with_suffix(".log.1"))
    return open(path, "a", encoding="utf-8", errors="replace")


def settings() -> dict:
    try:
        return json.loads((ROOT / "cambrian.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ Windows jobs
if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE

    class _BasicLimit(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class _ExtendedLimit(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _CpuRate(ctypes.Structure):
        _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

    _EXTENDED_LIMIT_INFORMATION, _CPU_RATE_CONTROL_INFORMATION = 9, 15
    _KILL_ON_JOB_CLOSE = 0x2000
    _CPU_RATE_ENABLE, _CPU_RATE_HARD_CAP = 0x1, 0x4
    _PROCESS_SET_QUOTA, _PROCESS_TERMINATE = 0x0100, 0x0001

    def _job(kill_on_close: bool = True):
        job = _k32.CreateJobObjectW(None, None)
        if kill_on_close:
            info = _ExtendedLimit()
            info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            _k32.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
        return job

    def _assign(job, pid: int) -> bool:
        h = _k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not h:
            return False
        ok = bool(_k32.AssignProcessToJobObject(job, h))
        _k32.CloseHandle(h)
        return ok

    def _cap(job, quota_pct: float) -> None:
        """Hard-cap the job at quota_pct of one core (150 = 1.5 cores)."""
        ncpu = os.cpu_count() or 1
        rate = int(max(1, min(10000, round(quota_pct / (100.0 * ncpu) * 10000))))
        info = _CpuRate(_CPU_RATE_ENABLE | _CPU_RATE_HARD_CAP, rate)
        _k32.SetInformationJobObject(job, _CPU_RATE_CONTROL_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))

    EVERYTHING = _job()                           # this process and all it starts
    _assign(EVERYTHING, os.getpid())
    BUDGET = _job()                               # the organism and its workers: the CPU cap
    BELOW_NORMAL = 0x00004000
    NO_WINDOW = 0x08000000
else:
    EVERYTHING = BUDGET = None


def quota_pct() -> float:
    try:
        return float(json.loads(HANDLER_STATE.read_text(encoding="utf-8-sig")).get("last_quota_pct", 150))
    except (OSError, ValueError):
        return 150.0


def start(args: list[str], logname: str, budget: bool = False) -> subprocess.Popen:
    kw = {"cwd": ROOT, "stdin": subprocess.DEVNULL, "stdout": log_file(logname), "stderr": subprocess.STDOUT}
    if WINDOWS:
        kw["creationflags"] = NO_WINDOW | (BELOW_NORMAL if budget else 0)
    elif budget:
        # macOS: children stay in this process group, so launchd ends them all
        # when it stops this job; the organism runs at low priority (macOS has
        # no hard CPU cap for a process -- its budget sets prices and workers).
        kw["preexec_fn"] = lambda: os.nice(10)
    p = subprocess.Popen(args, **kw)
    if WINDOWS and budget:
        _assign(BUDGET, p.pid)
    return p


def main() -> int:
    py = sys.executable.replace("pythonw.exe", "python.exe")
    cfg = settings()
    source = str(cfg.get("source", "0"))
    port = str(cfg.get("viewer_port", 8090))
    # The livecam has the camera (it made this yield; docs/suite.md): stay off
    # -- at boot too -- until `cambrian --start`. A clean exit, so neither
    # Task Scheduler nor launchd starts it again.
    import suite
    y = suite.yielded()
    if y:
        log(f"service: not starting, {suite.describe_yielded(y)}")
        return 0
    # Autostart off (docs/suite.md): at boot it stays off unless `cambrian
    # --start` ran this boot; the camera goes back to the livecam. A clean
    # exit, so neither Task Scheduler nor launchd starts it again.
    if not suite.may_start():
        log("service: not starting at boot -- autostart is off (`cambrian --start` starts it)")
        suite.start_livecam()
        return 0
    SERVICE_STOP.unlink(missing_ok=True)
    log(f"service: start (source {source}, viewer port {port}, python {py})")
    env_extra = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    os.environ.update(env_extra)
    if WINDOWS:
        _cap(BUDGET, quota_pct())
    viewer = start([py, "tools/viewer.py", "--port", port], "viewer.log")
    organism = None
    next_run = 0.0
    next_handler = time.time() + 60
    while True:
        now = time.time()
        if SERVICE_STOP.exists():
            log("service: stop requested")
            break
        if viewer.poll() is not None:
            log(f"service: viewer exited ({viewer.returncode}); restarting")
            viewer = start([py, "tools/viewer.py", "--port", port], "viewer.log")
        if organism is not None and organism.poll() is not None:
            log(f"service: organism exited ({organism.returncode}); restarting in {RESTART_AFTER_S} s")
            organism, next_run = None, now + RESTART_AFTER_S
        if organism is None and now >= next_run:
            source = str(settings().get("source", source))
            organism = start([py, "run_vision.py", source],  # one long process: no hourly bound
                             "run.log", budget=True)
            log(f"service: organism started (pid {organism.pid}, source {source}, cap {quota_pct():.0f}%)")
        if now >= next_handler:
            next_handler = now + HANDLER_EVERY_S
            subprocess.run([py, "tools/resource_handler.py"], cwd=ROOT, stdout=log_file("service.log"),
                           stderr=subprocess.STDOUT, creationflags=NO_WINDOW if WINDOWS else 0)
            if WINDOWS:
                _cap(BUDGET, quota_pct())
        time.sleep(2)
    # A graceful stop: the organism finishes its generation and saves.
    if organism is not None and organism.poll() is None:
        STOP_REQUEST.touch()
        try:
            organism.wait(timeout=STOP_GRACE_S)
            log("service: organism saved and exited")
        except subprocess.TimeoutExpired:
            log("service: organism did not stop in time; ending it")
            organism.kill()
    viewer.terminate()
    SERVICE_STOP.unlink(missing_ok=True)
    log("service: stopped")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):  # UTF-8 into its log, whatever the platform's code page (run_vision.utf8_stdio)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError, OSError):
            pass
    sys.exit(main())
