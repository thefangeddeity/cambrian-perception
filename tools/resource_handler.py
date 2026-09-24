#!/usr/bin/env python3
from __future__ import annotations

"""
The external "handler" (the user's own word -- "its tube") that decides how
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
    signal. the user's own framing: "a relatively permissive feeder circuit
    that nevertheless ruthlessly cuts off food when it's bad for it,
    regardless of how hungry it is."

  HUNGER (reward for real, demonstrated cognition) -- quota moves UP,
    within the ceiling, when evolution_log.json shows real recent
    fitness improvement. Never based on the organism merely asking.

  DISGUST / WASTE (the "own den" signal, unconditional) -- quota moves
    DOWN, regardless of hunger, when REAL system telemetry (load
    average, free memory) shows the shared machine is under genuine
    strain. Never based on anything the organism reports about
    itself -- unfakeable from inside the sandbox.

  HUNGER EXTINCTION (this pass's addition) -- User: "I don't want an
    eternally hungry, unhappy organism... if its resource hunger gets
    out of line and it's constantly biting at its handler, [the
    handler] should have some sort of long-term antagonist feedback
    such as making it less hungry the more it uselessly attempts to
    eat resources... only gets hungrier when it detects new food."
    Exactly conspec.Habituation's own shape, reapplied here: repeated
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
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.json"
REQUESTS_PATH = STATE_DIR / "requests.json"
HANDLER_STATE_PATH = STATE_DIR / "handler_state.json"

SERVICE_NAME = "cambrian-perception.service"

MIN_QUOTA_PCT = 50    # hard floor -- never starve it completely
MAX_QUOTA_PCT = 300   # hard ceiling -- never exceed 3 of 8 real cores,
# regardless of any signal below. Change this only as a deliberate,
# written decision (see README's fishbowl boundary on what "explicit,
# not slid into" means), not a tuning knob to nudge casually.
STEP_PCT = 25         # bounded step per invocation, up or down --
# no single run can swing the quota far, so a bad read of the signals
# costs at most one small step, not a lurch.

# Real, external, unfakeable-from-inside-the-sandbox strain thresholds.
LOAD_STRAIN_PER_CORE = 1.3   # loadavg / nproc above this = real strain
FREE_MEM_STRAIN_MB = 800     # system-wide free memory below this = real strain

HUNGER_DECAY = 0.85  # EMA decay for the extinction/satiation signal --
# tuned faster than conspec.Habituation's 0.98 since this runs far
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
    # sudo, not a root-owned service -- this script runs as the same
    # unprivileged user as everything else here, using the same
    # already-verified passwordless sudo rule the initial deployment
    # used, scoped to exactly this one real command.
    subprocess.run(
        ["sudo", "systemctl", "set-property", SERVICE_NAME, f"CPUQuota={pct}%"],
        check=True,
    )


def _recent_fitness_improved(window: int = 200) -> bool:
    log = _read_json(EVOLUTION_LOG_PATH, [])
    if len(log) < 2:
        return False
    recent = log[-window:]
    fitnesses = [e["best_fitness"] for e in recent if e.get("best_fitness") is not None]
    if len(fitnesses) < 2:
        return False
    return fitnesses[-1] > fitnesses[0] + 1e-6


def _recent_request_count(since_seconds: float = 1800.0) -> int:
    requests = _read_json(REQUESTS_PATH, [])
    if not requests:
        return 0
    # requests.json entries carry their own elapsed_seconds from THAT
    # run's own start, not a wall-clock timestamp -- real, honest
    # limitation: this counts requests in the tail of the log instead
    # of a true time window, since sandbox.py doesn't stamp wall-clock
    # time (see its own log_request()). Close enough for a coarse,
    # periodic handler decision, not pretended to be more precise.
    return len(requests[-20:])


def _load_average_strain() -> bool:
    import os
    load1, _, _ = os.getloadavg()
    nproc = os.cpu_count() or 1
    return (load1 / nproc) > LOAD_STRAIN_PER_CORE


def _free_memory_strain() -> bool:
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
    strained = _load_average_strain() or _free_memory_strain()

    # Hunger extinction -- the user's own addition this pass. Real new food
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

    if strained:
        # Unconditional -- overrides hunger/fitness entirely. The
        # "own den" signal, never negotiable regardless of how well
        # the organism is doing.
        target = max(MIN_QUOTA_PCT, current - STEP_PCT)
        reason = "real system strain (load or memory) -- cutting back regardless of fitness"
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
          f"fitness_improved={fitness_improved} requests={request_count}")
    print(f"[resource_handler] {reason}")

    if not dry_run:
        if target != current:
            _set_quota_pct(target)
        _write_json_atomic(HANDLER_STATE_PATH, {
            "satiation": satiation,
            "last_run_unix": time.time(),
            "last_quota_pct": target,
            "last_reason": reason,
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute and print, but don't apply or persist")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
