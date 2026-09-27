#!/usr/bin/env python3
"""
`cambrian` where systemd isn't (Windows now, macOS next; Linux has
tools/cambrian over systemd). Same verbs everywhere:

  cambrian --start     start the organism, its viewer and its budget
  cambrian --stop      stop them (the organism saves its checkpoint first)
  cambrian --restart   stop, then start
  cambrian --status    what runs, and how the organism is doing
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
TASK = "cambrian-perception"
WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if WINDOWS else 0


def _ps(cmd: str) -> str:
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
                         capture_output=True, text=True, creationflags=NO_WINDOW)
    return out.stdout.strip()


def _service_pid() -> int | None:
    """The running supervisor (tools/cambrian_service.py), if any."""
    if WINDOWS:
        out = _ps("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*cambrian_service.py*' "
                  "-and $_.Name -like 'python*' } | Select-Object -First 1 -ExpandProperty ProcessId")
        return int(out) if out.isdigit() else None
    out = subprocess.run(["pgrep", "-f", "[c]ambrian_service.py"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def start() -> int:
    if _service_pid():
        print("already running")
        return status()
    if WINDOWS:
        _ps(f"Start-ScheduledTask -TaskName '{TASK}'")
    else:  # macOS: the boot LaunchDaemon (deploy/macos/install.sh); a clean stop keeps it down
        subprocess.run(["sudo", "launchctl", "kickstart", "system/org.cambrian.perception"])
    for _ in range(30):
        if _service_pid():
            break
        time.sleep(1)
    return status()


def stop() -> int:
    pid = _service_pid()
    if not pid:
        print("not running")
        return 0
    (STATE / "service.stop").touch()
    print("stopping (the organism finishes its generation and saves) ...", flush=True)
    for _ in range(200):
        if not _service_pid():
            print("stopped (checkpoint saved)")
            return 0
        time.sleep(1)
    print("still running after 200 s -- check state/service.log")
    return 1


def status() -> int:
    pid = _service_pid()
    print(f"service   : {'running (pid %d)' % pid if pid else 'NOT RUNNING'}")
    if WINDOWS:
        t = _ps(f"$t = Get-ScheduledTask -TaskName '{TASK}' -ErrorAction SilentlyContinue; "
                "if ($t) { \"$($t.State), at boot as $($t.Principal.UserId)\" } else { 'none' }")
        print(f"boot task : {t}")
    elif sys.platform == "darwin":
        loaded = subprocess.run(["launchctl", "print", "system/org.cambrian.perception"], capture_output=True).returncode == 0
        print(f"boot job  : {'LaunchDaemon org.cambrian.perception' if loaded else 'none loaded'}")
    try:
        cfg = json.loads((ROOT / "cambrian.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        cfg = {}
    live = STATE / "live_status.json"
    for runtime in (os.environ.get("CAMBRIAN_RUNTIME_DIR"), "/dev/shm/cambrian-perception"):
        if runtime and (Path(runtime) / "live_status.json").exists():
            live = Path(runtime) / "live_status.json"
    try:
        d = json.loads(live.read_text(encoding="utf-8-sig"))
        b = d.get("body") or {}
        age = time.time() - live.stat().st_mtime
        print(f"organism  : generation {d.get('generation', 0):,}, eye {d.get('receptors', '?')}x{d.get('receptors', '?')}, "
              f"energy {b.get('energy', 0):.2f}, {'asleep' if (b.get('asleep') or 0) >= 0.5 else 'awake'} "
              f"(status {age:.0f} s old)")
        print(f"watching  : {d.get('clip_name', cfg.get('source', '?'))}")
    except (OSError, ValueError):
        print("organism  : no live status yet")
    print(f"quota     : {json.loads((STATE / 'handler_state.json').read_text(encoding="utf-8-sig")).get('last_quota_pct', '?') if (STATE / 'handler_state.json').exists() else '?'}% of a core (hard cap)")
    print(f"viewer    : http://{socket.gethostname()}:{cfg.get('viewer_port', 8090)}/")
    return 0 if pid else 1


def main() -> int:
    verb = (sys.argv[1] if len(sys.argv) > 1 else "--status").lstrip("-")
    if verb == "restart":
        stop()
        return start()
    return {"start": start, "stop": stop, "status": status}.get(verb, lambda: print(__doc__) or 2)()


if __name__ == "__main__":
    sys.exit(main())
