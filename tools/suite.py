"""
The camera suite (docs/suite.md): the organism and the livecam are two
products that share one camera and never run together. Starting either
stops the other; the one that was stopped stays off -- across reboots --
until it is started again. Each side only ever calls the other's own
public command, never its internals:

  the livecam's         Windows: camdash --yield organism | --start
                        macOS:   livecam yield organism   | start
                        Linux:   systemd (hls-livecam.target Conflicts=
                                 cambrian.target; see tools/cambrian)

This module is the organism's half on Windows and macOS: where the livecam
is, whether it runs, and how to make it yield. Stdlib only.
"""

import json
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YIELDED = ROOT / "state" / "yielded.json"   # present = off because the livecam took the camera
WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
NO_WINDOW = 0x08000000 if WINDOWS else 0
LIVECAM_TASK = "hls-livecam-win"
LIVECAM_AGENT = Path.home() / "Library/LaunchAgents/com.livecam.autostart.plist"


def _ps(cmd: str) -> str:
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
                         capture_output=True, text=True, creationflags=NO_WINDOW)
    return out.stdout.strip()


def livecam_command() -> list[str] | None:
    """The livecam's own control command on this host, or None if it isn't installed."""
    if WINDOWS:
        exe = _ps(f"(Get-ScheduledTask -TaskName '{LIVECAM_TASK}' -ErrorAction SilentlyContinue).Actions.Execute")
        return [exe.strip('"')] if exe else None
    if MACOS:
        try:
            args = plistlib.loads(LIVECAM_AGENT.read_bytes()).get("ProgramArguments") or []
        except (OSError, ValueError):
            return None
        return [args[0]] if args else None
    return None


def livecam_running() -> bool:
    if WINDOWS:
        return _ps(f"(Get-ScheduledTask -TaskName '{LIVECAM_TASK}' -ErrorAction SilentlyContinue).State") == "Running"
    if MACOS:
        return subprocess.run(["pgrep", "-f", "bin/broadcast-api"], capture_output=True).returncode == 0
    return False


def yield_livecam() -> bool:
    """Make the livecam stop and stay off (it records that the organism has the
    camera). False if it could not be asked."""
    cmd = livecam_command()
    if not cmd:
        return True
    verb = ["--yield", "organism"] if WINDOWS else ["yield", "organism"]
    if livecam_running():
        print("the livecam is yielding the camera to the organism ...", flush=True)
    try:
        subprocess.run(cmd + verb, creationflags=NO_WINDOW, timeout=120)
        return True
    except (OSError, subprocess.TimeoutExpired) as e:
        # Windows: camdash runs elevated (its manifest), so only an elevated
        # shell may start it (WinError 740 otherwise).
        print(f"could not ask the livecam to yield ({e}) -- run this from an elevated terminal", flush=True)
        return False


def mark_yielded(to: str) -> None:
    YIELDED.parent.mkdir(parents=True, exist_ok=True)
    YIELDED.write_text(json.dumps({"to": to, "at": time.time()}), encoding="utf-8")


def yielded() -> dict | None:
    try:
        return json.loads(YIELDED.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def describe_yielded(y: dict) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(y.get("at", 0))))
    return f"off -- the {y.get('to', 'livecam')} took the camera at {when}; `cambrian --start` takes it back"
