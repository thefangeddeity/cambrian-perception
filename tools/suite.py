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


# ---- autostart (docs/suite.md) ----------------------------------------------
# Whether the organism comes back on its own at boot. On while it is being
# developed; the finished default is off: the livecam is the always-on default
# and the organism starts only on `cambrian --start` (or comes back at boot
# only if that was this same boot -- a crash restart, not a reboot).
AUTOSTART_DEFAULT = True  # development; False once testing is done
SETTINGS = ROOT / "cambrian.json"
STARTED = ROOT / "state" / "started.json"   # when `cambrian --start` last ran: this boot's, or an older one


def autostart() -> bool:
    try:
        return bool(json.loads(SETTINGS.read_text(encoding="utf-8-sig")).get("autostart", AUTOSTART_DEFAULT))
    except (OSError, ValueError):
        return AUTOSTART_DEFAULT


def set_autostart(on: bool) -> None:
    try:
        cfg = json.loads(SETTINGS.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        cfg = {}
    cfg["autostart"] = bool(on)
    tmp = SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS)


def boot_time() -> float:
    """When this machine booted (seconds since the epoch)."""
    if WINDOWS:
        import ctypes
        return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000.0
    if MACOS:
        out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True).stdout
        try:
            return float(out.split("sec =")[1].split(",")[0])
        except (IndexError, ValueError):
            return 0.0
    try:
        return time.time() - float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def mark_started() -> None:
    """`cambrian --start` ran now: this boot's starts (a crash restart) go ahead."""
    STARTED.parent.mkdir(parents=True, exist_ok=True)
    STARTED.write_text(json.dumps({"boot": boot_time(), "t": time.time()}), encoding="utf-8")


def started_this_boot() -> bool:
    try:
        return abs(json.loads(STARTED.read_text(encoding="utf-8"))["boot"] - boot_time()) < 300  # boot times read a little apart
    except (OSError, ValueError, KeyError, TypeError):
        return False


def may_start() -> bool:
    """May the organism start now: autostart on, or `cambrian --start` this boot."""
    return autostart() or started_this_boot()


def start_livecam() -> None:
    """The camera back to the livecam, through its own public command (when
    the organism declines to start at boot and the livecam is installed)."""
    cmd = livecam_command()
    if cmd is None and not WINDOWS and not MACOS:
        import shutil
        cmd = [shutil.which("camdash")] if shutil.which("camdash") else None
    if not cmd:
        return
    try:
        subprocess.run(cmd + (["start"] if MACOS else ["--start"]), capture_output=True, timeout=120,
                       creationflags=NO_WINDOW if WINDOWS else 0)
    except (OSError, subprocess.SubprocessError):
        pass


if __name__ == "__main__":
    # --may-start: the Linux unit's ExecCondition (0 = start; 1 = skip, and the camera goes to the livecam)
    # --mark-started / --autostart on|off: for tools/cambrian
    verb = sys.argv[1] if len(sys.argv) > 1 else ""
    if verb == "--may-start":
        if may_start():
            sys.exit(0)
        print("not starting at boot: autostart is off (`cambrian --start` starts it; `cambrian --autostart on` brings it back at boot)")
        start_livecam()
        sys.exit(1)
    if verb == "--mark-started":
        mark_started()
        sys.exit(0)
    if verb == "--autostart":
        if len(sys.argv) > 2 and sys.argv[2] in ("on", "off"):
            set_autostart(sys.argv[2] == "on")
        print(f"autostart: {'on' if autostart() else 'off'}")
        sys.exit(0)
    sys.exit(2)
