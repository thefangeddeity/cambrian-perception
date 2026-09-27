# Packaging: one organism, four platforms (design, 2026-09-27)

cambrian-perception ships separately from laptop-livecam on every platform.
Either can be installed alone. Installed together on one machine, they must
not clash. The organism's code and library versions are the same everywhere
(the host parity rule); only the *handling* differs per platform: the
supervisor, the CPU budget, where frames live, and the camera.

## What every package provides

- **The organism**, running in the background from boot, with no window and
  no login needed, at low priority, restarted after a crash. A deliberate stop
  saves the checkpoint first.
- **Its viewer** on port **8090**.
- **A resource budget**: a hard CPU cap that the resource handler moves as
  cores go idle or busy.
- **One command**: `cambrian --start | --stop | --restart | --status`
  (`--status` needs no privileges).
- **The pinned libraries** (`requirements.lock`) in the organism's own venv,
  never the system's. The YOLO model is at `models/yolov8n.onnx` next to the
  code, unless `CAMBRIAN_PREY_MODEL` says otherwise.
- **State that survives upgrades and uninstall** (`state/`: checkpoint,
  logs). Removing it takes an explicit purge.

## Living beside laptop-livecam (no clash, by construction)

| Resource | laptop-livecam | cambrian-perception |
|---|---|---|
| Ports | 80, 8554, 8888, 8889, 8890, 1935, 8189 | 8090 |
| **Camera** | owns it | when laptop-livecam is installed and running, reads its output, `rtsp://127.0.0.1:8554/cam` (as on tina); otherwise opens the camera itself. Decided at install and re-checked at each start. |
| Service | hls-livecam's own (`broadcast-api`, `hls-livecam-win`) | `cambrian-perception` (target, task or agent) |
| Python | the system's (Linux/macOS), or its CV sidecar's | its own venv |
| Install and data | the livecam's paths | its own (below) |

## Per platform

| | Linux: Debian/Ubuntu `.deb`, Arch AUR | macOS | Windows 11 |
|---|---|---|---|
| Code | `/srv/cambrian/cambrian-perception` | `~/Library/Application Support/cambrian-perception` | `C:\ProgramData\cambrian\cambrian-perception` |
| Runs as | system user `cambrian` | the installing user | the installing user (S4U: no stored password) |
| Supervisor | systemd `cambrian.target` (organism, viewer, handler timer) | a boot LaunchDaemon (`org.cambrian.perception`, runs as the user) running `tools/cambrian_service.py` | a Task Scheduler boot task running `tools/cambrian_service.py` |
| CPU budget | cgroup `CPUQuota` | `taskpolicy` / nice (no hard cap in macOS) | Job Object CPU-rate hard cap |
| Frames (RAM only) | `/dev/shm` | held in the organism's memory, served to the viewer on 127.0.0.1 (`video_source.FrameRing`) | the same as macOS: no RAM disk in Windows |
| Idle measure | `/proc/stat` | `host_processor_info` (via `sysctl`/`top`) | `GetSystemTimes` |
| Install | `.deb` / PKGBUILD running `deploy/install.sh` | `deploy/macos/install.sh` (Homebrew's Python 3.12+ and ffmpeg, else python.org's Python); a `.pkg` later | `deploy/windows/install.ps1` (an MSI later, like hls-livecam-win) |
| Camera device | `/dev/video0` | index 0 (AVFoundation) | index 0 (Media Foundation) |
| Streams, web video | OpenCV's own FFmpeg | an `ffmpeg` process (the macOS OpenCV wheels have no FFmpeg; H.264 decodes bit-exact either way) | OpenCV's own FFmpeg |

**Why a Python supervisor on Windows and macOS.** systemd restarts the
organism whenever it exits: every hour by design, and on every video switch.
Task Scheduler and launchd only restart on failure. So a small supervisor
(`tools/cambrian_service.py`) does it:
- it runs the organism in a loop and keeps the viewer up;
- it calls the resource handler every 15 minutes and applies its quota;
- everything dies with it (a Job Object on Windows, a process group on
  macOS).

**Graceful stop everywhere.** A stop-request file (`state/stop.request`) is
checked every generation; the organism finishes that generation, saves, and
exits. It is the portable form of systemd's SIGTERM, which also still works.

**Lineages.** Each host evolves its own lineage (Tina's lean one; the default
i5's full one). A new install can seed its first checkpoint from another
host's (`-Checkpoint` / `--checkpoint`).

## Order of work

1. Portable code (above).
2. Windows installer, tested on hera.
3. `.deb`, tested on tina.
4. AUR, which needs an Arch box.
5. macOS, on ariana (done: `deploy/macos/install.sh`).
