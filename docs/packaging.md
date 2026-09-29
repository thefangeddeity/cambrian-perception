# Packaging: one organism, four platforms (design, 2026-09-27)

cambrian-perception ships separately from laptop-livecam on every platform.
Either can be installed alone. Installed together on one machine, they must
not clash. The organism's code and library versions are the same everywhere
(the host parity rule); only the *handling* differs per platform: the
supervisor, the CPU budget, where frames live, and the camera.

## What every package provides

- **The organism**, running in the background with no window, at low
  priority, restarted after a crash. It runs from boot on Linux and Windows,
  and from login on macOS, where only the logged-in session may use the
  camera. A deliberate stop saves the checkpoint first.
- **Its viewer** on port **8090**.
- **A resource budget**: a hard CPU cap that the resource handler moves as
  cores go idle or busy.
- **One command**: `cambrian --start | --stop | --restart | --yield | --status`
  (`--status` needs no privileges).
- **The pinned libraries** (`requirements.lock`) in the organism's own venv,
  never the system's. The YOLO model is at `models/yolov8n.onnx` next to the
  code, unless `CAMBRIAN_PREY_MODEL` says otherwise. Beside it, optional:
  `yolov8n-oiv7.onnx` and `yolov8n-oiv7.names.json` (Ultralytics' Open
  Images V7 YOLOv8n, exported to ONNX at 640 with its class names), which
  finds every plant (Open Images' "Plant" classes) as a sugar source. Without
  them only potted plants are.
- **State that survives upgrades and uninstall** (`state/`: checkpoint,
  logs). Removing it takes an explicit purge.

## Installing it

| Host | How | Update |
|---|---|---|
| Arch | `cd deploy/arch && makepkg -si` (or `yay -Bi .`) in a clone; the PKGBUILD builds from GitHub, no AUR needed | the same again |
| Debian / Ubuntu | `sh deploy/debian/build-deb.sh`, then `sudo apt install ./dist/cambrian-perception_*.deb` | build and install the new one |
| a git checkout (Linux) | `sudo sh /srv/cambrian/cambrian-perception/deploy/install.sh` | the same again (it pulls) |
| macOS | `sh deploy/macos/install.sh` from the repo | the same again |
| Windows | `deploy\windows\install.ps1`, elevated | the same again |

The packages and the Linux installer make the same host: `/srv/cambrian/cambrian-perception`,
the `cambrian` account (in the video group), `requirements.lock` in its own venv, the units
under `cambrian.target`, the polkit rule and the `cambrian` command -- the packages through
one shared post-install (`deploy/packaging/post-install.sh`). A packaged host's code is
root's (the organism can't change its own program); its state, model and venv are its own.
Removing a package stops it and keeps its state. Versions come from the release tag:
`0.1.0`, then `0.1.0.r3.gabc1234` (Arch) / `0.1.0+r3.gabc1234` (Debian) three commits later.
Don't install a package over a git-checkout host without moving its `state/` aside first.

## Living beside laptop-livecam (no clash, by construction)

The two are a camera suite and never run together: starting either stops the
other, and the stopped one stays off until it is started again (docs/suite.md).

| Resource | laptop-livecam | cambrian-perception |
|---|---|---|
| Ports | 80, 8554, 8888, 8889, 8890, 1935, 8189 | 8090 |
| **Camera** | has it while it runs | has it while it runs (the livecam is off then) |
| Service | hls-livecam's own (`broadcast-api`, `hls-livecam-win`) | `cambrian-perception` (target, task or agent) |
| Python | the system's (Linux/macOS), or its CV sidecar's | its own venv |
| Install and data | the livecam's paths | its own (below) |

## Per platform

| | Linux: Debian/Ubuntu `.deb`, Arch AUR | macOS | Windows 11 |
|---|---|---|---|
| Code | `/srv/cambrian/cambrian-perception` | `~/Library/Application Support/cambrian-perception` | `C:\ProgramData\cambrian\cambrian-perception` |
| Runs as | system user `cambrian` | the installing user | the installing user (S4U: no stored password) |
| Supervisor | systemd `cambrian.target` (organism, viewer, handler timer) | a login LaunchAgent (`org.cambrian.perception`, in the user's GUI session: the camera needs it) running `tools/cambrian_service.py` | a Task Scheduler boot task running `tools/cambrian_service.py` |
| CPU budget | cgroup `CPUQuota` | `taskpolicy` / nice (no hard cap in macOS) | Job Object CPU-rate hard cap |
| Frames (RAM only) | `/dev/shm` | held in the organism's memory, served to the viewer on 127.0.0.1 (`video_source.FrameRing`) | the same as macOS: no RAM disk in Windows |
| Idle measure | `/proc/stat` | `host_processor_info` (via `sysctl`/`top`) | `GetSystemTimes` |
| Install | `.deb` / PKGBUILD running `deploy/install.sh` | `deploy/macos/install.sh` (Homebrew's Python 3.12+ and ffmpeg, else python.org's Python); a `.pkg` later | `deploy/windows/install.ps1` (an MSI later, like hls-livecam-win) |
| Camera device | `/dev/v4l/by-id/...` (stable across reboots) or `/dev/video0` | index 0 (AVFoundation) | index 0 (Media Foundation) |
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

## Breeding across the fleet (`tools/fleet.py`)

Finds the organisms on the Tailnet (or `--hosts`), fetches each whole organism
(genome, body, memory, episodes) from its viewer (`/organism/...`, numbers
only), and judges them in a tournament: all on the same frames, captured into
RAM from the stream the fleet is watching, at the same prices. Each host's own
fitness is not comparable (its own CPU prices, its own frames).

- `python tools/fleet.py`: breed. The ranking is shown; you choose the parent
  and tick the hosts to clone it onto; nothing happens without "yes".
- `python tools/fleet.py --auto`: Sewall Wright's shifting balance. 3
  tournaments on 3 snapshots; if the best beats the worst in all of them, the
  best replaces the worst -- at most one per run. `--protect a,b` keeps hosts
  out of it (none by default). `--dry-run` judges and changes nothing.

Every replaced lineage is backed up to `state/backup-<time>-before-<parent>/`
first. Linux and macOS targets are cloned over ssh (the host's own name), the
local machine directly; a remote Windows target gets the steps printed. Runs
are logged in `state/fleet_log.jsonl`.
