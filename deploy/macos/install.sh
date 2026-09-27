#!/bin/sh
# Installs (or updates) cambrian-perception on macOS as a background service
# (docs/packaging.md). Run from the repo, as the user who will own it:
#
#   sh deploy/macos/install.sh [--checkpoint <path>] [--source auto|0|rtsp://...] [--model <yolov8n.onnx>]
#
# A LaunchDaemon starts it at boot (no login needed) as this user; a clean
# stop keeps it down, a crash restarts it. Beside the Mac livecam it never
# clashes: it reads the livecam's output (rtsp://127.0.0.1:8554/cam) instead
# of taking the camera, uses port 8090 only, its own folder and Python venv.
# Re-running it updates the code and keeps state/ (checkpoint, logs).
set -eu
SOURCE=auto CHECKPOINT="" MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --source) SOURCE=$2; shift 2 ;;
        --checkpoint) CHECKPOINT=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DIR="$HOME/Library/Application Support/cambrian-perception"
LABEL=org.cambrian.perception
PLIST=/Library/LaunchDaemons/$LABEL.plist
echo "cambrian-perception: installing from $REPO to $DIR"
sudo -v

# 1. Stop a running copy (it saves first), then lay down the code.
if pgrep -f "[c]ambrian_service.py" >/dev/null; then
    echo "  stopping the running organism (it saves first)..."
    touch "$DIR/state/service.stop"
    i=0; while pgrep -f "[c]ambrian_service.py" >/dev/null && [ $i -lt 200 ]; do sleep 1; i=$((i+1)); done
fi
mkdir -p "$DIR/state" "$DIR/models"
if [ "$REPO" != "$DIR" ]; then
    rsync -a --delete --exclude .git --exclude .venv --exclude state --exclude models \
        --exclude cambrian.json --exclude __pycache__ "$REPO/" "$DIR/"
fi

# 2. Python 3.12+ (requirements.lock's numpy needs it): Homebrew's if present
#    (a non-login shell may not have it on PATH), else python.org's build.
#    Then the organism's own venv.
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
PY=""
for p in python3.14 python3.13 python3.12 /Library/Frameworks/Python.framework/Versions/3.1[2-9]/bin/python3 python3; do
    c=$(command -v "$p" 2>/dev/null || true)
    [ -n "$c" ] && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null && { PY=$c; break; }
done
if [ -z "$PY" ] && command -v brew >/dev/null; then
    echo "  no Python 3.12+ -- installing Homebrew's python@3.14"
    brew install python@3.14 && PY=$(brew --prefix)/bin/python3.14
fi
if [ -z "$PY" ]; then
    V=3.14.0
    echo "  no Python 3.12+ -- installing python.org Python $V"
    curl -fsSL -o /tmp/python-$V.pkg "https://www.python.org/ftp/python/$V/python-$V-macos11.pkg"
    sudo installer -pkg /tmp/python-$V.pkg -target / >/dev/null
    PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
fi
echo "  python: $PY ($($PY --version))"
[ -x "$DIR/.venv/bin/python" ] || "$PY" -m venv "$DIR/.venv"
"$DIR/.venv/bin/python" -m pip install -q --disable-pip-version-check -r "$DIR/requirements.lock"

# 3. The prey detector's model.
if [ -n "$MODEL" ]; then cp "$MODEL" "$DIR/models/yolov8n.onnx"; fi
[ -f "$DIR/models/yolov8n.onnx" ] || echo "  WARNING: no model at $DIR/models/yolov8n.onnx -- no prey (snacks only) until one is added (--model)"

# 4. What it watches: the Mac livecam's output when that's here, else the camera.
if [ "$SOURCE" = auto ]; then
    if nc -z -G 2 127.0.0.1 8554 2>/dev/null; then SOURCE="rtsp://127.0.0.1:8554/cam"; else SOURCE=0; fi
fi
echo "  source: $SOURCE"
printf '{"source": "%s", "viewer_port": 8090}\n' "$SOURCE" > "$DIR/cambrian.json"

# 5. A lineage to continue, if given and none is here yet.
if [ -n "$CHECKPOINT" ] && [ ! -f "$DIR/state/checkpoint.json" ]; then
    cp "$CHECKPOINT" "$DIR/state/checkpoint.json"; echo "  seeded the organism from $CHECKPOINT"
fi

# 6. The boot job: the supervisor, as this user, at boot; restarted after a
#    crash (KeepAlive unless it exits cleanly -- `cambrian --stop` stays stopped).
cat > /tmp/$LABEL.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>UserName</key><string>$(id -un)</string>
    <key>ProgramArguments</key>
    <array>
        <string>$DIR/.venv/bin/python</string>
        <string>tools/cambrian_service.py</string>
    </array>
    <key>WorkingDirectory</key><string>$DIR</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
    <key>StandardOutPath</key><string>$DIR/state/launchd.log</string>
    <key>StandardErrorPath</key><string>$DIR/state/launchd.log</string>
</dict>
</plist>
EOF
sudo launchctl bootout system/$LABEL 2>/dev/null || true
sudo install -m 644 -o root -g wheel /tmp/$LABEL.plist "$PLIST"
sudo launchctl bootstrap system "$PLIST"

# 7. `cambrian` on the PATH.
sudo mkdir -p /usr/local/bin
sudo ln -sf "$DIR/deploy/macos/cambrian" /usr/local/bin/cambrian
chmod +x "$DIR/deploy/macos/cambrian"
sleep 15
"$DIR/.venv/bin/python" "$DIR/tools/cambrian_ctl.py" --status || true
echo "installed. Control it with: cambrian --start | --stop | --restart | --status"
