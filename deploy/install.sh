#!/bin/sh
# Installs or updates cambrian-perception on a host whose checkout
# is /srv/cambrian/cambrian-perception, owned by the "cambrian" system user.
# Run as root from anywhere: sh /srv/cambrian/cambrian-perception/deploy/install.sh
#
# It installs the units, cambrian.target (one switch for everything), the
# polkit rule (the viewer may restart the organism on a video switch) and the
# `cambrian` command, and enables the target at boot. Host-specific settings
# stay in drop-ins (e.g. cambrian-perception.service.d/10-camera-source.conf:
# which camera, what priority) and are left alone. Running services keep
# running. Installing is starting: the organism takes the camera, and the
# livecam, if installed, stops and stays off at boot (the camera suite,
# docs/suite.md; `livecam`'s own start takes it back).
#
#   --hive      join the hive: after an extinction it may take a migrant from
#               its peers, and it serves its own organism to them
#   --no-hive   solo (the default for a new install); without either, an
#               update keeps what the host had
set -eu
HIVE=""
for a in "$@"; do
    case "$a" in
        --hive) HIVE=true ;;
        --no-hive) HIVE=false ;;
        *) echo "unknown option $a (--hive | --no-hive)"; exit 2 ;;
    esac
done
REPO=/srv/cambrian/cambrian-perception
OWNER=$(stat -c %U "$REPO/.git")
# 1. The code: pull as the checkout's owner (an update is a re-run of this).
OLD=$(sudo -u "$OWNER" git -C "$REPO" rev-parse HEAD)
sudo -u "$OWNER" git -C "$REPO" pull -q --ff-only
# 2. Python 3.12+ and the organism's own venv with the pinned libraries --
#    the same step the macOS and Windows installers take, so every host runs
#    requirements.lock exactly.
PY=""
for p in python3.14 python3.13 python3.12 python3; do
    c=$(command -v "$p" 2>/dev/null || true)
    [ -n "$c" ] && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null && { PY=$c; break; }
done
[ -n "$PY" ] || { echo "no Python 3.12+ -- install one (Debian/Ubuntu: apt install python3; Arch: pacman -S python) and re-run"; exit 1; }
[ -x "$REPO/.venv/bin/python" ] || sudo -u "$OWNER" "$PY" -m venv "$REPO/.venv"
sudo -u "$OWNER" "$REPO/.venv/bin/python" -m pip install -q --disable-pip-version-check -r "$REPO/requirements.lock"
# 2b. Its self-test (tools/selftest.py) before it runs the new code: failing,
#     the checkout goes back to the code that is running, which keeps running.
#     (From the repo: its worker processes start in the working directory, and the
#     owner may not be able to enter the one this was run from.)
if ! (cd "$REPO" && sudo -u "$OWNER" "$REPO/.venv/bin/python" "$REPO/tools/selftest.py") > /tmp/cambrian-selftest.log 2>&1; then
    echo "self-test FAILED -- kept the running code ($OLD); see /tmp/cambrian-selftest.log"
    tail -20 /tmp/cambrian-selftest.log
    sudo -u "$OWNER" git -C "$REPO" reset -q --hard "$OLD"
    exit 1
fi
echo "self-test passed"
# 2c. The host's settings (cambrian.json): in the hive or solo.
sudo -u "$OWNER" "$REPO/.venv/bin/python" "$REPO/tools/settings.py" "$REPO/cambrian.json" "${HIVE:-keep}"
# 3. The services.
cd "$REPO/deploy"
install -m 644 cambrian.target cambrian-perception.service cambrian-viewer.service \
    cambrian-resource-handler.service cambrian-resource-handler.timer /etc/systemd/system/
install -d /etc/polkit-1/rules.d
install -m 644 50-cambrian.rules /etc/polkit-1/rules.d/
# The command, as a link into the repo (current with every pull). Others may
# pass through /srv/cambrian to reach it, not list it.
chmod 711 /srv/cambrian
chmod 755 "$REPO/tools/cambrian"
ln -sf "$REPO/tools/cambrian" /usr/local/bin/cambrian
systemctl daemon-reload
# Members are enabled under the target now, not directly at boot.
systemctl disable -q cambrian-perception.service cambrian-viewer.service cambrian-resource-handler.timer 2>/dev/null || true
systemctl enable -q cambrian-perception.service cambrian-viewer.service cambrian-resource-handler.timer
# 4. Start, or restart onto the new code (installing is starting).
if systemctl is-active -q cambrian-perception.service; then "$REPO/tools/cambrian" --restart; else "$REPO/tools/cambrian" --start; fi
echo "installed: cambrian --start | --stop | --restart | --yield | --status"
