#!/bin/sh
# After a package installs or upgrades cambrian-perception: the Arch
# package's .install and the Debian postinst both run this, so a packaged
# host ends up as deploy/install.sh leaves a git-checkout host -- the same
# account, pinned libraries, services and start.
set -eu
DIR=/srv/cambrian/cambrian-perception
# 1. Its account: a system user in the video group, no login.
if ! id cambrian >/dev/null 2>&1; then
    if command -v systemd-sysusers >/dev/null 2>&1; then systemd-sysusers cambrian-perception.conf
    else useradd --system --home-dir /srv/cambrian --shell /usr/sbin/nologin cambrian; fi
fi
if getent group video >/dev/null 2>&1; then usermod -aG video cambrian; fi
# 2. Where it may write (state, models, venv); the code stays root's.
if command -v systemd-tmpfiles >/dev/null 2>&1; then systemd-tmpfiles --create cambrian-perception.conf
else chmod 711 /srv/cambrian; install -d -o cambrian -g cambrian "$DIR/state" "$DIR/models" "$DIR/.venv"; fi
# 3. Python 3.12+ and its own venv with requirements.lock (needs the network
#    once), as its own account -- the step every installer takes.
PY=""
for p in python3.14 python3.13 python3.12 python3; do
    c=$(command -v "$p" 2>/dev/null || true)
    [ -n "$c" ] && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null && { PY=$c; break; }
done
[ -n "$PY" ] || { echo "cambrian-perception: needs Python 3.12+ (with its venv module)"; exit 1; }
[ -x "$DIR/.venv/bin/python" ] || runuser -u cambrian -- "$PY" -m venv "$DIR/.venv"
runuser -u cambrian -- "$DIR/.venv/bin/python" -m pip install -q --disable-pip-version-check -r "$DIR/requirements.lock"
# 4. Its self-test, as every installer runs it. A package has already
#    replaced the code, so a failure can't keep the old code: it leaves the
#    running organism (the old code, in memory) running and says so.
if ! (cd "$DIR" && runuser -u cambrian -- "$DIR/.venv/bin/python" "$DIR/tools/selftest.py") > /tmp/cambrian-selftest.log 2>&1; then
    tail -20 /tmp/cambrian-selftest.log
    echo "cambrian-perception: self-test FAILED -- not (re)started; see /tmp/cambrian-selftest.log"
    exit 0
fi
echo "cambrian-perception: self-test passed"
# 5. The detector models, fetched from the release if missing (checksummed;
#    an install never fails for want of them: it runs with snacks only),
#    before it starts: its detector loads them once, at start.
runuser -u cambrian -- "$DIR/.venv/bin/python" "$DIR/tools/fetch_models.py" "$DIR/models" || true
# 6. Its services, under cambrian.target. Installing is starting: the
#    livecam, if installed, yields (the camera suite, docs/suite.md).
systemctl daemon-reload
systemctl enable -q cambrian-perception.service cambrian-viewer.service cambrian-resource-handler.timer
if systemctl is-active -q cambrian-perception.service; then "$DIR/tools/cambrian" --restart; else "$DIR/tools/cambrian" --start; fi
echo "cambrian-perception: cambrian --start | --stop | --restart | --yield | --status; its camera and priority go in drop-ins (docs/packaging.md)"
