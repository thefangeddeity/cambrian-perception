#!/bin/sh
# Installs (or updates) cambrian-perception's services on a host whose checkout
# is /srv/cambrian/cambrian-perception, owned by the "cambrian" system user.
# Run as root from anywhere: sh /srv/cambrian/cambrian-perception/deploy/install.sh
#
# It installs the units, cambrian.target (one switch for everything), the
# polkit rule (the viewer may restart the organism on a video switch) and the
# `cambrian` command, and enables the target at boot. Host-specific settings
# stay in drop-ins (e.g. cambrian-perception.service.d/10-camera-source.conf:
# which camera, what priority) and are left alone. Running services keep
# running; nothing is stopped.
set -eu
REPO=/srv/cambrian/cambrian-perception
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
systemctl enable -q cambrian.target cambrian-perception.service cambrian-viewer.service cambrian-resource-handler.timer
systemctl start cambrian.target
echo "installed: cambrian --start | --stop | --restart | --status"
