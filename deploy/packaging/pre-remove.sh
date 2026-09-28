#!/bin/sh
# Before a package removes cambrian-perception: stop it (it saves its
# checkpoint first) and take it off the boot. Its state stays in
# /srv/cambrian/cambrian-perception/state: the lineage is never deleted.
/srv/cambrian/cambrian-perception/tools/cambrian --stop || true
systemctl disable -q cambrian-perception.service cambrian-viewer.service cambrian-resource-handler.timer 2>/dev/null || true
