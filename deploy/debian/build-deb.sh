#!/bin/sh
# Builds cambrian-perception_<version>_all.deb from this checkout, on any
# Debian/Ubuntu box with dpkg-deb:  sh deploy/debian/build-deb.sh [out dir]
# It lays out what deploy/install.sh makes on a git-checkout host -- don't
# install it over such a checkout without moving its state/ aside first.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# the release tag (v0.1.0 -> 0.1.0), commits since it, the commit: 0.1.0+r3.gabc1234
if VER=$(git -C "$REPO" describe --long --tags --abbrev=7 2>/dev/null); then
    VER=$(echo "$VER" | sed 's/^v//; s/-\([0-9]*\)-g/+r\1.g/')
else
    VER="0~r$(git -C "$REPO" rev-list --count HEAD).g$(git -C "$REPO" rev-parse --short HEAD)"
fi
OUT=${1:-$REPO/dist}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
D="$STAGE/srv/cambrian/cambrian-perception"
mkdir -p "$D" "$STAGE/DEBIAN" "$OUT"
git -C "$REPO" archive HEAD | tar -x -C "$D"
chmod 755 "$D/tools/cambrian"
install -Dm644 -t "$STAGE/usr/lib/systemd/system/" "$REPO/deploy/cambrian.target" "$REPO/deploy/cambrian-perception.service" \
    "$REPO/deploy/cambrian-viewer.service" "$REPO/deploy/cambrian-resource-handler.service" "$REPO/deploy/cambrian-resource-handler.timer"
install -Dm644 "$REPO/deploy/50-cambrian.rules" "$STAGE/usr/share/polkit-1/rules.d/50-cambrian.rules"
install -Dm644 "$REPO/deploy/packaging/sysusers.conf" "$STAGE/usr/lib/sysusers.d/cambrian-perception.conf"
install -Dm644 "$REPO/deploy/packaging/tmpfiles.conf" "$STAGE/usr/lib/tmpfiles.d/cambrian-perception.conf"
install -d "$STAGE/usr/share/doc/cambrian-perception"
{
    echo "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/"
    echo "Upstream-Name: cambrian-perception"
    echo "Source: https://github.com/thefangeddeity/cambrian-perception"
    echo ""
    echo "Files: *"
    echo "License: GPL-3.0-only"
    echo " On Debian systems, the full text is in /usr/share/common-licenses/GPL-3,"
    echo " and in /srv/cambrian/cambrian-perception/LICENSE."
} > "$STAGE/usr/share/doc/cambrian-perception/copyright"
mkdir -p "$STAGE/usr/bin"
ln -s /srv/cambrian/cambrian-perception/tools/cambrian "$STAGE/usr/bin/cambrian"
{
    echo "Package: cambrian-perception"
    echo "Version: $VER"
    echo "Architecture: all"
    echo "Maintainer: thefangeddeity <thefangeddeity@users.noreply.github.com>"
    echo "Depends: systemd, passwd"
    echo "Recommends: python3-venv, polkitd | policykit-1"
    echo "Section: science"
    echo "Priority: optional"
    echo "Homepage: https://github.com/thefangeddeity/cambrian-perception"
    echo "Description: evolving artificial visual organism living in a camera feed"
    echo " A sandboxed, self-programming organism that evolves its own vision,"
    echo " body and behaviour on a live camera or stream, with a small web viewer."
} > "$STAGE/DEBIAN/control"
{
    echo '#!/bin/sh'
    echo 'set -e'
    echo 'if [ "$1" = configure ]; then sh /srv/cambrian/cambrian-perception/deploy/packaging/post-install.sh; fi'
} > "$STAGE/DEBIAN/postinst"
{
    echo '#!/bin/sh'
    echo 'if [ "$1" = remove ]; then sh /srv/cambrian/cambrian-perception/deploy/packaging/pre-remove.sh; fi'
    echo 'exit 0'
} > "$STAGE/DEBIAN/prerm"
chmod 755 "$STAGE/DEBIAN/postinst" "$STAGE/DEBIAN/prerm"
dpkg-deb --root-owner-group --build "$STAGE" "$OUT/cambrian-perception_${VER}_all.deb" >/dev/null
echo "$OUT/cambrian-perception_${VER}_all.deb"
