"""The host's own settings (cambrian.json), for the installers:

    python tools/settings.py <cambrian.json> <hive: true|false|keep> [key=value ...]

Keeps what is there, sets what is given; "hive" is false (solo) unless an
install joined the hive (--hive): joining is the owner's choice (a 2026-09-30
panel -- fishbowl/sandbox.py hive()). Prints where the host stands.
"""
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    path, hive = Path(argv[0]), (argv[1] if len(argv) > 1 else "")
    try:
        s = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(s, dict):
            s = {}
    except (OSError, ValueError):
        s = {}
    if hive in ("true", "false"):
        s["hive"] = hive == "true"
    s.setdefault("hive", False)
    for kv in argv[2:]:
        k, _, v = kv.partition("=")
        s[k] = int(v) if k == "viewer_port" else v  # a source stays a string, even "0"
    path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    print("  hive: " + ("on -- it may take migrants from its peers and serves its own to them"
                        if s["hive"] else "off (solo) -- --hive / -Hive joins it"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
