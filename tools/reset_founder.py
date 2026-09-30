"""
Reset a lineage to a fresh install's founder: the being's state (its genome,
memories, episodes, habits, the individuals it knows, its histories) moves
aside into state/backup-<time>-before-reset/, and the next start makes a
default brain (run_vision._default_brain: a random genome, every seed set,
random draws). Nothing is deleted; moving the folder's files back restores it.

    python tools/reset_founder.py <state dir>      # with its organism stopped

Kept in place, because they belong to the host, not the being: the stream the
user chose, the host's limits, the resource handler's state, the service's
logs, and older backups. (The settings panel's Reset button does this.)
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

KEEP = {".gitkeep", "selected_source.json", "host_limits.json", "handler_state.json", "yielded.json",
        "service.log", "run.log", "viewer.log", "launchd.log", "service.stop",
        "life_history.json", "life_history.prev.json",  # the host's record of its lives, across lineages (and its previous copy)
        "experiments.json"}   # the host's trials


def reset(state: Path, suffix: str = "before-reset") -> tuple[Path, list]:
    """Moves the being's state aside; returns (the backup folder, what moved)."""
    backup = state / f"backup-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
    backup.mkdir()
    moved = []
    for f in sorted(state.iterdir()):
        if f == backup or f.name in KEEP or f.name.startswith("backup-") or ".bak" in f.name:
            continue
        shutil.move(str(f), str(backup / f.name))
        moved.append(f.name)
    return backup, moved


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    state = Path(sys.argv[1])
    if not (state / "checkpoint.json").exists():
        print("no checkpoint here: nothing to reset")
        return 1
    backup, moved = reset(state)
    print(f"reset to a founder: {len(moved)} files moved to {backup.name} ({', '.join(moved)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
