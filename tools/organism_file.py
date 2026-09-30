"""A whole organism in one file (.cambrioid), to keep and bring back.

    python tools/organism_file.py save <state dir> [file or folder] [--name <name>]
    python tools/organism_file.py check <file>
    python tools/organism_file.py load <file> <state dir>    # with its organism stopped

Saved files go to ~/Games/Lifeforms/Cambrioids/ unless another place is given
(/home/<you>/Games/Lifeforms/Cambrioids on Linux, /Users/<you>/Games/Lifeforms/
Cambrioids on macOS, C:\\Users\\<you>\\Games\\Lifeforms\\Cambrioids on Windows); a
bare file name given to check or load is looked for there too.

The file is a zip:
  manifest.json      format, name, the ecohost it came from, its generation,
                     when it was saved, the code's version, and every member's
                     SHA-256
  checkpoint.json    its genome, body, memories, learned weights
  episodes.npz       its episodes, sleep test set and map (if it has them)
  cortex.json        the individuals it knows (if any)
  tally.json         its life's tally
  founder.json, founder_seeded.txt, seeded.txt
                     its founder, so Reset to founder still works after a load
  lineage_history.json  its ecohost's life history when saved (a record, for
                     provenance; a load appends to the receiving ecohost's own)

Numbers only, never camera frames (the fishbowl boundary). It is saved from
the organism's latest checkpoint (written every 10 minutes and at every stop).

A file from anywhere is untrusted input (a 2026-09-30 panel: Schneier, as for
a hive migrant): reading one checks its size before unpacking anything, every
member's checksum, JSON and plain arrays only (never pickles), and the genome
against this ecohost's own limits (tools/fleet.py vet_genome). A file that
fails is refused with the reason. Loading moves the current lineage aside to
state/backup-<time>-before-load/ first (nothing is deleted) and records the
load in the life history (Gelman: a loaded lineage must never pass for a
native one). The viewer and `cambrian --save / --load` use this module; a
running organism loads at its next generation (run_vision.py's load request).
"""
from __future__ import annotations

import hashlib
import io
import json
import socket
import sys
import time
import zipfile
from pathlib import Path

FORMAT = 1
SUFFIX = ".cambrioid"
BEING = ("checkpoint.json", "episodes.npz", "cortex.json", "tally.json",
         "founder.json", "founder_seeded.txt", "seeded.txt")
RECORD = "lineage_history.json"
MEMBERS = set(BEING) | {RECORD, "manifest.json"}
# (H, 2026-09-30) at most this much per member, and in all: 20x and 80x the
# largest checkpoint measured in the fleet (0.83 MB) -- a larger file is not an
# organism of this program, and is refused before anything is unpacked.
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def _repo_on_path() -> None:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


class Refused(ValueError):
    """The file isn't a sound organism: why, in words."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def default_name(state: Path) -> str:
    """ecohost-gen<generation>-<date>.cambrioid"""
    gen = "unknown"  # (never "?": Windows refuses it in a file name)
    try:
        g = json.loads((state / "checkpoint.json").read_text(encoding="utf-8")).get("total_generation")
        if isinstance(g, int):
            gen = g
    except (OSError, ValueError):
        pass
    host = socket.gethostname().split(".")[0].lower()
    return f"{host}-gen{gen}-{time.strftime('%Y-%m-%d')}{SUFFIX}"


def save_bytes(state: Path, name: str | None = None) -> tuple[bytes, dict]:
    """The organism in `state`, as a .cambrioid file's bytes, and its manifest."""
    state = Path(state)
    ck = state / "checkpoint.json"
    if not ck.exists():
        raise Refused("no checkpoint here yet: nothing to save")
    data = {}
    try:
        for f in BEING:
            p = state / f
            if p.exists():
                data[f] = p.read_bytes()
        life = state / "life_history.json"
        if life.exists():
            data[RECORD] = life.read_bytes()
    except PermissionError as e:
        raise Refused(f"can't read {Path(e.filename).name} (its organism's own account owns it: try with sudo)")
    c = json.loads(data["checkpoint.json"])
    host = socket.gethostname().split(".")[0].lower()
    manifest = {"format": FORMAT, "name": name or f"{host} at generation {c.get('total_generation', '?')}",
                "ecohost": host, "generation": c.get("total_generation"), "saved_at": time.time(),
                "checkpoint_saved_at": c.get("saved_at"), "code_version": c.get("code_version"),
                "files": {k: _sha(v) for k, v in data.items()}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for k, v in data.items():
            z.writestr(k, v)
    return buf.getvalue(), manifest


def games_dir() -> Path:
    """Where saved organisms live by default: ~/Games/Lifeforms/Cambrioids (on
    Windows C:\\Users\\<you>\\Games\\Lifeforms\\Cambrioids; room beside it for other
    kinds of artificial life) -- made on first save."""
    return Path.home() / "Games" / "Lifeforms" / "Cambrioids"


# Where the viewer's "Save on this ecohost" puts one on a Linux install: the
# games_dir() of the organism's own account (home /srv/cambrian), which the
# viewer runs as; on Windows and macOS the viewer runs as you, so it is yours.
LINUX_ECOHOST_SAVES = Path("/srv/cambrian/Games/Lifeforms/Cambrioids")


def find(name: str) -> Path:
    """A file as given, else the same name (with or without .cambrioid) in
    games_dir(), then in the Linux ecohost's own save folder."""
    p = Path(name).expanduser()
    if p.exists():
        return p
    for d in (games_dir(), LINUX_ECOHOST_SAVES):
        for q in (d / name, d / (name + SUFFIX)):
            if q.exists():
                return q
    raise Refused(f"no such file: {name} (nor in {games_dir()} or {LINUX_ECOHOST_SAVES})")


def save(state: Path, dest: Path | None = None, name: str | None = None) -> Path:
    blob, _ = save_bytes(state, name)
    # Any folder that doesn't exist yet is made, parents and all (~/Games
    # included): the default one, one given, or the one a given file goes in.
    # A given path is a folder unless it ends in .cambrioid.
    dest = games_dir() if dest is None else Path(dest).expanduser()
    if dest.is_dir() or dest.suffix != SUFFIX:
        dest.mkdir(parents=True, exist_ok=True)
        dest = dest / default_name(Path(state))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    return dest


def read(blob: bytes) -> tuple[dict, dict]:
    """(manifest, {member: bytes}) of a .cambrioid file's bytes -- every check
    passed -- or Refused with the reason."""
    if len(blob) > MAX_TOTAL_BYTES:
        raise Refused(f"larger than {MAX_TOTAL_BYTES // 2**20} MB")
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        raise Refused("not a .cambrioid file (not a zip)")
    with z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        unknown = [n for n in names if n not in MEMBERS]
        if unknown:
            raise Refused(f"unexpected contents: {', '.join(unknown[:5])}")
        if len(set(names)) != len(names):
            raise Refused("a member appears twice")
        if "manifest.json" not in names or "checkpoint.json" not in names:
            raise Refused("no manifest or no checkpoint")
        big = [i.filename for i in infos if i.file_size > MAX_MEMBER_BYTES]
        if big or sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:  # sizes checked before anything is unpacked
            raise Refused("a member is too large")
        try:  # a member whose header lies about its size, or damaged data, fails here: refused, never a crash
            files = {n: z.read(n) for n in names}
        except Exception as e:
            raise Refused(f"damaged ({type(e).__name__})")
    try:
        manifest = json.loads(files.pop("manifest.json"))
    except ValueError:
        raise Refused("its manifest isn't JSON")
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise Refused(f"format {manifest.get('format') if isinstance(manifest, dict) else '?'} (this program reads format {FORMAT})")
    sums = manifest.get("files") or {}
    for n, b in files.items():
        if sums.get(n) != _sha(b):
            raise Refused(f"{n} doesn't match its checksum (damaged or altered)")
    if set(sums) != set(files):
        raise Refused("its manifest and its contents disagree")
    for n in ("checkpoint.json", "cortex.json", "tally.json", "founder.json", RECORD):
        if n in files:
            try:
                json.loads(files[n])
            except ValueError:
                raise Refused(f"{n} isn't JSON")
    c = json.loads(files["checkpoint.json"])
    if not isinstance(c, dict) or not isinstance(c.get("genome"), dict):
        raise Refused("its checkpoint has no genome")
    _repo_on_path()
    from tools import fleet
    for n in ("checkpoint.json", "founder.json"):
        if n in files:
            why = fleet.vet_genome(json.loads(files[n]).get("genome"))
            if why:
                raise Refused(f"{n}: its genome is refused -- {why}")
    if "episodes.npz" in files:
        import numpy as np
        try:
            with np.load(io.BytesIO(files["episodes.npz"]), allow_pickle=False) as e:
                for k in e.files:
                    e[k]  # every array readable, none a pickle
        except Exception as ex:
            raise Refused(f"its episodes can't be read ({type(ex).__name__})")
    return manifest, files


def install(blob: bytes, state: Path) -> tuple[dict, Path]:
    """Loads a .cambrioid into `state` (its organism stopped): the current
    lineage moved aside first. Returns (its manifest, the backup folder)."""
    manifest, files = read(blob)
    state = Path(state)
    _repo_on_path()
    from tools.reset_founder import reset
    from fishbowl import sandbox
    backup, _ = reset(state, "before-load")
    for n, b in files.items():
        if n == RECORD:
            continue  # its old ecohost's record stays in the file; this ecohost's history goes on
        tmp = state / (n + ".tmp")
        tmp.write_bytes(b)
        tmp.replace(state / n)
    # the record goes to THIS state folder's history (from the command line,
    # `state` needn't be the one sandbox was imported for): its paths pointed
    # there for the call, and put back after
    keep = {k: getattr(sandbox, k) for k in ("STATE_DIR", "LIFE_HISTORY_PATH", "LIFE_HISTORY_PREV_PATH")}
    try:
        sandbox.STATE_DIR = state
        sandbox.LIFE_HISTORY_PATH = state / "life_history.json"
        sandbox.LIFE_HISTORY_PREV_PATH = state / "life_history.prev.json"
        sandbox.record_life({"event": "loaded", "name": manifest.get("name"), "from": manifest.get("ecohost"),
                             "generation": manifest.get("generation"), "saved_at": manifest.get("saved_at"),
                             "backup": backup.name})
    finally:
        for k, v in keep.items():
            setattr(sandbox, k, v)
    return manifest, backup


def safe_name(name: str) -> bool:
    """A save's file name as the viewer accepts one: letters, digits, dots,
    dashes, underscores, ending .cambrioid -- no folders, nothing hidden."""
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}\.cambrioid", name or "")) and ".." not in name


def list_saves(folder: Path | None = None) -> list[dict]:
    """This ecohost's saved organisms (its games_dir()), newest first: the file,
    its size, and its manifest's name, ecohost, generation and when saved --
    read from the manifest alone (size-checked), never unpacking the rest."""
    folder = games_dir() if folder is None else Path(folder)
    out = []
    for p in sorted(folder.glob("*" + SUFFIX)) if folder.is_dir() else []:
        if not safe_name(p.name):
            continue
        entry = {"file": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        try:
            with zipfile.ZipFile(p) as z:
                info = z.getinfo("manifest.json")
                if info.file_size <= MAX_MEMBER_BYTES:
                    m = json.loads(z.read("manifest.json"))
                    entry.update({k: m.get(k) for k in ("name", "ecohost", "generation", "saved_at")})
        except (zipfile.BadZipFile, KeyError, ValueError, OSError):
            entry["damaged"] = True
        out.append(entry)
    return sorted(out, key=lambda e: -(e.get("saved_at") or e["mtime"]))


def store(blob: bytes, name: str, folder: Path | None = None) -> Path:
    """An arriving file (copied from another ecohost, or uploaded), checked as
    a load checks it, kept in this ecohost's folder -- under its own name, or
    name-2, name-3 ... when that one is taken (nothing is overwritten)."""
    read(blob)  # refused files are never kept
    if not safe_name(name):
        raise Refused("not a plain .cambrioid file name")
    folder = games_dir() if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dest, stem, k = folder / name, name[:-len(SUFFIX)], 2
    while dest.exists():
        dest, k = folder / f"{stem}-{k}{SUFFIX}", k + 1
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    return dest


def stage(path: Path, state: Path) -> dict:
    """Checks a file and stages it for the organism in `state` to load at its
    next generation (the request its main loop answers, as for the viewer's
    Load organism). Run as root (Linux: `cambrian --load`), the staged files
    are given to the state folder's owner. Returns its manifest."""
    import os
    blob = Path(path).read_bytes()
    manifest, _ = read(blob)
    state = Path(state)
    staged, req = state / "load.cambrioid", state / "load.request"
    tmp = state / "load.cambrioid.tmp"
    tmp.write_bytes(blob)
    tmp.replace(staged)
    req.write_text(json.dumps({"t": time.time()}), encoding="utf-8")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        st = state.stat()
        for p in (staged, req):
            os.chown(p, st.st_uid, st.st_gid)
    return manifest


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "find":
        try:
            print(find(argv[1]))
        except Refused as e:
            print(e, file=sys.stderr)
            return 1
        return 0
    if len(argv) == 3 and argv[0] == "stage":
        try:
            m = stage(find(argv[1]), Path(argv[2]))
        except Refused as e:
            print(f"refused: {e}")
            return 1
        print(f"{m.get('name')} (from {m.get('ecohost')}, generation {m.get('generation')}) is staged: "
              "the organism saves, keeps itself in a backup, and the loaded one takes its place at its next "
              "generation (at its next start, if it isn't running)")
        return 0
    if len(argv) >= 2 and argv[0] == "save":
        args, name = list(argv[1:]), None
        if "--name" in args:
            i = args.index("--name")
            name = args[i + 1] if i + 1 < len(args) else None
            del args[i:i + 2]
        try:
            dest = save(Path(args[0]), Path(args[1]) if len(args) > 1 else None, name)
        except Refused as e:
            print(f"refused: {e}")
            return 1
        print(f"saved: {dest}")
        return 0
    if len(argv) == 2 and argv[0] == "check":
        try:
            m, files = read(find(argv[1]).read_bytes())
        except Refused as e:
            print(f"refused: {e}")
            return 1
        print(f"a sound organism: {m.get('name')} (from {m.get('ecohost')}, generation {m.get('generation')}, "
              f"saved {time.ctime(m.get('saved_at') or 0)}; {', '.join(sorted(files))})")
        return 0
    if len(argv) == 3 and argv[0] == "load":
        try:
            m, backup = install(find(argv[1]).read_bytes(), Path(argv[2]))
        except Refused as e:
            print(f"refused: {e}")
            return 1
        print(f"loaded {m.get('name')}; the lineage it replaced is in {backup.name}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):  # a console that can't show jīng or 精 shows "?", never a crash
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    sys.exit(main(sys.argv[1:]))
