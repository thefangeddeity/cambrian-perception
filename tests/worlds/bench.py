"""
The worlds bench: its horizon organ against worlds whose horizon is known.

  python -m tests.worlds.bench run                 every world: synthetic + labelled captures
  python -m tests.worlds.bench capture NAME [-n 12]   frames from a stream in streams.json
  python -m tests.worlds.bench sheet FILE OUT.jpg  a contact sheet of a capture, to label by eye
  python -m tests.worlds.bench label FILE Y        its horizon by eye: a height (0 top, 1 bottom),
                                                   or "above" / "below" when out of view

Synthetic worlds (synthetic.py) are drawn from a known camera: exact. Stream
captures go to media/worlds/ (gitignored: other people's footage, not ours
to republish), each an .npz of grey frames, 320 wide, with its source, time
and -- once a person has looked -- its label. The report: for each world, the
median estimate, how many frames it answered, its error, and whether the
organism would believe it (organism.HORIZON_MAX_SPREAD: the estimates'
spread). It is a report, not a gate; the self-test holds the gates.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CAPTURES = ROOT / "media" / "worlds"
STREAMS = pathlib.Path(__file__).with_name("streams.json")


def _grey320(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    s = 320 / max(g.shape)
    return cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)


def _url(entry: dict) -> str:
    if entry.get("url"):
        return entry["url"]
    exe = pathlib.Path(sys.executable).parent / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
    out = subprocess.run([str(exe) if exe.exists() else "yt-dlp", "--flat-playlist", "-j", "ytsearch12:" + entry["search"]],
                         capture_output=True, text=True, timeout=120).stdout
    for line in out.splitlines():
        e = json.loads(line)
        if e.get("live_status") == "is_live":
            return "https://www.youtube.com/watch?v=" + e["id"]
    raise SystemExit(f"no live result for {entry['search']!r}")


def capture(name: str, n: int = 12, every: int = 8) -> pathlib.Path:
    import run_vision
    entry = next((e for e in json.loads(STREAMS.read_text(encoding="utf-8"))["streams"] if e["name"] == name), None)
    if entry is None:
        raise SystemExit(f"no stream {name!r} in {STREAMS.name}")
    url = _url(entry)
    cap = cv2.VideoCapture(run_vision._resolve_live_url(url), cv2.CAP_FFMPEG)
    frames, k = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if k % every == 0:
            frames.append(_grey320(f))
        k += 1
    cap.release()
    if not frames:
        raise SystemExit(f"{name}: no frames")
    CAPTURES.mkdir(parents=True, exist_ok=True)
    path = CAPTURES / f"{name}-{time.strftime('%Y%m%d-%H%M')}.npz"
    meta = {"name": name, "url": url, "time": time.strftime("%Y-%m-%d %H:%M %Z"), "label": None}
    np.savez_compressed(path, frames=np.stack(frames), meta=json.dumps(meta))
    print(f"{path.name}: {len(frames)} frames -- look (sheet) and label it")
    return path


def _load(path: pathlib.Path):
    z = np.load(path)
    return list(z["frames"]), json.loads(str(z["meta"]))


def label(path: pathlib.Path, y: str) -> None:
    frames, meta = _load(path)
    meta["label"] = y if y in ("above", "below") else float(y)
    np.savez_compressed(path, frames=np.stack(frames), meta=json.dumps(meta))
    print(f"{path.name}: horizon {meta['label']}")


def sheet(path: pathlib.Path, out: pathlib.Path) -> None:
    from fishbowl import organs
    frames, meta = _load(path)
    tiles = []
    for i, g in enumerate(frames[:8]):
        t = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        h, w = g.shape
        for f in (0.25, 0.5, 0.75):  # a scale to judge by
            cv2.line(t, (0, int(f * h)), (6, int(f * h)), (0, 200, 0), 1)
        e = organs.horizon(g, None, np.random.default_rng(i))
        if e:
            y, dy = e["y"] * h, np.tan(e["roll"]) * w / 2
            cv2.line(t, (0, int(y - dy)), (w, int(y + dy)), (0, 255, 255), 1)
            cv2.putText(t, f"{e['y']:.2f} {e['how']}", (4, 14), 0, 0.45, (0, 255, 255), 1)
        tiles.append(cv2.resize(t, (320, 180)))
    while len(tiles) % 4:
        tiles.append(np.zeros((180, 320, 3), np.uint8))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(f"{out}: the organ's estimate in yellow; ticks at 0.25 / 0.5 / 0.75")


def _error(est: float, truth) -> float:
    if truth == "above":
        return max(0.0, est)
    if truth == "below":
        return max(0.0, 1.0 - est)
    return abs(est - float(truth))


def run() -> None:
    from fishbowl import organs
    from fishbowl.organism import HORIZON_MAX_SPREAD
    from tests.worlds import synthetic
    worlds = [(name, frames, truth) for name, frames, truth in synthetic.worlds()]
    for path in sorted(CAPTURES.glob("*.npz")) if CAPTURES.exists() else []:
        frames, meta = _load(path)
        if meta.get("label") is not None:
            worlds.append((path.stem, frames, meta["label"]))
        else:
            print(f"(unlabelled, skipped: {path.name})")
    print(f"{'world':28s} {'truth':>6s} {'median':>7s} {'answered':>9s} {'error':>6s}  believed")
    for name, frames, truth in worlds:
        ys = [e["y"] for e in (organs.horizon(g, None, np.random.default_rng(i)) for i, g in enumerate(frames)) if e]
        med = float(np.median(ys)) if ys else None
        spread = float(np.subtract(*np.percentile(ys, [75, 25]))) if len(ys) > 1 else 0.0
        believed = med is not None and spread <= HORIZON_MAX_SPREAD
        t = truth if isinstance(truth, str) else f"{truth:.2f}"
        print(f"{name:28s} {t:>6s} {'-' if med is None else f'{med:.2f}':>7s} {len(ys):>4d}/{len(frames):<4d}"
              f" {'-' if med is None else f'{_error(med, truth):.2f}':>6s}  {'yes' if believed else 'no (learned stands)'}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    c = sub.add_parser("capture"); c.add_argument("name"); c.add_argument("-n", type=int, default=12)
    s = sub.add_parser("sheet"); s.add_argument("file", type=pathlib.Path); s.add_argument("out", type=pathlib.Path)
    lab = sub.add_parser("label"); lab.add_argument("file", type=pathlib.Path); lab.add_argument("y")
    a = ap.parse_args()
    if a.cmd == "run":
        run()
    elif a.cmd == "capture":
        capture(a.name, a.n)
    elif a.cmd == "sheet":
        sheet(a.file, a.out)
    else:
        label(a.file, a.y)


if __name__ == "__main__":
    main()
