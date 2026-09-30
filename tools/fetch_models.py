"""The detector models, fetched when missing -- every installer runs this.

    python tools/fetch_models.py <models dir>

They are this repo's release assets (github.com/thefangeddeity/cambrian-perception,
release "models-v1": Ultralytics' YOLOv8n and its Open Images V7 variant,
exported to ONNX, AGPL-3.0 -- the release notes say where they came from and
how to make them yourself). Each is checked against the SHA-256 below before
it is kept; a failed or partial download is discarded, never used. A file
already present is left alone. The standard library only (it runs before
anything else is installed). Without the first model the organism runs with no
hosts to feed on (snacks only); the other two are optional (plants beyond
potted ones).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/thefangeddeity/cambrian-perception/releases/download/models-v1/"
MODELS = {  # name: (sha256, required)
    "yolov8n.onnx": ("b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd", True),
    "yolov8n-oiv7.onnx": ("375889a98a1e70c66bcfe711c3fc5351e639e2990af31ec9b8451959526b6c8d", False),
    "yolov8n-oiv7.names.json": ("508523a62720c6ee9e20a52259c4a58e35993bcc5dc72c9fdd478a124b175ba0", False),
}


def fetch(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    missing_required = False
    for name, (sha, required) in MODELS.items():
        dest = folder / name
        if dest.exists():
            continue
        part = dest.with_name(name + ".part")
        try:
            with urllib.request.urlopen(BASE + name, timeout=60) as r, open(part, "wb") as f:
                h = hashlib.sha256()
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != sha:
                raise ValueError(f"checksum mismatch ({h.hexdigest()[:12]}...)")
            part.replace(dest)
            print(f"  model: {name} fetched")
        except Exception as e:
            part.unlink(missing_ok=True)
            print(f"  model: {name} not fetched ({e})" + ("" if required else " -- optional"))
            missing_required |= required
    if missing_required:
        print(f"  WARNING: no {next(iter(MODELS))} in {folder} -- it runs with no hosts to feed on until one is there "
              f"(re-run this, or see the models-v1 release notes to make it yourself)")
    return 0  # never fails an install: the organism runs without it


if __name__ == "__main__":
    sys.exit(fetch(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "models"))
