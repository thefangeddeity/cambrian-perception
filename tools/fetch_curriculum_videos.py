#!/usr/bin/env python3
from __future__ import annotations

"""
Downloads short clips from real, established, long-running public
nature-cam sources into media/ (gitignored -- see README's fishbowl
boundary: raw footage is never committed, and this tool itself is NOT
part of the sandboxed organism -- it's a human-run utility, same
status as any other tools/ script, outside the "no subprocess"
boundary that applies to fishbowl/ specifically.

Sources chosen for being established institutions (Cornell Lab of
Ornithology, Explore.org) with long-running public streams, not
random uploads that could vanish -- and staged easiest-to-hardest by
how evolutionarily ancient the real reflex each one exercises is (see
README's curriculum section).

Usage:
    python3 tools/fetch_curriculum_videos.py [--seconds N] [--stage NAME]
"""

import argparse
import subprocess
import sys
from pathlib import Path

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"

# stage -> [(name, url), ...]
CURRICULUM = {
    "stage1_luminance_and_motion": [
        ("cornell_feederwatch", "https://www.youtube.com/watch?v=x10vL6_47Dw"),
        ("explore_african_lookout", "https://explore.org/livecams/currently-live/african-animal-lookout-camera"),
    ],
    "stage3_loom": [
        ("explore_kitten_rescue", "https://explore.org/livecams/kitten-rescue/kitten-rescue-cam"),
        ("pixcams_wildlife", "https://www.youtube.com/watch?v=XfNhPa26fP8"),
    ],
}


def fetch_clip(name: str, url: str, out_dir: Path, seconds: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.mp4"
    if dest.exists():
        print(f"  {name}: already present, skipping")
        return
    print(f"  {name}: fetching {seconds}s from {url}")
    # --download-sections grabs just the first N seconds of a live/
    # long stream via yt-dlp's own real support for that, rather than
    # downloading the whole thing and trimming after.
    result = subprocess.run(
        [
            "yt-dlp",
            "--download-sections", f"*0-{seconds}",
            "-f", "mp4",
            "-o", str(dest),
            url,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()[-500:]}")
    else:
        print(f"    OK -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--stage", default=None, help="only fetch this stage")
    args = parser.parse_args()

    stages = {args.stage: CURRICULUM[args.stage]} if args.stage else CURRICULUM

    for stage, clips in stages.items():
        print(f"[{stage}]")
        out_dir = MEDIA_DIR / stage
        for name, url in clips:
            fetch_clip(name, url, out_dir, args.seconds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
