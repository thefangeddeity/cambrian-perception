from __future__ import annotations

"""
Real frame capture -- a video file (curated curriculum clip) or a
live device (Tanzania's own /dev/video*). Fixed, human-written, never
evolved, same category as retina.py/reflexes.py/conspec.py: the
genome never touches this code, it only ever sees frame_to_vector()'s
already-reduced output. Frames are streamed and handed to the caller
one at a time -- nothing here ever writes a frame to disk. See
README's fishbowl boundary.
"""

from typing import Iterator

import cv2
import numpy as np


DEFAULT_MAX_DIM = 320  # real HD sources (Cornell's own feed is 1920x1080)


def read_frames(source: str, stride: int = 1, max_frames: int | None = None, max_dim: int = DEFAULT_MAX_DIM) -> Iterator[np.ndarray]:
    """
    source: a file path (curated clip) or a device path/index (e.g.
    "/dev/video0" or 0) for a live camera. Yields grayscale uint8
    frames one at a time -- never buffers the whole clip in memory
    itself, never writes anything to disk.

    Real bug, caught live (deployed service on Tanzania thrashing in
    swap, generation counter stalled): this used to cast every frame
    to float64 before yielding it -- 8 bytes/pixel, and run_vision.py
    holds up to max_frames of these in a list AT ONCE for the whole
    run (real, deliberate -- see its own docstring on why every
    generation needs the SAME in-memory clip for fair comparisons).
    At a real HD source and max_frames=600 that's multiple real
    gigabytes just for one clip. retina.py's own frame_to_vector()
    already does `.astype(np.float64)` itself, per frame, at the
    moment it's actually used -- this cast was pure duplicated,
    wasted memory, not needed for correctness at all. uint8 (the
    format cv2 already hands back) is 8x smaller and exactly what
    frame_to_vector() already expects (it checks `frame.max() > 1.5`
    to detect an un-normalized 0-255 input).

    max_dim: also resizes each frame so its longer side is at most
    this many pixels, preserving aspect ratio -- a real HD source
    (Cornell's own feed is 1920x1080) has no reason to stay full
    resolution when retina.py reduces everything to a 12x12 grid
    downstream anyway. This is the bigger of the two real fixes:
    dtype alone (uint8 vs float64) still left ~1.24GB for 600 real HD
    frames, uncomfortably close to a real resource cap on a shared
    machine; resizing to 320px cuts that by ~35x on top of it, and
    makes every per-generation fovea/retina computation proportionally
    cheaper too -- the actual root cause of the deployed service
    stalling (swapping, not computing) was memory pressure, not raw
    compute cost.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")

    try:
        count = 0
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % stride == 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield gray
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            count += 1
    finally:
        cap.release()
