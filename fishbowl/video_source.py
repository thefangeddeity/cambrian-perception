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


def read_frames(source: str, stride: int = 1, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """
    source: a file path (curated clip) or a device path/index (e.g.
    "/dev/video0" or 0) for a live camera. Yields grayscale float64
    frames one at a time -- never buffers the whole clip in memory,
    never writes anything to disk.
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                yield gray
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            count += 1
    finally:
        cap.release()
