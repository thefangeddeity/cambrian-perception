from __future__ import annotations

"""
Real frame capture -- a video file (curated curriculum clip) or a
live device (Tanzania's own /dev/video*). Fixed, human-written, never
evolved, same category as retina.py/reflexes.py/prey.py: the
genome never touches this code, it only ever sees frame_to_vector()'s
already-reduced output. Frames are streamed and handed to the caller
one at a time -- nothing here ever writes a frame to disk. See
README's fishbowl boundary.
"""

import collections
import threading
import time
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


class LiveFeed:
    """
    A live camera kept continuously in memory: a background thread reads
    every frame, keeps every `stride`-th one (15 frames/s from a 30 fps
    camera), reduces it to its retina grid as it arrives, and holds only
    the newest `window` of them (about 40 s). Nothing is ever written to
    disk. Each generation takes a snapshot, so the organism is always
    scored on what the camera is seeing now, not on a clip captured when
    the process started. Reconnects if the device drops.
    """

    def __init__(self, source: str, stride: int = 2, window: int = 600, max_dim: int = DEFAULT_MAX_DIM,
                 detector=None, detect_every: int = 3):
        from .retina import frame_to_vector
        self._to_vector = frame_to_vector
        # Prey (see prey.py): detected on the full-resolution colour frame
        # every detect_every kept frames; only boxes are kept, and each
        # result is held until the next detection.
        self.detector, self.detect_every = detector, detect_every
        self._last_prey: list = []
        self.source, self.stride, self.max_dim = source, stride, max_dim
        self._buf: collections.deque = collections.deque(maxlen=window)
        self._lock = threading.Lock()
        self.total = 0  # frames ever kept -- lets callers find what's new since their last look
        self._times: collections.deque = collections.deque(maxlen=window)
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="LiveFeed", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                time.sleep(2.0)
                continue
            count = 0
            try:
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    count += 1
                    if count % self.stride:
                        continue
                    if self.detector is not None and (self.total % self.detect_every == 0):
                        self._last_prey = self.detector.detect(frame)
                    h, w = frame.shape[:2]
                    if max(h, w) > self.max_dim:
                        scale = self.max_dim / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    vec = self._to_vector(gray)
                    with self._lock:
                        self._buf.append((gray, vec, self._last_prey))
                        self._times.append(time.time())
                        self.total += 1
            finally:
                cap.release()
            time.sleep(1.0)

    def wait_for(self, n: int, timeout: float = 180.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if len(self._buf) >= n:
                    return True
            time.sleep(0.2)
        return False

    def snapshot(self) -> tuple[list[np.ndarray], np.ndarray, int, list]:
        """(frames, their retina vectors, total frames ever kept, prey boxes per frame) -- a consistent copy of the current window."""
        with self._lock:
            items = list(self._buf)
            total = self.total
        return [f for f, _, _ in items], np.array([v for _, v, _ in items]), total, [p for _, _, p in items]

    def frames_per_second(self) -> float:
        """Real rate of kept frames (the camera's own rate varies with light)."""
        with self._lock:
            t = list(self._times)
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def close(self) -> None:
        self._stop = True


def read_frames_with_prey(source: str, stride: int = 2, max_frames: int | None = None,
                          max_dim: int = DEFAULT_MAX_DIM, detector=None, detect_every: int = 3) -> tuple[list[np.ndarray], list]:
    """read_frames() for a fixed clip, plus prey boxes per kept frame
    (detected on the colour frame every detect_every kept frames, held in
    between). Frames stay in memory only, never written to disk."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    frames, prey, last = [], [], []
    try:
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if (count - 1) % stride:
                continue
            if detector is not None and len(frames) % detect_every == 0:
                last = detector.detect(frame)
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            prey.append(last)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames, prey
