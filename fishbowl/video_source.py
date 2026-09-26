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
import os
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


# Frames kept for the viewer's replay: twice the default window, so the
# frames of the run it is showing are still there when it shows them.
FRAME_RING = 1200
# A camera slower than this is kept at every frame (a 15 fps camera keeps
# every 2nd, 7.5/s; an 8 fps one would drop to 4/s -- too coarse for motion).
SLOW_SOURCE_FPS = 12.0
# Prey detection at most this often (it ran every 3rd kept frame, ~0.4 s).
DETECT_INTERVAL_S = 0.3


class LiveFeed:
    """
    A live camera kept continuously in memory: a background thread reads
    every frame, keeps every `stride`-th one (7.5 frames/s from a 15 fps
    camera) -- or every frame from a slow camera (under SLOW_SOURCE_FPS),
    whose motion would otherwise be too coarse to follow -- reduces it to
    its retina grid as it arrives, and holds only the newest `window` of
    them. Prey detection runs on its own thread (below), so a slow detector
    on a busy host never makes the camera drop frames. Nothing is ever written to
    disk. Each generation takes a snapshot, so the organism is always
    scored on what the camera is seeing now, not on a clip captured when
    the process started. Reconnects if the device drops.
    """

    def __init__(self, source: str, stride: int = 2, window: int = 600, max_dim: int = DEFAULT_MAX_DIM,
                 detector=None, frames_dir=None):
        from .retina import frame_to_vector
        self._to_vector = frame_to_vector
        # Prey (see prey.py): detected on its own thread, on the newest
        # full-resolution colour frame, at most every DETECT_INTERVAL_S;
        # only boxes are kept, each result is attached to the frames that
        # arrive until the next one (so boxes trail by one detection).
        self.detector = detector
        self._last_prey: list = []
        self._latest_full = None
        self.source, self.stride, self.max_dim = source, stride, max_dim
        # Each kept frame as a small JPEG, for the viewer's replay of its
        # latest run: the last FRAME_RING of them, f<index>.jpg. Only ever
        # pointed at the RAM runtime dir (run_vision.py), never at disk;
        # None = off.
        self.frames_dir = frames_dir
        self.newest_time = None  # set by snapshot()
        self.snapshot_first = 0  # index of the first frame in the last snapshot
        self._buf: collections.deque = collections.deque(maxlen=window)
        self._lock = threading.Lock()
        self.total = 0  # frames ever kept -- lets callers find what's new since their last look
        self._times: collections.deque = collections.deque(maxlen=window)
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="LiveFeed", daemon=True)
        self._thread.start()
        if detector is not None:
            threading.Thread(target=self._detect_loop, name="LiveFeed-prey", daemon=True).start()

    def _detect_loop(self) -> None:
        done = None
        while not self._stop:
            frame = self._latest_full
            if frame is None or frame is done:
                time.sleep(0.05)
                continue
            done = frame
            try:
                self._last_prey = self.detector.detect(frame)
            except cv2.error:
                pass
            time.sleep(DETECT_INTERVAL_S)

    def _run(self) -> None:
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            # RTSP over UDP drops packets under load (corrupt H.264
            # macroblocks); TCP delivers whole frames.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        while not self._stop:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                time.sleep(2.0)
                continue
            count = 0
            rate = cap.get(cv2.CAP_PROP_FPS)
            stride = 1 if 0 < rate < SLOW_SOURCE_FPS else self.stride
            try:
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    count += 1
                    if count % stride:
                        continue
                    self._latest_full = frame  # for the prey thread
                    if self.frames_dir is not None:
                        self._write_frame(frame, self.total)
                    h, w = frame.shape[:2]
                    if max(h, w) > self.max_dim:
                        scale = self.max_dim / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    vec = self._to_vector(gray)
                    with self._lock:
                        # colour frame kept (downscaled, memory only) for the
                        # gaze's colour receptors -- see retina.opponent_planes
                        self._buf.append((gray, vec, self._last_prey, frame))
                        self._times.append(time.time())
                        self.total += 1
            finally:
                cap.release()
            time.sleep(1.0)

    def _write_frame(self, frame: np.ndarray, index: int, max_dim: int = 640) -> None:
        """Best effort: a replay missing a frame is fine."""
        try:
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                s = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                tmp = self.frames_dir / "tmp.jpg"
                tmp.write_bytes(jpg.tobytes())
                os.replace(tmp, self.frames_dir / f"f{index}.jpg")
            (self.frames_dir / f"f{index - FRAME_RING}.jpg").unlink(missing_ok=True)
        except (OSError, cv2.error):
            pass

    def wait_for(self, n: int, timeout: float = 180.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if len(self._buf) >= n:
                    return True
            time.sleep(0.2)
        return False

    def snapshot(self) -> tuple[list[np.ndarray], np.ndarray, int, list, list[np.ndarray]]:
        """(grey frames, their retina vectors, total frames ever kept, prey boxes per frame, colour frames) -- a consistent copy of the current window."""
        with self._lock:
            items = list(self._buf)
            total = self.total
            # when the newest frame of this snapshot arrived (for "how far
            # behind live is its gaze")
            self.newest_time = self._times[-1] if self._times else None
            self.snapshot_first = total - len(items)
        return ([it[0] for it in items], np.array([it[1] for it in items]), total,
                [it[2] for it in items], [it[3] for it in items])

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
    frames, prey, last, colour = [], [], [], []
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
            colour.append(frame)
            prey.append(last)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames, prey, colour
