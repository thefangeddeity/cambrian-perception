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

# Quiet OpenCV and FFmpeg (a 2026-09-28 infrastructure review): their warnings
# and errors printed whole signed stream URLs -- carrying the host's public IP
# -- into the logs, dozens a day, and our own stall and failure handling
# already reports what matters. Set before OpenCV loads; an operator can still
# override them in the environment.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET
import shutil
import subprocess
import threading
import time
from typing import Iterator

import cv2
import numpy as np


DEFAULT_MAX_DIM = 320  # real HD sources (Cornell's own feed is 1920x1080)


class _FFmpegPipe:
    """
    The part of cv2.VideoCapture this module uses, decoded by an ffmpeg
    process into raw BGR frames on a pipe (memory only, like everything
    here). For hosts whose OpenCV has no FFmpeg of its own -- the macOS
    wheels, which can't open a stream (rtsp://) or a web video at all.
    H.264 decoding is bit-exact by the standard, and the conversion to
    BGR is swscale's bicubic, as in OpenCV's own FFmpeg backend: the
    organism sees the same frames it would on any other host.
    """

    def __init__(self, source: str, ffmpeg: str):
        self._proc = None
        self._fps = 0.0
        net = ["-rtsp_transport", "tcp"] if source.startswith("rtsp") else []
        probe = shutil.which("ffprobe", path=os.path.dirname(ffmpeg)) or "ffprobe"
        try:
            out = subprocess.run([probe, "-v", "error", *net, "-select_streams", "v:0", "-show_entries",
                                  "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "csv=p=0", source],
                                 capture_output=True, text=True, timeout=30).stdout.split()[0].split(",")
            self._w, self._h = int(out[0]), int(out[1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return
        for rate in out[2:4]:
            num, _, den = rate.partition("/")
            if den and float(den) and float(num):
                self._fps = float(num) / float(den)
                break
        self._proc = subprocess.Popen([ffmpeg, "-nostdin", "-loglevel", "error", *net, "-i", source,
                                       "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def read(self):
        if self._proc is None:
            return False, None
        size = self._w * self._h * 3
        buf = self._proc.stdout.read(size)
        if len(buf) < size:
            return False, None
        return True, np.frombuffer(buf, np.uint8).reshape(self._h, self._w, 3).copy()

    def get(self, prop) -> float:
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None


def _ffmpeg_binary() -> str | None:
    # A boot service's PATH is minimal: also look where Homebrew puts it.
    return shutil.which("ffmpeg") or shutil.which("ffmpeg", path="/opt/homebrew/bin:/usr/local/bin")


def open_capture(source):
    """cv2.VideoCapture, or an ffmpeg pipe for a stream/file when this
    OpenCV was built without FFmpeg (and an ffmpeg is installed)."""
    if not isinstance(source, int) and not cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG):
        ffmpeg = _ffmpeg_binary()
        if ffmpeg:
            return _FFmpegPipe(str(source), ffmpeg)
    if not isinstance(source, int) and cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)  # a URL straight to FFmpeg, not tried as an image sequence first
    return cv2.VideoCapture(source)


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
    resolution when retina.py reduces everything to a few hundred receptors
    downstream anyway. This is the bigger of the two real fixes:
    dtype alone (uint8 vs float64) still left ~1.24GB for 600 real HD
    frames, uncomfortably close to a real resource cap on a shared
    machine; resizing to 320px cuts that by ~35x on top of it, and
    makes every per-generation fovea/retina computation proportionally
    cheaper too -- the actual root cause of the deployed service
    stalling (swapping, not computing) was memory pressure, not raw
    compute cost.
    """
    cap = open_capture(source)
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



class FrameRing:
    """
    The replay frames (LiveFeed._write_frame) held in this process's own
    memory, for hosts with no RAM directory to put them in (Windows,
    macOS: no /dev/shm). Served to the viewer on 127.0.0.1 only; the
    port is published in `port_file`. Nothing reaches disk.
    """

    def __init__(self, size: int = FRAME_RING):
        self.size = size
        self._jpgs: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def put(self, epoch: int, index: int, jpg: bytes) -> None:
        with self._lock:
            self._jpgs[(epoch, index)] = jpg
            while len(self._jpgs) > self.size:
                self._jpgs.popitem(last=False)

    def get(self, epoch: int, index: int) -> bytes | None:
        with self._lock:
            return self._jpgs.get((epoch, index))

    def serve(self, port_file) -> int:
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse
        ring = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                q = parse_qs(urlparse(self.path).query)
                e, i = (q.get("e") or [""])[0], (q.get("i") or [""])[0]
                body = ring.get(int(e), int(i)) if e.isdigit() and i.isdigit() else None
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        class QuietServer(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):  # a viewer closing mid-frame is not an error
                import sys
                if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                    super().handle_error(request, client_address)

        server = QuietServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, name="FrameRing", daemon=True).start()
        port = server.server_address[1]
        tmp = port_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
        os.replace(tmp, port_file)
        return port


def _is_live_source(source) -> bool:
    """A camera (device index or /dev/video*) or a network camera stream."""
    return isinstance(source, int) or str(source).startswith(("/dev/video", "/dev/v4l/", "rtsp://", "rtsps://", "srt://"))


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
                 detector=None, frames_dir=None, epoch: int = 0, resolve=None):
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
        # A stream's address is signed and expires (hours): on a failed
        # reconnect it is resolved afresh (resolve() -> a new address), so one
        # long process can watch one stream for days.
        self.resolve = resolve
        # Each kept frame as a small JPEG, for the viewer's replay of its
        # latest run: the last FRAME_RING of them, f<index>.jpg. Only ever
        # pointed at the RAM runtime dir (run_vision.py), never at disk --
        # or a FrameRing in this process's memory where there is none;
        # None = off.
        self.frames_dir, self.epoch = frames_dir, epoch
        if frames_dir is not None and not isinstance(frames_dir, FrameRing):
            self._drop_old_runs()
        self.newest_time = None  # set by snapshot()
        self.ended = False       # a recording played to its end (a live source never ends: it reconnects)
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
            started = time.time()
            try:
                self._last_prey = self.detector.detect(frame)
            except cv2.error:
                pass
            # At most a quarter of the time detecting: on a busy host where a
            # detection takes seconds (3.5 s on a laptop running a livecam
            # server), back-to-back detection starved everything else.
            time.sleep(max(DETECT_INTERVAL_S, 3.0 * (time.time() - started)))

    def _run(self) -> None:
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            # RTSP over UDP drops packets under load (corrupt H.264
            # macroblocks); TCP delivers whole frames.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        first = True
        while not self._stop:
            if not first and self.resolve is not None:
                try:
                    self.source = self.resolve()
                except Exception:
                    pass  # the old address, then; the caller's stall watch decides when a stream has ended
            first = False
            cap = open_capture(self.source)
            if not cap.isOpened():
                time.sleep(2.0)
                continue
            count = 0
            rate = cap.get(cv2.CAP_PROP_FPS)
            stride = 1 if 0 < rate < SLOW_SOURCE_FPS else self.stride
            # Anything that isn't a camera (a video URL or file) is played at
            # its own declared frame rate -- a recording decoded as fast as
            # possible would make the body's real-clock time fly; a live stream
            # can't outrun real time anyway, so pacing only smooths its bursts
            # -- and when it ends, it has ended. (Frame counts can't tell a
            # recording from a stream: YouTube's report garbage.) Cameras
            # arrive in real time and reconnect if they drop.
            recording = not _is_live_source(self.source)
            period = 1.0 / rate if recording and 0 < rate < 1000 else 0.0
            due = time.time()
            try:
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        if recording:
                            self.ended = True
                            return
                        break
                    if period:
                        due += period
                        wait = due - time.time()
                        if wait > 0:
                            time.sleep(wait)
                        elif wait < -1.0:
                            due = time.time()  # fell behind (a stall): carry on from now
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
            if isinstance(self.frames_dir, FrameRing):
                if ok:
                    self.frames_dir.put(self.epoch, index, jpg.tobytes())
                return
            if ok:
                tmp = self.frames_dir / "tmp.jpg"
                tmp.write_bytes(jpg.tobytes())
                os.replace(tmp, self.frames_dir / f"f{self.epoch}_{index}.jpg")
            (self.frames_dir / f"f{self.epoch}_{index - FRAME_RING}.jpg").unlink(missing_ok=True)
            if index % 600 == 0:
                self._drop_old_runs()
        except (OSError, cv2.error):
            pass

    def _drop_old_runs(self, older_than_s: float = 300.0) -> None:
        """Earlier runs' frames: kept a while (the viewer plays them until this
        run reports in), then removed."""
        now = time.time()
        for p in self.frames_dir.glob("f*.jpg"):
            try:
                if not p.name.startswith(f"f{self.epoch}_") and now - p.stat().st_mtime > older_than_s:
                    p.unlink()
            except OSError:
                pass

    def longest_gap(self) -> float:
        """The longest wait between kept frames in the current window: the
        source's own cadence, bursts included (a live stream arrives in
        segments of a few seconds)."""
        with self._lock:
            t = list(self._times)
        return max((b - a for a, b in zip(t, t[1:])), default=0.0)

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

    def since(self, index: int | None) -> list:
        """Frames kept after the feed's frame `index` (None: just the newest),
        oldest first, as (index, grey, prey boxes, colour frame, arrival time):
        what a live organism hasn't lived yet (fishbowl/livelife.py)."""
        with self._lock:
            n = len(self._buf)
            first = self.total - n
            start = max(0, n - 1) if index is None else max(0, index + 1 - first)
            if start >= n:
                return []
            items, times = list(self._buf)[start:], list(self._times)[start:]
        return [(first + start + k, it[0], it[2], it[3], t) for k, (it, t) in enumerate(zip(items, times))]

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
    cap = open_capture(source)
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
