#!/usr/bin/env python3
from __future__ import annotations

"""
A tiny, dependency-free local viewer -- watching noise that, over
millions of iterations, is meant to become vision. Serves ONE page that
polls state/live_status.json and renders two blocky, pixelated
canvases from real backend-computed retina.py reductions -- genuinely
what it's seeing, not a reconstruction, since a 144-value luminance
grid is already reduced far past anything resembling real footage (see
run_vision.py's own comment on why including it there doesn't touch
the no-raw-frames rule): the RETINA (the fovea's own cropped view) and
the WORLD RETINA (the same reduction run on the full frame, i.e. what
run_vision.py's fitness grading itself sees), with the FOVEA's own box
(cx/cy/fraction, fovea.py's real pan/tilt window) drawn on top of the
world retina to show where it's currently pointed. This replaced an
earlier YouTube-embed crop-preview that kept hitting real, unfixable
constraints (embedding restrictions, cross-origin pixel access, URL
format parsing).

Deliberately NOT built into the HLSLS stack -- this has to work
whether or not broadcast-api/mediamtx are up (see the coordination
with HLSLS's own agent on that). Stdlib only, no new dependency.

Usage:
    python3 tools/viewer.py [--port 8090]
    Then open http://<host>:<port>/ in a browser.
"""

import argparse
import html
import json
import os
import socket
import subprocess
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import urllib.request


# The last reset of each kind this viewer passed on (time.time()): a second
# one within RESET_REPEAT_S -- another tab, another person, a double
# submission -- would wipe the fresh founder or brain the first just made
# (Ariana, 2026-09-30: two Amnesias 2.5 min apart, the first founder lived
# 99 s). Refused, with when the first was asked.
_LAST_RESET: dict = {}
RESET_REPEAT_S = 120.0


def _reset_refused(what: str) -> str | None:
    t = _LAST_RESET.get(what)
    if t is not None and time.time() - t < RESET_REPEAT_S:
        return f"already asked at {time.strftime('%H:%M:%S', time.localtime(t))} -- a fresh one wasn't wiped by a second"
    _LAST_RESET[what] = time.time()
    return None


_PEERS: dict = {}  # the hive's ecohosts the page last listed: name -> (address, port); Copy from only asks these


class QuietHTTPServer(ThreadingHTTPServer):
    """A browser closing a tab mid-response (a broken pipe, a reset, an aborted
    connection) is not an error worth a traceback in the logs."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def _read_shared(path) -> bytes | None:
    """A file the organism replaces several times a second: on Windows, a read
    that lands on the replace is refused for an instant -- try again briefly,
    then give up (the page asks again within a second)."""
    for _ in range(10):
        try:
            return path.read_bytes() if path.exists() else None
        except PermissionError:
            time.sleep(0.02)
    return None


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# LIVE_SOURCES imported (not duplicated) from run_vision.py -- a
# second copy here would drift out of sync with the real list. Safe
# to import: run_vision.py only runs main() under __main__.
from run_vision import LIVE_SOURCES  # noqa: E402

# Which organism on this host this viewer shows. PARKED feature (a second
# organism per host, docs/second-organism.md): unset -- as on every host
# today -- it is the one organism, exactly as before. Set to "b" (by the
# parked units in deploy/second-organism/), every path below is that
# organism's own; the derivations MUST match fishbowl/sandbox.py's.
_INSTANCE = os.environ.get("CAMBRIAN_INSTANCE", "").strip()
STATE_DIR = Path(__file__).resolve().parent.parent / ("state" if not _INSTANCE else f"state-{_INSTANCE}")
ORGANISM_UNIT = "cambrian-perception.service" if not _INSTANCE else f"cambrian-perception-{_INSTANCE}.service"
DEFAULT_PORT = 8090 if not _INSTANCE else 8091  # the second organism's viewer (only "b" is supported)
# Same place run_vision writes it: RAM (/dev/shm) where available -- per
# organism (without the suffix, a second viewer showed the FIRST organism's
# live status: found and fixed while parking the feature, 2026-09-30).
_RUNTIME_DIR = Path(os.environ.get("CAMBRIAN_RUNTIME_DIR", "/dev/shm/cambrian-perception" + (f"-{_INSTANCE}" if _INSTANCE else "")))
LIVE_STATUS_PATH = (_RUNTIME_DIR if _RUNTIME_DIR.parent.is_dir() else STATE_DIR) / "live_status.json"
# Its recent frames, small JPEGs run_vision.py keeps next to it in RAM (a
# short ring, see video_source.LiveFeed), for replaying its latest run.
FRAMES_DIR = LIVE_STATUS_PATH.with_name("frames")
# Where there's no RAM dir (Windows, macOS) the organism holds them in its
# own memory instead and serves them on localhost at the port named here.
FRAMES_PORT_PATH = LIVE_STATUS_PATH.with_name("frames.json")


def _livecam_installed() -> bool:
    """The camera suite (docs/suite.md): is the livecam on this host too?
    Then it is off whenever this page has anything to show."""
    if sys.platform.startswith("linux"):
        return any(Path(d, "hls-livecam.target").exists()
                   for d in ("/etc/systemd/system", "/usr/lib/systemd/system", "/lib/systemd/system"))
    try:
        from tools import suite
        return suite.livecam_command() is not None
    except Exception:
        return False


SUITE_CHIP = ('<span class="chip" title="The camera suite: the livecam and the organism never run together -- '
              'starting the livecam stops the organism">livecam <b>off while it runs</b></span>\n  ')


def _frame_from_organism(e: str, i: str) -> bytes | None:
    from urllib.request import urlopen
    try:
        port = int(json.loads(FRAMES_PORT_PATH.read_text(encoding="utf-8-sig"))["port"])
        with urlopen(f"http://127.0.0.1:{port}/frame?e={e}&i={i}", timeout=2) as r:
            return r.read()
    except (OSError, ValueError, KeyError):
        return None
EVOLUTION_LOG_PREV_PATH = STATE_DIR / "evolution_log.1.jsonl"
# Paths defined independently here, not imported from fishbowl.sandbox
# -- same deliberate independence as everything else in this module
# (see its own module docstring: this has to work standalone).
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
HISTORY_MAX_LINES = 5000
HISTORY_MAX_BYTES = 12_000_000  # real tail, not a full-file read -- the
# log can grow to ~20000 lines/rotation; this stays cheap regardless.

# Human-only control -- this is the first thing this viewer ever
# WRITES (everything else is read-only). Written here, read by
# sandbox.load_selected_source() in run_vision.py. Only ever a name
# from LIVE_SOURCES, or a verified-live https YouTube link -- never an arbitrary URL,
# since this viewer has no auth (a known, documented gap) and
# accepting free-text URLs here would hand anyone on the LAN control
# over what real content the organism trains against.
SELECTED_SOURCE_PATH = STATE_DIR / "selected_source.json"


def _held(fn, tries: int = 30):
    """fn(), retried while Windows holds the file (WinError 32: another
    process -- the organism reading it, a virus scanner on a file just
    written -- has it open; it can't be deleted or replaced until it lets
    go, a few ms to a second later). Raises the last error if it never does."""
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.05 * (1 + k))


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    _held(lambda: temp.replace(path))


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_URL_CHECK = threading.BoundedSemaphore(1)  # one yt-dlp check at a time


def _allowed_url(url: str) -> bool:
    """Only https YouTube links -- the viewer is reachable on the LAN, and
    yt-dlp would otherwise fetch anything it understands (security audit:
    SSRF, arbitrary stream decoding)."""
    try:
        u = urlparse(url)
    except ValueError:
        return False
    return u.scheme == "https" and (u.hostname or "").lower() in YOUTUBE_HOSTS and not u.username and not u.password


# yt-dlp's own give-up time with its default settings (a 20 s socket
# timeout, 10 retries): the check waits as long as yt-dlp itself would.
YTDLP_GIVE_UP_S = 20 * (10 + 1)


def _check_live_url(url: str) -> tuple[bool, str | None]:
    """
    Real yt-dlp metadata check, not a URL-shape guess: the link must resolve
    to a playable YouTube video, live or recorded (a recording plays at its
    own pace, and it goes back to its camera when the video ends).
    """
    if not _allowed_url(url):
        return False, "only https youtube.com / youtu.be links are accepted"
    if not _URL_CHECK.acquire(blocking=False):
        return False, "another check is already running -- try again in a moment"
    yt_dlp = Path(sys.executable).parent / ("yt-dlp.exe" if os.name == "nt" else "yt-dlp")
    if not yt_dlp.exists():
        yt_dlp = Path("yt-dlp")
    try:
        # "--" so the URL can never be read as a yt-dlp option.
        result = subprocess.run(
            [str(yt_dlp), "--skip-download", "--no-warnings", "--print", "is_live", "--", url],
            capture_output=True, text=True, timeout=YTDLP_GIVE_UP_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "yt-dlp could not reach YouTube (it gave up) -- try again"
    finally:
        _URL_CHECK.release()
    if result.returncode != 0:
        return False, "not a playable YouTube video"
    return True, None


def _tail_jsonl(path: Path, max_lines: int = HISTORY_MAX_LINES, max_bytes: int = HISTORY_MAX_BYTES) -> list[dict]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # drop the partial first line from the seek
        data = f.read()
    records = []
    for line in data.decode("utf-8", errors="ignore").splitlines()[-max_lines:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records

HISTORY_POINTS = 600


def _history_summary(records: list[dict]) -> dict:
    """
    Up to HISTORY_POINTS chart points from a long tail of the log: each
    point is the last record in its bucket (its current state), plus
    every mutation kind that was ACCEPTED anywhere in the bucket and the
    largest accepted gain -- so downsampling never hides an acceptance.
    """
    n = len(records)
    if not n:
        return {"records": [], "span_generations": 0}
    size = max(1, -(-n // HISTORY_POINTS))
    out = []
    for i in range(0, n, size):
        bucket = records[i:i + size]
        r = bucket[-1]
        bd = r.get("breakdown") or {}
        ts = (r.get("tree_stats") or {}).get("response") or {}
        acc = [b for b in bucket if b.get("accepted")]
        gains = [b["fitness_delta"] for b in acc if b.get("fitness_delta") is not None]
        out.append({
            "best_fitness": r.get("best_fitness"),
            "peak_fitness_seen": r.get("peak_fitness_seen"),
            "mean_energy": bd.get("mean_energy"),
            "mean_food": bd.get("mean_food"),
            "mean_prey": bd.get("mean_prey"),
            "mean_drive": bd.get("mean_drive"),
            "mean_aperture": bd.get("mean_aperture"),
            "mv": bd.get("movement"),
            "fovea_fraction": r.get("fovea_fraction"),
            "quota_pct": r.get("quota_pct"),
            "pace": r.get("pace"),
            "tree_nodes": ts.get("nodes"),
            "tree_depth": ts.get("depth"),
            "accepted_types": sorted({b.get("mutation_type") for b in acc}),
            "accepted_delta": max(gains) if gains else None,
        })
    return {"records": out, "span_generations": n}


# Raw string: the page is served byte-for-byte, no Python escape
# processing (an unescaped apostrophe in the JS broke the page once).
# The target-lock HUD, shared by the viewer's live panel and the client page
# (/live): its gaze drawn as a fighter jet's target lock, with the ID of what
# it is on. Self-contained on purpose -- it is the piece a livecam server's
# CV module takes over (where LOCK, "eating", becomes "take a snapshot").
# The tab's name: this host, then CP (cambrian-perception) -- several hosts' tabs side by side
HOST_NAME = html.escape(socket.gethostname().split(".")[0] or "cambrian")

LOCK_HUD_JS = r"""
  // ---- One clock for the visual field, the real picture and the target lock ----
  // Its gaze runs over a snapshot of the newest frames, refreshed every few
  // generations, so the frames it has gazed at trail the live camera by a
  // few seconds. The viewer plays them forward at real speed with a delay
  // just long enough that every frame shown has been gazed at: a continuous
  // delayed feed, not a loop. The picture is the real frame it saw at that
  // moment (run_vision keeps a short ring of them in RAM), the lock is where
  // its gaze was on it -- picture, gaze and prey marks describe one instant.
  // clk keeps the clock between frames ({ t0 } to start). A source without
  // frame indices (a file) falls back to looping its latest run.
  function replayAt(d, now, clk, defaultFps) {
    const traj = d.trajectory && d.trajectory.length ? d.trajectory : [[d.fovea_cx ?? 0.5, d.fovea_cy ?? 0.5, d.fovea_fraction || 0.35, 0]];
    const fps = d.frames_per_second || defaultFps || 15;
    const lastIdx = traj[traj.length - 1][3] ?? (traj.length - 1);
    let cur, delay = null, catching = false;
    if (d.world_first_index != null && d.world_age_s != null) {
      const key = d.world_epoch + ':' + d.world_first_index + ':' + d.world_age_s + ':' + d.generation;
      // A new run (the organism restarted, e.g. switching source) starts the
      // clock afresh: the minute it took to report in is not staleness.
      if (clk.epoch !== d.world_epoch) { clk.epoch = d.world_epoch; clk.first = null; clk.need = null; clk.lastT = null; clk.stale = []; }
      if (clk.key !== key) {
        // A new snapshot: how stale its newest gazed-at frame got before it
        // came is the delay that never stalls.
        if (clk.first != null && d.world_first_index !== clk.first) {
          (clk.stale = clk.stale || []).push([now, clk.age + (now - clk.recvT) / 1000 + 0.5]);
        }
        clk.key = key; clk.recvT = now; clk.age = d.world_age_s; clk.first = d.world_first_index;
      }
      const age = clk.age + (now - clk.recvT) / 1000;  // of the newest frame it has gazed at
      const dt = clk.lastT != null ? Math.min(1, Math.max(0, (now - clk.lastT) / 1000)) : 0; clk.lastT = now;
      if (clk.need == null) clk.need = age + 3.0;
      // The floor: the stalest a snapshot got in the last 30 s -- the shortest
      // delay at which every frame shown has been gazed at. One slow
      // generation raises it only for as long as it is remembered.
      clk.stale = (clk.stale || []).filter(([t]) => now - t < 30000);
      clk.target = Math.max(1.0, ...clk.stale.map(([, s]) => s), clk.stale.length ? 0 : age + 0.5);
      // Below the floor: slow to half speed (never jump back). Above it: jump
      // straight to the floor -- the present, as its body lives it (stale
      // frames are skipped, never caught up on; a 2026-09-29 panel).
      if (clk.need < clk.target) clk.need = Math.min(clk.target, clk.need + dt * 0.5);
      else clk.need = clk.target;
      catching = clk.need > clk.target + 0.2;
      delay = clk.need;
      const newest = d.world_first_index + lastIdx;
      const g = Math.floor(newest + (age - delay) * fps);
      cur = Math.max(0, Math.min(lastIdx, g - d.world_first_index));
    } else {
      cur = Math.floor(Math.max(0, now - clk.t0) / 1000 * fps) % (lastIdx + 1);
    }
    return frameAt(d, { traj, fps, lastIdx, delay, catching }, cur);
  }
  // The replay's state at one frame (cur: frames after world_first_index).
  function frameAt(d, base, cur) {
    const { traj, fps, lastIdx, delay, catching } = base;
    let i = 0; while (i + 1 < traj.length && (traj[i + 1][3] ?? (i + 1)) <= cur) i++;
    const at = a => a && a.length ? a[Math.min(cur, a.length - 1)] : null;
    return {
      traj, fps, lastIdx, cur, i, cx: traj[i][0], cy: traj[i][1], f: traj[i][2] || 0.35,
      // the gaze is square in pixels: f of the frame's height, fw of its width
      fw: (traj[i][2] || 0.35) * (d.frame_h || 9) / (d.frame_w || 16),
      eat: at(d.eating) || 0, boxes: at(d.prey_boxes) || [],
      guess: at(d.tree_guess) ?? null, label: at(d.teacher_label) ?? null, snack: at(d.snacks) || 0,
      asleep: !!at(d.asleep_frames), warn: !!at(d.alarm_frames),
      contact: at(d.contact_frames) || 0, ahead: at(d.ahead_boxes) || null,
      frame: d.world_first_index != null ? d.world_first_index + cur : null, epoch: d.world_epoch || 0,
      delay, catching,
    };
  }
  // The HUD belongs to the picture under it. When the picture lags (a slow or
  // missing frame), the HUD is drawn for the frame actually on screen -- never
  // a live HUD over a stale picture; null = that frame has left the record
  // (draw no HUD until the picture catches up).
  function hudFor(d, R, F) {
    if (R.frame == null || F.shown == null) return R;
    const [e, g] = String(F.shown).split(':').map(Number);
    if (e !== R.epoch) return null;
    if (g === R.frame) return R;
    const off = g - d.world_first_index;
    if (off < 0 || off > R.lastIdx) return null;
    return frameAt(d, R, off);
  }
  // Keeps an <img> on the wanted frame of the replay. Each frame is loaded
  // off-screen first and only shown once it has loaded, so a missing frame
  // never replaces the picture (it just keeps the last good one). One request
  // at a time; frames are immutable per (run, index), so the browser caches them.
  function frameLoader(img) {
    const L = { shown: null, want: null, busy: false, aspect: null, failed: false };
    const pre = new Image();
    pre.addEventListener('load', () => {
      L.busy = false; L.failed = false; L.shown = L.want;
      img.src = pre.src; img.style.visibility = 'visible';
      if (pre.naturalWidth) L.aspect = pre.naturalWidth / pre.naturalHeight;
    });
    pre.addEventListener('error', () => { L.busy = false; L.failed = true; });
    L.show = (g, epoch) => {
      const key = g == null ? null : epoch + ':' + g;
      if (key == null || key === L.shown || L.busy) return;
      L.busy = true; L.want = key; pre.src = '/frame?i=' + g + '&e=' + epoch;
    };
    return L;
  }

  // ---- Target lock: its gaze on the picture ----
  // The reticle is its gaze -- the box it sees detail through, with the
  // diamond marking its centre -- its mouth: prey under it is a meal.
  //   SCAN  nothing it hunts is in its gaze
  //   TRACK prey (person/animal) somewhere in its gaze
  //   LOCK  prey under its gaze centre (its mouth): a bite -- in a livecam, the
  //         moment to take a snapshot
  //   SLEEP asleep: eyes shut, the gaze sees nothing and is parked. The
  //         picture stays -- the world doesn't go dark when it sleeps, and
  //         the whole field still reaches it, as light through closed eyes.
  // ID = the prey nearest its gaze center, per the detector; pink corner
  // marks = all prey the detector found in that frame.
  function lockCorners(ctx, X0, Y0, X1, Y1, L) {
    ctx.beginPath();
    [[X0, Y0, 1, 1], [X1, Y0, -1, 1], [X0, Y1, 1, -1], [X1, Y1, -1, -1]].forEach(([x, y, sx, sy]) => { ctx.moveTo(x + sx * L, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * L); });
    ctx.stroke();
  }
  function lockState(fs) {
    if (fs.asleep) return { mode: 'SLEEP', id: null };
    const cx = fs.cx, cy = fs.cy, hx = (fs.fw ?? fs.f) / 2, hy = fs.f / 2;
    const overlap = (b, k) => Math.max(0, Math.min(b[4], cx + hx * k) - Math.max(b[2], cx - hx * k)) * Math.max(0, Math.min(b[5], cy + hy * k) - Math.max(b[3], cy - hy * k));
    let id = null, best = 0, inGaze = false;
    (fs.boxes || []).forEach(b => {
      const whole = overlap(b, 1), score = 4 * overlap(b, 0.5) + whole;
      if (whole > 0) inGaze = true;
      if (score > best) { best = score; id = b; }
    });
    return { mode: fs.eat > 0.01 ? 'LOCK' : inGaze ? 'TRACK' : 'SCAN', id };
  }
  // The readout in Arabic (the HUD's right side): modern Lebanese words,
  // Arabic-Indic digits, and the unit written out.
  const AR_FONT = '"Noto Naskh Arabic", "Geeza Pro", "Segoe UI", Tahoma, sans-serif';
  const AR_WORDS = [['catching up:', 'عم يلحّق:'], ['its latest run, looped', 'آخر دورة، عم تعيد'], ['delayed', 'متأخّر'],
    ['calibrating', 'عم يظبّط'], ['snacks', 'لقمات'], ['locks', 'مسكات'],
    ['SCAN', 'عم دوّر'], ['TRACK', 'لاحقو'], ['LOCK', 'مسكتو'], ['SLEEP', 'نايم'], ['TARGET', 'هدف'], ['WARN', 'دير بالك'],
    ['looks', 'نظرات']];
  function toArabic(text) {
    let t = String(text);
    AR_WORDS.forEach(([en, ar]) => { t = t.split(en).join(ar); });
    return t.replace(/ s(?= |$)/g, ' ثانية').replace(/(\d)\.(\d)/g, '$1٫$2').replace(/%/g, '٪')
            .replace(/[0-9]/g, c => '٠١٢٣٤٥٦٧٨٩'[c]);
  }
  // The readout in Ukrainian (bottom left) and in Traditional Chinese as
  // written in Taiwan (bottom right): the same rows, each language's own
  // decimal mark and unit.
  const UK_WORDS = [['catching up:', 'наздоганяю:'], ['its latest run, looped', 'останній прогін, по колу'], ['delayed', 'затримка'],
    ['calibrating', 'калібрування'], ['snacks', 'перекуси'], ['locks', 'захоплення'],
    ['SCAN', 'ПОШУК'], ['TRACK', 'СТЕЖУ'], ['LOCK', 'ЗАХОПЛЕНО'], ['SLEEP', 'СПЛЮ'], ['TARGET', 'ЦІЛЬ'], ['WARN', 'УВАГА'],
    ['looks', 'погляди']];
  const TW_WORDS = [['catching up:', '追趕中：'], ['its latest run, looped', '最近一輪，循環播放'], ['delayed', '延遲'],
    ['calibrating', '校準中'], ['snacks', '神'], ['locks', '鎖定次數'],
    ['SCAN', '搜尋'], ['TRACK', '追蹤'], ['LOCK', '鎖定'], ['SLEEP', '睡眠'], ['TARGET', '目標'], ['WARN', '警告'],
    ['looks', '次注視']];
  const TW_FONT = '"Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", sans-serif';
  function translate(text, words) { let t = String(text); words.forEach(([en, x]) => { t = t.split(en).join(x); }); return t; }
  function toUkrainian(text) { return translate(text, UK_WORDS).replace(/([0-9])\.([0-9])/g, '$1,$2').replace(/ s(?= |$)/g, ' с'); }
  function toTaiwanese(text) { return translate(text, TW_WORDS).replace(/ s(?= |$)/g, ' 秒'); }
  // The translated readouts are page text laid over the HUD, not canvas text:
  // browsers' canvases disagree on right-to-left alignment and mis-measure
  // shaped Arabic (one clipped the first words, another fell short of the
  // margin), while the page's own text engine lays out bidirectional text and
  // justifies it the same everywhere. Arabic top right (right-to-left,
  // right-justified), Ukrainian bottom left, Taiwanese bottom right.
  const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  function hudOverlay(canvas, rows) {
    const host = canvas && canvas.parentElement; if (!host) return;
    if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
    host._hudT = performance.now();
    const hk = canvas._hudK || 1, m = Math.round(14 * hk), side = Math.round(24 * hk), lh = Math.round(16 * hk);
    const spots = { ar: ['rtl', `top:${m}px;right:${side}px;text-align:right`, AR_FONT, toArabic],
                    uk: ['ltr', `bottom:${m}px;left:${side}px;text-align:left`, 'monospace', toUkrainian],
                    tw: ['ltr', `bottom:${m}px;right:${side}px;text-align:right`, TW_FONT, toTaiwanese] };
    rows = rows.filter(r => r[3] !== 'telemetry');  // already at the corners' size (drawLock)
    Object.entries(spots).forEach(([key, [dir, where, family, conv]]) => {
      let el = host.querySelector(':scope > .hud-' + key);
      if (!el) {
        el = document.createElement('div'); el.className = 'hud-' + key + ' hud-lang'; el.dir = dir;
        host.appendChild(el);
      }
      el.style.cssText = 'position:absolute;pointer-events:none;white-space:pre;opacity:0.72;'
        + `text-shadow:1px 1px 3px #000,1px 1px 1px #000,0 0 6px ${HUD_GLOW};` + where;
      el.style.display = '';
      // every row, blank ones too (WARN blinking off), so each line stays level with the English
      // the bottom blocks run upward (mirror images of the top ones): first row at the bottom
      const html = (key === 'ar' ? rows : rows.slice().reverse()).map(([t, colour, font]) =>
        `<div style="color:${colour};font:${font.replace('monospace', family)};line-height:${lh}px;height:${lh}px">${t ? esc(conv(t)) : '&nbsp;'}</div>`).join('');
      if (el._html !== html) { el._html = html; el.innerHTML = html; }
    });
  }
  // A HUD switched off (or not drawn) takes its page text with it.
  setInterval(() => document.querySelectorAll('.hud-lang').forEach(el => {
    if (!el.parentElement._hudT || performance.now() - el.parentElement._hudT > 500) el.style.display = 'none';
  }), 250);
  function lockIdText(id) { return id ? ` TARGET ${(id[1] * 100).toFixed(0)}%` : ''; }  // every host is a target
  // Draws the HUD on a bw x bh box. fs = the replay's current frame
  // (replayAt), d = status, picture = a frame is shown (the reticle is drawn
  // only then), imgAspect = the picture's width / height (fitted inside the
  // box), st = glide state kept between frames, opts.gen = show generation.
  // The HUD's palette, "red phosphor" (a 2026-09-28 UX panel, 8-2): one red
  // ramp, meaning carried by brightness and weight (it survives without
  // colour), bone-white only for what matters now -- a lock, a bite, a warning.
  // Phosphor, as a CRT's: each glyph a pale core blooming red (its glow is the
  // red); brightness is the hierarchy -- white-hot for what matters now, then
  // pale red, then a dimmer red; a drop shadow under everything keeps it
  // legible over any picture.
  const HUD_BONE = '#fff4f0', HUD_HOT = '#ffc2b8', HUD_EMBER = '#ff8a78', HUD_ASH = '#b8584a';
  // Each state keeps its own colour, as before the palette changed -- only the
  // hues moved into the palette: scan, track, lock, sleep; agree / disagree;
  // catching up; snacks and meals; a lit warning.
  const HUD_SCANNING = '#ff8a78', HUD_TRACK = '#ffb066', HUD_LOCK = '#ff4f6f', HUD_SLEEP = '#c98aa8';
  const HUD_AGREE = '#fff0c0', HUD_DISAGREE = '#ffb066', HUD_CATCH = '#ffb066';
  const HUD_SNACK = '#e0909c', HUD_MEAL = '#ff6f8a', HUD_WARN = '#ff3b28', HUD_HOST = '#ff6f8a';
  // Roles (a 2026-09-28 critique, taken up by the UX panel): red/magenta = target
  // and acquisition; amber = behaviour; cool steel = system telemetry; grey =
  // labels. Three tiers of text: the state, then the delay, then small telemetry.
  const HUD_BEHAV = '#d9a066', HUD_SYS = '#a9bcc8', HUD_LABEL = '#8c8680';
  const HUD_MOUTH = 22 / 64 / 2;  // its mouth, of the frame's height (prey.MOUTH_SIDE)
  const HUD_GLOW = 'rgba(255, 40, 20, 0.95)';
  const HUD_FILM = 'rgba(80, 0, 0, 0.14)', HUD_SCAN = 'rgba(0, 0, 0, 0.08)';
  function drawLock(ctx, bw, bh, fs, d, picture, imgAspect, now, st, opts) {
    // Red phosphor (a 2026-09-28 UX panel): the picture under a thin red film
    // and faint scanlines, so it reads as seen by a machine.
    ctx.fillStyle = HUD_FILM; ctx.fillRect(0, 0, bw, bh);
    ctx.fillStyle = HUD_SCAN; for (let y = 0; y < bh; y += 3) ctx.fillRect(0, y, bw, 1);
    const dt = Math.min(1, (now - (st.lastT || now)) / 1000); st.lastT = now;
    const L = lockState(fs), aspect = bw / bh;
    const sleeping = L.mode === 'SLEEP';
    const col = L.mode === 'LOCK' ? HUD_LOCK : L.mode === 'TRACK' ? HUD_TRACK : sleeping ? HUD_SLEEP : HUD_SCANNING;
    const blink = Math.floor(now / 350) % 2 === 0;
    if (picture) {
      const ia = imgAspect || aspect;
      const w = ia >= aspect ? bw : bh * ia, h = ia >= aspect ? bw / ia : bh;
      ctx.save(); ctx.translate((bw - w) / 2, (bh - h) / 2);
      ctx.strokeStyle = HUD_HOST; ctx.lineWidth = 1.5;
      (fs.boxes || []).forEach(b => lockCorners(ctx, b[2] * w, b[3] * h, b[4] * w, b[5] * h, 8));
      if (fs.ahead) {  // where it expects the host it follows (its extrapolation): a lighter, dashed ghost ahead of it
        ctx.save(); ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55; ctx.lineWidth = Math.max(1, ctx.lineWidth * 0.6);
        lockCorners(ctx, fs.ahead[0] * w, fs.ahead[1] * h, fs.ahead[2] * w, fs.ahead[3] * h, 6); ctx.restore();
      }
      // glide between its gazes (the eye moves between them too); snap when the replay loops
      if (st.i != null && fs.i < st.i) st.rx = null;
      st.i = fs.i;
      const k = 1 - Math.pow(0.0005, dt);
      st.rx = st.rx == null ? fs.cx : st.rx + (fs.cx - st.rx) * k;
      st.ry = st.ry == null ? fs.cy : st.ry + (fs.cy - st.ry) * k;
      st.rs = st.rs == null ? fs.f : st.rs + (fs.f - st.rs) * k;
      const x = st.rx * w, y = st.ry * h, gw = st.rs * h, gh = st.rs * h;  // square: its side is a fraction of the height
      // Asleep, its eyes are shut: short, thin brackets and no centre (it can't eat).
      ctx.strokeStyle = col; ctx.lineWidth = sleeping ? 1 : 2;
      lockCorners(ctx, x - gw / 2, y - gh / 2, x + gw / 2, y + gh / 2, Math.min(gw, gh) * (sleeping ? 0.08 : 0.16));
      // Client view only (opts.snap): when it locks on -- starts eating, where a
      // livecam would take its snapshot -- the brackets snap in from the edges of
      // the picture onto its gaze (with a soft shutter flash, below). At most
      // once a second, so a flickering lock doesn't strobe.
      if (opts && opts.snap) {
        if (L.mode === 'LOCK' && st.mode !== 'LOCK' && now - (st.snapT ?? -1e9) > 1000) st.snapT = now;
        const age = now - (st.snapT ?? -1e9);
        if (age < 280) {
          const k = age / 280, e = 1 - Math.pow(1 - k, 3);
          const X0 = (x - gw / 2) * e, Y0 = (y - gh / 2) * e, X1 = w + (x + gw / 2 - w) * e, Y1 = h + (y + gh / 2 - h) * e;
          ctx.save(); ctx.strokeStyle = `rgba(255, 79, 111, ${0.9 * (1 - k)})`; ctx.lineWidth = 3;
          lockCorners(ctx, X0, Y0, X1, Y1, Math.min(X1 - X0, Y1 - Y0) * 0.14);
          ctx.restore();
        }
      }
      // Its mouth, the one real feature at the centre: its true square at its
      // true size (prey under it is a meal); filled, pulsing, while it bites.
      if (!sleeping) {
        const m = HUD_MOUTH * h;
        if (L.mode === 'LOCK') { ctx.fillStyle = `rgba(255, 79, 111, ${0.18 + 0.14 * (0.5 + 0.5 * Math.sin(now / 110))})`; ctx.fillRect(x - m / 2, y - m / 2, m, m); }
        ctx.lineWidth = 1.5; ctx.strokeRect(x - m / 2, y - m / 2, m, m);
        // where its eye is moving: a short vector from the mouth (its real eye motion between looks)
        const tr = fs.traj || [], j = fs.i || 0;
        if (tr.length > 1 && j > 0) {
          const vx = (tr[j][0] - tr[j - 1][0]) * w, vy = (tr[j][1] - tr[j - 1][1]) * h, v = Math.hypot(vx, vy);
          if (v > 1) {
            const len = Math.min(gw * 0.6, 3 * v), ux = vx / v, uy = vy / v;
            ctx.strokeStyle = HUD_BEHAV; ctx.lineWidth = 1.5; ctx.beginPath();
            ctx.moveTo(x + ux * m / 2, y + uy * m / 2); ctx.lineTo(x + ux * (m / 2 + len), y + uy * (m / 2 + len)); ctx.stroke();
          }
        }
      }
      ctx.restore();
    }
    if (opts && opts.snap) {  // the shutter flash (client view only)
      const age = now - (st.snapT ?? -1e9);
      if (age < 180) { ctx.fillStyle = `rgba(255, 255, 255, ${0.16 * (1 - age / 180)})`; ctx.fillRect(0, 0, bw, bh); }
    }
    st.mode = L.mode;
    const b = d.body_now || d.body || {}, threat = Math.max(0, Math.min(1, b.threat || 0));
    if (threat > 0.05) { ctx.strokeStyle = `rgba(255, 42, 26, ${0.85 * threat})`; ctx.lineWidth = 8; ctx.strokeRect(4, 4, bw - 8, bh - 8); }
    // Meals and snacks it had, counted as the feed plays (since this page
    // opened): a feeding act is a look that caught prey (in a livecam, a
    // snapshot) or any surprise; acts closer together than its bout
    // criterion -- measured from its own feeding gaps (d.bouts, run_vision /
    // fishbowl/bouts.py) -- are one meal / snack. No criterion yet: calibrating.
    const gazeFrame = fs.traj && fs.traj[fs.i] ? (d.world_first_index ?? 0) + (fs.traj[fs.i][3] ?? fs.i) : null;
    if (gazeFrame != null && (st.lastGaze == null || gazeFrame > st.lastGaze)) {
      st.lastGaze = gazeFrame;
      const fps = fs.fps || 15, bouts = d.bouts || {};
      [['meal', fs.eat > 0, 'meals', 'lastEat'], ['snack', (fs.snack || 0) > 0, 'snacks', 'lastSnack']].forEach(([kind, act, count, last]) => {
        const fit = bouts[kind] && bouts[kind].fit;
        if (!act || !fit) return;
        if (st[last] == null || gazeFrame - st[last] > fit.criterion_s * fps) st[count] = (st[count] || 0) + 1;
        st[last] = gazeFrame;
      });
    }
    // its text scales with the picture: full size from 560 px wide, down to 60% on a phone
    const hk = Math.max(0.6, Math.min(1, bw / 560)), hpx = n => Math.max(8, Math.round(n * hk)), lh = 16 * hk, y0 = 18 * hk;
    ctx.canvas._hudK = hk;
    ctx.font = `${hpx(11)}px monospace`; ctx.textBaseline = 'middle';
    // LIVE: the organism acting live (fishbowl/livelife.py)
    const tag = (fs.delay != null ? (d.live_actor ? 'LIVE' : 'DELAYED') : 'REPLAY') + (opts && opts.gen ? `  gen ${d.generation !== undefined ? Number(d.generation).toLocaleString() : '--'}` : '');
    lockShadow(ctx, true);
    // The live indicator top centre; the readout top left, left-justified (so
    // changing numbers move away from the edge), mirrored in Arabic top right,
    // right-justified as Arabic reads.
    const tw = ctx.measureText(tag).width, tx = bw / 2 - (tw + 9) / 2;
    ctx.fillStyle = blink ? HUD_HOT : HUD_ASH; ctx.beginPath(); ctx.arc(tx + 4, y0, 4 * hk, 0, 7); ctx.fill();
    ctx.fillStyle = HUD_SYS; ctx.textAlign = 'left'; ctx.fillText(tag, tx + 13 * hk, y0);
    // Left-hand readout, top to bottom: mode + ID, delay, then its
    // warning. The same rows, size and look as the other three corners (their
    // translations).
    const rf = n => Math.max(7, Math.round(n * 0.85 * hk));
    const rows = [[L.mode + lockIdText(L.id), col, `bold ${rf(14)}px monospace`],
                  [fs.delay != null ? (fs.catching ? `catching up: ${fs.delay.toFixed(1)} s` : `delayed ${fs.delay.toFixed(1)} s`) : 'its latest run, looped', fs.catching ? HUD_CATCH : HUD_SYS, `${rf(11)}px monospace`]];
    // Its warning, last, for the owner only until the owner's feedback has trained
    // it (untrained, it drifts: one lineage warned in every waking frame): lit when
    // it warns, a dim lamp otherwise. The client page never shows it.
    if (opts && opts.internals) rows.push(fs.warn ? [blink ? 'WARN' : '', HUD_WARN, `bold ${rf(12)}px monospace`]
                                                  : ['WARN', HUD_ASH, `bold ${rf(12)}px monospace`]);
    ctx.textAlign = 'left';
    // each line twice: its drop shadow, then its phosphor bloom on top; as translucent as the other corners
    ctx.globalAlpha = 0.72;
    rows.forEach(([text, colour, font], k) => { ctx.font = font; ctx.fillStyle = colour; ctx.fillText(text, 16 * hk, y0 + lh * k); });
    ctx.save(); ctx.shadowColor = HUD_GLOW; ctx.shadowBlur = 6; ctx.shadowOffsetX = 0; ctx.shadowOffsetY = 0;
    rows.forEach(([text, colour, font], k) => { ctx.font = font; ctx.fillStyle = colour; ctx.fillText(text, 16 * hk, y0 + lh * k); });
    ctx.restore();
    ctx.globalAlpha = 1;
    hudOverlay(ctx.canvas, rows);  // Arabic, Ukrainian, Taiwanese: page text over the HUD
    lockShadow(ctx, false);
    ctx.textAlign = 'left';
  }
  // Text stands out by a soft drop shadow, not a dark box over the picture.
  function lockShadow(ctx, on) {  // a drop shadow, always, under every glyph and line
    ctx.shadowColor = on ? 'rgba(0, 0, 0, 0.95)' : 'transparent';
    ctx.shadowBlur = on ? 3 : 0; ctx.shadowOffsetX = on ? 1 : 0; ctx.shadowOffsetY = on ? 1 : 0;
  }
"""

# The client-facing page: only the picture and the target lock (its latest
# run, replayed). The viewer ("/") is the operators' view with its insides.
LIVE_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#0b0706">
<script>
// Uncaught errors, recorded on the page's root (invisible; for audits: a headless browser's DOM dump shows them)
addEventListener('error', e => { const r = document.documentElement; r.dataset.jsErrors = (+(r.dataset.jsErrors || 0) + 1) + ''; r.dataset.jsLast = String(e.message).slice(0, 200) + ' @' + (e.lineno || '?'); });
</script>

<title>__HOST__ | CP</title>
<style>
  html, body { margin: 0; height: 100%; background: #05080c; color: #f0d6cf; font-family: ui-monospace, Menlo, Consolas, monospace; overflow: hidden; }
  #wrap { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; }
  #box { position: relative; background: #000; }
  #box img, #box canvas { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; object-fit: contain; }
  #box canvas { pointer-events: none; }
  #fs { position: absolute; right: 8px; bottom: 8px; z-index: 6; width: 28px; height: 28px; padding: 5px; border-radius: 4px; cursor: pointer;
        background: rgba(5, 7, 10, 0.55); border: 1px solid #3a2a26; color: #ffe2d6; line-height: 0; }
</style></head>
<body>
<div id="wrap"><div id="box"><img id="cam" alt="" style="visibility:hidden"><canvas id="hud"></canvas><button id="fs" type="button" title="full screen" aria-label="full screen"><svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M1 5V1h4M11 1h4v4M15 11v4h-4M5 15H1v-4"/></svg></button></div></div>
<script>
// a click (or tap) takes the video full screen; another brings it back -- and so does its button
function toggleFull() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
}
document.addEventListener('click', toggleFull);
document.getElementById('fs').addEventListener('click', e => { e.stopPropagation(); toggleFull(); });
/*LOCK_HUD_JS*/
  const $ = id => document.getElementById(id);
  let D = null;
  const clk = { t0: performance.now() }, st = {}, F = frameLoader($('cam'));
  async function poll() { try { D = await (await fetch('/state')).json(); } catch (e) { } setTimeout(poll, 1000); }
  function frame(now) {
    requestAnimationFrame(frame);
    const d = D || {};
    const aspect = d.frame_w && d.frame_h ? d.frame_w / d.frame_h : (F.aspect || 16 / 9);
    const bw = Math.min(window.innerWidth, window.innerHeight * aspect), bh = bw / aspect;
    const box = $('box'); box.style.width = bw + 'px'; box.style.height = bh + 'px';
    const c = $('hud'), dpr = window.devicePixelRatio || 1, W = Math.round(bw * dpr), H = Math.round(bh * dpr);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, bw, bh);
    if (d.generation === undefined) return;
    const R0 = replayAt(d, now, clk);
    F.show(R0.frame, R0.epoch);
    const R = hudFor(d, R0, F);
    if (R) drawLock(ctx, bw, bh, R, d, F.shown != null, F.aspect, now, st, { gen: false, snap: true });
  }
  poll(); requestAnimationFrame(frame);
</script>
</body></html>
""".replace("/*LOCK_HUD_JS*/", LOCK_HUD_JS)

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#0b0706">
<script>
// Uncaught errors, recorded on the page's root (invisible; for audits: a headless browser's DOM dump shows them)
addEventListener('error', e => { const r = document.documentElement; r.dataset.jsErrors = (+(r.dataset.jsErrors || 0) + 1) + ''; r.dataset.jsLast = String(e.message).slice(0, 200) + ' @' + (e.lineno || '?'); });
</script>

<title>__HOST__ | CP</title>
<style>
  :root { --bg:#0b0706; --panel:#130b0a; --line:#2b1714; --text:#f0d6cf; --dim:#9a6f67; --cyan:#ffb4a6; --green:#ffe2d6; --orange:#ff7a45; --red:#ff3b28; --yellow:#ffb066; --violet:#e0909c; --pink:#ff6f8a; --magenta:#ff4f6f; }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font: 13px/1.45 ui-monospace, Menlo, Consolas, monospace; margin: 0; padding: 16px 20px 40px; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 22px; margin-bottom: 14px; }
  .chips { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 22px; }
  h1 { font-size: 17px; font-weight: normal; color: var(--green); margin: 0; }
  .chip { color: var(--dim); } .chip b { color: var(--cyan); font-weight: normal; }
  h2 { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--cyan); margin: 0 0 4px; font-weight: normal; }
  .cap { color: var(--dim); font-size: 12px; margin: 0 0 10px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; min-width: 0; }
  /* Tap/click a panel to show it full screen; tap again (or Esc) to put it back.
     Where the browser has no full screen for a card (an iPhone), it fills the page instead. */
  .panel.maximized { position: fixed; inset: 8px; z-index: 1000; overflow: auto; box-shadow: 0 0 0 100vmax rgba(0, 0, 0, 0.75); }
  .panel.maximized:fullscreen { inset: 0; border-radius: 0; border: 0; box-shadow: none; }
  .panel.maximized::backdrop { background: var(--bg, #000); }
  body.has-max { overflow: hidden; touch-action: none; }
  .video16x9 { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; }
  .video16x9 iframe, .video16x9 img, .video16x9 canvas { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; object-fit: contain; }
  .video16x9 canvas { pointer-events: none; }
  /* the picture alone full screen (a click on it); where a browser can't, it fills the page */
  /* (no zoom cursor: a click on the video does nothing -- its full-screen button does) */
  /* its full-screen button (a 2026-09-29 UX panel: a visible control, where every video player puts one) */
  .panel { position: relative; }
  .fsbtn { position: absolute; z-index: 6; width: 28px; height: 28px; padding: 5px; border-radius: 4px; cursor: pointer;
           background: rgba(5, 7, 10, 0.55); border: 1px solid var(--line); color: #ffe2d6; line-height: 0; }
  .fsbtn:hover { background: rgba(5, 7, 10, 0.85); }
  /* its row: just below its picture, on the card -- never over what the picture shows */
  .fsrow { display: flex; justify-content: flex-end; margin-top: 4px; }
  .fsrow > .fsbtn { position: static; }
  .video16x9:fullscreen, .video16x9.filling { aspect-ratio: auto; background: #000; }
  /* iPhone: no full screen for anything but <video>, so the picture is lifted out of the page and fills the
     screen -- black to the edges, under the notch and the home bar, the page frozen behind it */
  .video16x9.filling { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; height: 100dvh; z-index: 2000; margin: 0; }
  .quad > .panel > canvas, .quad > .panel > .video16x9 { margin-bottom: 8px; }
  .stack { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
  #left-stack { align-self: start; }  /* its own height, so charts can fill the rest beside the body */
  .quad { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-bottom: 16px; }
  .panel.maximized::before { content: 'Esc or the corner button closes it'; float: right; color: var(--dim); font-size: 11px; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; margin-top: 16px; }
  @media (max-width: 1250px) { .quad { grid-template-columns: 1fr; } }  /* narrow: one column, in the same order */
  @media (max-width: 480px) { body { padding: 10px; } .charts { grid-template-columns: 1fr; } }
  canvas { display: block; max-width: 100%; }
  canvas.px { image-rendering: pixelated; }
  canvas.steer { cursor: pointer; } canvas.steer.steering { outline: 1px solid #9a6f67; outline-offset: -1px; cursor: grab; }
  /* its life's tally over the mushroom body (a 2026-09-29 UX panel): good in its phosphor green, bad in the HUD's red */
  /* Its tally HUD, after MIL-STD-1787 / MIL-STD-1472 practice (as the video's
     HUD): uppercase monospace, colour by meaning -- green normal, amber
     caution, red warning, white data -- and a dark outline on every character
     instead of a backing, so it reads over any cell it crosses. */
  /* a control and its own dropdowns, boxed together: which list belongs to which link */
  /* the organism's actions, a row of their own below the caption, each boxed like the page's buttons
     (watch this, back to camera) together with its own lists and notes */
  .orgrow { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; }
  #tree-panel { align-self: start; }  /* the tree's card ends where the tree does: the grid doesn't stretch it to its row's tallest */
  .grp { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px; vertical-align: middle; }  /* a button and its own list, close together; groups further apart */
  #brain-panel .orgrow .grp a, #brain-panel .orgrow .orgsel { font: inherit; background: var(--bg); color: var(--cyan); border: 1px solid var(--line); padding: 3px 6px; text-decoration: none; display: inline-block; }  /* as the page's buttons */
  #brain-panel .orgrow .grp a:hover, #brain-panel .orgrow .orgsel:hover { border-color: var(--cyan); }
  #brain-panel .orgrow [hidden] { display: none !important; }  /* the button style must not unhide what waits (Copy here, a peer's saves) */
  /* its life's tally, under the navigation card: four blocks side by side, wrapping */
  #mb-hud { margin-top: 10px; font: bold 12px monospace; text-transform: uppercase; display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px 18px; }
  #mb-hud .q { position: static !important; text-align: left !important; line-height: 15px; }
  #mb-hud .g { color: #7cffa0; } #mb-hud .b { color: #ffb000; } #mb-hud .w { color: #ff4040; } #mb-hud .n { color: #e8e8e8; }
  #mb-hud .h { color: #c8c8c8; }
  #brain-mode a, #mb-mode a, #randomize, #amnesia, #reset-founder, #save-organism, #save-here, #load-organism, #copy-here, #upload-organism { color: #b88a80; } .orgsel { font: inherit; font-size: 12px; max-width: 16em; background: #10161c; color: #e8d8d0; border: 1px solid #3a2a26; }  /* the page's own link colour (not the browser's blue, unreadable on it) */ #brain-mode b, #mb-mode b { color: #ffe2d6; }
  .sleep-views { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px; }
  .sleep-views canvas { width: 100%; height: auto; aspect-ratio: 1; display: block; }
  .legend span { display: inline-block; margin-right: 12px; white-space: normal; }
  .gauge { display: grid; grid-template-columns: 84px 1fr 44px; align-items: center; gap: 8px; margin: 3px 0; }
  .gauge .track { height: 10px; background: #162029; border-radius: 2px; overflow: hidden; }
  .gauge .fill { height: 10px; }
  .gauge .v { text-align: right; color: var(--cyan); }
  .note { color: var(--dim); font-size: 11px; margin: -1px 0 4px 92px; }
  .section { margin-top: 12px; }
  table.drives { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.drives td, table.drives th { padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }
  table.drives th { color: var(--dim); font-weight: normal; }
  .plus { color: var(--green); } .minus { color: var(--red); } .inn { color: var(--yellow); }
  .tag { font-size: 10px; padding: 1px 6px; border-radius: 8px; border: 1px solid var(--line); color: var(--dim); white-space: nowrap; }
  .tag.body { color: var(--green); border-color: #1f5a44; } .tag.hand { color: var(--orange); border-color: #5a3f1f; } .tag.innate { color: var(--yellow); border-color: #5a531f; }
  #trees { display: flex; flex-wrap: wrap; gap: 14px; overflow-x: auto; }
  .stale { color: var(--red); }
  input, button { font: inherit; background: var(--bg); color: var(--cyan); border: 1px solid var(--line); padding: 3px 6px; }
</style>
</head>
<body>
<header>
  <h1>cambrian-perception</h1>
  <span class="chip">generation <b id="h-gen">--</b></span>
  <span class="chip" id="h-stale"></span>
</header>

<!-- The page follows the signal through the animal (UX panel, 2026-09-28):
     what it sees (the stream, its gaze); perceiving (its visual field, its
     perception tree); thinking and memory (its brain, its mushroom body); its
     inner life (sleep, replay and dreams; its body); then the charts. -->
<div class="quad">
  <div class="panel" id="live-panel">
    <h2 id="live-title">live view</h2>
    <div class="video16x9" id="live-box"><img id="cam" alt="" style="visibility:hidden"><canvas id="hud"></canvas></div>
    <div class="cap" id="live-cap">--</div>
    <div class="cap" id="cam-note" style="margin-top:6px"></div>
    <div class="cap" id="hud-legend" style="margin-top:6px"><label><input type="checkbox" id="hud-on" checked> HUD</label> <span id="hud-legend-text"></span></div>
    <div class="cap" id="stream-link-row" style="margin-top:6px; display:none"><a id="stream-link" target="_blank" rel="noopener" style="color:var(--cyan)">open on YouTube</a></div>
    <!-- what it watches: a YouTube video instead of its camera (frames are never saved) -->
    <div id="dessert-card" style="margin-top:12px">
      <div id="dessert-status" class="cap" style="color:var(--cyan)">--</div>
      <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:6px">
        <input type="text" id="custom-url" placeholder="a YouTube URL to watch instead of the camera (frames are never saved)" style="flex:1 1 260px">
        <button id="custom-url-submit">watch this</button>
        <button id="dessert-cancel">back to camera</button>
      </div>
      <label class="cap" style="display:block; margin-top:6px"><input type="checkbox" id="dessert-timed"> back to the camera by itself at <input type="time" id="dessert-until" value="07:00"></label>
      <div id="submit-status" class="cap" style="margin-top:6px"></div>
    </div>
  </div>
  <div class="panel" id="space-panel">
    <h2>navigation</h2>
    <canvas id="space"></canvas>
    <div class="cap" id="space-cap">--</div>
    <div class="cap" id="cortex-cap"></div>
    <div class="cap" id="senses-strip"></div>
    <div id="mb-hud"></div>
  </div>
  <div class="panel" id="look-panel">
    <h2>gaze</h2>
    <canvas id="look" class="px"></canvas>
    <div class="cap"><span id="look-px">--</span></div>
    <div class="cap" style="margin-top:8px" id="look-scale"></div>
  </div>
  <div class="panel" id="field-panel">
    <h2>visual field</h2>
    <canvas id="field" class="px"></canvas>
    <div class="cap" id="layers">Layers:
      <label><input type="checkbox" data-layer="place"> <b style="color:#ffb066">food places</b></label>
      <label><input type="checkbox" data-layer="people"> <b style="color:#ff6f8a">people expected</b></label>
      <label><input type="checkbox" data-layer="familiar"> <b style="color:#ffb4a6">still surprising</b></label>
      <label><input type="checkbox" data-layer="priority"> <b style="color:#ff66cc">priority</b></label>
</div>
    <div class="cap"><span id="field-px">--</span> &middot; box: gaze &middot; <b style="color:#ff6f8a">dashed</b>: host &middot; <b style="color:#9ccf7a">dotted</b>: plant &middot; red frame: looming</div>
    <div class="cap" id="replay-clock">--</div>
  </div>
  <div class="panel" id="brain-panel" style="grid-column: 1 / -1">
    <h2>its brain</h2>
    <canvas id="brain" height="520"></canvas>
    <div class="cap"><b id="brain-mb"></b><b id="brain-units">--</b> units, <b id="brain-layers">--</b> stacked (a column each beside its units: what it adds to each unit now; faint adds nothing). <b style="color:var(--cyan)">Cyan</b> excites, <b style="color:var(--orange)">orange</b> inhibits. <span id="brain-mode"></span></div>
    <div class="cap orgrow"><span class="grp"><a href="#" id="brain-portrait" title="Its brain as a picture: every wire, from an oblique angle, saved to the device you are viewing on">Get 3D image</a> <span id="brain-portrait-note"></span></span> <span class="grp"><a href="#" id="randomize">New random brain</a> <span id="randomize-note"></span></span> <span class="grp"><a href="#" id="amnesia">New random founder</a> <span id="amnesia-note"></span></span> <span class="grp"><a href="#" id="reset-founder">Reset to founder</a> <span id="reset-founder-note"></span></span> <span class="grp"><a href="#" id="save-here" title="Save the organism into this ecohost's Games/Lifeforms/Cambrioids folder">Save</a> <span id="save-here-note"></span></span> <span class="grp"><select id="load-list" class="orgsel" title="This ecohost's saved organisms"><option value="">its saves&hellip;</option></select> <a href="#" id="load-organism" title="Load the chosen one in this one's place (this one is kept in a backup)">Load</a> <span id="load-organism-note"></span></span> <span class="grp">Copy from <select id="peer-list" class="orgsel" title="The hive's other ecohosts"><option value="">ecohost&hellip;</option></select> <select id="peer-saves" class="orgsel" hidden></select> <a href="#" id="copy-here" hidden>Copy here</a> <span id="copy-note"></span></span> <span class="grp"><a href="#" id="save-organism" title="Download the organism to the device you are viewing on">Download</a> <span id="save-organism-note"></span></span> <span class="grp"><a href="#" id="upload-organism" title="Put a file from the device you are viewing on into this ecohost's saves">Upload&hellip;</a><input type="file" id="load-file" accept=".cambrioid" hidden> <span id="upload-note"></span></span></div>
  </div>
  <div class="panel" id="tree-panel">
    <h2>its perception tree</h2>
    <div id="trees"></div>
  </div>
  <div class="stack" id="left-stack">
    <div class="panel" id="dream-panel">
      <h2>sleep, replay &amp; dreams</h2>
      <div class="cap" id="sleep-line">--</div>
      <div class="sleep-views">
        <div><canvas id="replay-eye" width="192" height="192" class="px"></canvas><div class="cap">recalled</div></div>
        <div><canvas id="replay-recon" width="192" height="192" class="px"></canvas><div class="cap" id="recon-cap">mind's eye</div></div>
        <div><canvas id="replay-seen" width="192" height="192"></canvas><div class="cap" id="seen-cap">what it saw</div></div>
      </div>
      <div class="cap" id="replay-eye-cap" style="margin-top:4px"></div>
      <canvas id="dream-map" class="px" style="margin-top:10px; display:block; margin-left:auto; margin-right:auto"></canvas>
      <div class="cap">blue NREM &middot; violet REM &middot; grey awake &middot; gold dreamt</div>
    </div>
    <div class="stack" id="charts-side"></div>
  </div>
  <div class="panel" id="body-panel">
    <h2>body</h2>
    <div class="cap">Gut &rarr; sugar &rarr; stores.</div>
    <div id="gauges"></div>
    <canvas id="energy-trace" height="60"></canvas>
    <div class="section"><h2>eating</h2>
      <div id="prey-gauge"></div>
      <canvas id="prey-trace" height="60"></canvas>
      <div id="food-gauge"></div>
      <canvas id="food-trace" height="60"></canvas>
    </div>
    <div class="section"><h2>how it moves</h2>
      <div id="movement"></div>
    </div>
  </div>
</div>
<div class="charts" id="charts"></div>

<div class="panel" style="margin-top:16px">
  <h2>what drives it</h2>
  <div class="cap">Every pressure on it, and its weight.</div>
  <table class="drives">
    <tr><th></th><th>weight</th><th>pressure</th><th>in a line</th></tr>
    <tr><td><span class="tag body">body</span></td><td class="minus">-3.0</td><td>homeostatic drive</td><td>Hunger, stores, threat, fatigue and sleep pressure over the run; only eating refills.</td></tr>
    <tr><td><span class="tag hand">teacher</span></td><td class="minus">-3.0</td><td>perception tree's error</td><td>Its tree guesses &ldquo;a host in my gaze?&rdquo;; YOLO grades it.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>trait + motor</td><td>tempo</td><td>How often it looks: an inherited pace its brain speeds or slows.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>trait</td><td>colour, prey sense, host preference</td><td>What it can see and smell of hosts, and which it favours; all priced.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>trait</td><td>eye</td><td>Size, cone patch, zoom lens, stabilizer, photoreceptor speed.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>trait</td><td>metabolism, feeding pump</td><td>Rest cost against stamina; how fast essence flows in at a bite.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>world</td><td>host defense</td><td>A host that comes at it takes back the bite's essence.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>physics</td><td>digestion</td><td>A fifth of every meal.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>trait</td><td>memory, replay, dreams</td><td>Where food was; replayed and dreamt paths while asleep.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>motor</td><td>sleep, vigilance</td><td>Its own choice within sleep pressure; a big enough change wakes it.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>input</td><td>hunger, search, curiosity, danger</td><td>Felt, not scored.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="plus">+1.0</td><td>flinch</td><td>Reacting to something looming.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="plus">+18 &times;</td><td>curiosity (gaze)</td><td>Reaching new gaze positions.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="minus">-1.0 / -0.5</td><td>dead field, corner / edge</td><td>Long stretches of nothing; its gaze centre in a corner or on an edge.</td></tr>
  </table>
</div>

<div class="panel" style="margin-top:16px" id="vitals-panel">
  <h2>genome and host</h2>
  <div class="chips">
  <span class="chip">fitness <b id="h-fit">--</b></span>
  <span class="chip">peak ever <b id="h-peak">--</b></span>
  <span class="chip">watching <b id="h-src">--</b></span>
  <span class="chip">gaze size <b id="h-look">--</b></span>
  <span class="chip">pace <b id="h-pace">--</b></span>
  <span class="chip">colour <b id="h-colour">--</b></span>
  <span class="chip">stabilizer <b id="h-stab">--</b></span>
  <span class="chip">zoom lens <b id="h-zoom">--</b></span>
  <span class="chip">metabolism <b id="h-metab">--</b></span>
  <span class="chip" title="its feeding pump: seconds of waking life each second on a host buys">pump <b id="h-pump">--</b></span>
  <span class="chip">vigilance <b id="h-vigil">--</b></span>
  <span class="chip">replay <b id="h-replay">--</b></span>
  <span class="chip">prey sense <b id="h-prey">--</b></span>
  <span class="chip">CPU quota <b id="h-quota">--</b></span>
  </div>
</div>

<script>
/*LOCK_HUD_JS*/
  const $ = id => document.getElementById(id);
  const INPUT_NAMES = ['light', 'motion', 'flow x', 'flow y', 'loom', 'gaze x', 'gaze y', 'eye size', 'sugar', 'arousal', 'threat', 'search', 'motion dx', 'motion dy', 'eye vx', 'eye vy', 'hunger', 'curiosity', 'tree', 'gut', 'reserve', 'sleep pressure', 'asleep', 'field light', 'light trend', 'prey scent', 'prey dir x', 'prey dir y', 'food value', 'place dx', 'place dy', 'place value', 'intruder', 'danger', 'plant scent', 'plant dir x', 'plant dir y', 'mismatch', 'mismatch dx', 'mismatch dy', 'recalled value', 'recalled dx', 'recalled dy', 'protein', 'host vx', 'host vy', 'own pace', 'missed', 'uncertainty', 'ground near', 'horizon', 'parallax', 'camera moving', 'archetype 1', 'archetype 2', 'archetype 3', 'archetype 4', 'collicular dx', 'collicular dy', 'collicular strength',
    'terrain', 'nearness', 'felt nearness', 'contact', 'turning', 'tilting', 'heading sin', 'heading cos', 'speed', 'acceleration', 'map value', 'riding',
    'texture contrast', 'texture fineness', 'texture grain', 'strangeness',
    'road heading', 'road curve', 'road crest', 'road offset', 'road sure'];  // controller.py's inputs, in order (81)
  const OUTPUT_NAMES = ['pan', 'tilt', 'zoom', 'alarm', 'tempo', 'sleep'];
  // Grown channels (controller.py): a latch feeds its output back; a
  // predictor feeds back how wrong it was about one of its inputs.
  function channelName(br, k, side) {
    const c = (br.channels || [])[k]; if (!c) return (side === 'in' ? 'in ' : 'out ') + k;
    if (c.kind === 'predict') return side === 'in' ? `err ${INPUT_NAMES[c.target]}` : `predict ${INPUT_NAMES[c.target]}`;
    return side === 'in' ? `loop ${k + 1} (back)` : `loop ${k + 1}` + (c.copy_of !== undefined ? ` (from ${OUTPUT_NAMES[c.copy_of] || 'loop'})` : '');
  }
  const REPLAY_FPS = 15;
  let D = null, t0 = performance.now();
  const CLK = { t0 };  // the one clock of the visual field and the picture (replayAt)

  function innerWidth(el) { const cs = getComputedStyle(el); return el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight); }
  // The quad's four displays share one shape -- its camera frame's -- and
  // the full width of their panels, so they are the same size and aligned.
  function quadAspect() { return D && D.frame_w && D.frame_h ? D.frame_h / D.frame_w : 9 / 16; }
  function isMax(el) { const p = el && el.closest ? el.closest('.panel') : null; return !!(p && p.classList.contains('maximized')); }
  // Width that also fits the screen's height when maximized, for a canvas of the given aspect (h/w).
  function fitWidth(panel, aspect) {
    const w = Math.floor(innerWidth(panel));
    return isMax(panel) ? Math.min(w, Math.floor((window.innerHeight - 140) / aspect)) : w;
  }

  // Phosphor heat (black, deep red, red, amber, bone): the page's own ramp.
  const HEAT = [[4, 2, 2], [60, 6, 4], [150, 18, 10], [230, 50, 24], [255, 120, 50], [255, 190, 120], [255, 240, 225]];
  function heatColour(v) {
    const t = Math.max(0, Math.min(1, v)) * (HEAT.length - 1), i = Math.min(HEAT.length - 2, Math.floor(t)), f = t - i;
    const [a, b] = [HEAT[i], HEAT[i + 1]];
    return `rgb(${Math.round(a[0] + f * (b[0] - a[0]))},${Math.round(a[1] + f * (b[1] - a[1]))},${Math.round(a[2] + f * (b[2] - a[2]))})`;
  }
  function drawGrid(ctx, values, shape, x, y, w, h, heat) {
    const [rows, cols] = shape;
    for (let i = 0; i < rows; i++) {
      const y0 = Math.round(y + i * h / rows), y1 = Math.round(y + (i + 1) * h / rows);
      for (let j = 0; j < cols; j++) {
        const x0 = Math.round(x + j * w / cols), x1 = Math.round(x + (j + 1) * w / cols);
        const v = Math.max(0, Math.min(1, values[i * cols + j])), g = Math.round(v * 255);
        ctx.fillStyle = heat ? heatColour(v) : `rgb(${g},${g},${g})`;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
  }
  // Colour display: rebuild RGB from its light receptor (L) and its
  // colour-opponent receptors (red-green a, blue-yellow b), inverting
  // L = .299R+.587G+.114B, a = R-G, b = B-(R+G)/2 (checked: round-trips
  // pure red, blue and grey exactly).
  function drawColourGrid(ctx, lum, colour, shape, x, y, w, h) {
    const [rows, cols] = shape, n = rows * cols;
    for (let i = 0; i < rows; i++) {
      const y0 = Math.round(y + i * h / rows), y1 = Math.round(y + (i + 1) * h / rows);
      for (let j = 0; j < cols; j++) {
        const k = i * cols + j, L = lum[k];
        const a = 2 * (colour[k] - 0.5), b = colour.length >= 2 * n ? 2 * (colour[n + k] - 0.5) : 0;
        const c = v => Math.round(Math.max(0, Math.min(1, v)) * 255);
        ctx.fillStyle = `rgb(${c(0.644 * a - 0.114 * b + L)},${c(-0.356 * a - 0.114 * b + L)},${c(0.144 * a + 0.886 * b + L)})`;
        const x0 = Math.round(x + j * w / cols), x1 = Math.round(x + (j + 1) * w / cols);
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
  }
  function crop(d, f) { const side = Math.round(d.frame_h * f); return [side, side]; }  // square, f of the frame's height
  function boxColor(cx, cy, f) {
    const nx = (cx - 0.5) / 0.5, ny = (cy - 0.5) / 0.5;  // where its gaze's CENTER is, over the whole frame
    const corner = Math.abs(nx * ny), edge = Math.max(Math.abs(nx), Math.abs(ny));
    return corner >= 0.5 ? '#ff3b28' : edge >= 0.75 ? '#ff7a45' : edge >= 0.4 ? '#ffb066' : '#fff4f0';
  }

  // Visual field + replay of the gaze's real path, frame by frame, no interpolation.
  function drawField(now) {
    requestAnimationFrame(drawField);
    const d = D;
    if (!d || !d.frame_w || !d.world_grid) return;
    const c = $('field'), W = Math.max(200, fitWidth($('field-panel'), d.frame_h / d.frame_w));
    const H = Math.round(W * d.frame_h / d.frame_w);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d');
    // What its wide-field eyes sense: where and how much things change, per
    // cell, scaled by its own motion gain -- as heat. (It never senses a
    // picture; without a live actor, the dim brightness grid stands in.)
    if (d.field_motion && d.world_grid_shape) {
      // each cell's share of the frame's movement -- the weights its motion-
      // location sense takes its centroid over (a walking camera saturates the
      // absolute amount everywhere; where it moves most is what it uses)
      const top = Math.max(1e-6, ...d.field_motion);
      drawGrid(ctx, d.field_motion.map(v => v / top), d.world_grid_shape, 0, 0, W, H, true);
    } else drawGrid(ctx, d.world_grid.map(v => 0.35 * v), d.world_grid_shape, 0, 0, W, H);
    drawLayers(ctx, d, W, H);
    // Its marks are as coarse as its receptors: everything snaps to cell edges.
    const [gR, gC] = d.world_grid_shape || [9, 16], cellW = W / gC, cellH = H / gR;
    const snapL = x => Math.floor(x * gC) * cellW, snapR = x => Math.ceil(x * gC) * cellW;
    const snapT = y => Math.floor(y * gR) * cellH, snapB = y => Math.ceil(y * gR) * cellH;
    // Gazes are unevenly spaced (its tempo changes): replay in real frame
    // time and show whichever gaze is current at that moment -- the same
    // clock as the replayed picture beside it (replayAt, LOCK_HUD_JS).
    const { traj, fps, lastIdx, cur, i, delay, catching } = replayAt(d, now, CLK, REPLAY_FPS);
    const ev = d.field_events && d.field_events[i];
    if (ev) {
      const [mx, my, act, loom, reflex] = ev;
      if (act > 0.05) {
        // where it senses motion: that one cell, lit
        ctx.fillStyle = `rgba(255, 244, 240, ${0.55 + 0.45 * act})`;
        ctx.fillRect(snapL(Math.min(0.9999, mx)), snapT(Math.min(0.9999, my)), cellW, cellH);
      }
      if (loom > 0.18) { ctx.strokeStyle = '#f44'; ctx.lineWidth = 6; ctx.strokeRect(3, 3, W - 6, H - 6); }
    }
    // Prey (YOLO: people and animals) -- coordinates only, never pixels.
    const pb = (d.prey_boxes && d.prey_boxes[Math.min(cur, d.prey_boxes.length - 1)]) || [];
    ctx.font = '12px monospace'; ctx.textBaseline = 'bottom';
    pb.forEach(([cls, conf, x0, y0, x1, y1]) => {
      ctx.fillStyle = 'rgba(255, 111, 138, 0.18)';  // a host: its cells tinted
      ctx.fillRect(snapL(x0), snapT(y0), snapR(x1) - snapL(x0), snapB(y1) - snapT(y0));
      ctx.strokeStyle = '#ff6f8a'; ctx.lineWidth = 3; ctx.setLineDash([8, 4]);
      ctx.strokeRect(snapL(x0), snapT(y0), snapR(x1) - snapL(x0), snapB(y1) - snapT(y0)); ctx.setLineDash([]);
    });
    // plants (its nectar): green, dotted, snapped to its cells like the hosts
    const plb = (d.plant_boxes && d.plant_boxes[Math.min(cur, d.plant_boxes.length - 1)]) || [];
    plb.forEach(([cls, conf, x0, y0, x1, y1]) => {
      ctx.strokeStyle = 'rgba(156, 207, 122, 0.9)'; ctx.lineWidth = 3; ctx.setLineDash([2, 4]);
      ctx.strokeRect(snapL(x0), snapT(y0), snapR(x1) - snapL(x0), snapB(y1) - snapT(y0)); ctx.setLineDash([]);
    });
    // the boxes carry no labels (they would clash on a coarse grid): one count for them all
    if (pb.length) {
      ctx.font = 'bold 15px monospace'; ctx.fillStyle = '#ff6f8a'; ctx.textBaseline = 'bottom';
      ctx.fillText(`TARGETS ${pb.length}`, 10, H - 8);
    }
    const eat = (d.eating && d.eating[Math.min(cur, d.eating.length - 1)]) || 0;
    if (eat > 0.01) {
      ctx.fillStyle = 'rgba(255, 95, 162, 0.9)'; ctx.font = 'bold 14px monospace'; ctx.textBaseline = 'top';
      ctx.fillText(`ABSORBING \u00b7 JĪNG ${(eat * 100).toFixed(0)}%`, 10, 10);
    } else {
      const sip = (d.snacks && d.snacks[Math.min(cur, d.snacks.length - 1)]) || 0;  // nectar: a surprise snack
      if (sip > 0.01) { ctx.fillStyle = 'rgba(200, 136, 255, 0.9)'; ctx.font = 'bold 14px monospace'; ctx.textBaseline = 'top'; ctx.fillText('SIPPING \u00b7 QÌ', 10, 10); }
    }
    // its trail over the last 3 s, in its own cells: each cell it passed, lit, fading with age
    let k0 = i; while (k0 > 0 && (traj[k0 - 1][3] ?? (k0 - 1)) >= cur - 3 * fps) k0--;
    const lit = new Map();
    for (let k = k0; k <= i; k++) lit.set(snapL(Math.min(0.9999, traj[k][0])) + ',' + snapT(Math.min(0.9999, traj[k][1])), (k - k0 + 1) / (i - k0 + 1));
    lit.forEach((age, key) => { const [tx, ty] = key.split(',').map(Number); ctx.fillStyle = `rgba(255, 180, 166, ${0.12 + 0.3 * age})`; ctx.fillRect(tx, ty, cellW, cellH); });
    const [cx, cy, f] = traj[i];
    const [cw, ch] = crop(d, f), s = W / d.frame_w;
    ctx.strokeStyle = boxColor(cx, cy, f); ctx.lineWidth = 5;
    const gx0 = snapL(Math.max(0, cx - cw / d.frame_w / 2)), gx1 = snapR(Math.min(1, cx + cw / d.frame_w / 2));
    const gy0 = snapT(Math.max(0, cy - ch / d.frame_h / 2)), gy1 = snapB(Math.min(1, cy + ch / d.frame_h / 2));
    ctx.strokeRect(gx0, gy0, Math.max(cellW, gx1 - gx0), Math.max(cellH, gy1 - gy0));
    $('replay-clock').textContent = (delay != null ? `delayed ${delay.toFixed(1)} s` : `replay ${(cur / fps).toFixed(1)} / ${((lastIdx + 1) / fps).toFixed(0)} s`) + (ev && ev[3] > 0.18 ? ' \u00b7 APPROACH' : '');
  }
  requestAnimationFrame(drawField);

  // Layers over the visual field: what it has learned about places (a
  // 2026-09-28 panel). Off by default; learned values, not thoughts.
  const LAYERS = {};
  try { Object.assign(LAYERS, JSON.parse(localStorage.getItem('layers') || '{}')); } catch (e) { }
  document.querySelectorAll('#layers input').forEach(cb => {
    cb.checked = !!LAYERS[cb.dataset.layer];
    cb.addEventListener('change', () => { LAYERS[cb.dataset.layer] = cb.checked; try { localStorage.setItem('layers', JSON.stringify(LAYERS)); } catch (e) { } });
  });
  // Its newer senses in one line under the visual field: its own pace, how
  // unsure its memory is, whether the camera itself moves, where its horizon
  // is, and what its archetype heads see now (a small bar each).
  function drawSenses(d) {
    const el = $('senses-strip'), s = d && d.senses; if (!el) return;
    if (!s) { el.textContent = ''; return; }
    const bar = v => `<span style="display:inline-block;width:40px;height:6px;background:#1c2a36;vertical-align:middle"><span style="display:block;height:6px;width:${Math.round(40 * Math.max(0, Math.min(1, v)))}px;background:#7fd4ff"></span></span>`;
    const parts = [`pace ${s.pace_s.toFixed(2)} s${s.missed ? ' <b style="color:#ff6f8a">missed</b>' : ''}`, `unsure ${bar(s.uncertainty)}`,
                   s.camera_moving ? '<b style="color:#7fd4ff">camera moving</b>' : 'camera still',
                   s.horizon != null ? `horizon ${Math.round(100 * s.horizon)}% down${s.horizon_from ? ` (${s.horizon_from})` : ''}` : 'no horizon yet',
                   s.new_places ? `${s.new_places} new place${s.new_places === 1 ? '' : 's'}` : '',
                   s.road && s.road.conf > 0.05 ? `road: ${Math.abs(s.road.curve) < 0.002 ? 'straight' : 'bends ' + (s.road.curve > 0 ? 'right' : 'left')}${Math.abs(s.road.crest) > 0.003 ? (s.road.crest > 0 ? ', a crest' : ', a dip') : ''}, ${Math.round(100 * s.road.conf)}% sure` : '',
                   d.oxygen && d.oxygen.stage ? `<b style="color:#ff6f8a">short of oxygen</b>: shed ${d.oxygen.shed.join(', ')} (${d.oxygen.behind_s.toFixed(1)} s behind)` : '',
                   Math.abs(s.turning || 0) > 0.01 ? `turning ${s.turning > 0 ? 'right' : 'left'} ${Math.round(100 * Math.abs(s.turning))}%` : '',
                   Math.abs(s.tilting || 0) > 0.01 ? `tilting ${s.tilting > 0 ? 'clockwise' : 'anticlockwise'} ${Math.round(100 * Math.abs(s.tilting))}%` : '',
                   s.heading != null ? `heading ${Math.round((s.heading + 360) % 360)}°` : '',
                   s.speed != null && Math.abs(s.speed) > 0.01 ? `moving ${s.speed.toFixed(2)} eye-heights/s` : '',
                   s.acceleration != null && Math.abs(s.acceleration) > 0.05 ? (s.acceleration > 0 ? 'starting' : 'stopping') : '',
                   s.place_value != null && Math.abs(s.place_value) > 0.01 ? `this place: ${s.place_value.toFixed(2)}` : '',
                   s.riding ? `riding: ${Math.round(100 * s.riding)}% of its view` : '',
                   s.texture && s.texture[0] ? `texture: contrast ${s.texture[0].toFixed(2)}, fineness ${s.texture[1].toFixed(2)}, grain ${s.texture[2].toFixed(2)}` : '',
                   s.contact ? `<b style="color:#ff6f8a">approaching</b>: contact in ~${(s.pace_s / s.contact).toFixed(1)} s` : '',  // tau is in its looks: shown in our seconds
                   s.felt_nearness ? `it feels: ${s.felt_nearness >= 0.67 ? 'near' : s.felt_nearness >= 0.33 ? 'mid' : 'far'} (${s.felt_nearness.toFixed(2)})` : '',
                   s.nearness != null && s.horizon != null ? `what it looks at: ${s.nearness >= 0.67 ? 'near' : s.nearness >= 0.33 ? 'mid' : s.nearness > 0 ? 'far' : 'beyond its ground'} (${s.nearness.toFixed(2)})` : ''].filter(Boolean);
    if (s.archetypes && s.archetypes.length) parts.push('sees ' + s.archetypes.map(([n, v]) => `${n} ${bar(v)}`).join(' '));
    el.innerHTML = 'Senses: ' + parts.join(' &middot; ');
  }
  function drawLayers(ctx, d, W, H) {
    // Each layer its own kind of mark, so they stack (a 2026-09-28 UX panel):
    // the heat dims to a ground, food places are outlines, people expected are
    // dots, habituation is shade (what is still surprising stays bright), and
    // replay and dreams are strokes.
    const m = d.maps, sn = d.senses || {}; if (!m || !Object.values(LAYERS).some(Boolean)) return;
    ctx.fillStyle = 'rgba(8, 4, 4, 0.5)'; ctx.fillRect(0, 0, W, H);
    const each = (vals, shape, draw) => {
      const [rows, cols] = shape, cw = W / cols, ch = H / rows;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) draw(vals[r * cols + c], c * cw, r * ch, cw, ch);
    };
    if (LAYERS.familiar) each(m.familiar, m.mem_shape, (v, x, y, w, h) => {  // habituated: shaded; never seen: darker
      const a = v < 0 ? 0.55 : 0.45 * v; if (a > 0.02) { ctx.fillStyle = `rgba(0, 0, 0, ${a})`; ctx.fillRect(x, y, w + 0.5, h + 0.5); }
    });
    if (LAYERS.place) {  // what has fed it there, plus what its dreams learned it leads to
      const worth = m.value ? m.place.map((v, k) => v + m.value[k]) : m.place;
      const top = Math.max(1e-6, ...worth.map(Math.abs));
      each(worth, m.shape, (v, x, y, w, h) => {
        if (v <= 0) return;
        const t = v / top, lw = 1 + 3 * t;
        ctx.strokeStyle = `rgba(255, 176, 102, ${0.35 + 0.6 * t})`; ctx.lineWidth = lw;
        ctx.strokeRect(x + lw / 2 + 1, y + lw / 2 + 1, w - lw - 2, h - lw - 2);
      });
    }
    if (LAYERS.priority && sn.priority) {  // its collicular priority map: bars by salience, the winner ringed
      const top = Math.max(1e-6, ...sn.priority);
      let best = -1, bi = -1; sn.priority.forEach((v, k) => { if (v > best) { best = v; bi = k; } });
      each(sn.priority, m.shape, (v, x, y, w, h) => {
        if (v <= 0) return;
        const t = v / top; ctx.fillStyle = `rgba(255, 102, 204, ${0.25 + 0.6 * t})`; ctx.fillRect(x + 2, y + h - 3 - (h - 6) * t, 4, (h - 6) * t);
      });
      if (best > 0) { const [rows, cols] = m.shape, cw = W / cols, ch = H / rows; ctx.strokeStyle = '#ff66cc'; ctx.lineWidth = 2; ctx.strokeRect((bi % cols) * cw + 1, Math.floor(bi / cols) * ch + 1, cw - 2, ch - 2); }
    }
    if (LAYERS.people) each(m.people, m.shape, (v, x, y, w, h) => {
      if (v <= 0.01) return;
      ctx.fillStyle = 'rgba(255, 111, 138, 0.9)'; ctx.beginPath();
      ctx.arc(x + w / 2, y + h / 2, 1.5 + Math.min(1, v) * Math.min(w, h) * 0.22, 0, 7); ctx.fill();
    });
  }

  // Its mushroom body: every Kenyon cell a dot; lit = firing for its latest
  // look (~5%); green = what it has learned means food, red = danger; dark =
  // lost to wasting. Learned values, not thoughts.
  // Its latest replayed memory on its eye (NREM, REM or awake): the receptors
  // the reactivated Kenyon cells sample, back-projected onto its gaze. Not a
  // picture it makes -- which parts of its eye the memory is built from.
  const SEEN = { key: null, ok: false, img: new Image() };
  // Its sleep in a line: asleep or awake (and for how long), sleep pressure,
  // whether it is dreaming now, and the inherited shape of its replay.
  // An empty view shows its medium, not a message: a faint grid at the
  // resolution it would draw at (the state goes in its caption).
  function idleView(ctx, W, k) {
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, W);
    ctx.strokeStyle = '#141a20'; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 1; i < k; i++) { const x = Math.round(i * W / k) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, W); ctx.moveTo(0, x); ctx.lineTo(W, x); }
    ctx.stroke();
  }
  const DIM = s => `<span style="color:#6a5550">${s}</span>`;
  const WOKE = { mismatch: 'the room changed', loom: 'something looming', motion: 'movement', rested: 'rested', choice: 'its own choice' };
  function drawSleepLine(d) {
    const el = $('sleep-line'), z = d.sleep; if (!el) return;
    if (!z) { el.textContent = '--'; return; }
    const t = z.traits, mins = Math.round(z.for_s / 60);
    el.innerHTML = (z.asleep ? `<b style="color:#e0909c">asleep</b> ${mins} min` : '<b style="color:#a9bcc8">awake</b>')
      + ` &middot; sleep pressure ${(100 * z.pressure).toFixed(0)}%`
      + (z.clock_day != null ? ` &middot; clock ${z.clock_day >= 0.5 ? 'day' : 'night'}` : '')
      + (z.dreaming ? ' &middot; <b style="color:#fff0c0">dreaming</b>' : '')
      + ` &middot; replays ${t.awake}/${t.asleep}, REM ${(100 * t.rem).toFixed(0)}%`
      + (z.plasticity ? ` &middot; habits ${z.distilled}` : '')
      + (z.maturation ? ` &middot; matured ${Math.round(100 * (z.locked || 0))}%` : '')
      + (z.sleep_set ? ` &middot; self-edits ${z.edits[0]}/${z.edits[1]}` : '')
      + (z.max_scenes > 1 ? ` &middot; scene ${z.scene}/${z.scenes}` : '')
      + (!z.asleep && z.woke_by ? ` &middot; woke: ${WOKE[z.woke_by] || z.woke_by}` : '')
      + (z.mismatch > 0.05 ? ` &middot; room changed ${(100 * z.mismatch).toFixed(0)}%` : '');
  }
  function drawReplayEye(d) {
    drawSleepLine(d);
    const c = $('replay-eye'); if (!c) return;
    const r = d.replay_eye, W = c.width, ctx = c.getContext('2d');
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, W);
    if (!r) {
      idleView(ctx, W, 16); ['replay-recon', 'replay-seen'].forEach(id => idleView($(id).getContext('2d'), W, 16));
      $('recon-cap').innerHTML = "mind's eye"; $('seen-cap').innerHTML = 'what it saw';
      $('replay-eye-cap').textContent = !d.kc ? 'no mushroom body yet' : 'nothing replayed lately';
      drawDreamMap(d); return;
    }
    const n = r.n, s = W / n;
    const fade = Math.max(0.25, 1 - r.age / 10);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = r.grid[i * n + j]; if (!v) continue;
      ctx.globalAlpha = fade; ctx.fillStyle = heatColour(v); ctx.fillRect(Math.floor(j * s), Math.floor(i * s), Math.ceil(s), Math.ceil(s));
    }
    ctx.globalAlpha = 1;
    // its own reconstruction (imagery, when its lineage has evolved it): its prototypes, summed
    const rc = $('replay-recon'), rctx = rc.getContext('2d'); rctx.fillStyle = '#05070a'; rctx.fillRect(0, 0, W, W);
    if (r.recon) {
      const k = r.recon_side, q = W / k;
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
        const g = Math.round(255 * Math.max(0, Math.min(1, r.recon[i * k + j])));
        rctx.fillStyle = `rgb(${g},${Math.round(g * 0.92)},${Math.round(g * 0.88)})`; rctx.fillRect(Math.floor(j * q), Math.floor(i * q), Math.ceil(q), Math.ceil(q));
      }
    } else idleView(rctx, W, 16);
    // what it saw then: the frame the memory formed on, cropped to its gaze, if the frame ring still holds it
    const sc = $('replay-seen'), sctx = sc.getContext('2d'); sctx.fillStyle = '#05070a'; sctx.fillRect(0, 0, W, W);
    let seenState = '';
    if (r.seen) {
      const key = (d.world_epoch || 0) + ':' + r.seen.i;
      if (SEEN.key !== key) { SEEN.key = key; SEEN.ok = false; SEEN.img.onload = () => { SEEN.ok = true; }; SEEN.img.onerror = () => { SEEN.ok = false; }; SEEN.img.src = '/frame?i=' + r.seen.i + '&e=' + (d.world_epoch || 0); }
      if (SEEN.ok && SEEN.img.naturalWidth) {
        const iw = SEEN.img.naturalWidth, ih = SEEN.img.naturalHeight, side = r.seen.f * ih;
        sctx.drawImage(SEEN.img, r.seen.cx * iw - side / 2, r.seen.cy * ih - side / 2, side, side, 0, 0, W, W);
      } else { idleView(sctx, W, 16); seenState = 'gone from RAM'; }
    } else { idleView(sctx, W, 16); seenState = r.kind === 'rem' ? 'no one moment' : 'none'; }
    $('seen-cap').innerHTML = 'what it saw' + (seenState ? ' ' + DIM('&middot; ' + seenState) : '');
    $('recon-cap').innerHTML = ((d.sleep && d.sleep.dreaming) ? 'dream' : "mind's eye") + (r.recon ? '' : ' ' + DIM('&middot; not evolved yet'));
    const name = { nrem: 'NREM replay', rem: 'REM (recombined)', awake: 'awake replay' }[r.kind] || r.kind;
    $('replay-eye-cap').textContent = `${name}, ${r.age.toFixed(1)} s ago`;
    drawDreamMap(d);
  }
  // Replayed and dreamt paths on its place map (its field's grid): a small
  // square per place, joined in order; each kind its own colour; fading over 3 s.
  const DM = { extra: 0 };  // extra height the place map takes to fill the left column
  function drawDreamMap(d) {
    const c = $('dream-map'); if (!c) return;
    const m = d.maps, shape = (m && m.shape) || d.world_grid_shape || [9, 16], [rows, cols] = shape;
    // its grid is coarse, so it starts small -- and grows into any space left beside the body card (balanceSide)
    const PW = c.parentElement.clientWidth, base = Math.max(160, Math.round(PW * 0.6));
    const W = Math.min(PW, Math.round((base * rows / cols + DM.extra) * cols / rows)), H = Math.round(W * rows / cols);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d'), cw = W / cols, ch = H / rows;
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
    if (m && m.place) {  // the place map itself, faint: where food has been
      const top = Math.max(1e-6, ...m.place.map(Math.abs));
      m.place.forEach((v, k) => { if (v > 0) { ctx.fillStyle = `rgba(255, 176, 102, ${0.25 * v / top})`; ctx.fillRect((k % cols) * cw, Math.floor(k / cols) * ch, cw, ch); } });
    }
    const hue = { nrem: '120, 160, 255', rem: '224, 144, 196', awake: '160, 150, 150', dream: '255, 214, 120' };
    let prev = null;
    (d.dreams || []).forEach(([kind, r, c2, age, seq]) => {
      const a = Math.max(0, 1 - age / 3);
      if (a) {
        const q = Math.min(cw, ch) * 0.3;
        ctx.fillStyle = ctx.strokeStyle = `rgba(${hue[kind] || '200, 200, 200'}, ${0.9 * a})`; ctx.lineWidth = 2;
        ctx.fillRect((c2 + 0.5) * cw - q / 2, (r + 0.5) * ch - q / 2, q, q);
        if (prev && seq && prev[4] === seq && (prev[1] !== r || prev[2] !== c2)) {
          ctx.beginPath(); ctx.moveTo((prev[2] + 0.5) * cw, (prev[1] + 0.5) * ch); ctx.lineTo((c2 + 0.5) * cw, (r + 0.5) * ch); ctx.stroke();
        }
      }
      prev = [kind, r, c2, age, seq];
    });
  }
  function int8s(b64) { if (!b64) return null; const s = atob(b64); return Int8Array.from(s, ch => ch.charCodeAt(0)); }
  // Its mushroom body in 3D, laid out as an insect's (Menzel): the Kenyon
  // cells in the calyx, a cup at the top, each coloured by what it has
  // learned (green food, red danger; grey once lost to wasting); the cells
  // firing now glow and send their axons down the peduncle, which splits into
  // the medial lobe (its food-value readout) and the vertical lobe (danger).
  // Orbit, zoom, hover as the brain; it redraws on new data or a touch.
  // Steerable graphics (brain, mushroom body, tree): the page keeps its own
  // scroll and pinch-zoom until you click or tap a graphic; then the wheel,
  // drag and pinch steer it (a thin outline shows which). Clicking outside it
  // or Esc releases it -- the "cooperative gestures" of web maps. Clicking a
  // graphic never expands its card (click its title or caption for that).
  // A phone (a touch screen that small): the 3D views are made for it --
  // fewer labels (the tapped unit's and the outputs'), nodes scaled to the
  // canvas, coarser meshes -- and the brain opens in 2D (its own choice, kept
  // apart from a desktop's).
  const MOBILE = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
  const SMALL = W => Math.min(1, W / 600);  // node sizes scale with a small canvas
  const STEER = { el: null };
  function steerable(c) { c.classList.add('steer'); c.style.touchAction = 'auto'; return () => STEER.el === c; }
  function steerOn(c) { if (STEER.el && STEER.el !== c) steerOff(); STEER.el = c; c.style.touchAction = 'none'; c.classList.add('steering'); }
  function steerOff() { if (!STEER.el) return; STEER.el.style.touchAction = 'auto'; STEER.el.classList.remove('steering'); STEER.el = null; }
  document.addEventListener('pointerdown', e => { if (STEER.el && e.target !== STEER.el) steerOff(); }, true);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && STEER.el) { steerOff(); e.stopImmediatePropagation(); } }, true);
  // Idle drift for the 3D views (brain, mushroom body, perception trees): a
  // slow turn, eased in and out, from 4 s to 24 s after it was last touched
  // (or the page opened), then still -- its loop then sleeps until the next
  // touch, so a viewer left open costs nothing (it shares its machine with an
  // organism). driftStep advances S.yaw and says whether to keep animating.
  function driftStep(S, now, busy) {
    const dt = Math.min(0.1, (now - (S.last || now)) / 1000); S.last = now;
    const since = now - S.touched, drifting = since > 4000 && since < 24000 && !busy;
    S.vyaw += ((drifting ? 0.12 : 0) - S.vyaw) * Math.min(1, dt * 1.5);
    S.yaw += S.vyaw * dt;
    return since < 24000 || Math.abs(S.vyaw) > 1e-4;
  }
  const MB3 = { yaw: 2.95, pitch: 0.1, zoom: 1, vyaw: 0, touched: performance.now() - 4000, loop: false, dirty: true, d: null, drag: null, pos: null, n: 0, hover: null };
  function mbPositions(n) {
    const pos = new Float32Array(n * 3), ga = Math.PI * (3 - Math.sqrt(5));
    for (let k = 0; k < n; k++) {  // a cup: the upper part of a sphere, cells spread evenly over it and through its wall
      const t = (k + 0.5) / n, y = 1 - 0.75 * t, rr = Math.sqrt(Math.max(0, 1 - y * y)), th = ga * k;
      const wall = 0.82 + 0.18 * ((k * 0.618034) % 1);
      pos[3 * k] = 0.62 * wall * rr * Math.cos(th); pos[3 * k + 1] = -0.55 - 0.45 * wall * y; pos[3 * k + 2] = 0.62 * wall * rr * Math.sin(th);
    }
    return pos;
  }
  function drawMB3D() {
    const d = MB3.d, c = $('mb'); if (!d || !d.mb || !c) return;
    const mb = d.mb, n = mb.n, W = c.width, H = c.height, ctx = c.getContext('2d');
    if (MB3.n !== n) { MB3.pos = mbPositions(n); MB3.n = n; }
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
    const cy = Math.cos(MB3.yaw), sy = Math.sin(MB3.yaw), cp = Math.cos(MB3.pitch), sp = Math.sin(MB3.pitch);
    const scale = Math.min(W, H) * 0.36 * MB3.zoom, F = 3.4;
    const proj = (x, y, z) => {
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy, y2 = (y + 0.08) * cp - z1 * sp, z2 = (y + 0.08) * sp + z1 * cp, w = F / (F + z2);
      return [W / 2 + x1 * scale * w, H / 2 + y2 * scale * w, z2, w];
    };
    const food = int8s(mb.food), danger = int8s(mb.danger), on = new Set(mb.active || []);
    // Colour: each cell by what it has learned, relative to the strongest cell
    // (green food, red danger, amber learned-bad food); faint unless firing.
    let fmax = 1e-6, gmax = 1e-6;
    for (let k = 0; k < Math.min(n, mb.live); k++) { if (food) fmax = Math.max(fmax, Math.abs(food[k])); if (danger) gmax = Math.max(gmax, Math.max(0, danger[k])); }
    const cellRGB = (k, lit) => {  // dim, bright when firing; green / red / amber by what it learned
      const f = food ? food[k] / fmax : 0, g = danger ? Math.max(0, danger[k]) / gmax : 0, base = lit ? 150 : 30;
      return [Math.min(255, Math.round(base + 200 * Math.max(g, f < 0 ? -f : 0))),
              Math.min(255, Math.round(base + 200 * Math.max(0, f) + (f < 0 ? 110 * -f : 0))), base];
    };
    // Firing cells are phosphor green (P1 phosphor), graded deep to bright by
    // how strongly the cell has learned anything.
    const PHOS = [70, 255, 130];
    const phosphor = k => 0.45 + 0.55 * Math.min(1, Math.max(food ? Math.abs(food[k]) / fmax : 0, danger ? Math.max(0, danger[k]) / gmax : 0));
    // the peduncle and the two lobes: a stalk down from the calyx, then a fork
    const PED = [0, -0.05, 0], MED = [0.95, 0.55, 0], VER = [-0.35, 0.95, 0.35];
    const tube = (a, b2, col, wd) => { const A = proj(...a), B = proj(...b2); ctx.strokeStyle = col; ctx.lineWidth = wd * (A[3] + B[3]) / 2; ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); };
    ctx.lineCap = 'round';
    tube(PED, MED, 'rgba(90,110,125,0.3)', 11);  // the axons themselves are the peduncle tube(PED, VER, 'rgba(90,110,125,0.3)', 11);
    // cells, far to near
    const P = MB3.pos, pts = [];
    for (let k = 0; k < n; k++) { const q = proj(P[3 * k], P[3 * k + 1], P[3 * k + 2]); pts.push([k, q]); }
    pts.sort((x, y) => y[1][2] - x[1][2]);
    const dot = Math.max(1.5, 0.8 * scale * Math.sqrt(2.4 / Math.max(1, n)));  // about the spacing between cells, so it grows as you zoom
    ctx.globalCompositeOperation = 'lighter';
    // the firing cells' axons: down the peduncle, into each lobe by what they carry
    let fsum = 0, gsum = 0;
    on.forEach(k => {
      if (k >= n) return;
      const f = food ? food[k] / 127 : 0, g = danger ? danger[k] / 127 : 0; fsum += f; gsum += g;
      const A = proj(P[3 * k], P[3 * k + 1], P[3 * k + 2]), B = proj(...PED);
      ctx.strokeStyle = 'rgba(255,240,200,0.10)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
    });
    if (on.size) {
      const ped = proj(...PED), m = proj(...MED), v = proj(...VER), fa = Math.min(1, Math.abs(fsum) / Math.max(1, on.size) * 3), ga2 = Math.min(1, Math.abs(gsum) / Math.max(1, on.size) * 3);
      ctx.strokeStyle = `rgba(${fsum >= 0 ? '120,230,120' : '255,153,0'},${0.15 + 0.7 * fa})`; ctx.lineWidth = 2 + 6 * fa; ctx.beginPath(); ctx.moveTo(ped[0], ped[1]); ctx.lineTo(m[0], m[1]); ctx.stroke();
      ctx.strokeStyle = `rgba(255,70,50,${0.15 + 0.7 * ga2})`; ctx.lineWidth = 2 + 6 * ga2; ctx.beginPath(); ctx.moveTo(ped[0], ped[1]); ctx.lineTo(v[0], v[1]); ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';
    pts.forEach(([k, q]) => {
      const fog = Math.max(0.35, Math.min(1, 0.7 - 0.4 * q[2]));
      if (k >= mb.live) { ctx.fillStyle = `rgba(70,70,70,${0.6 * fog})`; ctx.fillRect(q[0] - dot / 2, q[1] - dot / 2, dot, dot); return; }
      const lit = on.has(k), a = lit ? fog : 0.6 * fog;  // unlit stay dim
      const cc = lit ? PHOS.map(c => Math.round(c * phosphor(k))) : cellRGB(k, false);  // firing: phosphor green, graded
      ctx.fillStyle = `rgba(${cc[0]},${cc[1]},${cc[2]},${a})`;
      const z = lit ? dot * 1.8 : dot;
      ctx.fillRect(q[0] - z / 2, q[1] - z / 2, z, z);
    });
    if (on.size) {  // the firing cells glow
      ctx.globalCompositeOperation = 'lighter';
      on.forEach(k => { if (k >= n) return; const q = proj(P[3 * k], P[3 * k + 1], P[3 * k + 2]), r = 2.5 * dot, v = phosphor(k); const gr = ctx.createRadialGradient(q[0], q[1], 0, q[0], q[1], r); gr.addColorStop(0, `rgba(60,255,120,${0.18 * v})`); gr.addColorStop(1, 'rgba(60,255,120,0)'); ctx.fillStyle = gr; ctx.fillRect(q[0] - r, q[1] - r, 2 * r, 2 * r); });  /* a faint green halo */
      ctx.globalCompositeOperation = 'source-over';
    }
    // the two readouts
    ctx.font = '11px monospace'; ctx.textBaseline = 'middle';
    [[MED, 'food value', d.food_value, '120,230,120'], [VER, 'danger', d.danger_value, '255,90,70']].forEach(([pt, name, val, col]) => {
      const q = proj(...pt), r = 9 * q[3] * Math.sqrt(MB3.zoom) * (MOBILE ? SMALL(W) : 1);
      ctx.fillStyle = '#0a2a1a'; ctx.strokeStyle = `rgb(${col})`; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = `rgb(${col})`; ctx.textAlign = q[0] > W / 2 ? 'left' : 'right';
      ctx.fillText(`${name}${val != null ? ' ' + Number(val).toFixed(2) : ''}`, q[0] + (q[0] > W / 2 ? r + 5 : -r - 5), q[1]);
    });
    const lab = proj(0, -1.12, 0); ctx.fillStyle = '#9a6f67'; ctx.textAlign = 'center'; ctx.fillText('calyx: Kenyon cells', lab[0], lab[1]);
    const pl = proj(...PED); ctx.textAlign = 'left'; ctx.fillText('peduncle', pl[0] + 10, pl[1]);
    if (MB3.hover != null && MB3.hover < n) {
      const k = MB3.hover, q = proj(P[3 * k], P[3 * k + 1], P[3 * k + 2]);
      ctx.fillStyle = '#ffe2d6'; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'left';
      ctx.fillText(`cell ${k}: food ${food ? (food[k] / 127).toFixed(2) : '--'}, danger ${danger ? (danger[k] / 127).toFixed(2) : '--'}${on.has(k) ? ', firing' : ''}${k >= mb.live ? ', lost' : ''}`, q[0] + 8, q[1] - 8);
    }
    MB3.proj = proj;
  }
  // 2D or 3D: a phone opens it in 2D, a desktop in 3D, each choice kept on its own
  const MB_VIEW_KEY = MOBILE ? 'mb-view-phone' : 'mb-view';
  MB3.flat = MOBILE;
  try { const v = localStorage.getItem(MB_VIEW_KEY); if (v) MB3.flat = v === '2d'; } catch (e) {}
  // Its mushroom body flat: the calyx's Kenyon cells as a grid (coloured as in
  // 3D: green food, red danger, amber food it learned to avoid, grey once lost;
  // firing cells glow phosphor green), and the two lobes as its two readouts.
  function drawMB2D() {
    const d = MB3.d, c = $('mb'); if (!d || !d.mb || !c) return;
    const mb = d.mb, n = mb.n, W = c.width, H = c.height, ctx = c.getContext('2d');
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
    const food = int8s(mb.food), danger = int8s(mb.danger), on = new Set(mb.active || []);
    let fmax = 1e-6, gmax = 1e-6;
    for (let k = 0; k < Math.min(n, mb.live); k++) { if (food) fmax = Math.max(fmax, Math.abs(food[k])); if (danger) gmax = Math.max(gmax, Math.max(0, danger[k])); }
    const barH = Math.max(28, Math.round(H * 0.16)), gh = H - barH - 12, pad = 6;
    const cols = Math.max(1, Math.round(Math.sqrt(n * (W - 2 * pad) / Math.max(1, gh)))), rows = Math.ceil(n / cols);
    const cw = (W - 2 * pad) / cols, ch = gh / rows;
    for (let k = 0; k < n; k++) {
      const x = pad + (k % cols) * cw, y = pad + Math.floor(k / cols) * ch;
      if (k >= mb.live) { ctx.fillStyle = '#2a2a2a'; }
      else if (on.has(k)) {
        const g = 0.45 + 0.55 * Math.min(1, Math.max(food ? Math.abs(food[k]) / fmax : 0, danger ? Math.max(0, danger[k]) / gmax : 0));
        ctx.fillStyle = `rgb(${Math.round(70 * g)}, ${Math.round(255 * g)}, ${Math.round(130 * g)})`;
      } else {
        const f = food ? food[k] / fmax : 0, g = danger ? Math.max(0, danger[k]) / gmax : 0;
        ctx.fillStyle = `rgb(${Math.min(255, Math.round(30 + 200 * Math.max(g, f < 0 ? -f : 0)))}, ${Math.min(255, Math.round(30 + 200 * Math.max(0, f) + (f < 0 ? 110 * -f : 0)))}, 30)`;
      }
      ctx.fillRect(x, y, Math.max(1, cw - (cw > 3 ? 1 : 0)), Math.max(1, ch - (ch > 3 ? 1 : 0)));
    }
    // the lobes: its two readouts, -1 .. 1 about the middle
    const bar = (i, name, v, col) => {
      const y = H - barH + i * (barH / 2), w = W - 2 * pad, mid = pad + w / 2, bh = barH / 2 - 4;
      ctx.fillStyle = '#10161c'; ctx.fillRect(pad, y, w, bh);
      const val = Math.max(-1, Math.min(1, v || 0));
      ctx.fillStyle = `rgb(${col})`; ctx.fillRect(Math.min(mid, mid + val * w / 2), y, Math.abs(val) * w / 2, bh);
      // its label as the HUD's: uppercase, white, a dark outline (no backing)
      const label = `${name} ${val.toFixed(2)}`.toUpperCase();
      ctx.font = `bold ${Math.max(10, Math.min(12, bh - 2))}px monospace`; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.lineJoin = 'round'; ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(label, pad + 3, y + bh / 2);
      ctx.fillStyle = '#e8e8e8'; ctx.fillText(label, pad + 3, y + bh / 2);
    };
    bar(0, 'food value (medial lobe)', d.food_value, '124,255,160');
    bar(1, 'danger (vertical lobe)', d.danger_value, '255,64,64');
  }
  function setMBView(flat) {
    MB3.flat = flat; try { localStorage.setItem(MB_VIEW_KEY, flat ? '2d' : '3d'); } catch (e) {}
    if (!flat) MB3.touched = performance.now() - 4000;  // into 3D: it drifts a while
    MB3.dirty = true; mbKick();
    if (MB3.d) drawMB(MB3.d);
  }
  function mbKick() {
    if (MB3.loop) return;
    MB3.loop = true;
    const tick = now => {
      const more = driftStep(MB3, now, MB3.flat || MB3.drag || MB3.hover != null);
      const c = $('mb'), r = c && c.getBoundingClientRect(), moving = Math.abs(MB3.vyaw) > 1e-4;
      if ((MB3.dirty || moving) && r && r.bottom > 0 && r.top < innerHeight && !document.hidden) { MB3.dirty = false; if (MB3.flat) drawMB2D(); else drawMB3D(); }
      if (more || MB3.dirty) requestAnimationFrame(tick); else { MB3.vyaw = 0; MB3.last = null; MB3.loop = false; }
    };
    requestAnimationFrame(tick);
  }
  (() => {
    const c = $('mb'); if (!c) return;
    const active = steerable(c);
    const pt = e => { const r = c.getBoundingClientRect(); return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]; };
    const touch = () => { MB3.touched = performance.now(); MB3.dirty = true; mbKick(); };
    c.addEventListener('wheel', e => { if (!active()) return; e.preventDefault(); MB3.zoom = Math.max(0.5, Math.min(8, MB3.zoom * Math.exp(-e.deltaY * 0.0015))); touch(); }, { passive: false });
    c.addEventListener('dblclick', () => { if (!active()) return; MB3.yaw = 2.95; MB3.pitch = 0.1; MB3.zoom = 1; touch(); });
    c.addEventListener('pointerdown', e => { if (MB3.flat) return; if (!active()) { steerOn(c); return; } c.setPointerCapture(e.pointerId); MB3.drag = pt(e); c.style.cursor = 'grabbing'; touch(); });  // flat: nothing to steer
    c.addEventListener('pointermove', e => {
      const now = pt(e);
      if (MB3.drag) { MB3.yaw += (now[0] - MB3.drag[0]) * 0.008; MB3.pitch = Math.max(-1.4, Math.min(1.4, MB3.pitch + (now[1] - MB3.drag[1]) * 0.008)); MB3.drag = now; touch(); return; }
      if (!MB3.proj || !MB3.pos) return;
      let best = null, bd = 8; const P = MB3.pos;
      for (let k = 0; k < MB3.n; k++) { const q = MB3.proj(P[3 * k], P[3 * k + 1], P[3 * k + 2]), dd = Math.hypot(q[0] - now[0], q[1] - now[1]); if (dd < bd) { bd = dd; best = k; } }
      if (best !== MB3.hover) { MB3.hover = best; touch(); }
    });
    const up = () => { MB3.drag = null; c.style.cursor = ''; };
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
    c.addEventListener('pointerleave', () => { if (MB3.hover != null) { MB3.hover = null; touch(); } });
  })();
  document.addEventListener('click', e => { const a = e.target.closest('#mb-mode a'); if (a) { e.preventDefault(); e.stopPropagation(); setMBView(a.dataset.mb === '2d'); } }, true);
  // Its life's tally (livelife.Tally): each event per hour lived (its last 60
  // minutes) and since its birth. Good, after the Three Treasures (README):
  // jīng (separate bites of a host, in kǒu), qì (sips of nectar, in xī), shén
  // (looks that fed on something new, in niàn). Bad: swats, missed looks (still
  // thinking when it had to act), minutes starving. Neither: approaches
  // (something began coming at its gaze).
  function drawMBHud(d) {
    const el = $('mb-hud'), t = d && d.tally; if (!el) return;
    if (!t) { el.innerHTML = ''; return; }
    const ph = t.per_hour, sb = t.since_birth, f = v => (v = v || 0) >= 100 ? String(Math.round(v)) : v >= 10 ? v.toFixed(0) : v.toFixed(1);  // always a string (padStart)
    const row = (cls, name, k, unit) => `<div class="${cls}">${name.padEnd(9, '\u00a0')} ${f(ph[k]).padStart(5, '\u00a0')}${unit ? ` <span style="text-transform:none">${unit}</span>` : ''}/h \u00b7 ${f(sb[k])}</div>`;  // a unit stays lowercase, as units are
    const dur = s => s < 172800 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} d`;
    const L = d.life || {}, age = dur(t.hours_lived * 3600);
    const span = L.expected_lifespan_s ? ` of ~${dur(L.expected_lifespan_s)}` : '';  // its lifespan, as its own damage rate so far says
    // colour by meaning (MIL-STD practice): a bad thing this hour is a warning
    // (red), one only in its past a caution (amber); torpor or a cyst now, amber
    const bad = k => ((ph[k] || 0) > 0 ? 'w' : 'b'), now = k => ((ph[k] || 0) > 0 ? 'b' : 'n');
    // in 2D its lobes' bars fill the canvas's foot: the bottom rows sit above them
    const cv = $('mb'), lift = MB3.flat && cv && cv.height ? Math.round(Math.max(28, Math.round(cv.height * 0.16)) * (cv.clientHeight / cv.height || 1)) + 12 : 8;
    el.innerHTML = `<div class="q" style="left:10px;top:8px"><div class="h">GOOD · /h · since birth</div>${row('g', 'JĪNG 精', 'meals', 'kǒu')}${row('g', 'QÌ 氣', 'sips', 'xī')}${row('g', 'SHÉN 神', 'snacks', 'niàn')}${row('g', 'EGGS', 'eggs')}</div>`
      + `<div class="q" style="right:10px;top:8px;text-align:right"><div class="h">BAD · /h · since birth</div>${row(bad('swats'), 'SWATS', 'swats')}${row(bad('missed'), 'MISSED', 'missed')}${row(bad('starving_min'), 'STARVING', 'starving_min').replace('/h', ' min/h')}<div class="${(L.deaths || 0) > 0 ? 'w' : 'b'}">DEATHS${' '.repeat(4)} ${L.deaths || 0} on this ecohost</div></div>`
      + `<div class="q" style="left:10px;bottom:${lift}px">${row('n', 'APPROACH', 'approaches')}${row(now('torpor_min'), 'TORPOR', 'torpor_min').replace('/h', ' min/h')}${row(now('cyst_min'), 'CYST', 'cyst_min').replace('/h', ' min/h')}${L.developed ? '<div class="g">DEVELOPED: beats its founder</div>' : ''}</div>`
      + `<div class="q n" style="right:10px;bottom:${lift}px;text-align:right">${L.hive ? 'HIVE on' : 'SOLO · hive off'}${L.migrant_from ? ` · lineage from ${String(L.migrant_from).replace(/[^\w.-]/g, '')}` : ''}<br>age ${age}${span}</div>`;
  }
  function drawMB(d) {
    const c = $('mb'), cap = $('mb-cap'); if (!c) return;
    const mb = d.mb;
    drawMBHud(d);
    if (!mb) {  // no mushroom body (yet): the card keeps its tally
      const W = Math.max(200, fitWidth($('mb-panel'), quadAspect())), H = Math.round(W * quadAspect());
      if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
      const ctx = c.getContext('2d'); ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#6b5a55'; ctx.font = '12px monospace'; ctx.textAlign = 'center'; ctx.fillText('no mushroom body yet', W / 2, H / 2);
      MB3.d = null; c.style.display = ''; cap.textContent = ''; return;
    }
    c.style.display = '';
    const W = Math.max(200, fitWidth($('mb-panel'), quadAspect())), H = Math.round(W * quadAspect());
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    MB3.d = d; MB3.dirty = true; mbKick();
    const mode = MB3.flat ? '<b>2D</b> &middot; <a href="#" data-mb="3d">3D</a>' : '<a href="#" data-mb="2d">2D</a> &middot; <b>3D</b>';
    cap.innerHTML = `${mb.n} Kenyon cells, ${(mb.active || []).length} firing &middot; <b style="color:#78e678">food</b> <b style="color:#ff5a46">danger</b> <b style="color:#ffb066">avoided</b> <b style="color:#888">lost</b> <span id="mb-mode">${mode}</span>`;
  }

  // Its sense of space, rendered in 3D: the world as it reconstructs it. Its
  // learned ground plane laid out in depth (each field cell below the horizon
  // at the distance the plane implies: depth ~ 1 / (y - horizon), sideways
  // ~ depth x x), the hosts it sees standing on it at their distance and
  // height, parallax as cyan columns where things move against the camera,
  // and, inset, the lines its perception trees read on its eye (oriented
  // pools as bars at their angle). Click it to steer, like the others.
  // Seen from its own eye by default (POV: at its camera's height, looking where
  // it looks -- it is driving into this, not studying a graph of it), or from
  // outside ("overview", the oblique orbit).
  const SP3 = { pov: true, hy: 0, hp: 0, yaw: -0.5, pitch: 0.45, zoom: 1, dirty: true, loop: false, drag: null, d: null };
  // Its navigation HUD (a 2026-09-29 UX panel -- Kare, Victor, Norman, Raskin,
  // Laurel -- after a fighter's HUD, MIL-STD-1787's layout): one colour for its
  // own symbology (phosphor green, as its mushroom body's), the world's colours
  // for the world; conformal where it can be (horizon, flight-path marker at
  // the focus of expansion), tapes for the rest: heading on top (its compass),
  // speed on the left (its speed cells; the caret leans with starting or
  // stopping), nearness on the right (NR: T what its ground model teaches, F what
  // its terrain head feels, their values beside the label; near at the top), a bank pointer (its tilting),
  // and a data block top left; its path and its tree's lines sit in the bottom
  // corners, the middle kept clear. Text scales with the picture, floored as the video's HUD.
  function hudNav(ctx, W, H, hz, sn, d) {
    const k = Math.max(0.6, Math.min(1, W / 560)), px = n => Math.max(8, Math.round(n * k)), G = a => `rgba(124, 255, 160, ${a})`;
    ctx.save(); ctx.lineWidth = 1; ctx.font = `${px(11)}px monospace`;
    // a dark outline around its symbology (as the video's HUD and the tally's):
    // legible over bright ground or video, with no backing behind it
    ctx.shadowColor = 'rgba(0, 0, 0, 0.9)'; ctx.shadowBlur = 3;
    const hy = hz * H;
    // horizon line, broken at the middle for the flight-path marker
    ctx.strokeStyle = G(0.55); ctx.beginPath(); ctx.moveTo(W * 0.14, hy); ctx.lineTo(W * 0.44, hy); ctx.moveTo(W * 0.56, hy); ctx.lineTo(W * 0.86, hy); ctx.stroke();
    // flight-path marker: where it is going (the focus of expansion, moved by its turning), shown while it moves
    if (sn.speed != null && Math.abs(sn.speed) > 0.01) {
      const fx = W * (0.5 + 0.5 * Math.max(-1, Math.min(1, sn.turning || 0))), r = px(6);
      ctx.strokeStyle = G(0.95); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(fx, hy, r, 0, 7);
      ctx.moveTo(fx - r, hy); ctx.lineTo(fx - 2.4 * r, hy); ctx.moveTo(fx + r, hy); ctx.lineTo(fx + 2.4 * r, hy); ctx.moveTo(fx, hy - r); ctx.lineTo(fx, hy - 1.8 * r); ctx.stroke(); ctx.lineWidth = 1;
    }
    // bank pointer: its tilting this look, on an arc under the heading tape
    const bx = W / 2, by = px(46), br = px(22), tilt = Math.max(-1, Math.min(1, sn.tilting || 0)) * Math.PI / 4;
    ctx.strokeStyle = G(0.5); ctx.beginPath(); ctx.arc(bx, by + br, br, -Math.PI / 2 - Math.PI / 4, -Math.PI / 2 + Math.PI / 4); ctx.stroke();
    ctx.fillStyle = G(0.95); ctx.beginPath(); const ta = -Math.PI / 2 + tilt; ctx.moveTo(bx + br * Math.cos(ta), by + br + br * Math.sin(ta));
    ctx.lineTo(bx + (br - px(6)) * Math.cos(ta - 0.08), by + br + (br - px(6)) * Math.sin(ta - 0.08)); ctx.lineTo(bx + (br - px(6)) * Math.cos(ta + 0.08), by + br + (br - px(6)) * Math.sin(ta + 0.08)); ctx.fill();
    // heading tape (its compass): 60 degrees wide, ticks every 10, numbers every 30
    if (sn.heading != null) {
      const hd = (sn.heading + 360) % 360, tw = Math.min(W * 0.36, px(220)), ty = px(14), x0 = W / 2 - tw / 2;
      ctx.strokeStyle = G(0.7); ctx.fillStyle = G(0.85); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      for (let a = Math.ceil((hd - 30) / 10) * 10; a <= hd + 30; a += 10) {
        const x = W / 2 + (a - hd) / 30 * (tw / 2), big = ((a % 30) + 30) % 30 === 0;
        ctx.beginPath(); ctx.moveTo(x, ty + px(4)); ctx.lineTo(x, ty + px(big ? 11 : 7)); ctx.stroke();
        if (big) ctx.fillText(String(((a % 360) + 360) % 360).padStart(3, '0'), x, ty + px(3));
      }
      ctx.beginPath(); ctx.moveTo(W / 2, ty + px(12)); ctx.lineTo(W / 2 - px(4), ty + px(18)); ctx.lineTo(W / 2 + px(4), ty + px(18)); ctx.closePath(); ctx.fill();
      if (sn.road && sn.road.conf > 0.2) {  // where its road goes: an open caret under the tape (its road organ)
        const rx = W / 2 + Math.max(-30, Math.min(30, sn.road.heading * 180 / Math.PI)) / 30 * (tw / 2);
        ctx.save(); ctx.strokeStyle = G(0.9); ctx.beginPath(); ctx.moveTo(rx, ty + px(20)); ctx.lineTo(rx - px(4), ty + px(27)); ctx.lineTo(rx + px(4), ty + px(27)); ctx.closePath(); ctx.stroke(); ctx.restore();
      }
      const tr = Math.max(-1, Math.min(1, sn.turning || 0));  // its turning: a caret along the tape
      if (Math.abs(tr) > 0.01) { ctx.fillRect(W / 2, ty + px(19), tr * tw / 2, 2); }
    }
    // data block, top left: what else it senses now (the tapes start below it)
    const rows = [sn.out_of_model ? 'LEARNING HELD' : '', sn.strangeness ? `ODD ${sn.strangeness.toFixed(2)}` : '',
                  sn.riding ? `RIDE ${Math.round(100 * sn.riding)}%` : '', sn.ram != null && sn.ram > 1.01 ? `RAM x${sn.ram.toFixed(1)}` : '',sn.texture && sn.texture[0] ? `TEX c${sn.texture[0].toFixed(2)} f${sn.texture[1].toFixed(2)} g${sn.texture[2].toFixed(2)}` : '',
                  sn.place_value != null && Math.abs(sn.place_value) > 0.01 ? `PLACE ${sn.place_value >= 0 ? '+' : ''}${sn.place_value.toFixed(2)}` : '',
                  sn.heading != null ? `HDG ${String(Math.round((sn.heading + 360) % 360)).padStart(3, '0')}` : ''].filter(Boolean);
    const TOP = Math.max(px(70), px(8) + rows.length * px(13) + px(22));
    // a tape with its value boxed at the middle
    const tape = (x, v, lo, hi, side, label, marks) => {
      const y0 = TOP, th = Math.max(px(30), H - Math.min(110, H * 0.3) - px(34) - y0), yv = t => y0 + th * (1 - (t - lo) / (hi - lo));
      ctx.strokeStyle = G(0.6); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + th); ctx.stroke();
      for (let t = 0; t <= 4; t++) { const yy = y0 + th * t / 4; ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(x + side * px(5), yy); ctx.stroke(); }
      (marks || []).forEach(([mv, col, name]) => { if (mv == null) return; const yy = yv(Math.max(lo, Math.min(hi, mv)));
        ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(x + side * px(8), yy - px(4)); ctx.lineTo(x + side * px(8), yy + px(4)); ctx.fill();
        ctx.textAlign = side > 0 ? 'left' : 'right'; ctx.textBaseline = 'middle'; ctx.fillText(name, x + side * px(10), yy); });
      ctx.fillStyle = G(0.85); ctx.textAlign = side > 0 ? 'left' : 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(label, x, y0 - px(3));
    };
    const spd = sn.speed, acc = sn.acceleration || 0;
    if (spd != null) {
      const top = Math.max(0.5, Math.ceil(Math.abs(spd) * 2) / 2);
      tape(px(12), spd, 0, top, 1, `SPD ${spd.toFixed(2)}${acc > 0.05 ? ' ▲' : acc < -0.05 ? ' ▼' : ''}`, [[Math.max(0, spd), G(0.95), '']]);
    }
    const nr = v => v == null ? '--' : v.toFixed(2);
    tape(W - px(12), null, 0, 1, -1, `NR T${nr(sn.nearness)} F${nr(sn.felt_nearness)}`, [[sn.nearness, 'rgb(156, 207, 122)', 'T'], [sn.felt_nearness, 'rgb(127, 212, 255)', 'F']]);
    ctx.fillStyle = G(0.85); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    rows.forEach((t, i) => ctx.fillText(t, px(8), px(8) + i * px(13)));
    ctx.restore();
  }
  function drawSpace(d) {
    const c = $('space'); if (!c || !d) return;
    const W = Math.max(240, Math.floor(c.parentElement.clientWidth - 24));
    const H = Math.round(Math.min(W * ((d.frame_h && d.frame_w) ? d.frame_h / d.frame_w : 0.5625), 560));  // the stream's own shape: the two side by side
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    SP3.d = d;
    const ctx = c.getContext('2d'), sn = d.senses || {};
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
    const m = d.maps, [rows, cols] = (m && m.shape) || d.world_grid_shape || [9, 16];
    const hz = sn.horizon;
    const cy = Math.cos(SP3.yaw), sy = Math.sin(SP3.yaw), cp = Math.cos(SP3.pitch), sp = Math.sin(SP3.pitch);
    const F = 4.0, scale = Math.min(W, H) * 0.42 * SP3.zoom;
    const proj = (x, y, z) => {  // world: x sideways, y up, z away (centred on the scene's middle)
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy, y2 = -y * cp - z1 * sp, z2 = -y * sp + z1 * cp, w = F / (F + z2);
      return [W / 2 + x1 * scale * w, H / 2 + y2 * scale * w, z2, w];
    };
    const ZMAX = SP3.pov ? 40.0 : 3.0;  // the far edge drawn (the horizon itself is infinitely far)
    const place = (fx, fy) => {  // a point of the frame on its ground: (sideways, depth), or null above the horizon
      if (hz == null || fy <= hz + 1e-3) return null;
      const z = Math.min(ZMAX, (1 - hz) / (fy - hz));  // 1 at the frame's bottom, growing toward the horizon
      return [(fx - 0.5) * z * 1.6, z];
    };
    // Its eye: a pinhole at its camera's height (1.6 (1 - horizon) here), the
    // frame's own geometry -- looking straight ahead at zoom 1, a point lands
    // where the stream shows it (frame x = 0.5 + x / 1.6 z; y = horizon + (h - y) / 1.6 z).
    const HC = hz != null ? 1.6 * (1 - hz) : 1.0;
    const hcy = Math.cos(SP3.hy), hsy = Math.sin(SP3.hy), hcp = Math.cos(SP3.hp), hsp = Math.sin(SP3.hp);
    const eye = (x, y, z) => {
      const ry = y - HC, x1 = x * hcy - z * hsy, z1 = x * hsy + z * hcy;
      const y2 = ry * hcp + z1 * hsp, z2 = Math.max(0.05, -ry * hsp + z1 * hcp);
      return [W / 2 + W * SP3.zoom * x1 / (1.6 * z2), H * (hz != null ? hz : 0.5) - H * SP3.zoom * y2 / (1.6 * z2), z2, Math.min(1, 1 / z2)];
    };
    const at = SP3.pov ? eye : (x, y, z) => proj(x, y, z - ZMAX / 2 - 0.5);
    let note = '';
    if (hz == null) {
      note = 'no horizon yet';
    } else {
      // Its ground, as its mind holds it (a 2026-09-29 panel: the surface
      // goes through feet and roots, never over heads; and it is the
      // organism's own terrain map, not the viewer's guess). Its plane is
      // flat; its terrain map says, per field cell, how far the ground under
      // the things it measured there rises or falls (camera heights; one
      // camera height here is 1.6 (1 - horizon)).
      const HCAM = 1.6 * (1 - hz), pts = [], terr = sn.terrain || [];
      const last = k => (d[k] && d[k].length) ? (d[k][d[k].length - 1] || []) : [];
      [['host', last('prey_boxes')], ['plant', last('plant_boxes')], ['thing', last('thing_boxes')]].forEach(([kind, bxs]) => bxs.forEach(([cls, conf, x0, y0, x1, y1]) => {
        const px = Math.max(1, (d.frame_w || 16) / (d.frame_h || 9)) / 640, cutBase = y1 >= 1 - px, cutTop = y0 <= px;  // one detector pixel (letterboxed)
        const g0 = place((x0 + x1) / 2, y1); if (!g0) return;
        pts.push({ kind, x: g0[0], z: g0[1], hgt: (y1 - y0) * g0[1] * 1.6, x0, x1, y1, conf, cutBase, cutTop });
      }));
      // the mesh reads its map -- already blended in its mind (neighbouring
      // cells share evidence: the ground is continuous) -- each node back to
      // the frame (depth z at frame row hz + (1 - hz) / z), read bilinearly
      let NU = MOBILE ? 24 : 48, NZ = MOBILE ? 16 : 32; const cells = [];  // the drawing's squares, finer than its map (which is its own, coarse)
      terr.forEach((t, k) => { if (t) cells.push([0, 0, t[0] * HCAM]); });
      const cellE = (r, c) => { r = Math.max(0, Math.min(rows - 1, r)); c = Math.max(0, Math.min(cols - 1, c)); const t = terr[r * cols + c]; return t ? t[0] * HCAM : 0; };
      const groundAt = (x, z) => {
        if (!cells.length) return 0;
        const fy = hz + (1 - hz) / z, fx = x / (1.6 * z) + 0.5, u = fx * cols - 0.5, v = fy * rows - 0.5;
        const c0 = Math.floor(u), r0 = Math.floor(v), du = u - c0, dv = v - r0;
        return (1 - dv) * ((1 - du) * cellE(r0, c0) + du * cellE(r0, c0 + 1)) + dv * ((1 - du) * cellE(r0 + 1, c0) + du * cellE(r0 + 1, c0 + 1));
      };
      const sticks = cells.map(q => ({ elev: q[2] }));
      const big = Math.max(1e-6, ...sticks.map(q => Math.abs(q.elev)));
      const node = [];
      // From its eye: equal squares on the ground (TS a side, out to TX either
      // side and TZ ahead), so lines running away meet at the horizon and
      // cross-lines crowd toward it -- perspective. From outside: its frame's grid.
      const TS = MOBILE ? 0.5 : 0.25, TX = 16, TZ = 24;
      if (SP3.pov) { NU = Math.round(2 * TX / TS); NZ = Math.round((TZ - 1) / TS); }
      for (let j = 0; j <= NZ; j++) {
        const z = SP3.pov ? 1 + j * TS : 1 + (ZMAX - 1) * j / NZ, row = [];
        for (let i = 0; i <= NU; i++) { const x = SP3.pov ? -TX + i * TS : (i / NU - 0.5) * 1.6 * z, h = groundAt(x, z); row.push([at(x, h, z), h]); }
        node.push(row);
      }
      const seg = (p, q) => {
        if ((p[0][0] < 0 && q[0][0] < 0) || (p[0][0] > W && q[0][0] > W) || (p[0][1] > H && q[0][1] > H)) return;  // off its view
        const t = Math.min(1, Math.abs(p[1] + q[1]) / 2 / big);
        const fade = SP3.pov ? Math.min(1, W * TS / (1.6 * p[0][2]) / 6) : 1;  // squares under ~6 px fade out, not into a solid band
        ctx.strokeStyle = `rgba(${Math.round(156 + 99 * t)}, ${Math.round(207 - 96 * t)}, ${Math.round(122 + 16 * t)}, ${(0.3 + 0.3 * t) * fade})`;
        ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); ctx.lineTo(q[0][0], q[0][1]); ctx.stroke();
      };
      ctx.lineWidth = 1;
      for (let j = NZ; j >= 0; j--) for (let i = 0; i < NU; i++) seg(node[j][i], node[j][i + 1]);
      for (let i = 0; i <= NU; i++) for (let j = NZ; j > 0; j--) seg(node[j][i], node[j - 1][i]);
      // parallax: cyan columns on its cells (nearness, not ground)
      if (sn.parallax) sn.parallax.forEach((v, k) => {
        if (v <= 0.05) return;
        const r = Math.floor(k / cols), cc = k % cols, g = place((cc + 0.5) / cols, (r + 0.5) / rows); if (!g) return;
        const base = groundAt(g[0], g[1]), a2 = at(g[0], base, g[1]), b2 = at(g[0], base + 0.6 * v, g[1]);
        ctx.strokeStyle = `rgba(127, 212, 255, ${0.5 + 0.5 * v})`; ctx.lineWidth = 3 * a2[3]; ctx.beginPath(); ctx.moveTo(a2[0], a2[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
      });
      // the things, standing on that ground at their distance and height; one
      // the frame cuts is drawn dashed with its cut side open (its base is
      // somewhere nearer, below the frame; or its top above it) and measures nothing
      ctx.save(); ctx.shadowColor = 'rgba(0, 0, 0, 0.9)'; ctx.shadowBlur = 3;  // its labels outlined dark, as the HUD's: legible over the grid, no backing
      pts.forEach(q => {
        const g0 = place(q.x0, q.y1), g1 = place(q.x1, q.y1); if (!g0 || !g1) return;
        const base = groundAt(q.x, q.z), top = base + q.hgt, cut = q.cutBase || q.cutTop;
        const c4 = [at(g0[0], base, g0[1]), at(g1[0], base, g1[1]), at(g1[0], top, g1[1]), at(g0[0], top, g0[1])];
        const rgb = q.kind === 'plant' ? '156, 207, 122' : q.kind === 'thing' ? '160, 160, 170' : '255, 111, 138';
        // stroked only (no fill: MIL-STD symbology); how sure the detector is, in the line's weight and brightness
        ctx.strokeStyle = `rgba(${rgb}, ${cut ? 0.45 + 0.3 * q.conf : 0.55 + 0.45 * q.conf})`; ctx.lineWidth = 1 + 1.5 * q.conf;
        ctx.setLineDash(cut ? [4, 3] : []);
        const edge = (i, j) => { ctx.beginPath(); ctx.moveTo(c4[i][0], c4[i][1]); ctx.lineTo(c4[j][0], c4[j][1]); ctx.stroke(); };
        if (!q.cutBase) edge(0, 1); edge(1, 2); if (!q.cutTop) edge(2, 3); edge(3, 0);
        ctx.setLineDash([]);
        // what it could give, inside its top left corner: 精 jīng (a host), 氣 qì
        // (a plant), 神 shén (a thing: only what is new in it); sized to the box
        const bh = Math.abs(c4[0][1] - c4[3][1]), bw = Math.abs(c4[1][0] - c4[0][0]), hs = Math.min(18, Math.round(Math.min(bh, bw) * 0.3));
        if (hs >= 9) {
          ctx.font = `bold ${hs}px "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", sans-serif`;
          ctx.fillStyle = `rgb(${rgb})`; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
          // placed by its ink, not its em box (a glyph's ink starts below the em box's top):
          // the same gap to the box's top as to its left
          const ch = q.kind === 'plant' ? '氣' : q.kind === 'thing' ? '神' : '精', m = ctx.measureText(ch), gap = 3;
          ctx.fillText(ch, Math.min(c4[3][0], c4[0][0]) + gap + (m.actualBoundingBoxLeft || 0), Math.min(c4[3][1], c4[2][1]) + gap + (m.actualBoundingBoxAscent || 0.88 * hs));
        }
        const tr = ((d.cortex && d.cortex.tracks) || []).find(k => k.who != null && Math.abs(k.box[0] - q.x0) < 1e-3 && Math.abs(k.box[3] - q.y1) < 1e-3);
        if (tr) { const m = at((g0[0] + g1[0]) / 2, top, (g0[1] + g1[1]) / 2); ctx.fillStyle = `rgb(${rgb})`; ctx.font = '11px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText('#' + tr.who, m[0], m[1] - 3); }
      });
      ctx.restore();
      const nCut = pts.filter(q => q.cutBase || q.cutTop).length;
      // its road organ's road, conformal: each edge laid on its ground where the
      // frame shows it (as a HUD draws a runway) -- the HUD's green, as bright as
      // the organ is sure; it bends with the land as its terrain map does
      const road = sn.road;
      if (road && road.lines && road.conf > 0.05) {
        ctx.save(); ctx.strokeStyle = `rgba(124, 255, 160, ${0.35 + 0.6 * Math.min(1, road.conf)})`; ctx.lineWidth = 1 + 1.5 * Math.min(1, road.conf);
        road.lines.forEach(ln => {
          ctx.beginPath(); let started = false;
          ln.forEach(([fx, fy]) => { const g = place(fx, fy); if (!g) return; const a = at(g[0], groundAt(g[0], g[1]), g[1]); if (started) ctx.lineTo(a[0], a[1]); else { ctx.moveTo(a[0], a[1]); started = true; } });
          ctx.stroke();
        });
        ctx.restore();
      }
      // where it expects the host it follows (its extrapolation), standing on its ground, dashed
      const ah = last('ahead_boxes');
      if (ah && ah.length === 4) {
        const g0 = place(ah[0], ah[3]), g1 = place(ah[2], ah[3]);
        if (g0 && g1) {
          const gz = place((ah[0] + ah[2]) / 2, ah[3]), base = gz ? groundAt(gz[0], gz[1]) : 0, top = base + (ah[3] - ah[1]) * g0[1] * 1.6;
          const c4 = [at(g0[0], base, g0[1]), at(g1[0], base, g1[1]), at(g1[0], top, g1[1]), at(g0[0], top, g0[1])];
          ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(255, 111, 138, 0.6)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(c4[0][0], c4[0][1]); c4.slice(1).forEach(r => ctx.lineTo(r[0], r[1])); ctx.closePath(); ctx.stroke(); ctx.restore();
        }
      }
      // its local frame (what rides with it: a cab, a bonnet): the region's
      // outline where its eye sees it, dashed in the HUD's green -- an
      // obstruction marked by its edge, not veiled (MIL-STD: no backing)
      if (SP3.pov && sn.local_frame && sn.local_frame.length === rows * cols) {
        const lf = sn.local_frame, on = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && !!lf[r * cols + c];
        const X = c => Math.round(c * W / cols) + 0.5, Y = r => Math.round(r * H / rows) + 0.5;
        ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(124, 255, 160, 0.55)'; ctx.lineWidth = 1; ctx.beginPath();
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          if (!on(r, c)) continue;
          if (!on(r - 1, c)) { ctx.moveTo(X(c), Y(r)); ctx.lineTo(X(c + 1), Y(r)); }
          if (!on(r + 1, c)) { ctx.moveTo(X(c), Y(r + 1)); ctx.lineTo(X(c + 1), Y(r + 1)); }
          if (!on(r, c - 1)) { ctx.moveTo(X(c), Y(r)); ctx.lineTo(X(c), Y(r + 1)); }
          if (!on(r, c + 1)) { ctx.moveTo(X(c + 1), Y(r)); ctx.lineTo(X(c + 1), Y(r + 1)); }
        }
        ctx.stroke(); ctx.restore();
      }
      // its gaze: a reticle where it looks (from its eye, a frame point is a screen point), the ring
      // tightening as something approaches (tau); beside it the nearness its ground model teaches and the one it feels
      if (SP3.pov && d.fovea_cx != null) {
        const gx = d.fovea_cx * W, gy = d.fovea_cy * H, ct = Math.max(0, Math.min(1, last('contact_frames') || sn.contact || 0));
        const rr = 16 * (1 - 0.6 * ct);
        ctx.strokeStyle = ct > 0 ? 'rgba(255, 90, 70, 0.95)' : 'rgba(127, 212, 255, 0.8)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(gx, gy, rr, 0, 7); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(gx - rr - 4, gy); ctx.lineTo(gx - rr + 3, gy); ctx.moveTo(gx + rr - 3, gy); ctx.lineTo(gx + rr + 4, gy); ctx.stroke();
      }
      if (SP3.pov) hudNav(ctx, W, H, hz, sn, d);
      note = `horizon ${Math.round(100 * hz)}% down &middot; <b style="color:#ff6f8a">hosts</b> <b style="color:#9ccf7a">plants</b> <b style="color:#a0a0aa">things</b>`
             + (nCut ? ` &middot; dashed: cut by the frame` : '') + (sn.parallax && sn.parallax.some(v => v > 0.05) ? ' &middot; <b style="color:#7fd4ff">parallax</b>' : '')
             + (sn.local_frame && sn.local_frame.some(Boolean) ? ' &middot; <b style="color:#7cffa0">dashed green: rides with it</b>' : '');
    }
    // its path (path integration: speed x heading), bottom left; north up = its first heading; its facing now as a tick
    const path = d.path, INSET = Math.min(110, H * 0.3);
    if (path && path.length > 1) {
      const side = INSET, ox = 8, oy = H - INSET - 20;
      ctx.fillStyle = '#05070a'; ctx.fillRect(ox - 4, oy - 4, side + 8, side + 20);  // a window, opaque (an MFD inset): nothing seen through it
      ctx.strokeStyle = 'rgba(124, 255, 160, 0.35)'; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, side, side);
      const xs = path.map(p => p[0]), ys = path.map(p => p[1]);
      const cxp = (Math.min(...xs) + Math.max(...xs)) / 2, cyp = (Math.min(...ys) + Math.max(...ys)) / 2;
      const span = Math.max(1, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.15;
      const P = p => [ox + side / 2 + (p[0] - cxp) / span * side, oy + side / 2 - (p[1] - cyp) / span * side];
      ctx.strokeStyle = 'rgba(124, 255, 160, 0.85)'; ctx.beginPath(); path.forEach((p, k) => { const q = P(p); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.stroke();  // the HUD's green (amber is caution)
      const e = P(path[path.length - 1]); ctx.strokeStyle = 'rgb(124, 255, 160)'; ctx.beginPath(); ctx.arc(e[0], e[1], 3, 0, 7); ctx.stroke();
      if (sn.heading != null) { const a = sn.heading * Math.PI / 180; ctx.beginPath(); ctx.moveTo(e[0], e[1]); ctx.lineTo(e[0] + 9 * Math.sin(a), e[1] - 9 * Math.cos(a)); ctx.stroke(); }
      ctx.fillStyle = 'rgba(124, 255, 160, 0.85)'; ctx.font = '10px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(`path ${span.toFixed(0)} eye-heights`, ox, oy + side + 3);
    }
    // inset: the lines its trees read on its eye
    const N = d.receptors || 12, side = INSET, cell = side / N, ox = W - side - 8, oy = H - INSET - 20;
    ctx.fillStyle = '#05070a'; ctx.fillRect(ox - 4, oy - 4, side + 8, side + 20);  // a window, opaque (an MFD inset)
    ctx.strokeStyle = 'rgba(124, 255, 160, 0.35)'; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, side, side);
    let edges = 0;
    const arc = (x, y, a, L, pr, bend, ys) => { ctx.beginPath(); for (let i = 0; i <= 12; i++) { const tp = (i / 6 - 1) * L, np = 0.5 * (bend || 0) * (tp / pr) * (tp / pr) * pr, px = x - tp * Math.sin(a) + np * Math.cos(a), py = y + (tp * Math.cos(a) + np * Math.sin(a)) * ys; if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); } ctx.stroke(); };  // an oriented pool's line (an arc when it bends: blocks.py)
    (function walk(n) {
      if (!n) return;
      if (n.kind === 'edge' || n.kind === 'pool') {
        const x = ox + (n.kx + N / 2 + 0.5) * cell, y = oy + (n.ky + N / 2 + 0.5) * cell, r = (Math.round(n.value) + 0.5) * cell, col = PLANE_COL[n.index] || '200,200,200';
        ctx.strokeStyle = `rgba(${col},0.5)`; ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
        if (n.kind === 'edge') { ctx.strokeStyle = `rgb(${col})`; ctx.lineWidth = 2; arc(x, y, n.angle || 0, r, cell, n.bend, 1); ctx.lineWidth = 1; edges++; }
      }
      (n.children || []).forEach(walk);
    })(d.trees && d.trees.response);
    ctx.fillStyle = 'rgba(124, 255, 160, 0.85)'; ctx.font = '10px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(edges ? `lines: ${edges} on its eye` : 'lines: none yet', ox, oy + side + 3);
    if (hz == null) { ctx.fillStyle = 'rgba(124, 255, 160, 0.85)'; ctx.font = '12px monospace'; ctx.textAlign = 'center'; ctx.fillText(note, (W - side) / 2, H / 2); }
    $('space-cap').innerHTML = note;
    const ts = d.scores && d.scores.terrain;
    $('space-cap').innerHTML += ts ? ` &middot; felt vs taught nearness: off by ${ts.mae.toFixed(2)} &plusmn; ${ts.mae_se.toFixed(2)} of 1 (${ts.n} looks)` : '';
    // who it knows, as a tally by kind (most first): "8 cats, 6 cars, 2 persons" --
    // one line however many it knows (each individual's own record was a list
    // of dozens once vehicles became hosts)
    const cx = d.cortex, plural = (n, w) => n === 1 || w === 'sheep' ? w : /(s|sh|ch|x)$/.test(w) ? w + 'es' : w + 's';
    const kinds = {}; ((cx && cx.known) || []).forEach(k => { kinds[k.name] = (kinds[k.name] || 0) + 1; });
    const tally = Object.entries(kinds).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([w, n]) => `${n} ${plural(n, w)}`).join(', ');
    $('cortex-cap').innerHTML = !cx ? '' : `<b>who it knows</b>: ${tally ? esc(tally) : 'nobody yet'}`;
  }
  // Its sense of space is not ours to steer: it shows what its eye sees, from
  // where its eye is (a 2026-09-29 decision) -- nothing here moves it.
  function spaceKick() {  // drawn on the next frame, only while on screen and the tab is visible
    if (SP3.loop) return;
    SP3.loop = true;
    requestAnimationFrame(() => {
      const c = $('space'), r = c && c.getBoundingClientRect();
      if (SP3.dirty && SP3.d && r && r.bottom > 0 && r.top < innerHeight && !document.hidden) { SP3.dirty = false; drawSpace(SP3.d); }
      SP3.loop = false;
    });
  }
  function spaceData(d) { SP3.d = d; SP3.dirty = true; spaceKick(); }
  addEventListener('scroll', () => { if (SP3.dirty) spaceKick(); }, { passive: true });
  document.addEventListener('visibilitychange', () => { if (SP3.dirty) spaceKick(); });
  // The retina panel runs on the same replay clock as the picture and the
  // visual field: its gaze where the replayed path has it, and its n x n
  // retina rebuilt here from the frame on screen with the same averaging its
  // eye does (grey: 0.299 R + 0.587 G + 0.114 B, like its own). A
  // reconstruction -- from the replay JPEG, not its exact frames; without
  // frames (a file source) it falls back to its retina at the end of its run.
  const LOOK = { key: null, cells: null };
  const MOUTH_SIDE = 22 / 64 / 2;  // of the frame's height -- prey.MOUTH_SIDE
  const lookSrc = document.createElement('canvas');
  function rebuildRetina(img, cx, cy, f, N) {
    const iw = img.naturalWidth, ih = img.naturalHeight;
    if (lookSrc.width !== iw || lookSrc.height !== ih) { lookSrc.width = iw; lookSrc.height = ih; }
    const c2 = lookSrc.getContext('2d', { willReadFrequently: true });
    c2.drawImage(img, 0, 0);
    const gw = Math.max(N, Math.round(f * ih)), gh = gw;  // square receptors: its side is f of the height
    const x0 = Math.round(cx * iw - gw / 2), y0 = Math.round(cy * ih - gh / 2);
    const sx0 = Math.max(0, x0), sy0 = Math.max(0, y0), sx1 = Math.min(iw, x0 + gw), sy1 = Math.min(ih, y0 + gh);
    const px = sx1 > sx0 && sy1 > sy0 ? c2.getImageData(sx0, sy0, sx1 - sx0, sy1 - sy0).data : null;
    const cells = [];
    for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) {
      const ya = y0 + Math.floor(r * gh / N), yb = y0 + Math.floor((r + 1) * gh / N);
      const xa = x0 + Math.floor(q * gw / N), xb = x0 + Math.floor((q + 1) * gw / N);
      let R = 0, G = 0, B = 0, n = 0;
      for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) {
        n++;  // past the frame's edge counts as black, as its eye sees it
        if (px && x >= sx0 && x < sx1 && y >= sy0 && y < sy1) { const k = ((y - sy0) * (sx1 - sx0) + (x - sx0)) * 4; R += px[k]; G += px[k + 1]; B += px[k + 2]; }
      }
      cells.push(n ? [R / n, G / n, B / n] : [0, 0, 0]);
    }
    return cells;
  }
  function drawLook(now) {
    requestAnimationFrame(drawLook);
    const d = D;
    if (!d || !d.frame_w) return;
    let R = replayAt(d, now, CLK, REPLAY_FPS);
    if (typeof F !== 'undefined') R = hudFor(d, R, F) || R;  // its gaze on the picture actually shown
    const fmax = d.max_fraction || 0.6, N = d.receptors || 12;
    const C = Math.min(N, d.cones ?? N), c0 = (N - C) >> 1;  // its cones: the central C x C (rods around them)
    const img = $('cam');
    let cells = null;
    if (typeof F !== 'undefined' && F.shown != null && img.naturalWidth) {
      const key = F.shown + '|' + R.cx + '|' + R.cy + '|' + R.f + '|' + N;
      if (LOOK.key !== key) { LOOK.cells = rebuildRetina(img, R.cx, R.cy, R.f, N); LOOK.key = key; }
      cells = LOOK.cells;
    }
    // Without frames: its retina and gaze at the end of its latest run.
    const f = cells ? R.f : (d.fovea_fraction || 0.35);
    const gx = cells ? R.cx : (d.fovea_cx ?? 0.5), gy = cells ? R.cy : (d.fovea_cy ?? 0.5);
    const [cw, ch] = crop(d, f);
    const c = $('look'), W = Math.max(160, fitWidth($('look-panel'), quadAspect())), H = Math.round(W * quadAspect());
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d'), s = W / d.frame_w, cx = gx * W, cy = gy * H;
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#2a3c4c'; ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
    ctx.setLineDash([6, 5]); ctx.strokeRect(cx - fmax * H / 2, cy - fmax * H / 2, fmax * H, fmax * H); ctx.setLineDash([]);
    const lw = f * H, lh = f * H, lx = cx - lw / 2, ly = cy - lh / 2;  // square: its side is f of the height
    const shut = cells && R.asleep;  // asleep: its eyes are shut, the gaze sees nothing
    if (shut) {
      ctx.fillStyle = '#000'; ctx.fillRect(lx, ly, lw, lh);
      ctx.fillStyle = '#a8c'; ctx.font = '12px monospace'; ctx.textAlign = 'center';
      ctx.fillText('eyes shut', cx, cy + 4); ctx.textAlign = 'left';
    } else if (cells) {
      const colour = (d.colour_channels ?? 0) > 0;
      for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) {
        const [Rv, Gv, Bv] = cells[r * N + q];
        const grey = Math.round(0.299 * Rv + 0.587 * Gv + 0.114 * Bv);
        const cone = r >= c0 && r < c0 + C && q >= c0 && q < c0 + C;
        ctx.fillStyle = colour && cone ? `rgb(${Math.round(Rv)},${Math.round(Gv)},${Math.round(Bv)})` : `rgb(${grey},${grey},${grey})`;
        const xa = Math.round(lx + q * lw / N), xb = Math.round(lx + (q + 1) * lw / N);
        const ya = Math.round(ly + r * lh / N), yb = Math.round(ly + (r + 1) * lh / N);
        ctx.fillRect(xa, ya, xb - xa, yb - ya);
      }
    } else if (d.grid) {
      if (d.colour_grid && d.colour_grid.length >= d.grid.length) drawColourGrid(ctx, d.grid, d.colour_grid, d.grid_shape, lx, ly, lw, lh);
      else drawGrid(ctx, d.grid, d.grid_shape, lx, ly, lw, lh);
    }
    if (C > 0 && C < N) {  // the cone patch
      ctx.strokeStyle = 'rgba(255, 190, 90, 0.8)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.strokeRect(lx + c0 * lw / N, ly + c0 * lh / N, C * lw / N, C * lh / N); ctx.setLineDash([]);
    }
    // Eating, in the other panels' prey pink: its mouth -- the centre of its
    // gaze, where a host under it is a bite -- glows as much as it bites and
    // pulses while it chews; each new bite sends a ring out from it (the
    // target lock's snap, here). Drawn only with the replay's own frames.
    const eat = cells && !shut ? R.eat : 0;
    if (eat > 0.01 && !LOOK.eating) LOOK.biteT = now;
    LOOK.eating = eat > 0.01;
    if (LOOK.eating) {
      const ms = MOUTH_SIDE * H, mx = cx - ms / 2, my = cy - ms / 2;  // its mouth, at its real size, at the gaze centre (prey.py)
      const pulse = 0.5 + 0.5 * Math.sin(now / 110);
      ctx.fillStyle = `rgba(255, 95, 162, ${(0.12 + 0.28 * eat) * (0.6 + 0.4 * pulse)})`; ctx.fillRect(mx, my, ms, ms);
      ctx.strokeStyle = `rgba(255, 95, 162, ${0.6 + 0.4 * pulse})`; ctx.lineWidth = 2; ctx.strokeRect(mx, my, ms, ms);
      ctx.fillStyle = 'rgba(255, 95, 162, 0.95)'; ctx.font = 'bold 14px monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText(`ABSORBING \u00b7 JĪNG ${(eat * 100).toFixed(0)}%`, W - 8, 6); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    } else if (cells && !shut && (R.snack || 0) > 0.01) {  // nectar: a surprise snack, sipped
      ctx.fillStyle = 'rgba(200, 136, 255, 0.95)'; ctx.font = 'bold 14px monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText('SIPPING \u00b7 QÌ', W - 8, 6); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    }
    const biteAge = now - (LOOK.biteT ?? -1e9);
    if (biteAge < 400) {
      const k = biteAge / 400, e = 1 - Math.pow(1 - k, 3), rs = MOUTH_SIDE * H + e * Math.max(lw, lh) * 1.2;
      ctx.strokeStyle = `rgba(255, 95, 162, ${0.9 * (1 - k)})`; ctx.lineWidth = 3;
      ctx.strokeRect(cx - rs / 2, cy - rs / 2, rs, rs);
    }
    ctx.strokeStyle = shut ? '#a8c' : LOOK.eating ? '#ff5fa2' : '#ffb4a6'; ctx.lineWidth = 2; ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = '#9a6f67'; ctx.font = '11px monospace';
    ctx.fillText(shut ? 'asleep -- the gaze sees nothing' : cells ? 'its whole visual field -- retina rebuilt from the frame on screen (a reconstruction)' : 'its whole visual field -- retina at the end of its latest run', 6, 14);
    $('look-px').textContent = `${N}x${N} receptors, ${C}x${C} cones`;
    $('field-px').textContent = `${(d.world_grid_shape || [12, 12])[1]}x${(d.world_grid_shape || [12, 12])[0]} receptors`;
    const zoomNow = N / 64 / f;  // its lens now: receptors x pitch over the side it covers
    $('look-scale').textContent = `${f.toFixed(2)} of the frame at (${gx.toFixed(2)}, ${gy.toFixed(2)})${zoomNow > 1.01 ? ` \u00b7 zoom ${zoomNow.toFixed(1)}x` : ''}`;
  }
  requestAnimationFrame(drawLook);

  function gauge(name, v, color, note) {
    const x = Math.max(0, Math.min(1, v || 0));
    return `<div class="gauge"><span>${name}</span><div class="track"><div class="fill" style="width:${(x * 100).toFixed(1)}%;background:${color}"></div></div><span class="v">${(v || 0).toFixed(2)}</span></div>` + (note && note.startsWith('=') ? `<div class="note">${note.slice(1)}</div>` : '');
  }
  function spark(id, series, color, label, band) {
    const c = $(id); if (!c) return;
    c.width = Math.max(200, Math.floor(innerWidth(c.parentElement)));
    const ctx = c.getContext('2d'), W = c.width, H = c.height;
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#1c2a36'; [0, 0.5, 1].forEach(v => { const y = (1 - v) * (H - 16) + 2; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); });
    if (band && band.length > 1) {
      ctx.fillStyle = 'rgba(200, 136, 255, 0.22)';
      band.forEach((v, k) => { if (v >= 0.5) { const x = k / (band.length - 1) * W; ctx.fillRect(x - W / (band.length - 1) / 2, 0, W / (band.length - 1) + 1, H - 14); } });
    }
    if (series && series.length > 1) {
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
      series.forEach((v, k) => { const x = k / (series.length - 1) * W, y = (1 - Math.max(0, Math.min(1, v))) * (H - 16) + 2; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    ctx.fillStyle = '#9a6f67'; ctx.font = '10px monospace'; ctx.fillText(label + ' over its latest run (0..1, start -> end)', 2, H - 3);
  }
  function drawBody(d) {
    const b = d.body_now || d.body || {};
    $('gauges').innerHTML =
      `<div class="cap" style="margin:0 0 4px">${d.life && d.life.encysted ? '<b style="color:#e8c170">IN ITS CYST</b> (starving in a world with no food; a developed body waits, unchanged, until a host comes within reach)' : d.life && d.life.torpid ? '<b style="color:#9cf">TORPID</b> (no world in its eyes: hibernating at 5% of its burn)' : (b.asleep || 0) >= 0.5 ? '<b style="color:#c8f">ASLEEP</b>' : '<b style="color:var(--green)">awake</b>'}`
      // its sense of time (a 2026-09-29 panel): a look is its moment -- how many it takes a second, beside ours (Rayner 1998: ~200-300 ms fixations)
      + (d.senses && d.senses.pace_s ? ` &middot; its time: a look every ${d.senses.pace_s.toFixed(2)} s (${(1 / d.senses.pace_s).toFixed(1)} a second; people fixate ~3&ndash;4)` : '') + `</div>` +
      gauge('sugar', b.energy, '#ffe2d6', 'pays for everything; ~10 min of waking burn') +
      gauge('gut', b.gut, '#e8c170', 'what it ate, digesting into sugar over minutes; full = cannot eat more') +
      gauge('glycogen', b.glycogen, '#6d9', '~6 h; fills first when fed, released fast for waking and bursts') +
      gauge('fat', b.reserve, '#2c9', '~3 days; made from a real surplus, burned slowly and only aerobically -- never for a burst or the brain') +
      gauge('protein', b.protein ?? 1, '#e0909c', 'for eggs: only essence fills it; spent over ~3 days (a gonotrophic cycle)') +
      gauge('phosphagen', b.phosphagen, '#fd6', 'the first ~10 s of a burst; refills in ~30 s') +
      gauge('ketosis', b.ketone, '#c9f', 'once glycogen is gone, fat feeds up to 2/3 of the brain as ketones') +
      gauge('wasting', b.wasting, '#f55', 'tissue burned for a brain with no sugar: Kenyon cells, hidden units and the outer rings of its eye go with it; at full, it dies') +
      (d.life ? gauge('toward an egg', d.life.egg_progress, '#9cf', `its reproduction buffer: ${Math.round(100 * (1 - d.life.kappa))}% of what it digests (kappa ${d.life.kappa.toFixed(2)}); an egg costs what a newborn is made of`) +
                gauge('age', d.life.age, '#b9a', 'the damage its own burning has done to its tissue; at full, it dies of age') : '') +
      gauge('sleep pressure', b.sleep_pressure, '#c8f', 'builds while awake, clears asleep; tiredness costs it half of what it catches at full pressure') +
      gauge('hunger', b.hunger, '#f6a', 'sugar and gut together') +
      gauge('search', b.search, '#fd4', 'urge to look around, driven by hunger') + gauge('curiosity', b.curiosity, '#c8f', 'appetite for something new') +
      gauge('arousal', b.arousal, '#ffb4a6', 'from motion anywhere in the field') + gauge('threat', b.threat, '#f44', 'from something dark approaching') +
      gauge('fatigue', b.fatigue, '#f90', 'from forceful eye movement') +
      gauge('anaerobic debt', b.debt, '#f60', 'from bursts beyond what it can sustain; felt as fatigue, repaid over hours') +
      `<div class="cap" style="margin-top:4px">tempo <b style="color:var(--cyan)">${d.pace ? ((d.frames_per_second || 15) / d.pace).toFixed(1) : '--'}</b> gazes/s (resting ${d.pace_accepted ? ((d.frames_per_second || 15) / d.pace_accepted).toFixed(1) : '--'})</div>` +
      gauge('metabolism', b.metabolic_rate ?? 1, '#fd4', 'acclimatizes to its tempo over ~2 min (1 = gazing every frame)') +
      (d.flinch ? `<div class="cap" style="margin-top:4px">flinch: <b style="color:var(--cyan)">${d.flinch.events}</b> approaches in its latest run, reacted to <b style="color:var(--cyan)">${d.flinch.reacted}</b>` + (d.flinch.mean_latency_frames !== null ? `, on average ${(d.flinch.mean_latency_frames / (d.frames_per_second || 15) * 1000).toFixed(0)} ms after onset` : '') + '</div>' : '');
    spark('energy-trace', d.energy_series, '#ffe2d6', 'sugar (purple band: asleep)', d.sleep_series);
    const boutNote = kind => { const bt = d.bouts && d.bouts[kind]; if (!bt) return ''; return bt.fit ? `=1 ${kind} = looks < ${bt.fit.criterion_s.toFixed(1)} s apart` : `=${kind} size: calibrating`; };
    $('prey-gauge').innerHTML = gauge('prey (meals)', d.mean_prey, '#ff5fa2', boutNote('meal'));
    spark('prey-trace', d.prey_series, '#ff5fa2', 'prey eaten');
    $('food-gauge').innerHTML = gauge('surprise (snacks)', d.mean_food, '#c8f', boutNote('snack'));
    spark('food-trace', d.food_series, '#c8f', 'surprise');
    const m = d.movement || {};
    $('movement').innerHTML =
      gauge('fixating', m.fixate, '#9a6f67', 'share of frames still') + gauge('gliding', m.glide, '#ffe2d6', 'slow, smooth') +
      gauge('saccades', m.saccade, '#f90', 'fast jumps') + gauge('scanning', m.scan_while_still, '#ffb4a6', 'gliding while the world is still') +
      `<div class="gauge"><span>tracking</span><span class="cap" style="margin:0">${m.tracking === null || m.tracking === undefined ? 'not enough movement in the room this run' : 'correlation ' + m.tracking.toFixed(2) + ' with where motion was'}</span><span></span></div>`;
  }

  // The whole recurrent brain.
  // Randomize: its brain drawn afresh as a founder's (run_vision.py), at its
  // next generation. Two clicks within 4 s (no browser dialog): the first arms it.
  // New random founder (the Amnesia request): randomized AND its eye, body, memories and traits back to a fresh
  // install's (tools/reset_founder.py; everything is kept in a backup).
  [['randomize', 'click again to draw a new brain', 'a new brain at its next generation'],
   ['amnesia', 'click again: a new random founder, everything it was kept in a backup', 'a new founder at its next generation'],
   ['reset-founder', 'click again: back to its founder as it was at birth (a backup is kept)', 'its founder at its next generation']].forEach(([id, arm, done]) => {
    const a = $(id), note = $(id + '-note'); if (!a) return;
    let armed = 0;
    a.addEventListener('click', ev => {
      ev.preventDefault();
      if (Date.now() - armed > 4000) { armed = Date.now(); note.textContent = arm; return; }
      armed = 0; note.textContent = 'asking...';
      fetch('/' + id, { method: 'POST', headers: { 'X-Cambrian': '1' } })
        .then(r => r.ok ? (note.textContent = done) : r.json().then(j => { note.textContent = j.error || 'refused'; }, () => { note.textContent = 'refused'; }))
        .catch(() => { note.textContent = 'no answer'; });
    });
  });
  // Its organism files (tools/organism_file.py), all in THIS ecohost's
  // Games/Lifeforms/Cambrioids folder: Save puts it there; Load (a list of
  // them, two clicks) takes one in its place at its next generation, this one
  // kept in a backup; Copy from pulls one of a hive peer's there; Download and
  // Upload move one between that folder and the device you are viewing on.
  (() => {
    const H = { 'X-Cambrian': '1' }, say = (id, t) => { const e = $(id); if (e) e.textContent = t; };
    const when = t => t ? new Date(t * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + new Date(t * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : '';
    const label = e => e.damaged ? `${e.file} (damaged)` : `${e.ecohost || '?'} gen ${e.generation ?? '?'} · ${when(e.saved_at)}`;
    const fill = (sel, list, empty) => {
      sel.innerHTML = list.length ? '' : `<option value="">${empty}</option>`;
      list.forEach(e => { const o = document.createElement('option'); o.value = e.file; o.textContent = label(e); o.title = e.file; sel.appendChild(o); });
    };
    const ll = $('load-list');
    const refresh = () => ll && fetch('/organism/saves', { headers: H }).then(r => r.json()).then(l => fill(ll, l, 'no saves yet')).catch(() => {});
    refresh();
    const post = (url, body) => fetch(url, { method: 'POST', headers: H, body }).then(r => r.json());
    // Save
    const sh = $('save-here');
    if (sh) sh.addEventListener('click', ev => {
      ev.preventDefault(); say('save-here-note', 'saving...');
      post('/save-here').then(j => { say('save-here-note', j.ok ? 'saved' : `refused: ${j.error}`); refresh(); }, () => say('save-here-note', 'no answer'));
    });
    // Load (from the list)
    let armed = 0;
    const lo = $('load-organism');
    if (lo) lo.addEventListener('click', ev => {
      ev.preventDefault();
      const f = ll && ll.value;
      if (!f) { say('load-organism-note', 'choose one of its saves first'); return; }
      if (Date.now() - armed > 4000) { armed = Date.now(); say('load-organism-note', 'click again: it takes this one’s place (a backup is kept)'); return; }
      armed = 0; say('load-organism-note', 'checking...');
      post('/load-saved?file=' + encodeURIComponent(f)).then(j => say('load-organism-note', j.ok ? `${j.name}: at its next generation` : `refused: ${j.error}`), () => say('load-organism-note', 'no answer'));
    });
    // Copy from (a hive peer)
    const pl = $('peer-list'), ps = $('peer-saves'), ch = $('copy-here');
    let asked = false;
    const askPeers = () => {
      if (asked) return;
      asked = true; say('copy-note', 'finding ecohosts...');
      fetch('/organism/peers', { headers: H }).then(r => r.json()).then(j => {
        say('copy-note', !j.hive ? 'solo: this ecohost is not in the hive' : j.peers.length ? '' : 'no other ecohost answers');
        j.peers.forEach(h => { const o = document.createElement('option'); o.value = h; o.textContent = h; pl.appendChild(o); });
      }, () => { asked = false; say('copy-note', 'no answer'); });
    };
    if (pl) { pl.addEventListener('pointerdown', askPeers); pl.addEventListener('focus', askPeers); }
    if (pl) pl.addEventListener('change', () => {
      ps.hidden = ch.hidden = !pl.value;
      if (!pl.value) return;
      say('copy-note', 'asking...');
      fetch('/organism/peer-saves?host=' + encodeURIComponent(pl.value), { headers: H }).then(r => r.json())
        .then(j => { if (j.error) { say('copy-note', j.error); return; } fill(ps, j.saves, 'no saves there'); say('copy-note', ''); }, () => say('copy-note', 'no answer'));
    });
    if (ch) ch.addEventListener('click', ev => {
      ev.preventDefault();
      if (!ps.value) return;
      say('copy-note', 'copying...');
      post('/copy-from?host=' + encodeURIComponent(pl.value) + '&file=' + encodeURIComponent(ps.value))
        .then(j => { say('copy-note', j.ok ? `copied: ${j.file} (now under Load)` : `refused: ${j.error}`); refresh(); }, () => say('copy-note', 'no answer'));
    });
    // Download (to the device you are viewing on)
    const dl = $('save-organism');
    if (dl) dl.addEventListener('click', ev => {
      ev.preventDefault(); say('save-organism-note', 'saving...');
      fetch('/organism/save', { headers: H }).then(r => {
        if (!r.ok) return r.json().then(j => say('save-organism-note', j.error || 'refused'), () => say('save-organism-note', 'refused'));
        const m = /filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') || ''), name = m ? m[1] : 'organism.cambrioid';
        return r.blob().then(b => {
          const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = name;
          document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 10000);
          say('save-organism-note', 'downloaded');
        });
      }).catch(() => say('save-organism-note', 'no answer'));
    });
    // Upload (from the device you are viewing on, into its saves)
    const up = $('upload-organism'), fi = $('load-file');
    if (up && fi) {
      up.addEventListener('click', ev => { ev.preventDefault(); fi.value = ''; fi.click(); });
      fi.addEventListener('change', () => {
        const f = fi.files && fi.files[0];
        if (!f) return;
        say('upload-note', 'checking...');
        f.arrayBuffer().then(buf => post('/upload?name=' + encodeURIComponent(f.name), buf))
          .then(j => { say('upload-note', j.ok ? `kept: ${j.file} (now under Load)` : `refused: ${j.error}`); refresh(); }, () => say('upload-note', 'no answer'));
      });
    }
  })();
  // A card's drawing, guarded: an error is recorded on <html data-js-errors / data-js-last>
  // (as an uncaught one is) with where it came from, and the other cards still draw.
  function guarded(fn, ...args) {
    try { fn(...args); }
    catch (e) {
      const r = document.documentElement;
      r.dataset.jsErrors = (+(r.dataset.jsErrors || 0) + 1) + '';
      r.dataset.jsLast = `${fn.name || 'card'}: ${String(e && e.message).slice(0, 160)} @${String(e && e.stack || '').split(/\n/)[1] || ''}`.slice(0, 300);
    }
  }
  // Its nervous system as one flatmap (a 2026-09-30 panel -- Shneiderman,
  // Holten, Tufte & Bertin, Swanson): its sense organs on the left, each a
  // block with a port for every input it gives the brain (the mushroom body
  // among them, its Kenyon cells drawn); their wires bundled per organ and
  // unit (thickness: the summed weight; colour: the net sign; hover an organ
  // for its real wires, one per input); its units and their recurrence; the
  // stream through its stacked layers -- each layer its own units, reading
  // the lanes below through W, keeping U, adding back through its gate --
  // then its outputs, read from the top of the stack, and what they drive.
  // Dashed: what closes through the world or the body (its eye muscles move
  // what its eye sees; its eye feeds the perception tree and the mushroom
  // body; what it eats or suffers teaches the mushroom body; a grown channel
  // comes back as an input). Nothing that exists is left off; an input no
  // organ claims goes in "other". Matte: one colour per sign, alpha by
  // strength. 3D is the same map with depth, turned.
  const NS_ORGANS = [
    ['eye & V4', ['light', 'motion', 'flow x', 'flow y', 'loom', 'gaze x', 'gaze y', 'eye size', 'motion dx', 'motion dy', 'eye vx', 'eye vy', 'field light', 'light trend', 'texture contrast', 'texture fineness', 'texture grain']],
    ['perception tree', ['tree']],
    ['mushroom body', ['food value', 'danger', 'archetype 1', 'archetype 2', 'archetype 3', 'archetype 4', 'felt nearness', 'uncertainty']],
    ['detector (the frame)', ['prey scent', 'prey dir x', 'prey dir y', 'plant scent', 'plant dir x', 'plant dir y', 'intruder', 'host vx', 'host vy']],
    ['colliculus & surprise', ['collicular dx', 'collicular dy', 'collicular strength', 'mismatch', 'mismatch dx', 'mismatch dy']],
    ['memory & place', ['recalled value', 'recalled dx', 'recalled dy', 'place dx', 'place dy', 'place value', 'map value']],
    ['geometry organs', ['horizon', 'ground near', 'parallax', 'camera moving', 'terrain', 'nearness', 'contact', 'turning', 'tilting', 'heading sin', 'heading cos', 'speed', 'acceleration', 'riding']],
    ['body', ['sugar', 'arousal', 'threat', 'search', 'hunger', 'curiosity', 'gut', 'reserve', 'sleep pressure', 'asleep', 'protein']],
    ['self-monitoring', ['own pace', 'missed', 'strangeness']],
    ['road organ', ['road heading', 'road curve', 'road crest', 'road offset', 'road sure']],
  ];
  const nsIn = (br, i) => i < INPUT_NAMES.length ? INPUT_NAMES[i] : channelName(br, i - INPUT_NAMES.length, 'in');
  const nsOut = (br, o) => o < OUTPUT_NAMES.length ? OUTPUT_NAMES[o] : channelName(br, o - OUTPUT_NAMES.length, 'out');
  function nsGroups(br) {
    const nIn = br.weights_ih[0].length, used = new Set(), groups = [];
    NS_ORGANS.forEach(([name, names]) => {
      const idx = names.map(n => INPUT_NAMES.indexOf(n)).filter(i => i >= 0 && i < nIn);
      idx.forEach(i => used.add(i));
      if (idx.length) groups.push({ name, idx, mb: name === 'mushroom body', eye: name === 'eye & V4', tree: name === 'perception tree', body: name === 'body' });
    });
    const loops = [], other = [];
    for (let i = 0; i < nIn; i++) if (!used.has(i)) (i >= INPUT_NAMES.length ? loops : other).push(i);
    if (other.length) groups.push({ name: 'other', idx: other });
    if (loops.length) groups.push({ name: 'its loops (back in)', idx: loops, loops: true });
    return groups;
  }
  function nsLayout(d, W, H) {
    const br = d.brain, groups = nsGroups(br), nH = br.weights_ih.length, nOut = br.weights_ho.length, ls = br.layers || [];
    const top = 34, bot = H - 30, gap = 6, ox = 30;
    const ow = Math.max(104, Math.min(190, W * 0.17));
    const hasMB = !!(d.mb && d.mb.n);
    const weight = g => 1 + Math.sqrt(g.idx.length) + (g.mb && hasMB ? 3 : 0);
    const tot = groups.reduce((t, g) => t + weight(g), 0), avail = bot - top - gap * (groups.length - 1);
    let y = top;
    groups.forEach(g => {
      g.h = Math.max(18, avail * weight(g) / tot); g.x = ox; g.y = y; g.w = ow; y += g.h + gap;
      g.ports = g.idx.map((i, k) => ({ i, x: ox + ow, y: g.y + 15 + (g.h - 19) * (k + 0.5) / g.idx.length }));
    });
    const xo = W - Math.max(92, Math.min(170, W * 0.15));
    const xu = ox + ow + Math.max(70, (xo - ox - ow) * 0.24);
    const xend = xo - Math.max(46, (xo - xu) * 0.16);
    const yAt = (k, n) => n === 1 ? (top + bot) / 2 : top + 12 + k * (bot - top - 24) / (n - 1);
    const units = Array.from({ length: nH }, (_, h) => ({ x: xu, y: yAt(h, nH) }));
    const layers = ls.map((L, k) => units.map(u => ({ x: xu + (k + 1) * (xend - xu) / (ls.length + 1), y: u.y })));
    const outs = Array.from({ length: nOut }, (_, o) => ({ x: xo, y: yAt(o, nOut) }));
    return { groups, units, layers, outs, xu, xo, xend, top, bot, ox, ow };
  }
  function drawNervous(d, three) {
    const c = $('brain'), br = d && d.brain; if (!c || !br) return;
    const W = c.width, H = c.height, ctx = c.getContext('2d');
    const { groups, units, layers, outs, xu, xo, xend, top, bot, ox, ow } = nsLayout(d, W, H);
    const nH = units.length, nOut = outs.length, ls = br.layers || [], lc = d.brain_layers || [], hid = d.brain_hidden || [];
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    // where a point of the map lands: 2D, the map zoomed and panned; 3D, the map
    // given depth (units and their lanes fanned in depth, organs a little), turned
    const Dz = Math.min(W, H) * 0.55;
    const zu = h => nH > 1 ? (h / (nH - 1) - 0.5) * Dz : 0, zo = o => nOut > 1 ? (o / (nOut - 1) - 0.5) * Dz * 0.6 : 0;
    const zg = gi => groups.length > 1 ? (gi / (groups.length - 1) - 0.5) * Dz * 0.35 : 0;
    let P;
    if (three) {
      const cy = Math.cos(B3.yaw), sy = Math.sin(B3.yaw), cp = Math.cos(B3.pitch), sp = Math.sin(B3.pitch), F = 2.2 * Math.max(W, H), s = 0.82 * B3.zoom;
      P = (x, y, z) => {
        const X = (x - W / 2) * s, Y = (y - H / 2) * s, Z = (z || 0) * s;
        const x1 = X * cy + Z * sy, z1 = -X * sy + Z * cy, y2 = Y * cp - z1 * sp, z2 = Y * sp + z1 * cp, w = F / (F + z2);
        return [W / 2 + x1 * w, H / 2 + y2 * w, w, z2];
      };
    } else P = (x, y) => [x * BZ.k + BZ.x, y * BZ.k + BZ.y, BZ.k, 0];
    const hov = B3.hover, pts = [];
    const sgn = (w, a) => (w >= 0 ? `rgba(127,212,255,${a})` : `rgba(255,153,0,${a})`);
    const curve = (A, B, w, rel, dim, width) => {
      const a = P(...A), b = P(...B), s = Math.min(1, rel), mx = (a[0] + b[0]) / 2;
      ctx.strokeStyle = sgn(w, (0.05 + 0.6 * s) * (dim ? 0.18 : 1)); ctx.lineWidth = Math.max(0.5, (width || 0.5 + 2 * s) * Math.min(2, (a[2] + b[2]) / 2));
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]); ctx.stroke();
    };
    const arc = (A, B, w, rel, side, dim) => {
      const a = P(...A), b = P(...B), s = Math.min(1, rel), off = side * (10 + 0.22 * Math.hypot(a[0] - b[0], a[1] - b[1]));
      ctx.strokeStyle = sgn(w, (0.05 + 0.5 * s) * (dim ? 0.18 : 1)); ctx.lineWidth = Math.max(0.5, (0.5 + 1.6 * s) * Math.min(2, a[2]));
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.quadraticCurveTo((a[0] + b[0]) / 2 + off, (a[1] + b[1]) / 2, b[0], b[1]); ctx.stroke();
    };
    const dashed = (pathPts, label, lx, ly) => {
      const q = pathPts.map(p => P(...p));
      ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(200,190,180,0.45)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]); q.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.stroke();
      const e = q[q.length - 1], f = q[q.length - 2], an = Math.atan2(e[1] - f[1], e[0] - f[0]);  // an arrowhead: which way it flows
      ctx.setLineDash([]); ctx.fillStyle = 'rgba(200,190,180,0.6)'; ctx.beginPath(); ctx.moveTo(e[0], e[1]);
      ctx.lineTo(e[0] - 6 * Math.cos(an - 0.45), e[1] - 6 * Math.sin(an - 0.45)); ctx.lineTo(e[0] - 6 * Math.cos(an + 0.45), e[1] - 6 * Math.sin(an + 0.45)); ctx.fill();
      if (label) { const L = P(lx, ly, 0); ctx.font = '9px monospace'; ctx.fillStyle = 'rgba(200,190,180,0.7)'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(label, L[0], L[1]); }
      ctx.restore();
    };
    const node = (A, r, fill, stroke, lw, dash) => {
      const a = P(...A), rr = Math.max(1.5, r * Math.min(2, a[2]));
      ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1;
      if (dash) ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.arc(a[0], a[1], rr, 0, 7); if (fill) ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
      return a;
    };
    const quad = (x, y, w, h, z, fill, stroke) => {
      const q = [P(x, y, z), P(x + w, y, z), P(x + w, y + h, z), P(x, y + h, z)];
      ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]); q.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.closePath(); ctx.fill(); ctx.stroke();
    };
    const text = (s, x, y, z, font, col, align, base) => { const a = P(x, y, z); ctx.font = font; ctx.fillStyle = col; ctx.textAlign = align || 'left'; ctx.textBaseline = base || 'middle'; ctx.fillText(s, a[0], a[1]); return a; };

    // what closes through the world or the body, behind everything
    const eye = groups.find(g => g.eye), mbG = groups.find(g => g.mb), treeG = groups.find(g => g.tree), bodyG = groups.find(g => g.body);
    const yTop = top - 16, xl = ox - 10;
    if (eye && nOut >= 3) dashed([[xo, outs[0].y, zo(0)], [xo + 0, yTop, 0], [eye.x + eye.w / 2, yTop, 0], [eye.x + eye.w / 2, eye.y, zg(groups.indexOf(eye))]],
                                 'pan, tilt, zoom move its gaze: what its eye sees next', eye.x + eye.w / 2 + 6, yTop - 1);
    if (eye && treeG) dashed([[eye.x, eye.y + eye.h * 0.7, zg(groups.indexOf(eye))], [xl, eye.y + eye.h * 0.7, 0], [xl, treeG.y + treeG.h / 2, 0], [treeG.x, treeG.y + treeG.h / 2, zg(groups.indexOf(treeG))]]);
    if (eye && mbG) dashed([[eye.x, eye.y + eye.h * 0.85, zg(groups.indexOf(eye))], [xl - 7, eye.y + eye.h * 0.85, 0], [xl - 7, mbG.y + mbG.h * 0.3, 0], [mbG.x, mbG.y + mbG.h * 0.3, zg(groups.indexOf(mbG))]]);
    if (bodyG && mbG) dashed([[bodyG.x, bodyG.y + bodyG.h / 2, zg(groups.indexOf(bodyG))], [xl - 14, bodyG.y + bodyG.h / 2, 0], [xl - 14, mbG.y + mbG.h * 0.7, 0], [mbG.x, mbG.y + mbG.h * 0.7, zg(groups.indexOf(mbG))]]);
    const loopsG = groups.find(g => g.loops);
    if (loopsG) (br.channels || []).forEach((ch, k) => {
      const o = OUTPUT_NAMES.length + k, port = loopsG.ports.find(p => p.i === INPUT_NAMES.length + k); if (!outs[o] || !port) return;
      const yb = H - 4 - 4 * k;
      dashed([[xo, outs[o].y, zo(o)], [xo + 12, outs[o].y, zo(o)], [xo + 12, yb, 0], [port.x + 10, yb, 0], [port.x + 10, port.y, 0], [port.x, port.y, zg(groups.indexOf(loopsG))]]);
    });

    // organs to units: bundled (open: its real wires)
    const wih = br.weights_ih;
    let bmax = 1e-9, wmax = 1e-9;
    wih.forEach(r => r.forEach(v => { wmax = Math.max(wmax, Math.abs(v)); }));
    const bund = groups.map(g => units.map((u, h) => { let s = 0, a = 0; g.idx.forEach(i => { s += wih[h][i]; a += Math.abs(wih[h][i]); }); bmax = Math.max(bmax, a); return [s, a]; }));
    groups.forEach((g, gi) => {
      const open = hov === 'g' + gi;
      units.forEach((u, h) => {
        const dim = hov != null && hov !== 'g' + gi && hov !== 'u' + h;
        if (open) g.ports.forEach(p => curve([p.x, p.y, zg(gi)], [u.x, u.y, zu(h)], wih[h][p.i], Math.abs(wih[h][p.i]) / wmax, false));
        else { const [s, a] = bund[gi][h]; curve([g.x + g.w, g.y + g.h / 2, zg(gi)], [u.x, u.y, zu(h)], s, a / bmax, dim, 0.6 + 4 * a / bmax); }
      });
    });
    // its units' recurrence (arcs, bowing left; a unit's own weight is its ring)
    const whh = br.weights_hh;
    let hmax = 1e-9; whh.forEach(r => r.forEach(v => { hmax = Math.max(hmax, Math.abs(v)); }));
    for (let r = 0; r < nH; r++) for (let q = 0; q < nH; q++) if (r !== q) {
      const dim = hov != null && hov !== 'u' + r && hov !== 'u' + q;
      arc([units[q].x, units[q].y, zu(q)], [units[r].x, units[r].y, zu(r)], whh[r][q], Math.abs(whh[r][q]) / hmax, -1, dim);
    }
    // the stream: a lane per unit, through its layers to the top of the stack
    units.forEach((u, h) => { const a = P(u.x, u.y, zu(h)), b = P(xend, u.y, zu(h)); ctx.strokeStyle = 'rgba(150,170,190,0.16)'; ctx.lineWidth = 2 * Math.min(2, a[2]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); });
    let lmax = 1e-9; ls.forEach(L => { (L.W || []).forEach(r => r.forEach(v => { lmax = Math.max(lmax, Math.abs(v)); })); (L.U || []).forEach(r => r.forEach(v => { lmax = Math.max(lmax, Math.abs(v)); })); });
    const gmax = Math.max(1e-9, ...ls.map(L => Math.abs(L.gate ? L.gate[0] : 0)));
    ls.forEach((L, k) => {
      const src = k === 0 ? units : layers[k - 1], col = layers[k], Wk = L.W || [], Uk = L.U || [], g = L.gate ? L.gate[0] : 0;
      for (let r = 0; r < nH; r++) for (let q = 0; q < nH; q++) {
        const dim = hov != null && hov !== `l${k}:${r}` && hov !== (k === 0 ? 'u' + q : `l${k - 1}:${q}`);
        if (Wk[r] && Wk[r][q]) curve([src[q].x, src[q].y, zu(q)], [col[r].x, col[r].y, zu(r)], Wk[r][q], Math.abs(Wk[r][q]) / lmax, dim, null);
        if (r !== q && Uk[r] && Uk[r][q]) arc([col[q].x, col[q].y, zu(q)], [col[r].x, col[r].y, zu(r)], Uk[r][q], Math.abs(Uk[r][q]) / lmax, 1, dim);
      }
    });
    // its outputs, read from the top of the stack
    const topCol = ls.length ? layers[ls.length - 1] : units;
    const who_ = h => (ls.length ? `l${ls.length - 1}:${h}` : 'u' + h);
    const who2 = br.weights_ho;
    let omax = 1e-9; who2.forEach(r => r.forEach(v => { omax = Math.max(omax, Math.abs(v)); }));
    for (let o = 0; o < nOut; o++) for (let h = 0; h < nH; h++) {
      const dim = hov != null && hov !== 'o' + o && hov !== who_(h);
      curve([topCol[h].x, topCol[h].y, zu(h)], [outs[o].x, outs[o].y, zo(o)], who2[o][h], Math.abs(who2[o][h]) / omax, dim, null);
    }

    // the organs themselves
    const mb = d.mb || {}, firing = new Set(mb.active || []);
    groups.forEach((g, gi) => {
      const z = zg(gi), dim = hov != null && hov !== 'g' + gi;
      quad(g.x, g.y, g.w, g.h, z, dim ? 'rgba(18,26,34,0.85)' : '#121a22', hov === 'g' + gi ? '#7fd4ff' : '#2e4050');
      const title = text(g.name, g.x + 6, g.y + 8, z, 'bold 10px monospace', g.mb ? '#e6c8ff' : '#c9b8b0', 'left', 'middle');
      pts.push({ sx: title[0] + 30, sy: title[1] + 4, key: 'g' + gi });
      // the mushroom body's Kenyon cells: lit while firing, coloured by what each learned
      if (g.mb && mb.n && g.h > 34) {
        const food = int8s(mb.food) || [], danger = int8s(mb.danger) || [];  // packed as bytes (livelife)
        let fmax = 1e-6, dmax = 1e-6;
        for (let k = 0; k < mb.n; k++) { fmax = Math.max(fmax, Math.abs(food[k] || 0)); dmax = Math.max(dmax, Math.max(0, danger[k] || 0)); }
        const ax = g.x + 6, ay = g.y + 16, aw = g.w - 26, ah = g.h - 22, cols = Math.max(1, Math.ceil(Math.sqrt(mb.n * aw / ah))), rows = Math.ceil(mb.n / cols);
        const cs = Math.max(1, Math.min(aw / cols, ah / rows) - 0.5);
        for (let k = 0; k < mb.n; k++) {
          const f = (food[k] || 0) / fmax, dg = Math.max(0, danger[k] || 0) / dmax, on = firing.has(k), dead = mb.live != null && k >= mb.live;
          const rC = Math.round(45 + 210 * Math.max(dg, f < 0 ? -f : 0)), gC = Math.round(45 + 210 * Math.max(0, f) + (f < 0 ? 110 * -f : 0));
          const a = P(ax + (k % cols) * (aw / cols), ay + Math.floor(k / cols) * (ah / rows), z);
          ctx.fillStyle = dead ? 'rgba(60,60,60,0.4)' : `rgba(${rC},${gC},45,${on ? 1 : 0.35})`;
          ctx.fillRect(a[0], a[1], Math.max(1, cs * Math.min(2, a[2])), Math.max(1, cs * Math.min(2, a[2])));
        }
      }
      g.ports.forEach(p => {
        node([p.x, p.y, z], 2.2, '#2a1512', '#b88a80', 1);
        if (hov === 'g' + gi) text(nsIn(br, p.i), p.x - 5, p.y, z, '9px monospace', '#e8d8d0', 'right');
      });
    });
    // the units
    units.forEach((u, h) => {
      const a = hid[h] || 0, self = whh[h][h] || 0, dim = hov != null && hov !== 'u' + h;
      const s = node([u.x, u.y, zu(h)], 8, sgn(a, (0.15 + 0.85 * Math.abs(a)) * (dim ? 0.4 : 1)), sgn(self, 0.25 + 0.6 * Math.abs(self) / hmax), 1 + 2 * Math.abs(self) / hmax);
      pts.push({ sx: s[0], sy: s[1], key: 'u' + h });
      text(`h${h + 1} ${a.toFixed(2)}`, u.x + 2, u.y - 13, zu(h), '9px monospace', '#a9bcc8', 'center', 'middle');
    });
    // the layers' units: filled by what each adds to its lane now, ringed by its gate (dashed: silent)
    const spacing = ls.length ? (xend - xu) / (ls.length + 1) : 0, lr = Math.max(2, Math.min(6, spacing / 3));
    ls.forEach((L, k) => {
      const g = L.gate ? L.gate[0] : 0, row = lc[k], Uk = L.U || [];
      layers[k].forEach((p, h) => {
        const v = row && row[h] != null ? row[h] : 0, av = Math.min(1, Math.abs(v) / 0.25), self = Uk[h] ? Uk[h][h] || 0 : 0;
        const s = g ? node([p.x, p.y, zu(h)], lr, sgn(v, 0.08 + 0.92 * av), `rgba(232,216,208,${0.25 + 0.75 * Math.abs(g) / gmax})`, 1 + Math.abs(self) / lmax)
                    : node([p.x, p.y, zu(h)], lr, null, '#8a9aaa', 1, true);
        pts.push({ sx: s[0], sy: s[1], key: `l${k}:${h}` });
      });
      if (spacing >= 22) {
        text(g ? `L${k + 1}` : 'wait', layers[k][0].x, top - 4, zu(0), '9px monospace', g ? '#9a6f67' : '#8a9aaa', 'center', 'bottom');
        if (g) text(g.toFixed(2).replace(/^(-?)0\./, '$1.'), layers[k][0].x, bot + 4, zu(nH - 1), '9px monospace', '#9a6f67', 'center', 'top');
      }
    });
    if (ls.length && spacing < 22) text(layerSummary(ls), xu + 14, top - 4, 0, '9px monospace', '#9a6f67', 'left', 'bottom');
    // the outputs and what they drive
    outs.forEach((p, o) => {
      const dim = hov != null && hov !== 'o' + o;
      const s = node([p.x, p.y, zo(o)], 9, dim ? 'rgba(10,42,26,0.5)' : '#0a2a1a', '#ffe2d6', 1);
      pts.push({ sx: s[0], sy: s[1], key: 'o' + o });
      text(nsOut(br, o), p.x + 13, p.y, zo(o), '10px monospace', '#ffe2d6', 'left');
    });
    if (nOut >= 3) text('eye muscles: pan, tilt, zoom', xo - 10, outs[0].y - 14, zo(0), '9px monospace', 'rgba(200,190,180,0.7)', 'left', 'bottom');
    // a hovered thing, named
    if (hov != null) {
      const lab = hov[0] === 'g' ? groups[+hov.slice(1)].name + ': ' + groups[+hov.slice(1)].idx.length + ' inputs, its real wires shown'
        : hov[0] === 'u' ? `unit h${+hov.slice(1) + 1}` : hov[0] === 'o' ? nsOut(br, +hov.slice(1))
        : (() => { const [k, h] = hov.slice(1).split(':').map(Number), L = ls[k], g = L && L.gate ? L.gate[0] : 0, v = lc[k] && lc[k][h] != null ? lc[k][h] : 0;
                   return `layer ${k + 1}, its unit ${h + 1}: gate ${g.toFixed(3)}, adds ${v.toFixed(3)} now`; })();
      ctx.font = '11px monospace'; ctx.fillStyle = '#ffe2d6'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(lab, W - 8, H - 6);
    }
    B3.pts = pts;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  // Its stacked layers in a line: how many, how many waiting, their gates' range.
  function layerSummary(ls) {
    const on = ls.filter(l => l.gate && l.gate[0]), gs = on.map(l => l.gate[0]), f = v => v.toFixed(2).replace(/^(-?)0\./, '$1.');
    return `${on.length} layer${on.length === 1 ? '' : 's'}${ls.length > on.length ? ` +${ls.length - on.length} waiting` : ''}${gs.length ? `, gates ${f(Math.min(...gs))} to ${f(Math.max(...gs))}` : ''}`;
  }
  function drawBrain(d) {
    const br = d.brain; if (!br) return;
    if ($('brain-units')) $('brain-units').textContent = br.bias_h ? br.bias_h.length : '--';
    drawMBHud(d);  // its life's tally (under the navigation card); the mushroom body is drawn on the flatmap
    drawReplayEye(d);
    if ($('brain-mb')) $('brain-mb').textContent = d.kc ? '' : 'No mushroom body yet (lifetime learning evolves). ';
    if ($('brain-layers')) { const ls = br.layers || [], on = ls.filter(l => l.gate && l.gate[0] !== 0); $('brain-layers').textContent = `${on.length} layer${on.length === 1 ? '' : 's'}` + (ls.length > on.length ? ` (+${ls.length - on.length} silent)` : '') + (on.length ? ` -- gates ${on.map(l => l.gate[0].toFixed(2)).join(', ')}` : ''); }
    const c = $('brain'), W = Math.max(200, fitWidth($('brain').closest('.panel'), quadAspect()));
    // tall enough for its organs' blocks: a phone gets more height than width
    const H = Math.round(W < 600 ? Math.max(480, W * 1.35) : Math.max(520, W * quadAspect() * 1.05));
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    BZ.d = d;
    if (B3.on) { drawBrain3D(); return; }
    drawNervous(d, false);
    const ctx = c.getContext('2d');
    if (BZ.k > 1.01) { ctx.font = '10px monospace'; ctx.fillStyle = '#6a5550'; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`${BZ.k.toFixed(1)}x  (double-click: fit)`, W - 8, 6); }
  }
  // Its nervous system, 2D or 3D (drawNervous): drag to orbit (3D) or pan
  // (2D, zoomed), wheel or pinch to zoom, double-click to reset; hover an
  // organ, unit or output to light its wires; in 3D, idle, it drifts.
  const B3 = { on: false, yaw: 0.35, pitch: 0.2, zoom: 1, vyaw: 0, touched: 0, hover: null, pts: [], dirty: false };
  B3.on = false;  // 2D by default; 3D is the option
  const BRAIN_VIEW_KEY = MOBILE ? 'brain-view-phone' : 'brain-view';
  try { B3.on = localStorage.getItem(BRAIN_VIEW_KEY) === '3d'; } catch (e) {}
  // The largest on-screen distance from the centre a point within radius R
  // of it can reach under perspective F, at any rotation (per unit of scale).
  function reach3D(R, F) {
    let m = 0;
    for (let i = 0; i <= 60; i++) { const z = -R + 2 * R * i / 60; m = Math.max(m, Math.sqrt(Math.max(0, R * R - z * z)) * F / (F - z)); }
    return m;
  }
  function drawBrain3D() { if (BZ.d) drawNervous(BZ.d, true); }  // the same flatmap, given depth and turned
  function brain3DAt(x, y) {
    let best = null, bd = 14;
    B3.pts.forEach(q => { const dd = Math.hypot(q.sx - x, q.sy - y); if (dd < bd) { bd = dd; best = q.key; } });
    return best;
  }
  // It draws when something changes (new data, a touch) and drifts a while
  // after (driftStep), then its loop sleeps.
  B3.loop = false; B3.touched = performance.now() - 4000;
  function brain3DKick() {
    if (B3.loop) return;
    B3.loop = true;
    const tick = now => {
      const more = driftStep(B3, now, !B3.on || B3.hover != null || BZ.pts.size > 0);
      const c = $('brain'), r = c && c.getBoundingClientRect(), moving = Math.abs(B3.vyaw) > 1e-4;
      if ((B3.dirty || moving) && B3.on && BZ.d && r && r.bottom > 0 && r.top < innerHeight && !document.hidden) { B3.dirty = false; drawBrain3D(); }
      if (more || B3.dirty) requestAnimationFrame(tick); else { B3.vyaw = 0; B3.last = null; B3.loop = false; }
    };
    requestAnimationFrame(tick);
  }
  brain3DKick();
  function setBrainView(on) {
    B3.on = on; try { localStorage.setItem(BRAIN_VIEW_KEY, on ? '3d' : '2d'); } catch (e) {}
    const m = $('brain-mode'); if (m) m.innerHTML = on ? '<a href="#">2D</a> &middot; <b>3D</b>' : '<b>2D</b> &middot; <a href="#">3D</a>';

    if (BZ.d) drawBrain(BZ.d);
    if (on) { B3.touched = performance.now() - 4000; brain3DKick(); }  // into 3D: it drifts a while
  }
  // Zoom the brain: wheel (or pinch) zooms about the pointer, drag pans,
  // double-click fits it back.
  const BZ = { k: 1, x: 0, y: 0, d: null, drag: null, pts: new Map(), pinch: null };
  function brainZoomTo(k, px, py) {
    const c = $('brain'); k = Math.max(1, Math.min(12, k));
    BZ.x = px - (px - BZ.x) * k / BZ.k; BZ.y = py - (py - BZ.y) * k / BZ.k; BZ.k = k;
    if (k === 1) { BZ.x = BZ.y = 0; }
    BZ.x = Math.min(0, Math.max(c.width * (1 - k), BZ.x)); BZ.y = Math.min(0, Math.max(c.height * (1 - k), BZ.y));
    if (BZ.d) drawBrain(BZ.d);
  }
  function brainPoint(e) { const c = $('brain'), r = c.getBoundingClientRect(); return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]; }
  (() => {
    const c = $('brain'); if (!c) return;
    const active = steerable(c);
    const m = $('brain-mode'); if (m) m.addEventListener('click', e => { e.preventDefault(); if (e.target.tagName === 'A') setBrainView(!B3.on); });
    setBrainView(B3.on);
    c.addEventListener('wheel', e => {
      if (!active()) return;
      e.preventDefault(); B3.touched = performance.now(); brain3DKick();
      if (B3.on) { B3.zoom = Math.max(0.5, Math.min(6, B3.zoom * Math.exp(-e.deltaY * 0.0015))); drawBrain3D(); return; }
      const [x, y] = brainPoint(e); brainZoomTo(BZ.k * Math.exp(-e.deltaY * 0.0015), x, y);
    }, { passive: false });
    c.addEventListener('dblclick', () => {
      if (!active()) return;
      if (B3.on) { B3.yaw = 0.35; B3.pitch = 0.2; B3.zoom = 1; drawBrain3D(); return; }
      BZ.k = 1; BZ.x = BZ.y = 0; if (BZ.d) drawBrain(BZ.d);
    });
    c.addEventListener('pointerleave', () => { if (B3.hover != null) { B3.hover = null; B3.dirty = true; if (B3.on) brain3DKick(); else if (BZ.d) drawBrain(BZ.d); } });
    c.addEventListener('pointerdown', e => { if (!active()) { steerOn(c); return; } c.setPointerCapture(e.pointerId); BZ.pts.set(e.pointerId, brainPoint(e)); BZ.pinch = null; B3.touched = performance.now(); brain3DKick(); if (B3.on) c.style.cursor = 'grabbing'; });
    c.addEventListener('pointermove', e => {
      if (!BZ.pts.has(e.pointerId)) {
        const [x, y] = brainPoint(e), h = brain3DAt(x, y);
        if (h !== B3.hover) { B3.hover = h; if (B3.on) { B3.dirty = true; brain3DKick(); } else if (BZ.d) drawBrain(BZ.d); }
        return;
      }
      const prev = BZ.pts.get(e.pointerId), now = brainPoint(e); BZ.pts.set(e.pointerId, now);
      B3.touched = performance.now(); brain3DKick();
      if (BZ.pts.size === 2) {
        const [a, b] = [...BZ.pts.values()], dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (BZ.pinch) {
          if (B3.on) { B3.zoom = Math.max(0.5, Math.min(6, B3.zoom * dist / BZ.pinch)); B3.dirty = true; }
          else brainZoomTo(BZ.k * dist / BZ.pinch, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
        }
        BZ.pinch = dist; return;
      }
      if (B3.on) {
        B3.yaw += (now[0] - prev[0]) * 0.008; B3.pitch = Math.max(-1.4, Math.min(1.4, B3.pitch + (now[1] - prev[1]) * 0.008));
        B3.hover = null; B3.dirty = true; return;
      }
      if (BZ.k > 1) { BZ.x += now[0] - prev[0]; BZ.y += now[1] - prev[1]; brainZoomTo(BZ.k, 0, 0); }
    });
    const up = e => { BZ.pts.delete(e.pointerId); BZ.pinch = null; c.style.cursor = ''; };
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
  })();

  // "Get 3D image": its brain as the old 3D view drew it -- every wire, faint
  // to bright, added light on light (bundles glow where they cross, as the
  // connectome renders do), each bowed a little its own way -- from an oblique
  // angle, no labels: rendered once, offscreen, and saved as a picture on the
  // device you view from. The flatmap is the instrument; this is its portrait.
  // Inputs on three planes at the left, units on a sphere, its stacked layers
  // stepping from the sphere toward the outputs' ring at the right.
  function brainPortrait(d) {
    const br = d && d.brain; if (!br) return null;
    const W = 1920, H = 1080, c = document.createElement('canvas'); c.width = W; c.height = H;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    const nIn = br.weights_ih[0].length, nH = br.weights_ih.length, nOut = br.weights_ho.length, ls = br.layers || [];
    const pts = [], rows = Math.ceil(nIn / 3);
    for (let i = 0; i < nIn; i++) { const col = Math.floor(i / rows), r = i % rows; pts.push({ kind: 'in', x: -1.2, y: -0.92 + 1.84 * r / Math.max(1, rows - 1), z: (col - 1) * 0.42 }); }
    const ga = Math.PI * (3 - Math.sqrt(5)), unit = [];
    for (let h = 0; h < nH; h++) {
      const y = nH > 1 ? 1 - 2 * (h + 0.5) / nH : 0, rad = Math.sqrt(1 - y * y), th = ga * h;
      unit.push({ x: 0.42 * rad * Math.cos(th), y: 0.55 * y, z: 0.55 * rad * Math.sin(th) });
    }
    const hidAt = [unit.map(u => { pts.push({ kind: 'hid', x: u.x, y: u.y, z: u.z }); return pts.length - 1; })];
    ls.forEach((L, k) => {  // each layer's units: its sphere carried a step toward the outputs, a little tighter
      const t = (k + 1) / (ls.length + 1), sh = 0.75 * t, sc = 1 - 0.35 * t;
      hidAt.push(unit.map(u => { pts.push({ kind: 'lay', x: u.x * sc + sh, y: u.y * sc, z: u.z * sc }); return pts.length - 1; }));
    });
    const outAt = [];
    for (let o = 0; o < nOut; o++) { const a = 2 * Math.PI * o / nOut; pts.push({ kind: 'out', x: 1.2, y: 0.55 * Math.cos(a), z: 0.55 * Math.sin(a) }); outAt.push(pts.length - 1); }
    const yaw = 0.65, pitch = -0.28, F = 3.2, cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    // one fixed angle: fitted to what it actually spans from there, centred
    let x0 = Infinity, x1m = -Infinity, y0 = Infinity, y1m = -Infinity;
    pts.forEach(q => {
      const x1 = q.x * cy + q.z * sy, z1 = -q.x * sy + q.z * cy, y2 = q.y * cp - z1 * sp, z2 = q.y * sp + z1 * cp, w = F / (F + z2);
      q.ux = x1 * w; q.uy = y2 * w; q.w = w; q.depth = z2; q.fog = Math.max(0.25, Math.min(1, 0.65 - 0.45 * z2));
      x0 = Math.min(x0, q.ux); x1m = Math.max(x1m, q.ux); y0 = Math.min(y0, q.uy); y1m = Math.max(y1m, q.uy);
    });
    const scale = 0.88 * Math.min(W / Math.max(1e-6, x1m - x0), H / Math.max(1e-6, y1m - y0)), cx0 = (x0 + x1m) / 2, cy0 = (y0 + y1m) / 2;
    pts.forEach(q => { q.sx = W / 2 + (q.ux - cx0) * scale; q.sy = H / 2 + (q.uy - cy0) * scale; });
    const edges = [];  // every wire: [from, to, weight, recurrent]
    for (let h = 0; h < nH; h++) for (let i = 0; i < nIn; i++) edges.push([i, hidAt[0][h], br.weights_ih[h][i], 0]);
    for (let r = 0; r < nH; r++) for (let q = 0; q < nH; q++) if (r !== q) edges.push([hidAt[0][q], hidAt[0][r], br.weights_hh[r][q], 1]);
    ls.forEach((L, k) => {
      for (let r = 0; r < nH; r++) for (let q = 0; q < nH; q++) {
        if (L.W && L.W[r]) edges.push([hidAt[k][q], hidAt[k + 1][r], L.W[r][q], 0]);
        if (r !== q && L.U && L.U[r]) edges.push([hidAt[k + 1][q], hidAt[k + 1][r], L.U[r][q], 1]);
      }
    });
    const topAt = hidAt[hidAt.length - 1];
    for (let o = 0; o < nOut; o++) for (let h = 0; h < nH; h++) edges.push([topAt[h], outAt[o], br.weights_ho[o][h], 0]);
    let maxW = 1e-9; edges.forEach(e => { maxW = Math.max(maxW, Math.abs(e[2])); });
    const hash = (x, y) => { const v = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return v - Math.floor(v); };
    ctx.globalCompositeOperation = 'lighter';
    edges.forEach(([a, b, w, rec]) => {
      const A = pts[a], B = pts[b], s = Math.min(1, Math.abs(w) / maxW), alpha = (0.05 + 0.55 * s) * (A.fog + B.fog) / 2;
      const mx = (A.sx + B.sx) / 2, my = (A.sy + B.sy) / 2;
      let cx, cyy;
      if (rec) { const ox = mx - W / 2, oy = my - H / 2, n = Math.hypot(ox, oy) || 1; cx = mx + ox / n * 40; cyy = my + oy / n * 40; }
      else { const dx = B.sx - A.sx, dy = B.sy - A.sy, bend = (hash(a, b) - 0.5) * 0.35; cx = mx - dy * bend; cyy = my + dx * bend; }
      ctx.strokeStyle = 'rgba(' + (w >= 0 ? '127,212,255' : '255,153,0') + ',' + Math.min(1, alpha) + ')';
      ctx.lineWidth = (0.6 + 2.2 * s) * (A.w + B.w) / 2;
      ctx.beginPath(); ctx.moveTo(A.sx, A.sy); ctx.quadraticCurveTo(cx, cyy, B.sx, B.sy); ctx.stroke();
    });
    const hid = d.brain_hidden || [];
    hidAt[0].forEach((idx, h) => {  // a soft glow behind each unit, as bright as it is active
      const q = pts[idx], a = hid[h] || 0, rr = 34 * q.w * (0.4 + Math.abs(a)), col = a >= 0 ? '127,212,255' : '255,153,0';
      const g = ctx.createRadialGradient(q.sx, q.sy, 0, q.sx, q.sy, rr);
      g.addColorStop(0, 'rgba(' + col + ',' + (0.35 * Math.abs(a) * q.fog) + ')'); g.addColorStop(1, 'rgba(' + col + ',0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(q.sx, q.sy, rr, 0, 7); ctx.fill();
    });
    ctx.globalCompositeOperation = 'source-over';
    pts.slice().sort((a, b) => b.depth - a.depth).forEach(q => {
      const r = (q.kind === 'hid' ? 9 : q.kind === 'out' ? 11 : q.kind === 'lay' ? 4 : 5) * q.w;
      ctx.globalAlpha = q.fog;
      ctx.fillStyle = q.kind === 'out' ? '#0a2a1a' : q.kind === 'in' ? '#2a1512' : 'rgba(127,212,255,0.55)';
      ctx.strokeStyle = q.kind === 'out' ? '#ffe2d6' : q.kind === 'in' ? '#b88a80' : '#345';
      ctx.beginPath(); ctx.arc(q.sx, q.sy, Math.max(1.5, r), 0, 7); ctx.fill(); ctx.stroke();
    });
    ctx.globalAlpha = 1;
    return { canvas: c, wires: edges.length };
  }
  (() => {
    const a = $('brain-portrait'), note = $('brain-portrait-note'); if (!a) return;
    a.addEventListener('click', e => {
      e.preventDefault();
      const p = brainPortrait(BZ.d); if (!p) { note.textContent = 'no brain yet'; return; }
      p.canvas.toBlob(b => {
        const url = URL.createObjectURL(b), link = document.createElement('a');
        link.href = url; link.download = 'cambrioid-brain-' + location.hostname + '-' + new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-') + '.png';
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        note.textContent = 'saved: ' + p.wires + ' wires';
      }, 'image/png');
    });
  })();
  // Trees (every tree the genome has).
  const PLANE_NAMES = ['now', 'prev', 'rg', 'by'];
  function nodeLabel(n) { return n.kind === 'var' ? 'x' + n.index : n.kind === 'cell' ? `${PLANE_NAMES[n.index] || 'p' + n.index}(${n.kx},${n.ky})` : n.kind === 'pool' ? `${PLANE_NAMES[n.index] || 'p' + n.index}[${n.kx},${n.ky}]±${Math.round(n.value)}` : n.kind === 'edge' ? `${PLANE_NAMES[n.index] || 'p' + n.index}/${Math.round((n.angle || 0) * 180 / Math.PI)}°[${n.kx},${n.ky}]±${Math.round(n.value)}` : n.kind === 'const' ? n.value.toFixed(2) : n.op; }
  // Its perception trees in 3D: a cone tree (Robertson, Mackinlay & Card
  // 1991) -- each node's children on a circle below it -- growing down onto
  // its eye: a leaf that reads a receptor plugs into the retina grid at the
  // bottom, at the receptor it reads (plane: now / previous look / colour).
  // Drag to orbit, wheel or pinch to zoom, double-click to reset; hover names
  // a node; it drifts for 20 s after a touch (driftStep), then is still.
  const TREE3 = {};
  const PLANE_COL = ['255,226,214', '154,111,103', '255,95,162', '120,160,255'];
  function coneLayout(root, half) {
    const pts = [], links = [];
    const leaves = n => n.children && n.children.length ? n.children.reduce((t, c) => t + leaves(c), 0) : 1;
    let maxK = Math.max(1, half || 1);  // its whole eye, receptors either side of the centre
    (function scan(n) { if (n.kind === 'cell' || n.kind === 'pool' || n.kind === 'edge') maxK = Math.max(maxK, Math.abs(n.kx) + 1, Math.abs(n.ky) + 1); (n.children || []).forEach(scan); })(root);
    let depthMax = 0;
    (function place(n, x, y, z, r, depth, parent) {
      const idx = pts.length;
      depthMax = Math.max(depthMax, depth);
      pts.push({ n, x, y, z, depth, retina: false });
      if (parent != null) links.push([parent, idx]);
      const kids = n.children || [];
      const total = kids.reduce((t, c) => t + leaves(c), 0);
      let acc = 0;
      kids.forEach(c => {
        const share = leaves(c), ang = 2 * Math.PI * (acc + share / 2) / Math.max(1, total) + depth * 0.7;
        acc += share;
        const rr = kids.length === 1 ? 0 : r;
        place(c, x + rr * Math.cos(ang), y + 0.34, z + rr * Math.sin(ang), r * 0.62, depth + 1, idx);
      });
    })(root, 0, 0, 0, 0.55, 0, null);
    // receptor leaves drop onto the retina plane below the deepest level
    const floor = 0.34 * (depthMax + 1);
    pts.forEach(q => { if (q.n.kind === 'cell' || q.n.kind === 'pool' || q.n.kind === 'edge') { q.retina = true; q.x = 0.9 * (q.n.kx + 0.5) / maxK; q.z = 0.9 * (q.n.ky + 0.5) / maxK; q.y = floor; } });
    const mid = floor / 2;
    pts.forEach(q => { q.y -= mid; });
    return { pts, links, floor: floor - mid, maxK };
  }
  // How far a tree's layout reaches from the centre on screen (x, y, at unit
  // scale) over a whole turn at this tilt: its nodes and its retina's corners.
  function treeExtent(L, pitch, F) {
    const key = Math.round(pitch * 50);
    if (L._ext && L._ext[0] === key) return L._ext[1];
    const cp = Math.cos(pitch), sp = Math.sin(pitch), g = 0.9;
    const pts = L.pts.map(q => [q.x, q.y, q.z]).concat([[-g, L.floor, -g], [g, L.floor, -g], [g, L.floor, g], [-g, L.floor, g]]);
    let mx = 0.1, my = 0.1;
    for (let a = 0; a < 24; a++) {
      const cy = Math.cos(a * Math.PI / 12), sy = Math.sin(a * Math.PI / 12);
      pts.forEach(([x, y, z]) => {
        const x1 = x * cy + z * sy, z1 = -x * sy + z * cy, y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp, w = F / (F + z2);
        mx = Math.max(mx, Math.abs(x1 * w)); my = Math.max(my, Math.abs(y2 * w));
      });
    }
    L._ext = [key, [mx, my]];
    return L._ext[1];
  }
  function drawTree3D(name) {
    const T = TREE3[name]; if (!T) return;
    const c = T.canvas, W = c.width, H = c.height, ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    const cy = Math.cos(T.yaw), sy = Math.sin(T.yaw), cp = Math.cos(T.pitch), sp = Math.sin(T.pitch);
    const L = T.lay, F = 3.4;
    ctx.font = '11px monospace';
    const Lw = Math.max(...L.pts.map(q => ctx.measureText(nodeLabel(q.n)).width), 20) / 2;
    // its fit: the tree and its retina as they actually reach, over a whole
    // turn at this tilt (so the drift doesn't make it breathe) -- not a sphere
    // around everything, which left a small tree in an empty card
    const ext = treeExtent(L, T.pitch, F);
    const scale = Math.max(20, Math.min((W / 2 - Lw - 10) / ext[0], (H / 2 - 18) / ext[1])) * T.zoom;
    const proj = (x, y, z) => {
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp, w = F / (F + z2);
      return [W / 2 + x1 * scale * w, H / 2 + y2 * scale * w, z2, w];
    };
    // the retina: a faint grid, a line per receptor
    const k = L.maxK, g = 0.9;
    ctx.strokeStyle = 'rgba(52,68,85,0.55)'; ctx.lineWidth = 1;
    for (let i = -k; i <= k; i++) {
      const u = g * i / k;
      let a1 = proj(u, L.floor, -g), a2 = proj(u, L.floor, g); ctx.beginPath(); ctx.moveTo(a1[0], a1[1]); ctx.lineTo(a2[0], a2[1]); ctx.stroke();
      a1 = proj(-g, L.floor, u); a2 = proj(g, L.floor, u); ctx.beginPath(); ctx.moveTo(a1[0], a1[1]); ctx.lineTo(a2[0], a2[1]); ctx.stroke();
    }
    const P = L.pts.map(q => { const [sx, sy2, d, w] = proj(q.x, q.y, q.z); return Object.assign(q, { sx, sy: sy2, d, w, fog: Math.max(0.3, Math.min(1, 0.7 - 0.4 * d)) }); });
    const hov = T.hover;
    ctx.globalCompositeOperation = 'lighter';
    L.links.forEach(([a, b]) => {
      const A = P[a], B = P[b], lit = hov === a || hov === b;
      ctx.strokeStyle = B.retina ? `rgba(${PLANE_COL[B.n.index] || '200,200,200'},${(lit ? 0.9 : 0.45) * B.fog})` : `rgba(127,212,255,${(lit ? 0.9 : 0.4) * (A.fog + B.fog) / 2})`;
      ctx.lineWidth = (lit ? 2.2 : 1.3) * (A.w + B.w) / 2;
      if (B.retina) ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(A.sx, A.sy); ctx.lineTo(B.sx, B.sy); ctx.stroke(); ctx.setLineDash([]);
    });
    ctx.globalCompositeOperation = 'source-over';
    const many = P.length > 60;
    P.map((q, i) => [q, i]).sort((x, y) => y[0].d - x[0].d).forEach(([q, i]) => {
      const op = q.n.kind === 'op', r = (q.retina ? 5 : op ? 15 : 12) * q.w * Math.sqrt(T.zoom);
      ctx.globalAlpha = q.fog;
      if (q.retina) {  // a receptor (a pool: its whole patch, flat on the retina)
        const k2 = q.n.kind === 'pool' || q.n.kind === 'edge' ? 2 * Math.round(q.n.value) + 1 : 1;
        ctx.fillStyle = `rgba(${PLANE_COL[q.n.index] || '200,200,200'},${k2 > 1 ? 0.55 : 1})`; ctx.fillRect(q.sx - r * k2, q.sy - r * k2 / 2, 2 * r * k2, r * k2);
        if (q.n.kind === 'edge') {  // its orientation: the line that splits its patch
          const arc = (x, y, a, L, pr, bend, ys) => { ctx.beginPath(); for (let i = 0; i <= 12; i++) { const tp = (i / 6 - 1) * L, np = 0.5 * (bend || 0) * (tp / pr) * (tp / pr) * pr, px = x - tp * Math.sin(a) + np * Math.cos(a), py = y + (tp * Math.cos(a) + np * Math.sin(a)) * ys; if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); } ctx.stroke(); };  // an oriented pool's line (an arc when it bends: blocks.py)
          ctx.strokeStyle = `rgb(${PLANE_COL[q.n.index] || '200,200,200'})`; ctx.lineWidth = 1.5; arc(q.sx, q.sy, q.n.angle || 0, r * k2, 2 * r, q.n.bend, 0.5);
        }
      }
      else { ctx.fillStyle = op ? '#0a2a1a' : '#1a1a2a'; ctx.strokeStyle = op ? '#ffe2d6' : '#ffb4a6'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(q.sx, q.sy, r, 0, 7); ctx.fill(); ctx.stroke(); }
      if (!many || i === hov || op) {
        ctx.fillStyle = q.retina ? `rgb(${PLANE_COL[q.n.index] || '200,200,200'})` : '#cfe3f0';
        ctx.font = (i === hov ? 'bold 12px' : '11px') + ' monospace'; ctx.textBaseline = 'middle';
        if (q.retina) { ctx.textAlign = 'center'; ctx.fillText(nodeLabel(q.n), q.sx, q.sy + r + 9); }
        else { ctx.textAlign = 'center'; ctx.fillText(nodeLabel(q.n), q.sx, q.sy); }
      }
    });
    ctx.globalAlpha = 1;
    T.P = P;
  }
  function treeKick(name) {
    const T = TREE3[name]; if (!T || T.loop) return;
    T.loop = true;
    const tick = now => {
      const more = driftStep(T, now, T.hover != null || T.drag);
      const r = T.canvas.getBoundingClientRect(), moving = Math.abs(T.vyaw) > 1e-4;
      if ((T.dirty || moving) && r.bottom > 0 && r.top < innerHeight && !document.hidden) { T.dirty = false; drawTree3D(name); }
      if (more || T.dirty) requestAnimationFrame(tick); else { T.vyaw = 0; T.last = null; T.loop = false; }
    };
    requestAnimationFrame(tick);
  }
  function renderTrees(trees, stats, limits) {
    const box = $('trees');
    const TW = Math.max(300, Math.floor(fitWidth($('tree-panel'), quadAspect())));
    const N = (D && D.receptors) || 12, key = N + ':' + TW + JSON.stringify(trees);  // its size too: an expanded card redraws
    if (box.dataset.key === key) return;  // unchanged: keep the view as it is
    box.dataset.key = key; box.innerHTML = '';
    for (const name of Object.keys(trees)) {
      const wrap = document.createElement('div');
      const s = stats && stats[name];
      wrap.innerHTML = `<div class="cap">${name}${s && limits ? ` -- ${s.nodes} / ${limits.max_nodes} nodes, depth ${s.depth} / ${limits.max_depth}` : ''}</div>`;
      const c = document.createElement('canvas'); c.style.width = '100%'; const active = steerable(c);
      wrap.appendChild(c); box.appendChild(wrap);
      const W = TW, lay = coneLayout(trees[name], N / 2), ext = treeExtent(lay, (TREE3[name] || {}).pitch ?? 0.3, 3.4);
      c.width = W; c.height = Math.round(W * Math.max(0.45, Math.min(quadAspect(), (ext[1] + 0.12) / (ext[0] + 0.25))));  // its own shape: no empty band
      const old = TREE3[name] || {};
      const T = TREE3[name] = { canvas: c, lay, yaw: old.yaw ?? 0.35, pitch: old.pitch ?? 0.3, zoom: old.zoom ?? 1,
                                vyaw: 0, touched: performance.now() - 4000, hover: null, drag: null, loop: false, dirty: true, P: [] };
      const pt = e => { const r = c.getBoundingClientRect(); return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]; };
      const touch = () => { T.touched = performance.now(); T.dirty = true; treeKick(name); };
      c.addEventListener('wheel', e => { if (!active()) return; e.preventDefault(); T.zoom = Math.max(0.4, Math.min(6, T.zoom * Math.exp(-e.deltaY * 0.0015))); touch(); }, { passive: false });
      c.addEventListener('dblclick', () => { if (!active()) return; T.yaw = 0.35; T.pitch = 0.3; T.zoom = 1; touch(); });
      c.addEventListener('pointerdown', e => { if (!active()) { steerOn(c); return; } c.setPointerCapture(e.pointerId); T.drag = pt(e); c.style.cursor = 'grabbing'; touch(); });
      c.addEventListener('pointermove', e => {
        const now = pt(e);
        if (T.drag) { T.yaw += (now[0] - T.drag[0]) * 0.008; T.pitch = Math.max(-1.4, Math.min(1.4, T.pitch + (now[1] - T.drag[1]) * 0.008)); T.drag = now; touch(); return; }
        let best = null, bd = 16; T.P.forEach((q, i) => { const dd = Math.hypot(q.sx - now[0], q.sy - now[1]); if (dd < bd) { bd = dd; best = i; } });
        if (best !== T.hover) { T.hover = best; touch(); }
      });
      const up = () => { T.drag = null; c.style.cursor = ''; };
      c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
      c.addEventListener('pointerleave', () => { if (T.hover != null) { T.hover = null; touch(); } });
      drawTree3D(name); treeKick(name);
    }
  }

  // History charts.
  const CHARTS = [
    { id: 'c-fit', title: 'fitness', cap: 'current / peak ever', series: [['fitness', '#ffe2d6', r => r.best_fitness], ['peak ever', '#9a6f67', r => r.peak_fitness_seen]] },
    { id: 'c-body', title: 'body over its runs', cap: 'per run', fixed: [0, 1], series: [['energy', '#ffe2d6', r => r.mean_energy], ['prey', '#ff5fa2', r => r.mean_prey], ['surprise', '#c8f', r => r.mean_food]] },
    { id: 'c-drive', title: 'homeostatic drive', cap: 'lower is healthier', series: [['drive', '#f6a', r => r.mean_drive]] },
    { id: 'c-look', title: 'gaze size', cap: 'fraction of the frame', fixed: [0, 1], series: [['at birth', '#ffb4a6', r => r.fovea_fraction], ['mean in run', '#c8f', r => r.mean_aperture]] },
    { id: 'c-pace', title: 'resting pace', cap: 'every Nth frame', series: [['every Nth frame', '#ffb4a6', r => r.pace]] },
    { id: 'c-quota', title: 'CPU quota granted', cap: '%', series: [['quota %', '#fd4', r => r.quota_pct]] },
    { id: 'c-tree', title: 'perception tree size', cap: 'response tree nodes / depth', series: [['nodes', '#f90', r => r.tree_nodes], ['depth', '#ffb4a6', r => r.tree_depth]] },
    { id: 'c-diet', title: 'jīng and qì', cap: 'per hour, each on its own scale', hourly: true, dual: true, series: [['jīng (bites)', '#ff5fa2', r => r.bites], ['qì (sips)', '#9ccf7a', r => r.nectar_sips]] },
    { id: 'c-mut', title: 'accepted changes by kind', cap: 'which mutation won, over time', mutations: true },
  ];
  const MUT = ['grow_kc', 'shrink_kc', 'mutate_learning', 'mutate_cones', 'duplicate_layer', 'remove_layer', 'grow_unit', 'shrink_unit', 'mutate_brain', 'mutate_pace', 'mutate_colour', 'mutate_stabilizer', 'mutate_zoom', 'mutate_metabolism', 'mutate_host', 'mutate_replay', 'mutate_vigilance', 'mutate_pump', 'mutate_aversive', 'mutate_receptor_speed', 'mutate_plant_sense', 'mutate_imagery', 'mutate_recall', 'mutate_scenes', 'mutate_sleep_set', 'mutate_pool', 'mutate_setpoints', 'mutate_bore', 'mutate_archetypes', 'mutate_apical', 'mutate_plasticity', 'mutate_maturation', 'mutate_colliculus', 'mutate_prey_sense', 'grow_channel', 'add_prediction', 'shrink_channel', 'mutate_fovea', 'mutate_const', 'mutate_op', 'grow', 'shrink', 'reroll_subtree'];
  const MUT_COLOR = { grow_kc: '#9f6', shrink_kc: '#595', mutate_learning: '#ff9', mutate_cones: '#fb5', duplicate_layer: '#6cf', remove_layer: '#468', grow_unit: '#e9f', shrink_unit: '#958', mutate_prey_sense: '#f8a', mutate_stabilizer: '#9fe', mutate_zoom: '#8ef', mutate_metabolism: '#e96', mutate_host: '#f7c', mutate_replay: '#b9f', mutate_vigilance: '#fe6', mutate_pump: '#e55', mutate_aversive: '#c66', mutate_receptor_speed: '#9cf', mutate_plant_sense: '#9ccf7a', mutate_imagery: '#fff0c0', mutate_recall: '#ffd8a0', mutate_scenes: '#a0e0ff', mutate_sleep_set: '#d0b0ff', mutate_pool: '#ffc070', mutate_setpoints: '#db7', mutate_bore: '#e77', mutate_archetypes: '#7ec', mutate_apical: '#c9f', mutate_plasticity: '#fc9', mutate_maturation: '#9cf', mutate_colliculus: '#f6c', grow_channel: '#fa6', add_prediction: '#fc4', shrink_channel: '#a86', mutate_colour: '#ff5fa2', mutate_pace: '#fd4', mutate_brain: '#f4f', mutate_fovea: '#c8f', mutate_const: '#ffe2d6', mutate_op: '#0af', grow: '#ffb4a6', shrink: '#f90', reroll_subtree: '#f66' };
  (function buildCharts() {
    // the mutation chart spans its whole row, with a line per kind (their names are long and many)
    $('charts').innerHTML = CHARTS.map(ch => `<div class="panel"${ch.mutations ? ' style="grid-column: 1 / -1"' : ''}><h2>${ch.title}</h2><div class="cap">${ch.cap}</div>` +
      (ch.series ? `<div class="legend cap">${ch.series.map(s => `<span><b style="color:${s[1]}">&#9644;</b> ${s[0]}</span>`).join('')}</div>` : '') +
      `<canvas id="${ch.id}" height="190"></canvas></div>`).join('');
  })();
  function lineChart(ch, recs) {
    const c = $(ch.id); c.width = Math.max(300, Math.floor(innerWidth(c.parentElement)));
    c.height = isMax(c) ? Math.max(190, window.innerHeight - 190) : 190 + (CH_EXTRA[ch.id] || 0);
    const ctx = c.getContext('2d'), W = c.width, H = c.height; ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 46, r: 8, t: 8, b: 18 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    if (ch.dual) { dualChart(ch, recs, ctx, W, H, pad, pw, ph); return; }
    let lo = Infinity, hi = -Infinity;
    ch.series.forEach(s => recs.forEach(r => { const v = s[2](r); if (v !== null && v !== undefined && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    if (ch.fixed) { lo = ch.fixed[0]; hi = ch.fixed[1]; }
    if (!isFinite(lo)) { ctx.fillStyle = '#9a6f67'; ctx.font = '11px monospace'; ctx.fillText('no data yet (recorded from this build on)', pad.l, H / 2); return; }
    if (lo === hi) { lo -= 1; hi += 1; }
    const n = recs.length, px = i => pad.l + (n <= 1 ? 0 : i / (n - 1)) * pw, py = v => pad.t + (1 - (v - lo) / (hi - lo)) * ph;
    ctx.strokeStyle = '#1c2a36'; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph); ctx.stroke();
    ctx.fillStyle = '#9a6f67'; ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.fillText(hi.toFixed(2), pad.l - 4, pad.t + 8); ctx.fillText(lo.toFixed(2), pad.l - 4, pad.t + ph);
    ctx.textAlign = 'left'; ctx.fillText(recs.label_left || 'older', pad.l, H - 4); ctx.textAlign = 'right'; ctx.fillText('now', pad.l + pw, H - 4);
    ch.series.forEach(s => { ctx.strokeStyle = s[1]; ctx.lineWidth = 1.5; ctx.beginPath(); let on = false;
      recs.forEach((r, i) => { const v = s[2](r); if (v === null || v === undefined || !isFinite(v)) { on = false; return; } on ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)); on = true; }); ctx.stroke(); });
  }
  function dualChart(ch, recs, ctx, W, H, pad, pw, ph) {
    pad.r = 46; pw = W - pad.l - pad.r;
    const n = recs.length, px = i => pad.l + (n <= 1 ? 0 : i / (n - 1)) * pw;
    ctx.strokeStyle = '#1c2a36'; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t); ctx.stroke();
    if (!n) { ctx.fillStyle = '#9a6f67'; ctx.font = '11px monospace'; ctx.fillText('no hours recorded yet', pad.l + 8, H / 2); return; }
    ch.series.forEach((s, k) => {
      const vals = recs.map(s[2]).map(v => (v === null || v === undefined || !isFinite(v)) ? null : v);
      const hi = Math.max(1, ...vals.filter(v => v !== null)), py = v => pad.t + (1 - v / hi) * ph;
      ctx.fillStyle = s[1]; ctx.font = '10px monospace'; ctx.textAlign = k ? 'left' : 'right';
      ctx.fillText(String(Math.round(hi)), k ? pad.l + pw + 4 : pad.l - 4, pad.t + 8); ctx.fillText('0', k ? pad.l + pw + 4 : pad.l - 4, pad.t + ph);
      ctx.strokeStyle = s[1]; ctx.lineWidth = 1.5; ctx.beginPath(); let on = false;
      vals.forEach((v, i) => { if (v === null) { on = false; return; } on ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)); on = true; }); ctx.stroke();
    });
    ctx.fillStyle = '#9a6f67'; ctx.textAlign = 'left'; ctx.fillText(`${n} hours ago`, pad.l, H - 4); ctx.textAlign = 'right'; ctx.fillText('now', pad.l + pw, H - 4);
  }
  function mutChart(ch, recs) {
    const c = $(ch.id); c.width = Math.max(300, Math.floor(innerWidth(c.parentElement)));
    c.height = isMax(c) ? Math.max(190, window.innerHeight - 190) : MUT.length * 14 + 26;  // 14 px a kind: labels never overlap
    const ctx = c.getContext('2d'), W = c.width, H = c.height; ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 150, r: 8, t: 8, b: 18 }, pw = W - pad.l - pad.r, rh = (H - pad.t - pad.b) / MUT.length;
    ctx.font = '10px monospace'; ctx.textAlign = 'left';
    MUT.forEach((t, k) => { ctx.fillStyle = MUT_COLOR[t]; ctx.fillText(t, 2, pad.t + k * rh + rh / 2 + 3); });
    const n = recs.length;
    recs.forEach((r, i) => (r.accepted_types || []).forEach(t => { const k = MUT.indexOf(t); if (k < 0) return; ctx.fillStyle = MUT_COLOR[t]; ctx.beginPath(); ctx.arc(pad.l + (n <= 1 ? 0 : i / (n - 1)) * pw, pad.t + k * rh + rh / 2, 3, 0, 7); ctx.fill(); }));
  }
  let lastHistory = null, lastHourly = [];

  // Tap/click any panel to maximize it; tap again (or Esc) to restore.
  // Controls inside a panel (inputs, buttons, links) keep working.
  // No wasted space: the charts nearest the body's subject move up under the
  // sleep card, beside the tall body card, as many as fit; the rest stay below.
  const SIDE_ORDER = ['c-body', 'c-drive', 'c-pace', 'c-look', 'c-quota', 'c-tree', 'c-fit'];
  function balanceSide() {
    const side = $('charts-side'), main = $('charts'), body = $('body-panel'), left = $('left-stack'), quad = document.querySelector('.quad');
    if (!side || !main || !body || !left || !quad) return;
    const panelOf = id => { const c = $(id); return c && c.closest('.panel'); };
    DM.extra = 0; if (D) drawDreamMap(D);
    for (const k in CH_EXTRA) delete CH_EXTRA[k];
    [...side.children].forEach(el => main.appendChild(el));  // all back, then in their order
    CHARTS.forEach(ch => { const pn = panelOf(ch.id); if (pn) main.appendChild(pn); });
    if (getComputedStyle(quad).gridTemplateColumns.split(' ').length < 2 || document.body.classList.contains('has-max')) return;
    for (const id of SIDE_ORDER) {
      const pn = panelOf(id); if (!pn) continue;
      if (left.offsetHeight + pn.offsetHeight + 16 <= body.offsetHeight) side.appendChild(pn); else break;
    }
    // what is still left over goes to the place map
    DM.extra = Math.max(0, body.offsetHeight - left.offsetHeight - 2);
    if (D) drawDreamMap(D);
    // the map stops at the card's width; the charts beside it take the rest
    const moved = [...side.children], gap = body.offsetHeight - left.offsetHeight - 2;
    if (gap > 4 && moved.length) moved.forEach(pn => { const c = pn.querySelector('canvas'); if (c) CH_EXTRA[c.id] = Math.floor(gap / moved.length); });
  }
  const CH_EXTRA = {};
  function redrawAll() {
    if (D) { drawBody(D); drawBrain(D); drawSenses(D); spaceData(D); if (D.trees) renderTrees(D.trees, D.tree_stats, D.tree_limits); }  // the retina panel redraws itself (drawLook, every frame)
    balanceSide();
    if (lastHistory) CHARTS.forEach(ch => ch.mutations ? mutChart(ch, lastHistory) : lineChart(ch, ch.hourly ? lastHourly : lastHistory));
  }
  // Where a browser has no full screen for it (an iPhone), the picture is lifted
  // to the page's top level -- so no card or grid around it can box it in --
  // and put back where it was.
  const FILL = { home: null, next: null };
  function fillScreen(el, on) {
    if (on) {
      FILL.home = el.parentNode; FILL.next = el.nextSibling;
      document.body.appendChild(el);
      el.classList.add('filling');
      document.body.classList.add('has-max');
    } else {
      el.classList.remove('filling');
      if (FILL.home) FILL.home.insertBefore(el, FILL.next);
      FILL.home = FILL.next = null;
      document.body.classList.remove('has-max');
    }
    requestAnimationFrame(redrawAll);
  }
  function setMax(panel) {
    document.querySelectorAll('.panel.maximized').forEach(p => { if (p !== panel) p.classList.remove('maximized'); });
    const on = panel ? panel.classList.toggle('maximized') : false;
    document.body.classList.toggle('has-max', on);
    // full screen where the browser allows it for a card (not an iPhone's): the page's own fill stays as the fallback
    if (on && panel.requestFullscreen) panel.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
    else if (!on && document.fullscreenElement) document.exitFullscreen().catch(() => {});
    requestAnimationFrame(redrawAll);
  }
  // leaving full screen by the browser's own way (Esc, a swipe) puts the card back too
  document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
      const p = document.querySelector('.panel.maximized');
      if (p) { p.classList.remove('maximized'); document.body.classList.remove('has-max'); }
    }
    requestAnimationFrame(redrawAll);
  });
  // (A click on a card or the video used to maximize it too; with a full-screen
  // button on each, the two stepped on each other, so only the buttons do it.
  // A card collapse toggle is parked for a UX/UI sprint: docs/next-designs.md.)
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { const p = document.querySelector('.panel.maximized'); if (p) setMax(p); }
  });
  // A full-screen button on every display that goes full screen: the video (bottom right) and
  // the cards drawn from its mind (top right) -- the one way in, and out (or Esc).
  function videoFull(vid) {
    if (document.fullscreenElement === vid) document.exitFullscreen().catch(() => {});
    else if (vid.classList.contains('filling')) fillScreen(vid, false);
    else if (vid.requestFullscreen) vid.requestFullscreen({ navigationUI: 'hide' }).catch(() => fillScreen(vid, true));
    else fillScreen(vid, true);
    requestAnimationFrame(redrawAll);
  }
  // Its full-screen button, in a row of its own just below its picture (the
  // video, or a card's drawing), right-aligned as a player's -- on the card,
  // so it never covers what the picture shows. (In the video's full screen,
  // Esc leaves it, as the browser says.)
  function fsButton(picture, act) {
    const row = document.createElement('div'), b = document.createElement('button');
    row.className = 'fsrow';
    b.className = 'fsbtn'; b.type = 'button'; b.title = 'full screen'; b.setAttribute('aria-label', 'full screen');
    b.innerHTML = '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M1 5V1h4M11 1h4v4M15 11v4h-4M5 15H1v-4"/></svg>';
    b.addEventListener('click', e => { e.stopPropagation(); act(); });
    row.appendChild(b);
    picture.after(row);
  }
  document.querySelectorAll('.video16x9').forEach(v => fsButton(v, () => videoFull(v)));
  ['space-panel', 'look-panel', 'field-panel', 'brain-panel'].forEach(id => {
    const p = $(id), c = p && p.querySelector('canvas');
    if (c) fsButton(c.parentElement === p ? c : c.parentElement, () => setMax(p));  // after its picture's own box (the mushroom body's holds its HUD too)
  });
  window.addEventListener('resize', () => requestAnimationFrame(redrawAll));

  async function fetchHistory() {
    try {
      const h = await (await fetch('/history')).json();
      const recs = h.records || []; recs.label_left = h.span_generations ? `${h.span_generations} generations ago` : 'older';
      lastHistory = recs;
      balanceSide();  // the body card fills in with data: rebalance as it grows
      try { lastHourly = await (await fetch('/metrics')).json(); } catch (e) { }
      CHARTS.forEach(ch => ch.mutations ? mutChart(ch, recs) : lineChart(ch, ch.hourly ? lastHourly : recs));
    } catch (e) { }
    setTimeout(fetchHistory, 20000);
  }
  fetchHistory();

  // YouTube video id from watch?v=, /live/, /embed/, /shorts/ or youtu.be links.
  function youtubeId(url) {
    if (!url) return null;
    try {
      const u = new URL(url);
      if (u.hostname.endsWith('youtu.be')) return u.pathname.slice(1).split('/')[0] || null;
      if (u.searchParams.get('v')) return u.searchParams.get('v');
      const m = u.pathname.match(/\/(?:live|embed|shorts)\/([^/?#]+)/);
      return m ? m[1] : null;
    } catch (e) { return null; }
  }
  // The picture panel: what it saw in its latest run (its camera or the
  // chosen stream), replayed in step with the visual field. The "watching"
  // chip follows what is SELECTED, so it says it is switching until the
  // restarted organism reports in (about a minute).
  let SEL = null;
  function showLive(d) {
    const want = SEL ? (SEL.active ? 'video' : 'camera') : (d && d.is_live ? 'video' : 'camera');
    const id = want === 'video' ? youtubeId((SEL && SEL.selected_url) || (d && d.clip)) : null;
    $('stream-link-row').style.display = id ? 'block' : 'none';
    if (id) $('stream-link').href = `https://www.youtube.com/watch?v=${id}`;
    $('live-title').textContent = d && d.is_live ? 'video stream' : 'camera';
    $('live-cap').textContent = '';
    const running = d && d.generation !== undefined ? (d.is_live ? 'video' : 'camera') : null;
    const switching = running && (running !== want || (want === 'video' && youtubeId(d.clip) !== id));
    $('h-src').textContent = switching
      ? `switching to ${want === 'video' ? 'the video' : 'its camera'}... (restarting, about a minute)`
      : ((d && (d.clip_name || d.clip)) || '--');
  }

  // ---- The replayed picture with its target lock (LOCK_HUD_JS, shared with /live) ----
  const HUD = { on: true }, F = frameLoader($('cam'));
  try { HUD.on = localStorage.getItem('hud-on') !== '0'; } catch (e) { }
  $('hud-on').checked = HUD.on;
  $('hud-on').addEventListener('change', e => { HUD.on = e.target.checked; try { localStorage.setItem('hud-on', HUD.on ? '1' : '0'); } catch (err) { } });
  $('hud-on').addEventListener('click', e => e.stopPropagation());
  $('hud-legend-text').textContent = '';
  function drawHud(now) {
    requestAnimationFrame(drawHud);
    const box = $('live-box'), c = $('hud'), panel = $('live-panel');
    // Same shape as the other three displays; the picture is fitted inside.
    const aspect = 1 / quadAspect();
    const bw = fitWidth(panel, 1 / aspect), bh = bw / aspect;
    box.style.aspectRatio = String(aspect); box.style.width = bw + 'px';
    const dpr = window.devicePixelRatio || 1, W = Math.round(bw * dpr), H = Math.round(bh * dpr);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, bw, bh);
    $('cam-note').textContent = F.failed && F.shown == null ? 'no frames to show (a file source, or CAMBRIAN_CAMERA_PREVIEW=0)' : '';
    if (!D || D.generation === undefined) return;
    const R0 = replayAt(D, now, CLK, REPLAY_FPS);
    F.show(R0.frame, R0.epoch);
    const R = hudFor(D, R0, F);
    if (HUD.on && R) drawLock(ctx, bw, bh, R, D, F.shown != null, F.aspect, now, HUD, { gen: true, internals: true });
  }
  requestAnimationFrame(drawHud);

  async function tick() {
    try {
      const d = await (await fetch('/state')).json();
      if (d && d.generation !== undefined) {
        if (!D || D.trajectory !== d.trajectory) { if (!D) t0 = performance.now(); }
        D = d;
        $('h-gen').textContent = d.generation;
        $('h-fit').textContent = d.best_fitness;
        $('h-peak').textContent = d.peak_fitness_seen;
        $('h-look').textContent = `${d.fovea_fraction_accepted ?? '--'} of frame at birth`;
        $('h-pace').textContent = d.pace_accepted ? `resting ${((d.frames_per_second || 15) / d.pace_accepted).toFixed(1)} gazes/s` : '--';
        $('h-colour').textContent = ['none (light only)', 'red-green', 'red-green + blue-yellow'][d.colour_channels ?? 0] || '--';
        $('h-stab').textContent = d.stabilizer !== undefined ? `gain ${Number(d.stabilizer).toFixed(2)}` : '--';
        $('h-zoom').textContent = d.zoom_gain !== undefined ? (d.zoom_gain > 0 ? `gain ${Number(d.zoom_gain).toFixed(2)}` : 'off') : '--';
        $('h-metab').textContent = d.metabolism !== undefined ? `${Number(d.metabolism).toFixed(2)} (${d.metabolism >= 0.95 ? 'endotherm' : d.metabolism <= 0.15 ? 'ectotherm' : 'between'})` : '--';
        $('h-vigil').textContent = d.vigilance !== undefined ? Number(d.vigilance).toFixed(2) : '--';
        $('h-pump').textContent = d.pump !== undefined ? Number(d.pump).toFixed(1) + ' s/s' : '--';
        const rt = d.replay_traits;
        $('h-replay').textContent = rt ? (rt.awake || rt.sleep ? `awake ${rt.awake}, asleep ${rt.sleep} (REM ${Math.round(rt.rem_share * 100)}%)` : 'off') : '--';
        $('h-prey').textContent = ['eyes only', 'scent', 'scent + direction'][d.prey_sense ?? 0] || '--';
        $('h-quota').textContent = d.quota_pct !== undefined ? (d.host_cores ? `${(d.quota_pct / 100).toFixed(1)} of ${d.host_cores} cores` : d.quota_pct + '% of a core') : '--';  // systemd's % = one core
        $('h-stale').innerHTML = '';
        // each card on its own: one card's error must not blank the rest, and is recorded, not swallowed
        [[drawBody, d], [drawBrain, d], [drawSenses, d], [spaceData, d], [showLive, d]].forEach(([fn, x]) => guarded(fn, x));
        if (d.trees) guarded(() => renderTrees(d.trees, d.tree_stats, d.tree_limits));
      } else { $('h-stale').innerHTML = '<span class="stale">no live_status.json yet</span>'; }
    } catch (e) { $('h-stale').innerHTML = '<span class="stale">error polling /state</span>'; }
    setTimeout(tick, 1000);
  }
  tick();

  // Dessert: next occurrence of the chosen local time, as a unix deadline.
  function nextTime(hhmm) {
    const [h, m] = hhmm.split(':').map(Number), t = new Date();
    t.setHours(h, m, 0, 0);
    if (t.getTime() <= Date.now()) t.setDate(t.getDate() + 1);
    return t;
  }
  // Switch results: in plain words, a refusal in red (it used to be easy to miss).
  function switchStatus(text, bad) { const el = $('submit-status'); el.textContent = text; el.style.color = bad ? 'var(--red)' : ''; }
  async function postSelect(query) {
    const res = await fetch('/select?' + query, { method: 'POST', headers: { 'X-Cambrian': '1' } });
    let r = null; try { r = await res.json(); } catch (e) { }
    return r || { ok: false, error: `refused by the viewer (HTTP ${res.status})` };
  }
  $('custom-url-submit').addEventListener('click', async () => {
    const url = $('custom-url').value.trim(); if (!url) return;
    const until = $('dessert-timed').checked ? nextTime($('dessert-until').value || '07:00') : null;
    switchStatus('checking the video can be played...');
    try {
      const r = await postSelect('url=' + encodeURIComponent(url) + (until ? '&until=' + Math.floor(until.getTime() / 1000) : ''));
      if (r.ok) switchStatus('switching to it' + (until ? ` until ${until.toLocaleString()}` : ' until you switch back') + ' -- restarting (watch "watching" above)');
      else switchStatus('NOT switched: ' + (r.error || 'unknown reason'), true);
      if (r.ok) $('custom-url').value = '';
      pollDessert();
    } catch (e) { switchStatus('NOT switched: the viewer could not be reached', true); }
  });
  $('dessert-cancel').addEventListener('click', async () => {
    switchStatus('going back to the camera...');
    try {
      const r = await postSelect('name=auto');
      if (r.ok) switchStatus('back to camera -- the organism restarts onto it within about a minute');
      else switchStatus('NOT switched back: ' + (r.error || 'unknown reason'), true);
    } catch (e) { switchStatus('NOT switched back: the viewer could not be reached', true); }
    pollDessert();
  });
  async function pollDessert() {
    try {
      const r = await (await fetch('/sources')).json();
      SEL = r; showLive(D);
      $('dessert-status').textContent = r.active
        ? `on video: ${r.selected_url}` + (r.until ? ` until ${new Date(r.until * 1000).toLocaleString()}` : ' until you switch back')
        : 'on its camera';
    } catch (e) { }
  }
  pollDessert();
  setInterval(pollDessert, 5000);
</script>
</body>
</html>
""".replace("/*LOCK_HUD_JS*/", LOCK_HUD_JS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- this is a debug viewer, not a service worth logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PAGE.replace("__HOST__", HOST_NAME, 1).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/live" or self.path.startswith("/live?"):
            body = LIVE_PAGE.replace("__HOST__", HOST_NAME, 1).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            body = _read_shared(LIVE_STATUS_PATH) or b"{}"
            # The organism acting live (fishbowl/livelife.py): its newest frames
            # replace the generation's replay, when it is fresh and of this run.
            actor = LIVE_STATUS_PATH.with_name("live_actor.json")
            try:
                # Whichever is newer: its live body's frames, unless the
                # generation's snapshot came after them. (It was "the body's,
                # if under 10 s old": on a host whose main loop stalls longer
                # while it publishes -- 11-16 s on 7elwe -- the page fell back to
                # a snapshot tens of seconds stale, and its replay clock took
                # that for the floor: "delayed 38.5 s" with its body current.)
                if actor.exists() and (time.time() - actor.stat().st_mtime < 10.0
                                       or actor.stat().st_mtime >= LIVE_STATUS_PATH.stat().st_mtime):
                    d, a = json.loads(body or b"{}"), json.loads(_read_shared(actor) or b"{}")
                    if a.get("world_epoch") == d.get("world_epoch"):
                        d.update(a)
                        body = json.dumps(d, separators=(",", ":")).encode("utf-8")
            except (OSError, ValueError):
                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/organism/saves" or self.path.startswith("/organism/saves/"):
            # Its saved organisms: the list, or one file. To its own page (which
            # sends X-Cambrian) always; to another ecohost's viewer only while
            # this one is in the hive -- a solo ecohost shares nothing.
            from fishbowl import sandbox
            from tools import organism_file as of
            if self.headers.get("X-Cambrian") != "1" and not sandbox.hive():
                self.send_response(403)
                self.end_headers()
                return
            if self.path == "/organism/saves":
                body, ctype = json.dumps(of.list_saves()).encode("utf-8"), "application/json"
            else:
                f = self.path.rsplit("/", 1)[1]
                p = of.games_dir() / f
                if not of.safe_name(f) or not p.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                body, ctype = p.read_bytes(), "application/zip"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/organism/peer-saves"):
            # A hive peer's saves, asked by this viewer for its page (a page may
            # not ask another machine itself); only peers the page was shown.
            from tools import organism_file as of
            if self.headers.get("X-Cambrian") != "1":
                self.send_response(403)
                self.end_headers()
                return
            peer = _PEERS.get((parse_qs(urlparse(self.path).query).get("host") or [""])[0])
            try:
                if peer is None:
                    raise ValueError("not an ecohost of the hive (open the list again)")
                with urllib.request.urlopen(f"http://{peer[0]}:{peer[1]}/organism/saves", timeout=10) as r:
                    saves = json.loads(r.read(of.MAX_MEMBER_BYTES))
                saves = [e for e in saves if isinstance(e, dict) and of.safe_name(str(e.get("file")))]
                body = json.dumps({"saves": saves}).encode("utf-8")
            except (OSError, ValueError) as e:
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/organism/peers":
            # The hive's other live ecohosts (tools/fleet.py discover), for Copy
            # from; only these may be copied from (remembered in _PEERS).
            from fishbowl import sandbox
            from tools import fleet
            if self.headers.get("X-Cambrian") != "1":
                self.send_response(403)
                self.end_headers()
                return
            peers = [h for h in fleet.discover(None) if not h.get("local") and h.get("hive")] if sandbox.hive() else []
            _PEERS.clear()
            _PEERS.update({h["host"]: (h["addr"], h["port"]) for h in peers})
            body = json.dumps({"hive": sandbox.hive(), "peers": sorted(_PEERS)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/organism/save":
            # Save as... (the brain card): the whole organism as a .cambrioid
            # file (tools/organism_file.py). Only this page's own fetch sends
            # X-Cambrian (a peer's never does), so a solo organism is still
            # never handed to another machine's tools.
            from tools import organism_file
            if self.headers.get("X-Cambrian") != "1":
                self.send_response(403)
                self.end_headers()
                return
            try:
                body, _ = organism_file.save_bytes(STATE_DIR)
                name = organism_file.default_name(STATE_DIR)
            except (organism_file.Refused, OSError) as e:
                body, name = json.dumps({"ok": False, "error": str(e)}).encode("utf-8"), None
            self.send_response(200 if name else 409)
            self.send_header("Content-Type", "application/zip" if name else "application/json")
            if name:
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/organism/info", "/organism/checkpoint", "/organism/episodes"):
            # For tools/fleet.py (breeding across the fleet): what this host
            # is, and its organism of record -- genome, body, memory and its
            # episodes' Kenyon-cell codes. Numbers only, never frames.
            import platform, socket
            from fishbowl import sandbox
            in_hive = sandbox.hive()
            if self.path == "/organism/info":
                # "siblings": the first organism names any second one this host
                # has (a state-<i>/ with a checkpoint; parked, so today always
                # []), so the fleet asks port 8091 only where one exists.
                root = STATE_DIR.parent
                siblings = [] if _INSTANCE else sorted(p.name[len("state-"):] for p in root.glob("state-*") if (p / "checkpoint.json").exists())
                body = json.dumps({"hostname": socket.gethostname(), "instance": _INSTANCE, "siblings": siblings, "hive": in_hive,
                                   "platform": sys.platform, "machine": platform.machine(),
                                   "state_dir": str(STATE_DIR), "has_checkpoint": (STATE_DIR / "checkpoint.json").exists()}).encode("utf-8")
                ctype = "application/json"
            elif not in_hive and self.client_address[0] not in ("127.0.0.1", "::1"):
                # Solo: its organism stays on its own machine (the 2026-09-30 panel, Schneier).
                self.send_response(403)
                self.end_headers()
                return
            else:
                path = STATE_DIR / ("checkpoint.json" if self.path.endswith("checkpoint") else "episodes.npz")
                body = _read_shared(path)
                ctype = "application/json" if path.suffix == ".json" else "application/octet-stream"
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/metrics":
            # The hourly metrics (fishbowl/metrics.py): numbers only, the last three days.
            recs = _tail_jsonl(STATE_DIR / "metrics.jsonl", max_lines=72) if (STATE_DIR / "metrics.jsonl").exists() else []
            body = json.dumps([{k: r.get(k) for k in ("t", "bites", "nectar_sips")} for r in recs]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/history":
            # Real structural-growth telemetry: when complexity increases
            # and whether it earns its structural cost. A real tail of evolution_log.jsonl, not the whole
            # (potentially ~20000-line) file.
            # The log rotates by renaming (evolution_log.1.jsonl holds the
            # previous chunk): stitch the two so charts keep their history.
            recs = _tail_jsonl(EVOLUTION_LOG_PATH)
            if len(recs) < HISTORY_MAX_LINES and EVOLUTION_LOG_PREV_PATH.exists():
                recs = _tail_jsonl(EVOLUTION_LOG_PREV_PATH, max_lines=HISTORY_MAX_LINES - len(recs)) + recs
            body = json.dumps(_history_summary(recs)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/frame?"):
            # One frame of its recent past, by index (immutable per run and
            # index -- the page adds the run's epoch -- so cacheable).
            q = parse_qs(urlparse(self.path).query)
            i, e = (q.get("i") or [""])[0], (q.get("e") or ["0"])[0]
            try:
                body = (FRAMES_DIR / f"f{int(e)}_{int(i)}.jpg").read_bytes() if i.isdigit() and e.isdigit() else None
            except OSError:
                body = _frame_from_organism(e, i) if i.isdigit() and e.isdigit() else None
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "max-age=600, immutable")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/sources"):
            # Reports BOTH a name-based selection (for the dropdown,
            # which can only show pre-defined options) and a raw-url
            # one (for an honest status line) -- a real gap found live:
            # the dropdown was falling back to "auto" for a custom URL
            # selection, which is actually active, just not nameable.
            selected_name, selected_url, until = None, None, None
            if SELECTED_SOURCE_PATH.exists():
                try:
                    data = json.loads(SELECTED_SOURCE_PATH.read_text(encoding="utf-8-sig"))
                    selected_name, selected_url, until = data.get("name"), data.get("url"), data.get("until")
                except (OSError, json.JSONDecodeError):
                    pass
            body = json.dumps({
                "options": [n for n, _ in LIVE_SOURCES],
                "selected": selected_name,
                "selected_url": selected_url,
                "until": until,
                "active": bool(selected_url) and (until is None or float(until) > time.time()),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/select"):
            # State-changing: POST only (see do_POST). A GET here could be
            # triggered by any web page on the LAN (an <img> tag).
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.end_headers()
            return
        else:
            self.send_response(404)
            self.end_headers()


    def do_POST(self):
        # Only this page's own fetch() sends X-Cambrian; a cross-site form
        # can't set custom headers, and a cross-site fetch with one needs a
        # CORS preflight this server never grants.
        if not self.path.startswith(("/select", "/randomize", "/amnesia", "/reset-founder", "/save-here", "/upload", "/load-saved", "/copy-from")) or self.headers.get("X-Cambrian") != "1":
            print(f"select: refused (not from this page) {self.path[:120]}", flush=True)
            self.send_response(403)
            self.end_headers()
            return
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host", ""):
            print(f"select: refused (origin {origin} is not this page's host {self.headers.get('Host', '')})", flush=True)
            self.send_response(403)
            self.end_headers()
            return
        if self.path.startswith("/reset-founder") and not (STATE_DIR / "founder.json").exists():
            body = json.dumps({"ok": False, "error": "no founder kept for this lineage (born before founders were kept)"}).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/save-here"):
            # Save (the brain card): the .cambrioid written straight into this
            # machine's Games/Lifeforms/Cambrioids (the viewer's own account's:
            # yours on Windows and macOS, /srv/cambrian/... on Linux) -- no
            # browser dialog. Save as... (the download) is for other devices.
            from tools import organism_file
            try:
                where = organism_file.save(STATE_DIR)
                body = json.dumps({"ok": True, "path": str(where)}).encode("utf-8")
                print(f"save-here: {where}", flush=True)
            except (organism_file.Refused, OSError) as e:
                body = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self.send_response(200 if b'"ok": true' in body else 409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith(("/upload", "/load-saved", "/copy-from")):
            # The organism files (the brain card; tools/organism_file.py). All of
            # them live in THIS ecohost's Games/Lifeforms/Cambrioids folder:
            #   /upload?name=      a file from the device you view on, kept there
            #   /load-saved?file=  one of them loaded: staged for its next generation
            #   /copy-from?host=&file=  one of a hive peer's, pulled and kept there
            # Every arriving file is checked as a load checks it (sizes,
            # checksums, no pickles, its genome held to this ecohost's limits).
            from tools import organism_file as of
            qs = parse_qs(urlparse(self.path).query)
            arg = lambda k: (qs.get(k) or [""])[0]
            ok, error, extra = False, None, {}
            try:
                if self.path.startswith("/upload"):
                    n = int(self.headers.get("Content-Length") or 0)
                    if not 0 < n <= of.MAX_TOTAL_BYTES:
                        raise of.Refused("no file, or larger than a .cambrioid can be")
                    kept = of.store(self.rfile.read(n), arg("name"))
                    ok, extra = True, {"file": kept.name}
                elif self.path.startswith("/load-saved"):
                    f = arg("file")
                    if not of.safe_name(f) or not (of.games_dir() / f).is_file():
                        raise of.Refused("no such saved organism here")
                    m = of.stage(of.games_dir() / f, STATE_DIR)
                    ok, extra = True, {"name": m.get("name")}
                else:
                    host, f = arg("host"), arg("file")
                    peer = _PEERS.get(host)
                    if peer is None:
                        raise of.Refused("not an ecohost of the hive (open the list again)")
                    if not of.safe_name(f):
                        raise of.Refused("not a plain .cambrioid file name")
                    with urllib.request.urlopen(f"http://{peer[0]}:{peer[1]}/organism/saves/{f}", timeout=60) as r:
                        blob = r.read(of.MAX_TOTAL_BYTES + 1)
                    if len(blob) > of.MAX_TOTAL_BYTES:
                        raise of.Refused("larger than a .cambrioid can be")
                    kept = of.store(blob, f)
                    ok, extra = True, {"file": kept.name}
            except of.Refused as e:
                error = str(e)
            except (OSError, ValueError) as e:
                error = f"{type(e).__name__}: {e}"
            print(f"{self.path.split('?')[0].lstrip('/')}: {'ok ' + json.dumps(extra) if ok else 'refused (' + str(error) + ')'}", flush=True)
            body = json.dumps({"ok": ok, "error": error, **extra}).encode("utf-8")
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/reset-founder"):
            refused = _reset_refused("reset-founder")
            if refused is None:
                _write_json_atomic(STATE_DIR / "reset_founder.request", {"t": time.time()})
            print(f"reset to founder: {'requested' if refused is None else 'refused, ' + refused}", flush=True)
            body = json.dumps({"ok": refused is None, "error": refused}).encode("utf-8")
            self.send_response(200 if refused is None else 409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith(("/randomize", "/amnesia")):
            # The owner's Randomize (its brain drawn afresh) or Amnesia (a
            # fresh install's founder): the organism picks it up at its next
            # generation (run_vision.py).
            what = "randomize" if self.path.startswith("/randomize") else "amnesia"
            refused = _reset_refused(what)
            if refused is not None:
                print(f"{what}: refused, {refused}", flush=True)
                body = json.dumps({"ok": False, "error": refused}).encode("utf-8")
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            _write_json_atomic(STATE_DIR / f"{what}.request", {"t": time.time()})
            print(f"{what}: requested", flush=True)
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # "auto" clears the override, resuming normal round-robin.
        # A whitelisted name is fine as-is. Free-text "url" has no
        # whitelist to fall back on, so it's validated for real
        # against yt-dlp (see _check_live_url) before being
        # accepted. Still human-only; the organism never
        # reaches this endpoint or picks its own source.
        qs = parse_qs(urlparse(self.path).query)
        name = (qs.get("name") or [""])[0]
        url = (qs.get("url") or [""])[0].strip()
        until_raw = (qs.get("until") or [""])[0].strip()
        until = float(until_raw) if until_raw.replace(".", "", 1).isdigit() else None
        valid_names = {n for n, _ in LIVE_SOURCES}
        ok, error = False, None
        if name == "auto":
            try:
                _held(lambda: SELECTED_SOURCE_PATH.unlink(missing_ok=True))
                ok = True
            except OSError as e:
                error = f"couldn't switch back to the camera ({e.strerror or e}); try again"
        elif name in valid_names:
            try:
                _write_json_atomic(SELECTED_SOURCE_PATH, {"name": name})
                ok = True
            except OSError as e:
                error = f"couldn't save the choice ({e.strerror or e}); try again"
        elif url:
            ok, error = _check_live_url(url)
            try:
                was_stream = bool(json.loads(SELECTED_SOURCE_PATH.read_text(encoding="utf-8")).get("url"))
            except (OSError, ValueError):
                was_stream = False
            if ok:
                # Dessert: a deadline after which the organism goes
                # back to its camera by itself (run_vision.py _dessert).
                try:
                    _write_json_atomic(SELECTED_SOURCE_PATH, {"url": url, **({"until": until} if until else {})})
                except OSError as e:
                    ok, error = False, f"couldn't save the choice ({e.strerror or e}); try again"
        if ok:
            # Submitting restarts cambrian-perception.service --
            # without this the
            # selection only took effect on whatever restart
            # happened to come next (up to an hour away), which is
            # exactly what caused the real "still watching the
            # kittens" confusion. Confirmed passwordless sudo for
            # this exact command before wiring it in.
            # Elsewhere (Windows, macOS: tools/cambrian_service.py) there is
            # nothing to call: the organism checks the choice every
            # generation and restarts itself onto it.
            # From one stream to another it switches in place (run_vision.py):
            # no restart. To or from its camera, the camera suite needs one.
            if sys.platform.startswith("linux") and not (url and was_stream):
                try:
                    subprocess.run(
                        ["systemctl", "restart", "--no-ask-password", ORGANISM_UNIT],
                        capture_output=True, text=True, timeout=15,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass  # selection is still saved even if the restart trigger itself failed
        # Every switch attempt is logged (journal): a refused one used to leave
        # no trace but a line of small print on the page.
        print(f"select: {'switched to' if ok else 'refused'} {name or url or '(nothing)'}"
              + (f" -- {error}" if error else ""), flush=True)
        body = json.dumps({"ok": ok, "error": error}).encode("utf-8")
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global PAGE
    if _livecam_installed():  # checked once, at start
        PAGE = PAGE.replace('<span class="chip">CPU quota', SUITE_CHIP + '<span class="chip">CPU quota', 1)
    server = QuietHTTPServer((args.host, args.port), Handler)
    print(f"Viewer running at http://{args.host}:{args.port}/ (polls {LIVE_STATUS_PATH})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):  # UTF-8 into its log, whatever the platform's code page (run_vision.utf8_stdio)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError, OSError):
            pass
    sys.exit(main())
