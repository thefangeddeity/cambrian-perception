#!/usr/bin/env python3
from __future__ import annotations

"""
A tiny, dependency-free local viewer -- "garbage that over millions of
iterations becomes vision" (the user's own framing). Serves ONE page that
polls state/live_status.json and renders two blocky, pixelated
canvases from real backend-computed retina.py reductions -- genuinely
what it's seeing, not a reconstruction, since a 144-value luminance
grid is already reduced far past anything resembling real footage (see
run_vision.py's own comment on why including it there doesn't touch
the no-raw-frames rule): the RETINA (the fovea's own cropped view) and
the WORLD RETINA (the same reduction run on the full frame, i.e. what
run_vision.py's fitness grading itself sees), with the FOVEA's own box
(cx/cy/fraction, fovea.py's real pan/tilt window) drawn on top of the
world retina to show where it's currently pointed. User: "Screw the
video. What is a pixel dump of what it's seeing?" -- replaced an
earlier YouTube-embed crop-preview that kept hitting real, unfixable
constraints (embedding restrictions, cross-origin pixel access, URL
format parsing). "Don't get retina and fovea confused."

Deliberately NOT built into the HLSLS stack -- this has to work
whether or not broadcast-api/mediamtx are up (see the coordination
with HLSLS's own agent on that). Stdlib only, no new dependency.

Usage:
    python3 tools/viewer.py [--port 8090]
    Then open http://<host>:<port>/ in a browser.
"""

import argparse
import json
import subprocess
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# LIVE_SOURCES imported (not duplicated) from run_vision.py -- a
# second copy here would drift out of sync with the real list. Safe
# to import: run_vision.py only runs main() under __main__.
from run_vision import LIVE_SOURCES  # noqa: E402

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
LIVE_STATUS_PATH = STATE_DIR / "live_status.json"
# Paths defined independently here, not imported from fishbowl.sandbox
# -- same deliberate independence as everything else in this module
# (see its own module docstring: this has to work standalone).
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
HISTORY_MAX_LINES = 5000
HISTORY_MAX_BYTES = 12_000_000  # real tail, not a full-file read -- the
# log can grow to ~20000 lines/rotation; this stays cheap regardless.

# User: "put a list of training videos I can pick from the viewer."
# Human-only control -- this is the first thing this viewer ever
# WRITES (everything else is read-only). Written here, read by
# sandbox.load_selected_source() in run_vision.py. Only ever a name
# from LIVE_SOURCES's own fixed whitelist -- never an arbitrary URL,
# since this viewer has no auth (a known, documented gap) and
# accepting free-text URLs here would hand anyone on the LAN control
# over what real content the organism trains against.
SELECTED_SOURCE_PATH = STATE_DIR / "selected_source.json"


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    temp.replace(path)


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


def _check_live_url(url: str) -> tuple[bool, str | None]:
    """
    Real yt-dlp metadata check, not a URL-shape guess -- same
    discipline as every curated source added this session ("confirmed
    live via yt-dlp metadata before wiring in"). Free-text input has
    no whitelist to fall back on, so this IS the validation.
    """
    if not _allowed_url(url):
        return False, "only https youtube.com / youtu.be links are accepted"
    if not _URL_CHECK.acquire(blocking=False):
        return False, "another check is already running -- try again in a moment"
    yt_dlp = Path(sys.executable).parent / "yt-dlp"
    if not yt_dlp.exists():
        yt_dlp = Path("yt-dlp")
    try:
        # "--" so the URL can never be read as a yt-dlp option.
        result = subprocess.run(
            [str(yt_dlp), "--skip-download", "--print", "is_live", "--", url],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "yt-dlp check failed or timed out"
    finally:
        _URL_CHECK.release()
    if result.returncode != 0:
        return False, "not a resolvable video"
    if result.stdout.strip() != "True":
        return False, "not currently live"
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
PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cambrian-perception</title>
<style>
  :root { --bg:#0a0e14; --panel:#0d141c; --line:#1c2a36; --text:#cfe3f0; --dim:#6f8798; --cyan:#7fd4ff; --green:#4fa; --orange:#f90; --red:#f44; --yellow:#fd4; --violet:#c8f; --pink:#f6a; --magenta:#f4f; }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font: 13px/1.45 ui-monospace, Menlo, Consolas, monospace; margin: 0; padding: 16px 20px 40px; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 22px; margin-bottom: 14px; }
  h1 { font-size: 17px; font-weight: normal; color: var(--green); margin: 0; }
  .chip { color: var(--dim); } .chip b { color: var(--cyan); font-weight: normal; }
  h2 { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--cyan); margin: 0 0 4px; font-weight: normal; }
  .cap { color: var(--dim); font-size: 12px; margin: 0 0 10px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; min-width: 0; }
  .stack { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
  .vision { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr) minmax(340px, 0.95fr); gap: 16px; }
  .brainrow { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 16px; margin-top: 16px; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; margin-top: 16px; }
  @media (max-width: 1250px) { .vision, .brainrow { grid-template-columns: 1fr; } }
  @media (max-width: 480px) { body { padding: 10px; } .charts { grid-template-columns: 1fr; } }
  canvas { display: block; max-width: 100%; }
  canvas.px { image-rendering: pixelated; }
  .legend span { margin-right: 12px; white-space: nowrap; }
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
  <span class="chip">fitness <b id="h-fit">--</b></span>
  <span class="chip">peak ever <b id="h-peak">--</b></span>
  <span class="chip">watching <b id="h-src">--</b></span>
  <span class="chip">gaze size <b id="h-look">--</b></span>
  <span class="chip">pace <b id="h-pace">--</b></span>
  <span class="chip">CPU quota <b id="h-quota">--</b></span>
  <span class="chip" id="h-stale"></span>
</header>

<div class="vision">
  <div class="panel" id="field-panel">
    <h2>visual field</h2>
    <div class="cap">The whole scene as its coarse wide-field eyes get it: 12x12 light receptors (fixed for now -- an evolvable, metabolically priced receptor count is queued) over <span id="field-px">--</span> (like a jumping spider's secondary eyes). It feels threat, arousal and <em>where</em> something moved from this, not detail. The box is its <b style="color:var(--cyan)">gaze</b>, replayed along its real path over its latest run (the newest ~600 frames), at real speed, trail = last 3 s.</div>
    <canvas id="field" class="px"></canvas>
    <div class="legend cap" style="margin-top:8px">
      <span><b style="color:var(--green)">&#9633;</b> gaze, centered</span>
      <span><b style="color:var(--yellow)">&#9633;</b> near an edge</span>
      <span><b style="color:var(--orange)">&#9633;</b> on an edge</span>
      <span><b style="color:var(--red)">&#9633;</b> in a corner</span>
      <span><b style="color:#8cff5a">&#9679;</b> where its wide-field eyes saw motion this frame (size = how much) -- its brain gets this location, so it can learn to swing its gaze there</span>
      <span><b style="color:var(--red)">red frame</b> something dark approaching (a flinch is rewarded if it reacts within 3 frames)</span>
    </div>
    <div class="cap" id="replay-clock">--</div>
  </div>

  <div class="stack">
    <div class="panel" id="look-panel">
      <h2>gaze</h2>
      <div class="cap">Its movable high-acuity eye (the spider's principal retina): the same 12x12 receptors over a smaller patch (<span id="look-px">--</span>), magnified. The dashed frame is the widest it can open; the gaze sits centered inside at its true relative size, so you can watch it widen and narrow. The only place it sees detail, and the only way it eats. View at the end of its latest run.</div>
      <canvas id="look" class="px"></canvas>
      <div class="cap" style="margin-top:8px" id="look-scale"></div>
    </div>
    <div class="panel" id="dessert-card">
      <h2>dessert</h2>
      <div class="cap">Switch it from the camera to a live YouTube stream -- to speed up learning with more going on, or overnight when the room is asleep. It stays on the video until you press "back to camera", or optionally until a set time. The stream is checked to really be live first; frames are never saved.</div>
      <div id="dessert-status" class="cap" style="color:var(--cyan)">--</div>
      <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:6px">
        <input type="text" id="custom-url" placeholder="paste a live YouTube URL..." style="flex:1 1 260px">
        <button id="custom-url-submit">switch to this video</button>
        <button id="dessert-cancel">back to camera now</button>
      </div>
      <label class="cap" style="display:block; margin-top:6px"><input type="checkbox" id="dessert-timed"> go back to the camera by itself at <input type="time" id="dessert-until" value="07:00"></label>
      <div id="submit-status" class="cap" style="margin-top:6px"></div>
    </div>
  </div>

  <div class="panel">
    <h2>body</h2>
    <div class="cap">Gemini's homeostasis (fishbowl/state.py) -- its body <b>right now</b>. It persists through the day and across restarts: hours of stillness really drain it; a rich stream feeds it back up. The traces below are its latest run.</div>
    <div id="gauges"></div>
    <canvas id="energy-trace" height="60"></canvas>
    <div class="section"><h2>eating</h2>
      <div class="cap">Food = surprise through its gaze: change beyond what each spot usually does (it remembers each spot's usual brightness and how much it usually varies). Sensor noise never feeds it; a swinging fan feeds it only until the swing becomes expected; something new in a still corner is a big meal.</div>
      <div id="food-gauge"></div>
      <canvas id="food-trace" height="60"></canvas>
    </div>
    <div class="section"><h2>how it moves</h2>
      <div class="cap">Measured, not rewarded. Yardstick: Land 1969, jumping-spider retinae (fixate / glide / saccade, scanning still things, tracking moving ones).</div>
      <div id="movement"></div>
    </div>
  </div>
</div>

<div class="brainrow">
  <div class="panel">
    <h2>its brain</h2>
    <div class="cap">Gemini's recurrent network (fishbowl/controller.py) that moves the gaze: 19 inputs (incl. the perception tree's output) &rarr; 16 recurrent units &rarr; pan / tilt / zoom / alarm / tempo. Its memory units also feed the perception tree (right). Lines are the evolved weights of the current accepted genome (<b style="color:var(--cyan)">cyan</b> excitatory, <b style="color:var(--orange)">orange</b> inhibitory, brighter = stronger). Unit fill = its activity at the end of the latest run. Right: the recurrent weights (unit &rarr; unit), which carry its memory from frame to frame.</div>
    <canvas id="brain" height="520"></canvas>
  </div>
  <div class="panel">
    <h2>its perception tree</h2>
    <div class="cap">The genome's evolved "response" tree: reads the gaze's 12x12 cells (x0-x143), the previous frame's (x144-x287), its own last movement (x288-x289) and the brain's 16 memory units (x290-x305). Its output goes into the brain as the "tree" input -- it is no longer graded by any score of its own, so it only matters if what it perceives helps the body.</div>
    <div id="trees"></div>
  </div>
</div>

<div class="charts" id="charts"></div>

<div class="panel" style="margin-top:16px">
  <h2>what drives it</h2>
  <div class="cap">Every pressure currently in the fitness, with its real weight in run_vision.py. <span class="tag body">body</span> = comes from staying alive (the direction this is going: Dennett's "whole iguana"). <span class="tag hand">hand-written</span> = an older score bolted on from outside, to be retired one at a time as the body takes over. Nothing is hard-wired: the old innate escape reflex was removed so the flinch can evolve.</div>
  <table class="drives">
    <tr><th></th><th>weight</th><th>pressure</th><th>what it means</th></tr>
    <tr><td><span class="tag body">body</span></td><td class="minus">-3.0</td><td>homeostatic drive</td><td>Mean over the run of (1-energy)&sup2; + threat&sup2; + fatigue&sup2;. The biggest term. Energy is spent every frame on basal metabolism (0.003 + 0.003 &times; arousal), muscle force (0.010 &times; force&sup2;) and gaze size (0.01 &times; area &times; 150 / CPU quota, so a wide gaze costs more when CPU is scarce), and only restored by eating (up to 0.012 per frame).</td></tr>
    <tr><td><span class="tag body">body</span></td><td>trait + motor</td><td>tempo / pace of life</td><td>How often it gazes. Inherited resting pace (every 1st-6th frame) plus a brain output that speeds up or slows down 3x either way, any time -- a continuum, not a fixed type. Its metabolic rate acclimatizes to its tempo over ~2 minutes (slowing down pays only once it has been slow a while, like a bear's winter). Time runs the same for all; each gaze costs compute plus the gaze-size cost. Slow = cheaper, fewer meals, slower reactions.</td></tr>
    <tr><td><span class="tag body">body</span></td><td>input</td><td>hunger, search, curiosity</td><td>Not scored directly: they are what it feels. Hunger builds while energy is low and drives search; curiosity grows while nothing new comes in and drops when it eats novelty. All three feed its brain.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="plus">+1.0</td><td>flinch</td><td>Not wired in: when something dark starts expanding anywhere in the whole field (locust-LGMD style, dark-only per Yilmaz &amp; Meister 2013), it earns up to +1 for widening its gaze or making a saccade within 3 frames of real time, more for faster. No approach, no reward, no penalty. The flinch has to evolve.</td></tr>
    <tr><td><span class="tag hand" style="text-decoration:line-through">retired</span></td><td>0</td><td>correlation scores</td><td>luminance_change, motion_energy, directional_motion, loom (old detector), conspec_drive, alarm, optokinetic_pursuit. Retired 2026-09-26 after two independent audits: they carried 80-95% of selection while moving nothing in the body (the perception tree memorised clips to satisfy them). Still measured and logged, not scored.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="plus">+18 &times;</td><td>curiosity (gaze)</td><td>Rate of reaching new gaze positions (5x5 grid). Overlaps with eating novelty now.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="plus">+1.0</td><td>seek</td><td>Ending up near a detected face-like pattern.</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="minus">-1.0 / -0.5</td><td>dead_field / movement_cost</td><td>Sustained stretches with nothing happening in the world; pushing the eye at all (on top of the body's energy cost for force).</td></tr>
    <tr><td><span class="tag hand">hand-written</span></td><td class="minus">-1.0 / -0.5</td><td>corner / edge penalty</td><td>Sitting in a corner, or hard against one edge.</td></tr>
  </table>
</div>

<script>
  const $ = id => document.getElementById(id);
  const INPUT_NAMES = ['light', 'motion', 'flow x', 'flow y', 'loom', 'gaze x', 'gaze y', 'zoom', 'energy', 'arousal', 'threat', 'search', 'motion dx', 'motion dy', 'eye vx', 'eye vy', 'hunger', 'curiosity', 'tree'];
  const OUTPUT_NAMES = ['pan', 'tilt', 'zoom', 'alarm', 'tempo'];
  const REPLAY_FPS = 15;
  let D = null, t0 = performance.now();

  function innerWidth(el) { const cs = getComputedStyle(el); return el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight); }

  function drawGrid(ctx, values, shape, x, y, w, h) {
    const [rows, cols] = shape;
    for (let i = 0; i < rows; i++) {
      const y0 = Math.round(y + i * h / rows), y1 = Math.round(y + (i + 1) * h / rows);
      for (let j = 0; j < cols; j++) {
        const x0 = Math.round(x + j * w / cols), x1 = Math.round(x + (j + 1) * w / cols);
        const g = Math.round(Math.max(0, Math.min(1, values[i * cols + j])) * 255);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
  }
  function crop(d, f) { return [2 * Math.floor(d.frame_w * f / 2), 2 * Math.floor(d.frame_h * f / 2)]; }
  function boxColor(cx, cy, f) {
    const hr = 0.5 - f / 2;
    const nx = hr > 1e-9 ? (cx - 0.5) / hr : 0, ny = hr > 1e-9 ? (cy - 0.5) / hr : 0;
    const corner = Math.abs(nx * ny), edge = Math.max(Math.abs(nx), Math.abs(ny));
    return corner >= 0.5 ? '#f44' : edge >= 0.75 ? '#f90' : edge >= 0.4 ? '#fd4' : '#4fa';
  }

  // Visual field + replay of the gaze's real path, frame by frame, no interpolation.
  function drawField(now) {
    requestAnimationFrame(drawField);
    const d = D;
    if (!d || !d.frame_w || !d.world_grid) return;
    const c = $('field'), W = Math.max(200, Math.floor(innerWidth($('field-panel'))));
    const H = Math.round(W * d.frame_h / d.frame_w);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d');
    drawGrid(ctx, d.world_grid, d.world_grid_shape, 0, 0, W, H);
    const traj = d.trajectory && d.trajectory.length ? d.trajectory : [[d.fovea_cx, d.fovea_cy, d.fovea_fraction || 0.35, 0]];
    // Gazes are unevenly spaced (its tempo changes): replay in real frame
    // time and show whichever gaze is current at that moment.
    const fps = d.frames_per_second || REPLAY_FPS;
    const lastIdx = traj[traj.length - 1][3] ?? (traj.length - 1);
    const cur = Math.floor((now - t0) / 1000 * fps) % (lastIdx + 1);
    let i = 0; while (i + 1 < traj.length && (traj[i + 1][3] ?? (i + 1)) <= cur) i++;
    const ev = d.field_events && d.field_events[i];
    if (ev) {
      const [mx, my, act, loom, reflex] = ev;
      if (act > 0.05) {
        ctx.fillStyle = 'rgba(140, 255, 90, 0.85)';
        ctx.beginPath(); ctx.arc(mx * W, my * H, 3 + act * 14, 0, 7); ctx.fill();
      }
      if (loom > 0.18) { ctx.strokeStyle = '#f44'; ctx.lineWidth = 6; ctx.strokeRect(3, 3, W - 6, H - 6); }
    }
    ctx.strokeStyle = 'rgba(127, 212, 255, 0.45)'; ctx.lineWidth = 1.5; ctx.beginPath();
    let k0 = i; while (k0 > 0 && (traj[k0 - 1][3] ?? (k0 - 1)) >= cur - 3 * fps) k0--;
    for (let k = k0; k <= i; k++) { const px = traj[k][0] * W, py = traj[k][1] * H; k === k0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
    ctx.stroke();
    const [cx, cy, f] = traj[i];
    const [cw, ch] = crop(d, f), s = W / d.frame_w;
    ctx.strokeStyle = boxColor(cx, cy, f); ctx.lineWidth = 3;
    ctx.strokeRect(cx * W - cw * s / 2, cy * H - ch * s / 2, cw * s, ch * s);
    $('replay-clock').textContent = `replay: gaze ${i + 1} / ${traj.length}  (t = ${(cur / fps).toFixed(1)} s of ${((lastIdx + 1) / fps).toFixed(0)} s)` + (ev && ev[3] > 0.18 ? '  -- APPROACH' : '');
  }
  requestAnimationFrame(drawField);

  function drawLook(d) {
    if (!d.grid || !d.frame_w) return;
    const f = d.fovea_fraction || 0.35, fmax = d.max_fraction || 0.6;
    const [cw, ch] = crop(d, f), [mw, mh] = crop(d, fmax);
    const c = $('look'), W = Math.max(160, Math.floor(innerWidth($('look-panel')))), H = Math.round(W * mh / mw);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = c.getContext('2d'), s = W / mw;
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#2a3c4c'; ctx.setLineDash([6, 5]); ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, H - 1); ctx.setLineDash([]);
    const lw = cw * s, lh = ch * s, lx = (W - lw) / 2, ly = (H - lh) / 2;
    drawGrid(ctx, d.grid, d.grid_shape, lx, ly, lw, lh);
    ctx.strokeStyle = '#7fd4ff'; ctx.lineWidth = 2; ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = '#6f8798'; ctx.font = '11px monospace'; ctx.fillText(`largest possible gaze (${fmax} of frame)`, 6, 14);
    $('look-px').textContent = `${cw}x${ch} real pixels`;
    $('field-px').textContent = `${d.frame_w}x${d.frame_h} real pixels`;
    $('look-scale').textContent = `Its gaze is ${(f).toFixed(3)} of the frame -- ${(100 * f / fmax).toFixed(0)}% of the widest it can open (dashed frame). Shown ${s.toFixed(1)}x real size.`;
  }

  function gauge(name, v, color, note) {
    const x = Math.max(0, Math.min(1, v || 0));
    return `<div class="gauge"><span>${name}</span><div class="track"><div class="fill" style="width:${(x * 100).toFixed(1)}%;background:${color}"></div></div><span class="v">${(v || 0).toFixed(2)}</span></div>` + (note ? `<div class="note">${note}</div>` : '');
  }
  function spark(id, series, color, label) {
    const c = $(id); if (!c) return;
    c.width = Math.max(200, Math.floor(innerWidth(c.parentElement)));
    const ctx = c.getContext('2d'), W = c.width, H = c.height;
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#1c2a36'; [0, 0.5, 1].forEach(v => { const y = (1 - v) * (H - 16) + 2; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); });
    if (series && series.length > 1) {
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
      series.forEach((v, k) => { const x = k / (series.length - 1) * W, y = (1 - Math.max(0, Math.min(1, v))) * (H - 16) + 2; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    ctx.fillStyle = '#6f8798'; ctx.font = '10px monospace'; ctx.fillText(label + ' over its latest run (0..1, start -> end)', 2, H - 3);
  }
  function drawBody(d) {
    const b = d.body_now || d.body || {};
    $('gauges').innerHTML =
      gauge('energy', b.energy, '#4fa') + gauge('hunger', b.hunger, '#f6a', 'builds while energy is low') +
      gauge('search', b.search, '#fd4', 'urge to look around, driven by hunger') + gauge('curiosity', b.curiosity, '#c8f', 'appetite for something new') +
      gauge('arousal', b.arousal, '#7fd4ff', 'from motion anywhere in the field') + gauge('threat', b.threat, '#f44', 'from something dark approaching') +
      gauge('fatigue', b.fatigue, '#f90', 'from forceful eye movement') +
      `<div class="cap" style="margin-top:4px">tempo: <b style="color:var(--cyan)">${d.pace ? ((d.frames_per_second || 15) / d.pace).toFixed(1) : '--'}</b> gazes/s on average in its latest run (resting: ${d.pace_accepted ? ((d.frames_per_second || 15) / d.pace_accepted).toFixed(1) : '--'}). It can speed up or slow down 3x either way, any time -- a continuum, like a bear sluggish in winter and hyper in spring.</div>` +
      gauge('metabolism', b.metabolic_rate ?? 1, '#fd4', 'acclimatizes to its tempo over ~2 min (1 = gazing every frame)') +
      (d.flinch ? `<div class="cap" style="margin-top:4px">flinch: <b style="color:var(--cyan)">${d.flinch.events}</b> approaches in its latest run, reacted to <b style="color:var(--cyan)">${d.flinch.reacted}</b>` + (d.flinch.mean_latency_frames !== null ? `, on average ${(d.flinch.mean_latency_frames / (d.frames_per_second || 15) * 1000).toFixed(0)} ms after onset` : '') + '</div>' : '');
    spark('energy-trace', d.energy_series, '#4fa', 'energy');
    $('food-gauge').innerHTML = gauge('food', d.mean_food, '#c8f', 'average per frame over its latest run');
    spark('food-trace', d.food_series, '#c8f', 'food');
    const m = d.movement || {};
    $('movement').innerHTML =
      gauge('fixating', m.fixate, '#6f8798', 'share of frames still') + gauge('gliding', m.glide, '#4fa', 'slow, smooth') +
      gauge('saccades', m.saccade, '#f90', 'fast jumps') + gauge('scanning', m.scan_while_still, '#7fd4ff', 'gliding while the world is still') +
      `<div class="gauge"><span>tracking</span><span class="cap" style="margin:0">${m.tracking === null || m.tracking === undefined ? 'not enough movement in the room this run' : 'correlation ' + m.tracking.toFixed(2) + ' with where motion was'}</span><span></span></div>`;
  }

  // The whole recurrent brain.
  function drawBrain(d) {
    const br = d.brain; if (!br) return;
    const c = $('brain'), W = Math.max(500, Math.floor(innerWidth(c.parentElement))), H = c.height;
    c.width = W;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    const nIn = br.weights_ih[0].length, nH = br.weights_ih.length, nOut = br.weights_ho.length;
    const hm = Math.min(220, W * 0.25), netW = W - hm - 40;
    const xin = 90, xh = xin + (netW - 90) * 0.5, xout = netW - 50;
    const yAt = (k, n) => 24 + k * (H - 48) / Math.max(1, n - 1);
    let maxW = 1e-9; br.weights_ih.forEach(r => r.forEach(v => maxW = Math.max(maxW, Math.abs(v)))); br.weights_ho.forEach(r => r.forEach(v => maxW = Math.max(maxW, Math.abs(v))));
    const edge = (x1, y1, x2, y2, w) => { const a = Math.min(1, Math.abs(w) / maxW); ctx.strokeStyle = w >= 0 ? `rgba(127,212,255,${0.08 + 0.8 * a})` : `rgba(255,153,0,${0.08 + 0.8 * a})`; ctx.lineWidth = 0.5 + 2 * a; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); };
    for (let h = 0; h < nH; h++) for (let i = 0; i < nIn; i++) edge(xin, yAt(i, nIn), xh, yAt(h, nH), br.weights_ih[h][i]);
    for (let o = 0; o < nOut; o++) for (let h = 0; h < nH; h++) edge(xh, yAt(h, nH), xout, yAt(o, nOut) , br.weights_ho[o][h]);
    ctx.font = '11px monospace'; ctx.textBaseline = 'middle';
    for (let i = 0; i < nIn; i++) { ctx.fillStyle = '#1a2a38'; ctx.beginPath(); ctx.arc(xin, yAt(i, nIn), 6, 0, 7); ctx.fill(); ctx.fillStyle = '#9fb6c6'; ctx.textAlign = 'right'; ctx.fillText(INPUT_NAMES[i] || ('in ' + i), xin - 10, yAt(i, nIn)); }
    const hid = d.brain_hidden || [];
    for (let h = 0; h < nH; h++) { const a = hid[h] || 0; ctx.fillStyle = a >= 0 ? `rgba(127,212,255,${0.15 + 0.85 * Math.abs(a)})` : `rgba(255,153,0,${0.15 + 0.85 * Math.abs(a)})`; ctx.strokeStyle = '#345'; ctx.beginPath(); ctx.arc(xh, yAt(h, nH), 9, 0, 7); ctx.fill(); ctx.stroke(); }
    for (let o = 0; o < nOut; o++) { ctx.fillStyle = '#0a2a1a'; ctx.strokeStyle = '#4fa'; ctx.beginPath(); ctx.arc(xout, yAt(o, nOut), 11, 0, 7); ctx.fill(); ctx.stroke(); ctx.fillStyle = '#4fa'; ctx.textAlign = 'left'; ctx.fillText(OUTPUT_NAMES[o] || ('out ' + o), xout + 16, yAt(o, nOut)); }
    const hx = W - hm - 10, hy = 30, cell = hm / nH;
    let mh = 1e-9; br.weights_hh.forEach(r => r.forEach(v => mh = Math.max(mh, Math.abs(v))));
    for (let r = 0; r < nH; r++) for (let q = 0; q < nH; q++) { const v = br.weights_hh[r][q], a = Math.abs(v) / mh; ctx.fillStyle = v >= 0 ? `rgba(127,212,255,${a})` : `rgba(255,153,0,${a})`; ctx.fillRect(hx + q * cell, hy + r * cell, cell - 1, cell - 1); }
    ctx.fillStyle = '#6f8798'; ctx.textAlign = 'left'; ctx.fillText('recurrent weights', hx, hy - 12);
    ctx.fillText('from unit ->  (rows: to unit)', hx, hy + hm + 14);
  }

  // Trees (every tree the genome has).
  function nodeLabel(n) { return n.kind === 'var' ? 'x' + n.index : n.kind === 'const' ? n.value.toFixed(2) : n.op; }
  function layout(n, depth, order) {
    if (!n.children || n.children.length === 0) return { node: n, depth, x: order.next++, children: [] };
    const kids = n.children.map(c => layout(c, depth + 1, order));
    return { node: n, depth, x: kids.reduce((s, k) => s + k.x, 0) / kids.length, children: kids };
  }
  function maxDepthOf(l) { return l.children.length ? 1 + Math.max(...l.children.map(maxDepthOf)) : 0; }
  function renderTrees(trees, stats, limits) {
    const box = $('trees'); box.innerHTML = '';
    for (const name of Object.keys(trees)) {
      const wrap = document.createElement('div');
      const s = stats && stats[name];
      wrap.innerHTML = `<div class="cap">${name}${s && limits ? ` -- ${s.nodes} / ${limits.max_nodes} nodes, depth ${s.depth} / ${limits.max_depth}` : ''}</div>`;
      const c = document.createElement('canvas'); wrap.appendChild(c); box.appendChild(wrap);
      const order = { next: 0 }, laid = layout(trees[name], 0, order), leaves = Math.max(1, order.next), md = maxDepthOf(laid);
      c.width = Math.max(300, (leaves + 1) * 52); c.height = Math.max(180, (md + 2) * 58);
      const ctx = c.getContext('2d'); ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, c.width, c.height);
      const xs = c.width / (leaves + 1), ys = c.height / (md + 2), pos = l => [(l.x + 1) * xs, (l.depth + 1) * ys];
      (function edges(l) { const [px, py] = pos(l); for (const k of l.children) { const [qx, qy] = pos(k); ctx.strokeStyle = '#345'; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(qx, qy); ctx.stroke(); edges(k); } })(laid);
      (function nodes(l) { const [x, y] = pos(l); const op = l.node.kind === 'op'; ctx.fillStyle = op ? '#0a2a1a' : '#1a1a2a'; ctx.strokeStyle = op ? '#4fa' : '#7fd4ff'; ctx.beginPath(); ctx.arc(x, y, 18, 0, 7); ctx.fill(); ctx.stroke(); ctx.fillStyle = '#cfe3f0'; ctx.font = '11px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(nodeLabel(l.node), x, y); l.children.forEach(nodes); })(laid);
    }
  }

  // History charts.
  const CHARTS = [
    { id: 'c-fit', title: 'fitness', cap: 'current genome, re-scored each generation / peak ever', series: [['fitness', '#4fa', r => r.best_fitness], ['peak ever', '#6f8798', r => r.peak_fitness_seen]] },
    { id: 'c-body', title: 'body over its runs', cap: 'mean energy and mean food per run (0..1)', fixed: [0, 1], series: [['energy', '#4fa', r => r.mean_energy], ['food', '#c8f', r => r.mean_food]] },
    { id: 'c-drive', title: 'homeostatic drive', cap: 'mean drive per run -- lower is healthier', series: [['drive', '#f6a', r => r.mean_drive]] },
    { id: 'c-look', title: 'gaze size', cap: 'inherited gaze size at birth, and mean gaze size over each run (fraction of frame)', fixed: [0, 0.65], series: [['at birth', '#7fd4ff', r => r.fovea_fraction], ['mean in run', '#c8f', r => r.mean_aperture]] },
    { id: 'c-pace', title: 'resting pace', cap: 'inherited resting gaze interval, every Nth frame (its temperament; the brain moves 3x either way around it)', series: [['every Nth frame', '#7fd4ff', r => r.pace]] },
    { id: 'c-quota', title: 'CPU quota granted', cap: 'resource_handler: grows with real improvement, shrinks under system strain (%)', series: [['quota %', '#fd4', r => r.quota_pct]] },
    { id: 'c-move', title: 'how it moves', cap: 'share of frames fixating / gliding / in saccades', fixed: [0, 1], series: [['fixate', '#6f8798', r => r.mv && r.mv.fixate], ['glide', '#4fa', r => r.mv && r.mv.glide], ['saccade', '#f90', r => r.mv && r.mv.saccade]] },
    { id: 'c-tree', title: 'perception tree size', cap: 'response tree nodes / depth', series: [['nodes', '#f90', r => r.tree_nodes], ['depth', '#7fd4ff', r => r.tree_depth]] },
    { id: 'c-delta', title: 'fitness gain of accepted changes', cap: 'how much each accepted change earned', series: [['gain', '#4fa', r => r.accepted_delta]] },
    { id: 'c-mut', title: 'accepted changes by kind', cap: 'which mutation won, over time', mutations: true },
  ];
  const MUT = ['mutate_brain', 'mutate_pace', 'mutate_fovea', 'mutate_const', 'mutate_op', 'grow', 'shrink', 'reroll_subtree'];
  const MUT_COLOR = { mutate_pace: '#fd4', mutate_brain: '#f4f', mutate_fovea: '#c8f', mutate_const: '#4fa', mutate_op: '#0af', grow: '#7fd4ff', shrink: '#f90', reroll_subtree: '#f66' };
  (function buildCharts() {
    $('charts').innerHTML = CHARTS.map(ch => `<div class="panel"><h2>${ch.title}</h2><div class="cap">${ch.cap}</div>` +
      (ch.series ? `<div class="legend cap">${ch.series.map(s => `<span><b style="color:${s[1]}">&#9644;</b> ${s[0]}</span>`).join('')}</div>` : '') +
      `<canvas id="${ch.id}" height="190"></canvas></div>`).join('');
  })();
  function lineChart(ch, recs) {
    const c = $(ch.id); c.width = Math.max(300, Math.floor(innerWidth(c.parentElement)));
    const ctx = c.getContext('2d'), W = c.width, H = c.height; ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 46, r: 8, t: 8, b: 18 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    let lo = Infinity, hi = -Infinity;
    ch.series.forEach(s => recs.forEach(r => { const v = s[2](r); if (v !== null && v !== undefined && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    if (ch.fixed) { lo = ch.fixed[0]; hi = ch.fixed[1]; }
    if (!isFinite(lo)) { ctx.fillStyle = '#6f8798'; ctx.font = '11px monospace'; ctx.fillText('no data yet (recorded from this build on)', pad.l, H / 2); return; }
    if (lo === hi) { lo -= 1; hi += 1; }
    const n = recs.length, px = i => pad.l + (n <= 1 ? 0 : i / (n - 1)) * pw, py = v => pad.t + (1 - (v - lo) / (hi - lo)) * ph;
    ctx.strokeStyle = '#1c2a36'; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph); ctx.stroke();
    ctx.fillStyle = '#6f8798'; ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.fillText(hi.toFixed(2), pad.l - 4, pad.t + 8); ctx.fillText(lo.toFixed(2), pad.l - 4, pad.t + ph);
    ctx.textAlign = 'left'; ctx.fillText(recs.label_left || 'older', pad.l, H - 4); ctx.textAlign = 'right'; ctx.fillText('now', pad.l + pw, H - 4);
    ch.series.forEach(s => { ctx.strokeStyle = s[1]; ctx.lineWidth = 1.5; ctx.beginPath(); let on = false;
      recs.forEach((r, i) => { const v = s[2](r); if (v === null || v === undefined || !isFinite(v)) { on = false; return; } on ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)); on = true; }); ctx.stroke(); });
  }
  function mutChart(ch, recs) {
    const c = $(ch.id); c.width = Math.max(300, Math.floor(innerWidth(c.parentElement)));
    const ctx = c.getContext('2d'), W = c.width, H = c.height; ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 104, r: 8, t: 8, b: 18 }, pw = W - pad.l - pad.r, rh = (H - pad.t - pad.b) / MUT.length;
    ctx.font = '10px monospace'; ctx.textAlign = 'left';
    MUT.forEach((t, k) => { ctx.fillStyle = MUT_COLOR[t]; ctx.fillText(t, 2, pad.t + k * rh + rh / 2 + 3); });
    const n = recs.length;
    recs.forEach((r, i) => (r.accepted_types || []).forEach(t => { const k = MUT.indexOf(t); if (k < 0) return; ctx.fillStyle = MUT_COLOR[t]; ctx.beginPath(); ctx.arc(pad.l + (n <= 1 ? 0 : i / (n - 1)) * pw, pad.t + k * rh + rh / 2, 3, 0, 7); ctx.fill(); }));
  }
  async function fetchHistory() {
    try {
      const h = await (await fetch('/history')).json();
      const recs = h.records || []; recs.label_left = h.span_generations ? `${h.span_generations} generations ago` : 'older';
      CHARTS.forEach(ch => ch.mutations ? mutChart(ch, recs) : lineChart(ch, recs));
    } catch (e) { }
    setTimeout(fetchHistory, 20000);
  }
  fetchHistory();

  async function tick() {
    try {
      const d = await (await fetch('/state')).json();
      if (d && d.generation !== undefined) {
        if (!D || D.trajectory !== d.trajectory) { if (!D) t0 = performance.now(); }
        D = d;
        $('h-gen').textContent = d.generation;
        $('h-fit').textContent = d.best_fitness;
        $('h-peak').textContent = d.peak_fitness_seen;
        $('h-src').textContent = d.clip_name || d.clip || '--';
        $('h-look').textContent = `${d.fovea_fraction_accepted ?? '--'} of frame at birth`;
        $('h-pace').textContent = d.pace_accepted ? `resting ${((d.frames_per_second || 15) / d.pace_accepted).toFixed(1)} gazes/s` : '--';
        $('h-quota').textContent = d.quota_pct !== undefined ? d.quota_pct + '%' : '--';
        $('h-stale').innerHTML = '';
        drawLook(d); drawBody(d); drawBrain(d);
        if (d.trees) renderTrees(d.trees, d.tree_stats, d.tree_limits);
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
  $('custom-url-submit').addEventListener('click', async () => {
    const url = $('custom-url').value.trim(); if (!url) return;
    const until = $('dessert-timed').checked ? nextTime($('dessert-until').value || '07:00') : null;
    $('submit-status').textContent = 'checking it is really live...';
    try {
      const r = await (await fetch('/select?url=' + encodeURIComponent(url) + (until ? '&until=' + Math.floor(until.getTime() / 1000) : ''), { method: 'POST', headers: { 'X-Cambrian': '1' } })).json();
      $('submit-status').textContent = r.ok ? ('switching to it' + (until ? ` until ${until.toLocaleString()}` : ' until you switch back') + ' -- restarting (watch "watching" above)') : ('not switched: ' + (r.error || 'unknown'));
      if (r.ok) $('custom-url').value = '';
      pollDessert();
    } catch (e) { $('submit-status').textContent = 'failed'; }
  });
  $('dessert-cancel').addEventListener('click', async () => {
    $('submit-status').textContent = 'going back to the camera...';
    try { await fetch('/select?name=auto', { method: 'POST', headers: { 'X-Cambrian': '1' } }); $('submit-status').textContent = 'back to camera -- restarting'; } catch (e) { $('submit-status').textContent = 'failed'; }
    pollDessert();
  });
  async function pollDessert() {
    try {
      const r = await (await fetch('/sources')).json();
      $('dessert-status').textContent = r.active
        ? `on video: ${r.selected_url}` + (r.until ? ` until ${new Date(r.until * 1000).toLocaleString()}` : ' until you switch back')
        : 'on its camera';
    } catch (e) { }
  }
  pollDessert();
  setInterval(pollDessert, 15000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- this is a debug viewer, not a service worth logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            if LIVE_STATUS_PATH.exists():
                body = LIVE_STATUS_PATH.read_bytes()
            else:
                body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/history":
            # Real structural-growth telemetry -- User: "plot fitness,
            # total nodes, nodes per channel, tree depth, accepted
            # mutation type, fitness delta... show exactly when
            # complexity increases and whether it earns its structural
            # cost." A real tail of evolution_log.jsonl, not the whole
            # (potentially ~20000-line) file.
            body = json.dumps(_history_summary(_tail_jsonl(EVOLUTION_LOG_PATH))).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
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
                    data = json.loads(SELECTED_SOURCE_PATH.read_text(encoding="utf-8"))
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
        if not self.path.startswith("/select") or self.headers.get("X-Cambrian") != "1":
            self.send_response(403)
            self.end_headers()
            return
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host", ""):
            self.send_response(403)
            self.end_headers()
            return
        # "auto" clears the override, resuming normal round-robin.
        # A whitelisted name is fine as-is. Free-text "url" has no
        # whitelist to fall back on, so it's validated for real
        # against yt-dlp (see _check_live_url) before being
        # accepted -- User: "add a field I can input the video to
        # be watched." Still human-only; the organism never
        # reaches this endpoint or picks its own source.
        qs = parse_qs(urlparse(self.path).query)
        name = (qs.get("name") or [""])[0]
        url = (qs.get("url") or [""])[0].strip()
        until_raw = (qs.get("until") or [""])[0].strip()
        until = float(until_raw) if until_raw.replace(".", "", 1).isdigit() else None
        valid_names = {n for n, _ in LIVE_SOURCES}
        ok, error = False, None
        if name == "auto":
            if SELECTED_SOURCE_PATH.exists():
                SELECTED_SOURCE_PATH.unlink()
            ok = True
        elif name in valid_names:
            _write_json_atomic(SELECTED_SOURCE_PATH, {"name": name})
            ok = True
        elif url:
            ok, error = _check_live_url(url)
            if ok:
                # Dessert: a deadline after which the organism goes
                # back to its camera by itself (run_vision.py _dessert).
                _write_json_atomic(SELECTED_SOURCE_PATH, {"url": url, **({"until": until} if until else {})})
        if ok:
            # User: "'Submit' should trigger a restart of
            # cambrian-perception.service" -- without this the
            # selection only took effect on whatever restart
            # happened to come next (up to an hour away), which is
            # exactly what caused the real "still watching the
            # kittens" confusion. Confirmed passwordless sudo for
            # this exact command before wiring it in.
            try:
                subprocess.run(
                    ["systemctl", "restart", "--no-ask-password", "cambrian-perception.service"],
                    capture_output=True, text=True, timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass  # selection is still saved even if the restart trigger itself failed
        body = json.dumps({"ok": ok, "error": error}).encode("utf-8")
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Viewer running at http://{args.host}:{args.port}/ (polls {LIVE_STATUS_PATH})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
