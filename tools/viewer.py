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
HISTORY_MAX_LINES = 500
HISTORY_MAX_BYTES = 800_000  # real tail, not a full-file read -- the
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


def _check_live_url(url: str) -> tuple[bool, str | None]:
    """
    Real yt-dlp metadata check, not a URL-shape guess -- same
    discipline as every curated source added this session ("confirmed
    live via yt-dlp metadata before wiring in"). Free-text input has
    no whitelist to fall back on, so this IS the validation.
    """
    yt_dlp = Path(sys.executable).parent / "yt-dlp"
    if not yt_dlp.exists():
        yt_dlp = Path("yt-dlp")
    try:
        result = subprocess.run(
            [str(yt_dlp), "--skip-download", "--print", "is_live", url],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "yt-dlp check failed or timed out"
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

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cambrian-perception -- live viewer</title>
<style>
  body { background: #0a0e14; color: #7fd4ff; font-family: monospace; padding: 16px; }
  h1 { font-size: 15px; font-weight: normal; color: #4fa; margin-bottom: 4px; }
  .sub { color: #567; font-size: 12px; margin-bottom: 20px; }
  .row { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }
  canvas { image-rendering: pixelated; border: 1px solid #234; }
  /* Trees render at their real, honest size (User: "since trees grow
     downward, page can just show its honest growth and scroll as far
     down as it grew" -- no shrinking the canvas to fake-fit). This
     wrapper is what actually adapts to a narrow/mobile screen: it
     scrolls horizontally WITHIN itself if the real canvas is wider
     than the viewport, so only that one panel needs a swipe, not the
     whole page. */
  .tree-panel { max-width: 100%; overflow-x: auto; }
  /* the user's own terms, final: "look" (fovea.py's own small cropped
     window) and "visual field" (the full frame it moves through --
     "always, BY DEFINITION, bigger than the looking part"). Both
     canvases are sized from ONE shared real-pixel scale (see tick()'s
     computeScale) so their ON-SCREEN sizes stay honestly proportional
     to their REAL pixel sizes -- no independently-capped canvas can
     make the smaller one (look, real 112x62) render as big as or
     bigger than the larger one (visual field, real 320x179) again.
     User: "Screw the video. What is a pixel dump of what it's seeing?"
     -- replaced an earlier YouTube-embed crop-preview entirely; no
     embedding restrictions, no autoplay games, no cross-origin pixel
     limits. */
  /* User: "Allow that long-ass explanation to wrap lol" -- flex items
     don't wrap text by default unless width-constrained; this label
     was stretching the whole panel across the page in one line
     instead of wrapping above its panel. */
  #look-label, #visual-field-label { max-width: 320px; }
  .chart-panel { max-width: 100%; overflow-x: auto; }
  .stats div { margin-bottom: 6px; }
  .label { color: #567; }
  .stale { color: #f66; }
  /* User: "Itemize tensions, instincts, 'feelings', rewards, and
     punishments, so we know our organism." A static caption, not
     live data -- these are the FIXED fitness terms (fishbowl/
     reflexes.py, conspec.py, run_vision.py), not something that
     changes per generation, so it doesn't need to poll /state. */
  .drives { font-size: 12px; margin-top: 8px; max-width: 900px; }
  .drives div { margin-bottom: 4px; }
  .plus { color: #4fa; }
  .minus { color: #f66; }
  @media (max-width: 480px) {
    body { padding: 10px; }
    .row { gap: 14px; }
    /* User: "resize tree to fit in mobile view" -- a real, deliberate
       exception to the "render honest size, let it scroll" rule for
       this one case: a tree wider than the ENTIRE phone screen with
       no visible scroll affordance just looks broken, not honest.
       Scales the real, already-computed canvas down to fit the
       viewport width (aspect ratio preserved, internal resolution
       unchanged) -- desktop/tablet keep the real per-panel scroll. */
    .tree-panel canvas { max-width: 100%; height: auto; }
    /* User: "Just optimize for mobile using Apple-compatible viewer
       and leave it there." Same shrink-to-fit logic extended to the
       retina grid and all chart canvases -- previously only trees got
       it. */
    #look { max-width: 100%; height: auto; }
    #visual-field { max-width: 100%; height: auto; }
    .chart-panel canvas { max-width: 100%; height: auto; }
  }
</style>
</head>
<body>
  <h1>cambrian-perception</h1>
  <div class="sub">live perception dashboard -- real, coarse 12x12 luminance receptors, not high-res video</div>
  <div class="row">
    <div>
      <div class="sub" id="look-label"><strong>Gaze / Fovea ("Look")</strong> -- what the organism's brain actually receives: a 12x12 luminance grid of its current focus window (real crop: <span id="look-pixels">--</span>). This is its sole visual input.</div>
      <canvas id="look" width="240" height="240"></canvas>
    </div>
    <div>
      <div class="sub" id="visual-field-label"><strong>Full Visual Field (Environment)</strong> -- 12x12 reference view of the entire scene (<span id="field-pixels">--</span>). The organism does not see this full view; the overlaid box shows where the fovea is currently aimed. Box color indicates penalties: <span style="color:#4fa">green</span> (centered), <span style="color:#fd4">yellow/orange</span> (edge proximity), <span style="color:#f44">red</span> (corner penalty).</div>
      <canvas id="visual-field" width="240" height="240"></canvas>
    </div>
    <div>
      <div class="stats" id="stats"></div>
      <!-- Kept as a SIBLING of #stats, not a child -- #stats gets
           wholesale-overwritten every 1s by tick() (real bug, found
           live, fixed once already: nesting this inside #stats wiped
           it out within a second of every load). Sitting in the same
           shared column puts it right below "watching:" without
           re-triggering that bug -- User: "that paste-in line should
           go below 'watching:' line." No separate "train on:" status
           line -- User: "Page already tells me what it's watching...
           your 'train on' line is redundant." No dropdown either --
           User: "I don't want hardcoded anything for now; I'll pick
           the training videos for now." -->
      <div id="url-picker" style="margin-top:6px; display:none;">
        <input type="text" id="custom-url" placeholder="paste a live YouTube URL..."
          style="width:220px; background:#0a0e14; color:#7fd4ff; border:1px solid #234; font-family:monospace; font-size:11px; padding:3px;">
        <button id="custom-url-submit" style="font-family:monospace; font-size:11px; background:#0a0e14; color:#7fd4ff; border:1px solid #234; cursor:pointer;">Submit</button>
        <span id="submit-status" style="color:#567; font-size:11px;"></span>
      </div>
      <div id="body-panel" class="stats" style="margin-top:14px;"></div>
    </div>
  </div>
  <div class="sub" style="margin-top:24px;">its perception tree -- the current ACCEPTED genome's response tree. Movement (pan / tilt / zoom) is driven by its recurrent brain (fishbowl/controller.py), not a tree.</div>
  <div class="row" id="trees"></div>

  <div class="sub" style="margin-top:24px;">what drives it -- FIXED, never-evolved pressures (fishbowl/reflexes.py, conspec.py, run_vision.py). Every one below is a REWARD or PUNISHMENT graded against real signals; none forces a specific behavior -- the organism evolves its own way to satisfy or avoid them.</div>
  <div class="drives">
    <div><span class="plus">+ luminance_change</span> -- notice global brightness change. The most primitive orienting response that exists.</div>
    <div><span class="plus">+ motion_energy</span> -- notice ANY change in the world, direction-blind.</div>
    <div><span class="plus">+ directional_motion</span> -- notice WHICH WAY something moved (a real Hassenstein-Reichardt correlator, the classic model of insect motion detection).</div>
    <div><span class="plus">+ loom</span> (heaviest weight) -- startle at something expanding/approaching. A real threat/food asymmetry: missing a threat costs more than missing an opportunity.</div>
    <div><span class="plus">+ conspec_drive</span> -- stay engaged with a being-like pattern. Dampened by habituation for familiar, harmless presence ("cats and humans are harmless noise"); a real threat paired with a being resensitizes it instead.</div>
    <div><span class="plus">+ curiosity</span> -- cover more of the reachable field of view over a run, instead of fixating by default.</div>
    <div><span class="plus">+ optokinetic_pursuit</span> -- move the eye in the SAME direction real motion is going. A real reflex found across nearly all motile visual animals.</div>
    <div><span class="plus">+ seek</span> -- get physically closer to a detected being, not just correlate with noticing it. Literal orienting, not a proxy.</div>
    <div><span class="minus">- dead_field_penalty</span> -- a SUSTAINED stretch with nothing happening in the world. Can't be dodged by moving the eye (that loophole's closed); curiosity above is the honest way to earn out of it.</div>
    <div><span class="minus">- movement_cost</span> -- real motor effort, every frame, whether or not it actually moved (a push against a wall still costs something). Doesn't cap or forbid a big jump -- pursuit/seek above can still justify one -- it just means an UNJUSTIFIED one is no longer free.</div>
    <div><span class="minus">- corner_penalty</span> -- sitting in a corner specifically (both axes maxed out at once, not just one edge). Soft, not a hard constraint -- real reward can still outweigh it.</div>
    <div><span class="minus">- edge_penalty</span> -- hard against just one edge. Real but smaller than corner_penalty -- a single edge is less wasteful than a true corner, so it costs less, not nothing.</div>
    <div><span class="minus">- homeostatic drive</span> (weight 3, the biggest single term) -- its BODY (fishbowl/state.py, Gemini's plan): every frame costs energy (basal burn, more when aroused; motor effort; a wider look costs more, and more still when CPU is scarce). It only eats by taking in genuinely NEW visual structure through its look -- a still room starves it, and panning over what it has already seen doesn't feed it. Threat (from something approaching anywhere in the whole field) and fatigue (from violent movement) also count against it.</div>
    <div><span class="minus">- giant-fiber reflex</span> -- not a reward: when something approaches fast (locust-LGMD-style expansion across the whole field), the reflex takes over the brain for that frame and widens the look. Innate, not learned.</div>
    <div><span class="plus">+ alarm</span> -- its brain's alarm output tracking real approaching objects.</div>
  </div>
  <div class="sub" style="margin-top:10px;">real tensions it has to balance, not resolve for it: loom's urgency to startle vs. curiosity's pull to keep exploring (a real predator/prey visual-field tradeoff, "scalding vs. freezing" -- a threat on one side, going numb on the other); habituation's dampening of familiar beings vs. resensitizing to any real threat paired with one; dead_field's "don't go numb" vs. seek's "stay locked on what you found." No fixed right answer to any of these is built in -- only the pressure to find its own.</div>

  <div class="sub" style="margin-top:24px;">growth over time -- real history from evolution_log.jsonl (last ~500 generations), refreshed every 20s, not live-polled every second like the panels above. User: "show exactly when complexity increases and whether it earns its structural cost."</div>
  <div class="row">
    <div class="chart-panel">
      <div class="sub"><span style="color:#4fa">fitness (live)</span> / <span style="color:#567">peak ever</span></div>
      <canvas id="chart-fitness" width="420" height="160"></canvas>
    </div>
    <div class="chart-panel">
      <div class="sub">tree size -- <span style="color:#7fd4ff">total</span> / <span style="color:#f90">response</span> / <span style="color:#0af">pan</span> / <span style="color:#f6f">tilt</span> (nodes)</div>
      <canvas id="chart-nodes" width="420" height="160"></canvas>
    </div>
    <div class="chart-panel">
      <div class="sub">tree depth -- <span style="color:#f90">response</span> / <span style="color:#0af">pan</span> / <span style="color:#f6f">tilt</span></div>
      <canvas id="chart-depth" width="420" height="160"></canvas>
    </div>
    <div class="chart-panel">
      <div class="sub">fitness delta, accepted generations only (how much each accepted structural change actually earned)</div>
      <canvas id="chart-delta" width="420" height="160"></canvas>
    </div>
    <div class="chart-panel">
      <div class="sub">accepted mutation type over time (which real change won each accepted generation)</div>
      <canvas id="chart-mutation" width="420" height="160"></canvas>
    </div>
  </div>

  <script>
    function nodeLabel(n) {
      if (n.kind === 'var') return 'x' + n.index;
      if (n.kind === 'const') return n.value.toFixed(2);
      return n.op;
    }
    function layout(n, depth, order) {
      // order: mutable {next: int} counter, in-order leaf position.
      if (!n.children || n.children.length === 0) {
        const x = order.next++;
        return { node: n, depth, x, children: [] };
      }
      const kids = n.children.map(c => layout(c, depth + 1, order));
      const x = kids.reduce((s, k) => s + k.x, 0) / kids.length;
      return { node: n, depth, x, children: kids };
    }
    function drawTree(ctx, laid, maxDepth, leafCount, w, h) {
      const xStep = w / Math.max(1, leafCount + 1);
      // maxDepth+2, not maxDepth+1 -- there are maxDepth+1 distinct
      // depth values (0..maxDepth), and the +1 slot pads BOTH edges
      // the same way xStep's own leafCount+1 already does for x.
      // Real bug, caught from a live screenshot: the old maxDepth+1
      // put the deepest row's y exactly AT h (the canvas's bottom
      // edge), so half of every leaf circle and its label rendered
      // outside the canvas -- padded on top only, clipped on bottom.
      const yStep = h / Math.max(1, maxDepth + 2);
      function pos(l) { return [(l.x + 1) * xStep, (l.depth + 1) * yStep]; }
      function walk(l) {
        const [px, py] = pos(l);
        for (const c of l.children) {
          const [cx, cy] = pos(c);
          ctx.strokeStyle = '#345';
          ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(cx, cy); ctx.stroke();
          walk(c);
        }
      }
      walk(laid);
      function drawNodes(l) {
        const [x, y] = pos(l);
        ctx.fillStyle = l.node.kind === 'op' ? '#0a2a1a' : '#1a1a2a';
        ctx.strokeStyle = '#4fa';
        ctx.beginPath(); ctx.arc(x, y, 16, 0, 7); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#7fd4ff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(nodeLabel(l.node), x, y);
        for (const c of l.children) drawNodes(c);
      }
      drawNodes(laid);
    }
    function maxDepthOf(l) { return l.children.length ? 1 + Math.max(...l.children.map(maxDepthOf)) : 0; }

    function renderTrees(trees, stats, treeLimits) {
      const container = document.getElementById('trees');
      container.innerHTML = '';
      for (const name of ['response', 'pan', 'tilt']) {
        const tree = trees[name];
        if (!tree) continue;
        const wrap = document.createElement('div');
        wrap.className = 'tree-panel';
        const label = document.createElement('div');
        label.className = 'sub';
        // Real node_count/depth next to the real ceiling -- honest
        // context for why a tree looks small: it's small because
        // growth hasn't been ACCEPTED yet (fitness-gated, same as
        // everywhere else in this project), not because it hit a
        // ceiling that isn't visible here.
        const s = (stats && stats[name]) || null;
        const budget = s && treeLimits
          ? ` -- ${s.nodes} / ${treeLimits.max_nodes} nodes, depth ${s.depth} / ${treeLimits.max_depth}`
          : '';
        label.textContent = name + budget;
        wrap.appendChild(label);
        const canvas = document.createElement('canvas');
        wrap.appendChild(canvas);
        container.appendChild(wrap);

        // Layout FIRST, size the canvas to what was actually laid out
        // SECOND -- User: "Borders should adapt as they grow." A fixed
        // canvas size meant a genuinely grown tree (more nodes than
        // today's small ones) would just get cramped/overlapping
        // instead of the canvas expanding to fit it. Real per-node
        // spacing floor (45px horizontal, 55px vertical -- enough for
        // the 16px-radius circles and their labels not to collide),
        // with a minimum so a tiny tree still gets a readable panel.
        // The existing CSS max-width:100% still scales this back down
        // to fit narrow/mobile screens regardless of the real size.
        const order = { next: 0 };
        const laid = layout(tree, 0, order);
        const leafCount = Math.max(1, order.next);
        const maxDepth = maxDepthOf(laid);
        canvas.width = Math.max(260, (leafCount + 1) * 45);
        canvas.height = Math.max(160, (maxDepth + 2) * 55);

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawTree(ctx, laid, maxDepth, leafCount, canvas.width, canvas.height);
      }
    }

    function drawGrid(canvas, values, shape) {
      const ctx = canvas.getContext('2d');
      if (values && shape) {
        const [rows, cols] = shape;
        for (let i = 0; i < rows; i++) {
          const y0 = Math.round(i * canvas.height / rows);
          const y1 = Math.round((i + 1) * canvas.height / rows);
          for (let j = 0; j < cols; j++) {
            const x0 = Math.round(j * canvas.width / cols);
            const x1 = Math.round((j + 1) * canvas.width / cols);
            const v = Math.max(0, Math.min(1, values[i * cols + j]));
            const g = Math.round(v * 255);
            ctx.fillStyle = `rgb(${g},${g},${g})`;
            ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
          }
        }
      } else {
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      return ctx;
    }


    // Replays the look's REAL per-frame path (cx, cy, aperture) from the
    // latest evaluated run, one analyzed frame per step at ~15/s (camera
    // at 30 fps, every 2nd frame analyzed) -- no interpolation, so a jump
    // shows as a jump and a glide as a glide. User: "Your box is still
    // jumping from place to place."
    let field = null;
    const REPLAY_FPS = 15;
    function cropSize(d, f) {
      return [2 * Math.floor(d.frame_w * f / 2), 2 * Math.floor(d.frame_h * f / 2)];
    }
    function animateField(now) {
      if (field && field.d.frame_w && field.d.frame_h) {
        const { d, scale } = field;
        const wc = document.getElementById('visual-field');
        const wctx = drawGrid(wc, d.world_grid, d.world_grid_shape);
        const traj = (d.trajectory && d.trajectory.length) ? d.trajectory
          : (d.fovea_cx !== undefined ? [[d.fovea_cx, d.fovea_cy, d.fovea_fraction || 0.35]] : []);
        if (traj.length) {
          const i = Math.floor((now - field.t0) / 1000 * REPLAY_FPS) % traj.length;
          wctx.strokeStyle = 'rgba(127, 212, 255, 0.35)';
          wctx.lineWidth = 1;
          wctx.beginPath();
          for (let k = Math.max(0, i - 45); k <= i; k++) {
            const px = traj[k][0] * wc.width, py = traj[k][1] * wc.height;
            if (k === Math.max(0, i - 45)) wctx.moveTo(px, py); else wctx.lineTo(px, py);
          }
          wctx.stroke();
          const [cx, cy, f] = traj[i];
          const [cw, ch] = cropSize(d, f);
          const bw = cw * scale, bh = ch * scale;
          // Same corner/edge penalties run_vision.py grades on: red = a
          // true corner; orange = hard against one edge; yellow =
          // approaching either; green = centered.
          const halfRange = 0.5 - f / 2;
          const nx = halfRange > 1e-9 ? (cx - 0.5) / halfRange : 0;
          const ny = halfRange > 1e-9 ? (cy - 0.5) / halfRange : 0;
          const corner = Math.abs(nx * ny), edge = Math.max(Math.abs(nx), Math.abs(ny));
          wctx.strokeStyle = corner >= 0.5 ? '#f44' : edge >= 0.75 ? '#f90' : edge >= 0.4 ? '#fd4' : '#4fa';
          wctx.lineWidth = 2;
          wctx.strokeRect(cx * wc.width - bw / 2, cy * wc.height - bh / 2, bw, bh);
        }
      }
      requestAnimationFrame(animateField);
    }
    requestAnimationFrame(animateField);

    // Gemini's homeostatic gauges -- the candidate's body at the end of
    // its latest run, its energy over that run, and how often the
    // giant-fiber escape reflex took over from the brain.
    function renderBody(d) {
      const el = document.getElementById('body-panel');
      if (!el || !d.body) return;
      const bar = (name, v, color) =>
        `<div><span class="label" style="display:inline-block;width:62px">${name}</span>` +
        `<span style="display:inline-block;width:120px;height:8px;background:#162029;vertical-align:middle">` +
        `<span style="display:block;width:${Math.round(Math.max(0, Math.min(1, v)) * 120)}px;height:8px;background:${color}"></span></span> ${v.toFixed(2)}</div>`;
      el.innerHTML = '<div class="sub" style="margin:0 0 6px 0">body (end of latest run)</div>' +
        bar('energy', d.body.energy, '#4fa') + bar('arousal', d.body.arousal, '#7fd4ff') +
        bar('threat', d.body.threat, '#f44') + bar('search', d.body.search, '#fd4') +
        bar('fatigue', d.body.fatigue, '#f90') +
        `<div><span class="label">reflex took over:</span> ${d.reflex_frames ?? '--'} frames</div>` +
        '<canvas id="energy-trace" width="200" height="40" style="margin-top:4px"></canvas>';
      const c = document.getElementById('energy-trace');
      const s = d.energy_series || [];
      if (c && s.length > 1) {
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, c.width, c.height);
        ctx.strokeStyle = '#4fa'; ctx.beginPath();
        s.forEach((v, k) => { const x = k / (s.length - 1) * c.width, y = (1 - v) * (c.height - 2) + 1; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.stroke();
      }
    }

    async function tick() {
      let el = document.getElementById('stats');
      try {
        const res = await fetch('/state');
        const d = await res.json();
        if (!d || d.generation === undefined) {
          el.innerHTML = '<div class="stale">no live_status.json yet -- run_vision.py hasn\\'t written one</div>';
          setTimeout(tick, 2000);
          return;
        }
        // YouTube URL box only when a YouTube source is actually in use
        // -- on the camera it would restart the service and change nothing.
        document.getElementById('url-picker').style.display = d.is_live ? 'block' : 'none';
        const watchingName = d.clip_name || d.clip || '';
        const watchingLive = d.is_live
          ? ` <a href="${d.clip}" target="_blank" rel="noopener" style="color:#4fa">(live &#8599;)</a>`
          : '';
        el.innerHTML = `
          <div><span class="label">generation:</span> ${d.generation}</div>
          <div><span class="label">fitness (live):</span> ${d.best_fitness}</div>
          <div><span class="label">peak ever:</span> ${d.peak_fitness_seen !== undefined ? d.peak_fitness_seen : '--'}</div>
          <div><span class="label">response:</span> ${d.response}</div>
          <div><span class="label">habituation:</span> ${d.habituation_exposure}</div>
          <div><span class="label">look size:</span> ${d.fovea_fraction_accepted !== undefined ? d.fovea_fraction_accepted : d.fovea_fraction} of frame (CPU quota ${d.quota_pct !== undefined ? d.quota_pct + '%' : '--'})</div>
          <div><span class="label">watching:</span> ${watchingName}${watchingLive}</div>
        `;

        // User: "the visual field is always, BY DEFINITION, bigger than
        // the looking part." ONE shared real-pixel scale for BOTH
        // canvases below -- fixes a real bug where "look" (real crop
        // 112x62) was independently capped at the same 240px max as
        // "visual field" (real 320x179), making the smaller thing
        // render as big as or bigger than the larger one. frac/cropW/
        // cropH mirror fovea.py's own extract() math exactly
        // (half_w = floor(w*frac/2), crop = 2*half) -- verified live
        // against the actual running fovea.py, not re-derived blind.
        const frac = d.fovea_fraction || 0.35;
        let scale = 1, cropW = 0, cropH = 0;
        if (d.frame_w && d.frame_h) {
          scale = 240 / Math.max(d.frame_w, d.frame_h);
          const halfW = Math.floor(d.frame_w * frac / 2);
          const halfH = Math.floor(d.frame_h * frac / 2);
          cropW = 2 * halfW; cropH = 2 * halfH;
        }

        // "look" -- fovea.py's own real crop (112x62 here), scaled by
        // the SAME factor as "visual field" below, so it renders
        // genuinely smaller, never equal or bigger.
        const lookC = document.getElementById('look');
        if (cropW && cropH) {
          lookC.width = Math.round(cropW * scale);
          lookC.height = Math.round(cropH * scale);
        }
        drawGrid(lookC, d.grid, d.grid_shape);

        // "visual field" -- the full real frame (320x179 here), same
        // scale.
        const wc = document.getElementById('visual-field');
        if (d.frame_w && d.frame_h) {
          wc.width = Math.round(d.frame_w * scale);
          wc.height = Math.round(d.frame_h * scale);
        }

        // The look's box is drawn by animateField() below, replaying its
        // real path -- keep this generation's data for it.
        if (!field || field.d !== d) field = { d, scale, t0: field ? field.t0 : performance.now() };

        renderBody(d);

        // Real pixel counts, computed live -- User: "get the right
        // pixel counts for each." Never a static guess.
        const lookPx = document.getElementById('look-pixels');
        const fieldPx = document.getElementById('field-pixels');
        if (lookPx) lookPx.textContent = cropW && cropH ? `${cropW}x${cropH} real pixels` : '--';
        if (fieldPx) fieldPx.textContent = d.frame_w && d.frame_h ? `${d.frame_w}x${d.frame_h} real pixels` : '--';

        if (d.trees) {
          renderTrees(d.trees, d.tree_stats, d.tree_limits);
        }
      } catch (e) {
        el.innerHTML = '<div class="stale">error polling /state</div>';
      }
      setTimeout(tick, 1000);
    }
    tick();

    // -- Growth-over-time charts (real history, /history) -----------
    function drawLineChart(canvasId, xs, series) {
      const c = document.getElementById(canvasId);
      const ctx = c.getContext('2d');
      const w = c.width, h = c.height;
      ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, w, h);
      if (!xs.length) {
        ctx.fillStyle = '#567'; ctx.font = '11px monospace';
        ctx.fillText('not enough history yet', 8, h / 2);
        return;
      }
      const pad = { l: 40, r: 8, t: 8, b: 16 };
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
      const xMin = xs[0], xMax = xs[xs.length - 1];
      let yMin = Infinity, yMax = -Infinity;
      for (const s of series) {
        for (const v of s.values) {
          if (v === null || v === undefined || !isFinite(v)) continue;
          yMin = Math.min(yMin, v); yMax = Math.max(yMax, v);
        }
      }
      if (!isFinite(yMin) || !isFinite(yMax)) { yMin = 0; yMax = 1; }
      if (yMin === yMax) { yMin -= 1; yMax += 1; }
      const yPad = (yMax - yMin) * 0.08;
      yMin -= yPad; yMax += yPad;
      const px = x => pad.l + (xMax === xMin ? 0 : (x - xMin) / (xMax - xMin)) * plotW;
      const py = y => pad.t + (1 - (y - yMin) / (yMax - yMin)) * plotH;

      ctx.strokeStyle = '#234'; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + plotH); ctx.lineTo(pad.l + plotW, pad.t + plotH);
      ctx.stroke();
      ctx.fillStyle = '#567'; ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(yMax.toFixed(2), pad.l - 4, pad.t + 8);
      ctx.fillText(yMin.toFixed(2), pad.l - 4, pad.t + plotH);
      ctx.textAlign = 'left';
      ctx.fillText('oldest', pad.l, h - 4);
      ctx.textAlign = 'right';
      ctx.fillText('newest', pad.l + plotW, h - 4);

      for (const s of series) {
        ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < xs.length; i++) {
          const v = s.values[i];
          if (v === null || v === undefined || !isFinite(v)) { started = false; continue; }
          const x = px(xs[i]), y = py(v);
          if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
        }
        ctx.stroke();
      }
    }

    const MUTATION_TYPES = ['mutate_const', 'mutate_op', 'grow', 'shrink', 'reroll_subtree', 'mutate_fovea'];
    const MUTATION_COLORS = { mutate_const: '#4fa', mutate_op: '#0af', grow: '#7fd4ff', shrink: '#f90', reroll_subtree: '#f66', mutate_fovea: '#c8f' };

    function drawMutationChart(records) {
      const c = document.getElementById('chart-mutation');
      const ctx = c.getContext('2d');
      const w = c.width, h = c.height;
      ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, w, h);
      const accepted = records.filter(r => r.accepted && MUTATION_TYPES.includes(r.mutation_type));
      const pad = { l: 78, r: 8, t: 8, b: 16 };
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
      const rowH = plotH / MUTATION_TYPES.length;
      ctx.font = '9px monospace'; ctx.fillStyle = '#567'; ctx.textAlign = 'left';
      MUTATION_TYPES.forEach((t, i) => ctx.fillText(t, 2, pad.t + i * rowH + rowH / 2 + 3));
      if (!accepted.length) {
        ctx.fillStyle = '#567'; ctx.fillText('no accepted mutations in this window yet', pad.l, h / 2);
        return;
      }
      // Real bug fix, User: "What's happening to the charts?" --
      // box.generation resets to 1 every hourly restart, but this
      // window can span multiple restarts, so raw generation numbers
      // aren't monotonic across it. Positioned by RECORD INDEX
      // instead (the tail is already in real chronological/file
      // order), which is always correct regardless of restarts.
      const n = records.length;
      const px = i => pad.l + (n <= 1 ? 0 : i / (n - 1)) * plotW;
      records.forEach((r, i) => {
        if (!r.accepted || !MUTATION_TYPES.includes(r.mutation_type)) return;
        const row = MUTATION_TYPES.indexOf(r.mutation_type);
        const x = px(i), y = pad.t + row * rowH + rowH / 2;
        ctx.fillStyle = MUTATION_COLORS[r.mutation_type];
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
      });
    }

    function treeNodes(r, ch) { return r.tree_stats && r.tree_stats[ch] ? r.tree_stats[ch].nodes : null; }
    function treeDepth(r, ch) { return r.tree_stats && r.tree_stats[ch] ? r.tree_stats[ch].depth : null; }

    async function fetchHistory() {
      try {
        const res = await fetch('/history');
        const records = await res.json();
        if (records && records.length) {
          const xs = records.map((r, i) => i);
          drawLineChart('chart-fitness', xs, [
            { values: records.map(r => r.best_fitness), color: '#4fa' },
            { values: records.map(r => r.peak_fitness_seen), color: '#567' },
          ]);
          drawLineChart('chart-nodes', xs, [
            { values: records.map(r => {
                const total = ['response', 'pan', 'tilt'].reduce((s, ch) => s + (treeNodes(r, ch) || 0), 0);
                return r.tree_stats ? total : null;
              }), color: '#7fd4ff' },
            { values: records.map(r => treeNodes(r, 'response')), color: '#f90' },
            { values: records.map(r => treeNodes(r, 'pan')), color: '#0af' },
            { values: records.map(r => treeNodes(r, 'tilt')), color: '#f6f' },
          ]);
          drawLineChart('chart-depth', xs, [
            { values: records.map(r => treeDepth(r, 'response')), color: '#f90' },
            { values: records.map(r => treeDepth(r, 'pan')), color: '#0af' },
            { values: records.map(r => treeDepth(r, 'tilt')), color: '#f6f' },
          ]);
          drawLineChart('chart-delta', xs, [
            { values: records.map(r => r.accepted ? r.fitness_delta : null), color: '#4fa' },
          ]);
          drawMutationChart(records);
        }
      } catch (e) { /* history is a nice-to-have; a failed fetch shouldn't break the live panels above */ }
      setTimeout(fetchHistory, 20000);
    }
    fetchHistory();

    // User: "'Submit' should trigger a restart of
    // cambrian-perception.service" -- /select now does that itself
    // (real sudo systemctl restart, confirmed passwordless) once the
    // URL's confirmed live, so this really does take effect right
    // away, not on some future restart. "watching:" in the stats
    // panel above (already polled every 1s) is the honest way to
    // confirm it actually applied -- User: "How do I know if it's
    // watching?" -- once the restart completes and the new process
    // writes its own live_status.json, that line updates on its own.
    document.getElementById('custom-url-submit').addEventListener('click', async () => {
      const input = document.getElementById('custom-url');
      const status = document.getElementById('submit-status');
      const url = input.value.trim();
      if (!url) return;
      status.textContent = 'checking it\\'s really live...';
      try {
        const res = await fetch('/select?url=' + encodeURIComponent(url));
        const d = await res.json();
        status.textContent = d.ok ? 'restarting -- watch "watching:" above' : ('failed: ' + (d.error || 'unknown'));
        if (d.ok) { input.value = ''; }
      } catch (err) { status.textContent = 'failed'; }
    });
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
            body = json.dumps(_tail_jsonl(EVOLUTION_LOG_PATH)).encode("utf-8")
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
            selected_name, selected_url = None, None
            if SELECTED_SOURCE_PATH.exists():
                try:
                    data = json.loads(SELECTED_SOURCE_PATH.read_text(encoding="utf-8"))
                    selected_name, selected_url = data.get("name"), data.get("url")
                except (OSError, json.JSONDecodeError):
                    pass
            body = json.dumps({
                "options": [n for n, _ in LIVE_SOURCES],
                "selected": selected_name,
                "selected_url": selected_url,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/select"):
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
                    _write_json_atomic(SELECTED_SOURCE_PATH, {"url": url})
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
                        ["sudo", "systemctl", "restart", "cambrian-perception.service"],
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
        else:
            self.send_response(404)
            self.end_headers()


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
