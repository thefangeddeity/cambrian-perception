#!/usr/bin/env python3
from __future__ import annotations

"""
A tiny, dependency-free local viewer -- "garbage that over millions of
iterations becomes vision" (the user's own framing). Serves ONE page that
polls state/live_status.json and renders the organism's actual 12x12
grid as a blocky, pixelated canvas -- genuinely what it's seeing, not
a reconstruction, since a 144-value luminance grid is already reduced
far past anything resembling real footage (see run_vision.py's own
comment on why including it there doesn't touch the no-raw-frames
rule). Also draws the fovea's own box (cx/cy/fraction) on a blank
frame-proportioned rectangle -- spatial context, no real background,
since no real frame is ever available at this point in the pipeline.

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
  /* User: "video can play to the right of fovea position box. Fovea
     position box should... have the actual image it's cropping out of
     underneath it." Two copies of the SAME live YouTube embed -- one
     plain (general viewing), one with the fovea box drawn on top of
     it. Both are the VIEWER'S OWN BROWSER connecting straight to
     YouTube -- cambrian-perception's own backend/state never touches,
     stores, or transmits a real frame either way, same no-raw-frames
     boundary as everywhere else, just satisfied by a completely
     different, browser-side mechanism this time. */
  /* User: "Allow that long-ass explanation to wrap lol" -- flex items
     don't wrap text by default unless width-constrained; this label
     was stretching the whole panel across the page in one line
     instead of wrapping above its 320px-wide video panel. */
  #fovea-label { max-width: 320px; }
  .video-wrap { position: relative; width: 320px; height: 180px; overflow: hidden; }
  .video-wrap iframe { display: block; position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none; }
  .fovea-overlay { position: absolute; top: 0; left: 0; pointer-events: none; border: none !important; }
  /* User: "if the video is stopped, I'm watching the box move over a
     stopped play button" -- real problem with hoping autoplay just
     works. The real YouTube Player API reports actual play state, so
     this overlay only shows when it's genuinely NOT playing (not a
     guess), and is itself clickable to start playback -- a real,
     reliable affordance instead of a caption someone might not read. */
  .play-overlay {
    position: absolute; inset: 0; z-index: 2;
    display: flex; align-items: center; justify-content: center;
    background: rgba(10, 14, 20, 0.55); color: #7fd4ff; font-family: monospace;
    font-size: 13px; cursor: pointer; text-align: center;
  }
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
  }
</style>
</head>
<body>
  <h1>cambrian-perception</h1>
  <div class="sub">what it's seeing -- a real, live 12x12 luminance grid, not a reconstruction</div>
  <div class="row">
    <div>
      <div class="sub">retina (grid)</div>
      <canvas id="grid" width="240" height="240"></canvas>
    </div>
    <div>
      <div class="sub" id="fovea-label">fovea position -- box shows where it's looking, drawn over the same real video it's cropping. Position reflects the LAST analyzed ~40s window, not this instant. On iOS, tap the video if it doesn't autoplay -- a real platform restriction, not a bug here.</div>
      <div class="video-wrap" id="fovea-video-wrap">
        <!-- Real YouTube Player API target (not a bare iframe) --
             lets JS ask the actual play state instead of hoping
             autoplay worked. See the script tag + player code below. -->
        <div id="fovea-embed"></div>
        <canvas id="fovea" class="fovea-overlay"></canvas>
        <div id="play-overlay" class="play-overlay" style="display:none;">tap to play</div>
      </div>
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
      <div style="margin-top:6px;">
        <input type="text" id="custom-url" placeholder="paste a live YouTube URL..."
          style="width:220px; background:#0a0e14; color:#7fd4ff; border:1px solid #234; font-family:monospace; font-size:11px; padding:3px;">
        <button id="custom-url-submit" style="font-family:monospace; font-size:11px; background:#0a0e14; color:#7fd4ff; border:1px solid #234; cursor:pointer;">Submit</button>
        <span id="submit-status" style="color:#567; font-size:11px;"></span>
      </div>
    </div>
  </div>
  <div class="sub" style="margin-top:24px;">its brain -- the current ACCEPTED genome's own trees (response / pan / tilt), not a rejected candidate's</div>
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

  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    // Real YouTube Player API, not a bare iframe -- lets the page ask
    // the actual play state instead of hoping autoplay worked. User:
    // "if the video is stopped, I'm watching the box move over a
    // stopped play button." ytReady/pendingVideoId handle the real
    // race: the API script loads async, and a video may need loading
    // before onYouTubeIframeAPIReady has fired.
    let ytPlayer = null, ytReady = false, pendingVideoId = null;

    function onYouTubeIframeAPIReady() {
      ytReady = true;
      ytPlayer = new YT.Player('fovea-embed', {
        width: '100%', height: '100%',
        playerVars: { autoplay: 1, mute: 1, playsinline: 1 },
        events: {
          onReady: () => { if (pendingVideoId) { ytPlayer.loadVideoById(pendingVideoId); pendingVideoId = null; } },
          onStateChange: (e) => {
            const overlay = document.getElementById('play-overlay');
            // YT.PlayerState.PLAYING === 1 -- only hide the "tap to
            // play" prompt on a REAL confirmed-playing state, not an
            // assumption.
            overlay.style.display = (e.data === YT.PlayerState.PLAYING) ? 'none' : 'flex';
          },
        },
      });
    }
    function loadFoveaVideo(vid) {
      if (!ytReady || !ytPlayer || !ytPlayer.loadVideoById) { pendingVideoId = vid; return; }
      ytPlayer.loadVideoById(vid);
    }
    document.getElementById('play-overlay').addEventListener('click', () => {
      if (ytPlayer && ytPlayer.playVideo) ytPlayer.playVideo();
    });

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

    // Tracks the currently-loaded embed so tick() (runs every 1s)
    // only touches iframe.src when the video actually changes -- not
    // on every poll, which would restart playback constantly.
    let currentEmbedId = null;

    function youtubeId(url) {
      const m = url && url.match(/[?&]v=([^&]+)/);
      return m ? m[1] : null;
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
          <div><span class="label">watching:</span> ${watchingName}${watchingLive}</div>
        `;

        if (d.grid && d.grid_shape) {
          const [rows, cols] = d.grid_shape;
          const c = document.getElementById('grid');
          const ctx = c.getContext('2d');
          const cw = c.width / cols, ch = c.height / rows;
          for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
              const v = Math.max(0, Math.min(1, d.grid[i * cols + j]));
              const g = Math.round(v * 255);
              ctx.fillStyle = `rgb(${g},${g},${g})`;
              ctx.fillRect(j * cw, i * ch, cw, ch);
            }
          }
        }

        // Real video underneath the fovea box -- User: "fovea position
        // box should... have the actual image it's cropping out of
        // underneath it." Both embeds are the BROWSER's own direct
        // connection to YouTube (never routed through this server's
        // own state/backend -- see the CSS comment on .video-wrap for
        // why that keeps the no-raw-frames boundary intact). Src is
        // only touched when the video actually changes, not every
        // poll tick, so playback doesn't restart every second.
        const vid = youtubeId(d.clip);
        if (vid && vid !== currentEmbedId) {
          currentEmbedId = vid;
          // Real Player API load, not a raw src= assignment -- see
          // loadFoveaVideo/onYouTubeIframeAPIReady above. autoplay/
          // mute/playsinline are set once in the player's own
          // playerVars now, not re-specified per video.
          loadFoveaVideo(vid);
        }

        const fc = document.getElementById('fovea');
        const foveaWrap = document.getElementById('fovea-video-wrap');
        // Honest shape, not a hardcoded 4:3 -- User: "make foveal
        // rectangle honest." Size BOTH the video container and the
        // overlay canvas to the REAL source frame's own aspect ratio
        // (frame_w/frame_h, sent once per run) so the embed and the
        // box drawn on top of it stay pixel-aligned, and the box is
        // actually shaped like what fovea.py really crops. Falls back
        // to a plain 320x180 16:9-ish box only if frame_w/h are missing
        // (an older live_status.json) or no YouTube id was found (a
        // local file/device source, not a live embed).
        const maxDim = 320;
        let vw = maxDim, vh = Math.round(maxDim * 9 / 16);
        if (d.frame_w && d.frame_h) {
          vh = d.frame_w >= d.frame_h ? Math.round(maxDim * d.frame_h / d.frame_w) : maxDim;
          vw = d.frame_w >= d.frame_h ? maxDim : Math.round(maxDim * d.frame_w / d.frame_h);
        }
        foveaWrap.style.width = vw + 'px';
        foveaWrap.style.height = vh + 'px';
        fc.width = vw; fc.height = vh;

        const fctx = fc.getContext('2d');
        fctx.clearRect(0, 0, fc.width, fc.height);
        const embedEl = document.getElementById('fovea-embed');
        if (!vid) {
          // No real video to show (local file/device source) --
          // fall back to the old blank spatial-context box.
          fctx.fillStyle = '#111';
          fctx.fillRect(0, 0, fc.width, fc.height);
          fctx.strokeStyle = '#345';
          fctx.strokeRect(0, 0, fc.width, fc.height);
        } else if (d.fovea_cx !== undefined) {
          // User: "Why can't fovea rectangle show the actual image
          // it's looking at, not plain video." Can't read pixels out
          // of a cross-origin YouTube iframe (real browser security
          // boundary, not a choice here) -- but scaling and shifting
          // the SAME iframe via CSS shows just its own region filling
          // the panel, no pixel access needed, same as any photo-crop
          // tool. frac of the frame becomes 100% of the visible panel.
          const frac = d.fovea_fraction || 0.35;
          embedEl.style.width = (100 / frac) + '%';
          embedEl.style.height = (100 / frac) + '%';
          embedEl.style.left = ((0.5 - d.fovea_cx / frac) * 100) + '%';
          embedEl.style.top = ((0.5 - d.fovea_cy / frac) * 100) + '%';

          // Same two real penalties run_vision.py actually grades
          // fitness on (corner_penalty, edge_penalty), applied to
          // THIS exact position -- User: "touching one edge also give
          // an orange-level penalty... shun edges unless they're
          // worth it." Now a border around the whole panel (which IS
          // the crop now) instead of a box drawn on top of it. Red =
          // a true corner; orange = hard against one edge; yellow =
          // approaching either; green = centered.
          const halfRange = 0.5 - frac / 2;
          const normX = halfRange > 1e-9 ? (d.fovea_cx - 0.5) / halfRange : 0;
          const normY = halfRange > 1e-9 ? (d.fovea_cy - 0.5) / halfRange : 0;
          const cornerness = Math.abs(normX * normY);
          const edgeCloseness = Math.max(Math.abs(normX), Math.abs(normY));
          let boxColor = '#4fa';
          if (cornerness >= 0.5) boxColor = '#f44';
          else if (edgeCloseness >= 0.75) boxColor = '#f90';
          else if (edgeCloseness >= 0.4) boxColor = '#fd4';
          foveaWrap.style.border = '2px solid ' + boxColor;
        }

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

    const MUTATION_TYPES = ['mutate_const', 'mutate_op', 'grow', 'shrink', 'reroll_subtree'];
    const MUTATION_COLORS = { mutate_const: '#4fa', mutate_op: '#0af', grow: '#7fd4ff', shrink: '#f90', reroll_subtree: '#f66' };

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
    loadSources();
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
