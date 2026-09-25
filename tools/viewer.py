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
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
LIVE_STATUS_PATH = STATE_DIR / "live_status.json"
# Path defined independently here, not imported from fishbowl.sandbox
# -- same deliberate independence as everything else in this module
# (see its own module docstring: this has to work standalone).
EVOLUTION_LOG_PATH = STATE_DIR / "evolution_log.jsonl"
HISTORY_MAX_LINES = 500
HISTORY_MAX_BYTES = 800_000  # real tail, not a full-file read -- the
# log can grow to ~20000 lines/rotation; this stays cheap regardless.


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
  .video-wrap { position: relative; width: 320px; height: 180px; }
  .video-wrap iframe { display: block; width: 100%; height: 100%; border: 1px solid #234; }
  .fovea-overlay { position: absolute; top: 0; left: 0; pointer-events: none; border: none !important; }
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
      <div class="sub" id="fovea-label">fovea position -- box shows where it's looking, drawn over the same real video it's cropping. Position reflects the LAST analyzed ~40s window, not this instant.</div>
      <div class="video-wrap" id="fovea-video-wrap">
        <iframe id="fovea-embed" src="" frameborder="0" allow="autoplay" title="fovea position source feed"></iframe>
        <canvas id="fovea" class="fovea-overlay"></canvas>
      </div>
    </div>
    <div class="stats" id="stats"></div>
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
          document.getElementById('fovea-embed').src = `https://www.youtube.com/embed/${vid}?autoplay=1&mute=1`;
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
        if (!vid) {
          // No real video to show underneath (local file/device
          // source) -- fall back to the old blank spatial-context box.
          fctx.fillStyle = '#111';
          fctx.fillRect(0, 0, fc.width, fc.height);
          fctx.strokeStyle = '#345';
          fctx.strokeRect(0, 0, fc.width, fc.height);
        }
        if (d.fovea_cx !== undefined) {
          const frac = d.fovea_fraction || 0.35;
          const bw = fc.width * frac, bh = fc.height * frac;
          const bx = d.fovea_cx * fc.width - bw / 2;
          const by = d.fovea_cy * fc.height - bh / 2;
          fctx.strokeStyle = '#4fa';
          fctx.lineWidth = 2;
          fctx.strokeRect(bx, by, bw, bh);
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
      ctx.fillText('gen ' + xMin, pad.l, h - 4);
      ctx.textAlign = 'right';
      ctx.fillText('gen ' + xMax, pad.l + plotW, h - 4);

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
      const gens = records.map(r => r.generation);
      const xMin = gens[0], xMax = gens[gens.length - 1];
      const px = x => pad.l + (xMax === xMin ? 0 : (x - xMin) / (xMax - xMin)) * plotW;
      for (const r of accepted) {
        const row = MUTATION_TYPES.indexOf(r.mutation_type);
        const x = px(r.generation), y = pad.t + row * rowH + rowH / 2;
        ctx.fillStyle = MUTATION_COLORS[r.mutation_type];
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
      }
    }

    function treeNodes(r, ch) { return r.tree_stats && r.tree_stats[ch] ? r.tree_stats[ch].nodes : null; }
    function treeDepth(r, ch) { return r.tree_stats && r.tree_stats[ch] ? r.tree_stats[ch].depth : null; }

    async function fetchHistory() {
      try {
        const res = await fetch('/history');
        const records = await res.json();
        if (records && records.length) {
          const xs = records.map(r => r.generation);
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
