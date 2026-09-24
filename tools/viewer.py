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
  canvas { image-rendering: pixelated; border: 1px solid #234; max-width: 100%; height: auto; }
  .stats div { margin-bottom: 6px; }
  .label { color: #567; }
  .stale { color: #f66; }
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
      <div class="sub">fovea position (field of view within the frame)</div>
      <canvas id="fovea" width="240" height="180"></canvas>
    </div>
    <div class="stats" id="stats"></div>
  </div>
  <div class="sub" style="margin-top:24px;">its brain -- the current ACCEPTED genome's own trees (response / pan / tilt), not a rejected candidate's</div>
  <div class="row" id="trees"></div>
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

    function renderTrees(trees) {
      const container = document.getElementById('trees');
      container.innerHTML = '';
      for (const name of ['response', 'pan', 'tilt']) {
        const tree = trees[name];
        if (!tree) continue;
        const wrap = document.createElement('div');
        const label = document.createElement('div');
        label.className = 'sub';
        label.textContent = name;
        wrap.appendChild(label);
        const canvas = document.createElement('canvas');
        canvas.width = 260; canvas.height = 160;
        wrap.appendChild(canvas);
        container.appendChild(wrap);

        const order = { next: 0 };
        const laid = layout(tree, 0, order);
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawTree(ctx, laid, maxDepthOf(laid), Math.max(1, order.next), canvas.width, canvas.height);
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
        el.innerHTML = `
          <div><span class="label">generation:</span> ${d.generation}</div>
          <div><span class="label">best_fitness:</span> ${d.best_fitness}</div>
          <div><span class="label">response:</span> ${d.response}</div>
          <div><span class="label">habituation:</span> ${d.habituation_exposure}</div>
          <div><span class="label">clip:</span> ${d.clip || ''}</div>
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

        const fc = document.getElementById('fovea');
        const fctx = fc.getContext('2d');
        fctx.fillStyle = '#111';
        fctx.fillRect(0, 0, fc.width, fc.height);
        fctx.strokeStyle = '#345';
        fctx.strokeRect(0, 0, fc.width, fc.height);
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
          renderTrees(d.trees);
        }
      } catch (e) {
        el.innerHTML = '<div class="stale">error polling /state</div>';
      }
      setTimeout(tick, 1000);
    }
    tick();
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
