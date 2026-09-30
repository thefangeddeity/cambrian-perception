#!/usr/bin/env python3
"""cambrian --tui: its navigation display in a terminal, for a headless Linux
host -- the picture alone, filling the terminal (a login profile can exec it on
the console: `[ "$(tty)" = /dev/tty1 ] && exec cambrian --tui`, in bash or zsh).
It never touches the camera itself: it reads only the numbers its viewer's
nav display draws from (/state on this machine). The one thing it can ask
for is the page's own "back to the camera" (key c, pressed twice).

The viewer's POV card (its own eye, tools/viewer.py drawSpace + hudNav) drawn
cheaply: Braille dots for the lines (2 x 4 per character cell), curses colours,
the Python standard library only -- a host's plain python3 runs it from a
login profile on the console.

  its ground   equal squares on its ground plane, seen from its eye: lines
               running away meet at the horizon (frame y = hz + (1 - hz) / z)
  the things   hosts (pink), plants (green), other things (grey), where the
               stream shows them -- its eye's projection at zoom 1 is the frame's
  the HUD      horizon, flight-path marker (where it is going, while it moves),
               heading tape (its compass), SPD tape (its speed cells), NR tape
               (T: what its ground model teaches, F: what its terrain head
               feels), the data block, all in its phosphor green

  On the kernel's own console (/dev/tty1 ...), whose fonts have no Braille,
  it draws in half blocks instead (half the detail); --blocks / --braille
  override the guess.

  python3 tools/tui.py [--port 8090]      keys: q quit; c c back to its camera
                                          (the chosen stream cleared; it
                                          restarts); m colour on/off; p the
                                          host's other organism (8090 / 8091)
"""
from __future__ import annotations

import curses
import json
import locale
import sys
import threading
import time
import urllib.request

PORT = 8090
POLL_S = 0.5  # the web viewer's own poll
BRAILLE = 0x2800
DOT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))  # [row][col] of a cell's 2 x 4 dots

# colour pairs
GREEN, HOST, PLANT, THING, CYAN, DIM, WARN, SHEN = range(1, 9)


HALF = {1: "▀", 2: "▄", 3: "█"}  # upper half, lower half, full block


def on_linux_console() -> bool:
    """Whether this is the kernel's own text console (/dev/tty1 ...), not a
    terminal emulator: its fonts have no Braille (checked 2026-09-30: Ubuntu's
    Terminus, Arch's eurlatgr) but do have the half blocks, as camdash uses."""
    import os
    import re
    try:
        return re.fullmatch(r"/dev/tty\d+", os.ttyname(sys.stdout.fileno())) is not None
    except OSError:
        return False


class Canvas:
    """A dot canvas, one colour a cell (the last line drawn through it):
    Braille, 2 x 4 dots a cell, in a terminal emulator; half blocks, 1 x 2
    dots a cell, on the kernel's console, whose fonts have no Braille. A dot
    is about square either way (a cell is about twice as tall as wide)."""

    def __init__(self, cols: int, rows: int, blocks: bool = False):
        self.cols, self.rows, self.blocks = cols, rows, blocks
        self.w, self.h = (cols, rows * 2) if blocks else (cols * 2, rows * 4)
        self.bits = [[0] * cols for _ in range(rows)]
        self.col = [[0] * cols for _ in range(rows)]

    def dot(self, x: float, y: float, c: int) -> None:
        xi, yi = int(x), int(y)
        if 0 <= xi < self.w and 0 <= yi < self.h:
            if self.blocks:
                r, cc, bit = yi >> 1, xi, 1 << (yi & 1)
            else:
                r, cc, bit = yi >> 2, xi >> 1, DOT[yi & 3][xi & 1]
            self.bits[r][cc] |= bit
            self.col[r][cc] = c

    def line(self, x0: float, y0: float, x1: float, y1: float, c: int, dash: int = 0) -> None:
        """A line of dots; dash > 0 draws `dash` dots on, `dash` off (lighter)."""
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if n > 4 * (self.w + self.h):  # far off the canvas: clip by sampling only
            n = 4 * (self.w + self.h)
        for i in range(n + 1):
            if dash and (i // dash) % 2:
                continue
            t = i / n
            self.dot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, c)

    def rect(self, x0, y0, x1, y1, c, dashed=False) -> None:
        if dashed:
            for (a, b, p, q) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
                n = int(max(abs(p - a), abs(q - b))) + 1
                for i in range(n + 1):
                    if (i // 3) % 2 == 0:
                        self.dot(a + (p - a) * i / n, b + (q - b) * i / n, c)
        else:
            self.line(x0, y0, x1, y0, c); self.line(x1, y0, x1, y1, c)
            self.line(x1, y1, x0, y1, c); self.line(x0, y1, x0, y0, c)

    def circle(self, cx, cy, r, c) -> None:
        import math
        n = max(12, int(2 * math.pi * r))
        for i in range(n):
            a = 2 * math.pi * i / n
            self.dot(cx + r * math.cos(a), cy + r * math.sin(a), c)

    def blit(self, scr, top: int, left: int, colour: bool) -> None:
        for r in range(self.rows):
            for cc in range(self.cols):
                b = self.bits[r][cc]
                if b:
                    attr = curses.color_pair(self.col[r][cc]) if colour else 0
                    put(scr, top + r, left + cc, HALF[b] if self.blocks else chr(BRAILLE + b), attr)


def put(scr, y: int, x: int, s: str, attr: int = 0) -> None:
    h, w = scr.getmaxyx()
    if 0 <= y < h and 0 <= x < w:
        try:
            scr.addstr(y, x, s[:max(0, w - x - (1 if y == h - 1 else 0))], attr)
        except curses.error:
            pass


class Feed:
    """Polls the viewer's /state in the background (the drawing never waits)."""

    def __init__(self, port: int):
        self.port, self.d, self.err, self.t = port, None, "", 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/state", timeout=3) as r:
                    self.d, self.err, self.t = json.loads(r.read() or b"{}"), "", time.time()
            except Exception as e:  # its viewer down or restarting: say so, keep the last picture
                self.err = f"no viewer on :{self.port} ({type(e).__name__})"
            time.sleep(POLL_S)


def camera(port: int) -> str:
    """Asks its viewer to go back to its camera (the page's own "auto": the
    chosen stream cleared, the organism restarted onto its camera). The one
    thing this display ever asks for."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/select?name=auto", method="POST", data=b"",
                                 headers={"X-Cambrian": "1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            ok = json.loads(r.read() or b"{}").get("ok")
        return "back to its camera: it restarts (a few seconds)" if ok else "its viewer refused the switch"
    except Exception as e:
        return f"couldn't reach its viewer ({type(e).__name__})"


def last(d: dict, k: str) -> list:
    v = d.get(k) or []
    return (v[-1] or []) if v else []


def draw_nav(scr, d: dict, top: int, left: int, cols: int, rows: int, colour: bool, blocks: bool = False) -> None:
    """Its eye's view: ground squares, the things, then the HUD's lines."""
    cv = Canvas(cols, rows, blocks)
    dpr = cv.h // rows  # dots per character row (4 Braille, 2 half blocks)
    W, H = cv.w, cv.h
    sn = d.get("senses") or {}
    hz = sn.get("horizon")
    fx = lambda x: x * (W - 1)
    fy = lambda y: y * (H - 1)
    if hz is not None:
        # its ground: squares TS on a side, TX either side, TZ ahead (viewer.py's POV)
        TS, TX, TZ = 0.5, 16, 24
        zs = [1 + j * TS for j in range(int((TZ - 1) / TS) + 1)]
        xs = [-TX + i * TS for i in range(int(2 * TX / TS) + 1)]
        sx = lambda x, z: 0.5 + x / (1.6 * z)
        sy = lambda z: hz + (1 - hz) / z
        # A dot can't fade as the web page's lines do: where lines would crowd
        # closer than GAP dots they stop instead (the far ground left open). In
        # half blocks (a dot a whole character wide) the ground is drawn dashed
        # -- lighter than the things standing on it -- and a little sparser.
        GAP = 4 if blocks else 3
        dash = 1 if blocks else 0
        z_far, prev = zs[0], None
        for z in zs:  # cross-lines, crowding toward the horizon
            y = fy(sy(z))
            if prev is not None and prev - y < GAP:
                break
            cv.line(fx(sx(-TX, z)), y, fx(sx(TX, z)), y, DIM, dash)
            prev, z_far = y, z
        z_far = min(z_far, W * TS / (1.6 * GAP))  # lines running away stop where they'd merge
        for x in xs:  # lines running away, toward the horizon
            cv.line(fx(sx(x, zs[0])), fy(sy(zs[0])), fx(sx(x, z_far)), fy(sy(z_far)), DIM, dash)
        # parallax: cyan ticks on its cells where things move against the camera
        par = sn.get("parallax")
        shape = (d.get("maps") or {}).get("shape") or d.get("world_grid_shape") or [9, 16]
        if par and not blocks:  # in half blocks a tick is a whole character: noise, so none
            gr, gc = shape
            for k, v in enumerate(par):
                if v > 0.05:
                    r, c = divmod(k, gc)
                    x, y = fx((c + 0.5) / gc), fy((r + 0.5) / gr)
                    cv.line(x, y, x, y - 6 * v, CYAN)
    # the things, where the stream shows them (cut by the frame: dashed)
    for kind, key in ((THING, "thing_boxes"), (PLANT, "plant_boxes"), (HOST, "prey_boxes")):
        for b in last(d, key):
            try:
                _cls, _conf, x0, y0, x1, y1 = b[:6]
            except (TypeError, ValueError):
                continue
            cv.rect(fx(x0), fy(y0), fx(x1), fy(y1), kind, dashed=(y1 >= 0.999 or y0 <= 0.001))
    # its target lock: the gaze reticle (viewer.py's), tightening and red on contact
    if d.get("fovea_cx") is not None and d.get("fovea_cy") is not None:
        gx, gy = fx(d["fovea_cx"]), fy(d["fovea_cy"])
        lc = last(d, "contact_frames")
        ct = max(0.0, min(1.0, lc if isinstance(lc, (int, float)) else (sn.get("contact") or 0.0)))
        rr = 6 * (1 - 0.6 * ct)
        c = WARN if ct > 0 else CYAN
        cv.circle(gx, gy, rr, c)
        cv.line(gx - rr - 3, gy, gx - rr + 2, gy, c); cv.line(gx + rr - 2, gy, gx + rr + 3, gy, c)
    # the HUD's lines
    if hz is not None:
        hy = fy(hz)
        cv.line(W * 0.14, hy, W * 0.44, hy, GREEN)
        cv.line(W * 0.56, hy, W * 0.86, hy, GREEN)
        spd = sn.get("speed")
        if spd is not None and abs(spd) > 0.01:  # flight-path marker, moved by its turning
            px = W * (0.5 + 0.5 * max(-1.0, min(1.0, sn.get("turning") or 0.0)))
            cv.circle(px, hy, 3, GREEN)
            cv.line(px - 3, hy, px - 8, hy, GREEN); cv.line(px + 3, hy, px + 8, hy, GREEN)
            cv.line(px, hy - 3, px, hy - 6, GREEN)
    # tapes: speed on the left, nearness on the right (near at the top)
    ty0, ty1 = 4 * dpr, H - 2 * dpr  # below the heading tape and the tapes' labels (text rows 0-3)
    for x, side in ((4, 1), (W - 5, -1)):
        cv.line(x, ty0, x, ty1, GREEN)
        for t in range(5):
            y = ty0 + (ty1 - ty0) * t / 4
            cv.line(x, y, x + 2 * side, y, GREEN)
    if sn.get("speed") is not None:
        spd = sn["speed"]
        hi = max(0.5, -(-abs(spd) * 2 // 1) / 2)
        y = ty1 - (ty1 - ty0) * max(0.0, min(1.0, spd / hi))
        cv.line(5, y, 8, y - 1, GREEN); cv.line(5, y, 8, y + 1, GREEN)
    for key, col in (("nearness", PLANT), ("felt_nearness", CYAN)):
        v = sn.get(key)
        if v is not None:
            y = ty1 - (ty1 - ty0) * max(0.0, min(1.0, v))
            cv.line(W - 6, y, W - 9, y - 1, col); cv.line(W - 6, y, W - 9, y + 1, col)
    cv.blit(scr, top, left, colour)
    # its text: heading tape, tape labels, data block
    g = curses.color_pair(GREEN) | curses.A_BOLD if colour else curses.A_BOLD
    hd = sn.get("heading")
    if hd is not None:
        hd %= 360
        half = min(cols // 5, 18)
        tape, step = [], 60.0 / (2 * half)  # 60 degrees across, ticks every 10, long every 30
        for i in range(-half, half + 1):
            a = hd + i * step
            m = round(a / 10) * 10
            tape.append(("|" if m % 30 == 0 else "'") if abs(a - m) < step / 2 else " ")
        put(scr, top, left + cols // 2 - half, "".join(tape), g)
        put(scr, top + 1, left + cols // 2 - 1, f"{int(round(hd)) % 360:03d}", g)
        tr = max(-1.0, min(1.0, sn.get("turning") or 0.0))
        if abs(tr) > 0.01:
            n = int(round(abs(tr) * half))
            put(scr, top + 2, left + cols // 2 - (n if tr < 0 else 0), "=" * max(1, n), g)
    acc = sn.get("acceleration") or 0.0
    if sn.get("speed") is not None:
        put(scr, top + 3, left + 1, f"SPD {sn['speed']:.2f}" + (" ^" if acc > 0.05 else " v" if acc < -0.05 else ""), g)
    nr = lambda v: "--" if v is None else f"{v:.2f}"
    lab = f"NR T{nr(sn.get('nearness'))} F{nr(sn.get('felt_nearness'))}"
    put(scr, top + 3, left + cols - len(lab) - 1, lab, g)
    # what it is taking in now, top left (the web viewer's character): jīng
    # while it absorbs from a host, qì while it sips, shén otherwise (awake,
    # anything new feeds it); nothing asleep. A console's font has no Chinese:
    # there, the word.
    eat, sip, slept = (d.get("eating") or [0])[-1] or 0, (d.get("snacks") or [0])[-1] or 0, (d.get("asleep_frames") or [0])[-1] or 0
    if not slept and d.get("live_actor"):
        han, word, col = ("精", "JĪNG", HOST) if eat > 0.01 else ("氣", "QÌ", PLANT) if sip > 0.01 else ("神", "SHÉN", SHEN)
        put(scr, top + 4, left + 1, word if blocks else f"{han} {word}", (curses.color_pair(col) if colour else 0) | curses.A_BOLD)
    tex = sn.get("texture")
    rows_ = [r for r in (
        "LEARNING HELD" if sn.get("out_of_model") else "",
        f"ODD {sn['strangeness']:.2f}" if sn.get("strangeness") else "",
        f"RIDE {round(100 * sn['riding'])}%" if sn.get("riding") else "",
        f"RAM x{sn['ram']:.1f}" if (sn.get("ram") or 1.0) > 1.01 else "",  # what the flow past its mouth pays its snacks
        f"TEX c{tex[0]:.2f} f{tex[1]:.2f} g{tex[2]:.2f}" if tex and tex[0] else "",
        f"PLACE {sn['place_value']:+.2f}" if sn.get("place_value") is not None and abs(sn["place_value"]) > 0.01 else "",
    ) if r]
    for i, r in enumerate(rows_):
        put(scr, top + 6 + i, left + 1, r, g)
    if hz is None:
        put(scr, top + rows // 2, left + max(0, cols // 2 - 7), "no horizon yet", curses.A_DIM)


def main(scr, port: int, blocks: bool = False) -> None:
    curses.curs_set(0)
    scr.nodelay(True)
    colour = curses.has_colors()
    if colour:
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        many = curses.COLORS >= 256
        for n, (c256, c8) in {GREEN: (120, curses.COLOR_GREEN), HOST: (211, curses.COLOR_MAGENTA),
                              PLANT: (150, curses.COLOR_GREEN), THING: (248, curses.COLOR_WHITE),
                              CYAN: (117, curses.COLOR_CYAN), DIM: (65, curses.COLOR_GREEN),
                              WARN: (203, curses.COLOR_RED), SHEN: (177, curses.COLOR_MAGENTA)}.items():
            curses.init_pair(n, c256 if many else c8, bg)
    feeds = {port: Feed(port)}
    armed, note = 0.0, ("", 0.0)  # the camera key's first press; a line shown for a few seconds
    while True:
        k = scr.getch()
        if k in (ord("q"), ord("Q"), 27):
            return
        if k in (ord("p"), ord("P")):
            port = PORT + 1 if port == PORT else PORT
            feeds.setdefault(port, Feed(port))
        if k in (ord("m"), ord("M")) and curses.has_colors():
            colour = not colour
        if k in (ord("c"), ord("C")):
            # Back to its camera: the viewer's own "auto" (the chosen stream
            # cleared). It restarts the organism, so it takes a second press.
            if time.time() - armed < 3.0:
                armed, note = 0.0, (camera(port), time.time())
            else:
                armed, note = time.time(), ("press c again to switch it to its camera (it restarts)", time.time())
        feed = feeds[port]
        d = feed.d or {}
        scr.erase()
        h, w = scr.getmaxyx()
        # the nav display alone, as big as the terminal allows, in the stream's
        # own shape (a character cell ~ twice as tall as wide)
        fw, fh = d.get("frame_w") or 16, d.get("frame_h") or 9
        cols = min(w, int(h * 2 * fw / fh))
        rows = min(h, int(cols * fh / fw / 2))
        if cols >= 20 and rows >= 8:
            draw_nav(scr, d, (h - rows) // 2, (w - cols) // 2, cols, rows, colour, blocks)
        if note[0] and time.time() - note[1] < 5.0:
            put(scr, h - 1, 0, note[0], curses.color_pair(GREEN) | curses.A_BOLD if colour else curses.A_BOLD)
        elif feed.err:  # its viewer is down or restarting: said once, small, over the last picture
            put(scr, h - 1, 0, feed.err, curses.color_pair(WARN) if colour else curses.A_DIM)
        scr.refresh()
        time.sleep(POLL_S / 2)


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    p = PORT
    if "--port" in sys.argv:
        p = int(sys.argv[sys.argv.index("--port") + 1])
    # Braille where the font has it; half blocks on the kernel's console (or
    # when asked: --blocks / --braille override the guess)
    use_blocks = "--blocks" in sys.argv or ("--braille" not in sys.argv and on_linux_console())
    try:
        curses.wrapper(main, p, use_blocks)
    except KeyboardInterrupt:
        pass
