"""
Its engineered organs (a 2026-09-30 panel -- Gibson, Hartley & Zisserman,
O'Keefe & Moser, Dennett, Gelman): senses it is GIVEN, as an animal is born
with a vestibular sense and a horizon-seeking reflex, instead of left to
learn them. Its learned horizon (organism.horizon: inferred from where
detected things stand) read a room as "9% down" after a tram ride and a
ceiling as "54% down"; evolution is far too slow for a sense like that.
Classic geometric CV only -- no trained model, cheap enough for Tina:

  the horizon     vanishing points of the frame's line segments (LSD): the
                  horizon is the line through the vanishing points of the
                  world's horizontal lines (walls, floors, kerbs, windows);
                  with one, it passes through it, level with the camera's
                  roll (from the vertical lines' vanishing point, else level).
                  Where they put it out of view, or say nothing, a visible
                  level line (the sea's edge, a plain's) is taken instead
  odometry        its turning from the frame's tracked corners: the
                  essential matrix (5-point, RANSAC) separates rotation from
                  travel, which the frame's plain shift mixes
  a place print   a coarse descriptor of the frame's structure (oriented
                  gradients on a 4 x 4 grid): "the same place" or "a new
                  one", judged against its own recent spread

Each is computed once per frame beside the frame's shift (run_vision.World
for evolution's snapshots, field.FieldSignals for the live body) and reaches
the organism in its per-frame signals; it can be switched off per ecohost
(cambrian.json "organs": {"horizon": false, ...}) to compare.

How well the horizon does (2026-09-30: twelve frames from each of the usual
streams, the median estimate against the horizon judged by eye): a tram's
cab 0.01 of the frame's height off, a canyon road 0.03, a beach 0.02 (by
its sea line), Tina's street and 7elwe's ceiling close; a traffic camera
over a crowded road 0.2 off; a tilted-up London street scattered over
0.36-0.74 (its true horizon itself unclear); a room seen steeply down from
high (a kitten room) unreliable; the same tram at night, through its
windscreen's reflections, scattered over -1..0.7; a bare synthetic room
mostly unanswered (None). The organism takes the median of its last nine
estimates, and only while they agree (organism.HORIZON_MAX_SPREAD) -- else
its learned horizon stands. A horizon-first rewrite (each edge
voting along candidate horizons, after Simon, Fond & Berger 2018) was tried
the same day and did worse on every stream but the beach; this simpler
design was kept.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# (H, 2026-09-30) a typical webcam's horizontal field of view -- used only
# until two orthogonal vanishing points have measured the camera's own focal
# length (then derived, and kept per source).
PRIOR_HFOV_DEG = 65.0
ANGLE_TOL = math.radians(2.0)  # (H) a segment belongs to a vanishing point within 2 degrees (LSD's own angle tolerance is 22.5; 2 is the RANSAC inlier band)
RANSAC_TRIES = 150             # (H) hypotheses per vanishing point (two segments each)
HORIZON_EVERY = 3              # (H) frames between horizon estimates: the horizon moves slowly; odometry runs every frame
MAX_ROLL = math.radians(20.0)  # (H) a mounted or carried camera is seldom tilted more: a steeper "horizon" is lines agreeing by chance, and is not taken
COLLINEAR_PX = 2.0             # (H) a segment whose middle lies within 2 px of another's line is a piece of the same edge
LEVEL_COVER = 0.5              # (H) a visible horizon spans at least half the frame's width
LEVEL_BAND = 0.02              # (H) pieces of one level line agree within 2% of the height
LEVEL_EDGE = 0.1               # (H) a level line within 10% of the frame's top or bottom is its surround (a cab's roof, a dashboard), not a horizon
MAX_LEVEL_LINES = 2            # (H) a horizon, perhaps with its shoreline: more long level lines than that is a square-on structure (a facade's floors)
BAR_LEVEL = 24                 # (H) letterbox bars: rows darker than this (of 255) -- just above video black (16, BT.601 studio range), allowing compression noise
STILL_PX = 0.2                 # (H) a frame shifted less than 0.2 px (of its 320) did not turn: odometry is skipped
NON_SCALAR = ("road_lines",)  # outputs for drawing, not signals: a snapshot's series leaves them out
PRINT_GRID, PRINT_BINS = 4, 8  # the place print: 4 x 4 cells x 8 orientations = 128 numbers


def settings() -> dict:
    """Which organs this ecohost runs (cambrian.json "organs"; all on by default)."""
    try:
        from . import sandbox
        o = (sandbox._read_json(sandbox.SETTINGS_PATH, {}) or {}).get("organs") or {}
        out = {k: bool(o.get(k, True)) for k in ("horizon", "odometry", "place", "road")}
        # a camera that banks (an aircraft's) may roll past the horizon organ's
        # 20 degrees: its ecohost raises the bound here
        out["max_roll_deg"] = float(o.get("max_roll_deg", math.degrees(MAX_ROLL)))
        return out
    except Exception:
        return {"horizon": True, "odometry": True, "place": True, "road": True, "max_roll_deg": math.degrees(MAX_ROLL)}


def _segments(grey: np.ndarray, min_len: float) -> np.ndarray:
    """Line segments (x0, y0, x1, y1) at least min_len long, found after its
    contrast is equalised (a dim, soft webcam -- Tanzania's room -- gave LSD
    no segment a field-cell long, and 27 half-cell ones once equalised)."""
    eq = cv2.equalizeHist(grey)
    try:
        lines = cv2.createLineSegmentDetector().detect(eq)[0]
    except (cv2.error, AttributeError):
        lines = cv2.HoughLinesP(cv2.Canny(eq, 60, 150), 1, np.pi / 180, 30, minLineLength=int(min_len), maxLineGap=3)
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4))
    s = lines.reshape(-1, 4).astype(float)
    return s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= min_len]


def _vanishing_point(seg: np.ndarray, rng: np.random.Generator):
    """The point the most segment length converges on -- RANSAC over pairs of
    segments' lines, every hypothesis scored at once -- and which segments
    agree: (point in homogeneous coordinates, inlier mask), or (None, None)."""
    n = len(seg)
    if n < 3:
        return None, None
    p0 = np.c_[seg[:, :2], np.ones(n)]
    p1 = np.c_[seg[:, 2:], np.ones(n)]
    lines = np.cross(p0, p1)
    mid = (seg[:, :2] + seg[:, 2:]) / 2.0
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    i = rng.integers(0, n, RANSAC_TRIES)
    j = (i + 1 + rng.integers(0, n - 1, RANSAC_TRIES)) % n  # a different segment
    V = np.cross(lines[i], lines[j])                        # (tries, 3)
    # two pieces of ONE edge (LSD splits a long one) meet anywhere along it:
    # such a pair is no hypothesis
    unit = lines / np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-9)[:, None]
    one_edge = np.abs(np.einsum("ij,ij->i", unit[i], np.c_[mid[j], np.ones(len(j))])) < COLLINEAR_PX
    V[one_edge] = 0.0
    fin = np.abs(V[:, 2]) > 1e-9
    safe = np.where(fin, V[:, 2], 1.0)
    # direction from each segment's middle toward each hypothesis (a point at
    # infinity: its direction itself)
    tx = np.where(fin[:, None], V[:, 0:1] / safe[:, None] - mid[None, :, 0], V[:, 0:1])
    ty = np.where(fin[:, None], V[:, 1:2] / safe[:, None] - mid[None, :, 1], V[:, 1:2])
    dang = np.abs(((np.arctan2(ty, tx) - ang[None, :]) + np.pi / 2) % np.pi - np.pi / 2)
    masks = dang < ANGLE_TOL
    scores = masks.astype(float) @ length
    scores[one_edge] = 0.0
    k = int(np.argmax(scores))
    if scores[k] <= 0:
        return None, None
    return V[k], masks[k]


def _letterbox(grey: np.ndarray) -> tuple[int, int]:
    """The rows of picture between a stream's black bars (top, bottom): a
    letterboxed frame's bar edges are long, level and perfectly straight --
    the very thing a horizon looks like -- and must not be read as one."""
    rows = grey.mean(axis=1)
    dark = rows < BAR_LEVEL
    top = int(np.argmin(dark)) if dark[0] else 0
    bottom = len(rows) - (int(np.argmin(dark[::-1])) if dark[-1] else 0)
    return (top, bottom) if bottom - top >= len(rows) // 3 else (0, len(rows))


def _cover(seg: np.ndarray, w: float) -> float:
    """The share of the frame's width the segments' x-extents cover together."""
    cover, end = 0.0, -1e9
    for x0, x1 in sorted(zip(np.minimum(seg[:, 0], seg[:, 2]), np.maximum(seg[:, 0], seg[:, 2]))):
        cover += max(0.0, x1 - max(x0, end))
        end = max(end, x1)
    return cover / w


def _level_line(seg: np.ndarray, w: float, h: float, max_roll: float = MAX_ROLL) -> tuple[float, float] | None:
    """A visible horizon: (its height at the frame's middle, its roll), or
    None. The one long level line in view -- level within MAX_ROLL, its
    collinear pieces covering at least LEVEL_COVER of the width (the sea's
    edge, a plain's, broken by a palm or a boat). A frame with more than
    MAX_LEVEL_LINES is a square-on structure (a facade's floors), whose level
    lines lie at every height; one hugging the frame's top or bottom is its
    surround (a cab's roof, a dashboard)."""
    if len(seg) == 0:
        return None
    d = seg[:, 2:] - seg[:, :2]
    ang = (np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2) % np.pi - np.pi / 2
    level = np.abs(ang) < max_roll
    s, a = seg[level], ang[level]
    mid = (s[:, :2] + s[:, 2:]) / 2.0
    yc = mid[:, 1] + np.tan(a) * (w / 2.0 - mid[:, 0])   # each piece's line at the frame's middle
    found = []  # (height, roll, its pieces' length)
    for k in np.argsort(-np.abs(s[:, 2] - s[:, 0]))[:24]:
        if any(abs(yc[k] - f[0]) < 2 * LEVEL_BAND * h for f in found):
            continue
        same = (np.abs(a - a[k]) < ANGLE_TOL) & (np.abs(yc - yc[k]) < LEVEL_BAND * h)
        if _cover(s[same], w) >= LEVEL_COVER:
            found.append((float(np.median(yc[same])), float(np.median(a[same])), float(np.hypot(*(s[same, 2:] - s[same, :2]).T).sum())))
    if len(found) > MAX_LEVEL_LINES:
        return None
    found = [f for f in found if LEVEL_EDGE * h < f[0] < (1 - LEVEL_EDGE) * h]
    best = max(found, key=lambda f: f[2], default=None)
    return None if best is None else (best[0], best[1])


def horizon(grey: np.ndarray, focal: float | None, rng: np.random.Generator, max_roll: float = MAX_ROLL) -> dict | None:
    """The horizon from the frame's lines: {"y": its height at the frame's
    middle (0 top, 1 bottom -- outside 0..1 when out of view), "roll": its
    tilt (radians), "conf": the share of the frame's line length that agrees,
    "focal": the focal length measured from two orthogonal vanishing points
    (pixels), or None, "how": "vp" or "line"}; None when the lines don't say."""
    H0 = grey.shape[0]
    top, bottom = _letterbox(grey)
    grey = grey[top:bottom]
    h, w = grey.shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    seg = _segments(grey, min_len=h / 18.0)  # at least half a cell of its field (9 rows) long
    if len(seg) < 6:
        return None
    est = _from_vanishing_points(seg, c, h, w, focal, rng, max_roll)
    if est is None or not 0.0 <= est["y"] <= h:
        # out of view, or nothing said: on the usual streams (2026-09-30) an
        # out-of-view answer was wrong wherever a level line was in view (a
        # beach's palm fronds converge on its crown; a crowded road's car
        # edges agree by chance), and the line was right or nearer
        line = _level_line(seg, w, h, max_roll)
        if line is not None:
            est = {"y": line[0], "roll": line[1], "conf": 0.0, "focal": None, "how": "line"}
    if est is None:
        return None
    est["y"] = float(np.clip((est["y"] + top) / H0, -1.0, 2.0))
    return est


def _from_vanishing_points(seg: np.ndarray, c: np.ndarray, h: float, w: float, focal: float | None,
                           rng: np.random.Generator, max_roll: float = MAX_ROLL) -> dict | None:
    """The horizon through the horizontal lines' vanishing points ("y" in
    pixels of the cropped frame), or None."""
    d = seg[:, 2:] - seg[:, :2]
    steep = np.abs(d[:, 1]) > np.abs(d[:, 0])  # nearer vertical than horizontal (45 degrees: the midpoint)
    total = float(np.hypot(d[:, 0], d[:, 1]).sum())
    vv, _ = _vanishing_point(seg[steep], rng) if steep.sum() >= 3 else (None, None)
    flat = seg[~steep]
    vps, support = [], 0.0
    for _ in range(2):
        v, mask = _vanishing_point(flat, rng)
        if v is None or mask is None or mask.sum() < 3:
            break
        dm = flat[mask, 2:] - flat[mask, :2]
        support += float(np.hypot(dm[:, 0], dm[:, 1]).sum())
        vps.append(v)
        flat = flat[~mask]
    finite = [v[:2] / v[2] for v in vps if abs(v[2]) > 1e-9]
    measured = None
    if len(finite) == 2:
        dot = -float(np.dot(finite[0] - c, finite[1] - c))
        if dot > 0:
            measured = math.sqrt(dot)  # two orthogonal directions' vanishing points: f^2 = -(v1 - c).(v2 - c)
    f = measured or focal or (w / 2.0) / math.tan(math.radians(PRIOR_HFOV_DEG) / 2.0)
    if len(finite) == 2 and abs(finite[1][0] - finite[0][0]) > 1e-6:
        (x0, y0), (x1, y1) = finite
        slope = (y1 - y0) / (x1 - x0)
        roll = math.atan(slope)
        y = y0 + slope * (c[0] - x0)
    elif len(vps) >= 1:
        # one horizontal direction: the horizon passes through its point (or
        # parallel to it, at infinity), level with the camera's roll
        if vv is not None and abs(vv[2]) > 1e-9:
            up = vv[:2] / vv[2] - c
            roll = math.atan2(-up[0], up[1]) if up[1] > 0 else math.atan2(up[0], -up[1])
        else:
            roll = 0.0
        v = vps[0]
        if abs(v[2]) > 1e-9:
            px, py = v[:2] / v[2]
            y = py + math.tan(roll) * (c[0] - px)
        elif vv is not None and abs(vv[2]) > 1e-9:
            vz = vv[:2] / vv[2]
            y = c[1] - f * f / (vz[1] - c[1]) if abs(vz[1] - c[1]) > 1e-6 else None
        else:
            return None
    elif vv is not None and abs(vv[2]) > 1e-9:
        # only the verticals: the horizon is where the camera's tilt puts it
        vz = vv[:2] / vv[2]
        if abs(vz[1] - c[1]) < 1e-6:
            return None
        y = c[1] - f * f / (vz[1] - c[1])
        roll = math.atan2(-(vz[0] - c[0]), vz[1] - c[1]) if vz[1] > c[1] else math.atan2(vz[0] - c[0], c[1] - vz[1])
    else:
        return None
    if y is None or not math.isfinite(y) or abs(roll) > max_roll:
        return None
    return {"y": float(y), "roll": float(roll), "conf": min(1.0, support / max(total, 1e-9)), "focal": measured, "how": "vp"}



# ---- the road organ --------------------------------------------------------
# Where the road goes (a 2026-10-01 panel -- Dickmanns' 4D approach, Hartley &
# Zisserman, Gibson, Gelman): the frame's line segments below its horizon,
# levelled by its roll and laid back onto the ground through its pitch and
# focal length (inverse perspective; camera height = 1 eye-height), then one
# road model fitted to all of them -- each edge or marking its own lateral
# offset, all sharing the road's shape: X(Z) = s(Z) x_k + psi Z + c0 Z^2/2 +
# c1 Z^3/6 (a clothoid: heading psi, curvature c0, its rate c1) -- and its
# rise and fall: on a crest the ground drops away, so a constant lane width
# looks narrower with distance than flat ground would make it, s(Z) =
# 1 / (1 + cv Z^2 / 2) (cv > 0 a crest, < 0 a dip; only seen with two lines or
# more). A scalar Kalman filter carries each estimate from frame to frame. It
# needs a horizon its organism believes, and says nothing without one.
ROAD_MIN_SEGS = 3                     # (H) fewer road segments than this: no road
ROAD_GAP = 0.4                        # (H) eye-heights between two edges' offsets that makes them two (a marking's own width is ~0.1)
ROAD_ZMAX = 40.0                      # (H) eye-heights: farther than this, a pixel's error is metres of road
ROAD_CV = np.linspace(-0.02, 0.02, 11)  # (H) the crest/dip curvatures tried (per eye-height): +-0.02 drops 16 eye-heights at 40 out
# its Kalman filter: how far each may wander per frame (Q) and how noisy one fit is when fully sure (R)
ROAD_Q = {"heading": 0.01 ** 2, "curve": 0.002 ** 2, "crest": 0.002 ** 2, "offset": 0.05 ** 2}   # (H)
ROAD_R = {"heading": 0.03 ** 2, "curve": 0.004 ** 2, "crest": 0.006 ** 2, "offset": 0.15 ** 2}   # (H)


def _ground_maps(roll: float, y_h: float, w: float, h: float, f: float):
    """The map from a frame point to the ground (X sideways, Z ahead, in
    eye-heights) and back, for this roll, horizon row and focal length."""
    cx, cy = w / 2.0, h / 2.0
    cr, sr = math.cos(roll), math.sin(roll)
    th = math.atan2(cy - y_h, f)  # its pitch: looking down puts the horizon above the middle
    ct, st = math.cos(th), math.sin(th)

    def to_ground(x, y):
        u, v = x - cx, y - cy
        lu, lv = u * cr + v * sr, -u * sr + v * cr          # levelled (rotated by -roll)
        yw, zw = lv * ct + f * st, -lv * st + f * ct
        ok = yw > 1e-6
        t = np.where(ok, 1.0 / np.where(ok, yw, 1.0), np.nan)
        return t * lu, t * zw

    def to_frame(X, Z):
        dy, dz = ct - Z * st, st + Z * ct                   # the ground point (X, 1, Z) in its camera
        lu, lv = f * X / dz, f * dy / dz
        return cx + lu * cr - lv * sr, cy + lu * sr + lv * cr  # back unlevelled (rotated by +roll)
    return to_ground, to_frame


def road(grey: np.ndarray, hz: dict, focal: float | None, rng: np.random.Generator) -> dict | None:
    """Its road this frame: {"heading" (radians, + right), "curve" (per
    eye-height, + bends right), "crest" (per eye-height, + falls away),
    "offset" (eye-heights right of its lane's middle), "conf" (0..1), "lines"
    (each edge as frame points, for drawing)}; None when it sees no road."""
    H0, W0 = grey.shape[:2]
    f = focal or (W0 / 2.0) / math.tan(math.radians(PRIOR_HFOV_DEG) / 2.0)
    y_h = hz["y"] * H0
    to_ground, to_frame = _ground_maps(hz.get("roll", 0.0), y_h, W0, H0, f)
    seg = _segments(grey, min_len=H0 / 24.0)
    if len(seg) < ROAD_MIN_SEGS:
        return None
    X0, Z0 = to_ground(seg[:, 0], seg[:, 1])
    X1, Z1 = to_ground(seg[:, 2], seg[:, 3])
    dX, dZ = X1 - X0, Z1 - Z0
    keep = (np.isfinite(Z0) & np.isfinite(Z1) & (np.minimum(Z0, Z1) > 0.8) & (np.maximum(Z0, Z1) < ROAD_ZMAX)
            & (np.abs(dZ) > np.abs(dX)))  # running away from it on the ground: road edges and markings, not crossings
    if keep.sum() < ROAD_MIN_SEGS:
        return None
    X0, Z0, X1, Z1 = X0[keep], Z0[keep], X1[keep], Z1[keep]
    # a piece's weight: its length in the frame over its distance squared --
    # how precisely it fixes a direction on the ground (a pixel far off is
    # metres of road; its length on the ground would trust the far ones most)
    L = np.hypot(*(seg[keep, 2:] - seg[keep, :2]).T) / ((Z0 + Z1) / 2) ** 2
    # The road's shape from the segments' DIRECTIONS on the ground first: a
    # piece of any edge runs at slope dX/dZ = psi + c0 Z + c1 Z^2/2 - cv Z x
    # (x: its edge's offset; the crest term is how edges diverge or converge
    # with distance) -- which edge it belongs to doesn't change that, so the
    # shape needs no grouping. Then each piece's offset, the edges grouped by
    # it, and the crest from how they spread; twice round.
    Zm, Xm = (Z0 + Z1) / 2, (X0 + X1) / 2
    slope = (X1 - X0) / (Z1 - Z0)
    xk = np.zeros(len(L))
    cv = 0.0
    inl = np.ones(len(L), bool)  # robust: pieces whose slope disagrees by more than 3 robust sigmas (texture, a crossing) drop out
    for it in range(4):
        # heading and curvature (its rate, c1, can't be told from them at
        # this range: left free, it traded against both), and the crest
        A = np.c_[np.ones_like(Zm), Zm, -Zm * xk]
        sw = np.sqrt(L * inl)
        cols = 3 if np.ptp(xk) > ROAD_GAP else 2  # one edge can't tell a crest from a turn
        sol, *_ = np.linalg.lstsq(A[:, :cols] * sw[:, None], slope * sw, rcond=None)
        psi, c0, c1 = sol[0], sol[1], 0.0
        cv = float(np.clip(sol[2], ROAD_CV[0], ROAD_CV[-1])) if cols == 3 else 0.0
        shape = psi * Zm + c0 * Zm ** 2 / 2 + c1 * Zm ** 3 / 6
        xk = (Xm - shape) * (1.0 + cv * Zm ** 2 / 2)  # each piece's offset, undone of the crest's narrowing
        res = slope - (psi + c0 * Zm + c1 * Zm ** 2 / 2 - cv * Zm * xk)
        mad = 1.4826 * float(np.median(np.abs(res[inl]))) if inl.any() else 0.0
        inl = np.abs(res) <= max(0.02, 3.0 * mad)
        if inl.sum() < ROAD_MIN_SEGS:
            return None
    L = L * inl  # what stays out of the fit stays out of the edges
    order = np.argsort(xk)
    label = np.zeros(len(L), int)
    for a, b in zip(order[:-1], order[1:]):
        label[b] = label[a] + (1 if xk[b] - xk[a] > ROAD_GAP else 0)
    K = label.max() + 1
    offs = np.array([np.average(xk[label == k], weights=L[label == k] + 1e-12) for k in range(K)])
    support = np.array([L[label == k].sum() for k in range(K)])
    keep_k = support >= 0.15 * support.max()  # an edge needs some length behind it: a stray piece is no edge
    offs_e = offs[keep_k]
    r = slope - (psi + c0 * Zm + c1 * Zm ** 2 / 2 - cv * Zm * xk)
    rms = float(np.sqrt(np.average(r ** 2, weights=L + 1e-12)))  # in slope: 0.05 is 3 degrees on the ground
    left, right = offs_e[offs_e < 0], offs_e[offs_e > 0]
    offset = -(left.max() + right.min()) / 2.0 if len(left) and len(right) else 0.0
    nE = int(keep_k.sum())
    conf = float(min(1.0, len(L) / 12.0) * math.exp(-rms / 0.05) * (1.0 if nE >= 2 else 0.6) * (1.0 if len(left) and len(right) else 0.7))
    lines = []
    for k in np.flatnonzero(keep_k):
        zk = np.r_[Z0[label == k], Z1[label == k]]
        Zs = np.linspace(max(0.8, zk.min()), zk.max(), 12)
        Xs = offs[k] / (1 + cv * Zs ** 2 / 2) + psi * Zs + c0 * Zs ** 2 / 2 + c1 * Zs ** 3 / 6
        fx, fy = to_frame(Xs, Zs)
        lines.append([[round(float(a) / W0, 4), round(float(b) / H0, 4)] for a, b in zip(fx, fy)])
    return {"heading": float(psi), "curve": float(c0), "crest": float(cv), "offset": float(offset), "conf": conf, "lines": lines}


class RoadFilter:
    """A scalar Kalman filter per road estimate: it trusts a fit as much as its confidence."""

    def __init__(self):
        self.m, self.P = {}, {}
        self.conf = 0.0  # how sure, smoothed: a few frames of no road fade it

    def update(self, est: dict | None) -> dict:
        for k in ROAD_Q:
            if k in self.P:
                self.P[k] += ROAD_Q[k]
        if est is not None and est["conf"] > 0.0:
            for k in ROAD_Q:
                R = ROAD_R[k] / max(1e-3, est["conf"])
                if k not in self.m:
                    self.m[k], self.P[k] = est[k], R
                else:
                    g = self.P[k] / (self.P[k] + R)
                    self.m[k] += g * (est[k] - self.m[k]); self.P[k] *= (1 - g)
        self.conf = 0.8 * self.conf + 0.2 * (est["conf"] if est is not None else 0.0)  # (H) ~5 frames
        return dict(self.m, conf=self.conf)


def place_print(grey: np.ndarray) -> np.ndarray:
    """A coarse, light-insensitive print of the frame's structure: the
    orientation histogram of its gradients on a 4 x 4 grid, normalised --
    taken at half size (a 4 x 4 print needs no finer gradients; a quarter of
    the pixels)."""
    g = cv2.resize(grey, (max(8, grey.shape[1] // 2), max(8, grey.shape[0] // 2)), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    b = (ang * np.float32(PRINT_BINS / np.pi)).astype(np.int32) % PRINT_BINS  # orientation, folded over a half turn (an edge's two sides alike)
    h, w = g.shape
    ys, xs = np.minimum(np.arange(h) * PRINT_GRID // h, PRINT_GRID - 1), np.minimum(np.arange(w) * PRINT_GRID // w, PRINT_GRID - 1)
    cell = (ys[:, None] * PRINT_GRID + xs[None, :]) * PRINT_BINS + b
    v = np.bincount(cell.ravel(), weights=mag.ravel(), minlength=PRINT_GRID * PRINT_GRID * PRINT_BINS).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class Organs:
    """Its organs over a sequence of grey frames (one stream): step(grey)
    returns this frame's {"cv_horizon", "cv_roll", "cv_hconf", "vo_yaw",
    "vo_conf", "place_sim"} (None where an organ is off or has nothing)."""

    def __init__(self, on: dict | None = None, seed: int = 0):
        self.on = on or settings()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.focal = None      # measured, then kept (median of the last measurements)
        self._focals: list = []
        self.hz = None          # the last estimate, held between estimates
        self.prev = None        # the previous frame, for odometry
        self.print_long = None  # the place's print, slowly averaged
        self.print_short = None # the last few seconds' print
        self._sim = [0, 0.0, 0.0]  # the place similarity's own running count, mean and M2 (Welford)
        self._hzs: list = []    # its horizon's last estimates: the road needs one its organism would believe
        self.road_f = RoadFilter()

    def step(self, grey: np.ndarray, shift: tuple | None = None) -> dict:
        """shift: the frame's global shift from the last (fractions of width
        and height, organism.global_shift), already measured beside it -- a
        still frame needs no essential matrix (most of the fleet's cameras
        are still, and odometry is the dearest organ, ~10 ms a frame)."""
        out = {"cv_horizon": None, "cv_roll": None, "cv_hconf": 0.0, "vo_yaw": None, "vo_conf": 0.0, "place_sim": None, "new_place": 0.0,
               "road_heading": None, "road_curve": None, "road_crest": None, "road_offset": None, "road_conf": 0.0, "road_lines": None}
        if self.on.get("horizon") and self.t % HORIZON_EVERY == 0:
            est = horizon(grey, self.focal, self.rng, math.radians(self.on.get("max_roll_deg", math.degrees(MAX_ROLL))))
            if est is not None:
                self._hzs = (self._hzs + [est["y"]])[-9:]
            if est is not None:
                if est["focal"]:
                    self._focals = (self._focals + [est["focal"]])[-15:]
                    self.focal = float(np.median(self._focals))
                self.hz = est
        if self.on.get("horizon") and self.hz is not None:
            out.update(cv_horizon=self.hz["y"], cv_roll=self.hz["roll"], cv_hconf=self.hz["conf"])
        if self.on.get("road"):
            from .organism import HORIZON_MAX_SPREAD
            q1, q3 = np.percentile(self._hzs, [25, 75]) if len(self._hzs) >= 3 else (0.0, 1.0)
            believed = self.hz is not None and len(self._hzs) >= 3 and q3 - q1 <= HORIZON_MAX_SPREAD
            est = road(grey, {"y": float(np.median(self._hzs)), "roll": self.hz["roll"]}, self.focal, self.rng) if believed else None
            m = self.road_f.update(est)
            if m.get("heading") is not None:
                out.update(road_heading=m["heading"], road_curve=m["curve"], road_crest=m["crest"], road_offset=m["offset"])
            out.update(road_conf=m["conf"], road_lines=est["lines"] if est is not None else None)
        if self.on.get("odometry") and self.prev is not None and self.prev.shape == grey.shape:
            h, w = grey.shape[:2]
            if shift is not None and math.hypot(shift[0] * w, shift[1] * h) < STILL_PX:
                yaw, conf = 0.0, 1.0  # still: no turn, surely
            else:
                yaw, conf = self._turn(self.prev, grey)
            out.update(vo_yaw=yaw, vo_conf=conf)
        if self.on.get("place"):
            p = place_print(grey)
            self.print_short = p if self.print_short is None else 0.7 * self.print_short + 0.3 * p
            self.print_long = p if self.print_long is None else 0.98 * self.print_long + 0.02 * p
            sim = float(np.dot(self.print_short, self.print_long) /
                        max(1e-9, np.linalg.norm(self.print_short) * np.linalg.norm(self.print_long)))
            out["place_sim"] = sim
            # A new place: its last seconds unlike this place's print by more than
            # its own surprise line (organism.SURPRISE_SIGMAS of its own spread,
            # once it has a spread): the print starts over from here -- a stream
            # that cuts, a lift's doors onto another floor (docs/next-designs.md).
            from .organism import SURPRISE_SIGMAS
            n, mean, m2 = self._sim
            if n >= 100 and sim < mean - SURPRISE_SIGMAS * math.sqrt(m2 / (n - 1)):
                out["new_place"] = 1.0
                self.print_long = self.print_short.copy()
                self._sim = [0, 0.0, 0.0]
            else:
                n += 1
                delta = sim - mean
                mean += delta / n
                self._sim = [n, mean, m2 + delta * (sim - mean)]
        self.prev = grey
        self.t += 1
        return out

    def _turn(self, a: np.ndarray, b: np.ndarray) -> tuple[float | None, float]:
        """Its turning between two frames (radians, + right) from the
        essential matrix of their tracked corners, and the inlier share."""
        pts = cv2.goodFeaturesToTrack(a, 80, 0.01, 7)
        if pts is None or len(pts) < 12:
            return None, 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        ok = st.ravel() == 1
        p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
        if len(p0) < 12:
            return None, 0.0
        h, w = a.shape[:2]
        f = self.focal or (w / 2.0) / math.tan(math.radians(PRIOR_HFOV_DEG) / 2.0)
        if float(np.median(np.hypot(*(p1 - p0).T))) < STILL_PX:
            return 0.0, 1.0  # still: no turn, surely
        try:
            E, mask = cv2.findEssentialMat(p0, p1, focal=f, pp=(w / 2.0, h / 2.0), method=cv2.RANSAC, prob=0.99, threshold=1.0)
            if E is None or E.shape != (3, 3):
                return None, 0.0
            n, R, t, mask2 = cv2.recoverPose(E, p0, p1, focal=f, pp=(w / 2.0, h / 2.0), mask=mask)
        except cv2.error:
            return None, 0.0
        conf = float(n) / len(p0)
        yaw = math.atan2(R[0, 2], R[2, 2])
        return -yaw, conf


def series(frames: list[np.ndarray], on: dict | None = None, shifts=None) -> dict:
    """Its organs over a recorded snapshot: each output as an array, one per
    frame (shifts: each frame's global shift, as Organs.step takes it)."""
    o = Organs(on)
    rows = [o.step(f, None if shifts is None else shifts[k]) for k, f in enumerate(frames)]
    return {k: np.array([r[k] if r[k] is not None else np.nan for r in rows], dtype=float) for k in rows[0] if k not in NON_SCALAR} if rows else {}


class _Running:
    """Its organs kept running across snapshots of one live feed: a new
    snapshot (every ~10 s) shares nearly all its frames with the last, so
    only the frames not seen yet are stepped -- as its live body steps each
    frame once. Keyed by the feed's frame index, checked by a fingerprint of
    each frame (a source switch restarts the indices: then it starts over)."""

    def __init__(self, on):
        self.organs = Organs(on)
        self.rows: dict[int, tuple] = {}   # feed index -> (fingerprint, outputs)
        self.next = None

    @staticmethod
    def fingerprint(f: np.ndarray) -> int:
        return int(f[::16, ::16].astype(np.int64).sum())


_RUNNING: _Running | None = None


def series_running(frames: list[np.ndarray], first: int, shifts=None, on: dict | None = None) -> dict:
    """series() for a live feed's snapshot whose first frame is the feed's
    frame `first`: the same outputs, computing only frames not seen before."""
    global _RUNNING
    on = on or settings()
    prints = [_Running.fingerprint(f) for f in frames]
    r = _RUNNING
    fresh = (r is None or r.organs.on != on or not r.rows or first < min(r.rows) or first > r.next
             or any(first + k in r.rows and r.rows[first + k][0] != p for k, p in enumerate(prints)))
    if fresh:
        r = _RUNNING = _Running(on)
        r.next = first
    rows = []
    for k, f in enumerate(frames):
        idx = first + k
        if idx in r.rows:
            rows.append(r.rows[idx][1])
            continue
        out = r.organs.step(f, None if shifts is None else shifts[k])
        r.rows[idx] = (prints[k], out)
        r.next = idx + 1
        rows.append(out)
    for idx in [i for i in r.rows if i < first]:  # older than this snapshot: never asked for again
        del r.rows[idx]
    return {k: np.array([row[k] if row[k] is not None else np.nan for row in rows], dtype=float) for k in rows[0] if k not in NON_SCALAR} if rows else {}
