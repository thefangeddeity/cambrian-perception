"""
Synthetic worlds for its engineered organs: frames drawn from a known
camera, so their horizon is exact (no eye judging it). Each world is
(name, frames, truth): truth is the horizon's height at the frame's middle
(0 top, 1 bottom; beyond 0..1 when out of view).

  floor-<yaw>     a tiled floor seen by a camera turned <yaw> degrees
  floor-boxed     the same, letterboxed (a stream's black bars)
  sea-palm        a sea's edge, with a palm's fronds converging on a crown
                  above the frame (a vanishing point that is not on the horizon)
  room-<p>-<y>-<r>  a bare room (four walls, floor, ceiling as plain shades)
                  from eye height, pitched up <p>, turned <y>, rolled <r>
"""
from __future__ import annotations

import math

import cv2
import numpy as np

W, H = 320, 180
F = (W / 2) / math.tan(math.radians(32.5))  # the organ's prior: 65 degrees across


def floor(yaw_deg: float = 35.0, horizon: float = 0.4) -> np.ndarray:
    img = np.full((H, W), 90, np.uint8)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def P(X, Z):
        Xc, Zc = c * X - s * Z, s * X + c * Z
        return (W / 2 + 250.0 * Xc / Zc, horizon * H + 250.0 / Zc) if Zc > 0.3 else None
    for i in range(-20, 21):
        for a, b in [((i, z), (i, z + 0.5)) for z in np.arange(0.5, 40, 0.5)] + [((x, i), (x + 0.5, i)) for x in np.arange(-20, 20, 0.5)]:
            p, q = P(*a), P(*b)
            if p and q and max(abs(p[0]), abs(q[0]), abs(p[1]), abs(q[1])) < 5000:
                cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 200, 1, cv2.LINE_AA)
    return img


def sea_palm(horizon: float = 0.45, seed: int = 0) -> np.ndarray:
    img = np.full((H, W), 200, np.uint8)
    img[int(horizon * H):] = 110
    for k in range(14):
        a = math.radians(35 + 2.5 * k)
        cv2.line(img, (330, -40), (int(330 - 150 * math.cos(a)), int(-40 + 150 * math.sin(a))), 40, 2, cv2.LINE_AA)
    return np.clip(img.astype(int) + np.random.default_rng(seed).normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def room(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0, seed: int = 0,
         size=(4.0, 2.6, 5.0), eye: float = 1.5) -> tuple[np.ndarray, float]:
    """A bare room, ray-cast (exact, no clipping), and its true horizon."""
    sx, sy, sz = size
    X, Z = (-sx / 2, sx / 2), (-1.0, sz - 1.0)
    faces = [([(X[0], 0, Z[0]), (X[1], 0, Z[0]), (X[1], 0, Z[1]), (X[0], 0, Z[1])], 120),     # floor
             ([(X[0], sy, Z[0]), (X[1], sy, Z[0]), (X[1], sy, Z[1]), (X[0], sy, Z[1])], 225),  # ceiling
             ([(X[0], 0, Z[1]), (X[1], 0, Z[1]), (X[1], sy, Z[1]), (X[0], sy, Z[1])], 185),     # far wall
             ([(X[0], 0, Z[0]), (X[0], 0, Z[1]), (X[0], sy, Z[1]), (X[0], sy, Z[0])], 165),     # left
             ([(X[1], 0, Z[0]), (X[1], 0, Z[1]), (X[1], sy, Z[1]), (X[1], sy, Z[0])], 200)]     # right
    p, yw, r = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    Ry = np.array([[math.cos(yw), 0, math.sin(yw)], [0, 1, 0], [-math.sin(yw), 0, math.cos(yw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry.T
    img = np.zeros((H, W), np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.stack([(xs - W / 2) / F, -(ys - H / 2) / F, np.ones_like(xs, float)], -1) @ R
    o = np.array([0.0, eye, 0.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        for corners, shade in faces:
            c = np.array(corners, float)
            n = np.cross(c[1] - c[0], c[3] - c[0])
            n /= np.linalg.norm(n)
            den = rays @ n
            t = np.where(np.abs(den) > 1e-9, ((c[0] - o) @ n) / np.where(np.abs(den) > 1e-9, den, 1), np.inf)
            P = o + rays * t[..., None]
            inside = (t > 0) & np.all((P >= c.min(0) - 1e-6) & (P <= c.max(0) + 1e-6), -1) & (t < zbuf)
            img[inside], zbuf[inside] = shade, t[inside]
    img = cv2.GaussianBlur(img, (3, 3), 0.8) + np.random.default_rng(seed).normal(0, 3, img.shape)
    truth = (H / 2 + F * math.tan(p)) / H  # this rotation looks UP by p: the horizon sinks below the middle
    return np.clip(img, 0, 255).astype(np.uint8), truth


ROOM_POSES = [(0, 0, 0), (0, 25, 0), (-10, 15, 0), (8, -20, 0), (-20, 30, 0), (0, 35, 4), (15, 10, 0), (5, 0, 0), (-5, 40, 0)]


def worlds() -> list[tuple[str, list[np.ndarray], float]]:
    out = [(f"floor-{y}", [floor(y)], 0.4) for y in (0, 20, 35)]
    boxed = np.vstack([np.zeros((30, W), np.uint8), floor(35), np.zeros((30, W), np.uint8)])
    out.append(("floor-boxed", [boxed], (0.4 * H + 30) / (H + 60)))
    out.append(("sea-palm", [sea_palm(seed=k) for k in range(3)], 0.45))
    for pose in ROOM_POSES:  # three looks each, a camera's noise differing between them
        looks = [room(*pose, seed=k) for k in range(3)]
        out.append(("room-%+d%+d%+d" % pose, [img for img, _ in looks], looks[0][1]))
    return out
