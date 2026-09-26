from __future__ import annotations

"""
Prey -- what the organism can eat. Fixed, human-written, never evolved,
same category as retina.py and reflexes.py.

YOLO (yolov8n, the model already used by the user's HLS livecam) plays the
role of the world's physics of food, NOT the organism's eyes: it decides
where living things (people, animals) are, and the organism eats only
when one of them is inside the center of its gaze. The organism itself
still sees nothing but its 12x12 grids, so its perception stays
self-built. Detection runs on the full-resolution colour frame at
capture time; only the resulting boxes (class, confidence, normalized
position) are kept -- the frame itself is never stored or sent anywhere.
"""

import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(os.environ.get("CAMBRIAN_PREY_MODEL", "/srv/cambrian/models/yolov8n.onnx"))

# COCO classes that are living things: prey.
PREY_CLASSES = {0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
                19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}
MIN_CONFIDENCE = 0.4
INPUT_SIZE = 640  # this export only accepts 640x640


class PreyDetector:
    """yolov8n via OpenCV's own ONNX loader (no extra dependency). If the
    model file is missing, detect() returns no prey -- the organism then
    only has its small surprise snack to live on."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        self.model_path = Path(model_path)
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path)) if self.model_path.exists() else None

    @property
    def available(self) -> bool:
        return self.net is not None

    def detect(self, bgr: np.ndarray) -> list[list[float]]:
        """[[class_id, confidence, x0, y0, x1, y1], ...], coordinates normalized to [0, 1]."""
        if self.net is None or bgr is None or bgr.ndim != 3:
            return []
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # (8400, 84): cx, cy, w, h, then 80 class scores
        scores = out[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(scores)), cls]
        keep = (conf >= MIN_CONFIDENCE) & np.isin(cls, list(PREY_CLASSES))
        if not keep.any():
            return []
        b = out[keep, :4] / INPUT_SIZE  # normalized: the blob stretches the whole frame to 640x640
        cls, conf = cls[keep], conf[keep]
        x0, y0 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
        boxes_px = [[float(x * w), float(y * h), float(bw * w), float(bh * h)] for x, y, bw, bh in zip(x0, y0, b[:, 2], b[:, 3])]
        idx = cv2.dnn.NMSBoxes(boxes_px, conf.astype(float).tolist(), MIN_CONFIDENCE, 0.45)
        result = []
        for i in np.array(idx).reshape(-1):
            result.append([int(cls[i]), round(float(conf[i]), 3),
                           round(float(np.clip(x0[i], 0, 1)), 4), round(float(np.clip(y0[i], 0, 1)), 4),
                           round(float(np.clip(x0[i] + b[i, 2], 0, 1)), 4), round(float(np.clip(y0[i] + b[i, 3], 0, 1)), 4)])
        return result


def prey_in_gaze(boxes: list[list[float]], cx: float, cy: float, fraction: float) -> float:
    """
    How much prey is in the CENTER of the gaze (its central half), 0..1:
    for each prey box, the share of the gaze center it covers, times the
    detector's confidence. Glancing at prey from the edge of the gaze
    doesn't feed it; holding it centered does -- so following a moving
    person is literally how it eats.
    """
    if not boxes:
        return 0.0
    half = fraction / 4.0  # central half of the gaze, each side
    gx0, gx1, gy0, gy1 = cx - half, cx + half, cy - half, cy + half
    area = (2 * half) ** 2
    total = 0.0
    for _, conf, x0, y0, x1, y1 in boxes:
        ix = max(0.0, min(x1, gx1) - max(x0, gx0))
        iy = max(0.0, min(y1, gy1) - max(y0, gy0))
        total += conf * min(1.0, ix * iy / area) if area > 0 else 0.0
    return float(min(1.0, total))
