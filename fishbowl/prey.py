from __future__ import annotations

"""
Prey -- what the organism can eat. Fixed, human-written, never evolved,
same category as retina.py and reflexes.py.

YOLO (yolov8n, the model the HLS livecam project already uses) plays the
role of the world's physics of food, NOT the organism's eyes: it decides
where living things (people, animals) are, and the organism eats only
when one of them is inside the center of its gaze. The organism itself
still sees nothing but its receptors, so its perception stays
self-built. Detection runs on the full-resolution colour frame at
capture time; only the resulting boxes (class, confidence, normalized
position) are kept -- the frame itself is never stored or sent anywhere.

YOLO is a SHORTCUT. The goal was always for the organism to grow its own
prey detector. Planned next step: its own perception learns "prey in my
gaze center or not" from its receptors with YOLO as the teacher (a
student running in parallel), scored on agreement; once it is reliable,
YOLO is weaned off and it eats by its own judgement. It only ever needs
people and animals -- not the 80 COCO classes.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# The model: CAMBRIAN_PREY_MODEL if set; else models/yolov8n.onnx next to the
# code (the packaged layout, every platform); else the first Linux hosts'
# /srv/cambrian/models.
_LOCAL_MODEL = Path(__file__).resolve().parents[1] / "models" / "yolov8n.onnx"
DEFAULT_MODEL = Path(os.environ.get("CAMBRIAN_PREY_MODEL")
                     or (_LOCAL_MODEL if _LOCAL_MODEL.exists() else "/srv/cambrian/models/yolov8n.onnx"))

# COCO classes that are living things: prey.
# Its food, the clade's rule (a design panel, 2026-09-27): living things --
# "protein", like every mosquito's blood meal or every crocodile's catch.
# Plants and COCO's food items are not food for it. Which of these it is
# drawn to evolves (genome.host_pref); people are always food and always
# sensed, so every lineage can track people.
PREY_CLASSES = {0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
                19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}
PERSON_CLASS = 0
# The detector's own calibration (Ultralytics' shipped defaults for YOLOv8
# prediction), not a hand-picked cut: confidence 0.25, NMS IoU 0.7. Food is
# weighted by confidence anyway, so a 0.3 bird counts, just less than a sure one.
MIN_CONFIDENCE = 0.25
NMS_IOU = 0.7
INPUT_SIZE = 640  # this export only accepts 640x640
LETTERBOX_GREY = 114  # the pad value YOLO's letterbox uses in training


class PreyDetector:
    """yolov8n via OpenCV's own ONNX loader (no extra dependency). If the
    model file is missing, detect() returns no prey -- the organism then
    only has its small surprise snack to live on."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        # 2 threads: measured on Tanzania, 129 ms per detection vs ~200 ms
        # for OpenCV's default fan-out, which also spread ~160% CPU across
        # worker threads competing with evolution itself.
        cv2.setNumThreads(2)
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
        # Letterboxed, as YOLO was trained: scaled to fit with its aspect kept,
        # centred, padded grey. (Stretching a 16:9 frame to a square distorted
        # every shape -- small ones like birds most.)
        scale = INPUT_SIZE / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        px, py = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), LETTERBOX_GREY, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # (8400, 84): cx, cy, w, h, then 80 class scores
        scores = out[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(scores)), cls]
        keep = (conf >= MIN_CONFIDENCE) & np.isin(cls, list(PREY_CLASSES))
        if not keep.any():
            return []
        raw = out[keep, :4]
        cls, conf = cls[keep], conf[keep]
        # From the letterboxed square back to the frame, normalized to [0, 1].
        b = np.stack([(raw[:, 0] - px) / (nw), (raw[:, 1] - py) / (nh), raw[:, 2] / nw, raw[:, 3] / nh], axis=1)
        x0, y0 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
        boxes_px = [[float(x * w), float(y * h), float(bw * w), float(bh * h)] for x, y, bw, bh in zip(x0, y0, b[:, 2], b[:, 3])]
        idx = cv2.dnn.NMSBoxes(boxes_px, conf.astype(float).tolist(), MIN_CONFIDENCE, NMS_IOU)
        result = []
        for i in np.array(idx).reshape(-1):
            result.append([int(cls[i]), round(float(conf[i]), 3),
                           round(float(np.clip(x0[i], 0, 1)), 4), round(float(np.clip(y0[i], 0, 1)), 4),
                           round(float(np.clip(x0[i] + b[i, 2], 0, 1)), 4), round(float(np.clip(y0[i] + b[i, 3], 0, 1)), 4)])
        return result


def prey_in_window(boxes: list[list[float]], cx: float, cy: float, hx: float, hy: float) -> float:
    """How much of the WHOLE gaze window prey covers, 0..1, confidence-
    weighted: the teacher's label the perception tree learns to predict from
    its own pixels (run_vision.py) -- "is there food in what I'm looking at".
    hx, hy: half the gaze's width and height (fovea.FoveaState.half_extents).
    """
    return _coverage(boxes, cx, cy, hx, hy)


def _coverage(boxes: list[list[float]], cx: float, cy: float, hx: float, hy: float) -> float:
    if not boxes or hx <= 0 or hy <= 0:
        return 0.0
    area = 4 * hx * hy
    total = 0.0
    for _, conf, x0, y0, x1, y1 in boxes:
        ix = max(0.0, min(x1, cx + hx) - max(x0, cx - hx))
        iy = max(0.0, min(y1, cy + hy) - max(y0, cy - hy))
        total += conf * min(1.0, ix * iy / area)
    return float(min(1.0, total))


# Its mouth: a square at the centre of its gaze, of a FIXED size -- physics,
# a given (a hagfish's mouth and a human's are much alike; not worth
# evolving). Exactly the eating zone a newborn's eye had (the central half of
# a 22-receptor gaze), now the same for every eye, so default eyes eat as
# before; square like its eye's mosaic (a design panel, 7-2: a round mouth
# of the same width would have cut every catch zone to pi/4 of its area). A
# mouth, not a point: a point made a catch rarer than any real mouth does
# (measured on tina: 0% of looks).
MOUTH_SIDE = 22 / 64 / 2  # of the frame's height (fovea: DEFAULT_RECEPTORS x RECEPTOR_PITCH / 2)


def prey_at_mouth(boxes: list[list[float]], cx: float, cy: float, aspect: float) -> float:
    """
    The meal a look catches, 0..1: prey under its mouth (a square MOUTH_SIDE
    wide at the gaze centre) -- the detector's confidence for it (the surest,
    if several), i.e. the chance the catch is real, times one bite.
    The meal is the prey item's, whatever the eye's size and however big the
    prey looks: in no living animal does eye size set how much a bite yields
    (after a design panel -- Land & Nilsson: eyes set finding, not eating;
    Holling 1959: a catch yields the prey's worth; where meal size is limited,
    it is by the mouth: gape limitation). Small or distant prey is harder to
    hit, not less filling. Glancing at prey from the edge of the gaze doesn't
    feed it; holding it centred does -- so following a moving person is
    literally how it eats. aspect: the frame's width / height.

    (It used to be the share of the gaze's central half the prey covered,
    which made the eye a mouth: a bigger eye diluted every meal, and prey
    far away fed less than the same prey nearby.)
    """
    hy = MOUTH_SIDE / 2.0
    hx = hy / aspect  # square in pixels
    best = 0.0
    for _, conf, x0, y0, x1, y1 in boxes or ():
        if min(x1, cx + hx) > max(x0, cx - hx) and min(y1, cy + hy) > max(y0, cy - hy):
            best = max(best, float(conf))
    return best
