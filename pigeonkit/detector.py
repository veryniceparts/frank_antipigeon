"""Bird proposals from a COCO SSD detector (alternative to MotionDetector).

Use it when motion detection can't work: moving / panning camera (most YouTube clips),
perched or walking birds. Same interface as MotionDetector: detect(frame) -> (N,4) xyxy boxes.
The pigeon/dove decision is still made by the crop classifier, so the detector only needs to
find "bird" (COCO class 15) with high recall.

Models (Apache-2.0, Google Coral test_data; CPU via LiteRT, or *_edgetpu.tflite on a Coral):
  ssdlite_mobiledet_coco_qat_postprocess.tflite   320x320, better on small objects
  ssd_mobilenet_v2_coco_quant_postprocess.tflite  300x300
"""
from __future__ import annotations

import os

import cv2
import numpy as np

SSD_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/ssdlite_mobiledet_coco_qat_postprocess.tflite"
COCO_BIRD = 15


class SSDBirdDetector:
    def __init__(self, path: str | None = None, score: float = 0.3, threads: int = 2, every: int = 1,
                 min_size: int = 12, classes=(COCO_BIRD,)):
        """every: run the network every N frames and reuse boxes in between (saves CPU)."""
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter
        path = path or download_detector()
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]
        outs = self.it.get_output_details()
        # TFLite_Detection_PostProcess: boxes [1,N,4] (ymin,xmin,ymax,xmax norm.), classes, scores, count
        self.o_boxes, self.o_cls, self.o_score, self.o_n = (o["index"] for o in outs[:4])
        self.score, self.every, self.min_size, self.classes = score, every, min_size, set(classes)
        self._n = 0
        self._last = np.zeros((0, 4), np.int32)
        self.s = 1  # MotionDetector compatibility (pipeline checks .s for Bayer input)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        self._n += 1
        if (self._n - 1) % self.every:
            return self._last
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        x = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)[None]
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        n = int(self.it.get_tensor(self.o_n)[0])
        b = self.it.get_tensor(self.o_boxes)[0][:n]
        c = self.it.get_tensor(self.o_cls)[0][:n].astype(int)
        s = self.it.get_tensor(self.o_score)[0][:n]
        keep = (s >= self.score) & np.isin(c, list(self.classes))
        b = b[keep]
        boxes = np.stack([b[:, 1] * W, b[:, 0] * H, b[:, 3] * W, b[:, 2] * H], 1).astype(np.int32) if len(b) \
            else np.zeros((0, 4), np.int32)
        boxes = np.clip(boxes, 0, [W - 1, H - 1, W - 1, H - 1])
        ok = ((boxes[:, 2] - boxes[:, 0]) >= self.min_size) & ((boxes[:, 3] - boxes[:, 1]) >= self.min_size)
        boxes, scores = boxes[ok], s[keep][ok]
        k = suppress_nested(boxes, scores)
        self._last, self.scores = boxes[k], scores[k]
        return self._last


def suppress_nested(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45, inside_thr: float = 0.7):
    """Greedy NMS that also drops a box lying mostly inside a kept one (SSD often returns the
    whole bird plus its body as two boxes -> two tracks, double labels)."""
    order = np.argsort(-scores)
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in order:
        dup = False
        for j in keep:
            iw = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            ih = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            if inter / (area[i] + area[j] - inter) > iou_thr or inter / max(1, min(area[i], area[j])) > inside_thr:
                dup = True
                break
        if not dup:
            keep.append(i)
    return np.array(keep, int)


def download_detector(dst_dir: str = "models") -> str:
    import urllib.request
    os.makedirs(dst_dir, exist_ok=True)
    out = os.path.join(dst_dir, os.path.basename(SSD_URL))
    if not os.path.exists(out):
        urllib.request.urlretrieve(SSD_URL, out)
    return out
