"""Moving-object proposals: background subtraction on a downscaled gray frame.

Measured on a 4-core Xeon, 1280x720 (median ms/frame incl. gray+resize):
    MOG2 1/4: 0.9   KNN 1/4: 1.0   CNT 1/4: 0.32   frame-diff: 0.27
CNT (opencv-contrib ``cv2.bgsegm``) is used when available, MOG2 otherwise.
findContours+boundingRect is ~10x faster than connectedComponentsWithStats.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(
        self,
        scale: int = 4,             # process at 1/scale resolution
        fps: float = 100.0,         # camera rate, sets CNT background memory
        method: str = "auto",       # auto | cnt | mog2 | diff
        min_area: int = 12,         # px on the *downscaled* mask
        max_area_frac: float = 0.2, # bigger blobs = lighting change / camera shake -> ignore
        merge_gap: int = 2,         # px (downscaled): merge boxes closer than this (flapping wings)
        ignore_mask: np.ndarray | None = None,  # full-res uint8, 255 = ignore (trees, flags)
        var_threshold: float = 32,  # MOG2 only; default 16 is too noisy for sky/foliage
    ):
        self.s = int(scale)
        self.min_area, self.max_area_frac, self.merge_gap = min_area, max_area_frac, merge_gap
        if method == "auto":
            method = "cnt" if hasattr(cv2, "bgsegm") else "mog2"
        self.method = method
        if method == "cnt":
            self.bg = cv2.bgsegm.createBackgroundSubtractorCNT(
                minPixelStability=15, useHistory=True,
                maxPixelStability=int(15 * fps), isParallel=True)
        elif method == "mog2":
            self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_threshold, detectShadows=False)
        elif method == "diff":
            self.bg, self._prev = None, None
        else:
            raise ValueError(method)
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._ignore_full = ignore_mask
        self._ignore = None
        self.mask = None  # last foreground mask (downscaled), for debugging

    def _small(self, frame: np.ndarray) -> np.ndarray:
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.s > 1:  # gray first, then INTER_LINEAR (INTER_AREA is ~9x slower)
            g = cv2.resize(g, (g.shape[1] // self.s, g.shape[0] // self.s), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(g, (5, 5), 0)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return (N,4) int32 boxes xyxy in full-resolution pixels."""
        small = self._small(frame)
        if self.method == "diff":
            prev, self._prev = self._prev, small
            if prev is None:
                return np.zeros((0, 4), np.int32)
            _, m = cv2.threshold(cv2.absdiff(small, prev), 15, 255, cv2.THRESH_BINARY)
        else:
            m = self.bg.apply(small)
            if self.method == "mog2":
                m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)[1]
        if self._ignore_full is not None:
            if self._ignore is None or self._ignore.shape != m.shape:
                self._ignore = cv2.resize(self._ignore_full, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            m[self._ignore] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k)
        m = cv2.dilate(m, self.k, iterations=2)  # also glues wing/body fragments
        self.mask = m
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = self.max_area_frac * m.shape[0] * m.shape[1]
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            a = w * h
            if self.min_area <= a <= max_area:
                boxes.append((x, y, x + w, y + h))
        if not boxes:
            return np.zeros((0, 4), np.int32)
        b = merge_boxes(np.array(boxes, np.int32), self.merge_gap)
        return b * self.s


def merge_boxes(b: np.ndarray, gap: int) -> np.ndarray:
    """Union boxes whose (gap-expanded) rectangles overlap, until stable."""
    b = b.copy()
    changed = True
    while changed and len(b) > 1:
        changed = False
        out = []
        used = np.zeros(len(b), bool)
        for i in range(len(b)):
            if used[i]:
                continue
            cur = b[i].copy()
            used[i] = True
            for j in range(i + 1, len(b)):
                if used[j]:
                    continue
                o = b[j]
                if (cur[0] - gap <= o[2] and o[0] - gap <= cur[2] and
                        cur[1] - gap <= o[3] and o[1] - gap <= cur[3]):
                    cur = np.array([min(cur[0], o[0]), min(cur[1], o[1]), max(cur[2], o[2]), max(cur[3], o[3])], np.int32)
                    used[j] = True
                    changed = True
            out.append(cur)
        b = np.array(out, np.int32)
    return b


def square_crops(frame: np.ndarray, boxes: np.ndarray, size: int, pad: float = 0.2, min_pad: int = 4):
    """Cut padded square crops around boxes and resize to size x size (uint8, same channels).

    Square + padding keeps the bird's aspect ratio and some context, matching how
    the classifier is trained (see scripts/harvest.py which uses the same function).
    Out-of-frame parts are filled by edge replication.
    """
    H, W = frame.shape[:2]
    n = len(boxes)
    ch = 1 if frame.ndim == 2 else frame.shape[2]
    out = np.empty((n, size, size, 3), np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        side = max(x1 - x0, y1 - y0)
        side = int(side + 2 * max(min_pad, pad * side))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        a0, b0 = cx - side // 2, cy - side // 2
        a1, b1 = a0 + side, b0 + side
        c = frame[max(0, b0):min(H, b1), max(0, a0):min(W, a1)]
        if a0 < 0 or b0 < 0 or a1 > W or b1 > H:
            c = cv2.copyMakeBorder(c, max(0, -b0), max(0, b1 - H), max(0, -a0), max(0, a1 - W), cv2.BORDER_REPLICATE)
        interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        c = cv2.resize(c, (size, size), interpolation=interp)
        if ch == 1:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        out[i] = c
    return out
