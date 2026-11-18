"""Greedy IoU tracker with per-track pigeon-probability EMA and classify scheduling.

At hundreds of fps objects move a few pixels per frame, so a Kalman filter adds
little; greedy max-IoU matching (+ centroid fallback) costs ~50 us for 20 blobs.
The classifier runs on a track when it is new, then every ``n_unsure`` frames
until the track is confident, then every ``n_sure`` frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


@dataclass
class Track:
    id: int
    box: np.ndarray
    first: int
    seen: int
    hits: int = 1
    p: float = 0.5       # EMA of P(pigeon)
    n_cls: int = 0       # classifier evaluations so far
    next_cls: int = 0    # frame index when it should be classified next
    name: str = ""       # optional species name from the classifier (iNat teacher)

    @property
    def confident(self) -> bool:
        return self.n_cls >= 2 and abs(self.p - 0.5) * 2 > 0.7


class IoUTracker:
    def __init__(self, iou_thr=0.1, max_age=15, alpha=0.6, n_unsure=5, n_sure=60,
                 center_gate=1.0, min_hits=3, threshold=0.6):
        self.iou_thr, self.max_age, self.alpha = iou_thr, max_age, alpha
        self.n_unsure, self.n_sure, self.center_gate = n_unsure, n_sure, center_gate
        self.min_hits, self.threshold = min_hits, threshold
        self.tracks: dict[int, Track] = {}
        self._nid = 0

    def update(self, boxes: np.ndarray, f: int) -> list[int]:
        """Associate boxes for frame f. Returns ids of tracks that need classification now."""
        ids = list(self.tracks)
        matched_t, matched_b = set(), set()
        if ids and len(boxes):
            tb = np.array([self.tracks[i].box for i in ids])
            M = iou_matrix(tb, boxes)
            # centroid fallback for small fast objects that don't overlap frame-to-frame
            tc = (tb[:, :2] + tb[:, 2:]) / 2
            bc = (boxes[:, :2] + boxes[:, 2:]) / 2
            d = np.linalg.norm(tc[:, None] - bc[None], axis=2)
            size = np.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])[:, None]
            near = d < self.center_gate * np.maximum(size, 8)
            score = np.where(M >= self.iou_thr, 1 + M, np.where(near, 1 - d / (size * 4 + 1e-9), -1))
            for k in np.argsort(-score, axis=None):
                r, c = divmod(int(k), score.shape[1])
                if score[r, c] <= 0:
                    break
                if r in matched_t or c in matched_b:
                    continue
                t = self.tracks[ids[r]]
                t.box, t.seen, t.hits = boxes[c], f, t.hits + 1
                matched_t.add(r)
                matched_b.add(c)
        for c in range(len(boxes)):
            if c not in matched_b:
                self.tracks[self._nid] = Track(self._nid, boxes[c], f, f, next_cls=f)
                self._nid += 1
        self.tracks = {i: t for i, t in self.tracks.items() if f - t.seen <= self.max_age}
        return [i for i, t in self.tracks.items() if t.seen == f and f >= t.next_cls]

    def feed(self, tid: int, p: float, f: int):
        t = self.tracks[tid]
        t.p = p if t.n_cls == 0 else self.alpha * t.p + (1 - self.alpha) * p
        t.n_cls += 1
        t.next_cls = f + (self.n_sure if t.confident else self.n_unsure)

    def active(self, f: int) -> list[Track]:
        return [t for t in self.tracks.values() if t.seen == f]

    def is_pigeon(self, t: Track) -> bool:
        return t.hits >= self.min_hits and t.n_cls > 0 and t.p >= self.threshold
