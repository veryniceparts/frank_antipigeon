"""camera frame -> motion boxes -> track -> (scheduled, batched) crop classification -> pigeon boxes."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .motion import MotionDetector, square_crops
from .tracker import IoUTracker, Track


@dataclass
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # xyxy full-res
    p: float                        # smoothed P(pigeon)
    pigeon: bool


class Timer:
    def __init__(self):
        self.t = {}
        self.n = 0

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt

    def report(self) -> str:
        n = max(1, self.n)
        return "  ".join(f"{k}={v / n * 1e3:.3f}ms" for k, v in self.t.items())


class PigeonPipeline:
    def __init__(self, classifier, motion: MotionDetector | None = None, tracker: IoUTracker | None = None,
                 pad: float = 0.2):
        self.clf = classifier
        self.motion = motion or MotionDetector()
        self.tracker = tracker or IoUTracker(threshold=getattr(classifier, "threshold", 0.5))
        self.pad = pad
        self.f = 0
        self.timer = Timer()
        self.n_classified = 0

    def process(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        boxes = self.motion.detect(frame)
        t1 = time.perf_counter()
        todo = self.tracker.update(boxes, self.f)
        t2 = time.perf_counter()
        if todo:
            tb = np.array([self.tracker.tracks[i].box for i in todo])
            crops = square_crops(frame, tb, self.clf.size, pad=self.pad)
            t3 = time.perf_counter()
            probs = self.clf(crops)  # one batched call per frame
            for i, p in zip(todo, probs):
                self.tracker.feed(i, float(p), self.f)
            self.n_classified += len(todo)
            t4 = time.perf_counter()
            self.timer.add("crop", t3 - t2)
            self.timer.add("classify", t4 - t3)
        self.timer.add("motion", t1 - t0)
        self.timer.add("track", t2 - t1)
        self.timer.n += 1
        dets = [Detection(t.id, tuple(int(v) for v in t.box), t.p, self.tracker.is_pigeon(t))
                for t in self.tracker.active(self.f)]
        self.f += 1
        return dets


def draw(frame: np.ndarray, dets: list[Detection], show_all: bool = True, fps: float | None = None) -> np.ndarray:
    if frame.ndim == 2:
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        out = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        out = frame
    for d in dets:
        if not d.pigeon and not show_all:
            continue
        x0, y0, x1, y1 = d.box
        col = (0, 0, 255) if d.pigeon else (160, 160, 160)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2 if d.pigeon else 1)
        label = f"PIGEON {d.p:.2f}" if d.pigeon else f"{d.p:.2f}"
        cv2.putText(out, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"{fps:.0f} fps", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out
