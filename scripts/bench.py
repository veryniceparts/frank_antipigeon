#!/usr/bin/env python3
"""Latency benchmark for classifiers and the motion stage.

  python scripts/bench.py --model models/pigeon.onnx models/pigeon.int8.onnx --threads 1 2 4
  python scripts/bench.py --motion --width 1280 --height 720
  python scripts/bench.py --compare models/pigeon.onnx models/pigeon.int8.onnx --data data/labeled
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import MotionDetector, load_classifier  # noqa: E402


def bench_clf(path, threads, batches, reps=200):
    for th in threads:
        clf = load_classifier(path, threads=th)
        for b in ([1] if path.endswith(".tflite") else batches):  # tflite model has fixed batch 1
            x = np.random.randint(0, 255, (b, clf.size, clf.size, 3), np.uint8)
            for _ in range(10):
                clf(x)
            t = time.perf_counter()
            r = reps if not path.endswith(".tflite") else 30
            for _ in range(r):
                clf(x)
            dt = (time.perf_counter() - t) / r
            print(f"{os.path.basename(path):40s} threads={th} batch={b:3d} "
                  f"{dt * 1e3:7.3f} ms/call  {dt / b * 1e3:7.3f} ms/crop  {b / dt:9.0f} crops/s")


def bench_motion(w, h, n=300):
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    for method in ("cnt", "avg", "mog2", "diff"):
        if method == "cnt" and not hasattr(cv2, "bgsegm"):
            continue
        for scale in (1, 2, 4):
            md = MotionDetector(scale=scale, method=method)
            ts = []
            for i in range(n):
                f = bg.copy()
                x = 50 + (i * 7) % (w - 150)
                cv2.ellipse(f, (x, h // 2), (40, 25), 0, 0, 360, (90, 90, 100), -1)
                f = cv2.add(f, rng.integers(0, 6, f.shape, dtype=np.uint8))
                t = time.perf_counter()
                md.detect(f)
                ts.append(time.perf_counter() - t)
            print(f"motion {method:5s} scale 1/{scale}: {np.median(ts[50:]) * 1e3:.3f} ms/frame @ {w}x{h}")


def compare(paths, data):
    files = []
    for cls, y in (("pigeon", 1), ("not_pigeon", 0)):
        files += [(f, y) for f in glob.glob(os.path.join(data, "**", cls, "*"), recursive=True)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    ys = np.array([y for _, y in files])
    for path in paths:
        clf = load_classifier(path)
        crops = np.stack([cv2.resize(cv2.imread(f), (clf.size, clf.size), interpolation=cv2.INTER_AREA) for f, _ in files])
        p = np.concatenate([clf(crops[i:i + 64]) for i in range(0, len(crops), 64)])
        pred = p >= clf.threshold
        tp, fp = (pred & (ys == 1)).sum(), (pred & (ys == 0)).sum()
        fn = (~pred & (ys == 1)).sum()
        print(f"{os.path.basename(path):40s} thr={clf.threshold:.3f} acc={(pred == ys).mean():.4f} "
              f"precision={tp / max(1, tp + fp):.4f} recall={tp / max(1, tp + fn):.4f}  (n={len(ys)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=[])
    ap.add_argument("--threads", nargs="+", type=int, default=[1])
    ap.add_argument("--batch", nargs="+", type=int, default=[1, 4, 16, 64])
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--compare", nargs="*", default=None)
    ap.add_argument("--data", default="data/labeled")
    a = ap.parse_args()
    for m in a.model:
        bench_clf(m, a.threads, a.batch)
    if a.motion:
        bench_motion(a.width, a.height)
    if a.compare:
        compare(a.compare, a.data)


if __name__ == "__main__":
    main()
