#!/usr/bin/env python3
"""Collect moving-object crops from your own camera/videos (the best training data you can get).

Uses the exact same motion detector + square_crops as the live pipeline, saves one crop per
track every --every frames.

  python scripts/harvest.py --source clip.mp4 --out data/crops --size 128
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import IoUTracker, MotionDetector, open_camera, square_crops  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, nargs="+")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--size", type=int, default=128, help="saved crop size (train resizes down)")
    ap.add_argument("--every", type=int, default=10, help="save a crop of each track every N frames")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = 0
    run = time.strftime("%Y%m%d-%H%M%S")
    for k, src in enumerate(a.source):
        # unique per run and per source: track ids restart at 0 for every source
        tag = f"{os.path.splitext(os.path.basename(src))[0] or 'cam'}-{run}-{k}"
        md = MotionDetector(scale=a.scale, min_area=a.min_area)
        tr = IoUTracker(n_unsure=a.every, n_sure=a.every)
        f = 0
        with open_camera(src) as cam:
            while (item := cam.read()) is not None:
                frame = item[2]
                todo = tr.update(md.detect(frame), f)
                if todo:
                    boxes = [tr.tracks[i].box for i in todo]
                    for tid, c in zip(todo, square_crops(frame, np.array(boxes), a.size)):
                        # file name groups crops by video+track -> split train/val by track, not by frame
                        cv2.imwrite(os.path.join(a.out, f"{tag}_t{tid:05d}_f{f:06d}.jpg"), c)
                        tr.feed(tid, 0.5, f)
                        total += 1
                f += 1
                if a.max_frames and f >= a.max_frames:
                    break
        print(f"{src}: {f} frames")
    print(f"saved {total} crops to {a.out}")


if __name__ == "__main__":
    main()
