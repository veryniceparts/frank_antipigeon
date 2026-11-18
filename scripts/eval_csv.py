#!/usr/bin/env python3
"""Score a detections CSV (from the C++ app: pigeon_detect --csv) against a ground-truth CSV.

  ./cpp/build/pigeon_detect --source data/synth/test.mp4 --fast --model runs/synth/tiny64.f32.onnx --csv runs/c.csv
  python scripts/eval_csv.py runs/c.csv data/synth/test_gt.csv

GT rows: frame,id,x0,y0,x1,y1,cls  (cls 'pigeon' or 'pigeon:<file>' = positive, anything else negative).
Same metrics as scripts/evaluate.py: frame-level box precision/recall + object level.
"""
import collections
import csv
import sys

import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main(det_path, gt_path, thr=0.3):
    det = collections.defaultdict(list)
    for r in csv.DictReader(open(det_path)):
        if r["pigeon"] == "1":
            det[int(r["frame"])].append([int(r[k]) for k in ("x0", "y0", "x1", "y1")])
    gt = collections.defaultdict(list)
    for r in csv.DictReader(open(gt_path)):
        gt[int(r["frame"])].append((r["id"], [int(r[k]) for k in ("x0", "y0", "x1", "y1")], r["cls"]))
    tp = fp = fn = 0
    obj_cls, obj_flag, obj_n = {}, collections.Counter(), collections.Counter()
    for f in sorted(set(det) | set(gt)):
        g = gt.get(f, [])
        hit = set()
        for d in det.get(f, []):
            scores = [iou(d, b) for _, b, _ in g]
            j = int(np.argmax(scores)) if scores else -1
            if j >= 0 and scores[j] >= thr:
                hit.add(j)
                fp += not g[j][2].startswith("pigeon")
            else:
                fp += 1
        for j, (gid, _, cls) in enumerate(g):
            obj_cls[gid] = cls
            obj_n[gid] += 1
            obj_flag[gid] += j in hit
            if cls.startswith("pigeon"):
                tp += j in hit
                fn += j not in hit
    print(f"frame-level PIGEON boxes: precision={tp / max(1, tp + fp):.3f} recall={tp / max(1, tp + fn):.3f} "
          f"(tp={tp} fp={fp} fn={fn})")
    pig = [g for g, c in obj_cls.items() if c.startswith("pigeon") and obj_n[g] >= 3]
    oth = [g for g, c in obj_cls.items() if not c.startswith("pigeon") and obj_n[g] >= 3]
    print(f"objects: pigeons detected {sum(obj_flag[g] > 0 for g in pig)}/{len(pig)}; non-pigeons ever flagged "
          f"{sum(obj_flag[g] > 0 for g in oth)}/{len(oth)} {sorted(obj_cls[g] for g in oth if obj_flag[g] > 0)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
