#!/usr/bin/env python3
"""Evaluate the full pipeline on a video with ground truth csv (frame,id,x0,y0,x1,y1,cls).

Reports frame-level box precision/recall for the PIGEON label and object-level results
(a pigeon object counts as detected if flagged in >=1 frame; a non-pigeon object is a false alarm
if it is ever flagged).

  python scripts/evaluate.py --source data/synth/test.mp4 --model models/pigeon.onnx
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import IoUTracker, MotionDetector, PigeonPipeline, load_classifier, open_camera  # noqa: E402
from pigeonkit.tracker import iou_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--iou", type=float, default=0.3)
    a = ap.parse_args()
    gt_path = a.gt or os.path.splitext(a.source)[0] + "_gt.csv"
    gt = collections.defaultdict(list)
    with open(gt_path) as fh:
        for r in csv.DictReader(fh):
            gt[int(r["frame"])].append((int(r["id"]), [int(r[k]) for k in ("x0", "y0", "x1", "y1")], r["cls"]))
    clf = load_classifier(a.model, threshold=a.threshold)
    pipe = PigeonPipeline(clf, MotionDetector(scale=a.scale), IoUTracker(threshold=clf.threshold))
    tp = fp = fn = 0
    obj_cls, obj_flag, obj_gt = {}, collections.Counter(), collections.Counter()
    unmatched_flags = 0
    with open_camera(a.source) as cam:
        f = 0
        while (item := cam.read()) is not None:
            dets = pipe.process(item[2])
            g = gt.get(f, [])
            for gid, _, cls in g:
                obj_cls[gid] = cls
                obj_gt[gid] += 1
            db = np.array([d.box for d in dets]) if dets else np.zeros((0, 4))
            gb = np.array([b for _, b, _ in g]) if g else np.zeros((0, 4))
            M = iou_matrix(db, gb) if len(db) and len(gb) else np.zeros((len(db), len(gb)))
            flagged_gt = set()
            for i, d in enumerate(dets):
                if not d.pigeon:
                    continue
                j = int(M[i].argmax()) if len(gb) else -1
                if j >= 0 and M[i, j] >= a.iou:
                    flagged_gt.add(j)
                    if g[j][2] != "pigeon":
                        fp += 1  # flagged a crow/sparrow/leaf
                    # extra fragments on an already-counted pigeon are neither tp nor fp
                else:
                    fp += 1
                    unmatched_flags += 1
            for j, (gid, _, cls) in enumerate(g):  # tp/fn once per ground-truth box
                if j in flagged_gt:
                    obj_flag[gid] += 1
                if cls == "pigeon":
                    tp += j in flagged_gt
                    fn += j not in flagged_gt
            f += 1
    print(f"frames={f}  per-frame classify+motion: {pipe.timer.report()}")
    print(f"frame-level PIGEON boxes: precision={tp / max(1, tp + fp):.3f} recall={tp / max(1, tp + fn):.3f} "
          f"(tp={tp} fp={fp} fn={fn}, flags on unmatched blobs={unmatched_flags})")
    # every object present >=3 GT frames counts, including ones motion detection never found
    pig = [g for g, c in obj_cls.items() if c == "pigeon" and obj_gt[g] >= 3]
    oth = [g for g, c in obj_cls.items() if c != "pigeon" and obj_gt[g] >= 3]
    print(f"objects: pigeons detected {sum(obj_flag[g] > 0 for g in pig)}/{len(pig)}; "
          f"non-pigeons ever flagged {sum(obj_flag[g] > 0 for g in oth)}/{len(oth)} "
          f"({collections.Counter(obj_cls[g] for g in oth if obj_flag[g] > 0)})")


if __name__ == "__main__":
    main()
