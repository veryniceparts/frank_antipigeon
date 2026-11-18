#!/usr/bin/env python3
"""Pseudo-label harvested crops with the iNat-birds teacher -> data for the fast student.

Writes labels.csv (path, p_teacher, label) and optionally copies crops into
<out>/pigeon, <out>/not_pigeon, <out>/unsure for a quick manual check.
Per-track averaging (crops of the same track share the _tNNNNN tag) makes labels more reliable.

  python scripts/autolabel.py --crops data/crops --out data/labeled --copy
"""
import argparse
import collections
import csv
import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import INatBirdClassifier, download_teacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--out", default="data/labeled")
    ap.add_argument("--teacher", default=None, help="iNat tflite (downloaded if omitted)")
    ap.add_argument("--hi", type=float, default=0.6, help="p >= hi -> pigeon")
    ap.add_argument("--lo", type=float, default=0.15, help="p <= lo -> not_pigeon")
    ap.add_argument("--track-mix", type=float, default=0.5, help="weight of per-track mean in final p")
    ap.add_argument("--copy", action="store_true", help="copy files into class folders for review")
    a = ap.parse_args()
    teacher = INatBirdClassifier(a.teacher or download_teacher("models"), threads=4)
    files = sorted(glob.glob(os.path.join(a.crops, "**", "*.jpg"), recursive=True) +
                   glob.glob(os.path.join(a.crops, "**", "*.png"), recursive=True))
    p = {}
    for i, f in enumerate(files):
        p[f] = float(teacher(cv2.imread(f)[None])[0])
        if i % 500 == 0:
            print(f"{i}/{len(files)}")
    tracks = collections.defaultdict(list)
    for f in files:
        m = re.search(r"^(.*_t\d+)_f\d+", os.path.basename(f))
        if m:
            tracks[m.group(1)].append(f)
    for fs in tracks.values():
        mean = float(np.mean([p[f] for f in fs]))
        for f in fs:
            p[f] = (1 - a.track_mix) * p[f] + a.track_mix * mean
    os.makedirs(a.out, exist_ok=True)
    counts = collections.Counter()
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "p_teacher", "label"])
        for f in files:
            lab = "pigeon" if p[f] >= a.hi else "not_pigeon" if p[f] <= a.lo else "unsure"
            counts[lab] += 1
            w.writerow([os.path.abspath(f), f"{p[f]:.4f}", lab])
            if a.copy:
                d = os.path.join(a.out, lab)
                os.makedirs(d, exist_ok=True)
                shutil.copy2(f, d)
    print(dict(counts), "->", os.path.join(a.out, "labels.csv"))
    print("Review the folders: move mistakes, then train with --data", a.out)


if __name__ == "__main__":
    main()
