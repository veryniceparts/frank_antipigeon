#!/usr/bin/env python3
"""Synthetic test scene for smoke-testing the pipeline end-to-end without real footage.

Draws flying "pigeons" (blue-grey body, dark head, iridescent neck, flapping wings) and
distractors ("sparrows", "crows", "leaves") over a noisy sky with a swaying tree.
Writes video + ground truth (frame, x0,y0,x1,y1, cls) and optionally a crop dataset in
<crops>/pigeon and <crops>/not_pigeon.  NOT a substitute for real data.

  python scripts/make_synthetic.py --out data/synth/train.mp4 --frames 1500 --crops data/synth_crops --seed 1
  python scripts/make_synthetic.py --out data/synth/test.mp4 --frames 600 --seed 2
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit.motion import square_crops  # noqa: E402

KINDS = {  # body color BGR, size range (half-length px), flap speed
    "pigeon": ((150, 135, 125), (14, 26), 0.5),
    "sparrow": ((60, 95, 130), (6, 10), 1.2),
    "crow": ((25, 25, 28), (16, 26), 0.35),
    "leaf": ((40, 140, 70), (5, 9), 0.0),
}


class Obj:
    def __init__(self, rng, W, H, kind):
        self.kind = kind
        self.r = rng.uniform(*KINDS[kind][1])
        left = rng.random() < 0.5
        self.x = -40.0 if left else W + 40.0
        self.y = rng.uniform(0.15, 0.85) * H
        sp = rng.uniform(3, 9) if kind != "leaf" else rng.uniform(1, 3)
        self.vx = sp if left else -sp
        self.vy = rng.uniform(-1.5, 1.5) if kind != "leaf" else rng.uniform(1, 3)
        self.phase = rng.uniform(0, 6.28)
        self.col = np.array(KINDS[kind][0], float) * rng.uniform(0.85, 1.15)
        self.flap = KINDS[kind][2]

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.phase += self.flap if self.flap else 0.2

    def draw(self, img):
        x, y, r = int(self.x), int(self.y), self.r
        d = 1 if self.vx > 0 else -1
        col = tuple(int(c) for c in self.col)
        if self.kind == "leaf":
            cv2.ellipse(img, (x, y), (int(r), int(r * 0.5)), np.degrees(self.phase) % 360, 0, 360, col, -1)
            return
        cv2.ellipse(img, (x, y), (int(r), int(r * 0.55)), 0, 0, 360, col, -1)  # body
        hx = x + d * int(r * 0.95)
        head = (40, 40, 45) if self.kind == "pigeon" else tuple(int(c * 0.8) for c in col)
        if self.kind == "pigeon":  # iridescent neck
            cv2.circle(img, (x + d * int(r * 0.6), y - int(r * 0.15)), max(2, int(r * 0.33)), (110, 150, 90), -1)
            cv2.ellipse(img, (x - d * int(r * 0.2), y), (int(r * 0.45), int(r * 0.2)), 0, 0, 360, (90, 80, 75), -1)
        cv2.circle(img, (hx, y - int(r * 0.2)), max(2, int(r * 0.3)), head, -1)  # head
        tail = (x - d * int(r * 1.5), y)
        cv2.line(img, (x - d * int(r * 0.8), y), tail, col, max(2, int(r * 0.25)))
        wing = int(r * 1.1 * np.sin(self.phase))  # flapping
        wc = tuple(int(c * 0.85) for c in col)
        pts = np.array([[x - int(r * 0.3), y], [x + int(r * 0.3), y], [x, y - wing]], np.int32)
        cv2.fillPoly(img, [pts], wc)

    def box(self):
        r = self.r
        if self.kind == "leaf":
            return [self.x - r, self.y - r, self.x + r, self.y + r]
        return [self.x - 1.5 * r, self.y - 1.15 * r, self.x + 1.3 * r, self.y + 1.15 * r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth/test.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--crops", default=None, help="also write labelled crops here")
    ap.add_argument("--crop-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    W, H = a.width, a.height
    grad = np.linspace(225, 165, H)[:, None, None] * np.array([1.0, 0.92, 0.78])[None, None]
    sky = np.broadcast_to(grad, (H, W, 3)).astype(np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    gt = open(os.path.splitext(a.out)[0] + "_gt.csv", "w", newline="")
    g = csv.writer(gt)
    g.writerow(["frame", "id", "x0", "y0", "x1", "y1", "cls"])
    if a.crops:
        for c in ("pigeon", "not_pigeon"):
            os.makedirs(os.path.join(a.crops, c), exist_ok=True)
    objs, nid = {}, 0
    tag = os.path.splitext(os.path.basename(a.out))[0]
    for f in range(a.frames):
        if len(objs) < 6 and rng.random() < 0.06:
            kind = rng.choice(list(KINDS), p=[0.4, 0.25, 0.2, 0.15])
            objs[nid] = Obj(rng, W, H, kind)
            nid += 1
        img = sky.copy()
        light = 1 + 0.04 * np.sin(f / 90)  # slow cloud lighting change
        img *= light
        sway = int(6 * np.sin(f / 7))  # swaying tree in the corner
        cv2.ellipse(img, (W - 90 + sway, H - 60), (110, 140), 0, 0, 360, (40, 90, 50), -1)
        img = img.astype(np.uint8)
        clean = None
        for i, o in list(objs.items()):
            o.step()
            if not (-60 < o.x < W + 60 and -60 < o.y < H + 60):
                del objs[i]
                continue
            o.draw(img)
        noisy = cv2.add(img, rng.integers(0, 8, img.shape, dtype=np.uint8))
        for i, o in objs.items():
            b = np.clip(np.array(o.box()), [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
            if b[2] - b[0] < 4 or b[3] - b[1] < 4:
                continue
            g.writerow([f, i, *b, o.kind])
            if a.crops and f % a.crop_every == 0:
                # jitter the box like an imperfect motion detector would
                j = (rng.normal(0, 0.12, 4) * (b[2] - b[0])).astype(int)
                c = square_crops(noisy, np.array([b + j]), 96)[0]
                cls = "pigeon" if o.kind == "pigeon" else "not_pigeon"
                cv2.imwrite(os.path.join(a.crops, cls, f"{tag}_t{i:05d}_f{f:06d}.jpg"), c)
        if a.crops and f % (a.crop_every * 10) == 0:  # pure background negatives
            boxes = [o.box() for o in objs.values()]
            for _ in range(20):  # only keep crops that contain no bird (incl. square_crops padding)
                x, y = rng.integers(0, W - 100), rng.integers(0, H - 100)
                s = int(rng.integers(20, 100))
                m = s * 0.5 + 4
                if all(x - m > b[2] or x + s + m < b[0] or y - m > b[3] or y + s + m < b[1] for b in boxes):
                    c = square_crops(noisy, np.array([[x, y, x + s, y + s]]), 96)[0]
                    cv2.imwrite(os.path.join(a.crops, "not_pigeon", f"{tag}_bg_f{f:06d}.jpg"), c)
                    break
        vw.write(noisy)
    vw.release()
    gt.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
