#!/usr/bin/env python3
"""Train the fast pigeon/not-pigeon student and export it to ONNX (uint8 BGR in, P(pigeon) out).

Data: one or more folders laid out as  <dir>/pigeon/*.jpg  <dir>/not_pigeon/*.jpg
(any depth below; 'unsure' folders are ignored). With --soft, labels.csv from autolabel.py
supplies the teacher's probabilities as soft targets (knowledge distillation).

  # default: ImageNet-pretrained LCNet-0.5 @96px  (~0.36 ms/crop, 0.13 ms batched, 1 CPU thread)
  python scripts/train.py --data data/labeled --out models/pigeon.onnx
  # fastest: 80k-param CNN @64px (~0.06 ms/crop), best when distilled from many in-domain crops
  python scripts/train.py --data data/labeled --soft --arch tiny --size 64 --epochs 40
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit.models import Deploy, build_model, export_onnx  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect(dirs, soft):
    items = {}  # path -> target in [0,1]
    for d in dirs:
        for cls, y in (("pigeon", 1.0), ("not_pigeon", 0.0)):
            for f in glob.glob(os.path.join(d, "**", cls, "**", "*"), recursive=True):
                if f.lower().endswith(IMG_EXT):
                    items[os.path.abspath(f)] = y
        lab = os.path.join(d, "labels.csv")
        if soft and os.path.exists(lab):
            has_folders = any(os.path.isdir(os.path.join(d, c)) for c in ("pigeon", "not_pigeon"))
            with open(lab) as fh:
                for r in csv.DictReader(fh):
                    if r["label"] == "unsure":
                        continue
                    pt, name = float(r["p_teacher"]), os.path.basename(r["path"])
                    if not has_folders:  # no review copy: train on the teacher's labels directly
                        if os.path.exists(r["path"]):
                            items[os.path.abspath(r["path"])] = pt
                        continue
                    # review folders are authoritative: deleted crops stay deleted, and a crop
                    # the user moved keeps its hard folder label; otherwise use the soft target
                    for cls, y in (("pigeon", 1.0), ("not_pigeon", 0.0)):
                        p = os.path.abspath(os.path.join(d, cls, name))
                        if p in items and (y == 1.0) == (r["label"] == "pigeon"):
                            items[p] = pt
    return sorted(items.items())


def group_of(path):
    m = re.search(r"^(.*_t\d+)_f\d+", os.path.basename(path))  # same video+track -> same split
    return m.group(1) if m else os.path.basename(path)


def motion_blur(img, k):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


class Crops(Dataset):
    def __init__(self, items, size, train, cache=False):
        # cache only when loading in the main process: DataLoader workers would each hold a copy
        self.items, self.size, self.train = items, size, train
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.items)

    def load(self, p):
        im = self.cache.get(p) if self.cache is not None else None
        if im is None:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                im = np.zeros((self.size, self.size, 3), np.uint8)
            s = self.size * 2  # keep some headroom for random crops
            if max(im.shape[:2]) > s:
                f = s / max(im.shape[:2])
                im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            if self.cache is not None:
                self.cache[p] = im
        return im

    def __getitem__(self, i):
        p, y = self.items[i]
        im = self.load(p)
        S = self.size
        if self.train:
            h, w = im.shape[:2]
            sc = random.uniform(0.6, 1.0)
            ar = random.uniform(0.8, 1.25)
            cw = int(min(w, max(4, np.sqrt(sc * w * h * ar))))
            ch = int(min(h, max(4, np.sqrt(sc * w * h / ar))))
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            im = im[y0:y0 + ch, x0:x0 + cw]
            if random.random() < 0.3:  # simulate tiny far-away birds
                r = random.uniform(0.25, 0.7)
                im = cv2.resize(im, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
            im = cv2.resize(im, (S, S), interpolation=cv2.INTER_LINEAR)
            if random.random() < 0.5:
                im = im[:, ::-1]  # horizontal flip only: birds have an up direction
            im = im.astype(np.float32)
            im = im * random.uniform(0.7, 1.3) + random.uniform(-25, 25)  # brightness/contrast
            if random.random() < 0.3:
                g = im.mean(2, keepdims=True)
                im = g + (im - g) * random.uniform(0.5, 1.4)  # saturation
            im = np.clip(im, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                im = motion_blur(im, random.choice((3, 5, 7)))
            elif random.random() < 0.2:
                im = cv2.GaussianBlur(im, (3, 3), 0)
            if random.random() < 0.3:
                q = random.randint(30, 90)
                im = cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
            if random.random() < 0.2:
                im = np.clip(im + np.random.normal(0, 6, im.shape), 0, 255).astype(np.uint8)
            if random.random() < 0.1:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # mono cameras
        else:
            im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA if max(im.shape[:2]) > S else cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(im)), torch.tensor(y, dtype=torch.float32)


def pick_threshold(p, y, min_precision):
    order = np.argsort(-p)
    tp = np.cumsum(y[order] == 1)
    fp = np.cumsum(y[order] == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(1, (y == 1).sum())
    ok = np.where(prec >= min_precision)[0]
    if len(ok) == 0:
        return 0.5, None
    k = ok[-1]  # lowest threshold that still meets the precision target -> max recall
    return float(p[order][k]), (float(prec[k]), float(rec[k]))


def evaluate(model, dl, dev):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(model(x.to(dev)).cpu())
            ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--soft", action="store_true", help="use teacher probabilities from labels.csv")
    ap.add_argument("--arch", default="lcnet_050.ra2_in1k",
                    help="'tiny' or any timm model, e.g. mobilenetv3_small_050.lamb_in1k, "
                         "mobilenetv4_conv_small_050.e3000_r224_in1k")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-precision", type=float, default=0.98, help="pick threshold for this val precision")
    ap.add_argument("--min-threshold", type=float, default=0.5, help="never go below this threshold")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default="models/pigeon.onnx")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    items = collect(a.data, a.soft)
    if not items:
        sys.exit("no images found: expected <data>/pigeon/*.jpg and <data>/not_pigeon/*.jpg")
    groups = sorted({group_of(p) for p, _ in items})
    random.shuffle(groups)
    val_g = set(groups[: max(1, int(len(groups) * a.val_frac))])
    tr = [it for it in items if group_of(it[0]) not in val_g]
    va = [it for it in items if group_of(it[0]) in val_g]
    npos = sum(y >= 0.5 for _, y in tr)
    print(f"train {len(tr)} ({npos} pigeon)  val {len(va)} ({sum(y >= 0.5 for _, y in va)} pigeon)")

    w = np.array([1.0 / max(1, npos) if y >= 0.5 else 1.0 / max(1, len(tr) - npos) for _, y in tr])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(Crops(tr, a.size, True, cache=a.workers == 0), a.bs, sampler=sampler, num_workers=a.workers,
                       drop_last=len(tr) > a.bs, persistent_workers=a.workers > 0)
    dl_va = DataLoader(Crops(va, a.size, False, cache=a.workers == 0), 256, num_workers=a.workers,
                       persistent_workers=a.workers > 0)

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net, mean, std = build_model(a.arch, pretrained=not a.no_pretrained)
    model = Deploy(net, mean, std).to(dev)
    head = set(map(id, (net.get_classifier() if hasattr(net, "get_classifier") else net.head).parameters()))
    params = [{"params": [p for p in net.parameters() if id(p) in head], "lr": a.lr},
              {"params": [p for p in net.parameters() if id(p) not in head],
               "lr": a.lr if a.arch == "tiny" or a.no_pretrained else a.lr * 0.3}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    steps = a.epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=steps, pct_start=0.1)
    bce = nn.BCELoss()
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            y = y * 0.95 + 0.025  # label smoothing
            loss = bce(model(x).clamp(1e-6, 1 - 1e-6), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x); n += len(x)
        p, y = evaluate(model, dl_va, dev)
        yb = (y >= 0.5).astype(int)
        acc = ((p >= 0.5) == yb).mean()
        # rank metric robust to imbalance: balanced accuracy at 0.5
        bacc = np.mean([((p >= 0.5) == yb)[yb == c].mean() for c in (0, 1) if (yb == c).any()])
        print(f"ep {ep + 1:02d} loss {tot / max(1, n):.4f} val acc {acc:.3f} bal_acc {bacc:.3f} ({time.time() - t0:.1f}s)")
        if bacc >= best:
            best, best_state = bacc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    p, y = evaluate(model, dl_va, "cpu")
    thr, pr = pick_threshold(p, (y >= 0.5).astype(int), a.min_precision)
    if thr < a.min_threshold:
        thr = a.min_threshold
        yb, pb = (y >= 0.5), (p >= thr)
        pr = (float((pb & yb).sum() / max(1, pb.sum())), float((pb & yb).sum() / max(1, yb.sum())))
    print(f"best val bal_acc {best:.3f}; threshold {thr:.3f} -> precision/recall {pr}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    meta = {"arch": a.arch, "size": a.size, "threshold": thr, "val_bal_acc": float(best),
            "val_precision_recall": pr, "n_train": len(tr), "n_val": len(va)}
    export_onnx(model.net, mean, std, a.size, a.out, meta)
    torch.save({"state_dict": model.net.state_dict(), "meta": meta}, os.path.splitext(a.out)[0] + ".pt")

    # parity + latency check with ONNX Runtime
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
    xb = np.random.randint(0, 255, (8, a.size, a.size, 3), np.uint8)
    diff = np.abs(s.run(None, {"crops": xb})[0] - model(torch.from_numpy(xb)).detach().numpy()).max()
    for bsz in (1, 16):
        xb = np.random.randint(0, 255, (bsz, a.size, a.size, 3), np.uint8)
        for _ in range(20):
            s.run(None, {"crops": xb})
        t = time.perf_counter()
        for _ in range(200):
            s.run(None, {"crops": xb})
        print(f"ONNX batch {bsz:2d}: {(time.perf_counter() - t) / 200 / bsz * 1e3:.3f} ms/crop (1 thread)")
    print(f"exported {a.out}  (max |onnx-torch| = {diff:.2e})  meta={json.dumps(meta)}")


if __name__ == "__main__":
    main()
