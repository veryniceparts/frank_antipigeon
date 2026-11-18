# Pipeline review

This review covers the whole chain: capture → motion → tracking → crop → classifier → decision, plus the
data/training tooling that produces the classifier. It combines a high-effort code review of
`pigeonkit/` and `scripts/` with end-to-end runs on the synthetic scene. Status: **all 10 code findings
fixed** (see table). The open design risks are listed at the end.

## End-to-end numbers (after fixes)

Setup: 4-core Xeon 2.1 GHz, CPU only, 960×540 synthetic test video (`make_synthetic.py --seed 2`, 600 frames),
tiny CNN at 64 px trained on `--seed 1` crops.

| stage | ms / frame (avg) |
|---|---|
| motion (CNT, 1/4 scale) | 0.364 |
| tracking | 0.122 |
| crop | 0.022 |
| classify (batched, only scheduled tracks) | 0.046 |
| **total** | **≈0.55 → ~1000+ fps** |

| metric (corrected evaluator) | value |
|---|---|
| frame-level pigeon-box precision / recall | 0.868 / 0.753 |
| pigeons detected (object level) | 6 / 6 |
| non-pigeons ever flagged | 1 / 10 (a crow) |
| flags on blobs with no GT match | 58 of 94 FP |

Earlier README numbers (precision 0.92, "1/8 distractors") came from the old evaluator, which
double-counted fragments and dropped objects the motion stage never found. These numbers are the honest ones.

## Code findings

| # | file | issue | impact | fix |
|---|---|---|---|---|
| 1 | `fetch_inat.py` | Wood pigeon fetched as a *negative*, but the teacher counts it as pigeon (idx 310) | contradictory training targets | moved `Columba palumbus` to positives |
| 2 | `harvest.py` | crop names `{video}_t{id}_f{frame}` restart every run/source | re-runs overwrite crops; unrelated tracks merged in autolabel averaging and train/val split | tag = `{video}-{timestamp}-{source#}` |
| 3 | `autolabel.py` | `--copy` re-run puts the same crop in two class folders | same image trained as 1.0 and 0.0 | an already-sorted crop is never re-copied, so the manual review is kept |
| 4 | `train.py` | `--soft`: crops deleted during review came back via the original path in `labels.csv` | removed mistakes are still trained on | when review folders exist they decide: deleted stays deleted, moved = hard label |
| 5 | `evaluate.py` | tp counted per detection, so wing fragments counted twice | inflated precision/recall | tp/fn counted once per GT box |
| 6 | `evaluate.py` | objects never segmented were excluded from the denominator | object recall overstated | denominator = objects present in ≥3 GT frames |
| 7 | `fetch_inat.py` | `urlretrieve` had no timeout and left partial files counted as done | hangs; truncated JPEG → black "pigeon" image | 30 s timeout, `.part` file + atomic rename |
| 8 | `fetch_inat.py` | license filter is per observation, but `photos[0]` may be all-rights-reserved | unlicensed images in the dataset | pick the first photo whose own license is CC0/CC-BY/CC-BY-SA |
| 9 | `make_synthetic.py` | "background" negatives cut at random could contain a pigeon | label noise | reject windows overlapping any bird (incl. crop padding) |
| 10 | `train.py` | image cache duplicated in every DataLoader worker; val cache rebuilt every epoch | ~8× RAM, OOM on big sets | cache only with `--workers 0`; persistent val workers |

Verified after the fixes: `pytest` 8/8 passing. Harvesting the same video twice gives 2 tag groups and no overwrites.
Re-running autolabel gives no duplicate copies. The soft-label collect handles a deleted crop, a moved crop and an
unsure crop correctly. `--workers 0` training works.

## Design review of the combined pipeline

What holds up:
- **Two-stage design** keeps full-resolution crops for small birds and spends compute only on moving
  objects. Classification is about 8% of per-frame time, so the classifier is not the bottleneck.
- **Classify-by-track scheduling**: 165 classifier calls for 600 frames. A heavier model (LCNet@96,
  0.4 ms) would still fit within a 1 kHz budget.
- **One batched ORT call per frame, with 1 thread.** Preprocessing is baked into the ONNX graph, so there is no
  Python-side normalisation and no train/deploy mismatch. The same `square_crops` function is used in
  harvest, training data and inference.

Open risks (not fixed; ranked):
1. **Merged objects.** Dilation plus gap-merging glues birds that overlap or fly close together into one box
   (seen in the annotated frame: a pigeon and a crow in one box). The crop is then judged as a whole, which
   explains most of the 58 "unmatched" false flags. Options: lower `merge_gap`/dilation at small `--scale`,
   or split boxes whose aspect ratio is far from a bird's.
2. **Early "confident" lock-in.** A track is marked confident after 2 calls and then re-checked only every 60
   frames. A pigeon that enters partly visible (edge of frame, motion-blurred) can be locked as "not pigeon"
   for 60 frames. Consider requiring a minimum crop size before a track can become confident, or lowering `n_sure`.
3. **Detection latency.** `min_hits=3` plus EMA delays the first flag by ≥3 frames. That is negligible at 500 fps,
   but at 30 fps it is 100 ms, and it is the main cause of recall 0.75 at frame level.
4. **Motion-only blind spots.** Perched or walking pigeons fade into CNT's background after
   `maxPixelStability` frames. Camera shake gives full-frame blobs, which `max_area_frac` drops, so nothing is
   detected during shake. If perched birds matter, add a slow (1–2 fps) full-frame detector pass.
5. **Lighting steps.** Clouds or auto-exposure produce a burst of blobs until CNT re-adapts. Lock exposure on
   the camera. An `--ignore-mask` handles trees and flags.
6. **Teacher domain gap.** The iNat model was trained on photos, not on blurry motion crops. On cartoon synthetic
   birds it labelled everything not-pigeon. On real crops, expect it to be weaker on small or blurry birds and on
   doves. That is why the review step in autolabel, and per-track averaging, matter.
7. **INT8 is not a guaranteed win.** It was 3× slower for the tiny net on this CPU. Keep the fp32 model unless
   `bench.py` shows a gain on the target hardware.
8. **Untested here:** real camera SDKs (pypylon/harvesters), `fetch_inat.py` against the live API, and timm
   pretrained weights (Hugging Face was blocked in the build environment).
