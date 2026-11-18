# frank_antipigeon: pigeonkit

A fast ML kit that decides whether a bird is a pigeon:

```
high-speed camera ─► motion blobs (CNT bg-subtraction @ 1/4 res) ─► IoU tracker
   (latest-frame grabber)      ~0.35 ms / 720p frame                  ~0.1 ms
                                                                         │ only new / unsure tracks
                                                                         ▼
          draw boxes ◄── per-track EMA of P(pigeon) ◄── batched square crops ─► tiny CNN (ONNX, uint8 in)
                                                                         0.07 ms (tiny@64) – 0.4 ms (LCNet@96) per crop
```

Measured on a 4-core 2.1 GHz Xeon, CPU only, one ONNX thread: the whole pipeline runs at
**~1000 fps on 960×540 video, video decoding included** (motion 0.34 ms, tracking 0.11 ms,
classification 0.05 ms per frame on average). Because of the tracker, each object is classified
a few times, not on every frame.

**Target hardware: Jetson Xavier NX + Teledyne DALSA Genie Nano (GigE Vision).** See [docs/DALSA.md](docs/DALSA.md)
for the camera: Aravis, network tuning, ROI and frame-rate table, raw-Bayer path. See [docs/JETSON.md](docs/JETSON.md) for the TensorRT engine, CSI/USB/GigE cameras,
power mode and setup. In short: train on a PC, copy `pigeon.f32.onnx` to the Jetson, run `scripts/build_trt.py` there,
then `run.py --model models/pigeon.engine`.

## Minimal examples: pigeons and doves only (no training needed)

```bash
pip install -r requirements.txt     # includes yt-dlp

# YouTube link -> downloaded, then replayed at its real speed with live detection
python examples/youtube_detect.py "https://www.youtube.com/watch?v=VIDEO_ID"
python examples/youtube_detect.py "https://www.youtube.com/watch?v=VIDEO_ID" --stream   # play while streaming
python examples/youtube_detect.py clip.mp4 --out runs/clip_detected.mp4 --no-show      # local file, save result

# USB camera (fixed camera -> motion detection; --find ssd also catches perched birds)
python examples/usb_camera_detect.py --device 0 --width 1280 --height 720 --fps 60
```

- **Only pigeons and doves get a box.** The decision comes from the iNaturalist bird classifier.
  "Pigeon/dove" is the sum of its **21 Columbidae species**: rock/feral, wood, collared, turtle, laughing,
  spotted, mourning and others. All other 943 bird species, and anything that isn't a bird, are negative.
  Use `--target pigeon` for pigeons only. `--debug` also draws the rejected candidates in grey.
- **Finding candidates.** The YouTube example defaults to `--find ssd`, a COCO SSD "bird" detector. It works when
  the camera pans and when birds sit still, as is common in YouTube clips. A fixed camera can use `--find motion`,
  which is about 20× cheaper.
- **Real-time replay.** Video plays at its native fps like a live camera. If the detector is slower, frames are dropped
  rather than letting playback lag.
- **Measured on real photos** (7 pigeons; 7 parrots, another bird, a cat and a person, moving across a scene):
  all 7 pigeons were boxed in 66–100% of their frames, labelled "Rock Pigeon", and none of the 10 others was ever boxed.
  This was with both `ssd` and `motion`. Dove photos were not available in the build environment, so dove recall is unmeasured.
- **Cost per frame (x86, 4 cores):** SSD about 20 ms, classifier about 1–4 ms. On a Xavier NX CPU, expect several times
  slower. For a fixed USB camera, use motion detection, and later the trained ONNX/TensorRT student (`--model`).

## Quick start

```bash
pip install -r requirements.txt

# 0) No training: use Google's iNaturalist-birds MobileNetV2 (964 species, Apache-2.0) as the classifier.
python -c "import pigeonkit; pigeonkit.download_teacher('models')"
python scripts/run.py --source 0 --model models/mobilenet_v2_1.0_224_inat_bird_quant.tflite --show
#    P(pigeon) = Rock Pigeon + Feral Pigeon + Wood-Pigeon. About 5–8 ms per crop on CPU.
#    On 7 real pigeon photos it scored 0.79–0.96; parrots scored about 0.

# 1) Collect crops from YOUR camera with the same motion detector and crop function as the live pipeline
python scripts/harvest.py --source clip1.mp4 clip2.mp4 --out data/crops

# 2) Auto-label the crops with the teacher, then fix mistakes by moving files between folders
python scripts/autolabel.py --crops data/crops --out data/labeled --copy

# 3) Optional: add public CC-licensed pigeons and look-alike birds as hard negatives
python scripts/fetch_inat.py --out data/inat

# 4) Train the fast student (distilled with the teacher's soft labels) and export ONNX
python scripts/train.py --data data/labeled data/inat --soft --out models/pigeon.onnx
#    fastest option:  --arch tiny --size 64   (80k params, 0.07 ms/crop)

# 5) Run it
python scripts/run.py --source pylon:// --model models/pigeon.onnx --scale 2 --fps 500 --show
python scripts/run.py --source clip.mp4 --model models/pigeon.onnx --out runs/annotated.mp4
```

Other tools:
- `scripts/bench.py`: latency per model, thread count and batch size; motion-stage timings; `--compare` checks fp32 vs int8 accuracy.
- `scripts/quantize.py`: INT8 static QDQ quantization calibrated on real crops. Benchmark before using it: on CPUs without VNNI, int8 was *slower* for the tiny net (0.21 vs 0.07 ms).
- `scripts/build_trt.py`: builds the TensorRT engine on the Jetson from `*.f32.onnx`, for the GPU or `--dla 0`.
- `scripts/jetson_setup.sh`: one-time Xavier NX setup.
- `scripts/gige_tune.sh`: GigE host tuning for the DALSA camera (MTU 9000, rmem, cap_net_raw).
- `scripts/evaluate.py`: precision/recall of the whole pipeline against a ground-truth csv.
- `scripts/make_synthetic.py`: synthetic scene of pigeons, crows, sparrows, leaves and a swaying tree, for smoke tests. Not real data.

The trigger hook for a deterrent is marked in `scripts/run.py` (`>>> hook your deterrent here <<<`).
`Detection.pigeon` is set only when a track has ≥3 hits **and** its smoothed probability is above
the model's threshold. `train.py` chooses that threshold for ≥98% validation precision, with a floor of 0.5.

## Cameras

`pigeonkit.open_camera(uri)` returns a grabber that always delivers the newest frame and drops stale ones:

| URI | backend |
|---|---|
| `0`, `1` … | OpenCV V4L2/UVC, MJPG, buffer size 1, threaded |
| `file.mp4` | every frame, no drops (offline) |
| `rtsp://…` | OpenCV, newest frame only |
| `csi://0?w=1280&h=720&fps=120` | Jetson CSI camera through PyGObject GStreamer; uses BGRx frames directly (no CPU colour conversion) |
| `gst://<pipeline>` | any GStreamer pipeline that outputs BGR, BGRx or GRAY8, e.g. USB MJPG with Jetson hardware decoding |
| `dalsa://<id or IP>?pf=BayerRG8&w=1280&h=480&exp=800&fps=300` | **Teledyne DALSA Genie Nano** / any GigE Vision camera through Aravis. Bayer frames stay raw: motion runs on the mosaic and only the crops are debayered |
| `pylon://` | Basler pypylon, `GrabStrategy_LatestImageOnly`, Mono8/Bayer8, ROI |
| `genicam://<path.cti>` | any GenICam camera through harvesters |

High-speed tips:
- Shrink the sensor ROI on the camera itself. This gives the biggest fps gain.
- Request Mono8 or Bayer8, never RGB8.
- Lock exposure. Auto-exposure silently caps the frame rate.
- For GigE, enable jumbo frames.
- For USB3, raise `usbfs_memory_mb`.

## Models (CPU, 1 thread, measured here)

| model | input | ms/crop (batch 1 / 16) | notes |
|---|---|---|---|
| `tiny` (TinyPigeonNet, 80k params) | 64 | 0.083 / 0.069 | fastest; needs in-domain data or distillation |
| `lcnet_050.ra2_in1k` (default) | 96 | 0.43 / 0.39 | ImageNet-pretrained, best for small datasets |
| `mobilenetv3_small_050.lamb_in1k` | 96 | ~0.35 | alternative |
| `mobilenetv4_conv_small_050.e3000_r224_in1k` | 96 | ~0.41 | if accuracy plateaus |
| same student as a TensorRT FP16 `.engine` | 64/96 | measure on the device | Jetson GPU or DLA; `scripts/build_trt.py` |
| iNat MobileNetV2 TFLite (teacher) | 224 | 5–8 | zero-training; ~1 ms on a Coral EdgeTPU (`*_edgetpu.tflite`) |

The ONNX student takes **raw uint8 BGR crops (N,S,S,3)**. BGR→RGB conversion and normalisation are
baked into the graph. The model's threshold and metadata are stored in the ONNX file.

## Design decisions (from the GitHub research)

11 research agents surveyed GitHub and the web for each stage. Key findings:

1. **Two stages, motion then classifier, not a single YOLO.** A detector at 320 px shrinks a 30 px
   pigeon to about 5 px. Cropping around motion keeps full resolution, as Frigate NVR does.
   Pigeon-specific prior work:
   - [adamp87/pigeon](https://github.com/adamp87/pigeon): bird detector, then a crop goes to the iNat MobileNetV2.
   - [LH-Tech-AI/dove-detector](https://github.com/LH-Tech-AI/dove-detector): YOLO, then CLIP.
   - [jpvoelz/pigeon-detector](https://github.com/jpvoelz/pigeon-detector): YOLOv7 on 244 images.
   - [tarraschk/GardenGuardian](https://github.com/tarraschk/GardenGuardian): Pi 5 + Hailo, a water gun, COCO "bird" only.
2. **Motion: CNT background subtraction** (opencv-contrib `cv2.bgsegm`) on a gray frame at 1/4
   scale. It took 0.32 ms per 720p frame, against 0.9 ms for MOG2 and 1.0 ms for KNN. Gray → `INTER_LINEAR`
   resize, because `INTER_AREA` is about 9x slower. `findContours` is about 10x faster than connectedComponents.
   Sources: [sagi-z/BackgroundSubtractorCNT](https://github.com/sagi-z/BackgroundSubtractorCNT),
   [andrewssobral/bgslibrary](https://github.com/andrewssobral/bgslibrary).
3. **Tracking: greedy IoU plus a centroid fallback**, no Kalman filter. At high fps objects move only a few pixels per frame.
   Classify a track when it is new, then every 5 frames until it is confident, then every 60 frames. Average the
   probabilities over the track with an EMA (as [bird-feeder-ai](https://github.com/christianloth/bird-feeder-ai) does).
   Gap-merging and dilation glue flapping-wing fragments together.
4. **Backbone:** benchmarked about 16 timm models exported to ONNX. GhostNet, RepViT, FastViT and
   EfficientNet-lite all take more than 1 ms at 96 px. LCNet-0.5 or MobileNetV3-S-0.5 at 96 px give the best accuracy per ms.
   Use 1 ONNX thread and batch the crops. Extra threads do not help at this size.
5. **Labels without hand-labelling:** the Coral iNat-birds TFLite model (indices 307 and 958 for Columba livia, 310 for Wood-Pigeon)
   is the teacher. Its soft labels are distilled into the student. For an even stronger offline teacher, use
   [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) (tree-of-life CLIP).
6. **Data:**
   - Crops from your own camera matter most. Validation is split by track, not by frame.
   - Public sources:
     - iNaturalist: CC0/CC-BY photos via the API, Columba livia plus 17 look-alike taxa as hard negatives.
     - Open Images V7: "Rock dove" `/m/01c08f` and "Pigeons and doves" `/m/0h0rv`, using the Bird boxes in those images.
     - Roboflow Universe pigeon sets (`wiings/pigeon-mg46t`, `robotica-kstcd/pigeon_chicken_detection`).
7. **Known hard cases:** collared and turtle doves look like pigeons, and so do grey pigeons against grey roofs.
   Close-up humans have been called "pigeon" in other projects. Add these as negatives. Motion-only detection misses perched birds.
8. **Edge hardware:**
   - Raspberry Pi 5 + Hailo-8L: the student runs at thousands of crops per second.
   - Jetson Orin Nano (TensorRT FP16): for cameras above about 120 fps.
   - Coral and OAK-D are not recommended for new builds.

## Verified here

- `pytest tests` passes 8 tests: motion (CNT, MOG2, diff), box merging, crops, tracker, end-to-end pipeline, ONNX round-trip.
- Synthetic end-to-end run: train the tiny model (15 s on CPU), export ONNX, then run the pipeline on a held-out video.
  All 6/6 pigeons were detected. 1 of 10 distractors (a crow) was flagged in some frames.
  Frame-level precision was 0.87 and recall 0.75; recall loses the first frames of each track because of the 3-hit warm-up.
  See [REVIEW.md](REVIEW.md) for the full pipeline review, the fixed findings and the open risks.
- The iNat teacher was checked on real pigeon and parrot photos.
- **Not verified here:** real camera SDKs (pypylon, harvesters), `fetch_inat.py` (the network was blocked in the build
  environment), and pretrained timm weights (Hugging Face was blocked, so timm was only trained from scratch).
