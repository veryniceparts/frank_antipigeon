# pigeon_detect: C++ runtime

This is the real-time path: camera / video / YouTube → candidates → tracker → crop classifier → pigeon & dove boxes.
Training stays in Python (`scripts/train.py`). The C++ app reads the exported models.

## Build

Dependencies:
- **OpenCV ≥ 4.5** with the `dnn` and `videoio` modules. With `opencv_contrib`, the CNT background subtractor is available;
  without it, MOG2 is used.
- **TensorRT**: optional, found automatically on JetPack.
- **yt-dlp**: needed only for YouTube links.

```bash
sudo apt install cmake g++ libopencv-dev            # Ubuntu (libopencv-contrib-dev adds CNT); JetPack already has OpenCV + TensorRT
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j$(nproc)
./cpp/build/pigeon_tests                             # run from the repo root (uses models/)
```

The models live in `models/`:
- `inat_birds.onnx`: 964-bird classifier. Pigeon/dove = the sum of 21 Columbidae species.
- `ssd_birds.onnx` with `ssd_birds.anchors`: COCO SSD bird detector.

Both were made by `tools/export_cpp_models.py` and are committed. Your own trained student is
`train.py` → `models/pigeon.f32.onnx`, then `python tools/export_cpp_models.py --student models/pigeon.f32.onnx` (writes the sidecar).

## Run

```bash
B=./cpp/build/pigeon_detect
$B --source 0 --show                                         # USB camera, motion + iNat teacher
$B --source 0 --width 640 --height 360 --cam-fps 260 --fourcc MJPG --model models/pigeon.f32.onnx --async --show
$B --source "https://www.youtube.com/watch?v=ID" --show       # YouTube: download, then real-time replay (SSD candidates)
$B --source "https://www.youtube.com/watch?v=ID" --stream --show
$B --source clip.mp4 --fast --csv runs/dets.csv               # offline, every frame, maximum speed
$B --source 0 --harvest data/crops                            # collect training crops with the live pipeline
$B --source 0 --on-pigeon "gpioset gpiochip0 12=1; sleep 0.5; gpioset gpiochip0 12=0" --cooldown 5   # deterrent
$B --bench --model models/pigeon.f32.onnx                     # per-stage latency
$B --help
```

Jetson:
- **Hardware MJPG decoding:** `--source "gst:v4l2src device=/dev/video0 io-mode=2 ! image/jpeg,width=1280,height=720,framerate=120/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR"`.
  JetPack's OpenCV is built with GStreamer, so this works directly.
- **GPU inference:** `python3 scripts/build_trt.py --onnx models/pigeon.f32.onnx --out models/pigeon.engine` (likewise for `inat_birds` and `ssd_birds`), then `--model models/pigeon.engine` or `--detector models/ssd_birds.engine`.

## Real-time design

| | |
|---|---|
| capture thread | grabs continuously and keeps only the newest frame; nothing queues up and stale frames are dropped (`dropped` in the summary) |
| motion | CNT/MOG2 on a gray frame downscaled ¼, then contours, then merged boxes |
| tracker | greedy IoU with a centroid fallback. Each object is classified when first seen, then every few frames until confident (intervals scale with fps) |
| `--async` | classification runs in a worker thread, so the frame loop never waits for the network; results arrive 1 frame later |
| crops | one batched inference call per frame; RAW Bayer input is debayered per crop only |

## Measured (x86, 4 cores, OpenCV 4.6, CPU only)

**Same pipeline, same OpenCV build, same models, frames preloaded:**

| per frame | Python | C++ |
|---|---|---|
| synthetic 960×540, mean / median | 0.75 / 0.63 ms | **0.52 / 0.39 ms** |
| real photos 960×540, mean / median / max | 1.03 / 0.81 / 10.6 ms | **0.66 / 0.46 / 4.7 ms** |

**Simulated 300 fps camera** (`--replay-fps 300`), motion + iNat teacher (4 ms/crop):

| | processed | dropped | pigeons found | other birds flagged |
|---|---|---|---|---|
| sync | 135 fps | 55% | 7/7, boxed in 19% of frames | 0/10 |
| `--async` | **300 fps** | 1% | 6/7, boxed in 51% of frames | 0/10 |

With the trained tiny student (0.1 ms/crop) both modes sustain 300 fps. At 1000 fps the limit here was video decoding.

**Output matches Python:** on the synthetic test video the C++ and Python pipelines give the same counts
(604 true positives, 85 false positives, 215 misses).

Real-photo test (7 pigeons; parrots, another bird, a cat and a person as negatives): all 7 pigeons were found and 0/10 others were boxed, with both `--find motion` and `--find ssd`.

## Backends

| backend | where | notes |
|---|---|---|
| `opencv` (default for .onnx) | everywhere | no extra dependency. Here: tiny student 0.15–0.22 ms/crop, teacher ~4 ms, SSD ~40 ms |
| `tensorrt` (default for .engine) | Jetson | compile-checked against TensorRT 8.5 (JetPack 5) and 10.3 headers. **Not run** (no NVIDIA GPU here) |
| `opencv-cuda` | OpenCV built with CUDA | not tested |

Not done: an ONNX Runtime C++ backend. Its headers couldn't be downloaded in the build environment. OpenCV DNN covers
CPU and TensorRT covers Jetson.

For a fixed USB camera, candidates come from motion detection. The SSD detector (`--find ssd`, about 40 ms/frame on CPU)
is for moving cameras and YouTube clips. On a Jetson, build it as a TensorRT engine.
