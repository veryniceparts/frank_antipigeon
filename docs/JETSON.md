# Deploying on Jetson Xavier NX

The Xavier NX has:
- a 6-core ARM Carmel CPU
- a 384-core Volta GPU with 48 tensor cores
- 2 NVDLA accelerators
- 8 or 16 GB of RAM

It runs **JetPack 5.1.x** (Ubuntu 20.04, Python 3.8, CUDA 11.4, TensorRT 8.5). JetPack 6 does
**not** support Xavier, so everything here targets Python 3.8; the kit is checked with `vermin` for 3.8.

## Split of work

| where | what |
|---|---|
| PC / cloud with a GPU | `train.py` (torch + timm), producing `pigeon.onnx` and `pigeon.f32.onnx` |
| Xavier NX | `harvest.py` (collect crops on site), `build_trt.py`, `run.py` |

TensorRT engines are tied to the GPU and the TensorRT version, so **always build the engine on the Jetson**.
Do not copy `.engine` files between machines.

## 1. Setup (once)

```bash
git clone <this repo> && cd frank_antipigeon
bash scripts/jetson_setup.sh          # max-performance power mode, pip deps, sanity checks
```

The script does the following:
- Sets `nvpmodel` to the 20 W 6-core mode (`MODE_20W_6CORE`) and runs `jetson_clocks`.
- Installs, for the system python3:
  - `numpy<1.25` and `opencv-contrib-python-headless`. Contrib provides the CNT background subtractor.
  - `onnxruntime` (CPU), used for metadata and as a CPU fallback.
  - `pypylon` (optional, Basler).
- Uses JetPack's own TensorRT Python bindings (`python3-libnvinfer`) and GStreamer through `python3-gi`.
  pip's OpenCV build lacks GStreamer, which is why camera capture goes through PyGObject (`GstCamera`) instead of OpenCV.
- Raises `usbfs_memory_mb` for USB3 industrial cameras.

## 2. Build the engine

```bash
scp pc:frank_antipigeon/models/pigeon.f32.onnx models/
python3 scripts/build_trt.py --onnx models/pigeon.f32.onnx --out models/pigeon.engine            # GPU FP16
python3 scripts/build_trt.py --onnx models/pigeon.f32.onnx --out models/pigeon.dla.engine --dla 0 # DLA (optional)
python3 scripts/bench.py --model models/pigeon.engine models/pigeon.dla.engine models/pigeon.onnx --batch 1 4 16
```

The `.f32.onnx` variant takes float32 NHWC BGR input in the 0–255 range; the uint8 cast happens on the host.
TensorRT 8.5 does not reliably accept uint8 network inputs. The engine has a dynamic batch of 1–32,
tuned for 4 crops. `build_trt.py` writes `pigeon.json` next to the engine with the threshold and input size.

**GPU, CPU or DLA: measure with `bench.py`, don't guess.**
- For the 80k-parameter `tiny` net, a GPU launch plus host↔device copies can cost more than the compute.
  ONNX Runtime on one Carmel core may be just as fast at batch 1, while the GPU wins at batch 8 and above.
- The DLA has higher per-call latency than the GPU. It is only useful to keep the GPU free for something else.
- Since tracking keeps classifications to about 0.3 crops per frame, either backend is far from being the bottleneck.

## 3. Cameras

| camera | `--source` | notes |
|---|---|---|
| CSI (IMX219 / IMX477) | `csi://0?w=1280&h=720&fps=120` | Argus ISP → VIC hardware converts NV12→BGRx, and the pipeline uses the **BGRx frames directly** (no CPU colour conversion). IMX219 tops out at 1280×720@120 |
| USB UVC, MJPG | `gst://v4l2src device=/dev/video0 io-mode=2 ! image/jpeg,width=1280,height=720,framerate=60/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx` | hardware JPEG decode; `--source 0` would decode MJPG on the CPU |
| USB3 / GigE Basler | `pylon://` | `pip install pypylon` (aarch64 wheels). Use Mono8/Bayer8 and a small ROI |
| other GenICam | `genicam:///opt/<vendor>/lib/<producer>.cti` | needs the vendor's aarch64 GenTL producer |

Example: `python3 scripts/run.py --source "csi://0?w=1280&h=720&fps=120" --model models/pigeon.engine --fps 120`

## 4. Motion stage on the Carmel CPU

Motion detection is the dominant CPU cost. The numbers in the README were measured on an x86 Xeon,
and Carmel cores are slower, so measure on the device:

```bash
python3 scripts/bench.py --motion --width 1280 --height 720
```

- **Keep CNT** (`--method auto` picks it when opencv-contrib is installed). It runs on multiple threads and
  tolerates swaying foliage. MOG2 is the fallback; it handles foliage just as well but is about 2.5× slower.
  `avg` is fast but flags tree edges. On the synthetic scene it scored precision 0.60 vs 0.87, so use it only together with `--ignore-mask`.
- Use `--scale 4`, which gives a 320×180 mask from 720p. Use `--scale 2` only if pigeons are under about 20 px.
- Crop the scene with the camera ROI or the sensor mode rather than processing pixels you will ignore anyway.
- Don't use `--show` in production: drawing and X11 cost several ms per frame.

## 5. Production checklist

- `sudo nvpmodel -q`: must report the 20 W 6-core mode. `sudo jetson_clocks` is lost on reboot, so add it as a systemd unit.
- `tegrastats`: watch CPU load, GPU load (`GR3D_FREQ`) and temperature. The Xavier NX dev kit fan must be on under sustained load.
- Deterrent output: `pip install Jetson.GPIO` and switch a pin at the hook in `scripts/run.py`.
  Add a cooldown so the deterrent does not fire on every frame of the same track.
- Record misclassified events (`--out`), harvest them, re-label them and retrain.
  Rebuild the engine after every retrain.

## Not verified

This kit was developed without Jetson hardware. What has been verified:
- PyGObject capture (`GstCamera`) and the BGRx path, tested with GStreamer `videotestsrc` on x86.
- The float-input ONNX export: identical output to the uint8 model.
- Python 3.8 syntax compatibility (`vermin`).

What has **not** been run:
- The TensorRT backend (`TensorRTPigeonClassifier` plus its ctypes CUDA calls) and `build_trt.py`.
- The `nvarguscamerasrc` and `nvv4l2decoder` pipelines.
- Any speed measurement on Carmel, Volta or the DLA.

Run `scripts/bench.py` and `pytest tests` on the device first.
