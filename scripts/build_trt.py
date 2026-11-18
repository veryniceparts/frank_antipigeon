#!/usr/bin/env python3
"""Build a TensorRT engine for the student ON THE JETSON (engines are tied to GPU + TensorRT version).

  python3 scripts/build_trt.py --onnx models/pigeon.f32.onnx --out models/pigeon.engine
  python3 scripts/build_trt.py --onnx models/pigeon.f32.onnx --out models/pigeon.dla.engine --dla 0
  python3 scripts/build_trt.py --onnx models/inat_birds.onnx --out models/inat_birds.engine   # teacher (C++ export)
  python3 scripts/build_trt.py --onnx models/ssd_birds.onnx  --out models/ssd_birds.engine    # bird detector

Uses trtexec (JetPack: /usr/src/tensorrt/bin/trtexec). FP16 by default. Use the *.f32.onnx that
train.py writes next to the uint8 model: TensorRT 8.5 (JetPack 5) does not reliably take uint8
network inputs. Writes <engine>.json with size/threshold so run.py picks the right threshold.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def read_meta(path):
    """-> input name, input size, dynamic batch?, metadata (sidecar <model>.json wins over ONNX metadata)."""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        meta = json.loads({p.key: p.value for p in m.metadata_props}.get("pigeonkit", "{}"))
        inp = m.graph.input[0]
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    except ImportError:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        inp, dims = i, i.shape
        meta = json.loads(s.get_modelmeta().custom_metadata_map.get("pigeonkit", "{}"))
    side = os.path.splitext(path)[0] + ".json"
    if os.path.exists(side):  # C++ sidecar (kind, size, threshold, targets, anchors ...)
        with open(side) as fh:
            meta = {**meta, **json.load(fh)}
    dynamic = not isinstance(dims[0], int) or dims[0] <= 0
    return inp.name, int(dims[1]), dynamic, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/pigeon.f32.onnx")
    ap.add_argument("--out", default="models/pigeon.engine")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--opt-batch", type=int, default=4, help="typical number of crops classified per frame")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "int8"],
                    help="int8 without calibration data only makes sense for speed tests")
    ap.add_argument("--dla", type=int, default=None, help="DLA core 0/1 (Xavier NX has 2); GPU fallback enabled")
    ap.add_argument("--workspace-mb", type=int, default=512)
    ap.add_argument("--trtexec", default=None)
    a = ap.parse_args()

    trtexec = a.trtexec or shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        sys.exit("trtexec not found: install JetPack's TensorRT (it lives in /usr/src/tensorrt/bin)")
    name, size, dynamic, meta = read_meta(a.onnx)
    shp = lambda b: f"{name}:{b}x{size}x{size}x3"  # noqa: E731
    cmd = [trtexec, f"--onnx={a.onnx}", f"--saveEngine={a.out}",
           f"--memPoolSize=workspace:{a.workspace_mb}", "--skipInference"]
    if dynamic:  # student: dynamic batch of crops; teacher/SSD exports have a fixed batch of 1
        cmd[3:3] = [f"--minShapes={shp(1)}", f"--optShapes={shp(a.opt_batch)}", f"--maxShapes={shp(a.max_batch)}"]
    if a.precision != "fp32":
        cmd.append(f"--{a.precision}")
    if a.dla is not None:
        cmd += [f"--useDLACore={a.dla}", "--allowGPUFallback"]
    print(" ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        # TensorRT < 8.4 uses --workspace=N instead of --memPoolSize
        cmd = [c if not c.startswith("--memPoolSize") else f"--workspace={a.workspace_mb}" for c in cmd]
        cmd = [c for c in cmd if c != "--skipInference"]
        print("retrying with legacy flags:\n" + " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(r.returncode)
    meta.setdefault("kind", "binary")
    meta.setdefault("rgb", False)
    meta.update(size=size, input=name, precision=a.precision, dla=a.dla, max_batch=a.max_batch if dynamic else 1)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {a.out}  -> test with: ./cpp/build/pigeon_detect --bench --model {a.out}")


if __name__ == "__main__":
    main()
