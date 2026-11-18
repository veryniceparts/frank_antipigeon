#!/usr/bin/env python3
"""One-time (offline, on a PC) conversion of the pretrained TFLite models for the C++ runtime.

    python -m venv tfenv && tfenv/bin/pip install "tensorflow-cpu==2.20.*" tf2onnx onnx ai-edge-litert
    tfenv/bin/python tools/export_cpp_models.py --out models/

Produces (every model: input float32 NHWC, pixel values 0..255, + a .json sidecar the C++ app reads):
  inat_birds.onnx    iNaturalist 964-bird classifier (pigeon/dove decision = sum of 21 Columbidae classes)
  ssd_birds.onnx     COCO SSDLite-MobileDet *without* the TFLite post-process op (box decoding + NMS
                     are done in C++), so it runs on OpenCV DNN, ONNX Runtime and TensorRT alike
  ssd_birds.anchors  2034x4 float32 anchors (ycenter, xcenter, h, w) for the box decoding

The student models from scripts/train.py are already C++-ready: use *.f32.onnx (+ write the sidecar
with --student).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit.classifier import INAT_COLUMBIDAE, INAT_LABELS_URL, INAT_URL, TARGETS  # noqa: E402
from pigeonkit.detector import SSD_URL  # noqa: E402


def fetch(url, d):
    p = os.path.join(d, os.path.basename(url))
    if not os.path.exists(p):
        print("download", url)
        urllib.request.urlretrieve(url, p)
    return p


def tflite_to_onnx(tfl, onnx_path):
    # --dequantize: float graph, identical behaviour on every backend (no runtime-specific int8 rounding)
    subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--tflite", tfl, "--output", onnx_path,
                    "--opset", "17", "--dequantize"], check=True)


def float_input(model):
    """uint8 input -> float32 input (values stay 0..255) and drop the now-useless leading Cast,
    which some backends (OpenCV DNN) don't implement."""
    from onnx import TensorProto
    inp = model.graph.input[0]
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    for n in list(model.graph.node):
        if n.op_type == "Cast" and n.input[0] == inp.name:
            for m in model.graph.node:
                m.input[:] = [inp.name if i == n.output[0] else i for i in m.input]
            model.graph.node.remove(n)
    return model


def opencv_compatible(model):
    """OpenCV DNN 4.5/4.6 (JetPack 5) only parses Clip with min/max *attributes* (opset <= 10).
    All Clips here are ReLU6 with constant bounds -> rewrite them and declare opset 10; the remaining
    ops (Conv, Add, Mul, Sub, Reshape, Transpose, Concat, Sigmoid, MatMul, AveragePool, Softmax over the
    last axis of a 2-D tensor) have identical semantics there."""
    from onnx import helper, numpy_helper
    vals = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    vals.update({n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in model.graph.node if n.op_type == "Constant"})
    allowed = {"Conv", "Add", "Mul", "Sub", "Reshape", "Transpose", "Concat", "Sigmoid", "MatMul", "AveragePool",
               "Softmax", "Clip", "Constant", "Relu", "Gemm", "Flatten", "GlobalAveragePool", "Div"}
    bad = {n.op_type for n in model.graph.node} - allowed
    if bad:
        raise RuntimeError(f"ops {bad} not checked for the opset-10 rewrite")
    for i, n in enumerate(list(model.graph.node)):
        if n.op_type == "Clip" and len(n.input) == 3:
            lo, hi = float(vals[n.input[1]]), float(vals[n.input[2]])
            model.graph.node.remove(n)
            model.graph.node.insert(i, helper.make_node("Clip", [n.input[0]], list(n.output), name=n.name, min=lo, max=hi))
        if n.op_type == "Softmax":
            assert not n.attribute or n.attribute[0].i in (-1, 1), "softmax axis"
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = 10
    used = {i for n in model.graph.node for i in n.input}
    for init in list(model.graph.initializer):  # the old Clip bound constants
        if init.name not in used:
            model.graph.initializer.remove(init)
    return model


def cut(src, dst, outputs, rename):
    """Keep the graph up to `outputs` and give them stable names."""
    import onnx
    onnx.utils.extract_model(src, dst, [onnx.load(src).graph.input[0].name], outputs)
    m = opencv_compatible(float_input(onnx.load(dst)))
    for o in m.graph.output:
        o.name = rename.get(o.name, o.name)
    for n in m.graph.node:
        n.output[:] = [rename.get(o, o) for o in n.output]
        n.input[:] = [rename.get(i, i) for i in n.input]
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--student", default=None, help="also write the sidecar for a trained *.f32.onnx")
    a = ap.parse_args()
    import onnx
    os.makedirs(a.out, exist_ok=True)

    if a.student:
        import onnxruntime as ort
        s = ort.InferenceSession(a.student, providers=["CPUExecutionProvider"])
        meta = json.loads(s.get_modelmeta().custom_metadata_map.get("pigeonkit", "{}"))
        side = {"kind": "binary", "input": s.get_inputs()[0].name, "size": int(s.get_inputs()[0].shape[1]),
                "rgb": False, "threshold": meta.get("threshold", 0.5), "gray": meta.get("gray", False)}
        json.dump(side, open(os.path.splitext(a.student)[0] + ".json", "w"), indent=1)
        print("sidecar for", a.student, side)
        return

    tmp = tempfile.mkdtemp()
    # --- iNat classifier
    tfl = fetch(INAT_URL, tmp)
    labels = fetch(INAT_LABELS_URL, a.out)
    out = os.path.join(a.out, "inat_birds.onnx")
    full = os.path.join(tmp, "inat_full.onnx")
    tflite_to_onnx(tfl, full)
    # stop at the float Softmax (drop the final QuantizeLinear back to uint8)
    m = cut(full, out, ["prediction_prequant"], {"prediction_prequant": "probs"})
    names = [l.strip() for l in open(labels)]
    json.dump({"kind": "inat", "input": m.graph.input[0].name, "size": 224, "rgb": True,
               # ONNX float softmax is less over-confident than TFLite's integer softmax: on real photos pigeons
               # score >= 0.57 (TFLite: >= 0.80), non-pigeons <= 0.01 -> 0.4 sits in the middle of the gap
               "threshold": 0.4,
               "targets": {k: list(v) for k, v in TARGETS.items()},
               "names": {str(i): names[i] for i in INAT_COLUMBIDAE}},
              open(os.path.join(a.out, "inat_birds.json"), "w"), indent=1)

    # --- SSD bird detector, cut before TFLite_Detection_PostProcess
    tfl = fetch(SSD_URL, tmp)
    full = os.path.join(tmp, "ssd_full.onnx")
    tflite_to_onnx(tfl, full)
    out = os.path.join(a.out, "ssd_birds.onnx")
    m = cut(full, out, ["Squeeze1_prequant", "convert_scores_prequant"],
            {"Squeeze1_prequant": "box_encodings", "convert_scores_prequant": "scores"})
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=tfl)
    it.allocate_tensors()
    pp = [o for o in it._get_ops_details() if o["op_name"] == "TFLite_Detection_PostProcess"][0]
    anchors = it.get_tensor(int(pp["inputs"][2])).astype(np.float32)  # (2034, 4) ycenter, xcenter, h, w
    anchors.tofile(os.path.join(a.out, "ssd_birds.anchors"))
    json.dump({"kind": "ssd", "input": m.graph.input[0].name, "size": 320, "rgb": True, "n_anchors": len(anchors),
               "n_classes": 91, "bird_class": 16, "scales": [10.0, 10.0, 5.0, 5.0], "score": 0.3, "nms_iou": 0.45,
               "anchors": "ssd_birds.anchors"},
              open(os.path.join(a.out, "ssd_birds.json"), "w"), indent=1)
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()
