#!/usr/bin/env python3
"""INT8 static quantization (QDQ, per-channel) of the exported student, calibrated on real crops.

  python scripts/quantize.py --model models/pigeon.onnx --crops data/crops --out models/pigeon.int8.onnx

Typically 1.5-3x faster on CPUs with VNNI / ARM dot-product. Always compare accuracy afterwards
(scripts/bench.py --compare). On old x86 without VNNI keep fp32.
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


class Reader(CalibrationDataReader):
    def __init__(self, files, size, name):
        self.it = iter(files)
        self.size, self.name = size, name

    def get_next(self):
        f = next(self.it, None)
        if f is None:
            return None
        im = cv2.resize(cv2.imread(f), (self.size, self.size), interpolation=cv2.INTER_AREA)
        return {self.name: im[None]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pigeon.onnx")
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="models/pigeon.int8.onnx")
    a = ap.parse_args()
    s = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    size = int(inp.shape[1])
    files = [f for f in glob.glob(os.path.join(a.crops, "**", "*"), recursive=True)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.seed(0)
    random.shuffle(files)
    pre = os.path.splitext(a.out)[0] + ".pre.onnx"
    quant_pre_process(a.model, pre)
    quantize_static(pre, a.out, Reader(files[: a.n], size, inp.name), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    os.remove(pre)
    # carry over metadata (threshold etc.)
    src, dst = onnx.load(a.model), onnx.load(a.out)
    del dst.metadata_props[:]
    dst.metadata_props.extend(src.metadata_props)
    onnx.save(dst, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
