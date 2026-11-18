"""Crop classifiers. All take a batch of uint8 BGR crops (N,S,S,3) and return P(pigeon) (N,).

* OnnxPigeonClassifier  - the fast student (scripts/train.py). Preprocessing (BGR->RGB,
  normalisation) is baked into the ONNX graph, so input is raw uint8 crops. ~0.1-0.4 ms/crop.
* INatBirdClassifier    - zero-training teacher: Google's MobileNetV2 iNaturalist-birds
  TFLite (964 species + background, Apache-2.0). P(pigeon) = Rock Pigeon + Feral Pigeon
  (+ Wood-Pigeon). ~5 ms/crop on CPU, ~1 ms on a Coral EdgeTPU.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

INAT_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v2_1.0_224_inat_bird_quant.tflite"
INAT_LABELS_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/inat_bird_labels.txt"
# 0-based indices in inat_bird_labels.txt
INAT_PIGEON = (307, 958)          # Columba livia, Columba livia domestica
INAT_WOODPIGEON = 310             # Columba palumbus
INAT_BACKGROUND = 964


class OnnxPigeonClassifier:
    def __init__(self, path: str, threads: int = 1, threshold: float | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # tiny models: 1 thread is fastest (thread wake-up dominates); batch crops instead
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider")
                     if p in ort.get_available_providers()] + ["CPUExecutionProvider"]
        if os.environ.get("PIGEON_CPU_ONLY"):
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, so, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[1])  # NHWC
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.meta = json.loads(meta.get("pigeonkit", "{}"))
        self.threshold = threshold if threshold is not None else float(self.meta.get("threshold", 0.5))

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros(0, np.float32)
        return self.sess.run(None, {self.input_name: np.ascontiguousarray(crops)})[0].reshape(-1)


class INatBirdClassifier:
    size = 224

    def __init__(self, path: str, threads: int = 2, include_woodpigeon: bool = True, threshold: float = 0.5):
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:  # older installs
            from tflite_runtime.interpreter import Interpreter, load_delegate
        delegates = [load_delegate("libedgetpu.so.1")] if "edgetpu" in os.path.basename(path) else None
        self.it = Interpreter(model_path=path, num_threads=threads, experimental_delegates=delegates)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.oscale, self.ozero = self.out["quantization"]
        self.idx = list(INAT_PIGEON) + ([INAT_WOODPIGEON] if include_woodpigeon else [])
        self.threshold = threshold

    def probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = crop_bgr
        if x.shape[0] != 224:
            x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)[None]
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        q = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        return (q - self.ozero) * self.oscale

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        # model has a fixed batch of 1
        return np.array([self.probs(c)[self.idx].sum() for c in crops], np.float32)


def load_classifier(path: str, **kw):
    kw = {k: v for k, v in kw.items() if v is not None}
    if path.endswith(".tflite"):
        return INatBirdClassifier(path, **{k: v for k, v in kw.items() if k in ("threads", "threshold")})
    return OnnxPigeonClassifier(path, **kw)


def download_teacher(dst_dir: str = "models") -> str:
    import urllib.request
    os.makedirs(dst_dir, exist_ok=True)
    out = os.path.join(dst_dir, os.path.basename(INAT_URL))
    for url in (INAT_URL, INAT_LABELS_URL):
        p = os.path.join(dst_dir, os.path.basename(url))
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
    return out
