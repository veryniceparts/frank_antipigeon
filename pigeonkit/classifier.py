"""Crop classifiers. All take a batch of uint8 BGR crops (N,S,S,3) and return P(pigeon) (N,).

* OnnxPigeonClassifier  - the fast student (scripts/train.py). Preprocessing (BGR->RGB,
  normalisation) is baked into the ONNX graph, so input is raw uint8 crops. ~0.1-0.4 ms/crop.
* TensorRTPigeonClassifier - same student as a TensorRT FP16 engine (Jetson GPU / DLA).
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
# All pigeons AND doves (family Columbidae) among the 964 species (0-based = line - 1).
# Deliberately NOT included: 459 Cepphus columba "Pigeon Guillemot" (a seabird).
INAT_COLUMBIDAE = (
    298, 302, 307, 310, 312, 313, 314,  # Turtle-Dove, Collared-Dove, Rock Pigeon, Wood-Pigeon, 3 Patagioenas
    325, 333, 338, 341, 342, 417, 441,  # White-tipped, Eared, Ground-Doves, Zebra Dove, Crested Pigeon, Mourning
    452, 551, 628, 683, 743, 893, 958,  # White-winged, Inca, Laughing, Zenaida, NZ Pigeon, Spotted, Feral Pigeon
)
COLUMBIDAE_GENERA = {
    "Columba", "Streptopelia", "Spilopelia", "Zenaida", "Columbina", "Patagioenas", "Geopelia", "Leptotila",
    "Ocyphaps", "Phaps", "Treron", "Ducula", "Ptilinopus", "Chalcophaps", "Hemiphaga", "Geophaps", "Leucosarcia",
    "Macropygia", "Gallicolumba", "Caloenas", "Goura", "Oena", "Turtur", "Metriopelia", "Claravis", "Geotrygon",
    "Lophophaps", "Petrophassa", "Alopecoenas", "Uropelia", "Zentrygon"}
TARGETS = {
    "columbidae": INAT_COLUMBIDAE,                      # pigeons + doves (default)
    "pigeon": INAT_PIGEON + (INAT_WOODPIGEON,),         # pigeons only (rock/feral/wood)
    "feral": INAT_PIGEON,                               # city pigeon only
}


def inat_label_names(labels_path: str | None = None) -> list[str]:
    p = labels_path or os.path.join("models", "inat_bird_labels.txt")
    try:
        with open(p) as fh:
            return [l.strip() for l in fh]
    except OSError:
        return []


class OnnxPigeonClassifier:
    def __init__(self, path: str, threads: int = 1, threshold: float | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # tiny models: 1 thread is fastest (thread wake-up dominates); batch crops instead
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # CPU by default: for tiny models a GPU launch + copies costs more than the compute.
        # On Jetson use the native TensorRT engine (TensorRTPigeonClassifier) instead.
        # Override e.g.  PIGEON_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
        env = os.environ.get("PIGEON_ORT_PROVIDERS")
        providers = env.split(",") if env else ["CPUExecutionProvider"]
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

    def __init__(self, path: str, threads: int = 2, target: str = "columbidae", threshold: float = 0.5):
        """target: 'columbidae' (pigeons + doves, default) | 'pigeon' | 'feral'.
        Every other bird species (and non-birds -> 'background' class) counts as negative."""
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
        self.idx = list(TARGETS[target])
        names = inat_label_names(os.path.join(os.path.dirname(path), "inat_bird_labels.txt"))
        if names:  # sanity check against the shipped label file
            bad = [i for i in self.idx if names[i].split()[0] not in COLUMBIDAE_GENERA]
            if bad:
                raise ValueError(f"label mismatch for indices {bad}: wrong labels file / model?")
        self.names = names
        self.last_top = []  # per crop: best matching target species name (for display)
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
        out, self.last_top = [], []
        for c in crops:
            p = self.probs(c)[self.idx]
            out.append(p.sum())
            if self.names:
                self.last_top.append(self.names[self.idx[int(p.argmax())]])
        return np.array(out, np.float32)


class _CudaRT:
    """Minimal ctypes binding to libcudart (always present on JetPack) -> no pycuda/cuda-python needed."""

    H2D, D2H = 1, 2

    def __init__(self):
        import ctypes
        self.ct = ctypes
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("libcudart not found (is CUDA installed / LD_LIBRARY_PATH set?)")

    def _ok(self, r, what):
        if r != 0:
            raise RuntimeError(f"{what} failed with CUDA error {r}")

    def malloc(self, n):
        p = self.ct.c_void_p()
        self._ok(self.lib.cudaMalloc(self.ct.byref(p), self.ct.c_size_t(n)), "cudaMalloc")
        return p.value

    def malloc_host(self, n, dtype, shape):
        """Pinned host buffer exposed as a numpy array (fast async DMA)."""
        p = self.ct.c_void_p()
        self._ok(self.lib.cudaMallocHost(self.ct.byref(p), self.ct.c_size_t(n)), "cudaMallocHost")
        buf = (self.ct.c_uint8 * n).from_address(p.value)
        return p.value, np.frombuffer(buf, dtype=dtype).reshape(shape)

    def stream(self):
        s = self.ct.c_void_p()
        self._ok(self.lib.cudaStreamCreate(self.ct.byref(s)), "cudaStreamCreate")
        return s.value

    def copy(self, dst, src, n, kind, stream):
        self._ok(self.lib.cudaMemcpyAsync(self.ct.c_void_p(dst), self.ct.c_void_p(src), self.ct.c_size_t(n),
                                          self.ct.c_int(kind), self.ct.c_void_p(stream)), "cudaMemcpyAsync")

    def sync(self, stream):
        self._ok(self.lib.cudaStreamSynchronize(self.ct.c_void_p(stream)), "cudaStreamSynchronize")


class TensorRTPigeonClassifier:
    """Native TensorRT engine (Jetson). Build it ON the device: scripts/build_trt.py.

    Engine input: NHWC BGR crops with dynamic batch (uint8 or float32 0..255, auto-detected);
    output P(pigeon). Batches larger than the engine's max profile are split.
    Supports TensorRT 8.5+ (execute_async_v3) and older bindings API (execute_async_v2).
    """

    def __init__(self, path: str, threshold: float | None = None, **_):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as fh:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {path}: engines are specific to the GPU + TensorRT "
                               "version they were built with -> rebuild on this device (scripts/build_trt.py)")
        self.ctx = self.engine.create_execution_context()
        self.v3 = hasattr(self.engine, "num_io_tensors") and hasattr(self.ctx, "execute_async_v3")
        if self.v3:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.in_name, self.out_name = ins[0], [n for n in names if n not in ins][0]
            in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
            self.in_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.in_name)))
            self.out_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.out_name)))
            prof = self.engine.get_tensor_profile_shape(self.in_name, 0) if hasattr(self.engine, "get_tensor_profile_shape") \
                else self.engine.get_profile_shape(0, self.in_name)
        else:
            self.in_idx = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)][0]
            self.out_idx = 1 - self.in_idx
            in_shape = tuple(self.engine.get_binding_shape(self.in_idx))
            self.in_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.in_idx)))
            self.out_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.out_idx)))
            prof = self.engine.get_profile_shape(0, self.in_idx)
        self.size = int(in_shape[1])
        self.max_batch = int(prof[2][0])
        meta = {}
        side = os.path.splitext(path)[0] + ".json"
        if os.path.exists(side):
            with open(side) as fh:
                meta = json.load(fh)
        self.meta = meta
        self.threshold = threshold if threshold is not None else float(meta.get("threshold", 0.5))

        self.cuda = _CudaRT()
        self.stream = self.cuda.stream()
        S, B = self.size, self.max_batch
        self.in_bytes_per = S * S * 3 * self.in_dtype.itemsize
        self.h_in_ptr, self.h_in = self.cuda.malloc_host(B * self.in_bytes_per, self.in_dtype, (B, S, S, 3))
        self.h_out_ptr, self.h_out = self.cuda.malloc_host(B * self.out_dtype.itemsize, self.out_dtype, (B,))
        self.d_in = self.cuda.malloc(B * self.in_bytes_per)
        self.d_out = self.cuda.malloc(B * self.out_dtype.itemsize)
        if self.v3:
            self.ctx.set_tensor_address(self.in_name, self.d_in)
            self.ctx.set_tensor_address(self.out_name, self.d_out)
        else:
            self.bindings = [0, 0]
            self.bindings[self.in_idx], self.bindings[self.out_idx] = self.d_in, self.d_out

    def _run(self, crops):
        n, S = len(crops), self.size
        self.h_in[:n] = crops  # uint8 -> engine dtype cast happens here if needed
        if self.v3:
            self.ctx.set_input_shape(self.in_name, (n, S, S, 3))
        else:
            self.ctx.set_binding_shape(self.in_idx, (n, S, S, 3))
        self.cuda.copy(self.d_in, self.h_in_ptr, n * self.in_bytes_per, _CudaRT.H2D, self.stream)
        ok = self.ctx.execute_async_v3(self.stream) if self.v3 else \
            self.ctx.execute_async_v2(self.bindings, self.stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        self.cuda.copy(self.h_out_ptr, self.d_out, n * self.out_dtype.itemsize, _CudaRT.D2H, self.stream)
        self.cuda.sync(self.stream)
        return self.h_out[:n].astype(np.float32)

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros(0, np.float32)
        B = self.max_batch
        return np.concatenate([self._run(crops[i:i + B]) for i in range(0, len(crops), B)])


def load_classifier(path: str, **kw):
    kw = {k: v for k, v in kw.items() if v is not None}
    if path.endswith((".engine", ".plan", ".trt")):
        return TensorRTPigeonClassifier(path, **kw)
    if path.endswith(".tflite"):
        return INatBirdClassifier(path, **{k: v for k, v in kw.items() if k in ("threads", "threshold", "target")})
    kw.pop("target", None)  # the trained student already encodes its target classes
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
