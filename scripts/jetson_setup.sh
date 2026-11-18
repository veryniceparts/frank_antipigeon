#!/usr/bin/env bash
# One-time setup for Jetson Xavier NX (JetPack 5.1.x, Python 3.8). See docs/JETSON.md.
set -euo pipefail

if [ -f /etc/nv_tegra_release ]; then
  head -1 /etc/nv_tegra_release
  grep -q "R35" /etc/nv_tegra_release || echo "WARNING: expected JetPack 5 (L4T R35.x); Xavier NX is not supported by JetPack 6"
else
  echo "WARNING: not a Jetson (no /etc/nv_tegra_release)"
fi

# --- performance mode: 20 W, 6 cores (mode id differs between releases -> look it up by name)
if command -v nvpmodel >/dev/null; then
  MODE=$(grep -E "POWER_MODEL.*NAME=MODE_20W_6CORE" /etc/nvpmodel.conf | sed -E 's/.*ID=([0-9]+).*/\1/' | head -1 || true)
  if [ -n "${MODE:-}" ]; then sudo nvpmodel -m "$MODE"; else echo "MODE_20W_6CORE not found; check: sudo nvpmodel -q --verbose"; fi
  sudo jetson_clocks || true
fi

# --- system packages: TensorRT python bindings + GStreamer introspection come with JetPack
sudo apt-get update
sudo apt-get install -y python3-pip python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-tools python3-libnvinfer || true

# --- python deps (system python3 so the JetPack tensorrt module stays importable)
python3 -m pip install --user --upgrade pip
python3 -m pip install --user "numpy<1.25" opencv-contrib-python-headless "onnxruntime<1.20" "onnx<1.17" pytest
python3 -m pip install --user pypylon || echo "pypylon not installed (only needed for Basler cameras)"

# --- USB3 industrial cameras need a larger usbfs buffer (default 16 MB)
echo 1000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null || true

# --- sanity checks
python3 - <<'EOF'
import cv2, numpy, sys
print("python", sys.version.split()[0], "numpy", numpy.__version__, "opencv", cv2.__version__, "CNT:", hasattr(cv2, "bgsegm"))
try:
    import tensorrt; print("tensorrt", tensorrt.__version__)
except ImportError:
    print("tensorrt python module MISSING (sudo apt install python3-libnvinfer)")
try:
    import gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; Gst.init(None)
    print("gstreamer", Gst.version_string(), "nvarguscamerasrc:", Gst.ElementFactory.find("nvarguscamerasrc") is not None)
except Exception as e:
    print("PyGObject/GStreamer problem:", e)
EOF
python3 -m pytest -q tests || true
echo "next: copy models/pigeon.f32.onnx here, then: python3 scripts/build_trt.py"
