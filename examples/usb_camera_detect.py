#!/usr/bin/env python3
"""USB camera -> real-time pigeon/dove detection. Other birds are ignored.

    python examples/usb_camera_detect.py                    # /dev/video0, 1280x720 MJPG
    python examples/usb_camera_detect.py --device 1 --width 640 --height 480 --fps 120
    python examples/usb_camera_detect.py --find ssd         # also finds perched / still birds
    # Jetson: hardware MJPG decoding instead of CPU decoding
    python examples/usb_camera_detect.py --gst "v4l2src device=/dev/video0 io-mode=2 ! image/jpeg,width=1280,height=720,framerate=60/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx"

Fixed camera -> motion detection (default) is the fastest way to find birds in flight.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import (IoUTracker, MotionDetector, OpenCVCamera, PigeonPipeline, SSDBirdDetector,  # noqa: E402
                       download_teacher, draw, load_classifier, open_camera)

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--gst", default=None, help="GStreamer pipeline instead of --device (Jetson HW decode)")
ap.add_argument("--find", default="motion", choices=["motion", "ssd"])
ap.add_argument("--model", default=None, help="trained models/pigeon.onnx|.engine (default: iNat teacher)")
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()

cam = open_camera("gst://" + a.gst) if a.gst else \
    OpenCVCamera(a.device, width=a.width, height=a.height, fps=a.fps, fourcc="MJPG")
clf = load_classifier(a.model or download_teacher("models"), threshold=a.threshold)
finder = MotionDetector(scale=4, fps=a.fps) if a.find == "motion" else SSDBirdDetector(every=2)
pipe = PigeonPipeline(clf, finder, IoUTracker(threshold=a.threshold))

with cam:
    while (item := cam.read()) is not None:
        frame = item[2]
        dets = pipe.process(frame)
        if any(d.pigeon for d in dets):
            pass  # >>> trigger your deterrent here <<<
        cv2.imshow("pigeon / dove detection  (q = quit)", draw(frame, dets, show_all=False, label="pigeon/dove"))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
print(f"camera {cam.stats.fps():.0f} fps, dropped {cam.stats.dropped}; per frame: {pipe.timer.report()}")
