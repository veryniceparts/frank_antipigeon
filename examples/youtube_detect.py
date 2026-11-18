#!/usr/bin/env python3
"""YouTube link (or local video) -> real-time pigeon/dove detection. Other birds are ignored.

    pip install yt-dlp opencv-contrib-python ai-edge-litert onnxruntime
    python examples/youtube_detect.py "https://www.youtube.com/watch?v=VIDEO_ID"            # download, then replay
    python examples/youtube_detect.py "https://www.youtube.com/watch?v=VIDEO_ID" --stream   # play while streaming
    python examples/youtube_detect.py clip.mp4 --out runs/clip_detected.mp4 --no-show

The video is replayed at its own fps (like a live camera); frames the detector can't keep up with
are dropped. Works without training: the iNaturalist bird classifier (964 species) decides
pigeon/dove vs. everything else. --model models/pigeon.onnx uses your trained fast model instead.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import (IoUTracker, MotionDetector, PigeonPipeline, SSDBirdDetector, download_teacher,  # noqa: E402
                       draw, load_classifier, open_camera, youtube)

ap = argparse.ArgumentParser()
ap.add_argument("video", help="YouTube/any yt-dlp URL, or a local file")
ap.add_argument("--stream", action="store_true", help="don't download, play the stream directly")
ap.add_argument("--find", default="ssd", choices=["ssd", "motion"],
                help="ssd: bird detector, works with moving camera + perched birds (default for videos); "
                     "motion: background subtraction, fastest, needs a fixed camera")
ap.add_argument("--model", default=None, help="default: iNat teacher (auto-download)")
ap.add_argument("--target", default="columbidae", choices=["columbidae", "pigeon", "feral"],
                help="columbidae = pigeons + doves (default)")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--out", default=None, help="save annotated video")
ap.add_argument("--no-show", action="store_true")
ap.add_argument("--debug", action="store_true", help="also draw non-pigeon candidates in grey")
a = ap.parse_args()

src = a.video
if youtube.is_url(src):
    src = youtube.fetch(src, mode="stream" if a.stream else "download")
    print("source:", src if not a.stream else "stream URL")

clf = load_classifier(a.model or download_teacher("models"), threshold=a.threshold, target=a.target)
finder = SSDBirdDetector(score=0.25) if a.find == "ssd" else MotionDetector(scale=2)
pipe = PigeonPipeline(clf, finder, IoUTracker(threshold=a.threshold, n_unsure=3, n_sure=15, min_hits=2))

writer, shown, t0 = None, 0, time.perf_counter()
with open_camera(src, realtime=True) as cam:
    while (item := cam.read()) is not None:
        frame = item[2]
        dets = pipe.process(frame)
        vis = draw(frame, dets, show_all=a.debug, label="pigeon/dove",
                   fps=shown / max(1e-6, time.perf_counter() - t0))
        shown += 1
        if a.out:
            if writer is None:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), cam.src_fps, vis.shape[1::-1])
            writer.write(vis)
        if not a.no_show:
            cv2.imshow("pigeon / dove detection  (q = quit)", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
if writer:
    writer.release()
print(f"video {cam.stats.grabbed} frames @ {cam.src_fps:.0f} fps, processed {shown}, dropped {cam.stats.dropped} "
      f"(real-time), per frame: {pipe.timer.report()}")
