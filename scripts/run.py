#!/usr/bin/env python3
"""Live / offline pigeon detection.

  python scripts/run.py --source 0 --model models/pigeon.onnx --show
  python scripts/run.py --source clip.mp4 --model models/mobilenet_v2_1.0_224_inat_bird_quant.tflite --out runs/out.mp4
  python scripts/run.py --source pylon:// --model models/pigeon.onnx --scale 2 --fps 500
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import IoUTracker, MotionDetector, PigeonPipeline, draw, load_classifier, open_camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index | video file | rtsp:// | dalsa://<id|ip>?pf=BayerRG8&w=1280&h=720&exp=800&fps=300 | "
                         "csi://0?w=1280&h=720&fps=120 | gst://<pipeline> | "
                         "pylon:// | genicam://<cti>")
    ap.add_argument("--model", default="models/pigeon.onnx", help=".engine (TensorRT, Jetson) | .onnx student | iNat .tflite teacher")
    ap.add_argument("--threshold", type=float, default=None, help="override model threshold")
    ap.add_argument("--scale", type=int, default=4, help="motion detection downscale factor")
    ap.add_argument("--fps", type=float, default=100, help="nominal camera fps (background memory)")
    ap.add_argument("--method", default="auto", choices=["auto", "cnt", "avg", "mog2", "diff"])
    ap.add_argument("--min-area", type=int, default=12, help="min blob area on downscaled mask")
    ap.add_argument("--ignore-mask", default=None, help="png, white = ignore region")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=None, help="write annotated video")
    ap.add_argument("--pigeons-only", action="store_true", help="draw only pigeon boxes")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    clf = load_classifier(a.model, threshold=a.threshold)
    ign = cv2.imread(a.ignore_mask, cv2.IMREAD_GRAYSCALE) if a.ignore_mask else None
    writer = None
    n_pigeon_frames = 0
    t_start = time.perf_counter()
    fps_ema = None
    with open_camera(a.source) as cam:
        pipe = PigeonPipeline(
            clf,
            MotionDetector(scale=a.scale, fps=a.fps, method=a.method, min_area=a.min_area, ignore_mask=ign),
            IoUTracker(threshold=clf.threshold),
            bayer=cam.bayer_code,  # RAW Bayer cameras (e.g. DALSA BayerRG8): debayer crops only
        )
        while (item := cam.read()) is not None:
            fid, _, frame = item
            t = time.perf_counter()
            dets = pipe.process(frame)
            dt = time.perf_counter() - t
            fps_ema = 1 / max(dt, 1e-6) if fps_ema is None else 0.95 * fps_ema + 0.05 / max(dt, 1e-6)
            if any(d.pigeon for d in dets):
                n_pigeon_frames += 1
                # >>> hook your deterrent here (GPIO, MQTT, serial ...) <<<
            if a.show or a.out:
                vis = draw(frame.copy(), dets, show_all=not a.pigeons_only, fps=fps_ema, bayer=cam.bayer_code)
                if a.out:
                    if writer is None:
                        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, vis.shape[1::-1])
                    writer.write(vis)
                if a.show:
                    cv2.imshow("pigeonkit", vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            if a.max_frames and pipe.f >= a.max_frames:
                break
    if writer:
        writer.release()
    el = time.perf_counter() - t_start
    print(f"frames={pipe.f} wall_fps={pipe.f / el:.1f} cam_dropped={cam.stats.dropped} "
          f"cam_incomplete={cam.stats.incomplete} "
          f"crops_classified={pipe.n_classified} frames_with_pigeon={n_pigeon_frames}")
    print("per-frame:", pipe.timer.report())
    if hasattr(cam, "network_stats"):  # GigE (DALSA via Aravis): packet loss / host underruns
        st = cam.network_stats()
        keys = ("n_completed_buffers", "n_failures", "n_underruns", "n_missing_packets", "n_resent_packets")
        print("gige:", {k: st[k] for k in keys if k in st} or st)


if __name__ == "__main__":
    main()
