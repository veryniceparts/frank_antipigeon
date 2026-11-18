import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pigeonkit import IoUTracker, MotionDetector, PigeonPipeline, merge_boxes, square_crops  # noqa: E402


def moving_square_frames(n=60, W=320, H=240):
    rng = np.random.default_rng(0)
    for i in range(n):
        f = np.full((H, W, 3), 180, np.uint8)
        x = 20 + i * 4
        cv2.rectangle(f, (x, 100), (x + 24, 124), (60, 60, 60), -1)
        yield cv2.add(f, rng.integers(0, 4, f.shape, dtype=np.uint8)), (x, 100, x + 24, 124)


@pytest.mark.parametrize("method", ["mog2", "diff", "avg"] + (["cnt"] if hasattr(cv2, "bgsegm") else []))
def test_motion_finds_moving_square(method):
    md = MotionDetector(scale=2, method=method, min_area=4)
    hits = 0
    for i, (f, gt) in enumerate(moving_square_frames()):
        b = md.detect(f)
        if i > 20 and len(b):
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            hits += np.any((abs(cx - (gt[0] + 12)) < 20) & (abs(cy - 112) < 20))
    assert hits > 30


def test_merge_boxes():
    b = np.array([[0, 0, 10, 10], [11, 0, 20, 10], [100, 100, 110, 110]])
    m = merge_boxes(b, gap=2)
    assert len(m) == 2 and [0, 0, 20, 10] in m.tolist()


def test_square_crops_edges():
    f = np.random.randint(0, 255, (100, 100, 3), np.uint8)
    c = square_crops(f, np.array([[0, 0, 30, 10], [90, 90, 100, 100]]), 64)
    assert c.shape == (2, 64, 64, 3) and c.dtype == np.uint8
    g = square_crops(f[..., 0], np.array([[10, 10, 20, 20]]), 32)
    assert g.shape == (1, 32, 32, 3)
    bgrx = np.dstack([f, np.full(f.shape[:2], 255, np.uint8)])  # Jetson BGRx frames
    assert np.array_equal(square_crops(bgrx, np.array([[0, 0, 30, 10]]), 64), c[:1])
    assert len(MotionDetector(scale=2).detect(bgrx)) == 0


def test_tracker_ema_and_schedule():
    tr = IoUTracker(n_unsure=3, n_sure=50, min_hits=2, threshold=0.6)
    todo_count = 0
    for f in range(30):
        box = np.array([[10 + f, 10, 40 + f, 40]])
        todo = tr.update(box, f)
        for i in todo:
            tr.feed(i, 0.95, f)
            todo_count += 1
    t = next(iter(tr.tracks.values()))
    assert len(tr.tracks) == 1 and t.hits == 30
    assert tr.is_pigeon(t) and t.p > 0.9
    assert todo_count <= 4  # classified a few times, then only every n_sure frames


class FakeClf:
    size = 32
    threshold = 0.5

    def __call__(self, crops):
        # "pigeon" = crop contains a dark object
        return (crops.reshape(len(crops), -1).min(1) < 100).astype(np.float32)


def test_pipeline_end_to_end():
    pipe = PigeonPipeline(FakeClf(), MotionDetector(scale=2, method="mog2", min_area=4))
    flagged = 0
    for f, _ in moving_square_frames(80):
        dets = pipe.process(f)
        flagged += any(d.pigeon for d in dets)
    assert flagged > 20


def test_onnx_export_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from pigeonkit.classifier import OnnxPigeonClassifier
    from pigeonkit.models import Deploy, TinyPigeonNet, export_onnx
    net = TinyPigeonNet().eval()
    p = export_onnx(net, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), 48, str(tmp_path / "m.onnx"), {"threshold": 0.7})
    clf = OnnxPigeonClassifier(p)
    assert clf.size == 48 and clf.threshold == 0.7
    x = np.random.randint(0, 255, (5, 48, 48, 3), np.uint8)
    ref = Deploy(net, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)).eval()(torch.from_numpy(x)).detach().numpy()
    assert np.allclose(clf(x), ref, atol=1e-5)


def test_gst_camera_bgrx_and_gray():
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
    except (ImportError, ValueError) as e:
        pytest.skip(f"PyGObject/GStreamer unavailable: {e}")
    if Gst.ElementFactory.find("videotestsrc") is None:
        pytest.skip("videotestsrc not available")
    from pigeonkit import open_camera
    with open_camera("gst://videotestsrc num-buffers=3 ! video/x-raw,width=64,height=48,format=BGRx") as cam:
        assert cam.read()[2].shape == (48, 64, 4)
    with open_camera("gst://videotestsrc num-buffers=3 ! video/x-raw,width=66,height=10,format=GRAY8") as cam:
        assert cam.read()[2].shape == (10, 66)  # padded row stride handled
