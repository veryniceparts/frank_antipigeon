"""Frame acquisition with "latest frame only" semantics.

Every backend runs one producer thread that only grabs and publishes into a
1-slot holder. The consumer always gets the newest frame; stale frames are
dropped (counted in ``stats``), so processing never lags behind the camera.

    cam = open_camera("0")                 # V4L2 / UVC webcam index
    cam = open_camera("video.mp4")         # file (paced=False -> as fast as possible)
    cam = open_camera("pylon://")          # Basler (pypylon), first device
    cam = open_camera("genicam:///opt/mvIMPACT/lib/x86_64/mvGenTLProducer.cti")
    with cam:
        while (item := cam.read()) is not None:
            fid, ts_ns, frame = item
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CamStats:
    grabbed: int = 0
    delivered: int = 0
    t0: float = field(default_factory=time.perf_counter)

    @property
    def dropped(self) -> int:
        return self.grabbed - self.delivered

    def fps(self) -> float:
        return self.grabbed / max(1e-9, time.perf_counter() - self.t0)


class _Latest:
    """1-slot mailbox: put() overwrites, get() blocks until something new."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self._closed = False

    def put(self, item):
        with self._cv:
            self._item = item
            self._cv.notify()

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def get(self, timeout: float | None = 2.0):
        with self._cv:
            if self._item is None and not self._closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item


class Camera:
    """Base class. Subclasses implement _open(), _grab() -> ndarray|None, _close()."""

    def __init__(self, latest_only: bool = True):
        self.latest_only = latest_only
        self.stats = CamStats()
        self._box = _Latest()
        self._run = False
        self._eof = False
        self._th: threading.Thread | None = None

    # -- subclass API --------------------------------------------------------
    def _open(self): ...
    def _grab(self) -> np.ndarray | None: raise NotImplementedError
    def _close(self): ...

    # -- public API ----------------------------------------------------------
    def start(self):
        self._open()
        self._run = True
        if self.latest_only:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
        return self

    def _loop(self):
        while self._run:
            f = self._grab()
            if f is None:
                self._eof = True
                break
            self.stats.grabbed += 1
            self._box.put((self.stats.grabbed, time.monotonic_ns(), f))
        self._box.close()

    def read(self):
        """Return (frame_id, t_ns, frame) or None at end of stream."""
        if not self.latest_only:  # synchronous: every frame, no drops (offline files)
            f = self._grab()
            if f is None:
                return None
            self.stats.grabbed += 1
            self.stats.delivered += 1
            return self.stats.grabbed, time.monotonic_ns(), f
        while True:
            item = self._box.get()
            if item is not None:
                self.stats.delivered += 1
                return item
            if self._eof or not self._run:
                return None

    def stop(self):
        self._run = False
        if self._th:
            self._th.join(timeout=2)
        self._close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()


class OpenCVCamera(Camera):
    """UVC/V4L2 cameras, video files, RTSP. For files use latest_only=False."""

    def __init__(self, src, width=None, height=None, fps=None, fourcc="MJPG", latest_only=True):
        super().__init__(latest_only)
        self.src, self.width, self.height, self.fps, self.fourcc = src, width, height, fps, fourcc
        self.cap = None

    def _open(self):
        if isinstance(self.src, int):
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(self.src)
            if self.fourcc:  # FOURCC must be set before size/fps
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.src!r}")

    def _grab(self):
        ok, f = self.cap.read()
        return f if ok else None

    def _close(self):
        if self.cap is not None:
            self.cap.release()


class PylonCamera(Camera):
    """Basler via pypylon. Request Mono8/Bayer8 (never RGB8: 3x bandwidth) and a small ROI."""

    def __init__(self, exposure_us=500, fps=None, pixel_format="Mono8", roi=None):
        super().__init__(latest_only=True)
        self.exposure_us, self.fps, self.pixel_format, self.roi = exposure_us, fps, pixel_format, roi

    def _open(self):
        from pypylon import pylon
        self._pylon = pylon
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        c = self.cam
        c.Open()
        c.PixelFormat.Value = self.pixel_format
        if self.roi:
            w, h, x, y = self.roi
            c.Width.Value, c.Height.Value, c.OffsetX.Value, c.OffsetY.Value = w, h, x, y
        c.ExposureTime.Value = self.exposure_us
        if self.fps:
            c.AcquisitionFrameRateEnable.Value = True
            c.AcquisitionFrameRate.Value = self.fps
        c.MaxNumBuffer.Value = 16
        # Driver keeps only the newest buffer -> zero latency backlog.
        c.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._bayer = _bayer_code(self.pixel_format)

    def _grab(self):
        while self.cam.IsGrabbing():
            with self.cam.RetrieveResult(2000, self._pylon.TimeoutHandling_ThrowException) as r:
                if not r.GrabSucceeded():
                    continue  # incomplete frame (e.g. GigE packet loss): skip
                img = r.Array  # view into SDK buffer -> copy/convert before release
                return cv2.cvtColor(img, self._bayer) if self._bayer is not None else img.copy()
        return None

    def _close(self):
        try:
            self.cam.StopGrabbing()
            self.cam.Close()
        except Exception:
            pass


class HarvestersCamera(Camera):
    """Any GenICam camera through a vendor GenTL producer (.cti)."""

    def __init__(self, cti_path: str, pixel_format: str | None = None):
        super().__init__(latest_only=True)
        self.cti, self.pixel_format = cti_path, pixel_format

    def _open(self):
        from harvesters.core import Harvester
        self.h = Harvester()
        self.h.add_file(self.cti)
        self.h.update()
        self.ia = self.h.create(0)
        nm = self.ia.remote_device.node_map
        if self.pixel_format:
            nm.PixelFormat.value = self.pixel_format
        self._bayer = _bayer_code(nm.PixelFormat.value)
        self.ia.start()

    def _grab(self):
        with self.ia.fetch(timeout=2.0) as buf:  # context manager re-queues the buffer
            c = buf.payload.components[0]
            img = c.data.reshape(c.height, c.width)
            return cv2.cvtColor(img, self._bayer) if self._bayer is not None else img.copy()

    def _close(self):
        try:
            self.ia.stop()
            self.ia.destroy()
            self.h.reset()
        except Exception:
            pass


def _bayer_code(fmt: str):
    # GenICam BayerRG8 == OpenCV COLOR_BayerRG2BGR in OpenCV>=4 naming. Verify with a test image.
    return {
        "BayerRG8": cv2.COLOR_BayerRG2BGR, "BayerBG8": cv2.COLOR_BayerBG2BGR,
        "BayerGR8": cv2.COLOR_BayerGR2BGR, "BayerGB8": cv2.COLOR_BayerGB2BGR,
    }.get(fmt)


def open_camera(uri: str, **kw) -> Camera:
    if uri.startswith("pylon://"):
        return PylonCamera(**kw)
    if uri.startswith("genicam://"):
        return HarvestersCamera(uri[len("genicam://"):], **kw)
    if uri.isdigit():
        return OpenCVCamera(int(uri), **kw)
    # files: process every frame by default (deterministic); live streams: latest only
    kw.setdefault("latest_only", uri.startswith(("rtsp://", "http://", "https://")))
    return OpenCVCamera(uri, **kw)
