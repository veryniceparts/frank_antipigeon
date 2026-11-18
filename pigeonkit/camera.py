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
    incomplete: int = 0  # GigE frames with lost packets (skipped)
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

    # set by cameras that deliver RAW Bayer frames: cv2.COLOR_Bayer*2BGR code. The pipeline then runs
    # motion on the mosaic and debayers only the crops (see PigeonPipeline(bayer=...)).
    bayer_code = None

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
    """UVC/V4L2 cameras, video files, RTSP, http(s) media URLs.

    realtime=True replays a file/URL at its native fps like a live camera: frames are released on
    the video's clock and, with latest_only, whatever the detector can't keep up with is dropped.
    """

    def __init__(self, src, width=None, height=None, fps=None, fourcc="MJPG", latest_only=True, api=None,
                 realtime=False):
        super().__init__(latest_only or realtime)
        self.src, self.width, self.height, self.fps, self.fourcc = src, width, height, fps, fourcc
        self.api = api
        self.realtime = realtime
        self.cap = None
        self.src_fps = None

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
        elif self.api is not None:
            self.cap = cv2.VideoCapture(self.src, self.api)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            hint = ""
            if self.api == cv2.CAP_GSTREAMER and "GStreamer:                   YES" not in cv2.getBuildInformation():
                hint = (" -- this OpenCV has no GStreamer. On Jetson use JetPack's system OpenCV "
                        "(pip uninstall opencv-python* opencv-contrib-python*)")
            raise RuntimeError(f"cannot open {self.src!r}{hint}")
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = f if 1 <= f <= 1000 else 30.0
        self._t0, self._n = None, 0

    def _grab(self):
        if not self.realtime:
            ok, f = self.cap.read()
            return f if ok else None
        import time as _t
        period = 1.0 / self.src_fps
        if self._t0 is None:
            self._t0 = _t.perf_counter()
        # decoder fell behind the video clock -> skip frames without converting them
        while _t.perf_counter() - (self._t0 + self._n * period) > 2 * period:
            if not self.cap.grab():
                return None
            self._n += 1
        ok, f = self.cap.read()
        if not ok:
            return None
        self._n += 1
        wait = self._t0 + self._n * period - _t.perf_counter()
        if wait > 0:
            _t.sleep(wait)
        return f

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


def _gst(desc: str, **kw) -> Camera:
    """Prefer PyGObject; fall back to OpenCV's GStreamer backend (needs a GStreamer-enabled cv2)."""
    try:  # check the full import: a gi built for another Python version fails only part-way
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
        return GstCamera(desc)
    except (ImportError, ValueError):
        if desc.rstrip().endswith("format=BGRx"):  # OpenCV's appsink wants BGR
            desc += " ! videoconvert ! video/x-raw,format=BGR"
        return OpenCVCamera(desc + " ! appsink drop=true max-buffers=1 sync=false",
                            api=cv2.CAP_GSTREAMER, latest_only=True, **kw)


def _bayer_code(fmt: str):
    # GenICam BayerRG8 == OpenCV COLOR_BayerRG2BGR in OpenCV>=4 naming. Verify with a test image.
    return {
        "BayerRG8": cv2.COLOR_BayerRG2BGR, "BayerBG8": cv2.COLOR_BayerBG2BGR,
        "BayerGR8": cv2.COLOR_BayerGR2BGR, "BayerGB8": cv2.COLOR_BayerGB2BGR,
    }.get(fmt)


class GstCamera(Camera):
    """GStreamer capture through PyGObject (python3-gi, preinstalled on JetPack).

    Independent of how OpenCV was built, so pip's opencv-contrib (CNT background subtractor)
    can be used together with Jetson CSI cameras. ``desc`` is a pipeline WITHOUT the sink;
    it must output video/x-raw with format BGR, BGRx (4-channel, used as-is) or GRAY8.
    """

    def __init__(self, desc: str, **_):
        super().__init__(latest_only=True)
        self.desc = desc

    def _open(self):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        self.Gst = Gst
        Gst.init(None)
        self.pipe = Gst.parse_launch(
            f"{self.desc} ! appsink name=pk_sink emit-signals=false max-buffers=1 drop=true sync=false")
        self.sink = self.pipe.get_by_name("pk_sink")
        if self.pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"cannot start GStreamer pipeline: {self.desc}")

    def _grab(self):
        Gst = self.Gst
        while self._run or not self.latest_only:
            sample = self.sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                if self.sink.get_property("eos"):
                    return None
                continue  # timeout: keep waiting unless stopped
            s = sample.get_caps().get_structure(0)
            w, h = s.get_value("width"), s.get_value("height")
            ch = {"GRAY8": 1, "BGRx": 4, "BGRA": 4}.get(s.get_value("format"), 3)
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                stride = info.size // h  # rows may be padded
                a = np.frombuffer(info.data, np.uint8, count=stride * h).reshape(h, stride)[:, : w * ch]
                return a.reshape(h, w, ch).copy() if ch > 1 else a.copy()
            finally:
                buf.unmap(info)
        return None

    def _close(self):
        try:
            self.pipe.set_state(self.Gst.State.NULL)
        except Exception:
            pass


class AravisCamera(Camera):
    """GigE Vision / USB3 Vision cameras through Aravis (open source, aarch64) -- e.g. Teledyne DALSA
    Genie Nano. Bayer formats are delivered RAW (bayer_code set) unless debayer=True.

    roi = (width, height, offset_x, offset_y). Exposure in microseconds.
    """

    def __init__(self, device_id=None, pixel_format=None, roi=None, exposure_us=None, fps=None,
                 packet_size=None, n_buffers=16, debayer=False, gain_db=None, **_):
        super().__init__(latest_only=True)
        self.device_id, self.pixel_format, self.roi = device_id, pixel_format, roi
        self.exposure_us, self.fps, self.packet_size, self.gain_db = exposure_us, fps, packet_size, gain_db
        self.n_buffers, self.debayer = n_buffers, debayer

    def _open(self):
        import gi
        for v in ("0.8", "0.6"):
            try:
                gi.require_version("Aravis", v)
                break
            except ValueError:
                continue
        from gi.repository import Aravis
        self.Arv = Aravis
        Aravis.update_device_list()
        self.cam = Aravis.Camera.new(self.device_id)  # None -> first camera found
        if self.cam is None:
            raise RuntimeError(f"Aravis camera {self.device_id!r} not found (check: arv-tool-0.8, NIC, firewall)")
        c = self.cam
        try:
            c.stop_acquisition()
        except Exception:
            pass
        if self.pixel_format:
            c.get_device().set_string_feature_value("PixelFormat", self.pixel_format)
        if self.roi:
            w, h, x, y = self.roi
            c.set_region(x, y, w, h)
        try:
            c.set_exposure_time_auto(Aravis.Auto.OFF)  # auto exposure silently caps fps
        except Exception:
            pass
        if self.exposure_us:
            c.set_exposure_time(float(self.exposure_us))
        if self.gain_db is not None:
            c.set_gain(float(self.gain_db))
        if self.fps:
            c.set_frame_rate(float(self.fps))
        if c.is_gv_device():
            if self.packet_size:
                c.gv_set_packet_size(int(self.packet_size))
            elif hasattr(c, "gv_auto_packet_size"):
                c.gv_auto_packet_size()  # largest size that passes the network (needs jumbo frames for 8k)
        c.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        self.pf = c.get_pixel_format_as_string()
        self.bayer = _bayer_code(self.pf)
        if self.bayer is not None and not self.debayer:
            self.bayer_code = self.bayer
        self.stream = c.create_stream(None, None)
        payload = c.get_payload()
        for _ in range(self.n_buffers):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        c.start_acquisition()

    def _grab(self):
        Arv = self.Arv
        while self._run:
            buf = self.stream.timeout_pop_buffer(1_000_000)  # us
            if buf is None:
                continue
            try:
                if buf.get_status() != Arv.BufferStatus.SUCCESS:
                    self.stats.incomplete += 1
                    continue
                w, h = buf.get_image_width(), buf.get_image_height()
                data = buf.get_data()  # bytes copy -> safe to re-queue the buffer
                if self.pf in ("RGB8", "RGB8Packed"):
                    return cv2.cvtColor(np.frombuffer(data, np.uint8, w * h * 3).reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                if self.pf in ("BGR8", "BGR8Packed"):
                    return np.frombuffer(data, np.uint8, w * h * 3).reshape(h, w, 3).copy()
                if self.pf.endswith("16") or self.pf.endswith("12") or self.pf.endswith("10"):
                    img = (np.frombuffer(data, np.uint16, w * h).reshape(h, w) >> (int(self.pf[-2:]) - 8)).astype(np.uint8)
                else:
                    img = np.frombuffer(data, np.uint8, w * h).reshape(h, w)
                if self.bayer is not None and self.debayer:
                    return cv2.cvtColor(img, self.bayer)
                return img.copy() if not img.flags.writeable else img
            finally:
                self.stream.push_buffer(buf)
        return None

    def _close(self):
        try:
            self.cam.stop_acquisition()
        except Exception:
            pass
        # drop references now so Aravis releases GigE control (otherwise the camera stays locked
        # until the heartbeat times out and the next start fails to grab frames for a few seconds)
        self._last_stats = self.network_stats() if getattr(self, "stream", None) is not None else {}
        self.stream = None
        self.cam = None

    def network_stats(self) -> dict:
        """GigE stream counters: completed/failed buffers, underruns (host too slow -> no free buffer),
        missing/resent packets (network problems: MTU, rmem_max, cable, switch)."""
        s = self.stream
        if s is None:
            return getattr(self, "_last_stats", {})
        if hasattr(s, "get_n_infos"):  # Aravis >= 0.8.11
            names = [s.get_info_name(i) for i in range(s.get_n_infos())]
            return {n: s.get_info_uint64_by_name(n) for n in names if n.startswith("n_")}
        return {"statistics": tuple(s.get_statistics())}


def csi_pipeline(sensor_id=0, width=1280, height=720, fps=60, flip=0) -> str:
    """Jetson CSI camera (IMX219/IMX477...) via the Argus ISP. NV12->BGRx runs on the VIC (hardware);
    the pipeline consumes BGRx (4-channel) frames directly, so no CPU videoconvert per frame."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! "
            f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx")


def open_camera(uri: str, **kw) -> Camera:
    if uri.startswith("csi://"):
        # csi://0?w=1280&h=720&fps=120&flip=0   (IMX219: 1280x720@120 is its fastest mode)
        from urllib.parse import parse_qs, urlparse
        u = urlparse(uri)
        q = {k: int(v[0]) for k, v in parse_qs(u.query).items()}
        pipe = csi_pipeline(int(u.netloc or 0), q.get("w", 1280), q.get("h", 720), q.get("fps", 60), q.get("flip", 0))
        return _gst(pipe, **kw)
    if uri.startswith("gst://"):  # any GStreamer pipeline producing BGR/GRAY8 (no sink)
        return _gst(uri[len("gst://"):], **kw)
    if uri.startswith(("aravis://", "dalsa://")):
        # dalsa://<device-id or IP>?pf=BayerRG8&w=1280&h=720&x=0&y=0&exp=800&fps=300&packet=8192&gain=0&debayer=0
        from urllib.parse import parse_qs, unquote, urlparse
        u = urlparse(uri)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        roi = None
        if "w" in q and "h" in q:
            roi = (int(q["w"]), int(q["h"]), int(q.get("x", 0)), int(q.get("y", 0)))
        num = lambda k, f=float: f(q[k]) if k in q else None  # noqa: E731
        return AravisCamera(unquote(u.netloc + u.path) or None, pixel_format=q.get("pf"), roi=roi,
                            exposure_us=num("exp"), fps=num("fps"), packet_size=num("packet", int),
                            gain_db=num("gain"), debayer=q.get("debayer", "0") == "1", **kw)
    if uri.startswith("pylon://"):
        return PylonCamera(**kw)
    if uri.startswith("genicam://"):
        return HarvestersCamera(uri[len("genicam://"):], **kw)
    if uri.isdigit():
        return OpenCVCamera(int(uri), **kw)
    # files: every frame, as fast as possible (deterministic) unless realtime=True;
    # rtsp: live -> latest only; http(s) media URLs (e.g. YouTube via yt-dlp): replay on the video clock
    if uri.startswith(("http://", "https://")):
        kw.setdefault("realtime", True)
    kw.setdefault("latest_only", uri.startswith("rtsp://"))
    return OpenCVCamera(uri, **kw)
