# Teledyne DALSA camera on Jetson Xavier NX

## Which DALSA cameras fit

| family | interface | on Xavier NX |
|---|---|---|
| **Genie Nano** (M = mono, C = colour) | 1 GigE Vision | ✅ the intended camera. Uses the dev kit's Gigabit port |
| Genie Nano-5G / 10G | 5/10 GigE | ⚠️ the dev kit has only 1 GbE. A 5GbE NIC on M.2 is possible but untested; at 1 GbE these cameras fall back to 1 Gbit/s |
| Linea, Piranha (line scan) | GigE / CameraLink | ❌ line scan does not fit motion → crop |
| Falcon / Genie TS with CameraLink (HS) | frame grabber | ❌ needs a PCIe frame grabber (Xtium) |

The rest of this guide assumes a **Genie Nano** over GigE Vision.

## Software path: Aravis

DALSA's Linux SDK (the "GigE-V Framework", which supports ARM) is a C API with no Python bindings.
The kit uses **[Aravis](https://github.com/AravisProject/aravis)** instead. Aravis is an open-source GigE Vision/GenICam
library that runs on aarch64, and Python reaches it through GObject introspection. It talks to the Nano through standard
GenICam features (`PixelFormat`, `Width`/`Height`/`OffsetX`/`OffsetY`, `ExposureTime`, `AcquisitionFrameRate`,
`GevSCPSPacketSize`).

```bash
# JetPack 5 (Ubuntu 20.04) ships Aravis 0.6; 0.8 is recommended (better stream stats, packet socket, auto packet size)
sudo apt install -y libxml2-dev libglib2.0-dev libusb-1.0-0-dev gobject-introspection libgirepository1.0-dev \
  ninja-build gettext libgstreamer-plugins-base1.0-dev
python3 -m pip install --user "meson>=0.57"
git clone -b 0.8.30 --depth 1 https://github.com/AravisProject/aravis && cd aravis
~/.local/bin/meson setup build -Dviewer=disabled -Dgst-plugin=enabled -Dintrospection=enabled -Ddocumentation=disabled
ninja -C build && sudo ninja -C build install && sudo ldconfig
arv-tool-0.8                       # lists cameras, e.g. "Teledyne DALSA-Nano-C1280-S1234567 (192.168.10.20)"
```

The code also accepts Aravis 0.6 from apt (`gir1.2-aravis-0.6`), with less detailed statistics.

## Network setup (once): `sudo bash scripts/gige_tune.sh eth0`

The script does the following:
- Sets a static IP on the camera port.
  The Nano defaults to DHCP, then falls back to link-local addressing (LLA). A static IP or DHCP both work; match the subnet.
- Sets **MTU 9000** (jumbo frames), so a 1280×720 frame takes about 110 packets instead of about 640. Packet count is what costs CPU on the Carmel cores.
  If the port refuses MTU 9000, the script keeps 1500 and Aravis negotiates a smaller packet size.
- Raises `net.core.rmem_max`/`rmem_default` to 32 MB. Socket buffers that are too small cause missing packets under load.
- Runs `setcap cap_net_raw` on python3. Aravis can then use packet sockets, which take less CPU than UDP sockets.
- Power: the Xavier NX dev kit has **no PoE**. Use a PoE injector or the Nano's 12–24 V aux connector.

## Choosing pixel format, ROI and frame rate

The camera only delivers what the link can carry. **1 GigE gives about 115 MB/s**, so the maximum frame rate is roughly
115e6 / (W·H·bytes per pixel). The sensor's own limit for that ROI also applies; see the Nano's datasheet.

| ROI (8-bit Mono/Bayer) | max fps at 1 GbE |
|---|---|
| 1280×1024 | ~88 |
| 1280×720 | ~125 |
| 1280×480 | ~185 |
| 800×600 | ~240 |
| 640×480 | ~375 |
| 640×240 | ~750 |

- **Colour Nano (C):** use **`BayerRG8`** (check the pattern name in `arv-tool-0.8 features`). Never use `RGB8`, which needs 3× the bandwidth.
  The kit passes the **raw mosaic** through the pipeline:
  - Motion detection runs on the mosaic. At an even `--scale`, each mask pixel is an R+G+G+B average.
  - Only the crops that get classified are debayered. An 80 px crop takes about 5 µs, against about 0.25 ms (x86) or ~1 ms (Carmel, estimated) for a full 720p frame.
  - Colour helps the classifier: blue-grey pigeons against brown sparrows and black crows.
- **Mono Nano (M):** use `Mono8` and train with `train.py --gray`.
- **Exposure:** flapping wings blur badly. Use ≤ 1/(4·fps); for example ≤ 800 µs at 300 fps. Raise gain before raising exposure.
  Auto-exposure is switched off, because it silently caps the frame rate.
- **ROI:** make it a horizontal band covering the ledge, roof edge or balcony rail. ROI is the biggest frame-rate multiplier.

## Run

```bash
python3 scripts/run.py --model models/pigeon.engine --fps 300 --scale 4 \
  --source "dalsa://192.168.10.20?pf=BayerRG8&w=1280&h=480&y=240&exp=800&fps=300"
#   dalsa://<device-id or IP>?pf=&w=&h=&x=&y=&exp=<us>&fps=&gain=<dB>&packet=<bytes>&debayer=0|1
#   (aravis:// is an alias; omitting the id takes the first camera found)
```

At the end, `run.py` prints the GigE counters:
- `n_missing_packets` or `n_resent_packets` above zero: a network problem (MTU, rmem_max, cable, or a switch without jumbo frames).
- `n_underruns` above zero: the host is too slow. Lower `fps`, make the ROI smaller, or use a larger `--scale`.
- `cam_dropped`: frames skipped on purpose to stay on the newest frame. This is fine, but if it is a large share of frames,
  processing cannot keep up with the camera rate.

For harvesting training crops, use the same source: `scripts/harvest.py --source "dalsa://..."`. It debayers crops the same way as the live pipeline.

## Tested vs untested

Tested here against **Aravis' fake GigE Vision camera** (`arv-fake-gv-camera-0.8`, real GVCP/GVSP over loopback):
- `dalsa://` URI parsing; ROI, BayerRG8 and exposure applied
- raw-mosaic frames through motion → crop-only debayer → classifier
- stream counters (0 failures, 0 missing packets)
- clean release of camera control between runs
- crop-only debayering matches full-frame debayering (unit test)

Not tested: a real Genie Nano, MTU/rmem/packet-socket tuning, the Xavier NX network port, and sustained 300+ fps.
