// pigeon_detect: camera / video / YouTube -> real-time pigeon & dove detection (other birds ignored).
//
//   pigeon_detect --source 0 --show                                   USB camera (motion + iNat teacher)
//   pigeon_detect --source 0 --width 640 --height 360 --cam-fps 260 --model models/pigeon.engine --async
//   pigeon_detect --source "https://youtu.be/ID" --find ssd --show     YouTube: download, replay in real time
//   pigeon_detect --source "https://youtu.be/ID" --stream --find ssd   ... or play while streaming
//   pigeon_detect --source clip.mp4 --fast --csv dets.csv              offline: every frame, max speed
//   pigeon_detect --source 0 --harvest data/crops                      collect training crops
//   pigeon_detect --bench --model models/pigeon.f32.onnx               latency per stage
#include "pigeon/runtime.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/stat.h>

using namespace pigeon;

static const char* kUsage = R"(pigeon_detect [options]
 input
  --source S        0 | usb:N | file | http(s)/YouTube URL | gst:<pipeline ending in format=BGR>   [0]
  --stream          URL: play the stream instead of downloading first
  --fast            files: process every frame as fast as possible (default: real-time replay)
  --replay-fps F    files: replay as if from a camera running at F fps (test high-speed behaviour)
  --width W --height H --cam-fps F --fourcc MJPG|YUYV   USB camera mode
 models
  --model M         crop classifier: models/inat_birds.onnx (teacher, default) | trained *.f32.onnx | *.engine
  --target T        teacher only: columbidae (pigeons+doves, default) | pigeon | feral
  --threshold P     decision threshold (default: from the model sidecar)
  --backend B       auto | opencv | opencv-cuda | tensorrt
  --threads N       OpenCV threads (0 = all cores)
 candidates
  --find motion|ssd motion: background subtraction (fixed camera, fastest; default for cameras)
                    ssd: COCO bird detector (moving camera, perched birds; default for URLs)
  --detector D      SSD model [models/ssd_birds.onnx]   --ssd-every N  run SSD every N frames [1]
  --scale N --method auto|cnt|mog2|avg|diff --min-area N --ignore-mask png --mog2-var V
 speed
  --async           classify in a worker thread (frame loop never waits for inference)
 output
  --show            window (refresh capped by --display-fps, default 30)
  --out FILE        annotated video          --csv FILE   detections per frame
  --debug           draw rejected candidates in grey
  --on-pigeon CMD   run shell command when a pigeon/dove is detected (--cooldown seconds, default 5)
  --harvest DIR     save classifier crops (training data)     --max-frames N
  --bench           print per-stage latency and exit
)";

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::stod(it->second); }
    bool flag(const std::string& k) const { return kv.count(k) > 0; }
};

static Args parse(int argc, char** argv) {
    static const char* flags[] = {"--stream", "--fast", "--async", "--show", "--debug", "--bench", "--help", "-h"};
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        bool isFlag = false;
        for (auto f : flags) isFlag |= k == f;
        if (isFlag) a.kv[k] = "1";
        else if (k.rfind("--", 0) == 0 && i + 1 < argc) a.kv[k] = argv[++i];
        else throw std::runtime_error("bad argument " + k);
    }
    return a;
}

static int bench(const Args& a) {
    cv::RNG rng(1);
    const std::string model = a.get("--model", "models/inat_birds.onnx");
    auto clf = loadClassifier(model, a.get("--backend", "auto"), -1, a.get("--target", "columbidae"));
    std::cout << "classifier " << model << " [" << clf->backend() << "], input " << clf->size() << "\n";
    for (int b : {1, 4, 16}) {
        std::vector<cv::Mat> crops(b);
        for (auto& c : crops) { c.create(clf->size(), clf->size(), CV_8UC3); rng.fill(c, cv::RNG::UNIFORM, 0, 255); }
        for (int i = 0; i < 5; ++i) clf->classify(crops);
        const int reps = clf->size() >= 200 ? 20 : 200;
        auto t = Clock::now();
        for (int i = 0; i < reps; ++i) clf->classify(crops);
        const double ms = msSince(t) / reps;
        std::cout << "  batch " << std::setw(2) << b << ": " << std::fixed << std::setprecision(3) << ms << " ms/call, "
                  << ms / b << " ms/crop\n";
    }
    const int W = int(a.num("--width", 1280)), H = int(a.num("--height", 720));
    cv::Mat bg(H, W, CV_8UC3);
    rng.fill(bg, cv::RNG::UNIFORM, 60, 200);
    cv::GaussianBlur(bg, bg, {31, 31}, 0);
    for (std::string method : {"auto", "mog2", "avg"}) {
        for (int scale : {2, 4}) {
            MotionParams mp;
            mp.method = method;
            mp.scale = scale;
            MotionDetector md(mp);
            double tot = 0;
            for (int i = 0; i < 300; ++i) {
                cv::Mat f = bg.clone();
                cv::ellipse(f, {50 + (i * 7) % (W - 150), H / 2}, {40, 25}, 0, 0, 360, {90, 90, 100}, -1);
                auto t = Clock::now();
                md.detect(f);
                if (i >= 50) tot += msSince(t);
            }
            std::cout << "motion " << md.method() << " 1/" << scale << " @" << W << "x" << H << ": " << tot / 250 << " ms/frame\n";
        }
    }
    const std::string det = a.get("--detector", "models/ssd_birds.onnx");
    struct stat st{};
    if (stat(det.c_str(), &st) == 0) {
        SsdBirdDetector ssd(det, a.get("--backend", "auto"));
        for (int i = 0; i < 3; ++i) ssd.detect(bg);
        auto t = Clock::now();
        for (int i = 0; i < 20; ++i) ssd.detect(bg);
        std::cout << "ssd bird detector [" << ssd.backend() << "]: " << msSince(t) / 20 << " ms/frame\n";
    }
    return 0;
}

int main(int argc, char** argv) try {
    Args a = parse(argc, argv);
    if (a.flag("--help") || a.flag("-h")) { std::cout << kUsage; return 0; }
    if (a.num("--threads", 0) > 0) cv::setNumThreads(int(a.num("--threads", 0)));
    if (a.flag("--bench")) return bench(a);

    std::string src = a.get("--source", "0");
    const bool url = isUrl(src);
    if (url) {
        src = resolveUrl(src, a.flag("--stream"));
        std::cout << "source: " << (a.flag("--stream") ? "stream URL" : src) << "\n";
    }
    const bool camera = src.find_first_not_of("0123456789") == std::string::npos || src.rfind("usb:", 0) == 0 ||
                        src.rfind("gst:", 0) == 0;
    SourceOptions so;
    so.width = int(a.num("--width", 0));
    so.height = int(a.num("--height", 0));
    so.fps = a.num("--cam-fps", 0);
    so.fourcc = a.get("--fourcc", "MJPG");
    so.realtime = !camera && !a.flag("--fast");
    so.replayFps = a.num("--replay-fps", 0);
    FrameSource cam(src, so);
    const double fps = camera ? (so.fps > 0 ? so.fps : cam.sourceFps()) : cam.sourceFps();

    // candidates
    const std::string find = a.get("--find", camera ? "motion" : "ssd");
    std::unique_ptr<Proposer> prop;
    if (find == "ssd") {
        prop = std::make_unique<SsdBirdDetector>(a.get("--detector", "models/ssd_birds.onnx"), a.get("--backend", "auto"),
                                                 float(a.num("--ssd-score", -1)), int(a.num("--ssd-every", 1)));
    } else {
        MotionParams mp;
        mp.scale = int(a.num("--scale", 4));
        mp.fps = fps;
        mp.method = a.get("--method", "auto");
        mp.minArea = int(a.num("--min-area", 12));
        mp.varThreshold = a.num("--mog2-var", 32);
        cv::Mat ign;
        if (a.flag("--ignore-mask")) ign = cv::imread(a.get("--ignore-mask"), cv::IMREAD_GRAYSCALE);
        prop = std::make_unique<MotionDetector>(mp, ign);
    }
    auto clf = loadClassifier(a.get("--model", "models/inat_birds.onnx"), a.get("--backend", "auto"),
                              float(a.num("--threshold", -1)), a.get("--target", "columbidae"));
    TrackerParams tp;
    tp.threshold = clf->threshold();
    if (find == "ssd") { tp.nUnsure = 3; tp.nSure = 15; tp.minHits = 2; }  // fewer, steadier boxes
    // re-check intervals are in frames: keep them ~constant in time for high-fps cameras
    if (fps > 100) { tp.nUnsure = int(tp.nUnsure * fps / 100); tp.nSure = int(tp.nSure * fps / 100); tp.maxAge = int(15 * fps / 100); }
    std::cout << "classifier [" << clf->backend() << "] threshold " << clf->threshold() << ", candidates: " << find
              << (a.flag("--async") ? ", async" : "") << ", source fps " << fps << "\n";
    Pipeline pipe(std::move(prop), std::move(clf), tp, a.flag("--async"));

    const std::string harvest = a.get("--harvest");
    if (!harvest.empty()) {
        mkdir(harvest.c_str(), 0755);
        const std::string tag = std::to_string(std::time(nullptr));
        pipe.onCrop = [harvest, tag](int tid, long f, const cv::Mat& c) {
            std::ostringstream n;
            n << harvest << "/cam-" << tag << "_t" << std::setw(5) << std::setfill('0') << tid << "_f" << std::setw(6) << f << ".jpg";
            cv::imwrite(n.str(), c);
        };
    }
    std::ofstream csv;
    if (a.flag("--csv")) { csv.open(a.get("--csv")); csv << "frame,track,x0,y0,x1,y1,p,pigeon,name\n"; }
    cv::VideoWriter writer;
    const bool show = a.flag("--show"), debug = a.flag("--debug");
    const double displayEvery = 1.0 / a.num("--display-fps", 30);
    const std::string onPigeon = a.get("--on-pigeon");
    const double cooldown = a.num("--cooldown", 5);
    const long maxFrames = long(a.num("--max-frames", 0));
    auto lastShow = Clock::now(), lastTrigger = Clock::now() - std::chrono::hours(1), t0 = Clock::now();
    long pigeonFrames = 0;
    double fpsEma = 0, latency = 0;
    Frame fr;
    while (cam.read(fr)) {
        const auto tp0 = Clock::now();
        auto dets = pipe.process(fr.img);
        const double dt = msSince(tp0);
        latency += std::chrono::duration<double, std::milli>(Clock::now() - fr.t).count();
        fpsEma = fpsEma == 0 ? 1000 / std::max(dt, 1e-3) : 0.95 * fpsEma + 0.05 * 1000 / std::max(dt, 1e-3);
        bool any = false;
        for (const auto& d : dets) {
            any |= d.pigeon;
            if (csv.is_open() && (d.pigeon || debug))
                // frame = camera/source frame number (gaps = dropped frames)
                csv << fr.id - 1 << ',' << d.trackId << ',' << d.box.x << ',' << d.box.y << ',' << d.box.br().x
                    << ',' << d.box.br().y << ',' << d.p << ',' << int(d.pigeon) << ",\"" << d.name << "\"\n";
        }
        if (any) {
            ++pigeonFrames;
            if (!onPigeon.empty() && std::chrono::duration<double>(Clock::now() - lastTrigger).count() > cooldown) {
                lastTrigger = Clock::now();
                std::thread([onPigeon] { int rc = std::system(onPigeon.c_str()); (void)rc; }).detach();  // deterrent
            }
        }
        const bool wantShow = show && std::chrono::duration<double>(Clock::now() - lastShow).count() >= displayEvery;
        if (a.flag("--out") || wantShow) {
            cv::Mat vis = draw(fr.img, dets, debug, fpsEma);
            if (a.flag("--out")) {
                if (!writer.isOpened())
                    writer.open(a.get("--out"), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), std::min(fps, 60.0), vis.size());
                writer.write(vis);
            }
            if (wantShow) {
                lastShow = Clock::now();
                cv::imshow("pigeon / dove detection (q = quit)", vis);
                const int k = cv::waitKey(1) & 0xFF;
                if (k == 'q' || k == 27) break;
            }
        }
        if (maxFrames && pipe.frameIndex() >= maxFrames) break;
    }
    cam.stop();
    const Timing& tm = pipe.timing();
    const double n = std::max<long>(1, tm.frames), el = msSince(t0) / 1000;
    std::cout << std::fixed << std::setprecision(3) << "frames " << tm.frames << " in " << el << " s (" << tm.frames / el
              << " fps), source grabbed " << cam.grabbed() << ", dropped " << cam.grabbed() - cam.delivered()
              << ", frames with pigeon/dove " << pigeonFrames << "\n"
              << "per frame: propose " << tm.propose / n << " ms, track " << tm.track / n << " ms, crop " << tm.crop / n
              << " ms, classify " << tm.classify / n << " ms (" << tm.crops << " crops"
              << (a.flag("--async") ? ", in worker thread" : "") << "), capture->result latency " << latency / n << " ms\n";
    if (a.flag("--async"))
        std::cout << "async: " << tm.jobs << " jobs, result delay avg " << double(tm.lagFrames) / std::max<long>(1, tm.jobs)
                  << " / max " << tm.maxLagFrames << " frames, frames where the worker was still busy " << tm.busySkips << "\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n" << kUsage;
    return 1;
}
