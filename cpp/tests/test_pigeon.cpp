// Self-contained unit tests (no framework). Run: ctest  or  ./pigeon_tests  (from the repo root,
// so that the optional model tests find models/*.onnx).
#include "pigeon/runtime.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <sys/stat.h>

using namespace pigeon;

static int failures = 0, checks = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        ++checks;                                                                    \
        if (!(c)) { ++failures; std::cerr << __FILE__ << ":" << __LINE__ << " FAILED: " #c "\n"; } \
    } while (0)

static bool exists(const std::string& p) { struct stat s{}; return stat(p.c_str(), &s) == 0; }

static std::vector<std::pair<cv::Mat, cv::Rect>> movingSquare(int n = 60, int W = 320, int H = 240) {
    cv::RNG rng(0);
    std::vector<std::pair<cv::Mat, cv::Rect>> v;
    for (int i = 0; i < n; ++i) {
        cv::Mat f(H, W, CV_8UC3, cv::Scalar::all(180));
        cv::Rect r(20 + i * 4, 100, 24, 24);
        cv::rectangle(f, r, cv::Scalar::all(60), -1);
        cv::Mat noise(H, W, CV_8UC3);
        rng.fill(noise, cv::RNG::UNIFORM, 0, 4);
        v.push_back({f + noise, r});
    }
    return v;
}

static cv::Mat mosaicRG(const cv::Mat& bgr) {  // BGR -> RAW BayerRG (R at 0,0) == cv::COLOR_BayerRG2BGR
    cv::Mat raw(bgr.size(), CV_8U);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);
            raw.at<uchar>(y, x) = (y % 2 == 0) ? (x % 2 == 0 ? p[2] : p[1]) : (x % 2 == 0 ? p[1] : p[0]);
        }
    return raw;
}

static void testMotion(const std::string& method) {
    MotionParams mp;
    mp.scale = 2;
    mp.method = method;
    mp.minArea = 4;
    MotionDetector md(mp);
    int hits = 0, i = 0;
    for (auto& [f, gt] : movingSquare()) {
        auto b = md.detect(f);
        if (i++ > 20)
            for (auto& r : b)
                if (std::abs(r.x + r.width / 2 - (gt.x + 12)) < 20 && std::abs(r.y + r.height / 2 - 112) < 20) { ++hits; break; }
    }
    std::cout << "  motion " << md.method() << ": hits " << hits << "/39\n";
    CHECK(hits > 30);
}

struct FakeClf : CropClassifier {  // "pigeon" = crop contains a dark object
    std::vector<float> classify(const std::vector<cv::Mat>& crops) override {
        std::vector<float> p;
        for (auto& c : crops) { double mn; cv::minMaxLoc(c.reshape(1), &mn); p.push_back(mn < 100 ? 1.f : 0.f); }
        lastNames.assign(crops.size(), "");
        return p;
    }
    int size() const override { return 32; }
    float threshold() const override { return 0.5f; }
    std::string backend() const override { return "fake"; }
};

int main() {
    std::cout << "motion\n";
    for (auto m : {"mog2", "diff", "avg", "auto"}) testMotion(m);

    std::cout << "merge / crops\n";
    {
        auto m = mergeBoxes({{0, 0, 10, 10}, {11, 0, 9, 10}, {100, 100, 10, 10}}, 2);
        CHECK(m.size() == 2);
        CHECK(m[0] == cv::Rect(0, 0, 20, 10));
        cv::Mat f(100, 100, CV_8UC3);
        cv::randu(f, 0, 255);
        auto c = squareCrops(f, {{0, 0, 30, 10}, {90, 90, 10, 10}}, 64);
        CHECK(c.size() == 2 && c[0].size() == cv::Size(64, 64) && c[0].type() == CV_8UC3);
        cv::Mat bgrx;
        cv::cvtColor(f, bgrx, cv::COLOR_BGR2BGRA);
        auto c4 = squareCrops(bgrx, {{0, 0, 30, 10}}, 64);
        CHECK(cv::norm(c4[0], c[0], cv::NORM_INF) == 0);
    }
    {  // crop-only debayer == full-frame debayer (except crop-border demosaicing)
        cv::Mat bgr(240, 320, CV_8UC3);
        for (int y = 0; y < 240; ++y)
            for (int x = 0; x < 320; ++x)
                bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(int(x * 0.8) % 256), uchar(y % 256), uchar(int((x + y) * 0.4) % 256));
        cv::GaussianBlur(bgr, bgr, {7, 7}, 0);
        cv::Mat raw = mosaicRG(bgr), full;
        cv::cvtColor(raw, full, cv::COLOR_BayerRG2BGR);
        std::vector<cv::Rect> boxes{{101, 51, 40, 32}, {0, 0, 20, 30}, {300, 220, 20, 20}};
        auto a = squareCrops(full, boxes, 48), b = squareCrops(raw, boxes, 48, 0.2, 4, cv::COLOR_BayerRG2BGR);
        for (size_t i = 0; i < boxes.size(); ++i) {
            cv::Mat d;
            cv::absdiff(a[i], b[i], d);
            CHECK(cv::mean(d)[0] < 3);
        }
    }

    std::cout << "tracker\n";
    {
        TrackerParams tp;
        tp.nUnsure = 3;
        tp.nSure = 50;
        tp.minHits = 2;
        tp.threshold = 0.6f;
        IoUTracker tr(tp);
        int calls = 0;
        for (int f = 0; f < 30; ++f)
            for (int id : tr.update({{10 + f, 10, 30, 30}}, f)) { tr.feed(id, 0.95f, f); ++calls; }
        CHECK(tr.tracks().size() == 1);
        const Track& t = tr.tracks().begin()->second;
        CHECK(t.hits == 30 && tr.isPigeon(t) && t.p > 0.9f);
        CHECK(calls <= 4);
        // fast small object with IoU < iouThr between frames is still one track (centroid fallback,
        // gate = 1 object size per frame)
        IoUTracker tr2;
        for (int f = 0; f < 10; ++f) tr2.update({{10 + f * 9, 50, 10, 10}}, f);
        CHECK(tr2.tracks().size() == 1);
    }
    {
        auto k = suppressNested({{10, 10, 100, 80}, {30, 25, 65, 55}, {200, 10, 60, 50}, {205, 12, 57, 49}}, {0.9f, 0.95f, 0.5f, 0.4f});
        std::sort(k.begin(), k.end());
        CHECK((k == std::vector<int>{1, 2}));
    }

    std::cout << "pipeline (sync + async)\n";
    for (bool async : {false, true}) {
        MotionParams mp;
        mp.scale = 2;
        mp.method = "mog2";
        mp.minArea = 4;
        Pipeline pipe(std::make_unique<MotionDetector>(mp), std::make_unique<FakeClf>(), TrackerParams{}, async);
        int flagged = 0;
        for (auto& [f, gt] : movingSquare(80)) {
            bool any = false;
            for (auto& d : pipe.process(f)) any |= d.pigeon;
            flagged += any;
        }
        std::cout << "  async=" << async << " flagged frames " << flagged << "/80\n";
        CHECK(flagged > 20);
    }

    if (exists("models/ssd_birds.onnx") && exists("models/inat_birds.onnx")) {
        std::cout << "models (OpenCV DNN)\n";
        auto clf = loadClassifier("models/inat_birds.onnx");
        cv::Mat grey(224, 224, CV_8UC3, cv::Scalar::all(128));
        auto p = clf->classify({grey, grey});
        CHECK(p.size() == 2 && p[0] >= 0 && p[0] < 0.2f);  // a grey square is not a pigeon
        SsdBirdDetector ssd("models/ssd_birds.onnx");
        auto b = ssd.detect(cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(128)));
        CHECK(b.empty());  // and contains no bird
    } else {
        std::cout << "models: skipped (run tools/export_cpp_models.py)\n";
    }

    std::cout << checks - failures << "/" << checks << " checks passed\n";
    return failures ? 1 : 0;
}
