// pigeon: fast moving-object -> pigeon/dove pipeline (C++ runtime).
// Core building blocks: motion proposals, square crops, IoU tracker.
#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace pigeon {

using Clock = std::chrono::steady_clock;
inline double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ---------------------------------------------------------------- motion
struct MotionParams {
    int scale = 4;               // process at 1/scale resolution (must be even for RAW Bayer input)
    double fps = 100;            // camera rate -> background memory
    std::string method = "auto"; // auto | cnt | mog2 | avg | diff   (auto: cnt if opencv_contrib, else mog2)
    int minArea = 12;            // px on the downscaled mask
    double maxAreaFrac = 0.2;    // larger blobs = lighting change / camera shake -> ignored
    int mergeGap = 2;            // px (downscaled): merge boxes closer than this (flapping wings)
    double varThreshold = 32;    // MOG2 only
};

class Proposer {  // anything that turns a frame into candidate boxes (motion, SSD detector)
public:
    virtual ~Proposer() = default;
    virtual std::vector<cv::Rect> detect(const cv::Mat& frame) = 0;
    virtual int evenScaleRequired() const { return 0; }  // != 0 -> scale used (Bayer check)
};

class MotionDetector : public Proposer {
public:
    explicit MotionDetector(const MotionParams& p = {}, const cv::Mat& ignoreMask = {});
    std::vector<cv::Rect> detect(const cv::Mat& frame) override;
    int evenScaleRequired() const override { return p_.scale; }
    const std::string& method() const { return method_; }
    const cv::Mat& mask() const { return mask_; }

private:
    cv::Mat small(const cv::Mat& frame);
    MotionParams p_;
    std::string method_;
    cv::Ptr<cv::Algorithm> bg_;
    cv::Mat prev_, bgf_, mask_, ignoreFull_, ignore_, kernel_;
    double alpha_ = 0.02;
};

std::vector<cv::Rect> mergeBoxes(std::vector<cv::Rect> boxes, int gap);

// Padded square crops resized to size x size, BGR uint8. Out-of-frame parts: edge replication.
// bayerCode >= 0 (cv::COLOR_Bayer*2BGR): frame is a RAW mosaic and only the crop is debayered.
std::vector<cv::Mat> squareCrops(const cv::Mat& frame, const std::vector<cv::Rect>& boxes, int size,
                                 double pad = 0.2, int minPad = 4, int bayerCode = -1);

// ---------------------------------------------------------------- tracker
struct Track {
    int id = 0;
    cv::Rect box;
    long first = 0, seen = 0;
    int hits = 1;
    float p = 0.5f;      // EMA of P(pigeon/dove)
    int nCls = 0;        // classifier evaluations
    long nextCls = 0;    // frame index of the next scheduled classification
    bool pending = false;// classification in flight (async mode)
    std::string name;    // species (iNat teacher only)
    bool confident() const { return nCls >= 2 && std::abs(p - 0.5f) * 2 > 0.7f; }
};

struct TrackerParams {
    float iouThr = 0.1f;
    int maxAge = 15;
    float alpha = 0.6f;     // EMA weight of the old value
    int nUnsure = 5;        // re-classify every n frames while unsure
    int nSure = 60;         // ... and every n frames once confident
    float centerGate = 1.0f;
    int minHits = 3;
    float threshold = 0.5f;
};

class IoUTracker {
public:
    explicit IoUTracker(const TrackerParams& p = {}) : p_(p) {}
    // Associate boxes of frame f; returns ids of tracks due for classification.
    std::vector<int> update(const std::vector<cv::Rect>& boxes, long f);
    void feed(int id, float prob, long f, const std::string& name = "");
    std::vector<const Track*> active(long f) const;
    bool isPigeon(const Track& t) const { return t.hits >= p_.minHits && t.nCls > 0 && t.p >= p_.threshold; }
    std::map<int, Track>& tracks() { return tracks_; }
    TrackerParams& params() { return p_; }

private:
    TrackerParams p_;
    std::map<int, Track> tracks_;
    int nid_ = 0;
};

float iou(const cv::Rect& a, const cv::Rect& b);

}  // namespace pigeon
