// Capture (latest-frame grabber, real-time replay), pipeline, drawing.
#pragma once

#include "pigeon/models.hpp"

#include <opencv2/videoio.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace pigeon {

// ---------------------------------------------------------------- capture
struct SourceOptions {
    int width = 0, height = 0;   // USB cameras: requested mode (0 = driver default)
    double fps = 0;
    std::string fourcc = "MJPG"; // USB: MJPG reaches the highest fps on most cameras
    bool realtime = false;       // files/URLs: release frames on the video clock (like a live camera)
    double replayFps = 0;        // realtime replay at this rate instead of the file's (simulate a fast camera)
    bool latestOnly = true;      // drop stale frames instead of queueing them
};

struct Frame {
    long id = 0;                 // 1-based source frame number (gaps = skipped/dropped frames)
    Clock::time_point t;         // capture time
    cv::Mat img;
};

// URIs: "0" | "usb:0" (V4L2) | file | http(s) media URL | "gst:<pipeline ending in BGR>"
class FrameSource {
public:
    FrameSource(const std::string& uri, const SourceOptions& o = {});
    ~FrameSource();
    bool read(Frame& out);      // blocks until a newer frame exists; false at end of stream
    void stop();
    double sourceFps() const { return srcFps_; }
    long grabbed() const { return grabbed_; }
    long delivered() const { return delivered_; }
    double grabFps() const;

private:
    void loop();
    bool grabOne(Frame& f);
    cv::VideoCapture cap_;
    SourceOptions o_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    Frame slot_;
    bool fresh_ = false, eof_ = false;
    std::atomic<bool> run_{true};
    std::atomic<long> grabbed_{0}, delivered_{0};
    double srcFps_ = 30;
    Clock::time_point t0_;
    long n_ = 0;
};

// ---------------------------------------------------------------- pipeline
struct Detection {
    int trackId;
    cv::Rect box;
    float p;          // smoothed P(pigeon/dove)
    bool pigeon;      // above threshold for >= minHits frames
    std::string name; // species (teacher)
};

struct Timing {
    double propose = 0, track = 0, crop = 0, classify = 0;
    long frames = 0, crops = 0;
    long jobs = 0, lagFrames = 0, maxLagFrames = 0, busySkips = 0;  // async: result delay in frames
};

class Pipeline {
public:
    // async: classification runs in a worker thread; the frame loop never waits for inference
    // (results are applied when ready, tracks stay "pending" meanwhile). Best for high fps.
    Pipeline(std::unique_ptr<Proposer> proposer, std::unique_ptr<CropClassifier> clf, const TrackerParams& tp,
             bool async = false, int bayerCode = -1, double pad = 0.2);
    ~Pipeline();
    std::vector<Detection> process(const cv::Mat& frame);
    const Timing& timing() const { return tm_; }
    long frameIndex() const { return f_; }
    IoUTracker& tracker() { return tracker_; }
    CropClassifier& classifier() { return *clf_; }
    // for harvesting: crops of the tracks due this frame
    std::function<void(int trackId, long frame, const cv::Mat& crop)> onCrop;

private:
    struct Job { long f; std::vector<int> ids; std::vector<cv::Mat> crops; std::vector<float> p; std::vector<std::string> names; double ms = 0; };
    void worker();
    void applyResults();
    std::unique_ptr<Proposer> prop_;
    std::unique_ptr<CropClassifier> clf_;
    IoUTracker tracker_;
    bool async_;
    int bayer_;
    double pad_;
    long f_ = 0;
    Timing tm_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::unique_ptr<Job> todo_, done_;
    bool busy_ = false, quit_ = false;
};

cv::Mat draw(const cv::Mat& frame, const std::vector<Detection>& dets, bool showAll, double fps,
             const std::string& label = "pigeon/dove", int bayerCode = -1);

// YouTube & co via the yt-dlp executable: returns a local file (download) or a direct media URL (stream)
std::string resolveUrl(const std::string& url, bool stream, const std::string& outDir = "data/youtube",
                       int maxHeight = 720);
bool isUrl(const std::string& s);

}  // namespace pigeon
