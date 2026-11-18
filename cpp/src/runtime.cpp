#include "pigeon/runtime.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace pigeon {

// ---------------------------------------------------------------- capture
static bool isNumber(const std::string& s) { return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos; }

bool isUrl(const std::string& s) { return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0; }

FrameSource::FrameSource(const std::string& uri, const SourceOptions& o) : o_(o) {
    std::string u = uri;
    if (u.rfind("usb:", 0) == 0) u = u.substr(4);
    if (isNumber(u)) {
        cap_.open(std::stoi(u), cv::CAP_V4L2);
        if (!cap_.isOpened()) cap_.open(std::stoi(u));
        if (!o.fourcc.empty() && o.fourcc.size() == 4)  // FOURCC before size/fps
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(o.fourcc[0], o.fourcc[1], o.fourcc[2], o.fourcc[3]));
        if (o.width) cap_.set(cv::CAP_PROP_FRAME_WIDTH, o.width);
        if (o.height) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, o.height);
        if (o.fps) cap_.set(cv::CAP_PROP_FPS, o.fps);
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        o_.realtime = false;  // live camera: its own clock
    } else if (u.rfind("gst:", 0) == 0) {
        cap_.open(u.substr(4) + " ! appsink drop=true max-buffers=1 sync=false", cv::CAP_GSTREAMER);
        o_.realtime = false;
    } else {
        cap_.open(u, cv::CAP_FFMPEG);
        if (!cap_.isOpened()) cap_.open(u);
        if (!o.realtime) o_.latestOnly = false;  // offline file: every frame, as fast as possible
    }
    if (!cap_.isOpened()) throw std::runtime_error("cannot open source " + uri);
    const double f = cap_.get(cv::CAP_PROP_FPS);
    srcFps_ = (f >= 1 && f <= 2000) ? f : 30.0;
    if (o_.realtime && o.replayFps > 0) srcFps_ = o.replayFps;
    // the grab thread starts on the first read(): no frames are dropped while models are still loading
}

FrameSource::~FrameSource() { stop(); }

void FrameSource::stop() {
    run_ = false;
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    if (cap_.isOpened()) cap_.release();
}

double FrameSource::grabFps() const {
    const double s = std::chrono::duration<double>(Clock::now() - t0_).count();
    return s > 0 ? grabbed_ / s : 0;
}

bool FrameSource::grabOne(Frame& f) {
    if (n_ == 0) t0_ = Clock::now();
    if (o_.realtime) {
        const double period = 1.0 / srcFps_;
        auto due = [&](long k) { return t0_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(k * period)); };
        while (Clock::now() - due(n_) > std::chrono::duration<double>(2 * period)) {  // decoder behind: skip
            if (!cap_.grab()) return false;
            ++n_;
        }
        if (!cap_.read(f.img)) return false;
        ++n_;
        std::this_thread::sleep_until(due(n_));
    } else {
        if (!cap_.read(f.img) || f.img.empty()) return false;
        ++n_;
    }
    f.t = Clock::now();
    f.id = n_;  // source frame number: files = position in the file (also across skipped frames)
    ++grabbed_;
    return true;
}

void FrameSource::loop() {
    while (run_) {
        Frame f;
        if (!grabOne(f)) break;
        {
            std::lock_guard<std::mutex> lk(mu_);
            slot_ = std::move(f);  // overwrite: consumer always gets the newest frame
            fresh_ = true;
        }
        cv_.notify_one();
    }
    std::lock_guard<std::mutex> lk(mu_);
    eof_ = true;
    cv_.notify_all();
}

bool FrameSource::read(Frame& out) {
    if (!o_.latestOnly) {
        if (!grabOne(out)) return false;
        ++delivered_;
        return true;
    }
    if (!th_.joinable() && run_) th_ = std::thread(&FrameSource::loop, this);
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return fresh_ || eof_ || !run_; });
    if (!fresh_) return false;
    out = std::move(slot_);
    fresh_ = false;
    ++delivered_;
    return true;
}

// ---------------------------------------------------------------- pipeline
Pipeline::Pipeline(std::unique_ptr<Proposer> proposer, std::unique_ptr<CropClassifier> clf, const TrackerParams& tp,
                   bool async, int bayerCode, double pad)
    : prop_(std::move(proposer)), clf_(std::move(clf)), tracker_(tp), async_(async), bayer_(bayerCode), pad_(pad) {
    if (bayer_ >= 0 && prop_->evenScaleRequired() % 2)
        throw std::runtime_error("RAW Bayer frames need an even motion scale (2 or 4)");
    if (async_) th_ = std::thread(&Pipeline::worker, this);
}

Pipeline::~Pipeline() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

void Pipeline::worker() {
    for (;;) {
        std::unique_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || todo_; });
            if (quit_) return;
            job = std::move(todo_);
        }
        const auto t = Clock::now();
        job->p = clf_->classify(job->crops);
        job->names = clf_->lastNames;
        job->ms = msSince(t);  // folded into tm_ by the frame thread (no shared writes)
        std::lock_guard<std::mutex> lk(mu_);
        done_ = std::move(job);
        busy_ = false;
    }
}

void Pipeline::applyResults() {
    std::unique_ptr<Job> job;
    {
        std::lock_guard<std::mutex> lk(mu_);
        job = std::move(done_);
    }
    if (!job) return;
    tm_.classify += job->ms;
    ++tm_.jobs;
    tm_.lagFrames += f_ - job->f;
    tm_.maxLagFrames = std::max(tm_.maxLagFrames, f_ - job->f);
    for (size_t i = 0; i < job->ids.size(); ++i)
        tracker_.feed(job->ids[i], job->p[i], f_, i < job->names.size() ? job->names[i] : "");
}

std::vector<Detection> Pipeline::process(const cv::Mat& frame) {
    auto t0 = Clock::now();
    if (async_) applyResults();
    std::vector<cv::Rect> boxes = prop_->detect(frame);
    auto t1 = Clock::now();
    std::vector<int> due = tracker_.update(boxes, f_);
    auto t2 = Clock::now();
    tm_.propose += std::chrono::duration<double, std::milli>(t1 - t0).count();
    tm_.track += std::chrono::duration<double, std::milli>(t2 - t1).count();
    bool canSubmit = true;
    if (async_) {
        std::lock_guard<std::mutex> lk(mu_);
        canSubmit = !busy_;  // one job in flight; due tracks simply stay due until the worker is free
        if (!canSubmit && !due.empty()) ++tm_.busySkips;
    }
    if (!due.empty() && canSubmit) {
        std::vector<cv::Rect> tb;
        for (int id : due) tb.push_back(tracker_.tracks()[id].box);
        auto crops = squareCrops(frame, tb, clf_->size(), pad_, 4, bayer_);
        tm_.crop += msSince(t2);
        tm_.crops += long(crops.size());
        if (onCrop)
            for (size_t i = 0; i < due.size(); ++i) onCrop(due[i], f_, crops[i]);
        if (async_) {
            auto job = std::make_unique<Job>();
            job->f = f_;
            job->ids = due;
            job->crops = std::move(crops);
            for (int id : due) tracker_.tracks()[id].pending = true;
            {
                std::lock_guard<std::mutex> lk(mu_);
                todo_ = std::move(job);
                busy_ = true;
            }
            cv_.notify_one();
        } else {
            const auto t3 = Clock::now();
            std::vector<float> p = clf_->classify(crops);
            tm_.classify += msSince(t3);
            for (size_t i = 0; i < due.size(); ++i)
                tracker_.feed(due[i], p[i], f_, i < clf_->lastNames.size() ? clf_->lastNames[i] : "");
        }
    }
    std::vector<Detection> dets;
    for (const Track* t : tracker_.active(f_))
        dets.push_back({t->id, t->box, t->p, tracker_.isPigeon(*t), t->name});
    ++f_;
    ++tm_.frames;
    return dets;
}

// ---------------------------------------------------------------- drawing
cv::Mat draw(const cv::Mat& frame, const std::vector<Detection>& dets, bool showAll, double fps,
             const std::string& label, int bayerCode) {
    cv::Mat out;
    if (frame.channels() == 1) cv::cvtColor(frame, out, bayerCode >= 0 ? bayerCode : cv::COLOR_GRAY2BGR);
    else if (frame.channels() == 4) cv::cvtColor(frame, out, cv::COLOR_BGRA2BGR);
    else out = frame.clone();
    for (const auto& d : dets) {
        if (!d.pigeon && !showAll) continue;
        const cv::Scalar col = d.pigeon ? cv::Scalar(0, 0, 255) : cv::Scalar(160, 160, 160);
        cv::rectangle(out, d.box, col, d.pigeon ? 2 : 1);
        std::ostringstream s;
        s << (d.pigeon ? (d.name.empty() ? label : d.name) + " " : "") << std::fixed << std::setprecision(2) << d.p;
        cv::putText(out, s.str(), {d.box.x, std::max(12, d.box.y - 4)}, cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
    }
    if (fps > 0) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(0) << fps << " fps";
        cv::putText(out, s.str(), {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 1, cv::LINE_AA);
    }
    return out;
}

// ---------------------------------------------------------------- yt-dlp
static std::string shellQuote(const std::string& s) {
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
}

static std::string runCapture(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("cannot run: " + cmd);
    while (fgets(buf.data(), int(buf.size()), p)) out += buf.data();
    const int rc = pclose(p);
    if (rc != 0) throw std::runtime_error("command failed (is yt-dlp installed? pip install yt-dlp): " + cmd);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    const auto nl = out.find_last_of('\n');  // last line = the path / URL
    return nl == std::string::npos ? out : out.substr(nl + 1);
}

std::string resolveUrl(const std::string& url, bool stream, const std::string& outDir, int maxHeight) {
    // H.264 MP4 video-only (no audio merge -> no ffmpeg needed), decodable by OpenCV everywhere
    const std::string h = std::to_string(maxHeight);
    const std::string fmt = "bv*[ext=mp4][vcodec^=avc1][height<=" + h + "]/b[ext=mp4][vcodec^=avc1][height<=" + h +
                            "]/bv*[height<=" + h + "]/b[height<=" + h + "]/b";
    if (stream) return runCapture("yt-dlp -q --no-warnings --no-playlist -f " + shellQuote(fmt) + " -g " + shellQuote(url));
    mkdir(outDir.c_str(), 0755);
    return runCapture("yt-dlp -q --no-warnings --no-playlist --no-progress -f " + shellQuote(fmt) + " -o " +
                      shellQuote(outDir + "/%(id)s_%(height)sp.%(ext)s") + " --print after_move:filepath " +
                      shellQuote(url));
}

}  // namespace pigeon
