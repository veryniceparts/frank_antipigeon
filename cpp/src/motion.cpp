#include "pigeon/core.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#if __has_include(<opencv2/bgsegm.hpp>)
#include <opencv2/bgsegm.hpp>
#define PIGEON_HAVE_CNT 1
#endif

#include <stdexcept>

namespace pigeon {

MotionDetector::MotionDetector(const MotionParams& p, const cv::Mat& ignoreMask) : p_(p), ignoreFull_(ignoreMask) {
    method_ = p.method;
    if (method_ == "auto") {
#ifdef PIGEON_HAVE_CNT
        method_ = "cnt";  // fastest, multi-threaded, tolerant to swaying foliage
#else
        method_ = "mog2";  // core OpenCV fallback, also copes with foliage (~2.5x slower)
#endif
    }
    if (method_ == "cnt") {
#ifdef PIGEON_HAVE_CNT
        bg_ = cv::bgsegm::createBackgroundSubtractorCNT(15, true, int(15 * p.fps), true);
#else
        throw std::runtime_error("CNT needs opencv_contrib (bgsegm); use --method mog2");
#endif
    } else if (method_ == "mog2") {
        bg_ = cv::createBackgroundSubtractorMOG2(500, p.varThreshold, false);
    } else if (method_ == "avg") {
        alpha_ = std::min(0.2, 2.0 / std::max(p.fps, 1.0));  // ~0.5 s time constant
    } else if (method_ != "diff") {
        throw std::runtime_error("unknown motion method " + method_);
    }
    kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
}

cv::Mat MotionDetector::small(const cv::Mat& frame) {
    cv::Mat g;
    if (frame.channels() == 1) g = frame;  // gray or RAW Bayer (even scale averages whole 2x2 cells)
    else cv::cvtColor(frame, g, frame.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    cv::Mat s;
    if (p_.scale > 1) cv::resize(g, s, {g.cols / p_.scale, g.rows / p_.scale}, 0, 0, cv::INTER_LINEAR);
    else s = g.clone();
    cv::GaussianBlur(s, s, {5, 5}, 0);
    return s;
}

std::vector<cv::Rect> MotionDetector::detect(const cv::Mat& frame) {
    cv::Mat s = small(frame), m;
    if (method_ == "diff") {
        if (prev_.empty()) { prev_ = s; return {}; }
        cv::absdiff(s, prev_, m);
        cv::threshold(m, m, 15, 255, cv::THRESH_BINARY);
        prev_ = s;
    } else if (method_ == "avg") {
        if (bgf_.empty()) { s.convertTo(bgf_, CV_32F); return {}; }
        cv::Mat bg8;
        bgf_.convertTo(bg8, CV_8U);
        cv::absdiff(s, bg8, m);
        cv::threshold(m, m, 15, 255, cv::THRESH_BINARY);
        cv::Mat moving, still;
        cv::dilate(m, moving, kernel_, {-1, -1}, 2);
        cv::bitwise_not(moving, still);
        cv::accumulateWeighted(s, bgf_, alpha_, still);          // learn only where nothing moves
        cv::accumulateWeighted(s, bgf_, alpha_ * 0.05, moving);  // slow update elsewhere
    } else {
        bg_.dynamicCast<cv::BackgroundSubtractor>()->apply(s, m);
        if (method_ == "mog2") cv::threshold(m, m, 200, 255, cv::THRESH_BINARY);
    }
    if (!ignoreFull_.empty()) {
        if (ignore_.size() != m.size()) cv::resize(ignoreFull_, ignore_, m.size(), 0, 0, cv::INTER_NEAREST);
        m.setTo(0, ignore_);
    }
    cv::morphologyEx(m, m, cv::MORPH_OPEN, kernel_);
    cv::dilate(m, m, kernel_, {-1, -1}, 2);
    mask_ = m;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    const double maxArea = p_.maxAreaFrac * m.total();
    std::vector<cv::Rect> boxes;
    for (const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        if (r.area() >= p_.minArea && r.area() <= maxArea) boxes.push_back(r);
    }
    boxes = mergeBoxes(std::move(boxes), p_.mergeGap);
    for (auto& r : boxes) r = cv::Rect(r.x * p_.scale, r.y * p_.scale, r.width * p_.scale, r.height * p_.scale);
    return boxes;
}

std::vector<cv::Rect> mergeBoxes(std::vector<cv::Rect> b, int gap) {
    bool changed = true;
    while (changed && b.size() > 1) {
        changed = false;
        std::vector<cv::Rect> out;
        std::vector<bool> used(b.size(), false);
        for (size_t i = 0; i < b.size(); ++i) {
            if (used[i]) continue;
            cv::Rect cur = b[i];
            used[i] = true;
            for (size_t j = i + 1; j < b.size(); ++j) {
                if (used[j]) continue;
                const cv::Rect& o = b[j];
                if (cur.x - gap <= o.br().x && o.x - gap <= cur.br().x && cur.y - gap <= o.br().y &&
                    o.y - gap <= cur.br().y) {
                    cur |= o;
                    used[j] = true;
                    changed = true;
                }
            }
            out.push_back(cur);
        }
        b.swap(out);
    }
    return b;
}

std::vector<cv::Mat> squareCrops(const cv::Mat& frame, const std::vector<cv::Rect>& boxes, int size, double pad,
                                 int minPad, int bayerCode) {
    const int W = frame.cols, H = frame.rows, ch = frame.channels();
    std::vector<cv::Mat> out;
    out.reserve(boxes.size());
    for (const auto& r : boxes) {
        int side = std::max(r.width, r.height);
        side = int(side + 2 * std::max<double>(minPad, pad * side));
        int cx = r.x + r.width / 2, cy = r.y + r.height / 2;
        int a0 = cx - side / 2, b0 = cy - side / 2, a1 = a0 + side, b1 = b0 + side;
        cv::Mat c;
        if (bayerCode >= 0) {
            // even-aligned window keeps the colour-filter phase; debayer only this crop
            int xa = std::max(0, a0) & ~1, ya = std::max(0, b0) & ~1;
            int xb = std::min(W & ~1, xa + std::max(2, (std::min(W, a1) - xa + 1) & ~1));
            int yb = std::min(H & ~1, ya + std::max(2, (std::min(H, b1) - ya + 1) & ~1));
            cv::cvtColor(frame(cv::Range(ya, yb), cv::Range(xa, xb)).clone(), c, bayerCode);
            int l = std::max(0, xa - a0), t = std::max(0, ya - b0), rr = std::max(0, a1 - xb), bb = std::max(0, b1 - yb);
            if (l || t || rr || bb) cv::copyMakeBorder(c, c, t, bb, l, rr, cv::BORDER_REPLICATE);
        } else {
            c = frame(cv::Range(std::max(0, b0), std::min(H, b1)), cv::Range(std::max(0, a0), std::min(W, a1)));
            if (ch == 4) cv::cvtColor(c, c, cv::COLOR_BGRA2BGR);
            if (a0 < 0 || b0 < 0 || a1 > W || b1 > H)
                cv::copyMakeBorder(c, c, std::max(0, -b0), std::max(0, b1 - H), std::max(0, -a0), std::max(0, a1 - W),
                                   cv::BORDER_REPLICATE);
        }
        cv::Mat o;
        cv::resize(c, o, {size, size}, 0, 0, side > size ? cv::INTER_AREA : cv::INTER_LINEAR);
        if (o.channels() == 1) cv::cvtColor(o, o, cv::COLOR_GRAY2BGR);
        out.push_back(o);
    }
    return out;
}

}  // namespace pigeon
