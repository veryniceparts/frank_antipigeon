#include "pigeon/core.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace pigeon {

float iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter = (a & b).area();
    return inter > 0 ? float(inter) / float(a.area() + b.area() - inter + 1e-9) : 0.f;
}

std::vector<int> IoUTracker::update(const std::vector<cv::Rect>& boxes, long f) {
    // greedy association on: IoU (score 1+IoU) or, for small fast objects that don't overlap
    // between frames, centroid distance within centerGate * size (score < 1)
    std::vector<int> ids;
    for (auto& kv : tracks_) ids.push_back(kv.first);
    std::vector<std::tuple<float, int, int>> cand;  // score, track idx, box idx
    for (size_t r = 0; r < ids.size(); ++r) {
        const cv::Rect& tb = tracks_[ids[r]].box;
        const float size = float(std::max(std::max(tb.width, tb.height), 8));
        const cv::Point2f tc(tb.x + tb.width / 2.f, tb.y + tb.height / 2.f);
        for (size_t c = 0; c < boxes.size(); ++c) {
            const float m = iou(tb, boxes[c]);
            float score = -1;
            if (m >= p_.iouThr) score = 1 + m;
            else {
                const cv::Point2f bc(boxes[c].x + boxes[c].width / 2.f, boxes[c].y + boxes[c].height / 2.f);
                const float d = float(cv::norm(tc - bc));
                if (d < p_.centerGate * size) score = 1 - d / (size * 4);
            }
            if (score > 0) cand.emplace_back(score, int(r), int(c));
        }
    }
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<bool> tUsed(ids.size(), false), bUsed(boxes.size(), false);
    for (auto& [s, r, c] : cand) {
        if (tUsed[r] || bUsed[c]) continue;
        Track& t = tracks_[ids[r]];
        t.box = boxes[c];
        t.seen = f;
        t.hits++;
        tUsed[r] = bUsed[c] = true;
    }
    for (size_t c = 0; c < boxes.size(); ++c) {
        if (bUsed[c]) continue;
        Track t;
        t.id = nid_++;
        t.box = boxes[c];
        t.first = t.seen = t.nextCls = f;
        tracks_[t.id] = t;
    }
    std::vector<int> due;
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (f - it->second.seen > p_.maxAge) { it = tracks_.erase(it); continue; }
        const Track& t = it->second;
        if (t.seen == f && f >= t.nextCls && !t.pending) due.push_back(t.id);
        ++it;
    }
    return due;
}

void IoUTracker::feed(int id, float prob, long f, const std::string& name) {
    auto it = tracks_.find(id);
    if (it == tracks_.end()) return;  // track expired while its crop was being classified
    Track& t = it->second;
    t.p = t.nCls == 0 ? prob : p_.alpha * t.p + (1 - p_.alpha) * prob;
    t.nCls++;
    t.pending = false;
    if (!name.empty()) t.name = name;
    t.nextCls = f + (t.confident() ? p_.nSure : p_.nUnsure);
}

std::vector<const Track*> IoUTracker::active(long f) const {
    std::vector<const Track*> out;
    for (const auto& kv : tracks_)
        if (kv.second.seen == f) out.push_back(&kv.second);
    return out;
}

}  // namespace pigeon
