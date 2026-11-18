#include "pigeon/models.hpp"

#include <opencv2/core/persistence.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace pigeon {

std::unique_ptr<Net> makeTensorRTNet(const std::string& path);  // trt_backend.cpp (or stub)

// ---------------------------------------------------------------- OpenCV DNN backend
class OpenCvNet : public Net {
public:
    OpenCvNet(const std::string& path, bool cuda, int fixedBatch, int threads) : fixed_(fixedBatch), cuda_(cuda) {
        net_ = cv::dnn::readNetFromONNX(path);
        if (cuda) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        }
        if (threads > 0) cv::setNumThreads(threads);
        outNames_ = net_.getUnconnectedOutLayersNames();
    }
    std::vector<cv::Mat> forward(const cv::Mat& blob) override {
        net_.setInput(blob);
        std::vector<cv::Mat> outs;
        net_.forward(outs, outNames_);
        for (auto& o : outs) o = o.clone();  // detach from the network's buffers
        return outs;
    }
    int maxBatch() const override { return fixed_ > 0 ? fixed_ : 64; }
    std::string name() const override { return cuda_ ? "opencv-cuda" : "opencv"; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> outNames_;
    int fixed_;
    bool cuda_;
};

static bool endsWith(const std::string& s, const std::string& e) {
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

std::unique_ptr<Net> loadNet(const std::string& path, const std::string& backend, int fixedBatch, int threads) {
    std::string b = backend;
    if (b == "auto") b = (endsWith(path, ".engine") || endsWith(path, ".plan")) ? "tensorrt" : "opencv";
    if (b == "tensorrt") return makeTensorRTNet(path);
    if (b == "opencv" || b == "opencv-cuda") return std::make_unique<OpenCvNet>(path, b == "opencv-cuda", fixedBatch, threads);
    throw std::runtime_error("unknown backend " + backend);
}

// ---------------------------------------------------------------- sidecar
ModelInfo readModelInfo(const std::string& modelPath, const std::string& target) {
    std::string base = modelPath.substr(0, modelPath.find_last_of('.'));
    // TensorRT engines built from inat_birds.onnx keep the sidecar name of the ONNX
    cv::FileStorage fs(base + ".json", cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) throw std::runtime_error("missing sidecar " + base + ".json (see tools/export_cpp_models.py)");
    ModelInfo m;
    fs["kind"] >> m.kind;
    fs["size"] >> m.size;
    m.rgb = int(fs["rgb"]) != 0;
    if (!fs["gray"].empty()) m.gray = int(fs["gray"]) != 0;
    if (!fs["threshold"].empty()) m.threshold = float(fs["threshold"]);
    if (m.kind == "inat") {
        cv::FileNode t = fs["targets"][target];
        if (t.empty()) throw std::runtime_error("unknown target " + target);
        for (const auto& v : t) m.targets.push_back(int(v));
        cv::FileNode names = fs["names"];
        for (const auto& kv : names) m.names[std::stoi(kv.name())] = std::string(kv);
    } else if (m.kind == "ssd") {
        fs["n_anchors"] >> m.nAnchors;
        fs["n_classes"] >> m.nClasses;
        fs["bird_class"] >> m.birdClass;
        cv::FileNode sc = fs["scales"];
        for (int i = 0; i < 4; ++i) m.scales[i] = float(sc[i]);
        m.score = float(fs["score"]);
        m.nmsIou = float(fs["nms_iou"]);
        std::string a;
        fs["anchors"] >> a;
        const auto slash = modelPath.find_last_of('/');
        m.anchors = (slash == std::string::npos ? std::string() : modelPath.substr(0, slash + 1)) + a;
    }
    return m;
}

cv::Mat makeBlob(const std::vector<cv::Mat>& imgs, bool rgb, bool gray) {
    const int n = int(imgs.size()), h = imgs[0].rows, w = imgs[0].cols;
    const int dims[4] = {n, h, w, 3};
    cv::Mat blob(4, dims, CV_32F);
    for (int i = 0; i < n; ++i) {
        cv::Mat src = imgs[i], tmp;
        if (gray) { cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY); cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR); src = tmp; }
        if (rgb) { cv::cvtColor(src, tmp, cv::COLOR_BGR2RGB); src = tmp; }
        cv::Mat dst(h, w, CV_32FC3, blob.ptr<float>(i));  // view into the blob (NHWC = HxW of 3-vectors)
        src.convertTo(dst, CV_32F);
    }
    return blob;
}

// ---------------------------------------------------------------- classifiers
class BinaryClassifier : public CropClassifier {
public:
    BinaryClassifier(const std::string& path, const ModelInfo& info, const std::string& backend, float thr)
        : info_(info), net_(loadNet(path, backend)), thr_(thr >= 0 ? thr : info.threshold) {}
    std::vector<float> classify(const std::vector<cv::Mat>& crops) override {
        std::vector<float> out;
        lastNames.assign(crops.size(), "");
        for (size_t i = 0; i < crops.size(); i += net_->maxBatch()) {
            std::vector<cv::Mat> part(crops.begin() + i, crops.begin() + std::min(crops.size(), i + net_->maxBatch()));
            cv::Mat p = net_->forward(makeBlob(part, info_.rgb, info_.gray))[0];
            const float* d = p.ptr<float>();
            out.insert(out.end(), d, d + part.size());
        }
        return out;
    }
    int size() const override { return info_.size; }
    float threshold() const override { return thr_; }
    std::string backend() const override { return net_->name(); }

private:
    ModelInfo info_;
    std::unique_ptr<Net> net_;
    float thr_;
};

class INatClassifier : public CropClassifier {
public:
    INatClassifier(const std::string& path, const ModelInfo& info, const std::string& backend, float thr)
        : info_(info), net_(loadNet(path, backend, 1)), thr_(thr >= 0 ? thr : info.threshold) {}
    std::vector<float> classify(const std::vector<cv::Mat>& crops) override {
        std::vector<float> out;
        lastNames.clear();
        for (const auto& c : crops) {  // exported with a fixed batch of 1
            cv::Mat probs = net_->forward(makeBlob({c}, info_.rgb))[0];
            const float* p = probs.ptr<float>();
            float sum = 0, best = -1;
            int bestIdx = -1;
            for (int idx : info_.targets) {
                sum += p[idx];
                if (p[idx] > best) { best = p[idx]; bestIdx = idx; }
            }
            out.push_back(sum);
            auto it = info_.names.find(bestIdx);
            std::string nm = it == info_.names.end() ? "" : it->second;  // "Columba livia (Rock Pigeon)"
            const auto a = nm.find('('), b = nm.find(')');
            lastNames.push_back(a != std::string::npos && b != std::string::npos ? nm.substr(a + 1, b - a - 1) : nm);
        }
        return out;
    }
    int size() const override { return info_.size; }
    float threshold() const override { return thr_; }
    std::string backend() const override { return net_->name(); }

private:
    ModelInfo info_;
    std::unique_ptr<Net> net_;
    float thr_;
};

std::unique_ptr<CropClassifier> loadClassifier(const std::string& path, const std::string& backend, float threshold,
                                               const std::string& target) {
    ModelInfo info = readModelInfo(path, target);
    if (info.kind == "binary") return std::make_unique<BinaryClassifier>(path, info, backend, threshold);
    if (info.kind == "inat") return std::make_unique<INatClassifier>(path, info, backend, threshold);
    throw std::runtime_error(path + ": kind '" + info.kind + "' is not a crop classifier");
}

// ---------------------------------------------------------------- SSD
SsdBirdDetector::SsdBirdDetector(const std::string& path, const std::string& backend, float score, int every)
    : info_(readModelInfo(path)), net_(loadNet(path, backend, 1)), every_(std::max(1, every)) {
    if (info_.kind != "ssd") throw std::runtime_error(path + " is not an ssd model");
    if (score >= 0) info_.score = score;
    anchors_.create(info_.nAnchors, 4, CV_32F);
    std::ifstream f(info_.anchors, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(anchors_.data), anchors_.total() * sizeof(float)))
        throw std::runtime_error("cannot read anchors " + info_.anchors);
}

std::vector<cv::Rect> SsdBirdDetector::detect(const cv::Mat& frame) {
    if (n_++ % every_) return last_;
    cv::Mat bgr = frame, rs;
    if (frame.channels() == 1) cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    else if (frame.channels() == 4) cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
    cv::resize(bgr, rs, {info_.size, info_.size}, 0, 0, cv::INTER_AREA);
    auto outs = net_->forward(makeBlob({rs}, info_.rgb));
    // identify outputs by size: box_encodings (A x 4) and scores (A x C)
    cv::Mat enc = outs[0], sc = outs[1];
    if (enc.total() != size_t(info_.nAnchors) * 4) std::swap(enc, sc);
    const float* e = enc.ptr<float>();
    const float* s = sc.ptr<float>();
    const float* a = anchors_.ptr<float>();
    const float W = float(frame.cols), H = float(frame.rows);
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (int i = 0; i < info_.nAnchors; ++i) {
        const float p = s[i * info_.nClasses + info_.birdClass];
        if (p < info_.score) continue;
        const float* ai = a + 4 * i;  // ycenter, xcenter, h, w
        const float* ei = e + 4 * i;  // ty, tx, th, tw
        const float yc = ei[0] / info_.scales[0] * ai[2] + ai[0];
        const float xc = ei[1] / info_.scales[1] * ai[3] + ai[1];
        const float h = std::exp(ei[2] / info_.scales[2]) * ai[2];
        const float w = std::exp(ei[3] / info_.scales[3]) * ai[3];
        cv::Rect r(cv::Point(int((xc - w / 2) * W), int((yc - h / 2) * H)),
                   cv::Point(int((xc + w / 2) * W), int((yc + h / 2) * H)));
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width >= 12 && r.height >= 12) { boxes.push_back(r); scores.push_back(p); }
    }
    last_.clear();
    lastScores.clear();
    for (int k : suppressNested(boxes, scores, info_.nmsIou)) { last_.push_back(boxes[k]); lastScores.push_back(scores[k]); }
    return last_;
}

std::vector<int> suppressNested(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iouThr,
                                float insideThr) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int x, int y) { return scores[x] > scores[y]; });
    std::vector<int> keep;
    for (int i : order) {
        bool dup = false;
        for (int j : keep) {
            const int inter = (boxes[i] & boxes[j]).area();
            if (inter <= 0) continue;
            const float u = float(boxes[i].area() + boxes[j].area() - inter);
            if (inter / u > iouThr || float(inter) / std::max(1, std::min(boxes[i].area(), boxes[j].area())) > insideThr) {
                dup = true;
                break;
            }
        }
        if (!dup) keep.push_back(i);
    }
    return keep;
}

}  // namespace pigeon
