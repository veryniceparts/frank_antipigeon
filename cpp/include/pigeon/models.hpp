// Inference backends + the three model kinds used by the pipeline.
//
// Every model takes float32 NHWC input with pixel values 0..255 and ships a JSON sidecar
// (<model>.json, written by tools/export_cpp_models.py or scripts/build_trt.py):
//   kind: "binary" (trained student, P(pigeon/dove)), "inat" (964-bird teacher), "ssd" (COCO bird boxes)
#pragma once

#include "pigeon/core.hpp"

#include <memory>
#include <string>
#include <vector>

namespace pigeon {

// ---------------------------------------------------------------- backends
class Net {
public:
    virtual ~Net() = default;
    // blob: CV_32F, dims {N, H, W, 3}. Returns one Mat per model output, first dim = N.
    virtual std::vector<cv::Mat> forward(const cv::Mat& blob) = 0;
    virtual int maxBatch() const = 0;  // largest N per call (callers split larger batches)
    virtual std::string name() const = 0;
};

// backend: "auto" (.engine/.plan -> tensorrt, else opencv), "opencv", "opencv-cuda", "tensorrt"
std::unique_ptr<Net> loadNet(const std::string& path, const std::string& backend = "auto", int fixedBatch = 0,
                             int threads = 0);
bool haveTensorRT();

// sidecar: <model>.json next to <model>.onnx/.engine
struct ModelInfo {
    std::string kind;            // binary | inat | ssd
    int size = 0;                // square input size
    bool rgb = false;            // channel order the network expects
    bool gray = false;           // binary only: trained on gray crops (mono camera)
    float threshold = 0.5f;
    std::vector<int> targets;    // inat: class indices summed into P(pigeon/dove)
    std::map<int, std::string> names;
    // ssd
    int nAnchors = 0, nClasses = 0, birdClass = 16;
    float scales[4] = {10, 10, 5, 5};
    float score = 0.3f, nmsIou = 0.45f;
    std::string anchors;
};
ModelInfo readModelInfo(const std::string& modelPath, const std::string& target = "columbidae");

// pack crops (BGR uint8, size x size) into a float NHWC blob
cv::Mat makeBlob(const std::vector<cv::Mat>& imgs, bool rgb, bool gray = false);

// ---------------------------------------------------------------- crop classifiers
class CropClassifier {
public:
    virtual ~CropClassifier() = default;
    virtual std::vector<float> classify(const std::vector<cv::Mat>& crops) = 0;  // P(pigeon/dove)
    virtual int size() const = 0;
    virtual float threshold() const = 0;
    virtual std::string backend() const = 0;
    std::vector<std::string> lastNames;  // per crop, best matching species (teacher only)
};

// path: student *.f32.onnx / *.engine (kind binary) or the iNat teacher (kind inat)
std::unique_ptr<CropClassifier> loadClassifier(const std::string& path, const std::string& backend = "auto",
                                               float threshold = -1, const std::string& target = "columbidae");

// ---------------------------------------------------------------- SSD bird proposals
class SsdBirdDetector : public Proposer {
public:
    SsdBirdDetector(const std::string& path, const std::string& backend = "auto", float score = -1, int every = 1);
    std::vector<cv::Rect> detect(const cv::Mat& frame) override;
    std::vector<float> lastScores;
    std::string backend() const { return net_->name(); }

private:
    ModelInfo info_;
    std::unique_ptr<Net> net_;
    cv::Mat anchors_;  // nAnchors x 4 (ycenter, xcenter, h, w), normalised
    int every_, n_ = 0;
    std::vector<cv::Rect> last_;
};

// greedy NMS that also drops boxes lying mostly inside a better one (SSD returns bird + body)
std::vector<int> suppressNested(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores,
                                float iouThr = 0.45f, float insideThr = 0.7f);

}  // namespace pigeon
