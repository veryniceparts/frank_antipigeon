// TensorRT backend (Jetson GPU / DLA). Uses the name-based I/O API available since TensorRT 8.5
// (JetPack 5.1) and still current in TensorRT 10 (JetPack 6). Engines must be built ON the device:
// scripts/build_trt.py. Compiled only when CMake finds TensorRT (PIGEON_WITH_TENSORRT).
#include "pigeon/models.hpp"

#include <stdexcept>

#ifdef PIGEON_WITH_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

namespace pigeon {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << "\n";
    }
};
Logger gLogger;

size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= size_t(std::max<int64_t>(d.d[i], 1));
    return v;
}

class TensorRTNet : public Net {
public:
    explicit TensorRTNet(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open engine " + path);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_)
            throw std::runtime_error(path + ": cannot deserialize -- engines are tied to the GPU and TensorRT "
                                            "version: rebuild on this device (scripts/build_trt.py)");
        ctx_.reset(engine_->createExecutionContext());
        ck(cudaStreamCreate(&stream_), "cudaStreamCreate");
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* n = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) in_ = n;
            else outs_.push_back(n);
        }
        if (engine_->getTensorDataType(in_.c_str()) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("engine input must be float32 (build it from the *.f32.onnx / exported models)");
        nvinfer1::Dims in = engine_->getTensorShape(in_.c_str());
        maxBatch_ = in.d[0] > 0 ? int(in.d[0])
                                : int(engine_->getProfileShape(in_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX).d[0]);
        nvinfer1::Dims maxIn = in;
        maxIn.d[0] = maxBatch_;
        inBytes_ = volume(maxIn) * sizeof(float);
        ck(cudaMalloc(&dIn_, inBytes_), "cudaMalloc");
        ck(cudaMallocHost(&hIn_, inBytes_), "cudaMallocHost");
        ctx_->setTensorAddress(in_.c_str(), dIn_);
        ctx_->setInputShape(in_.c_str(), maxIn);
        for (const auto& o : outs_) {  // allocate for the max batch
            const size_t b = volume(ctx_->getTensorShape(o.c_str())) * sizeof(float);
            void *d = nullptr, *h = nullptr;
            ck(cudaMalloc(&d, b), "cudaMalloc");
            ck(cudaMallocHost(&h, b), "cudaMallocHost");
            dOut_.push_back(d);
            hOut_.push_back(h);
            ctx_->setTensorAddress(o.c_str(), d);
        }
    }
    ~TensorRTNet() override {
        cudaFree(dIn_);
        cudaFreeHost(hIn_);
        for (void* p : dOut_) cudaFree(p);
        for (void* p : hOut_) cudaFreeHost(p);
        cudaStreamDestroy(stream_);
    }
    std::vector<cv::Mat> forward(const cv::Mat& blob) override {
        const int n = blob.size[0];
        if (n > maxBatch_) throw std::runtime_error("batch larger than the engine's max profile");
        nvinfer1::Dims s = engine_->getTensorShape(in_.c_str());
        s.d[0] = n;
        for (int i = 1; i < s.nbDims; ++i)
            if (s.d[i] < 0) s.d[i] = blob.size[i];
        ctx_->setInputShape(in_.c_str(), s);
        const size_t bytes = blob.total() * sizeof(float);
        std::memcpy(hIn_, blob.ptr<float>(), bytes);
        ck(cudaMemcpyAsync(dIn_, hIn_, bytes, cudaMemcpyHostToDevice, stream_), "H2D");
        if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
        std::vector<cv::Mat> res;
        std::vector<size_t> sizes;
        for (size_t k = 0; k < outs_.size(); ++k) {
            nvinfer1::Dims d = ctx_->getTensorShape(outs_[k].c_str());
            sizes.push_back(volume(d));
            ck(cudaMemcpyAsync(hOut_[k], dOut_[k], sizes.back() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "D2H");
        }
        ck(cudaStreamSynchronize(stream_), "sync");
        for (size_t k = 0; k < outs_.size(); ++k) {
            const int rows = n, cols = int(sizes[k] / size_t(n));
            res.push_back(cv::Mat(rows, cols, CV_32F, hOut_[k]).clone());
        }
        return res;
    }
    int maxBatch() const override { return maxBatch_; }
    std::string name() const override { return "tensorrt"; }

private:
    struct Del { template <class T> void operator()(T* p) const { delete p; } };
    std::unique_ptr<nvinfer1::IRuntime, Del> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, Del> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, Del> ctx_;
    cudaStream_t stream_{};
    std::string in_;
    std::vector<std::string> outs_;
    int maxBatch_ = 1;
    size_t inBytes_ = 0;
    void *dIn_ = nullptr, *hIn_ = nullptr;
    std::vector<void*> dOut_, hOut_;
};

}  // namespace

std::unique_ptr<Net> makeTensorRTNet(const std::string& path) { return std::make_unique<TensorRTNet>(path); }
bool haveTensorRT() { return true; }

}  // namespace pigeon

#else  // ---------------------------------------------------------------- built without TensorRT
namespace pigeon {
std::unique_ptr<Net> makeTensorRTNet(const std::string& path) {
    throw std::runtime_error(path + ": this build has no TensorRT (configure with TensorRT installed, e.g. on JetPack)");
}
bool haveTensorRT() { return false; }
}  // namespace pigeon
#endif
