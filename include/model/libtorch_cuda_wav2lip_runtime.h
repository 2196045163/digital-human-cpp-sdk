#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "model/input_processor.h"
#include "model/output_processor.h"

namespace digital_human {
namespace model {

enum class LibTorchCudaStatus {
    kOk,
    kCudaUnavailable,
    kModelFileNotFound,
    kModelLoadFailed,
    kModelNotReady,
    kInvalidInput,
    kForwardFailed,
    kInvalidOutput,
    kOutputConversionFailed
};

struct LibTorchCudaLoadResult {
    bool success = false;
    LibTorchCudaStatus status = LibTorchCudaStatus::kModelLoadFailed;
    std::string error_message;
};

struct LibTorchCudaInferenceInfo {
    std::vector<std::int64_t> output_shape;
    bool output_was_cuda = false;
    double gpu_peak_memory_mb = 0.0;
    double gpu_inference_memory_delta_mb = 0.0;
};

/// @brief 单次 LibTorch CUDA 推理的同步分段计时。
struct LibTorchCudaInferenceTiming {
    double h2d_time_ms = 0.0;              ///< CPU tensor 传输到 CUDA 的耗时
    double cuda_forward_time_ms = 0.0;     ///< 已同步的 TorchScript CUDA 前向耗时
    double d2h_time_ms = 0.0;              ///< CUDA output 传回 CPU 并 contiguous 的耗时
    double gpu_backend_total_time_ms = 0.0;///< 输入进入 backend 到 CPU output 可读取的耗时
};

struct LibTorchCudaInferenceResult {
    bool success = false;
    LibTorchCudaStatus status = LibTorchCudaStatus::kForwardFailed;
    std::string error_message;
    ProcessedModelOutput output;
    LibTorchCudaInferenceInfo info;
    LibTorchCudaInferenceTiming timing;
};

/// @brief 独立的 TorchScript CUDA Wav2Lip runtime。
/// @note 直接消费 Wav2LipInputData，不执行人脸、Mel 或归一化预处理。
class LibTorchCudaWav2LipRuntime {
public:
    LibTorchCudaWav2LipRuntime();
    ~LibTorchCudaWav2LipRuntime();

    LibTorchCudaWav2LipRuntime(const LibTorchCudaWav2LipRuntime&) = delete;
    LibTorchCudaWav2LipRuntime& operator=(const LibTorchCudaWav2LipRuntime&) = delete;
    LibTorchCudaWav2LipRuntime(LibTorchCudaWav2LipRuntime&&) noexcept;
    LibTorchCudaWav2LipRuntime& operator=(LibTorchCudaWav2LipRuntime&&) noexcept;

    static bool IsCudaAvailable();

    LibTorchCudaLoadResult Load(const std::filesystem::path& model_path);
    bool IsReady() const;
    LibTorchCudaInferenceResult Infer(const Wav2LipInputData& input) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace model
} // namespace digital_human
