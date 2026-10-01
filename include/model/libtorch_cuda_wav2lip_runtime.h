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
};

struct LibTorchCudaInferenceResult {
    bool success = false;
    LibTorchCudaStatus status = LibTorchCudaStatus::kForwardFailed;
    std::string error_message;
    ProcessedModelOutput output;
    LibTorchCudaInferenceInfo info;
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
