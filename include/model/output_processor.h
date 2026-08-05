#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "model/model_inference.h"

namespace digital_human {
namespace model {

/// @brief 模型输出转换阶段的语义状态码。
/// @note  与 InferenceStatus 分离：这些错误发生在 pred 已交给 OutputProcessor 后。
enum class OutputProcessStatus {
    kOk,
    kEmptyPrediction,
    kInvalidPredictionShape,
    kUnsupportedPredictionLayout,
    kNonFinitePrediction,
    kRangeViolation,
    kInvalidSharpnessThreshold,
    kAllocationFailed,
    kOpenCvError,
    kUnknownError
};

/// @brief 调用方可选提供的清晰度阈值；不设置时只记录分数、不判定通过与否。
struct OutputProcessOptions {
    std::optional<double> minimum_sharpness;
};

/// @brief 本次转换的可观察诊断，不承担模型质量结论。
struct OutputConversionInfo {
    double conversion_ms = 0.0;
    std::size_t corrected_value_count = 0;
    double sharpness_score = 0.0;
    bool sharpness_threshold_applied = false;
    bool sharpness_passed = true;
};

/// @brief 成功时交给 FaceMaskGenerator / FaceBlender 的完整对齐生成脸。
struct ProcessedModelOutput {
    cv::Mat generated_face_bgr;
    ModelInputMetadata metadata;
    std::uint64_t model_generation = 0;
    OutputConversionInfo conversion_info;
};

/// @brief 输出转换的统一结果；失败时 metadata/generation 仍保留，图像必须为空。
struct OutputProcessResult {
    bool success = false;
    OutputProcessStatus status = OutputProcessStatus::kUnknownError;
    std::string error_message;
    ProcessedModelOutput value;
};

/// @brief 将 guarded ncnn pred 转为 OpenCV BGR 对齐脸的无状态处理器。
/// @note  不推理、不生成 mask、不锐化、不逆仿射、不融合，也不保存上一次结果。
class OutputProcessor {
public:
    OutputProcessResult Convert(
        const InferenceOutput& input,
        const OutputProcessOptions& options = OutputProcessOptions()) const;

    static std::string StatusToString(OutputProcessStatus status);
};

} // namespace model
} // namespace digital_human
