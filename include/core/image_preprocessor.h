#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>


namespace digital_human {
namespace core {

    enum class ImagePreprocessStatus {
        kOk,
        kEmptyImage,
        kInvalidImageType,
        kInvalidChannels,
        kInvalidFaceRect,
        kInvalidLandmarkCount,
        kDegenerateEyeGeometry,
        kInvalidTargetSize,
        kInvalidAffineMatrix,
        kWarpFailed,
        kOpenCvError
    };

    enum class ColorOrder {
        kBgr,
        kRgb
    };

    enum class NormalizeMode {
        kNone,
        kZeroToOne,
        kMinusOneToOne
    };

    struct ImagePreprocessOptions {
        cv::Size target_size = {96, 96};         // 输出尺寸
        ColorOrder output_color = ColorOrder::kBgr;  // 输出颜色通道顺序：BGR 或 RGB
        NormalizeMode normalize = NormalizeMode::kZeroToOne; // 归一化模式
    };

    struct ImagePreprocessResult {
        bool success = false;
        ImagePreprocessStatus status = ImagePreprocessStatus::kOpenCvError;
        std::string error_message;
        cv::Mat image;                           // 预处理后的图像
        ColorOrder color_order = ColorOrder::kBgr;   // 实际输出的颜色顺序
        NormalizeMode normalize_mode = NormalizeMode::kNone; // 实际使用的归一化模式
        float observed_min = 0.0f;               // 处理后的实际最小像素值
        float observed_max = 0.0f;               // 处理后的实际最大像素值
        double time_ms = 0.0;                    // 耗时
    };

    class ImagePreprocessor {
    public:
        ImagePreprocessResult Process(
            const cv::Mat& image,
            const ImagePreprocessOptions& options = ImagePreprocessOptions()
        );

        std::vector<ImagePreprocessResult> ProcessBatch(
            const std::vector<cv::Mat>& images,
            const ImagePreprocessOptions& options = ImagePreprocessOptions()
        );
    };

} // namespace core
} // namespace digital_human