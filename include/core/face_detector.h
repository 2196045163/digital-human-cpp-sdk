#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace digital_human {
namespace core {

    enum class FaceDetectStatus {
        kOk,
        kEmptyImage,
        kInvalidImageChannels,
        kNoFace,
        kFaceTooSmall,
        kInvalidFaceRect,
        kModelPathEmpty,
        kModelFileNotFound,
        kModelLoadFailed,
        kLandmarkModelNotLoaded,
        kLandmarkFailed,
        kDlibError
    };

    // 人脸框检测选项
    struct FaceDetectOptions {
        int max_width = 1600;               // 大图先缩小，避免 dlib 太慢
        int min_width = 800;                // 小图先放大，提高小人脸检出率
        int min_face_size = 64;             // 小于该尺寸时关键点不稳定
        int detector_upsample_times = 0;    // dlib 首次检测的内部上采样次数
        int retry_upsample_times = 1;       // 检测不到时的二次尝试 次数
        bool retry_when_no_face = true;     // 是否开启二次检测
        bool clamp_rect_to_image = true;    // 是否把人脸框裁剪到图像边界内
    };

    // - dlib 默认 HOG 检测器不直接返回常规置信度，所以 `score` 暂时保留为 `0.0`。
    // - 保留 `score` 字段是为了后续替换 RetinaFace、YuNet、MediaPipe、ONNX 检测器时接口不大改。
    // 一张人脸的检测参数
    struct FaceBox {
        cv::Rect rect;          // 大小 cv::Rect: x, y, width, height
        int index = 0;          // 第几个 人脸
        double score = 0.0;     //
    };

    // - `kOk + success=true + faces 非空`：检测成功且找到人脸。
    // - `kNoFace + success=true + faces 为空`：算法正常运行，但没有人脸。这不是程序错误。
    // - `success=false`：输入、模型、dlib 异常等问题，调用方需要处理。
    // 一张图 = 一个 FaceDetectionResult
    struct FaceDetectionResult {
        bool success = false;
        FaceDetectStatus status = FaceDetectStatus::kDlibError;
        std::string error_message;
        std::vector<FaceBox> faces;
        double time_ms = 0.0;
    };

    // 模型加载要检查：
    // - 路径是否为空。
    // - 文件是否存在。
    // - 是否为普通文件。
    // - dlib 是否能 `deserialize`。
    // - 加载失败时不能把 `is_model_loaded` 留成 `true`。
    struct ModelLoadResult {
        bool success = false;
        FaceDetectStatus status = FaceDetectStatus::kModelLoadFailed;
        std::string error_message;
        std::string model_path;
    };

    // 关键点结果
    struct LandmarkResult {
        bool success = false;
        FaceDetectStatus status = FaceDetectStatus::kLandmarkFailed;
        std::string error_message;
        cv::Rect face_rect;
        std::vector<cv::Point> landmarks;
        std::vector<cv::Point> mouth_landmarks;
        double quality_score = 0.0; // 关键点质量的启发式评分（0.0~1.0）。
        // 脸框宽高 > 64px ？→ 越大分越高
        // 68 个点都在图像范围内 ？→ 有越界就扣分
        // 嘴部外接矩形有效 ？→ 异常就扣分
    };

    // 检测和关键点组合结果
    // - `FaceAligner` 可以拿 `landmarks[0].landmarks`。
    // - 嘴部模块可以拿 `landmarks[0].mouth_landmarks`。
    // - 多人脸场景可以由上层选择面积最大的人脸。
    struct FaceAnalyzeResult {
        bool success = false;
        FaceDetectStatus status = FaceDetectStatus::kDlibError;
        std::string error_message;
        FaceDetectionResult detection;
        std::vector<LandmarkResult> landmarks;
        double time_ms = 0.0;  // 操作耗时（毫秒），用于性能诊断
    };

    class FaceDetector {
    public:
        FaceDetector();
        ~FaceDetector();

        FaceDetector(const FaceDetector&) = delete;
        FaceDetector& operator=(const FaceDetector&) = delete;

        FaceDetector(FaceDetector&&) noexcept;
        FaceDetector& operator=(FaceDetector&&) noexcept;

        // 加载 68 点模型文件
        ModelLoadResult LoadLandmarkModel(const std::string& model_path);

        // 检测人脸框
        FaceDetectionResult Detect(
            const cv::Mat& image,
            const FaceDetectOptions& options = FaceDetectOptions()
        );

        // 人脸框关键点定位
        LandmarkResult GetLandmarks(
            const cv::Mat& image,
            const cv::Rect& face_rect,
            const FaceDetectOptions& options = FaceDetectOptions()
        );
        // 就是 Detect() + GetLandmarks() 打包成一个调用
        FaceAnalyzeResult DetectAndLandmark(
            const cv::Mat& image,
            const FaceDetectOptions& options = FaceDetectOptions()
        );

        bool IsLandmarkModelLoaded() const;

        static std::vector<cv::Point> ExtractMouthLandmarks(
            const std::vector<cv::Point>& landmarks
        );

        static bool IsValidFaceRect(
            const cv::Rect& rect,
            const cv::Size& image_size
        );

        // 把矩形限制在图像边界内
        static cv::Rect ClampRect(
            const cv::Rect& rect,
            const cv::Size& image_size
        );

        static std::string StatusToString(FaceDetectStatus status);

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl_;
    };

} // namespace core
} // namespace digital_human