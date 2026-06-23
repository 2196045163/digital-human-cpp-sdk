#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

#include "core/image_preprocessor.h"

namespace digital_human {
namespace core {

    enum class AlignmentMode {
        kEyes,
        kFaceRect
    };

    // 修改模型时（如换用 MuseTalk 256x256），修改此处的 target_size 和比例参数
    struct FaceAlignmentOptions {
        int target_size = 96;                    // 对齐输出图的边长（默认 96x96）
        AlignmentMode mode = AlignmentMode::kEyes; // kEyes=双眼对齐, kFaceRect=人脸框对齐（降级方案）
        double eye_distance_ratio = 0.40;        // 对齐后双眼距离占 target_size 的比例（96×0.4=38.4px）
        double eye_center_x_ratio = 0.50;        // 双眼中心在输出图中水平位置：0.5=正中间
        double eye_center_y_ratio = 0.40;        // 双眼中心在输出图中垂直位置：0.4=从上往下40%处
        double padding_ratio = 0.15;             // kFaceRect 模式：人脸框四周扩展的比例
    };

    struct FaceAlignmentResult {
        bool success = false;
        ImagePreprocessStatus status = ImagePreprocessStatus::kWarpFailed;
        std::string error_message = "";
        cv::Mat aligned_face;                  // CV_8UC3, BGR（不是说颜色是BGR，只是表明是三种颜色）, 0~255
        cv::Mat transform;                     // M: original -> aligned
        cv::Mat inverse_transform;             // M_inv: aligned -> original
        cv::Rect source_face_rect;             // 原 人脸框
        std::vector<cv::Point2f> aligned_landmarks; // 对齐后的五官关键点
        double time_ms = 0.0;
    };

    class FaceAligner {
    public:
        FaceAligner();
        ~FaceAligner();

        FaceAligner(const FaceAligner&) = delete;
        FaceAligner& operator=(const FaceAligner&) = delete;
        FaceAligner(FaceAligner&&) noexcept;
        FaceAligner& operator=(FaceAligner&&) noexcept;


        // image:      原图 CV_8UC3 BGR，来自 ImageLoader::LoadFromFile()
        // face_rect:  原图坐标系下的人脸框，来自 FaceDetector::Detect()
        // landmarks:  原图坐标系下的 68 个关键点，来自 FaceDetector::GetLandmarks()
        FaceAlignmentResult Align(
            const cv::Mat& image,
            const cv::Rect& face_rect,
            const std::vector<cv::Point>& landmarks,
            const FaceAlignmentOptions& options = FaceAlignmentOptions()
        );

        // AlignBatch() — 循环调 Align() 批量处理
        std::vector<FaceAlignmentResult> AlignBatch(
            const std::vector<cv::Mat>& images,
            const std::vector<cv::Rect>& face_rects,
            const std::vector<std::vector<cv::Point>>& landmarks_batch,
            const FaceAlignmentOptions& options = FaceAlignmentOptions()
        );

        // TransformLandmarks() — 对每个点做 2x3 矩阵乘法
        // 矩阵可能是 `CV_64F` 或 `CV_32F`。可以像参考源码一样提供安全读取函数，或统一在内部转换为 `CV_64F` 后再计算
        static std::vector<cv::Point2f> TransformLandmarks(
            const std::vector<cv::Point>& landmarks,
            const cv::Mat& transform
        );

        static std::string StatusToString(ImagePreprocessStatus status);

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl_;
    };

} // namespace core
} // namespace digital_human