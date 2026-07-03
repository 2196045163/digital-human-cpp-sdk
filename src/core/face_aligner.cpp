
#include <opencv2/imgproc.hpp>  // warpAffine, getRotationMatrix2D, invertAffineTransform
#include <cmath>
#include <memory>
#include <chrono>

#include "core/face_aligner.h"
#include "core/face_landmark_constants.h"

namespace digital_human {
namespace core {

    struct FaceAligner::Impl {
        // 可以放辅助函数，比如 getCenter()
        static cv::Point2f getCenter(const std::vector<cv::Point>& points) {
            float x = 0, y = 0;
            for (const auto& p : points) {
                x += p.x;
                y += p.y;
            }

            return cv::Point2f(x / points.size(), y / points.size());
        }

        FaceAlignmentResult Align(const cv::Mat& image, const cv::Rect& face_rect,
            const std::vector<cv::Point>& landmarks, const FaceAlignmentOptions& options) {
            FaceAlignmentResult fa_res;

            auto start = std::chrono::high_resolution_clock::now();
            // 1. 检查 image 非空、类型/通道符合约定
            if (image.empty()) {
                fa_res.success = false;
                fa_res.status = ImagePreprocessStatus::kEmptyImage;
                fa_res.error_message = "Input image is empty";

                return fa_res;
            }
            
            if (image.type() != CV_8UC3) {
                fa_res.success = false;
                fa_res.status = ImagePreprocessStatus::kInvalidImageType;
                fa_res.error_message = "Expected CV_8UC3, got type=" + std::to_string(image.type());

                return fa_res;
            }
            // 2. 检查 face_rect 有效并与图像边界求交
            // 因为用户传进来的人脸框可能在图像外面，需要把它裁到图像范围内 —— 防范未然
            // (0, 0) 是矩形左上角坐标——图像的起点。image.cols 是图像宽度，image.rows 是高度
            cv::Rect image_bounds(0, 0, image.cols, image.rows);
            cv::Rect clipped_rect = face_rect & image_bounds;   // & = 求交集
            if (clipped_rect.width <= 0 || clipped_rect.height <= 0) {
                fa_res.success = false;
                fa_res.status = ImagePreprocessStatus::kInvalidFaceRect;
                fa_res.error_message = "Face rect is outside image bounds";

                return fa_res;
            }

            // 3. 检查 landmarks.size() == kFaceLandmarkCount
            if (landmarks.size() != kFaceLandmarkCount) {
                fa_res.success = false;
                fa_res.status = ImagePreprocessStatus::kInvalidLandmarkCount;
                fa_res.error_message = "Expected " + std::to_string(kFaceLandmarkCount)
                    + " landmarks, got " + std::to_string(landmarks.size());

                return fa_res;
            }
            // 4. 提取 36~41、42~47 两组眼睛点
            std::vector<cv::Point> eye_a, eye_b;    // 后面按照 x 坐标区分
            for (int i = 36; i <= 41; i++) {
                eye_a.push_back(landmarks[i]);
                eye_b.push_back(landmarks[i + 6]);
            }
            
            // 5. 计算眼睛中心，并按 x 排序
            // 计算每组眼睛的中心
            cv::Point2f center_a(0, 0), center_b(0, 0);
            center_a = getCenter(eye_a);
            center_b = getCenter(eye_b);

            // 按 x 坐标排序：x 小的作为图像左侧眼睛 L，大的作为 R
            cv::Point2f L, R;
            if (center_a.x < center_b.x) {
                L = center_a;
                R = center_b;
            } else {
                L = center_b;
                R = center_a;
            }

            // 6. 计算 angle、current_distance、scale
            // 以下是公式：
            // dx = R.x - L.x
            // dy = R.y - L.y
            // angle = atan2(dy, dx) * 180 / PI
            // current_distance = sqrt(dx*dx + dy*dy)
            // desired_distance = target_size * eye_distance_ratio
            // scale = desired_distance / current_distance
            // eyes_center = (L + R) / 2
            float dx = R.x - L.x;
            float dy = R.y - L.y;
            double angle = std::atan2(dy, dx) * 180.0 / CV_PI;
            double current_dist = std::sqrt(dx * dx + dy * dy);
            // 检查眼距不是 0 或接近 0
            if (current_dist < 1e-6) {
                fa_res.success = false;
                fa_res.status = ImagePreprocessStatus::kDegenerateEyeGeometry;
                fa_res.error_message = "Eye distance is too samll or zero, cannot compute sacle";

                return fa_res;
            }

            double desired_dist = options.target_size * options.eye_distance_ratio;
            double scale = desired_dist / current_dist;
            // struct FaceAlignmentOptions {
            //     int target_size = 96;
            //     AlignmentMode mode = AlignmentMode::kEyes;
            //     double eye_distance_ratio = 0.40;
            //     double eye_center_x_ratio = 0.50;
            //     double eye_center_y_ratio = 0.40;
            //     double padding_ratio = 0.15;
            // };
            cv::Point2f eyes_center = (L + R) / 2;
            // 7. cv::getRotationMatrix2D 得到 M
            cv::Mat M = cv::getRotationMatrix2D(eyes_center, angle, scale);
            
            // 8. 调整 M 的平移项，使双眼中心落到目标位置
            double target_x = options.target_size * options.eye_center_x_ratio;
            double target_y = options.target_size * options.eye_center_y_ratio;
            M.at<double>(0, 2) += (target_x - eyes_center.x);
            M.at<double>(1, 2) += (target_y - eyes_center.y);

            // 9. cv::warpAffine 输出 target_size x target_size
            // 用 M 对原图执行仿射变换，生成 target_size×target_size 的对齐人脸。
            // 插值方式 INTER_CUBIC，边界填充黑色（BORDER_CONSTANT）
            // 用 M 矩阵把原图变换成对齐图
            cv::Mat aligned_face;
            cv::warpAffine(image, aligned_face, M, cv::Size(options.target_size, options.target_size),
                            cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            // 10. cv::invertAffineTransform 得到 M_inv
            cv::Mat M_inv;
            cv::invertAffineTransform(M, M_inv);

            // 11. 使用 M 变换全部 68 点
            std::vector<cv::Point2f> aligned_landmarks = FaceAligner::TransformLandmarks(landmarks, M);

            // 12. 填写结果和耗时
            // struct FaceAlignmentResult {
            //     bool success = false;
            //     ImagePreprocessStatus status = ImagePreprocessStatus::kWarpFailed;
            //     std::string error_message;
            //     cv::Mat aligned_face;                  // CV_8UC3, BGR（不是说颜色是BGR，只是表明是三种颜色）, 0~255
            //     cv::Mat transform;                     // M: original -> aligned
            //     cv::Mat inverse_transform;             // M_inv: aligned -> original
            //     cv::Rect source_face_rect;
            //     std::vector<cv::Point2f> aligned_landmarks;
            //     double time_ms = 0.0;
            // };
            fa_res.success = true;
            fa_res.status = ImagePreprocessStatus::kOk;
            fa_res.error_message = "";
            fa_res.aligned_face = aligned_face;
            fa_res.transform = M;
            fa_res.inverse_transform = M_inv;
            fa_res.source_face_rect = face_rect;
            fa_res.aligned_landmarks = aligned_landmarks;

            auto end = std::chrono::high_resolution_clock::now();
            fa_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            return fa_res;
        }

        // AlignBatch() — 循环调 Align()
        std::vector<FaceAlignmentResult> AlignBatch(const std::vector<cv::Mat>& images,
                const std::vector<cv::Rect>& face_rects,
                const std::vector<std::vector<cv::Point>>& landmarks_batch,
                const FaceAlignmentOptions& options) {
            std::vector<FaceAlignmentResult> results;
            results.reserve(images.size());
            for (size_t i = 0; i < images.size(); i++) {
                results.push_back(Align(images[i], face_rects[i], landmarks_batch[i], options));
            }

            return results;
        }
    };

    // 构造/析构/移动
    FaceAligner::FaceAligner() : pImpl_(std::make_unique<Impl>()) {}
    FaceAligner::~FaceAligner() = default;

    FaceAligner::FaceAligner(FaceAligner&&) noexcept = default;
    FaceAligner& FaceAligner::operator=(FaceAligner&&) noexcept = default;

    // Align() — 统一对齐人脸与关键点
    FaceAlignmentResult FaceAligner::Align(const cv::Mat& image, const cv::Rect& face_rect,
        const std::vector<cv::Point>& landmarks, const FaceAlignmentOptions& options) {

        return pImpl_->Align(image, face_rect, landmarks, options);
    }

    // AlignBatch() — 循环调 Align()
    std::vector<FaceAlignmentResult> FaceAligner::AlignBatch(const std::vector<cv::Mat>& images,
            const std::vector<cv::Rect>& face_rects,
            const std::vector<std::vector<cv::Point>>& landmarks_batch,
            const FaceAlignmentOptions& options) {
        
        return pImpl_->AlignBatch(images, face_rects, landmarks_batch, options);
    }

    // TransformLandmarks() — 对每个点做 2x3 矩阵乘法
    std::vector<cv::Point2f> FaceAligner::TransformLandmarks(const std::vector<cv::Point>& landmarks,
            const cv::Mat& transform) {
        // 矩阵可能是 `CV_64F` 或 `CV_32F`。可以像参考源码一样提供安全读取函数，或统一在内部转换为 `CV_64F` 后再计算
        cv::Mat M;
        if (transform.type() == CV_64F) {
            M = transform;
        } else {
            transform.convertTo(M, CV_64F);
        }

        std::vector<cv::Point2f> res;
        res.reserve(landmarks.size());
        for (const auto &p : landmarks) {
            float x = static_cast<float>(
                M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y + M.at<double>(0, 2));
            float y = static_cast<float>(
                M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y + M.at<double>(1, 2));

            res.push_back(cv::Point2f(x, y));
        }

        return res;
    }

    // StatusToString() — switch 11 个状态码
    std::string FaceAligner::StatusToString(ImagePreprocessStatus status) {
        switch (status) {
            case ImagePreprocessStatus::kOk:                    return "Ok";
            case ImagePreprocessStatus::kEmptyImage:            return "EmptyImage";
            case ImagePreprocessStatus::kInvalidImageType:      return "InvalidImageType";
            case ImagePreprocessStatus::kInvalidChannels:       return "InvalidChannels";
            case ImagePreprocessStatus::kInvalidFaceRect:       return "InvalidFaceRect";
            case ImagePreprocessStatus::kInvalidLandmarkCount:  return "InvalidLandmarkCount";
            case ImagePreprocessStatus::kDegenerateEyeGeometry: return "DegenerateEyeGeometry";
            case ImagePreprocessStatus::kInvalidTargetSize:     return "InvalidTargetSize";
            case ImagePreprocessStatus::kInvalidAffineMatrix:   return "InvalidAffineMatrix";
            case ImagePreprocessStatus::kWarpFailed:            return "WarpFailed";
            case ImagePreprocessStatus::kOpenCvError:           return "OpenCvError";
            default:                                            return "Unknown";
        }
    }

} // namespace core
} // namespace digital_human
