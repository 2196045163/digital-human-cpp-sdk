

#include <chrono>
#include <filesystem>
#include <fstream>
#include <system_error>

#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "core/face_detector.h"

namespace digital_human {
namespace core {
    struct FaceDetector::Impl {
        dlib::frontal_face_detector detector;   // HOG + SVM 检测器
        dlib::shape_predictor landmarks_predictor;  // 68 个关键点预测器
        bool is_model_loaded = false;               

        Impl() {
            detector = dlib::get_frontal_face_detector();   // 构造时就初始化
        }

        // 加载模型文件
        // struct ModelLoadResult {
        //     bool success = false;
        //     FaceDetectStatus status = FaceDetectStatus::kModelLoadFailed;
        //     std::string error_message;
        //     std::string model_path;
        // };
        ModelLoadResult LoadLandmarkModel(const std::string& model_path) {
            ModelLoadResult mload_res;

            // 空路径 → kModelPathEmpty
            if (model_path.empty()) {
                mload_res.success = false;
                mload_res.status = FaceDetectStatus::kModelPathEmpty;
                mload_res.error_message = "The model path is empty.";
                mload_res.model_path = "";

                return mload_res;
            }

            // 文件不存在 → kModelFileNotFound 依旧使用错误码返回
            std::error_code ec;
            if (!std::filesystem::exists(model_path, ec)) {
                mload_res.success = false;
                mload_res.status = FaceDetectStatus::kModelFileNotFound;
                mload_res.error_message = "The model file not found: " + model_path;
                mload_res.model_path = "";

                return mload_res;
            }

            // 不是普通文件（目录啥的） → kModelFileNotFound
            if (!std::filesystem::is_regular_file(model_path, ec)) {
                mload_res.success = false;
                mload_res.status = FaceDetectStatus::kModelFileNotFound;
                mload_res.error_message = "The model path isn't regular file: " + model_path;
                mload_res.model_path = "";

                return mload_res;
            }

            // dlib::deserialize(...) >> predictor
            // 从 .dat 文件里把 68 点的模型读出来，存到 landmarks_predictor 里
            try {
                dlib::deserialize(model_path) >> landmarks_predictor;
                // 成功 → is_model_loaded=true
                is_model_loaded = true;

                mload_res.success = true;
                mload_res.status = FaceDetectStatus::kOk;
                mload_res.error_message = "";
                mload_res.model_path = model_path;

                return mload_res;
                
            } catch (const dlib::serialization_error& e) { // 捕获 dlib::serialization_error → kModelLoadFailed
                is_model_loaded = false;

                mload_res.success = false;
                mload_res.status = FaceDetectStatus::kModelLoadFailed;
                mload_res.error_message = std::string("Failed to deserialize model: ") + e.what();
                mload_res.model_path = "";

                return mload_res;
            }

        }

        // 检测人脸框 
        // 一张图 = 一个 FaceDetectionResult
        // struct FaceDetectionResult {
        //     bool success = false;
        //     FaceDetectStatus status = FaceDetectStatus::kDlibError;
        //     std::string error_message;
        //     std::vector<FaceBox> faces;
        //     double time_ms = 0.0;
        // };
        FaceDetectionResult Detect(const cv::Mat& image, const FaceDetectOptions& options) {
            // 1. 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();
            // 2. image.empty() → kEmptyImage
            FaceDetectionResult fdetect_res;
            if (image.empty()) {
                fdetect_res.success = false;
                fdetect_res.status = FaceDetectStatus::kEmptyImage;
                fdetect_res.error_message = "The image is empty!";

                return fdetect_res;
            }
            // 3. 检查通道（1→转BGR，3→直接用，4→转BGR，其他→kInvalidImageChannels）
            cv::Mat tmp_img;
            if (image.channels() == 3) {
                tmp_img = image;    // BGR，直接用                       
            } else if (image.channels() == 1) {
                cv::cvtColor(image, tmp_img, cv::COLOR_GRAY2BGR);   // 灰度转 BGR
            } else if (image.channels() == 4) {
                cv::cvtColor(image, tmp_img, cv::COLOR_BGRA2BGR);   // BGRA 转 BGR
            } else {
                // 其它通道数 → 报错
                fdetect_res.success = false;
                fdetect_res.status = FaceDetectStatus::kInvalidImageChannels;
                fdetect_res.error_message = "Unsupported channel count: " + std::to_string(image.channels());

                return fdetect_res;
            }
            // 4. 根据宽度缩放图片（>1600缩小，<800放大）
            cv::Mat process_img;
            float scale = 1.0f;

            if (tmp_img.cols > options.max_width) {
                scale = static_cast<float>(options.max_width) / tmp_img.cols;
                cv::resize(tmp_img, process_img, cv::Size(), scale, scale); // 为什么这里传参传了两个scale
            } else if (tmp_img.cols < options.min_width) {
                scale = static_cast<float>(options.min_width) / tmp_img.cols;
                cv::resize(tmp_img, process_img, cv::Size(), scale, scale); // 第一个 scale 是横向缩放比例（fx），第二个是纵向（fy）,实现等比例缩放
            } else {
                process_img = tmp_img;
            }
            // 5. dlib::cv_image<dlib::bgr_pixel> 包装
            // 就是把 cv::Mat 包一层，让 dlib 能直接读，不拷贝像素数据
            // dlib_img 不持有数据，process_img 被销毁后 dlib_img 就悬空了
            dlib::cv_image<dlib::bgr_pixel> dlib_img(process_img);

            // 6. detector(dlib_img, detector_upsample_times) 检测并返回他检测到底人脸框
            // detector(...) 返回的 dlib::rectangle 的列表——这些人脸框还是 dlib 格式
            std::vector<dlib::rectangle> dets = detector(dlib_img, options.detector_upsample_times);

            // 7. 未检测到 && retry → 用 retry_upsample_times 再试
            if (dets.empty() && options.retry_when_no_face) {
                dets = detector(dlib_img, options.retry_upsample_times);
            }
            // 8. 坐标映射回原图 + 边界裁剪
            // 坐标+长度映射
            for (const auto& d : dets) {
                int x = static_cast<int>(d.left() / scale);
                int y = static_cast<int>(d.top() / scale);
                int w = static_cast<int>(d.width() / scale);
                int h = static_cast<int>(d.height() / scale);

                // 边界裁剪
                if (options.clamp_rect_to_image) {
                    x = std::max(0, x);
                    y = std::max(0, y);
                    w = std::min(image.cols - x, w);
                    h = std::min(image.rows - y, h);
                }

                // 9. 过滤无效框
                if (w > 0 && h > 0) {
                    FaceBox box;
                    box.rect = cv::Rect(x, y, w, h);
                    box.index = static_cast<int>(fdetect_res.faces.size());
                    box.score = 0.0;// 这里0.0是什么意思？ 无效框的得分
                    fdetect_res.faces.push_back(box);
                }
            }
            
            // 10. 设置结果状态，返回 FaceDetectionResult
            if (!fdetect_res.faces.empty()) {
                fdetect_res.success = true;
                fdetect_res.status = FaceDetectStatus::kOk;
            } else {
                fdetect_res.success = true;
                fdetect_res.status = FaceDetectStatus::kNoFace;
            }

            auto end = std::chrono::high_resolution_clock::now();
            fdetect_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            return fdetect_res;
        }

        // // 关键点结果
        // struct LandmarkResult {
        //     bool success = false;
        //     FaceDetectStatus status = FaceDetectStatus::kLandmarkFailed;
        //     std::string error_message;
        //     cv::Rect face_rect;
        //     std::vector<cv::Point> landmarks;
        //     std::vector<cv::Point> mouth_landmarks;
        //     double quality_score = 0.0; // 关键点质量的启发式评分（0.0~1.0）。
        //      项目中是怎么给关键点评分的？
        //     // 脸框宽高 > 64px ？→ 越大分越高
        //     // 68 个点都在图像范围内 ？→ 有越界就扣分
        //     // 嘴部外接矩形有效 ？→ 异常就扣分
        // };
        // 人脸框关键点定位
        LandmarkResult GetLandmarks(const cv::Mat& image, const cv::Rect& face_rect,
            const FaceDetectOptions& options) {

            LandmarkResult flandmark_res;
            // 1. 模型没加载 → kLandmarkModelNotLoaded
            if (!is_model_loaded) {
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kLandmarkModelNotLoaded;
                flandmark_res.error_message = "LandmarkModel not loaded!";

                return flandmark_res;
            }
            // 2. 图像为空 → kEmptyImage
            if (image.empty()) {
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kEmptyImage;
                flandmark_res.error_message = "Image is empty!";

                return flandmark_res;
            }
            // 3. 检查通道 → 必要时转 BGR
            cv::Mat tmp_img;
            if (image.channels() == 3) {
                tmp_img = image;    // BGR，直接用                       
            } else if (image.channels() == 1) {
                cv::cvtColor(image, tmp_img, cv::COLOR_GRAY2BGR);   // 灰度转 BGR
            } else if (image.channels() == 4) {
                cv::cvtColor(image, tmp_img, cv::COLOR_BGRA2BGR);   // BGRA 转 BGR
            } else {
                // 其它通道数 → 报错
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kInvalidImageChannels;
                flandmark_res.error_message = "Unsupported channel count: " + std::to_string(image.channels());

                return flandmark_res;
            }
            // 4. 人脸框无效 (IsValidFaceRect) → kInvalidFaceRect
            if (!IsValidFaceRect(face_rect, cv::Size(image.cols, image.rows))) {
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kInvalidFaceRect;
                flandmark_res.error_message = "Face rectangle is invalid!";

                return flandmark_res;
            }
            // 5. 人脸框太小 (< min_face_size) → kFaceTooSmall
            if (face_rect.width < options.min_face_size || face_rect.height < options.min_face_size) {
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kFaceTooSmall;
                flandmark_res.error_message = "Face is too small!";

                return flandmark_res;
            }
            // 6. cv::Rect → dlib::rectangle（注意 -1）
            // 坐标转换，GetLandmarks() 需要 dlib 格式的人脸框，但从 Detect() 拿到的是 OpenCV 格式
            dlib::rectangle dlib_rect(
                face_rect.x,
                face_rect.y,
                face_rect.x + face_rect.width - 1,
                face_rect.y + face_rect.height - 1
            );

            // 7. 包装 dlib::cv_image<dlib::bgr_pixel>
            dlib::cv_image<dlib::bgr_pixel> dlib_img(tmp_img);

            // 8. landmarks_predictor(dlib_img, dlib_rect)
            // 用已经加载好的 68 点模型，在人脸框里预测关键点位置
            dlib::full_object_detection shape;
            try {
                shape = landmarks_predictor(dlib_img, dlib_rect);
            } catch (const std::exception& e) {
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kDlibError;
                flandmark_res.error_message = std::string("Landmark prediction failed: ") + e.what();

                return flandmark_res;
            }
            // 9. num_parts() != 68 → kLandmarkFailed
            // shape 是预测结果，num_parts() 告诉你它预测出来几个关键点。
            // 68 点模型正常情况下应该返回 68 个。
            // 如果不是 68，说明预测失败了（比如人脸框位置不对、模型不匹配）
            if (shape.num_parts() != 68) {
                flandmark_res.success = false;
                flandmark_res.status = FaceDetectStatus::kLandmarkFailed;
                flandmark_res.error_message = "Expected 68 landmarks, got " + std::to_string(shape.num_parts());
                
                return flandmark_res;
            }

            // 10. 转成 std::vector<cv::Point>
            // 把 dlib 的 68 个点逐个转成 cv::Point，存到 vector 里。后面画图、提嘴部点都只认 cv::Point，不认 dlib 格式
            std::vector<cv::Point> landmarks;
            landmarks.reserve(68);
            for (int i = 0; i < 68; ++i) {
                landmarks.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
            }

            // 11. ExtractMouthLandmarks 提取嘴部点
            flandmark_res.mouth_landmarks = FaceDetector::ExtractMouthLandmarks(landmarks);
            // 12. 计算本次关键点预测的 quality_score（框≥64px + 68点全在界内 → 高分）
            double score = 0.0;
            // 脸框大小检查
            if (face_rect.width >= options.min_face_size) {
                score += 0.3;
            }
            if (face_rect.height >= options.min_face_size) {
                score += 0.3;
            }
            // 检查有没有越界的点
            bool all_inside = true;
            for (const auto& p : landmarks) {
                if (p.x < 0 || p.x >= image.cols || p.y < 0 || p.y >= image.rows) {
                    all_inside = false;
                    break;
                }
            }
            if (all_inside) {
                score += 0.4;
            }
            flandmark_res.quality_score = score;

            // 13. 返回 LandmarkResult
            flandmark_res.success = true;
            flandmark_res.status = FaceDetectStatus::kOk;
            flandmark_res.error_message = "";
            flandmark_res.face_rect = face_rect;
            flandmark_res.landmarks = landmarks;
            
            return flandmark_res;
        }

        // 就是 Detect() + GetLandmarks() 打包成一个调用
        // struct FaceAnalyzeResult {
        //     bool success = false;
        //     FaceDetectStatus status = FaceDetectStatus::kDlibError;
        //     std::string error_message;
        //     FaceDetectionResult detection;
        //     std::vector<LandmarkResult> landmarks;
        //     double time_ms = 0.0; // 这个操作花了多少毫秒，方便对比 Debug vs Release 性能 也方便面试时说 Release 下人脸检测耗时约 XXms
        // };
        FaceAnalyzeResult DetectAndLandmark(const cv::Mat& image, const FaceDetectOptions& options) {
            FaceAnalyzeResult res;
            // 1.调 Detect()，拿到 FaceDetectionResult
            auto start = std::chrono::high_resolution_clock::now();
            res.detection = Detect(image, options);
            // 2.如果 Detect() 失败（success=false），直接返回
            if (res.detection.success == false) {
                res.success = false;
                res.status = res.detection.status; //  Detect() 失败 直接用Detect 的处理信息
                res.error_message = res.detection.error_message;

                return res;
            }
            // 3.如果没检测到脸（kNoFace），返回空关键点列表
            if (res.detection.status == FaceDetectStatus::kNoFace) {
                res.success = true;
                res.status = FaceDetectStatus::kNoFace;
                res.error_message = "No face in the image!";
                res.landmarks = {};

                return res;
            }
            // 4.遍历每个 FaceBox，调 GetLandmarks()
            for (const auto& face : res.detection.faces) {
                // 不管成功失败都保留
                LandmarkResult lm = GetLandmarks(image, face.rect, options);
                res.landmarks.push_back(lm);
            }
            // 填 FaceAnalyzeResult 并返回
            res.success = true;
            res.status = FaceDetectStatus::kOk;
            
            auto end = std::chrono::high_resolution_clock::now();
            res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            return res;
        }
    };

    FaceDetector::FaceDetector() : pImpl_(std::make_unique<Impl>()) {}
    FaceDetector::~FaceDetector() = default;

    FaceDetector::FaceDetector(FaceDetector &&) noexcept = default;
    FaceDetector& FaceDetector::operator=(FaceDetector &&) noexcept = default;

    ModelLoadResult FaceDetector::LoadLandmarkModel(const std::string& model_path) {
        return pImpl_->LoadLandmarkModel(model_path);
    }

    FaceDetectionResult FaceDetector::Detect(const cv::Mat& image, 
                               const FaceDetectOptions& options) {
        return pImpl_->Detect(image, options);
    }

    LandmarkResult FaceDetector::GetLandmarks( const cv::Mat& image, const cv::Rect& face_rect,
        const FaceDetectOptions& options) {
        return pImpl_->GetLandmarks(image, face_rect, options);
    }

    FaceAnalyzeResult FaceDetector::DetectAndLandmark(const cv::Mat& image, const FaceDetectOptions& options) {
        return pImpl_->DetectAndLandmark(image, options);
    }

    bool FaceDetector::IsLandmarkModelLoaded() const {
        return pImpl_->is_model_loaded;
    }

    // 从 68 点里取 48~67 共 20 个点 即嘴部的关键点
    std::vector<cv::Point> FaceDetector::ExtractMouthLandmarks(
        const std::vector<cv::Point>& landmarks) {
        if (landmarks.size() < 68) {
            return {};// 不够68点，返回空
        }

        return std::vector<cv::Point>(landmarks.begin() + 48, landmarks.begin() + 68);
    }

    bool FaceDetector::IsValidFaceRect(const cv::Rect& rect, const cv::Size& image_size) {
        // x、y 不能为负
        if (rect.x < 0 || rect.y < 0) {
            return false;
        }
        // width、height 必须 > 0
        if (rect.width <= 0 || rect.height <= 0) {
            return false;
        }
        // 右边界和下边界不能超出图像宽度和高度
        if (rect.x + rect.width > image_size.width) {
            return false;
        }
        if (rect.y + rect.height > image_size.height) {
            return false;
        }

        return true;
    }

    // 把矩形限制在图像边界内    
    cv::Rect FaceDetector::ClampRect(const cv::Rect& rect, const cv::Size& image_size) {
        int x = std::max(0, rect.x);
        int y = std::max(0, rect.y);
        int w = std::min(image_size.width - x, rect.width);
        int h = std::min(image_size.height - y, rect.height);
        if (w < 0) w = 0;
        if (h < 0) h = 0;
        return cv::Rect(x, y, w, h);
    }

    // enum class FaceDetectStatus {
    //     kOk,
    //     kEmptyImage,
    //     kInvalidImageChannels,
    //     kNoFace,
    //     kFaceTooSmall,
    //     kInvalidFaceRect,
    //     kModelPathEmpty,
    //     kModelFileNotFound,
    //     kModelLoadFailed,
    //     kLandmarkModelNotLoaded,
    //     kLandmarkFailed,
    //     kDlibError
    // };
    std::string FaceDetector::StatusToString(FaceDetectStatus status) {
        switch (status) {
            case FaceDetectStatus::kOk:                     return "Ok";
            case FaceDetectStatus::kEmptyImage:             return "EmptyImage";
            case FaceDetectStatus::kInvalidImageChannels:   return "InvalidImageChannels";
            case FaceDetectStatus::kNoFace:                 return "NoFace";
            case FaceDetectStatus::kFaceTooSmall:           return "FaceTooSmall";
            case FaceDetectStatus::kInvalidFaceRect:        return "InvalidFaceRect";
            case FaceDetectStatus::kModelPathEmpty:         return "ModelPathEmpty";
            case FaceDetectStatus::kModelFileNotFound:      return "ModelFileNotFound";
            case FaceDetectStatus::kModelLoadFailed:        return "ModelLoadFailed";
            case FaceDetectStatus::kLandmarkModelNotLoaded: return "LandmarkModelNotLoaded";
            case FaceDetectStatus::kLandmarkFailed:         return "LandmarkFailed";
            case FaceDetectStatus::kDlibError:              return "DlibError";
            default:                                        return "Unknown";
        }
    }

    


}   // namespace core
}   // digital_human