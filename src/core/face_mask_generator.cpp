
#include <memory>
#include <string>
#include <chrono>
#include <vector>

#include <opencv2/imgproc.hpp> //minMaxLoc、GaussianBlur、dilate、fillConvexPoly 

#include "core/face_landmark_constants.h"
#include "core/face_mask_generator.h"

namespace digital_human {
namespace core {

    struct FaceMaskGenerator::Impl {
        // 放在 Impl 里 —— 得到mask的统计信息 
        static FaceMaskInfo BuildMaskInfo(const cv::Mat& mask) {
            FaceMaskInfo info;
            if (mask.empty()) return info;

            info.width = mask.cols;
            info.height = mask.rows;
            info.channels = mask.channels();
            info.type = mask.type();
            cv::minMaxLoc(mask, &info.min_value, &info.max_value);
            info.non_zero_count = cv::countNonZero(mask);
            return info;
        }

        // 在调用 GaussianBlur 之前把偶数核自动加 1，省得调用方踩坑
        static int NormalizeKernelSize(int size) {
            if (size < 1) return 1;
            if (size % 2 == 0) return size + 1;
            return size;
        }

        // struct FaceMaskResult {
        //     bool success = false;                                      ///< 是否成功
        //     FaceMaskStatus status = FaceMaskStatus::kOpenCvError;      ///< 状态码
        //     std::string error_message;                                  ///< 失败时的错误描述
        //     cv::Mat alpha_mask;         ///< alpha mask（CV_32FC1, 范围 0~1）
        //     FaceMaskInfo info;          ///< mask 统计信息
        //     FaceMaskOptions options_used; ///< 实际使用的选项（可能经过修正）
        //     double time_ms = 0.0;       ///< 耗时（毫秒）
        // };
        FaceMaskResult GenerateMouthMask(const cv::Size &image_size,
            const std::vector<cv::Point> &landmarks, const FaceMaskOptions &options) {
            
            FaceMaskResult fm_res;
            auto start = std::chrono::high_resolution_clock::now();
            // 1.检查 image_size.width > 0 && image_size.height > 0，否则返回 kInvalidImageSize
            if (image_size.width <= 0 || image_size.height <= 0) {
                fm_res.success = false;
                fm_res.status = FaceMaskStatus::kInvalidImageSize;
                fm_res.error_message = "The image size is invalid, image_size.width = " + std::to_string(image_size.width)
                                        + ", image_size.height = " + std::to_string(image_size.height);
                
                return fm_res;
            }
            // 2.检查 landmarks.size() == kFaceLandmarkCount，否则返回 kInvalidLandmarkCount
            // 原图版用 != kFaceLandmarkCount 是因为 FaceDetector 输出固定关键点数量——多一个少一个都不正常
            if (landmarks.size() != kFaceLandmarkCount) {
                fm_res.success = false;
                fm_res.status = FaceMaskStatus::kInvalidLandmarkCount;
                fm_res.error_message = "Expected " + std::to_string(kFaceLandmarkCount)
                    + " landmarks, got " + std::to_string(landmarks.size());
                
                return fm_res;
            }
            // 3.根据 options.mouth_region 取嘴部点（48~59 或 48~67）
            // 后续填白、膨胀、羽化都基于这组点。外轮廓够基础可视化用，完整嘴部（含内轮廓）画出的 mask 在张嘴时更稳定
            std::vector<cv::Point> mouth_pts;
            int pts_start = kMouthLandmarkStart;
            int pts_end = (options.mouth_region == MouthRegionMode::kFullMouth)
                ? kMouthFullLandmarkEnd
                : kMouthOuterLandmarkEnd;  // 是取内外嘴唇还是外嘴唇
            for (int i = pts_start; i <= pts_end; i++) {
                mouth_pts.push_back(landmarks[i]);
            }
            
            // 4.创建 CV_8UC1 全黑画布 cv::Mat::zeros(image_size, CV_8UC1)
            // 即做一张和输入图像同尺寸的，但是像素值与通道都设置为0和CV_8UC1
            cv::Mat mask = cv::Mat::zeros(image_size, CV_8UC1);

            // 5.如果 use_convex_hull：先算凸包 → cv::fillConvexPoly（填凸多边形） 填白；否则直接 cv::fillPoly（填任意多边形） 填白
            if (options.use_convex_hull) {
                // 算凸包，然后把凸包填白
                std::vector<cv::Point> hull;
                cv::convexHull(mouth_pts, hull);    // 根据关键点围成的区域算凸包

                cv::fillConvexPoly(mask, hull, cv::Scalar(255)); // 把凸多边形内部涂成白色 255
            } else {
                // 直接按嘴部关键点的顺序填白多边形（可能是凹的）
                std::vector<std::vector<cv::Point>> pts = {mouth_pts};

                cv::fillPoly(mask, pts, cv::Scalar(255));
            }

            // 6.如果 dilate_radius > 0：cv::dilate 膨胀 —— 即控制嘴部区域向外扩展的像素数
            // 核的形状用椭圆（MORPH_ELLIPSE）——膨胀出来的区域也是圆润的，比方形核更贴近嘴的自然形状
            // 校验膨胀半径和模糊核大小的预设值 非负
            if (options.dilate_radius < 0 || options.blur_kernel_size < 0) {
                fm_res.success = false;
                fm_res.status = FaceMaskStatus::kInvalidMaskParameters;
                fm_res.error_message = "dilate_radius or blur_kernel_size must be >= 0";

                return fm_res;
            }

            if (options.dilate_radius > 0) {
                int kernel_size = options.dilate_radius * 2 + 1;    // 核的边长（奇数才有对称像素点，高斯模糊才能用）
                // 创建一个椭圆的"模板"（核），kernel_size × kernel_size 的椭圆形
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                    cv::Size(kernel_size, kernel_size));

                cv::dilate(mask, mask, kernel);
            }

            // 7.修正 blur_kernel_size 为奇数 → cv::GaussianBlur 羽化
            int kernel_size = NormalizeKernelSize(options.blur_kernel_size);
            if (kernel_size > 0) {
                // 0 —— sigmaX	自动算高斯标准差（传 0 让 OpenCV 根据核大小自己算）
                cv::GaussianBlur(mask, mask, cv::Size(kernel_size, kernel_size), 0);
            }

            // 8.convertTo(CV_32FC1, 1.0/255.0) 转为 0~1 浮点 mask —— 即像素值归一化
            cv::Mat mask_float;
            mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);

            // 9.BuildMaskInfo 填统计信息
            fm_res.info = BuildMaskInfo(mask_float);

            // 返回成功结果
            fm_res.success = true;
            fm_res.status = FaceMaskStatus::kOk;
            fm_res.error_message = "";
            fm_res.alpha_mask = mask_float;
            fm_res.options_used = options;
            fm_res.options_used.blur_kernel_size = kernel_size; // 记录修正后的实际值
            
            auto end = std::chrono::high_resolution_clock::now();
            fm_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            return fm_res;
        }

        FaceMaskResult GenerateAlignedMouthMask(const cv::Size &aligned_size,
            const std::vector<cv::Point2f> &aligned_landmarks, const FaceMaskOptions &options) {

            FaceMaskResult fm_res;
            auto start = std::chrono::high_resolution_clock::now();
            // 1.检查 aligned_size 有效
            if (aligned_size.width <= 0 || aligned_size.height <= 0) {
                fm_res.success = false;
                fm_res.status = FaceMaskStatus::kInvalidImageSize;
                fm_res.error_message = "The aligned image size is invalid, aligned_size.width = " + std::to_string(aligned_size.width)
                                        + ", aligned_size.height = " + std::to_string(aligned_size.height);
                
                return fm_res;
            }

            // 2.检查 aligned_landmarks.size() >= kFaceLandmarkCount
            // 万一 FaceAligner 后续输出了多于标准关键点数量的点，本模块仍然能用前面的标准关键点，不报错
            if (aligned_landmarks.size() < kFaceLandmarkCount) {
                fm_res.success = false;
                fm_res.status = FaceMaskStatus::kInvalidLandmarkCount;
                fm_res.error_message = "Expected at least " + std::to_string(kFaceLandmarkCount)
                    + " landmarks, got " + std::to_string(aligned_landmarks.size());
                
                return fm_res;
            }

            // 2.5 参数校验（dilate_radius 和 blur_kernel_size 不能为负数）
            if (options.dilate_radius < 0 || options.blur_kernel_size < 0) {
                fm_res.success = false;
                fm_res.status = FaceMaskStatus::kInvalidMaskParameters;
                fm_res.error_message = "dilate_radius or blur_kernel_size must be >= 0";
                return fm_res;
            }

            // 3.取嘴部点（48~67，默认 kFullMouth），用 cv::Point2f
            std::vector<cv::Point2f> mouth_pts;
            int pts_start = kMouthLandmarkStart;
            int pts_end = (options.mouth_region == MouthRegionMode::kFullMouth)
                ? kMouthFullLandmarkEnd
                : kMouthOuterLandmarkEnd;  // 是取内外嘴唇还是外嘴唇
            for (int i = pts_start; i <= pts_end; i++) {
                mouth_pts.push_back(aligned_landmarks[i]);
            }

            // 4.将浮点坐标 clamp 到 [0, width-1] / [0, height-1]
            // 把坐标值限制在一个范围内——太小就拉到最小值（0），太大就拉到最大值
            // 为什么需要：FaceAligner 输出的对齐关键点可能是浮点数，
            // 张嘴时部分点可能落在 96×96 图像外面（x < 0 或 x > 95），
            // 直接拿去画多边形会报错或画出奇怪的形状。clamp 在填白前把越界坐标推回图像边界内。
            for (auto& p : mouth_pts) {
                p.x = std::max(0.0f, std::min(p.x, static_cast<float>(aligned_size.width - 1)));
                p.y = std::max(0.0f, std::min(p.y, static_cast<float>(aligned_size.height - 1)));
            }

            // 5.计算嘴部外接矩形 bbox —— 在对齐图中标记处矩形的嘴部范围，便于"清掉外侧区域"这一步做裁剪
            // 用 cv::boundingRect 包住所有嘴部点，得到嘴部的最小外包矩形
            cv::Rect mouth_bbox = cv::boundingRect(mouth_pts);

            // 6.创建 CV_8UC1 全黑画布
            cv::Mat mask = cv::Mat::zeros(aligned_size, CV_8UC1);

            // 7.convexHull + fillConvexPoly 填白
            if (options.use_convex_hull) {
                // 算凸包（浮点），转为整数坐标
                std::vector<cv::Point2f> hull_f;
                cv::convexHull(mouth_pts, hull_f);
                std::vector<cv::Point> hull;
                hull.reserve(hull_f.size());
                for (const auto& p : hull_f) {
                    hull.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
                }
                cv::fillConvexPoly(mask, hull, cv::Scalar(255)); // 把凸多边形内部涂成白色 255
            } else {
                // 直接按嘴部关键点的顺序填白多边形（可能是凹的）
                std::vector<std::vector<cv::Point>> pts;
                std::vector<cv::Point> pts_int;
                for (const auto& p : mouth_pts) {
                    pts_int.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
                }
                pts.push_back(pts_int);
                cv::fillPoly(mask, pts, cv::Scalar(255));
            }

            // 8.dilate 小幅膨胀
            if (options.dilate_radius > 0) {
                int kernel_size = options.dilate_radius * 2 + 1;    // 核的边长（奇数才有对称像素点，高斯模糊才能用）
                // 创建一个椭圆的"模板"（核），kernel_size × kernel_size 的椭圆形
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                    cv::Size(kernel_size, kernel_size));

                cv::dilate(mask, mask, kernel);
            }

            // 9.如果 limit_to_mouth_bbox：按 bbox 扩展范围清掉外侧区域
            // 是什么：膨胀把白区向外推了——可能推到下巴、脸颊。把 bbox 扩一圈后，bbox 外的东西全清零，只留嘴部附近。
            // 为什么：96×96 图很小，膨胀稍微大一点白区就扩散到不应改的地方（比如半边脸）。bbox 限制确保 mask 只在嘴部附近"画饼"。
            if (options.limit_to_mouth_bbox) {
                // 按比例扩展 bbox
                // 左右各扩 bbox 宽度的 55%
                int expand_w = static_cast<int>(mouth_bbox.width * options.bbox_expand_x);
                // 往上扩 bbox 高度的 90%（张嘴时上嘴唇上移需要空间）
                int expand_top = static_cast<int>(mouth_bbox.height * options.bbox_expand_top);
                // 往下扩 bbox 高度的 100%（下巴方向更要留空间）
                int expand_bottom = static_cast<int>(mouth_bbox.height * options.bbox_expand_bottom);

                // 构造扩展后的矩形，与图像边界求交集确保不越界
                cv::Rect expanded_bbox(std::max(0, mouth_bbox.x - expand_w),
                                     std::max(0, mouth_bbox.y - expand_top),
                                     std::min(aligned_size.width, mouth_bbox.width + expand_w * 2),
                                     std::min(aligned_size.height, mouth_bbox.height + expand_top + expand_bottom));
                expanded_bbox &= cv::Rect(0, 0, aligned_size.width, aligned_size.height); // 交集裁剪 —— 与对齐图边界求交，确保 expanded_bbox 不越界
                    
                // 新建一张全黑画布，和原 mask 同尺寸
                cv::Mat limited_mask = cv::Mat::zeros(aligned_size, CV_8UC1);

                // 把原 mask 中 expanded_bbox 矩形区域里的像素，拷到新画布同一位置
                //    矩形外的像素——全黑（0），相当于被清掉了
                mask(expanded_bbox).copyTo(limited_mask(expanded_bbox));

                // 用裁剪后的 mask 替换原来的，后续模糊、归一化都用它
                mask = limited_mask;
            }

            // 10.清边界（border_clear）
            if (options.border_clear > 0) {
                int b = options.border_clear;
                // 前 b 行和后 b 行清零
                mask(cv::Rect(0, 0, mask.cols, b)).setTo(cv::Scalar(0));
                mask(cv::Rect(0, mask.rows - b, mask.cols, b)).setTo(cv::Scalar(0));
                // 前 b 列和后 b 列清零
                mask(cv::Rect(0, 0, b, mask.rows)).setTo(cv::Scalar(0));
                mask(cv::Rect(mask.cols - b, 0, b, mask.rows)).setTo(cv::Scalar(0));
            }

            // 11.GaussianBlur 羽化
            int kernel_size = NormalizeKernelSize(options.blur_kernel_size);
            if (kernel_size > 0) {
                // 0 —— sigmaX	自动算高斯标准差（传 0 让 OpenCV 根据核大小自己算）
                cv::GaussianBlur(mask, mask, cv::Size(kernel_size, kernel_size), 0);
            }

            // 12.再次清边界 - 安全性保证（保守）
            if (options.border_clear > 0) {
                int b = options.border_clear;
                // 前 b 行和后 b 行清零
                mask(cv::Rect(0, 0, mask.cols, b)).setTo(cv::Scalar(0));
                mask(cv::Rect(0, mask.rows - b, mask.cols, b)).setTo(cv::Scalar(0));
                // 前 b 列和后 b 列清零
                mask(cv::Rect(0, 0, b, mask.rows)).setTo(cv::Scalar(0));
                mask(cv::Rect(mask.cols - b, 0, b, mask.rows)).setTo(cv::Scalar(0));
            }
            // 13.convertTo CV_32FC1, 1/255
            cv::Mat mask_float;
            mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);

            // 14.BuildMaskInfo + 返回
            fm_res.info = BuildMaskInfo(mask_float);

            fm_res.success = true;
            fm_res.status = FaceMaskStatus::kOk;
            fm_res.error_message = "";
            fm_res.alpha_mask = mask_float;
            fm_res.options_used = options;
            fm_res.options_used.blur_kernel_size = kernel_size; // 记录修正后的实际值
            
            auto end = std::chrono::high_resolution_clock::now();
            fm_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            return fm_res;
        }

        // 是什么：把单通道 CV_32FC1 的 alpha mask 复制为三通道 CV_32FC3。
        // 为什么：融合公式 generated_bgr × mask + ... 是逐通道乘法，B、G、R 三通道图要对上三通道 mask——每个通道乘以同样的 alpha 值。
        cv::Mat To3ChannelMask(const cv::Mat& alpha_mask) const {
            // 1. 空 mask 直接返回空 Mat
            if (alpha_mask.empty()) return cv::Mat();

            // 2. 统一转为 CV_32FC1：如果输入是 CV_8UC1（0~255），则除以 255 归一化；
            // 如果已是 CV_32FC1（0~1），则直接使用；其他类型返回空
            cv::Mat mask;
            if (alpha_mask.type() == CV_8UC1) {
                alpha_mask.convertTo(mask, CV_32FC1, 1.0 / 255.0);
            } else if (alpha_mask.type() == CV_32FC1) {
                mask = alpha_mask;
            } else {
                return cv::Mat();           // 不支持的类型
            }

            // 3. 将同一个 alpha 通道复制三份，合并为 CV_32FC3
            // B、G、R 三个通道存同一份 alpha 值——逐通道乘法时每个通道按相同比例混合
            std::vector<cv::Mat> channels = {mask, mask, mask};
            cv::Mat mask_3c;
            cv::merge(channels, mask_3c);
            return mask_3c;
        }
    };


    FaceMaskGenerator::FaceMaskGenerator() : pImpl_(std::make_unique<Impl>()) {}
    FaceMaskGenerator::~FaceMaskGenerator() = default;


    FaceMaskGenerator::FaceMaskGenerator(FaceMaskGenerator &&) noexcept = default;
    FaceMaskGenerator& FaceMaskGenerator::operator=(FaceMaskGenerator &&) noexcept = default;

    FaceMaskResult FaceMaskGenerator::GenerateMouthMask(const cv::Size &image_size,
        const std::vector<cv::Point> &landmarks, const FaceMaskOptions &options) {
        
        return pImpl_->GenerateMouthMask(image_size, landmarks, options);
    }

    FaceMaskResult FaceMaskGenerator::GenerateAlignedMouthMask(const cv::Size &aligned_size,
        const std::vector<cv::Point2f> &aligned_landmarks, const FaceMaskOptions &options) {

        return pImpl_->GenerateAlignedMouthMask(aligned_size, aligned_landmarks, options);
    }

    cv::Mat FaceMaskGenerator::To3ChannelMask(const cv::Mat& alpha_mask) const {
        return pImpl_->To3ChannelMask(alpha_mask);
    }

    std::string FaceMaskGenerator::StatusToString(FaceMaskStatus status) {
        switch (status) {
            case FaceMaskStatus::kOk:                            return "Ok";
            case FaceMaskStatus::kInvalidImageSize:              return "InvalidImageSize";
            case FaceMaskStatus::kInvalidLandmarkCount:          return "InvalidLandmarkCount";
            case FaceMaskStatus::kInvalidLandmarkGeometry:       return "InvalidLandmarkGeometry";
            case FaceMaskStatus::kInvalidMaskParameters:         return "InvalidMaskParameters";
            case FaceMaskStatus::kMaskEmpty:                     return "MaskEmpty";
            case FaceMaskStatus::kOpenCvError:                   return "OpenCvError";
            default:                                             return "Unknown";
        }
    }

}   // namespace core
}   // namespace digital_human
