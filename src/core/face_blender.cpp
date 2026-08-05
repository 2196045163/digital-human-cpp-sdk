#include <chrono>              // std::chrono::high_resolution_clock
#include <string>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // cv::GaussianBlur、cv::addWeighted、cv::warpAffine、cv::threshold

#include "core/face_blender.h"

namespace digital_human {
namespace core {

namespace {

    // 内部工具函数（匿名 namespace，仅本文件可见）

    /// @brief 快速构造一个"失败"的 FaceBlendResult
    /// @param status 错误状态码
    /// @param msg    给调用方的错误描述
    /// @param opt    本次使用的选项（用于调试图对比）
    FaceBlendResult MakeErrorResult(FaceBlendStatus status, const std::string& msg,
                                    const FaceBlendOptions& opt) {
        FaceBlendResult fb_res;
        fb_res.success = false;
        fb_res.status = status;
        fb_res.error_message = msg;
        fb_res.options_used = opt;
        return fb_res;
    }

    // /// @brief 检查图像是否为合法的 BGR 彩色图（CV_8UC3）
    // bool IsValidBgrImage(const cv::Mat& image) {
    //     if (image.empty()) {
    //         return false;
    //     }
    //     if (image.type() != CV_8UC3) {
    //         return false;
    //     }
    //     return true;
    // }

    /// @brief 检查逆仿射矩阵是否合法：非空、2×3、CV_64F 或 CV_32F
    /// @note  2×3 仿射矩阵： [a11 a12 tx]
    ///                       [a21 a22 ty]
    bool IsValidInverseTransform(const cv::Mat& M_inv) {
        if (M_inv.empty()) {
            return false;
        }
        if (M_inv.rows != 2 || M_inv.cols != 3) {
            return false;
        }
        int t = M_inv.type();
        if (t != CV_64FC1 && t != CV_32FC1) {
            return false;
        }
        return true;
    }

    /// @brief 将各种 mask 类型统一为 CV_32FC3、值范围 0~1
    /// @param mask 输入 mask（支持 CV_32FC1 / CV_8UC1 / CV_32FC3）
    /// @return 统一后的 CV_32FC3 mask；不支持的类型返回空 Mat
    cv::Mat NormalizeMaskToFloat3C(const cv::Mat& mask) {
        if (mask.empty()) {
            return cv::Mat();
        }

        cv::Mat res;

        if (mask.type() == CV_32FC1) {
            // 单通道浮点，值已在 0~1：复制为三通道
            std::vector<cv::Mat> channels = {mask, mask, mask};
            cv::merge(channels, res);
            return res;

        } else if (mask.type() == CV_8UC1) {
            // 单通道 uint8，值在 0~255：先归一化到 0~1，再复制三通道
            cv::Mat mask_float;
            mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
            std::vector<cv::Mat> channels = {mask_float, mask_float, mask_float};
            cv::merge(channels, res);
            return res;

        } else if (mask.type() == CV_32FC3) {
            // 已经是三通道浮点：直接 clone
            res = mask.clone();
            return res;
        }

        // 不支持的类型 → 返回空 Mat
        return res;
    }

    /// @brief 从输出图和 mask 提取统计信息，填充 FaceBlendInfo
    /// @param output_bgr  最终融合图（CV_8UC3）
    /// @param mask_3c     融合时使用的 mask（CV_32FC3，0~1）
    /// @param time_ms     总耗时（毫秒）
    /// @note  mask 三通道的 alpha 值都一样，取第一个通道做统计即可
    FaceBlendInfo CollectBlendInfo(const cv::Mat& output_bgr, const cv::Mat& mask_3c,
                                   double time_ms) {
        FaceBlendInfo info;

        // —— 输出图基本信息 ——
        info.output_width = output_bgr.cols;
        info.output_height = output_bgr.rows;
        info.output_channels = output_bgr.channels();
        info.output_type = output_bgr.type();

        // —— mask 统计（cv::minMaxLoc 只能处理单通道，所以先 split 取第一通道） ——
        std::vector<cv::Mat> chs;
        cv::split(mask_3c, chs);          // chs[0] = B 通道（三通道值相同，取任意一个）
        double min_val = 0.0, max_val = 0.0;
        cv::minMaxLoc(chs[0], &min_val, &max_val);
        info.mask_min = min_val;
        info.mask_max = max_val;
        info.mask_mean = cv::mean(mask_3c)[0];     // mean 返回 Scalar，[0] 取第一通道均值
        info.mask_non_zero_count = cv::countNonZero(chs[0]);

        // —— 性能 ——
        info.time_ms = time_ms;

        return info;
    }

} // namespace


// PImpl 内部实现

struct FaceBlender::Impl {

    /// @brief 对 96x96 生成图做轻微锐化（反锐化遮罩法）
    /// @note  公式：blur = GaussianBlur(src); sharp = src × (1+amount) + blur × (−amount)
    cv::Mat Sharpen96(const cv::Mat& generated_96,
                      const FaceBlendOptions& options) const {
        cv::Mat res;

        // 1. 空输入 → 返回空 Mat
        if (generated_96.empty()) {
            return res;
        }

        // 2. 未开启锐化 → 返回原图副本
        if (!options.enable_sharpen) {
            res = generated_96.clone();
            return res;
        }

        // 3. 反锐化遮罩：
        //    模糊图 = 高斯模糊(原图)
        //    边缘   = 原图 − 模糊图（隐含在 addWeighted 的负权重里）
        //    锐化   = 原图 + 边缘 × sharpen_amount
        cv::Mat blur;
        cv::GaussianBlur(generated_96, blur, cv::Size(0, 0), options.sharpen_sigma);

        cv::addWeighted(generated_96, 1.0 + options.sharpen_amount,  // 原图 × 1.25
                        blur,         -options.sharpen_amount,       // 模糊 × (−0.25)
                        0.0,                                         // gamma：不额外调亮/压暗
                        res);

        return res;
    }

    /// @brief 使用 M_inv 将 96x96 生成图回贴到原图坐标系
    /// @note  cv::warpAffine + BORDER_CONSTANT + 黑色：超出 96×96 的区域填黑，融合时不干扰原图
    cv::Mat RestoreToOriginal(const cv::Mat& generated_96, const cv::Mat& inverse_transform,
            const cv::Size& output_size, const FaceBlendOptions& options) const {
        cv::Mat res;

        // 1. 输入合法性检查
        if (generated_96.empty() || !IsValidInverseTransform(inverse_transform)) {
            return res;
        }
        if (output_size.width <= 0 || output_size.height <= 0) {
            return res;
        }

        // 2. 创建输出画布（全黑，原图尺寸）
        res = cv::Mat::zeros(output_size, CV_8UC3);

        // 3. warpAffine：将 96×96 小图"定位"到原图的正确位置
        cv::warpAffine(generated_96,                   // src: 96×96 生成图
                       res,                             // dst: 输出画布
                       inverse_transform,               // M: 对齐→原图的 2×3 矩阵
                       output_size,                     // dsize: 原图尺寸
                       options.image_interpolation,     // flags: INTER_CUBIC
                       cv::BORDER_CONSTANT,             // 越界填常数
                       cv::Scalar(0, 0, 0));            // 常数=黑色

        return res;
    }

    /// @brief 将 96×96 mask 统一为三通道 0~1 浮点图，并回贴到原图坐标系
    /// @note  支持 CV_32FC1 / CV_8UC1 / CV_32FC3 三种输入；
    ///        回贴后可选羽化（GaussianBlur）和 clamp（0~1 裁剪）
    cv::Mat RestoreMaskToOriginal(const cv::Mat& mask_96, const cv::Mat& inverse_transform,
            const cv::Size& output_size, const FaceBlendOptions& options) const {
        cv::Mat res;

        // 1. 输入合法性检查
        if (mask_96.empty() || !IsValidInverseTransform(inverse_transform)) {
            return res;
        }
        if (output_size.width <= 0 || output_size.height <= 0) {
            return res;
        }

        // 2. 统一 mask 类型 → CV_32FC3, 0~1
        cv::Mat mask_32fc3 = NormalizeMaskToFloat3C(mask_96);
        if (mask_32fc3.empty()) {
            return res;
        }

        // 3. warpAffine 回贴到原图坐标系
        //    用 INTER_LINEAR 保持 alpha 的线性渐变不被破坏
        //    BORDER_CONSTANT + 0：对齐脸之外的区域 mask=0（完全不动原图）
        cv::warpAffine(mask_32fc3, res, inverse_transform, output_size,
                       options.mask_interpolation,      // INTER_LINEAR
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

        // 4. 可选羽化（回贴后轻微高斯模糊，柔化可能出现的插值硬边）
        if (options.enable_mask_blur) {
            // src 和 dst 同一张图 = 原地模糊；sigma=0 让 OpenCV 根据核大小自动计算
            cv::GaussianBlur(res, res,
                             cv::Size(options.restored_mask_blur_kernel,
                                      options.restored_mask_blur_kernel),
                             0);
        }

        // 5. 可选 clamp（修正 warpAffine + GaussianBlur 引入的微小浮点越界）
        //    不 clamp 会导致 mask<0 或 mask>1，融合公式中出现负权重，边缘颜色偏移
        if (options.clamp_mask) {
            cv::threshold(res, res, 0.0, 0.0, cv::THRESH_TOZERO);  // <0 → 0
            cv::threshold(res, res, 1.0, 1.0, cv::THRESH_TRUNC);   // >1 → 1
        }

        return res;
    }

    /// @brief alpha 融合 + 可选细节恢复（本模块最核心的融合函数）
    /// @note  alpha 融合公式： out = gen×mask + base×(1−mask)
    ///         细节恢复公式：   out += (base − blur(base)) × mask × detail_strength
    FaceBlendResult BlendWithDetail(const cv::Mat& base_bgr, const cv::Mat& restored_face_bgr,
            const cv::Mat& restored_mask, const FaceBlendOptions& options) const {
        // 1. 输入验证 —— 逐个检查三张图，失败时返回明确的错误码和描述
        if (base_bgr.empty()) {
            return MakeErrorResult(FaceBlendStatus::kEmptyBaseImage,
                                   "base_bgr is empty", options);
        }
        if (base_bgr.type() != CV_8UC3) {
            return MakeErrorResult(FaceBlendStatus::kInvalidBaseImageType,
                                   "base_bgr type is not CV_8UC3, actual type="
                                   + std::to_string(base_bgr.type()), options);
        }

        if (restored_face_bgr.empty()) {
            return MakeErrorResult(FaceBlendStatus::kEmptyGeneratedImage,
                                   "restored_face_bgr is empty", options);
        }
        if (restored_face_bgr.type() != CV_8UC3) {
            return MakeErrorResult(FaceBlendStatus::kInvalidGeneratedType,
                                   "restored_face_bgr type is not CV_8UC3, actual type="
                                   + std::to_string(restored_face_bgr.type()), options);
        }

        if (restored_mask.empty()) {
            return MakeErrorResult(FaceBlendStatus::kEmptyMask,
                                   "restored_mask is empty", options);
        }

        // 三张图尺寸必须一致（width 和 height 都相同）
        if (base_bgr.size() != restored_face_bgr.size() ||
            base_bgr.size() != restored_mask.size()) {
            std::string msg = "Size mismatch: base="
                + std::to_string(base_bgr.cols) + "x" + std::to_string(base_bgr.rows)
                + " restored_face="
                + std::to_string(restored_face_bgr.cols) + "x" + std::to_string(restored_face_bgr.rows)
                + " mask="
                + std::to_string(restored_mask.cols) + "x" + std::to_string(restored_mask.rows);
            return MakeErrorResult(FaceBlendStatus::kSizeMismatch, msg, options);
        }

        // 2. 统一 mask 类型 → CV_32FC3, 0~1
        cv::Mat mask_f = NormalizeMaskToFloat3C(restored_mask);
        if (mask_f.empty()) {
            std::string msg = "Invalid mask type: type=" + std::to_string(restored_mask.type())
                + " (expected CV_32FC1=" + std::to_string(CV_32FC1)
                + ", CV_8UC1=" + std::to_string(CV_8UC1)
                + ", or CV_32FC3=" + std::to_string(CV_32FC3) + ")";
            return MakeErrorResult(FaceBlendStatus::kInvalidMaskType, msg, options);
        }

        // 3. 转为 float 精度做融合
        //    必须转：uint8(0~255) 直接做 mul 会截断溢出；
        //    float(0~1) 保证 alpha 融合的数学精度
        cv::Mat base_f, gen_f;
        base_bgr.convertTo(base_f, CV_32FC3, 1.0 / 255.0);
        restored_face_bgr.convertTo(gen_f, CV_32FC3, 1.0 / 255.0);
        // mask_f 已经是 CV_32FC3, 0~1，无需转换

        // 局部颜色匹配：对 generated patch 在 mask 区域内做分通道均值-标准差匹配。
        // 目的：Wav2Lip 模型生成的 96×96 纹理与原图皮肤常有色差和亮度差，
        // 简单 alpha blend 不能纠正这种差异，导致可见的"贴片"边界。
        // 本段在融合前对 gen 做局部颜色归一化，减轻色差，改善融合自然度。
        // 限制：这只改善颜色一致性，不能恢复 96×96 模型输出经上采样后损失的细节。
        if (options.enable_color_match) {
            const float kEpsilon = 1e-6f;  // 防止 std=0 时除零
            // mask_f 是 CV_32FC3，只取第一通道做统计（三通道 alpha 值相同）
            std::vector<cv::Mat> mask_chs;
            cv::split(mask_f, mask_chs);
            cv::Mat mask_active = mask_chs[0] > 0.01f;  // CV_8UC1

            // 只在 mask 有效像素足够时执行匹配，避免统计量不稳定
            if (cv::countNonZero(mask_active) >= 4) {
                std::vector<cv::Mat> base_chs, gen_chs;
                cv::split(base_f, base_chs);
                cv::split(gen_f, gen_chs);

                const float kMeanDegradationTolerance = 1e-6f;
                std::vector<cv::Mat> corrected_chs(3);
                for (int ch = 0; ch < 3; ++ch) {
                    cv::Scalar base_mean_scalar, base_std_scalar;
                    cv::Scalar gen_mean_scalar, gen_std_scalar;
                    cv::meanStdDev(base_chs[ch], base_mean_scalar, base_std_scalar, mask_active);
                    cv::meanStdDev(gen_chs[ch], gen_mean_scalar, gen_std_scalar, mask_active);

                    float base_mean = static_cast<float>(base_mean_scalar[0]);
                    float base_std = static_cast<float>(base_std_scalar[0]);
                    float gen_mean = static_cast<float>(gen_mean_scalar[0]);
                    float gen_std = static_cast<float>(gen_std_scalar[0]);

                    float before_error = std::abs(gen_mean - base_mean);

                    // 分通道颜色匹配：使 gen 的均值和标准差对齐 base
                    float scale = base_std / std::max(gen_std, kEpsilon);
                    cv::Mat corrected = (gen_chs[ch] - gen_mean) * scale + base_mean;
                    // 限制到合法像素范围 [0, 1]
                    cv::threshold(corrected, corrected, 0.0, 0.0, cv::THRESH_TOZERO);
                    cv::threshold(corrected, corrected, 1.0, 1.0, cv::THRESH_TRUNC);

                    // 非恶化保护：clamp 后重新计算均值，若校正在有效 mask 内使均值更偏离 base，
                    // 则对该通道回退到原始 generated 值，避免"越校正越差"。
                    cv::Scalar corrected_mean_scalar, _;
                    cv::meanStdDev(corrected, corrected_mean_scalar, _, mask_active);
                    float corrected_mean = static_cast<float>(corrected_mean_scalar[0]);
                    float after_error = std::abs(corrected_mean - base_mean);

                    if (after_error > before_error + kMeanDegradationTolerance) {
                        corrected_chs[ch] = gen_chs[ch].clone();  // 回退到校正前
                    } else {
                        corrected_chs[ch] = corrected;
                    }
                }
                cv::merge(corrected_chs, gen_f);
            }
        }

        // 4. alpha 融合（核心公式）
        //    out = gen × mask + base × (1 − mask)
        //    mul() 是逐像素乘法，Scalar(1,1,1)−mask 对 B/G/R 三通道分别做 1−mask
        cv::Mat blended = gen_f.mul(mask_f) + base_f.mul(cv::Scalar(1, 1, 1) - mask_f);

        // 5. 可选细节恢复：从原图提取高频纹理，在 mask 区域内少量加回
        //    高频层 detail = base − blur(base)（原图减去模糊图）
        //    只加回 mask 区域（× mask）、只加一点点（× detail_strength）
        if (options.enable_detail_restore) {
            cv::Mat base_blur;
            cv::GaussianBlur(base_f, base_blur, cv::Size(0, 0), options.detail_sigma);
            cv::Mat detail = base_f - base_blur;                     // 原图的高频纹理
            blended += detail.mul(mask_f) * options.detail_strength;  // 仅在 mask 区域以极小权重加回
        }

        // 6. 转回 uint8
        //    float 精度下算完 → 最后一刻转回 CV_8UC3（下游模块和 cv::imwrite 需要）
        FaceBlendResult result;
        blended.convertTo(result.final_bgr, CV_8UC3, 255.0);

        // 7. 填充调试图和统计信息
        result.generated_96_sharp = cv::Mat();        // BlendWithDetail 不处理锐化，由 BlendMouthToOriginal 填充
        result.restored_face_bgr = restored_face_bgr.clone();
        result.restored_mask_3c = mask_f.clone();
        result.info = CollectBlendInfo(result.final_bgr, mask_f, 0.0);
        result.options_used = options;
        result.success = true;
        result.status = FaceBlendStatus::kOk;

        return result;
    }

    /// @brief 一站式嘴部融合接口（后续 Pipeline 推荐直接调用）
    ///
    /// 串起完整流程：
    ///   验证 → 计时开始
    ///     → Sharpen96（轻微锐化 96×96 生成图）
    ///     → RestoreToOriginal（回贴嘴部图到原图坐标系）
    ///     → RestoreMaskToOriginal（回贴 mask 到原图坐标系）
    ///     → BlendWithDetail（alpha 融合 + 细节恢复）
    ///   → 计时结束
    ///     → 填充调试图 + 统计信息 → 返回完整 FaceBlendResult
    ///
    /// 任何一个子步骤失败都会提前返回带错误码的 FaceBlendResult
    FaceBlendResult BlendMouthToOriginal(const cv::Mat& base_bgr,const cv::Mat& generated_96,
        const cv::Mat& mask_96,const cv::Mat& inverse_transform,
        const FaceBlendOptions& options) const {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. 输入验证

        // 1a. base_bgr
        if (base_bgr.empty()) {
            return MakeErrorResult(FaceBlendStatus::kEmptyBaseImage,
                                   "base_bgr is empty", options);
        }
        if (base_bgr.type() != CV_8UC3) {
            return MakeErrorResult(FaceBlendStatus::kInvalidBaseImageType,
                                   "base_bgr type is not CV_8UC3, actual type="
                                   + std::to_string(base_bgr.type()), options);
        }

        // 1b. generated_96
        if (generated_96.empty()) {
            return MakeErrorResult(FaceBlendStatus::kEmptyGeneratedImage,
                                   "generated_96 is empty", options);
        }
        if (generated_96.type() != CV_8UC3) {
            return MakeErrorResult(FaceBlendStatus::kInvalidGeneratedType,
                                   "generated_96 type is not CV_8UC3, actual type="
                                   + std::to_string(generated_96.type()), options);
        }
        if (generated_96.cols != options.expected_aligned_size ||
            generated_96.rows != options.expected_aligned_size) {
            std::string msg = "generated_96 size is " + std::to_string(generated_96.cols)
                            + "x" + std::to_string(generated_96.rows)
                            + ", expected " + std::to_string(options.expected_aligned_size)
                            + "x" + std::to_string(options.expected_aligned_size);
            return MakeErrorResult(FaceBlendStatus::kInvalidGeneratedSize, msg, options);
        }

        // 1c. mask_96
        if (mask_96.empty()) {
            return MakeErrorResult(FaceBlendStatus::kEmptyMask,
                                   "mask_96 is empty", options);
        }

        // 1d. inverse_transform
        if (!IsValidInverseTransform(inverse_transform)) {
            std::string msg = "inverse_transform is invalid: "
                            + std::to_string(inverse_transform.rows) + "x"
                            + std::to_string(inverse_transform.cols)
                            + " type=" + std::to_string(inverse_transform.type());
            return MakeErrorResult(FaceBlendStatus::kInvalidInverseTransform, msg, options);
        }

        // 1e. 输出尺寸
        if (base_bgr.cols <= 0 || base_bgr.rows <= 0) {
            return MakeErrorResult(FaceBlendStatus::kInvalidOutputSize,
                                   "base_bgr has invalid size: "
                                   + std::to_string(base_bgr.cols) + "x"
                                   + std::to_string(base_bgr.rows), options);
        }

        // 2. Sharpen96 — 轻微锐化 96×96 生成图
        cv::Mat generated_96_sharp = Sharpen96(generated_96, options);


        // 3. RestoreToOriginal — 将锐化后的 96×96 嘴部图回贴到原图坐标系
        cv::Mat restored_face = RestoreToOriginal(generated_96_sharp,
                                                  inverse_transform,
                                                  base_bgr.size(),
                                                  options);
        if (restored_face.empty()) {
            return MakeErrorResult(FaceBlendStatus::kWarpFailed,
                                   "RestoreToOriginal returned empty", options);
        }


        // 4. RestoreMaskToOriginal — 将 96×96 mask 回贴到原图坐标系
        cv::Mat restored_mask = RestoreMaskToOriginal(mask_96,
                                                      inverse_transform,
                                                      base_bgr.size(),
                                                      options);
        if (restored_mask.empty()) {
            return MakeErrorResult(FaceBlendStatus::kWarpFailed,
                                   "RestoreMaskToOriginal returned empty", options);
        }


        // 5. BlendWithDetail — alpha 融合 + 可选细节恢复
        FaceBlendResult result = BlendWithDetail(base_bgr,
                                                 restored_face,
                                                 restored_mask,
                                                 options);
        if (!result.success) {
            return result;   // BlendWithDetail 内部已经填好了错误信息，直接透传
        }


        // 6. 计时结束，填充调试图（覆盖为完整链路中实际产生的中间图）
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        result.generated_96_sharp = generated_96_sharp.clone();
        result.restored_face_bgr = restored_face.clone();
        result.restored_mask_3c = restored_mask.clone();
        result.info = CollectBlendInfo(result.final_bgr, restored_mask, elapsed_ms);
        result.options_used = options;

        return result;
    }
};

// FaceBlender 公开接口（委托给 pImpl_）

FaceBlender::FaceBlender() : pImpl_(std::make_unique<Impl>()) {}
FaceBlender::~FaceBlender() = default;

FaceBlender::FaceBlender(FaceBlender&&) noexcept = default;
FaceBlender& FaceBlender::operator=(FaceBlender&&) noexcept = default;

cv::Mat FaceBlender::Sharpen96(const cv::Mat& generated_96,
                                const FaceBlendOptions& options) const {
    return pImpl_->Sharpen96(generated_96, options);
}

cv::Mat FaceBlender::RestoreToOriginal(const cv::Mat& generated_96, const cv::Mat& inverse_transform,
    const cv::Size& output_size, const FaceBlendOptions& options) const {

    return pImpl_->RestoreToOriginal(generated_96, inverse_transform, output_size, options);
}

cv::Mat FaceBlender::RestoreMaskToOriginal(const cv::Mat& mask_96, const cv::Mat& inverse_transform,
    const cv::Size& output_size, const FaceBlendOptions& options) const {
    return pImpl_->RestoreMaskToOriginal(mask_96, inverse_transform, output_size, options);
}

FaceBlendResult FaceBlender::BlendWithDetail(const cv::Mat& base_bgr, const cv::Mat& restored_face_bgr,
    const cv::Mat& restored_mask, const FaceBlendOptions& options) const {
    return pImpl_->BlendWithDetail(base_bgr, restored_face_bgr, restored_mask, options);
}

FaceBlendResult FaceBlender::BlendMouthToOriginal(const cv::Mat& base_bgr, const cv::Mat& generated_96,
    const cv::Mat& mask_96, const cv::Mat& inverse_transform,
    const FaceBlendOptions& options) const {
    return pImpl_->BlendMouthToOriginal(base_bgr, generated_96, mask_96,
                                        inverse_transform, options);
}

std::string FaceBlender::StatusToString(FaceBlendStatus status) {
    switch (status) {
        case FaceBlendStatus::kOk:                      return "Ok";
        case FaceBlendStatus::kEmptyBaseImage:          return "EmptyBaseImage";
        case FaceBlendStatus::kEmptyGeneratedImage:     return "EmptyGeneratedImage";
        case FaceBlendStatus::kEmptyMask:               return "EmptyMask";
        case FaceBlendStatus::kInvalidGeneratedSize:    return "InvalidGeneratedSize";
        case FaceBlendStatus::kInvalidBaseImageType:    return "InvalidBaseImageType";
        case FaceBlendStatus::kInvalidGeneratedType:    return "InvalidGeneratedType";
        case FaceBlendStatus::kInvalidMaskType:         return "InvalidMaskType";
        case FaceBlendStatus::kInvalidInverseTransform: return "InvalidInverseTransform";
        case FaceBlendStatus::kInvalidOutputSize:       return "InvalidOutputSize";
        case FaceBlendStatus::kSizeMismatch:            return "SizeMismatch";
        case FaceBlendStatus::kWarpFailed:              return "WarpFailed";
        case FaceBlendStatus::kBlendFailed:             return "BlendFailed";
        case FaceBlendStatus::kOpenCvError:             return "OpenCvError";
        default:                                        return "Unknown";
    }
}

} // namespace core
} // namespace digital_human
