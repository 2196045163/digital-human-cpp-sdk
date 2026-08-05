#pragma once

#include <opencv2/core.hpp>    // cv::Mat, cv::Size
#include <opencv2/imgproc.hpp> // cv::INTER_CUBIC, cv::INTER_LINEAR
#include <memory>             // std::unique_ptr
#include <string>             // std::string

namespace digital_human {
namespace core {

/// @brief 人脸融合模块状态码，描述成功或失败原因
/// @note  SDK 普通错误使用状态码，不主动抛异常；
///        OpenCV 内部异常用 try/catch 捕获，转成 kOpenCvError
enum class FaceBlendStatus {
    kOk,                       ///< 成功
    kEmptyBaseImage,           ///< 原图为空
    kEmptyGeneratedImage,      ///< 生成的 96x96 嘴部图为空
    kEmptyMask,                ///< alpha mask 为空
    kInvalidGeneratedSize,     ///< generated_96 尺寸不是预期的 96x96
    kInvalidBaseImageType,     ///< base_bgr 类型不是 CV_8UC3
    kInvalidGeneratedType,     ///< generated_96 类型不是 CV_8UC3
    kInvalidMaskType,          ///< mask 类型不支持（需为 CV_32FC1 / CV_8UC1 / CV_32FC3）
    kInvalidInverseTransform,  ///< 逆仿射矩阵为空、类型不对或不是 2x3
    kInvalidOutputSize,        ///< 输出尺寸不合法（宽或高 <= 0）
    kSizeMismatch,             ///< 融合时三张图尺寸不一致
    kWarpFailed,               ///< cv::warpAffine 执行失败
    kBlendFailed,              ///< 融合过程失败（默认兜底错误）
    kOpenCvError               ///< OpenCV 内部抛异常，被 try/catch 捕获
};

/// @brief 人脸融合可调选项
/// @note  所有字段都有默认值，调用方可只改关心的部分；
///        真实模型上线后可微调这些参数优化画质
struct FaceBlendOptions {
    // —— 对齐图尺寸 ——
    int expected_aligned_size = 96;   ///< 期望的生成图边长（默认 96，Wav2Lip 标准）

    // —— 锐化（缓解 96x96 放大到原图后的软糊感） ——
    bool enable_sharpen = true;       ///< 是否对 generated_96 做轻微锐化
    double sharpen_amount = 0.25;     ///< 锐化强度（0=不锐化，建议 0.15~0.35，太大→边缘噪声/光晕）
    double sharpen_sigma = 1.0;       ///< 锐化前高斯模糊的 sigma（控制"模糊"到什么程度再提取边缘）

    // —— mask 羽化（已优化：缩小二次羽化核，配合 FaceMaskGenerator 的缩小 mask） ——
    bool enable_mask_blur = true;     ///< 回贴后是否对 mask 再做一次高斯模糊
    int restored_mask_blur_kernel = 3;///< 回贴后 mask 高斯模糊核大小（奇数）

    // —— 细节恢复（保持原有默认值，本轮不修改） ——
    bool enable_detail_restore = true;///< 是否从原图提取高频细节并少量加回融合区
    double detail_sigma = 1.2;        ///< 提取原图细节前的高斯模糊 sigma
    double detail_strength = 0.12;    ///< 细节回填强度（保持原值 0.12，本轮不修改）

    // —— mask 范围控制 ——
    bool clamp_mask = true;           ///< 融合前是否强制将 mask 限制在 0~1

    // —— 局部颜色匹配 ——
    // 在 alpha blend 前对 generated patch 做分通道均值-标准差匹配，
    // 使生成嘴部的色调和对比度接近原始目标区域，减轻色差和贴片感。
    // 只在 mask 有效区域内计算统计量，mask 外像素不改变。
    bool enable_color_match = true;   ///< 是否启用 mask 内局部颜色匹配

    // —— 插值方式 ——
    int image_interpolation = cv::INTER_CUBIC;   ///< 图像回贴插值（16 邻域三次插值，放大更平滑）
    int mask_interpolation = cv::INTER_LINEAR;   ///< mask 回贴插值（4 邻域线性插值，保持 alpha 梯度不被破坏）
};

/// @brief 融合结果统计信息，方便 example 打印和测试验证
struct FaceBlendInfo {
    // —— 输出图基本信息 ——
    int output_width = 0;          ///< 输出图宽度（像素）
    int output_height = 0;         ///< 输出图高度（像素）
    int output_channels = 0;       ///< 输出图通道数（固定为 3，BGR）
    int output_type = 0;           ///< OpenCV 类型编码（固定为 CV_8UC3）

    // —— mask 统计（用于验证 mask 范围、覆盖区域是否合理） ——
    double mask_min = 0.0;         ///< mask 最小像素值（应在 0~1 之间，clamp 后 >= 0）
    double mask_max = 0.0;         ///< mask 最大像素值（应在 0~1 之间，clamp 后 <= 1）
    double mask_mean = 0.0;        ///< mask 像素平均值（粗略反映 mask 覆盖比例）
    int mask_non_zero_count = 0;   ///< 非零像素数量（反映 mask 实际覆盖了多少原图像素）

    // —— 性能 ——
    double time_ms = 0.0;          ///< 完整融合耗时（毫秒），从 BlendMouthToOriginal 入口计时
};

/// @brief 人脸融合统一返回结果
/// @note  包含成功标志、错误信息、最终图、调试图、统计信息、实际使用的选项。
///        初学阶段重点看 "中间调试图"，便于定位问题是出在锐化/回贴/mask/融合哪一步
struct FaceBlendResult {
    // —— 状态 ——
    bool success = false;                              ///< 是否成功
    FaceBlendStatus status = FaceBlendStatus::kBlendFailed; ///< 状态码
    std::string error_message;                         ///< 失败时的错误描述

    // —— 最终输出 ——
    cv::Mat final_bgr;             ///< 最终融合结果，CV_8UC3，原图尺寸

    // —— 中间调试图（用于肉眼定位问题） ——
    cv::Mat generated_96_sharp;    ///< 锐化后的 96x96 生成图（CV_8UC3，96x96）
    cv::Mat restored_face_bgr;     ///< 回贴到原图尺寸的嘴部图（CV_8UC3，原图尺寸，仅嘴部有内容其余黑色）
    cv::Mat restored_mask_3c;      ///< 回贴到原图尺寸的 mask（CV_32FC3，原图尺寸，0~1）

    // —— 统计与选项 ——
    FaceBlendInfo info;            ///< 输出图与 mask 的统计信息
    FaceBlendOptions options_used; ///< 本次实际使用的选项（便于调试时确认参数）
};

/// @brief 人脸融合模块（Face Blender）
///
/// 本模块负责把模型输出的 96x96 嘴部图像和 96x96 alpha mask，
/// 借助 FaceAligner 提供的逆仿射矩阵 M_inv 映射回原图尺寸，
/// 并在原图上只替换嘴部区域，使输出帧既有生成口型又尽量保留原图人脸细节。
///
/// 职责边界：
/// - 不做人脸检测（复用 FaceDetector）
/// - 不做人脸对齐（复用 FaceAligner）
/// - 不生成 mask（复用 FaceMaskGenerator）
/// - 不做模型推理（不依赖 ncnn）
/// - 不做视频编码/窗口显示（属于 RenderProcessor）
///
/// 调用方式：
/// - 分步调试：Sharpen96 → RestoreToOriginal → RestoreMaskToOriginal → BlendWithDetail
/// - 生产使用：BlendMouthToOriginal（一站式，返回包含调试图的 FaceBlendResult）
class FaceBlender {
public:
    FaceBlender();
    ~FaceBlender();

    // —— 禁止拷贝，允许移动 ——
    FaceBlender(const FaceBlender&) = delete;
    FaceBlender& operator=(const FaceBlender&) = delete;
    FaceBlender(FaceBlender&&) noexcept;
    FaceBlender& operator=(FaceBlender&&) noexcept;

    /// @brief 对 96x96 生成图做轻微锐化（反锐化遮罩法）
    /// @param generated_96 模型输出的 96x96 嘴部图（CV_8UC3）
    /// @param options 融合选项（控制是否开启、锐化强度、sigma）
    /// @return 锐化后的 96x96 图（CV_8UC3），空输入返回空图
    /// @note  公式：blur = GaussianBlur(src); sharp = src * (1+amount) + blur * (-amount)
    cv::Mat Sharpen96(
        const cv::Mat& generated_96,
        const FaceBlendOptions& options = FaceBlendOptions()
    ) const;

    /// @brief 使用 M_inv 将 96x96 生成图回贴到原图坐标系
    /// @param generated_96 模型输出的 96x96 嘴部图（CV_8UC3）
    /// @param inverse_transform 2x3 逆仿射矩阵 M_inv（CV_64F 或 CV_32F），96x96 对齐坐标系 → 原图坐标系
    /// @param output_size 输出图尺寸（即原图尺寸）
    /// @param options 融合选项（控制插值方式）
    /// @return 原图尺寸的回贴图（CV_8UC3），仅嘴部/人脸区域有内容，其余为黑色
    /// @note  使用 cv::warpAffine + BORDER_CONSTANT + 黑边，INTER_CUBIC 插值
    cv::Mat RestoreToOriginal(
        const cv::Mat& generated_96,
        const cv::Mat& inverse_transform,
        const cv::Size& output_size,
        const FaceBlendOptions& options = FaceBlendOptions()
    ) const;

    /// @brief 将 96x96 mask 统一为三通道 0~1 浮点图并回贴到原图坐标系
    /// @param mask_96 96x96 alpha mask（支持 CV_32FC1 / CV_8UC1 / CV_32FC3）
    /// @param inverse_transform 2x3 逆仿射矩阵 M_inv
    /// @param output_size 输出图尺寸（即原图尺寸）
    /// @param options 融合选项（控制是否羽化、是否 clamp、插值方式）
    /// @return 原图尺寸的 mask（CV_32FC3，0~1），非嘴部区域值接近 0
    /// @note  自动将 CV_8UC1 除以 255 归一化，CV_32FC1 复制为三通道，
    ///        使用 INTER_LINEAR 插值保持 alpha 梯度，可选再羽化 + clamp
    cv::Mat RestoreMaskToOriginal(
        const cv::Mat& mask_96,
        const cv::Mat& inverse_transform,
        const cv::Size& output_size,
        const FaceBlendOptions& options = FaceBlendOptions()
    ) const;

    /// @brief 使用回贴后的生成图和 mask，与原图做 alpha 融合 + 可选细节恢复
    /// @param base_bgr 原始图像（CV_8UC3，原图尺寸）
    /// @param restored_face_bgr 回贴到原图尺寸的生成嘴部图（CV_8UC3，原图尺寸）
    /// @param restored_mask 回贴到原图尺寸的 mask（CV_32FC1 或 CV_32FC3，0~1，原图尺寸）
    /// @param options 融合选项
    /// @return FaceBlendResult，包含最终融合图和统计信息
    /// @note  alpha 融合公式：out = gen × mask + base × (1 - mask)；
    ///         细节恢复公式：blended += (base − blur(base)) × mask × detail_strength
    FaceBlendResult BlendWithDetail(
        const cv::Mat& base_bgr,
        const cv::Mat& restored_face_bgr,
        const cv::Mat& restored_mask,
        const FaceBlendOptions& options = FaceBlendOptions()
    ) const;

    /// @brief 一站式嘴部融合接口（后续 Pipeline 推荐使用）
    ///
    /// 串起完整流程：
    /// 1. 验证全部输入
    /// 2. Sharpen96 锐化生成图
    /// 3. RestoreToOriginal 回贴嘴部图到原图
    /// 4. RestoreMaskToOriginal 回贴 mask 到原图
    /// 5. BlendWithDetail alpha 融合 + 细节恢复
    /// 6. 收集统计信息，返回完整 FaceBlendResult
    ///
    /// @param base_bgr 原始图像（CV_8UC3，任意尺寸）
    /// @param generated_96 模型输出的 96x96 嘴部图（CV_8UC3）
    /// @param mask_96 96x96 alpha mask（CV_32FC1 等，0~1，来自 FaceMaskGenerator）
    /// @param inverse_transform 2x3 逆仿射矩阵 M_inv（来自 FaceAligner::inverse_transform）
    /// @param options 融合选项
    /// @return FaceBlendResult，包含 final_bgr + 中间调试图 + 统计信息
    FaceBlendResult BlendMouthToOriginal(
        const cv::Mat& base_bgr,
        const cv::Mat& generated_96,
        const cv::Mat& mask_96,
        const cv::Mat& inverse_transform,
        const FaceBlendOptions& options = FaceBlendOptions()
    ) const;

    /// @brief 将状态码转为可读字符串（用于日志/example 打印）
    static std::string StatusToString(FaceBlendStatus status);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;  ///< PImpl 惯用法，隐藏实现细节和 OpenCV 调用
};

} // namespace core
} // namespace digital_human
