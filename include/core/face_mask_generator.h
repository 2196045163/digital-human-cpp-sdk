#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace digital_human {
namespace core {

    /// @brief 掩码生成状态码，描述成功或失败原因
    enum class FaceMaskStatus {
        kOk,                      ///< 成功
        kInvalidImageSize,        ///< 图像尺寸无效（width/height <= 0）
        kInvalidLandmarkCount,    ///< 关键点数量不足 68
        kInvalidLandmarkGeometry, ///< 嘴部几何异常（点重合、凸包不足 3 个点、全部越界）
        kInvalidMaskParameters,   ///< 参数非法（膨胀半径负数、模糊核负数等）
        kMaskEmpty,               ///< mask 非零区域为空
        kOpenCvError              ///< OpenCV 内部异常
    };

    /// @brief 嘴部区域模式
    enum class MouthRegionMode {
        kOuterLip,   ///< 使用 48~59（嘴唇外轮廓 12 点）
        kFullMouth   ///< 使用 48~67（完整嘴部 20 点，含内外轮廓）
    };

    /// @brief mask 生成选项
    struct FaceMaskOptions {
        MouthRegionMode mouth_region = MouthRegionMode::kFullMouth; ///< 嘴部区域模式
        bool use_convex_hull = true;         ///< 是否使用凸包（让多边形区域更稳定）
        int dilate_radius = 3;               ///< 膨胀半径（控制嘴部区域向外扩展的像素数）
        // mask 羽化和范围参数已优化：缩小 bbox 扩展和羽化核，减少对脸颊/下巴原始纹理的不必要替换，
        // 同时完整保留嘴唇、嘴角和合理张嘴区域的覆盖。96×96 模型分辨率限制见 FaceBlender 文档。
        int blur_kernel_size = 7;            ///< 96×96 mask 高斯模糊核大小（奇数，自动修正）
        int border_clear = 2;                ///< 96x96 mask 边界清零像素数（防止回贴边框）
        bool limit_to_mouth_bbox = true;     ///< 是否限制 mask 不超出嘴部外接矩形扩展范围
        float bbox_expand_x = 0.28f;         ///< 嘴部外接矩形水平扩展比例
        float bbox_expand_top = 0.45f;       ///< 嘴部外接矩形上方扩展比例
        float bbox_expand_bottom = 0.50f;    ///< 嘴部外接矩形下方扩展比例
    };

    /// @brief 本模块生成的 alpha mask 这张图的 mask 统计信息，方便 example 打印和测试验证
    struct FaceMaskInfo {
        int width = 0;              ///< mask 宽度
        int height = 0;             ///< mask 高度
        int channels = 0;           ///< 通道数
        int type = 0;               ///< OpenCV 类型（CV_32FC1 等）
        double min_value = 0.0;     ///< mask 最小像素值（应在 0~1 之间）
        double max_value = 0.0;     ///< mask 最大像素值（应在 0~1 之间）
        int non_zero_count = 0;     ///< 非零像素数量
    };

    /// @brief 掩码生成统一返回结果
    struct FaceMaskResult {
        bool success = false;                                      ///< 是否成功
        FaceMaskStatus status = FaceMaskStatus::kOpenCvError;      ///< 状态码
        std::string error_message;                                  ///< 失败时的错误描述
        cv::Mat alpha_mask;         ///< alpha mask（CV_32FC1, 范围 0~1）
        FaceMaskInfo info;          ///< mask 统计信息
        FaceMaskOptions options_used; ///< 实际使用的选项（可能经过修正）
        double time_ms = 0.0;       ///< 耗时（毫秒）
    };

    /// @brief 人脸掩码生成器
    /// 根据人脸关键点生成嘴部 alpha mask，为后续模型生成结果的融合做准备
    class FaceMaskGenerator {
    public:
        FaceMaskGenerator();
        ~FaceMaskGenerator();

        FaceMaskGenerator(const FaceMaskGenerator&) = delete;
        FaceMaskGenerator& operator=(const FaceMaskGenerator&) = delete;
        FaceMaskGenerator(FaceMaskGenerator&&) noexcept;
        FaceMaskGenerator& operator=(FaceMaskGenerator&&) noexcept;

        /// @brief 在原图坐标系下生成嘴部 alpha mask（用于可视化和调试）
        /// @param image_size 原图尺寸
        /// @param landmarks 原图坐标系下的 68 个关键点
        /// @param options mask 生成选项
        /// @return 与原图同尺寸的 CV_32FC1 alpha mask
        FaceMaskResult GenerateMouthMask(
            const cv::Size& image_size,
            const std::vector<cv::Point>& landmarks,
            const FaceMaskOptions& options = FaceMaskOptions()
        );

        /// @brief 在对齐坐标系下生成精细嘴部 alpha mask（用于后续模型融合）
        /// @param aligned_size 对齐图尺寸（默认 96x96）
        /// @param aligned_landmarks FaceAligner 输出的对齐后 68 个关键点
        /// @param options mask 生成选项
        /// @return 与对齐图同尺寸的 CV_32FC1 alpha mask
        FaceMaskResult GenerateAlignedMouthMask(
            const cv::Size& aligned_size,
            const std::vector<cv::Point2f>& aligned_landmarks,
            const FaceMaskOptions& options = FaceMaskOptions()
        );

        /// @brief 将单通道 alpha mask 转为三通道（B/G/R 三个通道均为同一 alpha 值）
        /// @param alpha_mask 单通道或三通道 alpha mask
        /// @return CV_32FC3 mask，用于逐通道融合公式
        cv::Mat To3ChannelMask(const cv::Mat& alpha_mask) const;

        /// @brief 将状态码转为可读字符串
        static std::string StatusToString(FaceMaskStatus status);

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl_;
    };

}   // namespace core
}   // namespace digital_human
