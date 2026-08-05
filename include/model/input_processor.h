#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <opencv2/core.hpp>

namespace digital_human {
namespace model {

/// @brief 模型输入构建状态码，每个失败点对应一个明确的枚举值
/// @note  Builder 只校验不修正——尺寸/类型/布局不满足契约时直接拒绝，不替上游"擦屁股"
enum class ModelInputStatus {
    // ---- 成功 ----
    kOk,                    ///< 构建成功

    // ---- 人脸图像校验 ----
    kEmptyAlignedFace,      ///< 对齐人脸 cv::Mat 为空
    kInvalidFaceSize,       ///< 对齐人脸尺寸不是 96×96（绝不 resize）
    kInvalidFaceType,       ///< 对齐人脸类型不是 CV_8UC3（绝不 convert）
    kNonFiniteFaceValue,    ///< Adapter 收到的人脸 float buffer 包含 NaN 或 Inf
    kEmptySourceImage,      ///< Wav2Lip 人脸裁剪收到空原图
    kInvalidSourceImageType,///< Wav2Lip 人脸裁剪原图不是 CV_8UC3
    kInvalidFaceRect,       ///< 检测框无效或 padding 后没有有效区域
    kInvalidFacePadding,    ///< padding 为负数
    kInvalidFaceLandmarks,  ///< 关键点数量不符合 68 点契约
    kOpenCvError,           ///< 裁剪、缩放或仿射矩阵计算失败

    // ---- Mel 音频校验 ----
    kInvalidMelChunkSize,   ///< Mel chunk 长度不是 1280（80×16）
    kInvalidMelLayout,      ///< Mel 布局异常（预留扩展）
    kNonFiniteMelValue,     ///< Mel chunk 包含 NaN 或 Inf

    // ---- 其他 ----
    kInvalidMetadata,       ///< 预留：当前 metadata 只透传，不做合法性判断
    kAllocationFailed,      ///< 内存分配失败
    kUnknownError           ///< 未知错误（兜底）
};

/// @brief 下半脸遮挡策略
/// @note  当前仅支持 Wav2Lip 的 lower-half 一刀切，预留自定义矩形接口
enum class FaceMaskPolicy {
    kWav2LipLowerHalf       ///< 默认：height/2 以下全部置零
};

/// @brief 透传的时间元数据
/// @note  Builder 不计算 pts/frame_index，只从上游原样传递到下游
struct ModelInputMetadata {
    std::optional<int64_t> pts_ms;       ///< 可选：渲染时间戳（毫秒）
    std::optional<int64_t> frame_index;  ///< 可选：帧序号
};

/// @brief Wav2Lip 官方推理风格的人脸裁剪选项。
/// @note 默认只在检测框下方增加 10 像素，用于包含下巴；不会根据双眼旋转人脸。
struct Wav2LipFacePrepareOptions {
    int pad_top = 0;
    int pad_bottom = 10;
    int pad_left = 0;
    int pad_right = 0;
};

/// @brief Wav2Lip 人脸裁剪产物及回贴所需的同一套坐标关系。
struct Wav2LipPreparedFace {
    cv::Mat face_bgr;                       ///< 96×96 CV_8UC3 BGR 模型输入脸
    cv::Rect source_crop_rect;              ///< 原图中实际使用的裁剪框
    cv::Mat transform;                      ///< 原图坐标到 96×96 坐标的 2×3 矩阵
    cv::Mat inverse_transform;              ///< 96×96 坐标回到原图的 2×3 矩阵
    std::vector<cv::Point2f> landmarks_96;  ///< 与 face_bgr 同坐标系的 68 点
};

/// @brief Wav2Lip 人脸裁剪统一结果。
struct Wav2LipFacePrepareResult {
    bool success = false;
    ModelInputStatus status = ModelInputStatus::kUnknownError;
    std::string error_message;
    Wav2LipPreparedFace value;
    double time_ms = 0.0;
};

/// @brief 后端无关的 Wav2Lip 语义输入数据
/// @note  不依赖 ncnn，纯 float buffer + 元数据。换推理框架时本结构体不变。
struct Wav2LipInputData {
    std::vector<float> face_chw;         ///< 六通道人脸 [6×96×96=55296]，CHW 排列，masked(0-2)+original(3-5)
    std::vector<float> mel_freq_time;    ///< Mel 频谱 [80×16=1280]，freq-major 排列
    ModelInputMetadata metadata;         ///< 透传的时间元数据
};

/// @brief 输入构建的诊断信息
struct Wav2LipInputInfo {
    int face_channels = 0;      ///< 实际输出的人脸通道数
    int face_height = 0;        ///< 实际输出的人脸高度
    int face_width = 0;         ///< 实际输出的人脸宽度
    int mel_bins = 0;           ///< 实际输出的 Mel 频率 bin 数
    int mel_frames = 0;         ///< 实际输出的 Mel 时间帧数
    float face_min_value = 0.0f;///< 人脸数据的最小值
    float face_max_value = 0.0f;///< 人脸数据的最大值
    float mel_min_value = 0.0f; ///< Mel 数据的最小值
    float mel_max_value = 0.0f; ///< Mel 数据的最大值
    int mask_start_row = 0;     ///< 下半脸 mask 起始行
    bool has_nan_or_inf = false;///< 输出是否包含 NaN 或 Inf
};

/// @brief 输入构建结果，包含成功/失败标志、语义数据、诊断信息和耗时
/// @note  成功时 data 非空且 info 与真实数据一致；失败时 data 为空
struct Wav2LipInputResult {
    bool success = false;                                        ///< 是否构建成功（默认失败）
    ModelInputStatus status = ModelInputStatus::kUnknownError;   ///< 具体状态码
    std::string error_message;                                   ///< 人类可读的错误描述
    Wav2LipInputData data;                                       ///< 语义输入数据
    Wav2LipInputInfo info;                                       ///< 诊断信息
    double time_ms = 0.0;                                        ///< 构建耗时（毫秒）
};

/// @brief 模型输入构建器（Wav2Lip Input Builder）
///
/// 负责把上游模块的产出转换为后端无关的 Wav2Lip 语义输入数据。
/// - 人脸：CV_8UC3 BGR [0,255] → 6 通道 float [0,1]（masked BGR + original BGR）
/// - 音频：1280 float freq-major → 原样透传（只校验不修改）
///
/// 核心设计：
/// - 翻译官 + 守门员：格式转换 + 契约校验。不满足契约直接拒绝，不偷偷修正。
/// - 后端无关：产出纯 float buffer（Wav2LipInputData），不依赖 ncnn。
///   换推理框架只需换 Adapter，Builder 不动。
/// - 下半脸 mask：固定 height/2 一刀切（Builder 不知道嘴的具体位置），
///   Wav2LipModelSpec 集中管理此常量。
///
/// 职责边界：
/// - 不做 resize / 颜色空间转换 / 灰度→BGR（Fail Fast）
/// - 不做人脸检测；PrepareFace 只消费已经确认的检测框和 68 点
/// - 不重新计算 Mel / FFT / 重采样（上游模块的事）
/// - 不做 ncnn 内存适配（NcnnInputAdapter 的事）
/// - 不做推理（ModelInference 的事）
class Wav2LipInputBuilder {
public:
    /// @brief 按 Wav2Lip 官方推理方式准备人脸：检测框 padding、裁剪、缩放到 96×96。
    /// @note 不做双眼旋转；返回的关键点和 inverse_transform 与模型输入使用同一坐标关系。
    Wav2LipFacePrepareResult PrepareFace(
        const cv::Mat& source_bgr,
        const cv::Rect& detected_face,
        const std::vector<cv::Point>& landmarks,
        const Wav2LipFacePrepareOptions& options =
            Wav2LipFacePrepareOptions()) const;

    /// @brief 构建 Wav2Lip 语义输入数据
    /// @param prepared_face PrepareFace 的产出：CV_8UC3、BGR、96×96、[0,255]
    /// @param freq_major_mel_chunk MelFeatureExtractor 的产出：1280 float、freq-major
    /// @param metadata 可选的时间元数据（pts_ms / frame_index），只透传
    /// @return Wav2LipInputResult，成功时 data 非空、info 与真实数据一致
    Wav2LipInputResult Build(
        const cv::Mat& prepared_face,
        const std::vector<float>& freq_major_mel_chunk,
        const ModelInputMetadata& metadata = ModelInputMetadata()) const;

    /// @brief 状态码 → 人类可读字符串
    /// @param status 状态码
    /// @return 非空字符串（覆盖所有状态码）
    static std::string StatusToString(ModelInputStatus status);
};

} // namespace model
} // namespace digital_human
