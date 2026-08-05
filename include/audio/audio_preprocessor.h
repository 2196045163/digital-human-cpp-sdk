#pragma once

#include <cstdint>   // int64_t
#include <memory>    // std::unique_ptr
#include <string>    // std::string
#include <vector>    // std::vector

namespace digital_human {
namespace audio {

/// @brief 预处理状态码，每个失败点对应一个明确的枚举值
enum class AudioPreprocessStatus {
    kOk,                        ///< 处理成功
    kEmptyInput,                ///< 输入 PCM 为空
    kInvalidSampleRate,         ///< 采样率 ≤ 0
    kInvalidTargetPeak,          ///< 目标峰值 ≤ 0
    kInvalidMaxGain,            ///< 最大增益 ≤ 0
    kInvalidDenoiseThreshold,   ///< 降噪阈值 ≥ 0（应为负值 dB）
    kInvalidPreEmphasisAlpha,   ///< 预加重系数不在 [0, 1) 范围内
    kInvalidVadFrameMs,         ///< VAD 帧长 ≤ 0
    kInvalidPcmData,            ///< PCM 包含 NaN 或 Inf
    kUnsupportedDenoiseMode,    ///< 选择了尚未实现的降噪模式（如谱减法）
    kInternalError              ///< 内部错误（兜底）
};

/// @brief 降噪模式：当前仅支持 None 和 NoiseGate，谱减法预留
enum class AudioDenoiseMode {
    kNone,                          ///< 不做降噪
    kNoiseGate,                     ///< 门限降噪：低于阈值的样本置零
    kSpectralSubtractionReserved    ///< 谱减法（预留，尚未实现）
};

/// @brief 预处理选项，控制归一化、降噪、预加重、VAD 的全部参数
/// @note  所有字段都有默认值，调用方可只改关心的部分
struct AudioPreprocessOptions {
    int sample_rate = 16000;                        ///< 采样率（Hz），需与 AudioLoader 输出一致

    // -- 归一化 --
    bool enable_normalize = true;                   ///< 是否启用峰值归一化
    float target_peak = 0.95f;                      ///< 目标峰值（归一化后最大绝对值）
    float max_gain = 20.0f;                         ///< 最大增益上限，防止静音被无限放大
    float silence_epsilon = 1e-6f;                  ///< 静音判定阈值：max_abs < 此值视为静音，跳过归一化
    bool clamp_output = true;                       ///< 归一化后是否 clip 到 [-1, 1]

    // -- 降噪 --
    bool enable_denoise = false;                    ///< 是否启用降噪
    AudioDenoiseMode denoise_mode = AudioDenoiseMode::kNoiseGate; ///< 降噪模式
    float denoise_threshold_db = -50.0f;            ///< 降噪阈值的 dB 值（负值，越小越严格）

    // -- 预加重 --
    bool enable_pre_emphasis = true;                ///< 是否启用预加重
    float pre_emphasis_alpha = 0.97f;               ///< 预加重系数，常用 0.97

    // -- VAD --
    bool enable_vad = false;                        ///< 是否启用语音活动检测
    int vad_frame_ms = 20;                          ///< VAD 每帧毫秒数
    float vad_threshold_scale = 0.5f;               ///< VAD 阈值缩放系数（乘以 RMS 均值）
    float vad_min_threshold = 0.01f;                ///< VAD 绝对最低阈值
    int vad_hangover_frames = 10;                   ///< VAD 挂起帧数（防止语音被切断）

    bool remove_non_speech = false;                 ///< 是否删除非语音段（默认不删，保持音画同步）
    bool collect_stats = true;                      ///< 是否统计输入/输出的 min/max/RMS 等信息
    bool validate_finite = true;                    ///< 是否检查 PCM 包含 NaN/Inf
};

/// @brief 语音段描述：VAD 检测到的一段连续语音的起止位置
struct SpeechSegment {
    int64_t start_sample = 0;               ///< 起始采样索引
    int64_t end_sample_exclusive = 0;       ///< 结束采样索引（不含）
    double start_ms = 0.0;                  ///< 起始时间（毫秒）
    double end_ms = 0.0;                    ///< 结束时间（毫秒）
    int start_frame = 0;                    ///< 起始 VAD 帧索引
    int end_frame_exclusive = 0;            ///< 结束 VAD 帧索引（不含）
};

/// @brief PCM 统计信息：遍历 PCM 计算 min/max/RMS/零样本占比/NaN/Inf
struct AudioPreprocessStats {
    int64_t num_samples = 0;                ///< 样本总数
    int sample_rate = 0;                    ///< 采样率（Hz）
    double duration_sec = 0.0;              ///< 时长（秒）

    float min_value = 0.0f;                 ///< 最小样本值
    float max_value = 0.0f;                 ///< 最大样本值
    float max_abs = 0.0f;                   ///< 最大绝对值
    float mean_abs = 0.0f;                  ///< 平均绝对值
    float rms = 0.0f;                       ///< 均方根值（Root Mean Square）
    double zero_ratio = 0.0;                ///< 零采样点占比

    bool has_nan = false;                   ///< 是否包含 NaN
    bool has_inf = false;                   ///< 是否包含 Inf
};

/// @brief 预处理流程的汇总信息：输入/输出统计、启用了哪些处理、耗时
struct AudioPreprocessInfo {
    AudioPreprocessStats input_stats;       ///< 输入 PCM 统计
    AudioPreprocessStats output_stats;      ///< 输出 PCM 统计

    bool normalized = false;                ///< 是否执行了归一化
    bool denoised = false;                  ///< 是否执行了降噪
    bool pre_emphasized = false;            ///< 是否执行了预加重
    bool vad_ran = false;                   ///< 是否执行了 VAD

    float applied_gain = 1.0f;              ///< 归一化实际应用的增益倍数
    float denoise_threshold_amp = 0.0f;     ///< 降噪阈值的线性幅度值
    int64_t processed_samples = 0;          ///< 实际处理的采样点数
    double process_time_ms = 0.0;           ///< 总处理耗时（毫秒）

    int speech_segment_count = 0;           ///< VAD 检测到的语音段数量
};

/// @brief 预处理统一返回结果
/// @note  无论成功与否都返回此结构体，失败时 success=false、error_message 有值
struct AudioPreprocessResult {
    bool success = false;                                       ///< 是否处理成功
    AudioPreprocessStatus status = AudioPreprocessStatus::kInternalError; ///< 具体状态码
    std::string error_message;                                  ///< 人类可读的错误描述

    std::vector<float> pcm;                                     ///< 处理后的输出 PCM
    AudioPreprocessInfo info;                                   ///< 处理统计信息
    std::vector<SpeechSegment> speech_segments;                 ///< VAD 检测到的语音段列表
};

/// @brief 音频预处理模块（Audio Preprocessor）
///
/// 本模块负责把 AudioLoader 输出的 float PCM 变成更稳定、更适合分帧和 Mel 特征的 PCM。
/// 默认处理顺序：归一化 → 门限降噪 → 预加重。VAD 作为独立分析能力提供。
///
/// 职责边界：
/// - 不做音频文件解码/重采样/声道转换（属于 AudioLoader）
/// - 不做分帧/加窗（属于 AudioFramer）
/// - 不做 FFT / Mel 滤波（属于 MelFeatureExtractor）
/// - 不默认删除静音段（保持音画同步）
///
/// 使用 PImpl 模式保持风格统一。禁止拷贝，允许移动。
class AudioPreprocessor {
public:
    /// @param options 预处理选项（所有字段都有默认值）
    explicit AudioPreprocessor(const AudioPreprocessOptions& options = AudioPreprocessOptions{});
    ~AudioPreprocessor();

    // 禁止拷贝，允许移动
    AudioPreprocessor(const AudioPreprocessor&) = delete;
    AudioPreprocessor& operator=(const AudioPreprocessor&) = delete;
    AudioPreprocessor(AudioPreprocessor&&) noexcept;
    AudioPreprocessor& operator=(AudioPreprocessor&&) noexcept;

    /// @brief 整段处理：归一化 → 降噪 → 预加重 → 统计（可选 VAD）
    /// @param pcm 输入 float PCM（来自 AudioLoader，16kHz/mono）
    /// @return AudioPreprocessResult，含处理后 PCM、统计、VAD 结果
    AudioPreprocessResult Process(const std::vector<float>& pcm);

    /// @brief 流式分块处理：保留跨块状态（预加重的上一个原始样本即块之间要记住"上一块最后一个原始样本"）
    /// @param frame 输入数据块
    /// @return AudioPreprocessResult，仅含当前块的处理结果和统计
    AudioPreprocessResult ProcessFrame(const std::vector<float>& frame);

    /// @brief VAD 语音活动检测：把 PCM 切成固定长度帧，逐帧算 RMS（均方根能量），
    ///        大于全局阈值判为语音帧，通过挂起帧数（语音结束后多等 N 帧才判结束，防换气停顿切碎语音段）
    ///        防短暂停顿切碎语音段
    /// @param pcm 输入 PCM
    /// @return 检测到的语音段列表
    std::vector<SpeechSegment> DetectSpeech(const std::vector<float>& pcm) const;

    /// @brief 计算 PCM 统计信息（min/max/RMS/零样本占比/NaN/Inf）
    AudioPreprocessStats ComputeStats(const std::vector<float>& pcm) const;

    /// @brief 重置流式状态（切换新音频时调用，清除上一段残留的最后一个样本，否则新音频的第一个样本会错误地用旧音频的最后一个样本做预加重）
    void ResetStreamingState();

    /// @name 选项查询与修改
    const AudioPreprocessOptions& GetOptions() const;
    void SetOptions(const AudioPreprocessOptions& options);

    /// @brief 将状态码转为可读字符串
    static std::string StatusToString(AudioPreprocessStatus status);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;  ///< PImpl 惯用法
};

}   // namespace audio
}   // namespace digital_human
