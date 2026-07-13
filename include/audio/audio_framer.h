#pragma once

#include <cstdint>   // int64_t
#include <memory>    // std::unique_ptr
#include <string>    // std::string
#include <vector>    // std::vector

namespace digital_human {
namespace audio {

/// @brief 窗函数类型：决定每个音频帧的边缘衰减方式
/// @note  Hann 更为规范，历史文献中"Hanning"常指 Hann window
enum class AudioWindowType {
    kNone,      ///< 不加窗（矩形窗），硬切边缘，频谱泄漏最大
    kHamming,   ///< Hamming 窗：两端约 0.08，语音处理常用
    kHann       ///< Hann 窗：两端严格为 0，最平滑但丢失边缘信息
};

/// @brief 尾部策略：当 PCM 末尾不足一帧时如何处理
/// @note  不同策略会算出不同帧数，必须显式指定，避免口径不一致
enum class AudioTailPolicy {
    kCoverLastSample,   ///< 最后一帧覆盖最后一个真实样本（默认推荐）
    kStartEveryHop,     ///< 每个 hop_size 整数倍位置都起一帧
    kDropIncomplete     ///< 尾部不足一帧直接丢弃，不补零
};

/// @brief 分帧模块状态码，描述成功或失败原因
enum class AudioFrameStatus {
    kOk,                    ///< 分帧成功
    kEmptyInput,            ///< 输入 PCM 为空
    kInvalidSampleRate,     ///< 采样率 ≤ 0
    kInvalidFrameDuration,  ///< 帧长毫秒 ≤ 0
    kInvalidHopDuration,    ///< 帧移毫秒 ≤ 0
    kInvalidFrameSize,      ///< 计算出的 frame_size ≤ 0
    kInvalidHopSize,        ///< 计算出的 hop_size ≤ 0
    kInvalidTailPolicy,     ///< 尾部策略取值范围非法
    kInvalidWindowType,     ///< 窗函数类型取值范围非法
    kInvalidPcmData,        ///< PCM 包含 NaN 或 Inf
    kNoFramesGenerated,     ///< 计算完成但帧数为 0
    kUnknownError           ///< 未知错误（兜底）
};

/// @brief 分帧选项，控制帧长、帧移、窗函数和尾部策略
/// @note  推荐通过 SpeechDefault() 或 Wav2LipDefault() 获取配置，避免手填魔法数字
struct AudioFrameOptions {
    int sample_rate = 16000;                                  ///< 采样率（Hz），需与 AudioLoader 输出一致
    double frame_duration_ms = 25.0;                          ///< 每帧覆盖毫秒数
    double hop_duration_ms = 10.0;                            ///< 相邻帧起点间隔毫秒数
    AudioWindowType window_type = AudioWindowType::kHamming;  ///< 窗函数类型
    AudioTailPolicy tail_policy = AudioTailPolicy::kCoverLastSample; ///< 尾部策略
    bool validate_finite = true;                              ///< 是否检查 PCM 中 NaN/Inf
    bool return_window = false;                               ///< 是否在结果中返回窗函数数组（供调试）
};

/// @brief 单帧数据 + 在原始 PCM 中的位置元数据
/// @note  使用左闭右开区间 [start_sample, end_sample_exclusive) 表示采样点范围
struct AudioFrame {
    std::vector<float> samples;       ///< 本帧的采样点（已加窗），长度通常 = frame_size，尾部帧可能更短
    int index = 0;                    ///< 第几帧（0-based）
    int64_t start_sample = 0;         ///< 在原始 PCM 中的起始采样索引
    int64_t end_sample_exclusive = 0; ///< 本帧覆盖的右边界（不含），即 start_sample + frame_size；超出原 PCM 末尾时 > original_sample_count
    double start_ms = 0.0;            ///< 起始时间（毫秒）= start_sample / sample_rate × 1000
    double end_ms = 0.0;              ///< 结束时间（毫秒）= end_sample_exclusive / sample_rate × 1000
    bool contains_padding = false;    ///< 本帧是否包含补零采样点（帧尾部超出原始 PCM 末尾即为 true）
};

/// @brief 分帧后的整体统计信息
struct AudioFrameInfo {
    int sample_rate = 16000;                               ///< 采样率（Hz）
    int frame_size = 0;                                    ///< 每帧采样点数
    int hop_size = 0;                                      ///< 帧移采样点数
    int overlap_size = 0;                                  ///< 相邻帧重叠点数 = frame_size - hop_size
    int64_t original_sample_count = 0;                     ///< 原始 PCM 采样点数
    int64_t padded_sample_count = 0;                       ///< 补零后总采样点数
    int64_t pad_sample_count = 0;                          ///< 补零个数
    int num_frames = 0;                                    ///< 总帧数
    double frame_duration_ms = 0.0;                        ///< 帧长（毫秒）
    double hop_duration_ms = 0.0;                          ///< 帧移（毫秒）
    double original_duration_sec = 0.0;                    ///< 原始音频时长（秒）
    AudioWindowType window_type = AudioWindowType::kHamming; ///< 窗函数类型
    AudioTailPolicy tail_policy = AudioTailPolicy::kCoverLastSample; ///< 尾部策略
};

/// @brief 分帧统一返回结果
/// @note  无论成功与否都返回此结构体，失败时 success=false、error_message 有值
struct AudioFrameResult {
    bool success = false;                                    ///< 是否分帧成功
    AudioFrameStatus status = AudioFrameStatus::kUnknownError; ///< 具体状态码
    std::string error_message;                               ///< 人类可读的错误描述
    std::vector<AudioFrame> frames;                          ///< 分帧结果，每帧含数据和元数据
    AudioFrameInfo info;                                     ///< 分帧统计信息
    std::vector<float> window;                               ///< 窗函数数组（return_window=true 时填充）
    double time_ms = 0.0;                                    ///< 分帧耗时（毫秒）
};

/// @brief 音频分帧模块（Audio Framer）
///
/// 本模块负责把 AudioLoader 输出的连续 float PCM 数组，按固定帧长和帧移
/// 切分成一组可重叠的短时音频帧，并对每帧加窗，为后续 FFT / Mel 频谱分析做准备。
///
/// 职责边界：
/// - 不做音频文件解码/重采样/声道转换（属于 AudioLoader）
/// - 不做 FFT / Mel 滤波器组（属于 MelFeatureExtractor）
/// - 不做音量归一化/预加重/VAD（属于 AudioPreprocessor）
/// - 不关心模型 tensor 拼接（属于 InputProcessor）
///
/// 使用 PImpl 模式保持风格统一。禁止拷贝，允许移动。
class AudioFramer {
public:
    /// @param options 分帧选项（推荐使用 SpeechDefault() 或 Wav2LipDefault()）
    explicit AudioFramer(const AudioFrameOptions& options = SpeechDefault());
    ~AudioFramer();

    // 禁止拷贝，允许移动
    AudioFramer(const AudioFramer&) = delete;
    AudioFramer& operator=(const AudioFramer&) = delete;
    AudioFramer(AudioFramer&&) noexcept;
    AudioFramer& operator=(AudioFramer&&) noexcept;

    /// @brief 通用语音 DSP 推荐配置：25ms 帧长 / 10ms 帧移 / Hamming 窗
    static AudioFrameOptions SpeechDefault();

    /// @brief Wav2Lip 模型推荐配置：50ms 帧长 / 12.5ms 帧移（与 n_fft=800 对齐）
    static AudioFrameOptions Wav2LipDefault();

    /// @brief 主分帧接口：完整流程（校验→计算→补零→切片→加窗→返回元数据）
    /// @param pcm 输入 float PCM 数组（需来自 AudioLoader：16kHz/mono）
    /// @return AudioFrameResult，含 frames / info / window(可选) / status
    AudioFrameResult Frame(const std::vector<float>& pcm) const;

    /// @brief 便捷接口：只返回二维 float 数组，不含元数据（供 Mel 模块直接使用）
    /// @param pcm 输入 float PCM 数组
    /// @return [num_frames][frame_size] 的已加窗音频帧数据
    std::vector<std::vector<float>> FrameSamplesOnly(
        const std::vector<float>& pcm
    ) const;

    /// @brief 流式分帧：每次传入一块 PCM chunk，攒够 frame_size 时自动切帧并加窗返回
    /// @param chunk 输入 PCM 数据块（来自 AudioStreamBuffer::PullSamples 等）
    /// @return 本次新产生的音频帧（可能 0~多帧），尾部不足一帧的数据留在内部 buffer
    /// @note  内部维护 pcm_buffer，跨调用保留状态。用完后调 FlushFrames 取尾部残留帧
    std::vector<AudioFrame> ProcessFrame(const std::vector<float>& chunk);

    /// @brief 取流式残留帧：处理完所有 chunk 后，内部 buffer 剩余不足一帧时按 tail_policy 处理
    std::vector<AudioFrame> FlushFrames();

    /// @brief 重置流式状态：清空内部 pcm_buffer
    void ResetStreaming();

    /// @brief 从 PCM 中切取一帧（不含加窗），供调试和测试
    /// @param pcm        补零后的 PCM 数组
    /// @param start_index 本帧起始采样索引
    /// @param pad_tail    尾部不足时是否补零
    /// @return frame_size 个采样点
    std::vector<float> CreateFrame(
        const std::vector<float>& pcm,
        size_t start_index,
        bool pad_tail
    ) const;

    /// @brief 对单帧应用窗函数（原地修改）：frame[i] *= window[i]
    /// @param frame [in/out] 待加窗的帧数据
    void ApplyWindow(std::vector<float>& frame) const;

    /// @name 属性查询
    /// @{
    int GetSampleRate() const;
    int GetFrameSize() const;
    int GetHopSize() const;
    int GetOverlapSize() const;
    const AudioFrameOptions& GetOptions() const;
    /// @}

    /// @brief 从毫秒计算帧采样点数（四舍五入）
    static int ComputeFrameSize(int sample_rate, double frame_duration_ms);

    /// @brief 从毫秒计算帧移采样点数（四舍五入）
    static int ComputeHopSize(int sample_rate, double hop_duration_ms);

    /// @brief 根据尾部策略计算帧数
    /// @param sample_count PCM 采样点总数
    /// @param frame_size   每帧采样点数
    /// @param hop_size     帧移采样点数
    /// @param tail_policy  尾部策略
    static int ComputeNumFrames(
        int64_t sample_count,
        int frame_size,
        int hop_size,
        AudioTailPolicy tail_policy
    );

    /// @brief 预生成窗函数数组（只在构造时计算一次，避免每帧重复计算）
    /// @param frame_size  帧采样点数
    /// @param window_type 窗函数类型
    static std::vector<float> GenerateWindow(
        int frame_size,
        AudioWindowType window_type
    );

    /// @brief 将状态码转为可读字符串（用于日志/example 打印）
    static std::string StatusToString(AudioFrameStatus status);

private:
    struct Impl;                      ///< 前向声明，实现在 .cpp 中
    std::unique_ptr<Impl> pImpl_;     ///< PImpl 惯用法
};

} // namespace audio
} // namespace digital_human
