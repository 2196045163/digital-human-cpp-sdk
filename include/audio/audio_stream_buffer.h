#pragma once

#include <cstdint>      // int64_t
#include <functional>   // std::function
#include <memory>       // std::unique_ptr
#include <string>       // std::string
#include <vector>       // std::vector

namespace digital_human {
namespace audio {

/// @brief 环形缓冲区状态码，每个失败点对应一个明确的枚举值
enum class AudioStreamBufferStatus {
    kOk,                    ///< 操作成功
    kInvalidCapacity,       ///< 容量 ≤ 0
    kInvalidSampleRate,     ///< 采样率 ≤ 0
    kInvalidDuration,       ///< 时长 ≤ 0（FromDuration 时）
    kEmptyInput,            ///< push 的输入数据为空
    kInvalidPcmData,        ///< PCM 包含 NaN 或 Inf
    kInvalidChunk,          ///< AudioChunk 的 pcm 为空或采样率非法
    kInvalidReadSize,       ///< pull 请求的样本数 = 0
    kInsufficientData,      ///< 非阻塞模式下数据不够
    kInsufficientSpace,     ///< 非阻塞模式下空间不够
    kTimeout,               ///< 等待超时
    kWouldBlock,            ///< 非阻塞模式下的瞬时状态（不是错误）
    kClosed,                ///< buffer 已关闭，push/pull 均返回此状态
    kUnsupportedOperation,  ///< 不支持的操作
    kInternalError          ///< 内部错误（兜底）
};

/// @brief 缓冲区溢出策略：满时如何处理新数据
enum class AudioBufferOverflowStrategy {
    kBlock,             ///< 阻塞等待，直到有空间（不丢数据）
    kOverwriteOldest,   ///< 覆盖最旧数据，腾出空间给新数据
    kDropNewest         ///< 丢弃新数据，直接返回失败
};

/// @brief 告警回调类型：occupancy(0~1), lost_samples(本次丢弃/覆盖量)
using WarningCallback = std::function<void(float occupancy, size_t lost_samples)>;

/// @brief 音频流缓冲配置选项
/// @note  支持两种构造方式：指定样本数或通过 FromDuration 按毫秒换算
struct AudioStreamBufferOptions {
    size_t capacity_samples = 0;                              ///< 缓冲区容量（采样点数）
    int sample_rate = 16000;                                  ///< 采样率（Hz），PTS 计算和毫秒换算时使用
    AudioBufferOverflowStrategy overflow_strategy = AudioBufferOverflowStrategy::kBlock; ///< 默认溢出策略
    int default_timeout_ms = 0;                               ///< 默认超时（0=永久阻塞，-1=非阻塞，>0=毫秒）
    bool enable_warning = true;                               ///< 是否启用告警回调
    bool validate_finite = true;                              ///< 是否检查 Push 数据含 NaN/Inf
};

/// @brief 带 PTS 时间戳的音频数据块
/// @note  PTS = Presentation Time Stamp（呈现时间戳），单位毫秒。
///        "某个时间点的音频数据应该在第几毫秒播放"。
///        例如 start_pts_ms=100 表示该块音频要在音频流的第 100ms 处播放。
///        下游音视频同步模块靠 PTS 对齐音频帧和视频帧的时间轴。
struct AudioChunk {
    std::vector<float> pcm;           ///< PCM 样本数据
    double start_pts_ms = 0.0;        ///< 起始 PTS（毫秒）= start_sample_index / sample_rate × 1000
    int sample_rate = 16000;          ///< 采样率（Hz）
    int channels = 1;                 ///< 声道数
    int64_t start_sample_index = 0;   ///< 在流中的起始采样索引
};

/// @brief 环形缓冲区实时统计信息
struct AudioStreamBufferStats {
    size_t capacity_samples = 0;               ///< 总容量（采样点数）
    size_t size_samples = 0;                   ///< 当前有效数据量
    size_t free_samples = 0;                   ///< 空闲空间
    double occupancy = 0.0;                    ///< 占用率 = size / capacity
    bool closed = false;                       ///< 是否已关闭

    int64_t total_pushed_samples = 0;          ///< 累计写入样本数
    int64_t total_pulled_samples = 0;          ///< 累计读取样本数
    int64_t total_dropped_samples = 0;         ///< 累计丢弃样本数（DropNewest）
    int64_t total_overwritten_samples = 0;     ///< 累计覆盖样本数（OverwriteOldest）

    int64_t push_count = 0;                    ///< 写入次数
    int64_t pull_count = 0;                    ///< 读取次数
    int64_t write_wrap_count = 0;              ///< 写指针回绕次数
    int64_t read_wrap_count = 0;               ///< 读指针回绕次数
    int64_t timeout_count = 0;                 ///< 超时次数
    int64_t high_watermark_count = 0;          ///< 高水位触发次数

    double stream_start_pts_ms = 0.0;          ///< 流起始 PTS（毫秒）
    int64_t next_read_sample_index = 0;        ///< 下次读取的采样索引
    int64_t next_write_sample_index = 0;       ///< 下次写入的采样索引
};

/// @brief Push 操作返回结果
struct AudioStreamPushResult {
    bool success = false;                                       ///< 是否成功
    AudioStreamBufferStatus status = AudioStreamBufferStatus::kInternalError; ///< 状态码
    std::string error_message;                                  ///< 错误描述

    size_t requested_samples = 0;                               ///< 请求写入的样本数
    size_t pushed_samples = 0;                                  ///< 实际写入的样本数
    size_t dropped_samples = 0;                                 ///< 被丢弃的样本数（DropNewest）
    size_t overwritten_samples = 0;                             ///< 被覆盖的样本数（OverwriteOldest）
    bool wrapped = false;                                       ///< 本次写入是否发生了回绕
    AudioStreamBufferStats stats;                               ///< 操作后的 buffer 统计快照
};

/// @brief Pull 操作返回结果
struct AudioStreamPullResult {
    bool success = false;                                       ///< 是否成功
    AudioStreamBufferStatus status = AudioStreamBufferStatus::kInternalError; ///< 状态码
    std::string error_message;                                  ///< 错误描述

    std::vector<float> pcm;                                     ///< 读出的 PCM 样本
    AudioChunk chunk;                                           ///< 带 PTS 的音频块
    size_t requested_samples = 0;                               ///< 请求读取的样本数
    size_t pulled_samples = 0;                                  ///< 实际读出的样本数
    bool wrapped = false;                                       ///< 本次读取是否发生了回绕
    AudioStreamBufferStats stats;                               ///< 操作后的 buffer 统计快照
};

/// @brief 音频流缓冲模块（Audio Stream Buffer）
///
/// 固定容量的环形缓冲区（Ring Buffer），为生产者和消费者之间提供流式 PCM 缓冲。
/// 支持三种溢出策略（Block/OverwriteOldest/DropNewest）、PTS 时间戳、告警回调。
/// 第一阶段单线程非阻塞，后续加入 mutex+condition_variable 做多线程阻塞。
///
/// 职责边界：
/// - 不做音频解码/重采样（属于 AudioLoader）
/// - 不做归一化/降噪/预加重/VAD（属于 AudioPreprocessor）
/// - 不做分帧/加窗（属于 AudioFramer）
/// - 不做 FFT/Mel（属于 MelFeatureExtractor）
///
/// 使用 PImpl 模式。禁止拷贝，允许移动。
class AudioStreamBuffer {
public:
    /// @param options 缓冲配置
    explicit AudioStreamBuffer(const AudioStreamBufferOptions& options);

    /// @brief 便捷构造：直接指定采样点数（其余用默认值）
    explicit AudioStreamBuffer(size_t capacity_samples);

    ~AudioStreamBuffer();

    // 禁止拷贝，允许移动
    AudioStreamBuffer(const AudioStreamBuffer&) = delete;
    AudioStreamBuffer& operator=(const AudioStreamBuffer&) = delete;
    AudioStreamBuffer(AudioStreamBuffer&&) noexcept;
    AudioStreamBuffer& operator=(AudioStreamBuffer&&) noexcept;

    /// @brief 写入 PCM 样本（生产者）
    /// @param samples  待写入的样本数据
    /// @param strategy 溢出策略
    /// @param timeout_ms 超时时间（0=永久阻塞，-1=非阻塞，>0=毫秒）
    /// @return Push 结果，含实际写入量和统计
    AudioStreamPushResult PushSamples(
        const std::vector<float>& samples,
        AudioBufferOverflowStrategy strategy = AudioBufferOverflowStrategy::kBlock,
        int timeout_ms = 0
    );

    /// @brief 读取 PCM 样本（消费者）
    /// @param sample_count 期望读取的样本数
    /// @param timeout_ms   超时时间
    /// @return Pull 结果，含读出的 PCM 数据和统计
    AudioStreamPullResult PullSamples(
        size_t sample_count,
        int timeout_ms = 0
    );

    /// @brief 写入带 PTS 的音频块
    AudioStreamPushResult PushChunk(
        const AudioChunk& chunk,
        AudioBufferOverflowStrategy strategy = AudioBufferOverflowStrategy::kBlock,
        int timeout_ms = 0
    );

    /// @brief 读取带 PTS 的音频块
    AudioStreamPullResult PullChunk(
        size_t sample_count,
        int timeout_ms = 0
    );

    /// @brief 关闭 buffer，唤醒所有阻塞线程，后续 push/pull 立即返回 kClosed
    void Close();

    /// @brief 查询是否已关闭
    bool IsClosed() const;

    /// @brief 重置 buffer：清空数据 + 重置统计（不改变容量和配置）
    void Reset();

    /// @brief 清空 buffer 内数据（read_pos = write_pos, size = 0）
    void Clear();

    /// @name 状态查询
    size_t Size() const;
    size_t Capacity() const;
    size_t FreeSpace() const;
    double Occupancy() const;
    AudioStreamBufferStats GetStats() const;
    /// @}

    /// @brief 设置告警回调（溢出/覆盖/丢弃时触发，锁外调用）
    void SetWarningCallback(WarningCallback callback);

    /// @brief 按毫秒和采样率换算容量（样本数 = sample_rate × duration_ms / 1000）
    static AudioStreamBufferOptions FromDuration(
        int sample_rate,
        double duration_ms
    );

    /// @brief 将状态码转为可读字符串
    static std::string StatusToString(AudioStreamBufferStatus status);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;  ///< PImpl 惯用法
};

} // namespace audio
} // namespace digital_human
