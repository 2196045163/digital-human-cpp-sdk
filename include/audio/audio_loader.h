#pragma once

#include <cstdint>   // int16_t, int64_t
#include <memory>    // std::unique_ptr
#include <string>    // std::string
#include <vector>    // std::vector

namespace digital_human {
namespace audio {

/// @brief 音频加载状态码，每个失败点对应一个明确的枚举值，便于错误定位
/// @note  不使用简单的 bool 返回值，让调用方能区分"路径错误/无音频流/解码器缺失/重采样失败"等
enum class AudioLoadStatus {
    kOk,                        ///< 加载成功
    kEmptyPath,                 ///< 传入路径为空字符串
    kFileNotFound,              ///< 文件不存在
    kPathIsDirectory,           ///< 路径指向目录而非文件
    kUnsupportedFormat,         ///< 文件扩展名不在支持列表中
    kInvalidTargetSampleRate,   ///< 目标采样率 ≤ 0
    kInvalidTargetChannels,     ///< 目标声道数非法（当前仅支持 1）
    kOpenInputFailed,           ///< avformat_open_input 失败（容器无法打开）
    kFindStreamInfoFailed,      ///< avformat_find_stream_info 失败（流信息探测失败）
    kNoAudioStream,             ///< 容器中不存在音频流
    kDecoderNotFound,           ///< 找不到对应的解码器（FFmpeg 未编译该编码支持）
    kCodecContextAllocFailed,   ///< avcodec_alloc_context3 分配失败
    kCodecParametersFailed,     ///< avcodec_parameters_to_context 复制参数失败
    kOpenDecoderFailed,         ///< avcodec_open2 打开解码器失败
    kInvalidSourceAudioParams,  ///< 源音频参数异常（采样率=0 或声道数=0）
    kResamplerAllocFailed,      ///< swr_alloc_set_opts 创建重采样器失败
    kResamplerInitFailed,       ///< swr_init 初始化重采样器失败
    kPacketAllocFailed,         ///< av_packet_alloc 分配失败
    kFrameAllocFailed,          ///< av_frame_alloc 分配失败
    kDecodeFailed,              ///< 解码过程中出错
    kResampleFailed,            ///< 重采样过程中出错
    kFlushFailed,               ///< flush 解码器或重采样器时出错
    kNoSamplesDecoded,          ///< 解码完成但 PCM 数组为空
    kInvalidPcmData,            ///< float PCM 包含 NaN 或 Inf 异常值
    kFileSystemError,           ///< 文件系统操作异常
    kUnknownError               ///< 未知错误（兜底）
};

/// @brief 采样格式：AudioLoader 对外支持的两种 PCM 数据类型
enum class AudioSampleFormat {
    kInt16,     ///< signed 16-bit 整数，范围 [-32768, 32767]，用于调试和 WAV 对照
    kFloat32    ///< IEEE 754 单精度浮点，范围约 [-1.0, 1.0]，用于后续算法链路
};

/// @brief 加载选项，控制 AudioLoader 的行为
/// @note  所有字段都有默认值，调用方可只改关心的部分
struct AudioLoadOptions {
    int target_sample_rate = 16000;                                ///< 目标采样率（Hz），默认 16000
    int target_channels = 1;                                       ///< 目标声道数，固定为 1（mono）
    AudioSampleFormat output_format = AudioSampleFormat::kFloat32; ///< 主输出格式，默认 float
    bool enable_format_check = true;                               ///< 是否先用扩展名过滤不支持格式
    bool collect_stats = true;                                     ///< 是否统计 min/max/RMS 等信息
    int64_t max_samples = 0;                                       ///< 最多解码多少采样点（0 = 不限制）
    int chunk_size_samples = 0;                                    ///< 预留：分块大小（0 = 一次性输出全部）
};

/// @brief 音频统计信息，用于判断 PCM 质量
struct AudioStats {
    double min_sample = 0.0;        ///< 最小采样值
    double max_sample = 0.0;        ///< 最大采样值
    double mean_abs = 0.0;          ///< 平均绝对值
    double rms = 0.0;               ///< 均方根值（Root Mean Square），反映整体音量
    double zero_ratio = 0.0;        ///< 零采样点占比（值过大说明音频可能全静音）
    bool has_nan_or_inf = false;    ///< 是否包含 NaN 或 Inf 异常值
};

/// @brief 音频基础信息，记录源文件和输出结果的完整参数
/// @note  分为"源文件信息"和"输出信息"两组，同时保留原始和转换后的参数便于调试
struct AudioInfo {
    // ---- 源文件信息 ----
    std::string file_path;               ///< 原始文件路径
    std::string container_format;        ///< 容器格式名称（如 "wav", "mp4"）
    std::string codec_name;              ///< 编解码器名称（如 "pcm_s16le", "aac"）
    int stream_index = -1;               ///< 音频流在容器中的索引（-1 = 未找到）
    int source_sample_rate = 0;          ///< 源采样率（Hz）
    int source_channels = 0;             ///< 源声道数
    std::string source_sample_format;    ///< 源采样格式名称（如 "s16", "fltp"）
    int64_t source_bit_rate = 0;         ///< 源比特率（bps），可能为 0 表示未知
    double source_duration_sec = 0.0;    ///< 源时长（秒），由容器记录，可能不精确

    // ---- 输出信息 ----
    int target_sample_rate = 16000;                              ///< 目标采样率（Hz）
    int target_channels = 1;                                     ///< 目标声道数
    AudioSampleFormat target_sample_format = AudioSampleFormat::kFloat32; ///< 目标采样格式
    int64_t sample_count = 0;                                    ///< 输出总采样点数
    double duration_sec = 0.0;                                   ///< 输出时长（秒）= sample_count / target_sample_rate
    AudioStats stats;                                            ///< 输出的统计信息
};

/// @brief PCM 数据 + 元信息，把数据数组和它的采样率/声道/格式绑定在一起
/// @note  下游模块拿到 AudioData 就能直接知道怎么解释 pcm 数组，无需额外传递参数
struct AudioData {
    std::vector<float> pcm;                                    ///< float PCM 样本数组，范围约 [-1.0, 1.0]
    int sample_rate = 16000;                                   ///< 采样率（Hz）
    int channels = 1;                                          ///< 声道数
    AudioSampleFormat format = AudioSampleFormat::kFloat32;    ///< 采样格式
};

/// @brief 加载结果（float PCM），包含成功/失败标志、PCM 数据、音频信息、耗时
/// @note  无论是否成功都返回此结构体，失败时 success=false、error_message 有值
struct AudioLoadResult {
    bool success = false;                                    ///< 是否加载成功
    AudioLoadStatus status = AudioLoadStatus::kUnknownError; ///< 具体状态码
    std::string error_message;                               ///< 人类可读的错误描述
    AudioData audio;                                         ///< PCM 数据（float）
    AudioInfo info;                                          ///< 音频信息
    double time_ms = 0.0;                                    ///< 加载耗时（毫秒），从入口计时到返回
};

/// @brief 加载结果（int16 PCM），与 AudioLoadResult 独立定义，避免同时持有两份大数组占内存
struct AudioInt16LoadResult {
    bool success = false;                                    ///< 是否加载成功
    AudioLoadStatus status = AudioLoadStatus::kUnknownError; ///< 具体状态码
    std::string error_message;                               ///< 人类可读的错误描述
    std::vector<int16_t> pcm;                                ///< PCM 数据（int16），范围 [-32768, 32767]
    AudioInfo info;                                          ///< 音频信息
    double time_ms = 0.0;                                    ///< 加载耗时（毫秒）
};

/// @brief 音频加载模块（Audio Loader）
///
/// 本模块是音频处理链路的统一输入入口。负责使用 FFmpeg 打开本地音频文件，
/// 完成解封装 → 解码 → 重采样 → 声道转换 → 格式转换，
/// 最终输出 16kHz / mono / float PCM（正式链路）和 int16 PCM（调试验证）。
///
/// 职责边界：
/// - 不做音频预加重/音量归一化/VAD（属于 AudioPreprocessor）
/// - 不做音频分帧（属于 AudioFramer）
/// - 不做 FFT / Mel 频谱（属于 MelFeatureExtractor）
/// - 不做麦克风实时采集/播放/音视频同步
///
/// 使用 PImpl 模式隐藏 FFmpeg 头文件依赖，用户代码无需引入 FFmpeg。
/// 禁止拷贝（内部持有 FFmpeg 资源不可浅复制），允许移动。
class AudioLoader {
public:
    /// @param target_sample_rate 目标采样率，默认 16000 Hz（匹配 Wav2Lip 类模型要求）
    explicit AudioLoader(int target_sample_rate = 16000);
    ~AudioLoader();

    // 禁止拷贝
    AudioLoader(const AudioLoader&) = delete;
    AudioLoader& operator=(const AudioLoader&) = delete;

    // 允许移动
    AudioLoader(AudioLoader&&) noexcept;
    AudioLoader& operator=(AudioLoader&&) noexcept;

    // =================== 主要接口 =========================

    /// @brief 主加载接口，返回 float PCM [-1.0, 1.0]，正式算法链路使用
    /// @param file_path 音频文件路径
    /// @param options 加载选项
    /// @return AudioLoadResult，包含 float PCM、音频信息和状态码
    AudioLoadResult LoadFromFile(
        const std::string& file_path,
        const AudioLoadOptions& options = AudioLoadOptions()
    );

    /// @brief 加载为 int16 PCM [-32768, 32767]，用于调试和 WAV 对照
    /// @param file_path 音频文件路径
    /// @param options 加载选项
    /// @return AudioInt16LoadResult，包含 int16 PCM
    AudioInt16LoadResult LoadInt16FromFile(
        const std::string& file_path,
        const AudioLoadOptions& options = AudioLoadOptions()
    );

    /// @brief 批量加载，对每个路径调用 LoadFromFile
    /// @param file_paths 音频文件路径列表
    /// @param options 加载选项（对每个文件复用同一组选项）
    /// @return 结果列表，下标与输入一一对应，失败项保留失败信息不清除
    std::vector<AudioLoadResult> LoadBatch(
        const std::vector<std::string>& file_paths,
        const AudioLoadOptions& options = AudioLoadOptions()
    );

    /// @brief 探测模式：只读取音频基础信息（采样率/声道/编码/时长），不解码完整 PCM
    /// @param file_path 音频文件路径
    /// @param options 加载选项（probe 模式下 affect_stats/format_check 等字段仍生效）
    /// @return AudioLoadResult，audio.pcm 为空，info 有值
    /// @note  适合先快速扫描文件信息，再决定是否完整解码
    AudioLoadResult Probe(
        const std::string& file_path,
        const AudioLoadOptions& options = AudioLoadOptions()
    );

    /// @brief 将 int16 PCM 转换为 float PCM
    /// @param pcm_int16 源 int16 PCM 数据
    /// @return float PCM，公式：out[i] = in[i] / 32768.0f
    /// @note  空输入返回空 vector；不做范围 clamp
    static std::vector<float> ConvertInt16ToFloat(
        const std::vector<int16_t>& pcm_int16
    );

    // ===================================================


    /// @brief 获取构造时设置的目标采样率
    int GetTargetSampleRate() const;

    /// @brief 通过扩展名判断是否为支持的音频格式（不区分大小写）
    /// @param file_path 文件路径
    /// @return true 表示扩展名在支持列表中
    static bool IsSupportedFormat(const std::string& file_path);

    /// @brief 返回支持的扩展名列表
    /// @return 小写含点号的扩展名列表，如 {".wav", ".mp3", ".aac", ".m4a", ".mp4", ".flac"}
    static std::vector<std::string> GetSupportedExtensions();

    /// @brief 将状态码转为人类可读字符串，用于日志和 example 打印
    /// @param status 状态码
    /// @return 非空字符串（覆盖所有状态码）
    static std::string StatusToString(AudioLoadStatus status);

private:
    struct Impl;                      ///< 前向声明，实现在 .cpp 中隐藏 FFmpeg 头文件
    std::unique_ptr<Impl> pImpl_;     ///< PImpl 惯用法，外部使用者看不到 FFmpeg 类型
};

} // namespace audio
} // namespace digital_human
