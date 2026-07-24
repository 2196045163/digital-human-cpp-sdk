#pragma once

#include <cstdint>        // int64_t
#include <memory>
#include <opencv2/core.hpp>  // cv::Mat
#include <string>
#include <vector>

namespace digital_human {
namespace audio {

/// @brief Mel 特征提取状态码，每个失败点对应一个明确的枚举值
enum class MelFeatureStatus {
    kOk,                     ///< 提取成功
    kEmptyFrame,             ///< 输入单帧为空
    kEmptyBatch,             ///< 批量输入为空
    kInvalidSampleRate,      ///< 采样率 ≤ 0
    kInvalidFftSize,         ///< n_fft ≤ 0
    kInvalidMelCount,        ///< n_mels ≤ 0
    kInvalidFrequencyRange,   ///< fmin ≥ fmax 或 fmax > sample_rate/2
    kInvalidNormalizeRange,  ///< 归一化参数非法
    kInvalidFrameData,       ///< 输入帧包含 NaN 或 Inf
    kMelBasisInitFailed,     ///< Mel 滤波器组构建失败
    kDftFailed,              ///< OpenCV DFT 执行失败
    kNoFeaturesGenerated,    ///< 提取完成但结果为空
    kInvalidChunkSize,       ///< chunk 大小非法
    kInvalidChunkLayout,     ///< chunk 展平方式无效
    kUnknownError            ///< 未知错误（兜底）
};

/// @brief 频谱模式：幅度谱或能量谱
/// @note  幅度谱 + 20×log10 与能量谱 + 10×log10 是等价的，但不要混用
enum class MelSpectrumMode {
    kMagnitude,  ///< 幅度谱：magnitude = sqrt(re² + im²)，配合 20×log10
    kPower       ///< 能量谱：power = magnitude²，配合 10×log10
};

/// @brief Mel 归一化模式
enum class MelNormalizeMode {
    kNone,              ///< 不归一化，直接返回原始 Mel 值
    kDb,                ///< dB 转换：20×log10(Mel)，clip 到 [min_level_db, max_level_db]
    kZeroOne,           ///< 线性缩放到 [0, 1]
    kWav2LipSymmetric   ///< Wav2Lip 对称归一化：除以 max_abs_value，clip [-4, 4]（默认）
};

/// @brief Mel 频率刻度。
enum class MelScale {
    kHtk,       ///< HTK 对数公式
    kSlaney     ///< librosa 默认的 Slaney 分段公式
};

/// @brief Mel 三角滤波器归一化方式。
enum class MelFilterNormalization {
    kNone,
    kSlaney     ///< 按相邻 Mel 边界带宽做面积归一化（librosa 默认）
};

/// @brief Mel 提取选项，控制 FFT / Mel 滤波 / 归一化的全部参数
/// @note  推荐通过 Wav2LipDefault() 或 SpeechDefault() 获取配置
struct MelFeatureOptions {
    int sample_rate = 16000;                              ///< 采样率（Hz）
    int n_fft = 800;                                      ///< FFT 点数，需与 AudioFramer frame_size 一致
    int n_mels = 80;                                      ///< Mel 滤波器个数（输出维度）
    float fmin = 55.0f;                                   ///< 最低频率（Hz）
    float fmax = 7600.0f;                                 ///< 最高频率（Hz）

    float min_level_db = -100.0f;                         ///< dB 模式下 clip 下界
    float ref_level_db = 20.0f;                           ///< dB 参考值
    float amin = 1e-5f;                                   ///< 幅度下限，防止 log(0)
    float max_abs_value = 4.0f;                           ///< Wav2Lip 对称归一化的 clip 值

    MelSpectrumMode spectrum_mode = MelSpectrumMode::kMagnitude;   ///< 幅度谱 / 能量谱
    MelNormalizeMode normalize_mode = MelNormalizeMode::kWav2LipSymmetric; ///< 归一化方式
    MelScale mel_scale = MelScale::kSlaney;                    ///< Wav2Lip 官方 librosa 默认刻度
    MelFilterNormalization filter_normalization =
        MelFilterNormalization::kSlaney;                       ///< Wav2Lip 官方滤波器归一化
    bool validate_finite = true;                          ///< 是否检查输入帧含 NaN/Inf
    bool return_debug = false;                            ///< 是否返回调试信息（mel_basis / 一半频谱）
};

/// @brief Mel 提取结果统计信息
struct MelFeatureInfo {
    int sample_rate = 16000;                              ///< 采样率（Hz）
    int n_fft = 0;                                        ///< FFT 点数
    int n_fft_bins = 0;                                   ///< 有效频率 bin 数 = n_fft/2+1
    int n_mels = 0;                                       ///< Mel 维数
    int num_frames = 0;                                   ///< 输入帧数
    int rows = 0;                                         ///< mel 矩阵行数 = num_frames
    int cols = 0;                                         ///< mel 矩阵列数 = n_mels
    float fmin = 0.0f;                                    ///< 最低频率（Hz）
    float fmax = 0.0f;                                    ///< 最高频率（Hz）
    float frequency_resolution_hz = 0.0f;                 ///< 频率分辨率（Hz）= sample_rate / n_fft
    float min_value = 0.0f;                               ///< mel 矩阵最小值
    float max_value = 0.0f;                               ///< mel 矩阵最大值
    bool has_nan_or_inf = false;                          ///< 是否包含异常值
    MelNormalizeMode normalize_mode = MelNormalizeMode::kWav2LipSymmetric; ///< 实际归一化方式
    MelSpectrumMode spectrum_mode = MelSpectrumMode::kMagnitude; ///< 实际频谱模式
    MelScale mel_scale = MelScale::kSlaney;               ///< 实际 Mel 频率刻度
    MelFilterNormalization filter_normalization =
        MelFilterNormalization::kSlaney;                  ///< 实际滤波器归一化
};

/// @brief Mel 提取统一返回结果
struct MelFeatureResult {
    bool success = false;                                      ///< 是否成功
    MelFeatureStatus status = MelFeatureStatus::kUnknownError; ///< 具体状态码
    std::string error_message;                                 ///< 人类可读的错误描述
    cv::Mat mel;                                               ///< Mel 矩阵：[1, n_mels]（单帧）或 [num_frames, n_mels]（批量）
    MelFeatureInfo info;                                       ///< 统计信息
    cv::Mat mel_basis;       ///< 调试：Mel 滤波器组 [n_mels, n_fft_bins]（return_debug=true 时填充）
    cv::Mat last_spectrum;   ///< 调试：最后一帧的一半频谱（return_debug=true 时填充）
    double time_ms = 0.0;                                      ///< 耗时（毫秒）
};

/// @brief chunk 展平顺序
enum class MelChunkLayout {
    kFreqMajor80x16,   ///< 频率优先：[f0_t0, f0_t1, ..., f0_t15, f1_t0, ...] → ncnn [1,1,80,16]
    kTimeMajor16x80    ///< 时间优先：[f0_t0, f1_t0, ..., f79_t0, f0_t1, ...]
};

/// @brief Wav2Lip chunk 切块选项
struct MelChunkOptions {
    int chunk_size = 16;                                   ///< 每个 chunk 多少帧（Wav2Lip 固定 16）
    int hop = 1;                                           ///< chunk 间帧移（1 = 逐帧滑动）
    MelChunkLayout layout = MelChunkLayout::kFreqMajor80x16; ///< 展平顺序
    bool pad_tail = true;                                  ///< 不足 chunk_size 时是否补空 chunk
};

/// @brief Wav2Lip chunk 切块结果
struct MelChunkResult {
    bool success = false;                                      ///< 是否成功
    MelFeatureStatus status = MelFeatureStatus::kUnknownError; ///< 状态码
    std::string error_message;                                 ///< 错误描述
    std::vector<std::vector<float>> chunks;                    ///< chunk 列表，每个 chunk 大小 = chunk_size × n_mels
    int chunk_size = 16;                                       ///< 每块帧数
    int n_mels = 80;                                           ///< Mel 维度
    MelChunkLayout layout = MelChunkLayout::kFreqMajor80x16;  ///< 展平顺序
};

/// @brief 梅尔频谱特征提取模块（Mel Feature Extractor）
///
/// 本模块负责把 AudioFramer 输出的短时音频帧，从时域转换到 Mel 频率特征空间。
/// 流程：FFT → 幅度/能量谱 → Mel 滤波器组 → 对数/dB/归一化 → Mel 频谱图。
/// 可选：BuildWav2LipChunks 将 [T,80] 切为 16 帧 chunck → freq-major 展平为 1280 float。
///
/// 使用 PImpl 模式隐藏 OpenCV 实现细节。禁止拷贝，允许移动。
class MelFeatureExtractor {
public:
    /// @param options Mel 提取选项（推荐 Wav2LipDefault()）
    explicit MelFeatureExtractor(
        const MelFeatureOptions& options = Wav2LipDefault()
    );
    ~MelFeatureExtractor();

    // 禁止拷贝，允许移动
    MelFeatureExtractor(const MelFeatureExtractor&) = delete;
    MelFeatureExtractor& operator=(const MelFeatureExtractor&) = delete;
    MelFeatureExtractor(MelFeatureExtractor&&) noexcept;
    MelFeatureExtractor& operator=(MelFeatureExtractor&&) noexcept;

    /// @brief Wav2Lip 推荐配置：n_fft=800, n_mels=80, fmin=55, fmax=7600, 对称归一化 [-4,4]
    static MelFeatureOptions Wav2LipDefault();

    /// @brief 通用语音推荐配置：n_fft=400, n_mels=40, fmin=80, fmax=8000, dB 归一化
    static MelFeatureOptions SpeechDefault();

    /// @brief 单帧 Mel 提取：一帧 PCM → FFT → Mel 滤波 → 归一化 → [1, n_mels]
    /// @param frame 已加窗的单帧 PCM（来自 AudioFramer::FrameSamplesOnly 的单帧）
    /// @return MelFeatureResult，mel 矩阵 shape 为 [1, n_mels]
    MelFeatureResult Extract(const std::vector<float>& frame) const;

    /// @brief 批量 Mel 提取：多帧 PCM → 逐帧 FFT+Mel → [num_frames, n_mels]
    /// @param frames 多帧 PCM（来自 AudioFramer::FrameSamplesOnly 的输出）
    /// @return MelFeatureResult，mel 矩阵 shape 为 [num_frames, n_mels]
    MelFeatureResult ExtractBatch(
        const std::vector<std::vector<float>>& frames
    ) const;

    /// @brief 便捷接口：单帧提取 → 直接返回 float vector（供调试/验证）
    /// @note 失败时返回空 vector；正式链路需要错误原因时请使用 Extract() 获取 MelFeatureResult
    std::vector<float> ExtractVector(
        const std::vector<float>& frame
    ) const;

    /// @brief 流式 Mel 攒帧：每来一帧 [1, n_mels] 就往内部 buffer 追加
    /// @param mel_frame 单帧 Mel 特征 [n_mels]
    /// @note  内部维护 mel_buffer。可配合 AudioFramer::ProcessFrame 使用
    void PushMelFrame(const std::vector<float>& mel_frame);

    /// @brief 把内部攒的 mel_buffer 转为 cv::Mat [T, n_mels]，然后清空 buffer
    cv::Mat FlushMelFrames();

    /// @brief 流式 Wav2Lip chunk：从内部 mel_buffer 切 16 帧 chunk → freq-major 展平
    /// @return 如果攒够了 chunk_size 帧就返回一个 chunk，否则返回空 vector
    /// @note  调用前先通过 PushMelFrame 攒帧
    std::vector<float> TryPopMelChunk(const MelChunkOptions& options = MelChunkOptions());

    /// @brief Wav2Lip chunk 切块：将 [T, n_mels] 切成连续 16 帧 chunck → freq-major 展平
    /// @param mel_spectrogram 批量 Mel 频谱矩阵 [num_frames, n_mels]
    /// @param options chunk 选项（每块帧数/展平方式/尾块处理）
    /// @return MelChunkResult，每个 chunk 大小 = 16 × 80 = 1280
    MelChunkResult BuildWav2LipChunks(const cv::Mat& mel_spectrogram,
        const MelChunkOptions& options = MelChunkOptions()) const;

    /// @name 属性查询
    /// @{
    const MelFeatureOptions& GetOptions() const;
    int GetSampleRate() const;
    int GetFftSize() const;
    int GetMelBins() const;
    int GetFftBins() const;
    float GetFrequencyResolutionHz() const;
    /// @}

    /// @brief 获取预构建的 Mel 滤波器组矩阵，shape [n_mels, n_fft_bins]
    cv::Mat GetMelBasis() const;

    /// @brief Hz → Mel 转换
    static float HzToMel(float hz);

    /// @brief Mel → Hz 转换
    static float MelToHz(float mel);

    /// @brief 将状态码转为可读字符串
    static std::string StatusToString(MelFeatureStatus status);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;  ///< PImpl 惯用法，隐藏 OpenCV 实现细节
};

}   // namespace audio
}   // namespace digital_human
