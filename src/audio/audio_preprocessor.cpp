#include <algorithm>   // std::max, std::min
#include <cmath>       // std::sqrt, std::isnan, std::isinf, std::log10, std::pow
#include <memory>

#include "audio/audio_preprocessor.h"

namespace digital_human {
namespace audio {

namespace {
    constexpr float kPcmMin = -1.0f;
    constexpr float kPcmMax = 1.0f;
    constexpr float kUnityGain = 1.0f;

    /// @brief 构造失败结果，避免错误路径中重复填充 success/status/error_message
    static AudioPreprocessResult MakeError(AudioPreprocessStatus status) {
        AudioPreprocessResult res;
        res.success = false;
        res.status = status;
        res.error_message = AudioPreprocessor::StatusToString(status);
        return res;
    }

    /// @brief 峰值归一化：把 PCM 的最大绝对值拉到 target_peak 附近，原地修改输入的pcm数据
    /// @return 实际应用的增益倍数
    static float NormalizePeak(std::vector<float>& pcm, float target_peak,
                                float max_gain, float silence_epsilon, bool clamp_output) {
        // 找最大绝对值
        float max_abs = 0.0f;
        for (auto v : pcm) {
            float av = std::abs(v);
            if (av > max_abs) {
                max_abs = av;
            }
        }

        // 静音保护：几乎没声音就不放大
        if (max_abs < silence_epsilon) {
            return kUnityGain;
        }

        // 算增益，受上限约束
        float gain = target_peak / max_abs;
        if (gain > max_gain) {
            gain = max_gain;
        }

        // 均匀放大 + clip 裁剪 [-1, 1]
        for (auto& v : pcm) {
            v *= gain;
            if (clamp_output) {
                if (v > kPcmMax) {
                    v = kPcmMax;
                }
                if (v < kPcmMin) {
                    v = kPcmMin;
                }
            }
        }
        return gain;
    }

    /// @brief 门限降噪：低于阈值（线性幅度）的样本直接置零，原地修改pcm
    /// @param pcm              [in/out] PCM 数据
    /// @param threshold_db     阈值（dB，一般值比较小为负值，如 -50）
    /// @return 阈值的线性幅度值（供 info 记录）
    static float ApplyNoiseGate(std::vector<float>& pcm, float threshold_db) {
        // dB → 线性幅度: amp = 10^(dB / 20)
        float threshold_amp = std::pow(10.0f, threshold_db / 20.0f);

        for (auto& v : pcm) {
            if (std::abs(v) < threshold_amp) { v = 0.0f; }
        }
        
        return threshold_amp;
    }

    /// @brief 整段预加重：y[n] = x[n] - alpha × x[n-1]，从后往前遍历避免覆盖未使用的值
    /// @param pcm   [in/out] PCM 数据
    /// @param alpha 预加重系数，常用 0.97
    static void ApplyPreEmphasis(std::vector<float>& pcm, float alpha) {
        if (pcm.empty()) { return; }
        // 从后往前：保证 x[n-1] 在本次迭代未被修改
        for (size_t i = pcm.size() - 1; i > 0; i--) {
            pcm[i] = pcm[i] - alpha * pcm[i - 1];
        }
        // pcm[0] 保持不变，因为无前值 即 x[-1] 不存在
    }
} // namespace

// ====== Impl ======
struct AudioPreprocessor::Impl {
    AudioPreprocessOptions options;

    // 流式预加重状态：跨块保留上一块最后一个归一化值（预加重前），做下一块首个样本的 x[n-1]
    float last_raw_sample = 0.0f;
    bool  has_last_raw_sample = false;
};

// ====== 构造/析构/移动 ======
AudioPreprocessor::AudioPreprocessor(const AudioPreprocessOptions& options)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->options = options;
}

AudioPreprocessor::~AudioPreprocessor() = default;
AudioPreprocessor::AudioPreprocessor(AudioPreprocessor&&) noexcept = default;
AudioPreprocessor& AudioPreprocessor::operator=(AudioPreprocessor&&) noexcept = default;

// ====== 静态工具 ======

std::string AudioPreprocessor::StatusToString(AudioPreprocessStatus status) {
    switch (status) {
        case AudioPreprocessStatus::kOk:                        return "成功";
        case AudioPreprocessStatus::kEmptyInput:                return "输入 PCM 为空";
        case AudioPreprocessStatus::kInvalidSampleRate:         return "无效的采样率";
        case AudioPreprocessStatus::kInvalidTargetPeak:         return "无效的目标峰值";
        case AudioPreprocessStatus::kInvalidMaxGain:            return "无效的最大增益";
        case AudioPreprocessStatus::kInvalidDenoiseThreshold:   return "无效的降噪阈值";
        case AudioPreprocessStatus::kInvalidPreEmphasisAlpha:   return "无效的预加重系数";
        case AudioPreprocessStatus::kInvalidVadFrameMs:         return "无效的 VAD 帧长";
        case AudioPreprocessStatus::kInvalidPcmData:            return "PCM 包含 NaN 或 Inf";
        case AudioPreprocessStatus::kUnsupportedDenoiseMode:    return "不支持的降噪模式";
        case AudioPreprocessStatus::kInternalError:             return "内部错误";
        default:                                                return "未定义错误";
    }
}

// ====== 音频预处理时的选项存取 ======

const AudioPreprocessOptions& AudioPreprocessor::GetOptions() const {
    return pImpl_->options;
}

void AudioPreprocessor::SetOptions(const AudioPreprocessOptions& options) {
    pImpl_->options = options;
    ResetStreamingState();  // 切换配置时清空跨块状态
}

// ====== 核心接口 ======

AudioPreprocessResult AudioPreprocessor::Process(const std::vector<float>& pcm) {
    AudioPreprocessResult result;

    // 1. 输入校验
    if (pcm.empty()) {
        result = MakeError(AudioPreprocessStatus::kEmptyInput);
        return result;
    }
    if (pImpl_->options.validate_finite) {
        for (auto v : pcm) {
            if (std::isnan(v) || std::isinf(v)) {
                result = MakeError(AudioPreprocessStatus::kInvalidPcmData);
                return result;
            }
        }
    }

    const auto& opt = pImpl_->options;
    auto& info = result.info;

    // 2. 统计输入
    if (opt.collect_stats) {
        info.input_stats = ComputeStats(pcm);
    }

    // 3. 拷贝输入到工作缓冲区
    std::vector<float> pcm_work = pcm;

    // 4. 归一化
    if (opt.enable_normalize) {
        info.applied_gain = NormalizePeak(pcm_work, opt.target_peak,
                                           opt.max_gain, opt.silence_epsilon, opt.clamp_output);
        info.normalized = true;
    }

    // 5. 门限降噪
    if (opt.enable_denoise && opt.denoise_mode == AudioDenoiseMode::kNoiseGate) {
        info.denoise_threshold_amp = ApplyNoiseGate(pcm_work, opt.denoise_threshold_db);
        info.denoised = true;
    }

    // 6. 整段预加重（从后往前，原地修改）
    if (opt.enable_pre_emphasis) {
        ApplyPreEmphasis(pcm_work, opt.pre_emphasis_alpha);
        info.pre_emphasized = true;
    }

    // 7. 统计输出
    if (opt.collect_stats) {
        info.output_stats = ComputeStats(pcm_work);
        info.processed_samples = static_cast<int64_t>(pcm_work.size());
    }

    result.success = true;
    result.status = AudioPreprocessStatus::kOk;
    result.pcm = std::move(pcm_work);
    return result;
}

AudioPreprocessResult AudioPreprocessor::ProcessFrame(const std::vector<float>& frame) {
    AudioPreprocessResult result;

    // 1. 输入校验
    if (frame.empty()) {
        result = MakeError(AudioPreprocessStatus::kEmptyInput);
        return result;
    }
    if (pImpl_->options.validate_finite) {
        for (auto v : frame) {
            if (std::isnan(v) || std::isinf(v)) {
                result = MakeError(AudioPreprocessStatus::kInvalidPcmData);
                return result;
            }
        }
    }

    const auto& opt = pImpl_->options;
    auto& info = result.info;

    // 2. 拷贝输入到工作缓冲区
    std::vector<float> pcm = frame;

    // 3. 归一化
    if (opt.enable_normalize) {
        info.applied_gain = NormalizePeak(pcm, opt.target_peak,
                                           opt.max_gain, opt.silence_epsilon, opt.clamp_output);
        info.normalized = true;
    }

    // 4. 门限降噪
    if (opt.enable_denoise && opt.denoise_mode == AudioDenoiseMode::kNoiseGate) {
        info.denoise_threshold_amp = ApplyNoiseGate(pcm, opt.denoise_threshold_db);
        info.denoised = true;
    }

    // 5. 流式预加重：块内从后往前（保证 x[n-1] 是归一化值），再单独处理首个样本
    if (opt.enable_pre_emphasis && !pcm.empty()) {
        float alpha = opt.pre_emphasis_alpha;

        // 5.1 保存预加重前的最后一个归一化值，跨块时用它做下一块首个样本的 x[-1]
        float last_normalized = pcm.back();

        // 5.2 块内从后往前遍历：此时 pcm[0] 尚未修改，和整段 ApplyPreEmphasis 行为一致
        for (size_t i = pcm.size() - 1; i > 0; i--) {
            pcm[i] = pcm[i] - alpha * pcm[i - 1];
        }

        // 5.3 最后处理首个样本：用上一块保存的归一化尾值做 x[-1]
        if (pImpl_->has_last_raw_sample) {
            pcm[0] = pcm[0] - alpha * pImpl_->last_raw_sample;
        }

        // 5.4 保存本块的归一化尾值，供下一块使用
        pImpl_->last_raw_sample = last_normalized;
        pImpl_->has_last_raw_sample = true;

        info.pre_emphasized = true;
    }

    // 6. 统计
    if (opt.collect_stats) {
        info.processed_samples = static_cast<int64_t>(pcm.size());
    }

    result.success = true;
    result.status = AudioPreprocessStatus::kOk;
    result.pcm = std::move(pcm);
    return result;
}

std::vector<SpeechSegment> AudioPreprocessor::DetectSpeech(const std::vector<float>& pcm) const {
    std::vector<SpeechSegment> segments;
    if (pcm.empty()) {
        return segments;
    }

    const auto& opt = pImpl_->options;
    int frame_len = opt.sample_rate * opt.vad_frame_ms / 1000;
    if (frame_len <= 0) {
        return segments;
    }

    int total_frames = static_cast<int>(pcm.size()) / frame_len;
    if (total_frames <= 0) {
        return segments;
    }

    // 1. 逐帧算 RMS
    std::vector<float> frame_rms(total_frames);
    double avg_rms = 0.0;
    for (int i = 0; i < total_frames; i++) {
        double sum_sq = 0.0;
        int start = i * frame_len;
        for (int j = 0; j < frame_len; j++) {
            float v = pcm[start + j];
            sum_sq += static_cast<double>(v) * v;
        }
        frame_rms[i] = static_cast<float>(std::sqrt(sum_sq / frame_len));
        avg_rms += frame_rms[i];
    }
    avg_rms /= total_frames;

    // 2. 全局阈值
    float threshold = std::max(static_cast<float>(avg_rms) * opt.vad_threshold_scale,
                               opt.vad_min_threshold);

    // 3. 逐帧判有/无语音
    std::vector<bool> is_speech(total_frames, false);
    for (int i = 0; i < total_frames; i++) {
        if (frame_rms[i] > threshold) {
            is_speech[i] = true;
        }
    }

    // 4. Hangover：语音结束后多等几帧，短停顿不切断
    int hangover = 0;
    for (int i = 0; i < total_frames; i++) {
        if (is_speech[i]) {
            hangover = opt.vad_hangover_frames;
        } else if (hangover > 0) {
            is_speech[i] = true;
            hangover--;
        }
    }

    // 5. 聚合为连续段落
    bool in_speech = false;
    SpeechSegment seg;
    for (int i = 0; i < total_frames; i++) {
        if (is_speech[i] && !in_speech) {
            // 语音段开始
            in_speech = true;
            seg.start_frame = i;
            seg.start_sample = static_cast<int64_t>(i) * frame_len;
            seg.start_ms = static_cast<double>(seg.start_sample) / opt.sample_rate * 1000.0;
        } else if (!is_speech[i] && in_speech) {
            // 语音段结束
            in_speech = false;
            seg.end_frame_exclusive = i;
            seg.end_sample_exclusive = static_cast<int64_t>(i) * frame_len;
            seg.end_ms = static_cast<double>(seg.end_sample_exclusive) / opt.sample_rate * 1000.0;
            segments.push_back(seg);
        }
    }
    // 尾部仍在语音中
    if (in_speech) {
        seg.end_frame_exclusive = total_frames;
        seg.end_sample_exclusive = static_cast<int64_t>(total_frames) * frame_len;
        seg.end_ms = static_cast<double>(seg.end_sample_exclusive) / opt.sample_rate * 1000.0;
        segments.push_back(seg);
    }

    return segments;
}

AudioPreprocessStats AudioPreprocessor::ComputeStats(const std::vector<float>& pcm) const {
    AudioPreprocessStats stats;

    // 空输入返回全零
    if (pcm.empty()) {
        return stats;
    }

    // 单次遍历 PCM，同时统计 min/max/RMS/零样本/NaN/Inf
    float min_val = pcm[0], max_val = pcm[0];
    double sum_abs = 0.0;   // 绝对值累加（用于 mean_abs）
    double sum_sq  = 0.0;   // 平方累加（用于 RMS），double 防 float 累积误差
    int64_t zeros = 0;
    bool has_nan = false, has_inf = false;

    for (auto v : pcm) {
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
        if (v == 0.0f) {
            zeros++;
        }
        if (std::isnan(v)) {
            has_nan = true;
        }
        if (std::isinf(v)) {
            has_inf = true;
        }
        sum_abs += std::abs(v);
        sum_sq  += static_cast<double>(v) * v;
    }

    // 填统计字段
    size_t n = pcm.size();
    stats.num_samples = static_cast<int64_t>(n);            // 样本总数
    stats.min_value   = min_val;                             // 最小值
    stats.max_value   = max_val;                             // 最大值
    stats.max_abs     = std::max(std::abs(min_val), std::abs(max_val)); // 最大绝对值
    stats.mean_abs    = static_cast<float>(sum_abs / n);     // 平均绝对值
    stats.rms         = static_cast<float>(std::sqrt(sum_sq / n)); // 均方根
    stats.zero_ratio  = static_cast<double>(zeros) / n;      // 零样本占比
    stats.has_nan     = has_nan;                              // 是否含 NaN
    stats.has_inf     = has_inf;                              // 是否含 Inf

    return stats;
}

void AudioPreprocessor::ResetStreamingState() {
    pImpl_->last_raw_sample = 0.0f;
    pImpl_->has_last_raw_sample = false;
}

}   // namespace audio
}   // namespace digital_human
