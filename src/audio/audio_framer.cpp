#include <cmath>    // std::round, std::cos, M_PI
#include <memory>
#include <vector>
#include <chrono>

#include "audio/audio_framer.h"

namespace digital_human {
namespace audio {

namespace {
    static AudioFrameResult MakeFrameErrorResult(AudioFrameStatus status) {
        AudioFrameResult res;
        res.success = false;
        res.status = status;
        res.error_message = AudioFramer::StatusToString(status);
        
        return res;
    }

    // 填 AudioFrameInfo：framer 的 getter 提供静态参数，剩余 3 个是 Frame 循环中计算出的动态值
    static AudioFrameInfo MakeFrameInfo(const AudioFramer& framer, const std::vector<float>& pcm,
            int64_t padded_sample_count, int64_t pad_sample_count, int num_frames) {
        
        AudioFrameInfo info;
        const auto& opts = framer.GetOptions();

        info.sample_rate          = framer.GetSampleRate();
        info.frame_size           = framer.GetFrameSize();
        info.hop_size             = framer.GetHopSize();
        info.overlap_size         = framer.GetOverlapSize();
        info.original_sample_count = static_cast<int64_t>(pcm.size());
        info.padded_sample_count  = padded_sample_count;
        info.pad_sample_count     = pad_sample_count;
        info.num_frames           = num_frames;
        info.frame_duration_ms    = opts.frame_duration_ms;
        info.hop_duration_ms      = opts.hop_duration_ms;
        info.original_duration_sec = static_cast<double>(pcm.size()) / framer.GetSampleRate();
        info.window_type          = opts.window_type;
        info.tail_policy          = opts.tail_policy;

        return info;
    }

} // 匿名空间

    // ====== Impl ======
    struct AudioFramer::Impl {
        AudioFrameOptions options;
        int64_t frame_size;             // 一个数据帧的采样点数
        int64_t hop_size;               // 数据帧之间间隔的采样点数
        int64_t overlap_size;           // 数据帧之间重叠的采样点数
        std::vector<float> window;      // 预计算的窗函数，避免每帧重复算
        std::vector<float> pcm_buffer;  // 流式用的 PCM 缓冲：攒够 frame_size 才切一帧
        int64_t stream_frame_index = 0; // 流式帧序号，跨 ProcessFrame 调用递增
    };

    // ====== 构造/析构/移动 ======
    AudioFramer::AudioFramer(const AudioFrameOptions& options) : pImpl_(std::make_unique<Impl>()) { 
        pImpl_->options = options;

        // 计算 frame_size
        pImpl_->frame_size = ComputeFrameSize(options.sample_rate, options.frame_duration_ms);
        // 计算 hop_size
        pImpl_->hop_size = ComputeHopSize(options.sample_rate, options.hop_duration_ms);
        // 预生成窗函数
        pImpl_->window = GenerateWindow(pImpl_->frame_size, options.window_type);
        // 算 overlap_size
        pImpl_->overlap_size = pImpl_->frame_size - pImpl_->hop_size;
    }
    AudioFramer::~AudioFramer() = default;
    AudioFramer::AudioFramer(AudioFramer&&) noexcept = default;
    AudioFramer& AudioFramer::operator=(AudioFramer&&) noexcept = default;

    // ======================== 核心接口 ========================

    /// @brief 对单帧应用窗函数（原地修改）：frame[i] *= window[i]
    void AudioFramer::ApplyWindow(std::vector<float>& frame) const {
        const std::vector<float>& window = pImpl_->window;

        // 如果外部传进来一个长度不对的 frame，frame.size() > window.size() 会越界访问
        // 尾部帧 由于策略的不同可能长度有所不同，这里做取最小做截断
        size_t n = std::min(frame.size(), window.size());

        for (size_t i = 0; i < n; i++) {
            frame[i] = frame[i] * window[i];
        }
    }

    /// @brief 从 PCM 中切取一帧（不含加窗），供调试和测试
    std::vector<float> AudioFramer::CreateFrame(const std::vector<float>& pcm,
        size_t start_index, bool pad_tail) const {

        std::vector<float> res{};
        if (pImpl_->frame_size <= 0) {
            return res;
        }
        // 1.预分配空间，避免频繁扩容
        res.reserve(pImpl_->frame_size);

        // 2.循环切取采样点，凑成一帧
        for (int i = 0; i < pImpl_->frame_size; i++) {
            // 2.1 算出当前采样点在 PCM 中的位置
            size_t src_idx = start_index + i;
            if (src_idx < pcm.size()) {
                // 2.2 正常复制
                res.push_back(pcm[src_idx]);
            } else {
                // 2.3 超出 PCM 末尾
                if (pad_tail) {
                    // 超出部分补零
                    res.push_back(0.0f);
                } else {
                    // 不补零，结束循环，返回已收集的部分帧
                    break;
                }
            }
        }

        return res;
    }

    /// @brief 主分帧接口：完整流程（校验→计算→补零→切片→加窗→返回元数据）
    AudioFrameResult AudioFramer::Frame(const std::vector<float>& pcm) const {
        AudioFrameResult af_res;

        // 1.输入校验
        if (pcm.empty()) { 
            af_res = MakeFrameErrorResult(AudioFrameStatus::kEmptyInput);
            return af_res;
        }
        if (pImpl_->frame_size <= 0) {
            af_res = MakeFrameErrorResult(AudioFrameStatus::kInvalidFrameDuration);
            return af_res;
        }
        if (pImpl_->hop_size <= 0) {
            af_res = MakeFrameErrorResult(AudioFrameStatus::kInvalidHopDuration);
            return af_res;
        }
        // 验证pcm数据中是否有 Nan/Inf 数据
        if (pImpl_->options.validate_finite) {
            bool has_bad = false;
            for (auto v : pcm) {
                if (std::isnan(v) || std::isinf(v)) {
                    has_bad = true;
                    break;  // 找到一个就可以停了
                }
            }
            if (has_bad) {
                af_res = MakeFrameErrorResult(AudioFrameStatus::kInvalidPcmData);
                return af_res;
            }
        }

        // 2.计算帧数 + 计算尾帧补零个数
        int frame_cnt = ComputeNumFrames(static_cast<int64_t>(pcm.size()), pImpl_->frame_size, 
                                         pImpl_->hop_size, pImpl_->options.tail_policy);
        if (frame_cnt <= 0) {
            af_res = MakeFrameErrorResult(AudioFrameStatus::kNoFramesGenerated);
            return af_res;
        }
        // 2.1 计算尾帧需要补多少个零
        int64_t needed_len = (frame_cnt - 1) * pImpl_->hop_size + pImpl_->frame_size;
        int64_t pad_cnt = std::max(int64_t(0), needed_len - static_cast<int64_t>(pcm.size()));

        // 2.2 构造补零后的 PCM
        std::vector<float> padded_pcm = pcm;
        padded_pcm.resize(padded_pcm.size() + static_cast<size_t>(pad_cnt), 0.0f);

        // 3.循环分帧
        for (int i = 0; i < frame_cnt; i++) {
            // 原始 PCM 中的起始和结束的采样点索引
            int64_t start = i * pImpl_->hop_size;
            int64_t end = start + pImpl_->frame_size;

            // 3.1 切一帧
            std::vector<float> frame_samples = CreateFrame(padded_pcm, start, true);

            // 3.2 给每一帧加窗
            // void AudioFramer::ApplyWindow(std::vector<float>& frame) 
            ApplyWindow(frame_samples);

            // 3.3 填 AudioFrame 元数据，即每一个数据帧的信息
            AudioFrame af;
            af.samples = std::move(frame_samples);
            af.index = i;
            af.start_sample = start;
            af.end_sample_exclusive = end;
            // 抽帧也是有时间尺度的
            af.start_ms = static_cast<double>(start) / pImpl_->options.sample_rate * 1000.0;
            af.end_ms = static_cast<double>(end) / pImpl_->options.sample_rate * 1000.0;
            af.contains_padding = (end > static_cast<int64_t>(pcm.size()));

            af_res.frames.push_back(std::move(af));
        }

        // 4. 填 AudioFrameInfo
        af_res.info = MakeFrameInfo(*this, pcm, static_cast<int64_t>(padded_pcm.size()),
                                    pad_cnt, frame_cnt);

        // 5.填窗函数
        if (pImpl_->options.return_window) {
            af_res.window = pImpl_->window;
        }

        // 6. 返回
        af_res.success = true;
        af_res.status = AudioFrameStatus::kOk;
        return af_res;
    }

    /// @brief 便捷接口：只返回二维 float 数组，不含元数据（供 Mel 模块直接使用）
    std::vector<std::vector<float>> AudioFramer::FrameSamplesOnly(
        const std::vector<float>& pcm) const {

        std::vector<std::vector<float>> res{};

        auto frame_result = Frame(pcm);
        if (!frame_result.success) {
            return res;
        }

        res.reserve(frame_result.frames.size());
        for (auto& f : frame_result.frames) {
            res.push_back(std::move(f.samples));
        }
        return res;
    }

    // =========================================================

    /// @brief 通用语音 DSP 推荐配置：25ms 帧长 / 10ms 帧移 / Hamming 窗
    AudioFrameOptions AudioFramer::SpeechDefault() {
        AudioFrameOptions opts;
        // 因为默认值和 SpeechDefault 要求的值完全相同，所以…什么都不用改，直接 return
        return opts;
    }

    /// @brief Wav2Lip 模型推荐配置：50ms 帧长 / 12.5ms 帧移（与 n_fft=800 对齐）
    AudioFrameOptions AudioFramer::Wav2LipDefault() {
        AudioFrameOptions opts;

        opts.frame_duration_ms = 50.0;
        opts.hop_duration_ms = 12.5;

        return opts;
    }

    /// @brief 从毫秒计算帧采样点数（四舍五入）
    int AudioFramer::ComputeFrameSize(int sample_rate, double frame_duration_ms) {
        int res = 0;
        double mid_res = 0.0;

        // frame_size = round(sample_rate × frame_duration_ms / 1000.0)
        mid_res = std::round(static_cast<double>(sample_rate) * frame_duration_ms / 1000.0);
        res = static_cast<int>(mid_res);

        return res;
    }

    /// @brief 从毫秒计算帧移采样点数（四舍五入）
    int AudioFramer::ComputeHopSize(int sample_rate, double hop_duration_ms) {
        int res = 0;
        double mid_res = 0.0;

        // hop_size   = round(sample_rate × hop_duration_ms / 1000.0)
        mid_res = std::round(static_cast<double>(sample_rate) * hop_duration_ms / 1000.0);
        res = static_cast<int>(mid_res);

        return res;
    }

    /// @brief 预生成窗函数数组（只在构造时计算一次，避免每帧重复计算）
    std::vector<float> AudioFramer::GenerateWindow(int frame_size,
        AudioWindowType window_type) {

        std::vector<float> window{};
        // 1.处理边界情况
        if (frame_size <= 0) {
            return window;
        }

        // 2.创建一个大小等于 frame_size 的 std::vector<float> window(frame_size)
        window.resize(frame_size);

        // 3.根据所选窗类型确定window的参数
        if (window_type == AudioWindowType::kNone) {
            std::fill(window.begin(), window.end(), 1.0f);
        } else {
            for (int i = 0; i < frame_size; i++) {
                if (frame_size == 1) {
                    window[0] = 1.0f;   // 单采样点加窗无意义，直接原值通过
                } else {
                    // 3.1 将索引 i 映射到 [0, 2π] 范围，cos 在此区间画出完整余弦波
                    double angle = 2.0 * M_PI * i / (frame_size - 1);
                    // 3.2 cos(0)=1 → cos(π)=-1 → cos(2π)=1，Hamming/Hann 公式将其映射为 0~1 的窗权重
                    double cos_angle = std::cos(angle);

                    if (window_type == AudioWindowType::kHamming) {
                        window[i] = 0.54 - 0.46 * cos_angle;   // 两端最低 ≈ 0.08，中间最高 = 1.0
                    } else if (window_type == AudioWindowType::kHann) {
                        window[i] = 0.5 * (1 - cos_angle);      // 两端 = 0，中间最高 = 1.0
                    } else {    
                        // 位置类型的窗函数直接返回空的数组
                        window.clear();
                        return window;
                    }
                }
            }
        }

        return window;
    }

    /// @brief 根据尾部策略计算帧数
    int AudioFramer::ComputeNumFrames(int64_t sample_count, int frame_size, int hop_size,
        AudioTailPolicy tail_policy) {
        
        int res = 0;
        // 1.检查三个参数合法性
        if (sample_count <= 0 || frame_size <= 0 || hop_size <= 0) {
            return res;
        }

        // 2.把 sample_count 转成 double，方便 ceil / floor 运算
        double sample_cnt = static_cast<double>(sample_count);

        // 3.三种不同的尾部数据帧方案
        // L < N 时特殊处理：补零策略返回 1 帧（全补零），丢弃策略返回 0
        if (sample_cnt < static_cast<double>(frame_size)) {
            if (tail_policy == AudioTailPolicy::kDropIncomplete) {
                return 0;
            }
            return 1;  // kCoverLastSample / kStartEveryHop：补零凑一帧
        }

        switch (tail_policy) {
            case AudioTailPolicy::kCoverLastSample :
                res = static_cast<int>(1 + std::ceil((sample_cnt - frame_size) / hop_size));
                return res;
            case AudioTailPolicy::kStartEveryHop :
                res = static_cast<int>(std::ceil(sample_cnt / hop_size));
                return res;
            case AudioTailPolicy::kDropIncomplete :
                res = static_cast<int>(1 + std::floor((sample_cnt - frame_size) / hop_size));
                return res;
            default:
                return res;
        }
    }

    /// @brief 将状态码转为人类可读字符串（用于日志/example 打印）
    std::string AudioFramer::StatusToString(AudioFrameStatus status) {
        switch (status) {
            case AudioFrameStatus::kOk:                     return "成功";
            case AudioFrameStatus::kEmptyInput:             return "输入 PCM 为空";
            case AudioFrameStatus::kInvalidSampleRate:      return "无效的采样率";
            case AudioFrameStatus::kInvalidFrameDuration:   return "无效的帧长";
            case AudioFrameStatus::kInvalidHopDuration:     return "无效的帧移";
            case AudioFrameStatus::kInvalidFrameSize:       return "计算出的帧长为 0";
            case AudioFrameStatus::kInvalidHopSize:         return "计算出的帧移为 0";
            case AudioFrameStatus::kInvalidTailPolicy:      return "无效的尾部策略";
            case AudioFrameStatus::kInvalidWindowType:      return "无效的窗函数类型";
            case AudioFrameStatus::kInvalidPcmData:         return "PCM 包含 NaN 或 Inf";
            case AudioFrameStatus::kNoFramesGenerated:      return "未生成任何帧";
            case AudioFrameStatus::kUnknownError:           return "未知错误";
            default:                                        return "未定义错误";
        }
    }

    // ====== 属性查询 ======

    int AudioFramer::GetSampleRate() const {
        return pImpl_->options.sample_rate;
    }

    int AudioFramer::GetFrameSize() const {
        return pImpl_->frame_size;
    }

    int AudioFramer::GetHopSize() const {
        return pImpl_->hop_size;
    }

    int AudioFramer::GetOverlapSize() const {
        return pImpl_->overlap_size;
    }

    const AudioFrameOptions& AudioFramer::GetOptions() const {
        return pImpl_->options;
    }

    // ====== 流式接口 ======

    std::vector<AudioFrame> AudioFramer::ProcessFrame(const std::vector<float>& chunk) {
        std::vector<AudioFrame> frames;

        // 把新 chunk 追加到内部缓冲
        pImpl_->pcm_buffer.insert(pImpl_->pcm_buffer.end(), chunk.begin(), chunk.end());

        // 攒够一帧就切一帧：每次取 frame_size 个，向前移动 hop_size
        while (pImpl_->pcm_buffer.size() >= static_cast<size_t>(pImpl_->frame_size)) {
            // 从缓冲头部切一帧（CreateFrame 的第一个参数要求 padded_pcm，但这里只切头部帧，
            // 实际就是取前 frame_size 个样本）
            std::vector<float> frame_samples(
                pImpl_->pcm_buffer.begin(),
                pImpl_->pcm_buffer.begin() + pImpl_->frame_size);

            ApplyWindow(frame_samples);

            AudioFrame af;
            af.samples = std::move(frame_samples);
            af.index = static_cast<int>(pImpl_->stream_frame_index);
            af.start_sample = pImpl_->stream_frame_index * pImpl_->hop_size;
            pImpl_->stream_frame_index++;
            af.end_sample_exclusive = af.start_sample + pImpl_->frame_size;
            af.start_ms = static_cast<double>(af.start_sample) / pImpl_->options.sample_rate * 1000.0;
            af.end_ms = static_cast<double>(af.end_sample_exclusive) / pImpl_->options.sample_rate * 1000.0;
            af.contains_padding = false;

            frames.push_back(std::move(af));

            // 从缓冲头部移除 hop_size 个样本（保持帧之间的重叠关系）
            pImpl_->pcm_buffer.erase(
                pImpl_->pcm_buffer.begin(),
                pImpl_->pcm_buffer.begin() + std::min(
                    static_cast<size_t>(pImpl_->hop_size), pImpl_->pcm_buffer.size()));
        }
        return frames;
    }

    // FlushFrames：处理流式末尾残留的不足一帧的 PCM 数据。
    // 注意：流式 ProcessFrame 的帧数可能和整段 Frame() 差 ±1 帧。
    // 原因：整段 Frame() 根据总长度预先算帧数；流式按"攒够→切帧→删 hop_size"滑动。
    // 末尾缓冲残差和整段 padding 公式可能偏差 1 帧。该帧几乎全是补零，影响可忽略。
    std::vector<AudioFrame> AudioFramer::FlushFrames() {
        std::vector<AudioFrame> frames;

        if (!pImpl_->pcm_buffer.empty() && pImpl_->pcm_buffer.size() < static_cast<size_t>(pImpl_->frame_size)) {
            if (pImpl_->options.tail_policy == AudioTailPolicy::kDropIncomplete) {
                // 丢弃，不生成帧
            } else {
                std::vector<float> frame_samples = pImpl_->pcm_buffer;
                frame_samples.resize(pImpl_->frame_size, 0.0f);
                ApplyWindow(frame_samples);

                AudioFrame af;
                af.samples = std::move(frame_samples);
                af.index = static_cast<int>(pImpl_->stream_frame_index);
                af.start_sample = pImpl_->stream_frame_index * pImpl_->hop_size;
                af.end_sample_exclusive = af.start_sample + pImpl_->frame_size;
                af.start_ms = static_cast<double>(af.start_sample) / pImpl_->options.sample_rate * 1000.0;
                af.end_ms = static_cast<double>(af.end_sample_exclusive) / pImpl_->options.sample_rate * 1000.0;
                af.contains_padding = true;
                pImpl_->stream_frame_index++;

                frames.push_back(std::move(af));
            }
        }

        pImpl_->pcm_buffer.clear();
        return frames;
    }

    void AudioFramer::ResetStreaming() {
        pImpl_->pcm_buffer.clear();
        pImpl_->stream_frame_index = 0;
    }

} // namespace audio
} // namespace digital_human
