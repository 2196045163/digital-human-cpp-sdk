#include <cmath>      // std::round, std::cos, std::sqrt, std::log10, M_PI
#include <memory>

#include "audio/audio_mel_feature_extract.h"

namespace digital_human {
namespace audio {

namespace {
    /// @brief 构造失败结果，避免错误路径中重复填充 success/status/error_message
    static MelFeatureResult MakeMelErrorResult(MelFeatureStatus status) {
        MelFeatureResult res;
        res.success = false;
        res.status = status;
        res.error_message = MelFeatureExtractor::StatusToString(status);
        return res;
    }

    /// @brief 校验 MelFeatureOptions 合法性
    /// @return kOk 通过；其他值对应首个失败点
    static MelFeatureStatus ValidateOptions(const MelFeatureOptions& opts) {
        if (opts.sample_rate <= 0) {
            return MelFeatureStatus::kInvalidSampleRate;
        }
        if (opts.n_fft <= 0) {
            return MelFeatureStatus::kInvalidFftSize;
        }
        if (opts.n_mels <= 0) {
            return MelFeatureStatus::kInvalidMelCount;
        }
        if (opts.fmin < 0.0f || opts.fmax <= opts.fmin) {
            return MelFeatureStatus::kInvalidFrequencyRange;
        }
        if (opts.fmax > opts.sample_rate / 2.0f) {
            return MelFeatureStatus::kInvalidFrequencyRange;
        }
        if (opts.max_abs_value <= 0.0f) {
            return MelFeatureStatus::kInvalidNormalizeRange;
        }

        return MelFeatureStatus::kOk;
    }

    /// @brief 填 MelFeatureInfo（Extract 和 ExtractBatch 共用）
    static MelFeatureInfo MakeMelInfo(const MelFeatureOptions& opt, int n_fft_bins,
                                       int num_frames, const cv::Mat& mel_spec) {
        MelFeatureInfo info;
        info.sample_rate = opt.sample_rate;
        info.n_fft = opt.n_fft;
        info.n_fft_bins = n_fft_bins;
        info.n_mels = opt.n_mels;
        info.num_frames = num_frames;
        info.rows = num_frames;
        info.cols = opt.n_mels;
        info.fmin = opt.fmin;
        info.fmax = opt.fmax;
        info.frequency_resolution_hz = static_cast<float>(opt.sample_rate) / opt.n_fft;
        info.normalize_mode = opt.normalize_mode;
        info.spectrum_mode = opt.spectrum_mode;
        info.mel_scale = opt.mel_scale;
        info.filter_normalization = opt.filter_normalization;
        double v_min, v_max;
        cv::minMaxLoc(mel_spec, &v_min, &v_max);
        info.min_value = static_cast<float>(v_min);
        info.max_value = static_cast<float>(v_max);
        info.has_nan_or_inf = false;
        return info;
    }

    static float HzToSlaneyMel(float hz) {
        constexpr float kFMin = 0.0f;
        constexpr float kFSp = 200.0f / 3.0f;
        constexpr float kMinLogHz = 1000.0f;
        constexpr float kMinLogMel = (kMinLogHz - kFMin) / kFSp;
        const float log_step = std::log(6.4f) / 27.0f;

        if (hz < kMinLogHz) {
            return (hz - kFMin) / kFSp;
        }
        return kMinLogMel + std::log(hz / kMinLogHz) / log_step;
    }

    static float SlaneyMelToHz(float mel) {
        constexpr float kFMin = 0.0f;
        constexpr float kFSp = 200.0f / 3.0f;
        constexpr float kMinLogHz = 1000.0f;
        constexpr float kMinLogMel = (kMinLogHz - kFMin) / kFSp;
        const float log_step = std::log(6.4f) / 27.0f;

        if (mel < kMinLogMel) {
            return kFMin + kFSp * mel;
        }
        return kMinLogHz * std::exp(log_step * (mel - kMinLogMel));
    }

    static cv::Mat BuildMelFilterBank(const MelFeatureOptions& opts, int n_fft_bins) {
        cv::Mat mel_basis = cv::Mat::zeros(opts.n_mels, n_fft_bins, CV_32F);

        const auto hz_to_mel = [&opts](float hz) {
            return opts.mel_scale == MelScale::kSlaney
                ? HzToSlaneyMel(hz)
                : MelFeatureExtractor::HzToMel(hz);
        };
        const auto mel_to_hz = [&opts](float mel) {
            return opts.mel_scale == MelScale::kSlaney
                ? SlaneyMelToHz(mel)
                : MelFeatureExtractor::MelToHz(mel);
        };

        const float mel_min = hz_to_mel(opts.fmin);
        const float mel_max = hz_to_mel(opts.fmax);
        const float mel_step = (mel_max - mel_min) / (opts.n_mels + 1);

        std::vector<float> mel_frequencies(opts.n_mels + 2);
        for (int index = 0; index < opts.n_mels + 2; ++index) {
            mel_frequencies[index] = mel_to_hz(mel_min + index * mel_step);
        }

        const float fft_frequency_step =
            static_cast<float>(opts.sample_rate) / opts.n_fft;
        for (int mel_index = 0; mel_index < opts.n_mels; ++mel_index) {
            const float left = mel_frequencies[mel_index];
            const float center = mel_frequencies[mel_index + 1];
            const float right = mel_frequencies[mel_index + 2];
            const float lower_width = center - left;
            const float upper_width = right - center;

            for (int fft_index = 0; fft_index < n_fft_bins; ++fft_index) {
                const float frequency = fft_index * fft_frequency_step;
                const float lower_slope = (frequency - left) / lower_width;
                const float upper_slope = (right - frequency) / upper_width;
                mel_basis.at<float>(mel_index, fft_index) =
                    std::max(0.0f, std::min(lower_slope, upper_slope));
            }

            if (opts.filter_normalization ==
                MelFilterNormalization::kSlaney) {
                const float normalization = 2.0f / (right - left);
                mel_basis.row(mel_index) *= normalization;
            }
        }
        return mel_basis;
    }

} // namespace

// ====== Impl ======
struct MelFeatureExtractor::Impl {
    MelFeatureOptions options;      // 构造时保存的配置
    int n_fft_bins = 0;             // n_fft / 2 + 1，如 800 → 401
    cv::Mat mel_basis;              // 预构建的 Mel 滤波器组 [n_mels, n_fft_bins]
    bool mel_basis_ok = false;      // mel_basis 是否构建成功
    std::vector<std::vector<float>> mel_buffer;  // 流式 Mel 攒帧缓冲
};

// ====== 构造/析构/移动 ======
MelFeatureExtractor::MelFeatureExtractor(const MelFeatureOptions& options)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->options = options;
    pImpl_->n_fft_bins = options.n_fft / 2 + 1;

    // 校验 options 合法性，通过后再构建 mel_basis
    MelFeatureStatus status = ValidateOptions(options);
    if (status == MelFeatureStatus::kOk) {
        pImpl_->mel_basis = BuildMelFilterBank(options, pImpl_->n_fft_bins);
        pImpl_->mel_basis_ok = !pImpl_->mel_basis.empty();
    }
    // 非法参数不崩溃，mel_basis_ok 保持 false，Extract 时返回错误
}

MelFeatureExtractor::~MelFeatureExtractor() = default;
MelFeatureExtractor::MelFeatureExtractor(MelFeatureExtractor&&) noexcept = default;
MelFeatureExtractor& MelFeatureExtractor::operator=(MelFeatureExtractor&&) noexcept = default;

// ====== 静态工具 ======

float MelFeatureExtractor::HzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelFeatureExtractor::MelToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::string MelFeatureExtractor::StatusToString(MelFeatureStatus status) {
    switch (status) {
        case MelFeatureStatus::kOk:                     return "成功";
        case MelFeatureStatus::kEmptyFrame:              return "输入帧为空";
        case MelFeatureStatus::kEmptyBatch:              return "批量输入为空";
        case MelFeatureStatus::kInvalidSampleRate:       return "无效的采样率";
        case MelFeatureStatus::kInvalidFftSize:          return "无效的 FFT 点数";
        case MelFeatureStatus::kInvalidMelCount:         return "无效的 Mel 维数";
        case MelFeatureStatus::kInvalidFrequencyRange:   return "无效的频率范围";
        case MelFeatureStatus::kInvalidNormalizeRange:   return "无效的归一化参数";
        case MelFeatureStatus::kInvalidFrameData:        return "帧数据包含 NaN 或 Inf";
        case MelFeatureStatus::kMelBasisInitFailed:      return "Mel 滤波器组构建失败";
        case MelFeatureStatus::kDftFailed:               return "DFT 执行失败";
        case MelFeatureStatus::kNoFeaturesGenerated:     return "未生成任何特征";
        case MelFeatureStatus::kInvalidChunkSize:        return "无效的 chunk 大小";
        case MelFeatureStatus::kInvalidChunkLayout:      return "无效的 chunk 展平方式";
        case MelFeatureStatus::kUnknownError:            return "未知错误";
        default:                                          return "未定义错误";
    }
}

// ====== 工厂函数 ======

MelFeatureOptions MelFeatureExtractor::Wav2LipDefault() {
    MelFeatureOptions opts;
    // 默认值即为 Wav2Lip 配置（800/80/55/7600/kWav2LipSymmetric），无需修改
    return opts;
}

MelFeatureOptions MelFeatureExtractor::SpeechDefault() {
    MelFeatureOptions opts;
    opts.n_fft = 400;
    opts.n_mels = 40;
    opts.fmin = 80.0f;
    opts.fmax = 8000.0f;
    opts.normalize_mode = MelNormalizeMode::kDb;
    opts.mel_scale = MelScale::kHtk;
    opts.filter_normalization = MelFilterNormalization::kNone;
    return opts;
}

// ====== 属性查询 ======

const MelFeatureOptions& MelFeatureExtractor::GetOptions() const {
    return pImpl_->options;
}

int MelFeatureExtractor::GetSampleRate() const {
    return pImpl_->options.sample_rate;
}

int MelFeatureExtractor::GetFftSize() const {
    return pImpl_->options.n_fft;
}

int MelFeatureExtractor::GetMelBins() const {
    return pImpl_->options.n_mels;
}

int MelFeatureExtractor::GetFftBins() const {
    return pImpl_->n_fft_bins;
}

float MelFeatureExtractor::GetFrequencyResolutionHz() const {
    if (pImpl_->options.n_fft <= 0) {
        return 0.0f;
    }
    return static_cast<float>(pImpl_->options.sample_rate) / pImpl_->options.n_fft;
}

cv::Mat MelFeatureExtractor::GetMelBasis() const {
    return pImpl_->mel_basis;
}

// =================== 核心接口 ===================

MelFeatureResult MelFeatureExtractor::Extract(const std::vector<float>& frame) const {
    MelFeatureResult mf_res;

    // 1. 输入校验
    if (frame.empty()) {
        mf_res = MakeMelErrorResult(MelFeatureStatus::kEmptyFrame);
        return mf_res;
    }
    if (!pImpl_->mel_basis_ok) {
        mf_res = MakeMelErrorResult(MelFeatureStatus::kMelBasisInitFailed);
        return mf_res;
    }
    if (pImpl_->options.validate_finite) {
        for (auto v : frame) {
            if (std::isnan(v) || std::isinf(v)) {
                mf_res = MakeMelErrorResult(MelFeatureStatus::kInvalidFrameData);
                return mf_res;
            }
        }
    }


    // 2. 准备输入矩阵：把帧数据拷贝到 [1, n_fft] 的 cv::Mat 中，不足 n_fft 尾部自动补零
    cv::Mat input_frame(1, pImpl_->options.n_fft, CV_32F, cv::Scalar(0));
    int copy_len = std::min(static_cast<int>(frame.size()), pImpl_->options.n_fft);
    std::memcpy(input_frame.ptr<float>(0), frame.data(), copy_len * sizeof(float));


    // 3. FFT 变换，把时域波形转成频域幅值
    // 3.1 准备两通道矩阵：通道0=实部(输入帧)，通道1=虚部(全零)
    cv::Mat planes[] = {input_frame, cv::Mat::zeros(input_frame.size(), CV_32F)};
    cv::Mat complex_img;
    // 3.2 合并实部和虚部为一个双通道复数矩阵
    cv::merge(planes, 2, complex_img);
    // 3.3 执行傅里叶变换，输入和输出是同一个矩阵（原地变换）
    cv::dft(complex_img, complex_img);
    // 3.4 分离回两个通道：planes[0]=变换后实部，planes[1]=变换后虚部
    cv::split(complex_img, planes);
    // 3.5 计算幅度谱：sqrt(实部² + 虚部²) → 结果写回 planes[0]
    cv::magnitude(planes[0], planes[1], planes[0]);
    // 3.6 只取前半频谱（401个频点）：实数信号FFT左右共轭对称，后半是重复的
    cv::Mat mag_spec = planes[0].colRange(0, pImpl_->n_fft_bins);  // shape = [1, 401]

    // 4. Mel 滤波：幅度谱 [1, 401] × mel_basis^T [80, 401] → [1, 80]
    cv::Mat mel_spec; // 滤波之后的
    cv::gemm(mag_spec, pImpl_->mel_basis, 1.0, cv::Mat(), 0.0, mel_spec, cv::GEMM_2_T);

    // 5. 归一化：Wav2Lip 对称归一化 [-4, 4]
    const auto& opt = pImpl_->options;
    for (int i = 0; i < opt.n_mels; i++) {
        float val = mel_spec.at<float>(0, i);
        // 5.1 幅度转 dB，clip 下界防止 log(0)
        val = 20.0f * std::log10(std::max(val, opt.amin));
        // 5.2 减去参考电平
        val -= opt.ref_level_db;
        // 5.3 clip 到最小电平
        val = std::max(val, opt.min_level_db);
        // 5.4 线性缩放到 [0, 1]——[-100, 0] 的 dB 范围线性映射到 [0, 1]
        float norm = (val - opt.min_level_db) / (-opt.min_level_db);
        // 5.5 映射到 [-max_abs_value, max_abs_value] 并 clip
        float sym = 2.0f * opt.max_abs_value * norm - opt.max_abs_value;
        sym = std::max(-opt.max_abs_value, std::min(opt.max_abs_value, sym));
        mel_spec.at<float>(0, i) = sym;
    }

    // 6. 填充结果并返回
    mf_res.mel = mel_spec;
    mf_res.info = MakeMelInfo(opt, pImpl_->n_fft_bins, 1, mel_spec);
    mf_res.success = true;
    mf_res.status = MelFeatureStatus::kOk;
    return mf_res;
}

MelFeatureResult MelFeatureExtractor::ExtractBatch(
    const std::vector<std::vector<float>>& frames) const {
    MelFeatureResult mf_res;

    // 1. 输入校验
    if (frames.empty()) {
        mf_res = MakeMelErrorResult(MelFeatureStatus::kEmptyBatch);
        return mf_res;
    }
    if (!pImpl_->mel_basis_ok) {
        mf_res = MakeMelErrorResult(MelFeatureStatus::kMelBasisInitFailed);
        return mf_res;
    }
    if (frames[0].empty()) {
        mf_res = MakeMelErrorResult(MelFeatureStatus::kEmptyFrame);
        return mf_res;
    }

    // 2. 预分配批量结果矩阵：每行一帧，每列一个 Mel 频段，shape = [帧数, 80]
    int n_mels = pImpl_->options.n_mels;
    cv::Mat batch_mel(static_cast<int>(frames.size()), n_mels, CV_32F);

    // 3. 逐帧调 Extract，取出每帧 [1, 80] 的结果，拷贝到批量矩阵的第 i 行
    for (size_t i = 0; i < frames.size(); i++) {
        auto single = Extract(frames[i]);
        if (!single.success) {
            mf_res = MakeMelErrorResult(single.status);
            return mf_res;
        }
        // 把 single.mel [1, 80] 的一行数据复制到 batch_mel 的第 i 行
        single.mel.copyTo(batch_mel.row(static_cast<int>(i)));
    }

    // 4. 填充结果并返回
    mf_res.mel = batch_mel;
    mf_res.info = MakeMelInfo(pImpl_->options, pImpl_->n_fft_bins,
                               static_cast<int>(frames.size()), batch_mel);
    mf_res.success = true;
    mf_res.status = MelFeatureStatus::kOk;
    return mf_res;
}

std::vector<float> MelFeatureExtractor::ExtractVector(const std::vector<float>& frame) const {
    auto res = Extract(frame);
    std::vector<float> vec;

    // 这是兼容性的便捷接口；需要失败状态时应直接调用 Extract().
    if (!res.success || res.mel.empty()) {
        return vec;
    }

    vec.assign(res.mel.begin<float>(), res.mel.end<float>());
    return vec;
}

MelChunkResult MelFeatureExtractor::BuildWav2LipChunks(
    const cv::Mat& mel_spectrogram, const MelChunkOptions& options) const {
    MelChunkResult r;

    // 1. 输入校验：mel 矩阵非空、列数必须等于 n_mels
    if (mel_spectrogram.empty()) {
        r.status = MelFeatureStatus::kEmptyBatch;
        r.error_message = StatusToString(r.status);
        return r;
    }
    if (mel_spectrogram.cols != pImpl_->options.n_mels) {
        r.status = MelFeatureStatus::kInvalidChunkSize;
        r.error_message = StatusToString(r.status);
        return r;
    }
    int total_frames = mel_spectrogram.rows;
    int chunk = options.chunk_size;
    int n_mels = pImpl_->options.n_mels;
    if (chunk <= 0 || total_frames <= 0) {
        r.status = MelFeatureStatus::kInvalidChunkSize;
        r.error_message = StatusToString(r.status);
        return r;
    }

    // 2. 滑动窗口切块：每次取 chunk 个连续帧，展平为一个 1280 float 数组
    r.chunks.clear();
    bool freq_major = (options.layout == MelChunkLayout::kFreqMajor80x16);

    // 滑动窗口：从第 0 帧开始，每次前进 hop 帧
    for (int start = 0; start < total_frames; start += options.hop) {

        // 剩余帧不够一整个 chunk：比如总共 299 帧，chunk=16，start=288 只能取到 288~298（11帧<16）
        if (start + chunk > total_frames) {
            if (options.pad_tail) {
                break;    // 开启尾块填充：直接结束，不生成这个残缺 chunk
            } else {
                continue; // 不填充：跳过这个残缺的起点，继续尝试下一个 hop
            }
        }

        // 当前 chunk 的展平结果：chunk=16, n_mels=80 → 1280 个 float
        std::vector<float> chunk_data;
        chunk_data.reserve(chunk * n_mels);

        if (freq_major) {
            // freq-major = 频率优先：先把 mel[第0帧][所有频段] 取完，再取 mel[第1帧][所有频段]，...
            // 内层循环是帧（row），外层循环是频段（col），取出的顺序是：
            //   f0_t0, f0_t1, ..., f0_t15,  ← 频段 0 的 16 个时间点
            //   f1_t0, f1_t1, ..., f1_t15,  ← 频段 1 的 16 个时间点
            //   ...,
            //   f79_t0, f79_t1, ..., f79_t15 ← 频段 79 的 16 个时间点
            for (int col = 0; col < n_mels; col++) {
                for (int row = start; row < start + chunk; row++) {
                    chunk_data.push_back(mel_spectrogram.at<float>(row, col));
                }
            }
        } else {
            // time-major = 时间优先：先把 mel[第0帧][所有频段] 取完，再取 mel[第1帧][所有频段]，...
            // 内层循环是频段（col），外层循环是帧（row），取出的顺序是：
            //   f0_t0, f1_t0, ..., f79_t0,  ← 帧 0 的 80 个频段
            //   f0_t1, f1_t1, ..., f79_t1,  ← 帧 1 的 80 个频段
            //   ...,
            //   f0_t15, f1_t15, ..., f79_t15 ← 帧 15 的 80 个频段
            for (int row = start; row < start + chunk; row++) {
                for (int col = 0; col < n_mels; col++) {
                    chunk_data.push_back(mel_spectrogram.at<float>(row, col));
                }
            }
        }

        r.chunks.push_back(std::move(chunk_data));
    }

    r.success = true;
    r.status = MelFeatureStatus::kOk;
    r.chunk_size = chunk;
    r.n_mels = n_mels;
    r.layout = options.layout;
    return r;
}

// ====== 流式接口 ======

void MelFeatureExtractor::PushMelFrame(const std::vector<float>& mel_frame) {
    pImpl_->mel_buffer.push_back(mel_frame);
}

cv::Mat MelFeatureExtractor::FlushMelFrames() {
    int rows = static_cast<int>(pImpl_->mel_buffer.size());
    int cols = pImpl_->options.n_mels;
    cv::Mat mel_mat(rows, cols, CV_32F);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            mel_mat.at<float>(r, c) = pImpl_->mel_buffer[r][c];
        }
    }
    pImpl_->mel_buffer.clear();
    return mel_mat;
}

std::vector<float> MelFeatureExtractor::TryPopMelChunk(const MelChunkOptions& options) {
    int chunk_size = options.chunk_size;
    if (static_cast<int>(pImpl_->mel_buffer.size()) < chunk_size) {
        return {};  // 不够一个 chunk，返回空
    }

    // 取前 chunk_size 帧
    std::vector<std::vector<float>> chunk_frames(
        pImpl_->mel_buffer.begin(),
        pImpl_->mel_buffer.begin() + chunk_size);

    // 转为 cv::Mat [chunk_size, n_mels]
    int n_mels = pImpl_->options.n_mels;
    cv::Mat chunk_mat(chunk_size, n_mels, CV_32F);
    for (int r = 0; r < chunk_size; r++) {
        for (int c = 0; c < n_mels; c++) {
            chunk_mat.at<float>(r, c) = chunk_frames[r][c];
        }
    }

    // 从 buffer 中移除已取帧（根据 hop 滑动）
    int hop = options.hop > 0 ? options.hop : 1;
    pImpl_->mel_buffer.erase(
        pImpl_->mel_buffer.begin(),
        pImpl_->mel_buffer.begin() + std::min(hop, static_cast<int>(pImpl_->mel_buffer.size())));

    // freq-major 展平
    std::vector<float> result;
    result.reserve(chunk_size * n_mels);
    for (int c = 0; c < n_mels; c++) {
        for (int r = 0; r < chunk_size; r++) {
            result.push_back(chunk_mat.at<float>(r, c));
        }
    }
    return result;
}

}   // namespace audio
}   // namespace digital_human
