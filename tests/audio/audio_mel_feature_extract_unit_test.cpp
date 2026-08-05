/// @file    audio_mel_feature_extract_unit_test.cpp
/// @brief   MelFeatureExtractor 单元测试：状态码/工厂/Hz↔Mel/滤波矩阵/单帧/批量/chunk
/// @note    参考 tests/audio/audio_framer_unit_test.cpp 风格

#include <algorithm>   // std::min, std::abs
#include <cmath>       // std::sin, M_PI, std::isnan, std::isinf
#include <iostream>
#include <limits>      // std::numeric_limits
#include <string>
#include <vector>

#include "audio/audio_mel_feature_extract.h"

using namespace digital_human::audio;

// ============================================================================
// 让枚举可打印
// ============================================================================
inline std::ostream& operator<<(std::ostream& os, MelFeatureStatus s) {
    return os << MelFeatureExtractor::StatusToString(s);
}

inline std::ostream& operator<<(std::ostream& os, MelNormalizeMode m) {
    switch (m) {
        case MelNormalizeMode::kNone:              return os << "kNone";
        case MelNormalizeMode::kDb:                return os << "kDb";
        case MelNormalizeMode::kZeroOne:           return os << "kZeroOne";
        case MelNormalizeMode::kWav2LipSymmetric:  return os << "kWav2LipSymmetric";
        default:                                    return os << "?";
    }
}

// ============================================================================
// 简易断言宏
// ============================================================================
static int g_fail = 0;

#define EXPECT_TRUE(cond, msg) \
    do { if (!(cond)) { std::cerr << "  FAIL: " << msg << "\n"; g_fail++; } } while(0)

#define EXPECT_FALSE(cond, msg) EXPECT_TRUE(!(cond), msg)

#define EXPECT_EQ(a, b, msg) \
    do { if ((a) != (b)) { std::cerr << "  FAIL: " << msg \
          << " (expected=" << (b) << ", actual=" << (a) << ")\n"; g_fail++; } } while(0)

// ============================================================================
// 辅助函数：生成 n_fft 长的 440Hz 正弦波（用作音频帧输入）
// ============================================================================
static std::vector<float> MakeSineFrame(int n, float freq_hz = 440.0f,
                                          float sample_rate = 16000.0f) {
    std::vector<float> f(n);
    for (int i = 0; i < n; i++) {
        f[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq_hz * i / sample_rate);
    }
    return f;
}

// ============================================================================
int main() {
    std::cout << "=== MelFeatureExtractor Unit Tests ===\n\n";

    // -----------------------------------------------------------------------
    // 1. StatusToString — 15 个状态码全部非空
    // -----------------------------------------------------------------------
    std::cout << "[1] StatusToString (15 codes) ...\n";

    std::vector<MelFeatureStatus> all_status = {
        MelFeatureStatus::kOk, MelFeatureStatus::kEmptyFrame,
        MelFeatureStatus::kEmptyBatch, MelFeatureStatus::kInvalidSampleRate,
        MelFeatureStatus::kInvalidFftSize, MelFeatureStatus::kInvalidMelCount,
        MelFeatureStatus::kInvalidFrequencyRange, MelFeatureStatus::kInvalidNormalizeRange,
        MelFeatureStatus::kInvalidFrameData, MelFeatureStatus::kMelBasisInitFailed,
        MelFeatureStatus::kDftFailed, MelFeatureStatus::kNoFeaturesGenerated,
        MelFeatureStatus::kInvalidChunkSize, MelFeatureStatus::kInvalidChunkLayout,
        MelFeatureStatus::kUnknownError,
    };
    for (auto s : all_status) {
        EXPECT_FALSE(MelFeatureExtractor::StatusToString(s).empty(), "");
    }

    // -----------------------------------------------------------------------
    // 2. HzToMel / MelToHz — 公式验证
    // -----------------------------------------------------------------------
    std::cout << "[2] HzToMel / MelToHz ...\n";

    // Hz→Mel→Hz 往返误差 < 1Hz
    float hz = MelFeatureExtractor::MelToHz(MelFeatureExtractor::HzToMel(1000.0f));
    EXPECT_TRUE(std::abs(hz - 1000.0f) < 1.0f, "Hz→Mel→Hz roundtrip: 1000Hz");

    // 低频 Mel 差 < 高频 Mel 差（同样 ΔHz 下）
    float d_mel_low  = MelFeatureExtractor::HzToMel(200.0f) - MelFeatureExtractor::HzToMel(100.0f);
    float d_mel_high = MelFeatureExtractor::HzToMel(7600.0f) - MelFeatureExtractor::HzToMel(7500.0f);
    EXPECT_TRUE(d_mel_low > d_mel_high, "low-freq delta Mel > high-freq delta Mel");

    // -----------------------------------------------------------------------
    // 3. Wav2LipDefault / SpeechDefault — 工厂配置
    // -----------------------------------------------------------------------
    std::cout << "[3] Factory functions ...\n";

    auto wl = MelFeatureExtractor::Wav2LipDefault();
    EXPECT_EQ(wl.n_fft, 800, "Wav2Lip: n_fft=800");
    EXPECT_EQ(wl.n_mels, 80, "Wav2Lip: n_mels=80");
    EXPECT_TRUE(std::abs(wl.fmin - 55.0f) < 0.1f, "Wav2Lip: fmin=55");
    EXPECT_TRUE(std::abs(wl.fmax - 7600.0f) < 1.0f, "Wav2Lip: fmax=7600");
    EXPECT_EQ(wl.normalize_mode, MelNormalizeMode::kWav2LipSymmetric, "Wav2Lip: symmetric");

    auto sp = MelFeatureExtractor::SpeechDefault();
    EXPECT_EQ(sp.n_fft, 400, "Speech: n_fft=400");
    EXPECT_EQ(sp.n_mels, 40, "Speech: n_mels=40");
    EXPECT_EQ(sp.normalize_mode, MelNormalizeMode::kDb, "Speech: dB");

    // -----------------------------------------------------------------------
    // 4. 构造 MelFeatureExtractor — mel_basis 就绪
    // -----------------------------------------------------------------------
    std::cout << "[4] Constructor + mel_basis ...\n";

    MelFeatureExtractor extractor;  // Wav2Lip 默认

    EXPECT_EQ(extractor.GetSampleRate(), 16000, "sample rate");
    EXPECT_EQ(extractor.GetFftSize(), 800, "n_fft");
    EXPECT_EQ(extractor.GetMelBins(), 80, "n_mels");
    EXPECT_EQ(extractor.GetFftBins(), 401, "n_fft_bins = 800/2+1 = 401");
    EXPECT_TRUE(std::abs(extractor.GetFrequencyResolutionHz() - 20.0f) < 1.0f,
                "freq resolution = 20Hz");

    // mel_basis 形状 [80, 401]、非负、非空
    auto basis = extractor.GetMelBasis();
    EXPECT_EQ(basis.rows, 80, "mel_basis rows = 80");
    EXPECT_EQ(basis.cols, 401, "mel_basis cols = 401");
    EXPECT_FALSE(basis.empty(), "mel_basis not empty");
    double b_min, b_max;
    cv::minMaxLoc(basis, &b_min, &b_max);
    EXPECT_TRUE(b_min >= 0.0, "mel_basis non-negative");

    // -----------------------------------------------------------------------
    // 5. 非法参数构造 — mel_basis 不可用但不崩溃
    // -----------------------------------------------------------------------
    std::cout << "[5] Constructor: invalid options ...\n";

    MelFeatureOptions bad_opts;
    bad_opts.n_fft = 0;
    MelFeatureExtractor bad_extractor(bad_opts);
    EXPECT_EQ(bad_extractor.GetMelBasis().rows, 0, "invalid opts: mel_basis empty");
    auto r_bad = bad_extractor.Extract(MakeSineFrame(800));
    EXPECT_FALSE(r_bad.success, "invalid opts: Extract fails");
    EXPECT_EQ(r_bad.status, MelFeatureStatus::kMelBasisInitFailed, "status = kMelBasisInitFailed");

    // -----------------------------------------------------------------------
    // 6. Extract — 空帧 / NaN / 正常单帧
    // -----------------------------------------------------------------------
    std::cout << "[6] Extract: error paths ...\n";

    // 空帧
    auto r_empty = extractor.Extract({});
    EXPECT_FALSE(r_empty.success, "empty frame -> fail");
    EXPECT_EQ(r_empty.status, MelFeatureStatus::kEmptyFrame, "status = kEmptyFrame");

    // NaN 帧
    std::vector<float> nan_frame(800, 0.0f);
    nan_frame[100] = std::nanf("");
    auto r_nan = extractor.Extract(nan_frame);
    EXPECT_FALSE(r_nan.success, "NaN frame -> fail");
    EXPECT_EQ(r_nan.status, MelFeatureStatus::kInvalidFrameData, "status = kInvalidFrameData");

    // -----------------------------------------------------------------------
    // 7. Extract — 正常单帧：shape / 范围 / 无 NaN
    // -----------------------------------------------------------------------
    std::cout << "[7] Extract: normal single frame ...\n";

    auto frame_800 = MakeSineFrame(800);
    auto r = extractor.Extract(frame_800);
    EXPECT_TRUE(r.success, "single frame: success");
    EXPECT_EQ(r.status, MelFeatureStatus::kOk, "status = kOk");

    // mel 矩阵 shape = [1, 80]
    EXPECT_EQ(r.mel.rows, 1, "mel rows = 1");
    EXPECT_EQ(r.mel.cols, 80, "mel cols = 80");

    // info 正确
    EXPECT_EQ(r.info.num_frames, 1, "info.num_frames = 1");
    EXPECT_EQ(r.info.n_fft_bins, 401, "info.n_fft_bins = 401");
    EXPECT_FALSE(r.info.has_nan_or_inf, "no NaN/Inf");

    // 数值范围在 [-4, 4]（Wav2Lip 对称归一化）
    EXPECT_TRUE(r.info.min_value >= -4.1f, "min >= -4");
    EXPECT_TRUE(r.info.max_value <= 4.1f, "max <= 4");
    EXPECT_TRUE(r.info.max_value > r.info.min_value, "max > min");

    // -----------------------------------------------------------------------
    // 8. Extract — 短帧（< n_fft，尾部补零后也能正常跑）
    // -----------------------------------------------------------------------
    std::cout << "[8] Extract: short frame (< n_fft) ...\n";

    auto short_frame = MakeSineFrame(200);  // 只有 200 个采样点
    auto r_short = extractor.Extract(short_frame);
    EXPECT_TRUE(r_short.success, "short frame: success");
    EXPECT_EQ(r_short.mel.cols, 80, "short frame: 80 Mel bins");

    // -----------------------------------------------------------------------
    // 9. ExtractBatch — 空批量
    // -----------------------------------------------------------------------
    std::cout << "[9] ExtractBatch: empty ...\n";

    auto r_batch_empty = extractor.ExtractBatch({});
    EXPECT_FALSE(r_batch_empty.success, "empty batch -> fail");
    EXPECT_EQ(r_batch_empty.status, MelFeatureStatus::kEmptyBatch, "status = kEmptyBatch");

    // -----------------------------------------------------------------------
    // 10. ExtractBatch — 正常批量
    // -----------------------------------------------------------------------
    std::cout << "[10] ExtractBatch: normal batch ...\n";

    // 构造 5 帧
    std::vector<std::vector<float>> frames(5, MakeSineFrame(800));
    auto r_batch = extractor.ExtractBatch(frames);
    EXPECT_TRUE(r_batch.success, "batch: success");
    EXPECT_EQ(r_batch.mel.rows, 5, "batch mel rows = 5");
    EXPECT_EQ(r_batch.mel.cols, 80, "batch mel cols = 80");
    EXPECT_EQ(r_batch.info.num_frames, 5, "info.num_frames = 5");
    EXPECT_EQ(r_batch.info.rows, 5, "info.rows = 5");

    // -----------------------------------------------------------------------
    // 11. ExtractVector — 便捷接口
    // -----------------------------------------------------------------------
    std::cout << "[11] ExtractVector ...\n";

    auto vec = extractor.ExtractVector(frame_800);
    EXPECT_EQ(vec.size(), size_t(80), "ExtractVector size = 80");
    // 空帧 → 空 vector
    EXPECT_TRUE(extractor.ExtractVector({}).empty(), "empty -> empty vector");

    // -----------------------------------------------------------------------
    // 12. BuildWav2LipChunks — 空矩阵
    // -----------------------------------------------------------------------
    std::cout << "[12] BuildWav2LipChunks: empty ...\n";

    auto r_chunk_empty = extractor.BuildWav2LipChunks(cv::Mat());
    EXPECT_FALSE(r_chunk_empty.success, "empty mel -> fail");

    // -----------------------------------------------------------------------
    // 13. BuildWav2LipChunks — 正常切块（freq-major）
    // -----------------------------------------------------------------------
    std::cout << "[13] BuildWav2LipChunks: normal ...\n";

    auto r_chunk = extractor.BuildWav2LipChunks(r_batch.mel);
    EXPECT_TRUE(r_chunk.success, "chunk: success");
    // 5 帧，chunk_size=16：不够一个完整 chunk → 0 个（pad_tail=true 直接 break）
    EXPECT_TRUE(r_chunk.chunks.empty(), "5 frames < 16: empty chunks");

    // 造 32 帧 → 应产出 2 个 chunk（hop=1, pad_tail=true）
    // start=0: [0..15] → valid chunk
    // start=1: [1..16] → valid chunk
    // ...
    // start=16: [16..31] → valid chunk
    // start=17: [17..32] → out of range, break
    // total: 17 chunks
    std::vector<std::vector<float>> frames_32(32, MakeSineFrame(800));
    auto r_batch_32 = extractor.ExtractBatch(frames_32);
    auto r_chunk_32 = extractor.BuildWav2LipChunks(r_batch_32.mel);
    EXPECT_TRUE(r_chunk_32.success, "32 frames: success");
    EXPECT_TRUE(r_chunk_32.chunks.size() > 0, "32 frames: chunks generated");
    // 每个 chunk 大小 = 16 × 80 = 1280
    for (auto& c : r_chunk_32.chunks) {
        EXPECT_EQ(static_cast<int>(c.size()), 1280, "chunk size = 1280");
    }
    EXPECT_EQ(r_chunk_32.n_mels, 80, "chunk n_mels = 80");
    EXPECT_EQ(r_chunk_32.chunk_size, 16, "chunk_size = 16");

    // -----------------------------------------------------------------------
    // 14. BuildWav2LipChunks — time-major 布局
    // -----------------------------------------------------------------------
    std::cout << "[14] BuildWav2LipChunks: time-major ...\n";

    MelChunkOptions opt_tm;
    opt_tm.layout = MelChunkLayout::kTimeMajor16x80;
    auto r_chunk_tm = extractor.BuildWav2LipChunks(r_batch_32.mel, opt_tm);
    EXPECT_TRUE(r_chunk_tm.success, "time-major: success");
    if (!r_chunk_tm.chunks.empty()) {
        EXPECT_EQ(static_cast<int>(r_chunk_tm.chunks[0].size()), 1280, "time-major: size=1280");
    }

    // -----------------------------------------------------------------------
    // 15. Move 语义
    // -----------------------------------------------------------------------
    std::cout << "[15] Move semantics ...\n";

    MelFeatureExtractor moved_extractor = std::move(extractor);
    auto r_moved = moved_extractor.Extract(frame_800);
    EXPECT_TRUE(r_moved.success, "moved extractor: success");

    // =======================================================================
    std::cout << "\n";
    if (g_fail == 0) {
        std::cout << "ALL PASSED.\n";
        return 0;
    } else {
        std::cerr << g_fail << " TEST(S) FAILED.\n";
        return 1;
    }
}
