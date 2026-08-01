/// @file    audio_framer_unit_test.cpp
/// @brief   AudioFramer 单元测试：状态码/工厂/计算/窗函数/分帧/边界
/// @note    参考 tests/audio/audio_loader_unit_test.cpp 风格

#include <algorithm>   // std::min, std::minmax_element, std::abs
#include <cmath>       // std::sqrt, std::isnan, std::isinf, M_PI
#include <iostream>
#include <limits>      // std::numeric_limits
#include <string>
#include <vector>

#include "audio/audio_framer.h"

using namespace digital_human::audio;

// ============================================================================
// 让 enum class 可被直接打印
// ============================================================================
inline std::ostream& operator<<(std::ostream& os, AudioFrameStatus s) {
    return os << AudioFramer::StatusToString(s);
}

inline std::ostream& operator<<(std::ostream& os, AudioWindowType w) {
    switch (w) {
        case AudioWindowType::kNone:    return os << "kNone";
        case AudioWindowType::kHamming: return os << "kHamming";
        case AudioWindowType::kHann:    return os << "kHann";
        default:                        return os << "?";
    }
}

inline std::ostream& operator<<(std::ostream& os, AudioTailPolicy p) {
    switch (p) {
        case AudioTailPolicy::kCoverLastSample: return os << "kCoverLastSample";
        case AudioTailPolicy::kStartEveryHop:   return os << "kStartEveryHop";
        case AudioTailPolicy::kDropIncomplete:  return os << "kDropIncomplete";
        default:                                 return os << "?";
    }
}

// ============================================================================
// 简易断言宏
// ============================================================================
static int g_fail = 0;

#define EXPECT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  FAIL: " << msg << "\n"; \
            g_fail++; \
        } \
    } while(0)

#define EXPECT_FALSE(cond, msg) EXPECT_TRUE(!(cond), msg)

#define EXPECT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "  FAIL: " << msg \
                      << " (expected=" << (b) << ", actual=" << (a) << ")\n"; \
            g_fail++; \
        } \
    } while(0)

// ============================================================================
// 辅助函数：生成递增 ramp PCM（0, 1, 2, ..., N-1），方便验证切片位置
// ============================================================================
static std::vector<float> MakeRampPcm(size_t n) {
    std::vector<float> pcm(n);
    for (size_t i = 0; i < n; i++) pcm[i] = static_cast<float>(i);
    return pcm;
}

// ============================================================================
int main() {
    std::cout << "=== AudioFramer Unit Tests ===\n\n";

    // -----------------------------------------------------------------------
    // 1. StatusToString — 12 个状态码全部返回非空
    // -----------------------------------------------------------------------
    std::cout << "[1] StatusToString (12 codes) ...\n";

    std::vector<AudioFrameStatus> all_status = {
        AudioFrameStatus::kOk,
        AudioFrameStatus::kEmptyInput,
        AudioFrameStatus::kInvalidSampleRate,
        AudioFrameStatus::kInvalidFrameDuration,
        AudioFrameStatus::kInvalidHopDuration,
        AudioFrameStatus::kInvalidFrameSize,
        AudioFrameStatus::kInvalidHopSize,
        AudioFrameStatus::kInvalidTailPolicy,
        AudioFrameStatus::kInvalidWindowType,
        AudioFrameStatus::kInvalidPcmData,
        AudioFrameStatus::kNoFramesGenerated,
        AudioFrameStatus::kUnknownError,
    };

    for (auto s : all_status) {
        EXPECT_FALSE(AudioFramer::StatusToString(s).empty(),
                     "StatusToString should be non-empty");
    }

    // -----------------------------------------------------------------------
    // 2. SpeechDefault / Wav2LipDefault — 工厂函数输出正确配置
    // -----------------------------------------------------------------------
    std::cout << "[2] Factory functions ...\n";

    auto sp = AudioFramer::SpeechDefault();
    EXPECT_EQ(sp.sample_rate, 16000, "SpeechDefault: sample_rate == 16000");
    EXPECT_TRUE(std::abs(sp.frame_duration_ms - 25.0) < 0.01, "SpeechDefault: 25ms");
    EXPECT_TRUE(std::abs(sp.hop_duration_ms - 10.0) < 0.01, "SpeechDefault: 10ms");
    EXPECT_EQ(sp.window_type, AudioWindowType::kHamming, "SpeechDefault: Hamming");
    EXPECT_EQ(sp.tail_policy, AudioTailPolicy::kCoverLastSample, "SpeechDefault: CoverLast");

    auto wl = AudioFramer::Wav2LipDefault();
    EXPECT_EQ(wl.sample_rate, 16000, "Wav2LipDefault: sample_rate == 16000");
    EXPECT_TRUE(std::abs(wl.frame_duration_ms - 50.0) < 0.01, "Wav2LipDefault: 50ms");
    EXPECT_TRUE(std::abs(wl.hop_duration_ms - 12.5) < 0.01, "Wav2LipDefault: 12.5ms");

    // -----------------------------------------------------------------------
    // 3. ComputeFrameSize / ComputeHopSize — 毫秒 → 采样点
    // -----------------------------------------------------------------------
    std::cout << "[3] ComputeFrameSize / ComputeHopSize ...\n";

    EXPECT_EQ(AudioFramer::ComputeFrameSize(16000, 25.0), 400, "16000×25ms = 400");
    EXPECT_EQ(AudioFramer::ComputeFrameSize(16000, 50.0), 800, "16000×50ms = 800");
    EXPECT_EQ(AudioFramer::ComputeHopSize(16000, 10.0), 160, "16000×10ms = 160");
    EXPECT_EQ(AudioFramer::ComputeHopSize(16000, 12.5), 200, "16000×12.5ms = 200");

    // -----------------------------------------------------------------------
    // 4. ComputeNumFrames — 三种尾部策略（L=1600, N=400, S=160）
    // -----------------------------------------------------------------------
    std::cout << "[4] ComputeNumFrames ...\n";

    // sample_count=0 返回 0
    EXPECT_EQ(AudioFramer::ComputeNumFrames(0, 400, 160, AudioTailPolicy::kCoverLastSample),
              0, "empty PCM -> 0 frames");

    // 三种策略
    int64_t sample_count = 1600;
    int frame_size = 400;
    int hop_size = 160;
    EXPECT_EQ(AudioFramer::ComputeNumFrames(sample_count, frame_size, hop_size, AudioTailPolicy::kCoverLastSample),
              9, "CoverLast: (1600,400,160) = 9");
    EXPECT_EQ(AudioFramer::ComputeNumFrames(sample_count, frame_size, hop_size, AudioTailPolicy::kStartEveryHop),
              10, "StartEveryHop: (1600,400,160) = 10");
    EXPECT_EQ(AudioFramer::ComputeNumFrames(sample_count, frame_size, hop_size, AudioTailPolicy::kDropIncomplete),
              8, "DropIncomplete: (1600,400,160) = 8");

    // L < N 时 CoverLast 返回 1 帧（补零）
    EXPECT_EQ(AudioFramer::ComputeNumFrames(100, 400, 160, AudioTailPolicy::kCoverLastSample),
              1, "L<N -> 1 frame");

    // -----------------------------------------------------------------------
    // 5. GenerateWindow — Hamming / Hann / None + 边界
    // -----------------------------------------------------------------------
    std::cout << "[5] GenerateWindow ...\n";

    // frame_size=0 → 空
    EXPECT_TRUE(AudioFramer::GenerateWindow(0, AudioWindowType::kHamming).empty(),
                "frame_size=0 -> empty");

    // frame_size=1 → [1.0]
    auto w1 = AudioFramer::GenerateWindow(1, AudioWindowType::kHamming);
    EXPECT_EQ(w1.size(), size_t(1), "frame_size=1 -> size 1");
    EXPECT_TRUE(std::abs(w1[0] - 1.0f) < 1e-6f, "frame_size=1 -> w[0]=1.0");

    // None → 全 1
    auto w_none = AudioFramer::GenerateWindow(100, AudioWindowType::kNone);
    EXPECT_EQ(w_none.size(), size_t(100), "None: size=100");
    for (auto v : w_none) EXPECT_TRUE(std::abs(v - 1.0f) < 1e-6f, "None: all 1.0");

    // Hamming: 两端 ≈ 0.08，中间 ≈ 1.0
    auto w_hamm = AudioFramer::GenerateWindow(400, AudioWindowType::kHamming);
    EXPECT_EQ(w_hamm.size(), size_t(400), "Hamming: size=400");
    EXPECT_TRUE(w_hamm[0] < 0.1,   "Hamming[0] ≈ 0.08");
    EXPECT_TRUE(w_hamm[0] > 0.06,  "Hamming[0] > 0.06");
    EXPECT_TRUE(w_hamm[399] < 0.1, "Hamming[399] ≈ 0.08");
    EXPECT_TRUE(w_hamm[199] > 0.99,"Hamming[199] ≈ 1.0");

    // Hann: 两端 ≈ 0.0
    auto w_hann = AudioFramer::GenerateWindow(400, AudioWindowType::kHann);
    EXPECT_TRUE(std::abs(w_hann[0]) < 1e-6f,   "Hann[0] ≈ 0");
    EXPECT_TRUE(std::abs(w_hann[399]) < 1e-6f, "Hann[399] ≈ 0");
    EXPECT_TRUE(w_hann[199] > 0.99,             "Hann[199] ≈ 1.0");

    // -----------------------------------------------------------------------
    // 6. CreateFrame — 正常切取 / pad_tail / 不补零
    // -----------------------------------------------------------------------
    std::cout << "[6] CreateFrame ...\n";

    AudioFramer framer;  // 默认 SpeechDefault: 25ms/10ms → frame=400, hop=160
    auto ramp = MakeRampPcm(1600);  // [0, 1, 2, ..., 1599]

    // 6a: 正常切帧（第 0 帧）
    auto f0 = framer.CreateFrame(ramp, 0, true);
    EXPECT_EQ(f0.size(), size_t(400), "frame 0: size=400");
    EXPECT_TRUE(std::abs(f0[0] - 0.0f) < 1e-6f, "frame 0[0] = 0");
    EXPECT_TRUE(std::abs(f0[399] - 399.0f) < 1e-6f, "frame 0[399] = 399");

    // 6b: 尾部补零（start=1280, 1600 只有 320 个真实点，缺 80 个补零）
    auto f_tail = framer.CreateFrame(ramp, 1280, true);
    EXPECT_EQ(f_tail.size(), size_t(400), "tail with pad: size=400");
    EXPECT_TRUE(std::abs(f_tail[0] - 1280.0f) < 1e-6f, "tail pad: [0]=1280");
    EXPECT_TRUE(std::abs(f_tail[319] - 1599.0f) < 1e-6f, "tail pad: [319]=1599");
    EXPECT_TRUE(std::abs(f_tail[320] - 0.0f) < 1e-6f, "tail pad: [320]=0 (padded)");

    // 6c: 不补零（start=1280, 只返回 320 个点）
    auto f_nopad = framer.CreateFrame(ramp, 1280, false);
    EXPECT_EQ(f_nopad.size(), size_t(320), "no pad: only 320 samples");

    // -----------------------------------------------------------------------
    // 7. ApplyWindow — 加窗后值被衰减
    // -----------------------------------------------------------------------
    std::cout << "[7] ApplyWindow ...\n";

    std::vector<float> test_frame(400, 1.0f);  // 全 1，方便观察窗形状
    framer.ApplyWindow(test_frame);
    // 加 Hamming 后两端应 < 0.1，中间 ≈ 1.0
    EXPECT_TRUE(test_frame[0] < 0.1f, "ApplyWindow: edge < 0.1");
    EXPECT_TRUE(test_frame[199] > 0.99f, "ApplyWindow: center ≈ 1.0");
    EXPECT_TRUE(test_frame[399] < 0.1f, "ApplyWindow: edge < 0.1");

    // -----------------------------------------------------------------------
    // 8. Frame — 空输入 / NaN / 正常流程
    // -----------------------------------------------------------------------
    std::cout << "[8] Frame: empty / NaN ...\n";

    // 8a: 空 PCM
    auto r_empty = framer.Frame({});
    EXPECT_FALSE(r_empty.success, "empty PCM -> fail");
    EXPECT_EQ(r_empty.status, AudioFrameStatus::kEmptyInput, "empty -> kEmptyInput");

    // 8b: PCM 含 NaN
    std::vector<float> nan_pcm = {1.0f, std::nanf(""), 3.0f};
    auto r_nan = framer.Frame(nan_pcm);
    EXPECT_FALSE(r_nan.success, "NaN PCM -> fail");
    EXPECT_EQ(r_nan.status, AudioFrameStatus::kInvalidPcmData, "NaN -> kInvalidPcmData");

    // 8c: PCM 含 Inf
    std::vector<float> inf_pcm = {1.0f, std::numeric_limits<float>::infinity()};
    auto r_inf = framer.Frame(inf_pcm);
    EXPECT_FALSE(r_inf.success, "Inf PCM -> fail");
    EXPECT_EQ(r_inf.status, AudioFrameStatus::kInvalidPcmData, "Inf -> kInvalidPcmData");

    // -----------------------------------------------------------------------
    // 9. Frame — 正常流程（1600 样本，frame=400, hop=160 → 9 帧）
    // -----------------------------------------------------------------------
    std::cout << "[9] Frame: normal 1600 samples ...\n";

    auto r = framer.Frame(ramp);
    EXPECT_TRUE(r.success, "normal frame: success");
    EXPECT_EQ(r.status, AudioFrameStatus::kOk, "status == kOk");
    EXPECT_EQ(r.info.num_frames, 9, "1600 samples -> 9 frames");
    EXPECT_EQ(static_cast<int>(r.frames.size()), 9, "frames.size() == 9");
    EXPECT_EQ(r.info.frame_size, 400, "frame_size == 400");
    EXPECT_EQ(r.info.hop_size, 160, "hop_size == 160");
    EXPECT_EQ(r.info.overlap_size, 240, "overlap == 240");
    EXPECT_EQ(r.info.original_sample_count, int64_t(1600), "original == 1600");
    EXPECT_EQ(r.info.padded_sample_count, int64_t(1680), "padded == 1680");
    EXPECT_EQ(r.info.pad_sample_count, int64_t(80), "pad == 80");

    // 每帧元数据验证
    EXPECT_EQ(r.frames[0].index, 0, "frame[0].index == 0");
    EXPECT_EQ(r.frames[0].start_sample, int64_t(0), "frame[0] start=0");
    EXPECT_TRUE(std::abs(r.frames[0].start_ms - 0.0) < 0.01, "frame[0] start_ms=0");
    EXPECT_FALSE(r.frames[0].contains_padding, "frame[0] no padding");

    EXPECT_EQ(r.frames[1].index, 1, "frame[1].index == 1");
    EXPECT_EQ(r.frames[1].start_sample, int64_t(160), "frame[1] start=160");

    // 最后一帧含有 padding
    EXPECT_EQ(r.frames[8].index, 8, "frame[8].index == 8");
    EXPECT_EQ(r.frames[8].start_sample, int64_t(1280), "frame[8] start=1280");
    EXPECT_TRUE(r.frames[8].contains_padding, "frame[8] contains padding");

    // 加窗后两端值被衰减、中间接近原值（ramp 位置准确已在 test 6 CreateFrame 验证）
    EXPECT_TRUE(r.frames[0].samples[0] < 1.0f, "frame[0][0] windowed (edge)");
    EXPECT_TRUE(r.frames[0].samples[199] > 199.0f * 0.99f,
                "frame[0][199] ≈ 199 (center, near original)");
    // 最后一帧 padding 部分为 0
    EXPECT_TRUE(std::abs(r.frames[8].samples[320] - 0.0f) < 1e-6f,
                "frame[8][320] = 0 (padded)");

    // -----------------------------------------------------------------------
    // 10. Frame — Wav2Lip 配置（50ms/12.5ms → frame=800, hop=200）
    // -----------------------------------------------------------------------
    std::cout << "[10] Frame: Wav2Lip config ...\n";

    AudioFramer wl_framer(AudioFramer::Wav2LipDefault());
    auto r_wl = wl_framer.Frame(ramp);
    EXPECT_TRUE(r_wl.success, "Wav2Lip: success");
    EXPECT_EQ(r_wl.info.frame_size, 800, "Wav2Lip: frame_size=800");
    EXPECT_EQ(r_wl.info.hop_size, 200, "Wav2Lip: hop_size=200");
    // 1600 样本, frame=800, hop=200, CoverLast: 1+ceil((1600-800)/200)=1+4=5
    EXPECT_EQ(r_wl.info.num_frames, 5, "Wav2Lip: 1600 samples -> 5 frames");

    // -----------------------------------------------------------------------
    // 11. Frame — kDropIncomplete（尾部不补零）
    // -----------------------------------------------------------------------
    std::cout << "[11] Frame: kDropIncomplete ...\n";

    AudioFrameOptions opt_drop;
    opt_drop.tail_policy = AudioTailPolicy::kDropIncomplete;
    AudioFramer drop_framer(opt_drop);
    auto r_drop = drop_framer.Frame(ramp);
    EXPECT_TRUE(r_drop.success, "DropIncomplete: success");
    EXPECT_EQ(r_drop.info.num_frames, 8, "DropIncomplete: 1600 -> 8 frames");
    EXPECT_EQ(r_drop.info.pad_sample_count, int64_t(0), "Drop: no padding");

    // -----------------------------------------------------------------------
    // 12. Frame — 短输入（L < frame_size）
    // -----------------------------------------------------------------------
    std::cout << "[12] Frame: short input ...\n";

    auto short_pcm = MakeRampPcm(200);  // 200 < 400
    auto r_short = framer.Frame(short_pcm);
    EXPECT_TRUE(r_short.success, "short PCM: success");
    EXPECT_EQ(r_short.info.num_frames, 1, "short: 1 frame (padded)");
    EXPECT_EQ(r_short.frames.size(), size_t(1), "short: 1 frame");
    EXPECT_TRUE(r_short.frames[0].contains_padding, "short: contains padding");
    EXPECT_TRUE(std::abs(r_short.frames[0].samples[0] - 0.0f) < 1e-6f,
                "short frame[0] = 0 (real)");
    EXPECT_TRUE(std::abs(r_short.frames[0].samples[399] - 0.0f) < 1e-6f,
                "short frame[399] = 0 (padded)");

    // -----------------------------------------------------------------------
    // 13. FrameSamplesOnly — 便捷接口
    // -----------------------------------------------------------------------
    std::cout << "[13] FrameSamplesOnly ...\n";

    auto fso = framer.FrameSamplesOnly(ramp);
    EXPECT_EQ(fso.size(), size_t(9), "FrameSamplesOnly: 9 frames");
    EXPECT_EQ(fso[0].size(), size_t(400), "FrameSamplesOnly: [0].size=400");
    // 空 PCM → 空结果
    EXPECT_TRUE(framer.FrameSamplesOnly({}).empty(), "FrameSamplesOnly empty -> empty");

    // -----------------------------------------------------------------------
    // 14. Getters
    // -----------------------------------------------------------------------
    std::cout << "[14] Getters ...\n";

    EXPECT_EQ(framer.GetSampleRate(), 16000, "GetSampleRate");
    EXPECT_EQ(framer.GetFrameSize(), 400, "GetFrameSize");
    EXPECT_EQ(framer.GetHopSize(), 160, "GetHopSize");
    EXPECT_EQ(framer.GetOverlapSize(), 240, "GetOverlapSize");
    EXPECT_TRUE(std::abs(framer.GetOptions().frame_duration_ms - 25.0) < 0.01,
                "GetOptions().frame_duration_ms");

    // -----------------------------------------------------------------------
    // 15. Move 语义
    // -----------------------------------------------------------------------
    std::cout << "[15] Move semantics ...\n";

    AudioFramer moved_framer = std::move(framer);
    auto r_moved = moved_framer.Frame(ramp);
    EXPECT_TRUE(r_moved.success, "moved framer: success");
    EXPECT_EQ(r_moved.info.num_frames, 9, "moved framer: 9 frames");

    // =======================================================================
    // 结果
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
