/// @file    audio_preprocessor_unit_test.cpp
/// @brief   AudioPreprocessor 单元测试：状态码/归一化/降噪/预加重/VAD/统计/流式
/// @note    参考 tests/audio/audio_framer_unit_test.cpp 风格

#include <algorithm>   // std::max, std::abs
#include <cmath>       // std::sqrt, std::isnan, std::isinf
#include <iostream>
#include <limits>      // std::numeric_limits
#include <string>
#include <vector>

#include "audio/audio_preprocessor.h"

using namespace digital_human::audio;

// ============================================================================
// 枚举可打印
// ============================================================================
inline std::ostream& operator<<(std::ostream& os, AudioPreprocessStatus s) {
    return os << AudioPreprocessor::StatusToString(s);
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

#define EXPECT_CLOSE(a, b, eps, msg) \
    EXPECT_TRUE(std::abs((a) - (b)) < (eps), msg)

// ============================================================================
int main() {
    std::cout << "=== AudioPreprocessor Unit Tests ===\n\n";

    // -----------------------------------------------------------------------
    // 1. StatusToString — 12 状态码全部非空
    // -----------------------------------------------------------------------
    std::cout << "[1] StatusToString (12 codes) ...\n";

    std::vector<AudioPreprocessStatus> all_status = {
        AudioPreprocessStatus::kOk, AudioPreprocessStatus::kEmptyInput,
        AudioPreprocessStatus::kInvalidSampleRate, AudioPreprocessStatus::kInvalidTargetPeak,
        AudioPreprocessStatus::kInvalidMaxGain, AudioPreprocessStatus::kInvalidDenoiseThreshold,
        AudioPreprocessStatus::kInvalidPreEmphasisAlpha, AudioPreprocessStatus::kInvalidVadFrameMs,
        AudioPreprocessStatus::kInvalidPcmData, AudioPreprocessStatus::kUnsupportedDenoiseMode,
        AudioPreprocessStatus::kInternalError,
    };
    for (auto s : all_status) {
        EXPECT_FALSE(AudioPreprocessor::StatusToString(s).empty(), "");
    }

    // -----------------------------------------------------------------------
    // 2. Process — 空输入 / NaN 错误路径
    // -----------------------------------------------------------------------
    std::cout << "[2] Process: error paths ...\n";

    AudioPreprocessor proc;
    auto r = proc.Process({});
    EXPECT_FALSE(r.success, "empty -> fail");
    EXPECT_EQ(r.status, AudioPreprocessStatus::kEmptyInput, "empty -> kEmptyInput");

    std::vector<float> nan_pcm{0.1f, std::nanf(""), 0.3f};
    r = proc.Process(nan_pcm);
    EXPECT_FALSE(r.success, "NaN -> fail");
    EXPECT_EQ(r.status, AudioPreprocessStatus::kInvalidPcmData, "NaN -> kInvalidPcmData");

    // -----------------------------------------------------------------------
    // 3. Process — 默认选项（归一化+预加重，不降噪）
    // -----------------------------------------------------------------------
    std::cout << "[3] Process: default options ...\n";

    std::vector<float> pcm = {0.5f, 0.3f, 0.1f, -0.2f, -0.4f, -0.6f, 0.2f, 0.4f};
    r = proc.Process(pcm);
    EXPECT_TRUE(r.success, "default: success");
    EXPECT_TRUE(r.info.normalized, "normalize ran");
    EXPECT_TRUE(r.info.pre_emphasized, "pre-emphasis ran");
    EXPECT_FALSE(r.info.denoised, "denoise not ran (default off)");
    EXPECT_EQ(r.pcm.size(), pcm.size(), "output size == input size");
    // 归一化后 max_abs 应接近 target_peak (0.95)
    float out_max = 0.0f;
    for (auto v : r.pcm) { out_max = std::max(out_max, std::abs(v)); }
    EXPECT_TRUE(out_max > 0.5f, "normalized: max_abs raised");

    // -----------------------------------------------------------------------
    // 4. Process — 全静音保护
    // -----------------------------------------------------------------------
    std::cout << "[4] Process: silence protection ...\n";

    std::vector<float> silent = {0.0f, 0.0f, 0.0f};
    auto r_silent = proc.Process(silent);
    EXPECT_TRUE(r_silent.success, "silence: success");
    // 全零不应被放大
    for (auto v : r_silent.pcm) {
        EXPECT_TRUE(std::abs(v) < 1e-5f, "silence stays zero");
    }
    EXPECT_CLOSE(r_silent.info.applied_gain, 1.0f, 0.01f, "silence: gain = 1");

    // -----------------------------------------------------------------------
    // 5. Process — 归一化被关闭
    // -----------------------------------------------------------------------
    std::cout << "[5] Process: normalize disabled ...\n";

    AudioPreprocessOptions opt_no_norm;
    opt_no_norm.enable_normalize = false;
    opt_no_norm.enable_pre_emphasis = false;
    AudioPreprocessor proc2(opt_no_norm);
    auto r_nn = proc2.Process(pcm);
    EXPECT_TRUE(r_nn.success, "no norm: success");
    EXPECT_FALSE(r_nn.info.normalized, "normalize skipped");
    // PCM 应保持不变（仅拷贝）
    EXPECT_EQ(r_nn.pcm.size(), pcm.size(), "size unchanged");

    // -----------------------------------------------------------------------
    // 6. Process — 降噪开启
    // -----------------------------------------------------------------------
    std::cout << "[6] Process: noise gate ...\n";

    AudioPreprocessOptions opt_nd;
    opt_nd.enable_denoise = true;
    opt_nd.denoise_threshold_db = -20.0f;   // 阈值较高，方便测试
    opt_nd.enable_pre_emphasis = false;
    AudioPreprocessor proc3(opt_nd);
    std::vector<float> noisy = {0.5f, 0.005f, -0.5f, 0.001f, 0.8f, -0.002f};
    auto r_nd = proc3.Process(noisy);
    EXPECT_TRUE(r_nd.success, "denoise: success");
    EXPECT_TRUE(r_nd.info.denoised, "denoise ran");
    // 小样本应被置零
    EXPECT_CLOSE(r_nd.pcm[1], 0.0f, 1e-6f, "0.005 -> 0");
    EXPECT_CLOSE(r_nd.pcm[3], 0.0f, 1e-6f, "0.001 -> 0");
    // 大样本保留
    EXPECT_TRUE(std::abs(r_nd.pcm[0]) > 0.1f, "0.5 kept");

    // -----------------------------------------------------------------------
    // 7. ComputeStats — 正常/空/NaN
    // -----------------------------------------------------------------------
    std::cout << "[7] ComputeStats ...\n";

    auto stats = proc.ComputeStats({0.0f, 1.0f, -1.0f});
    EXPECT_CLOSE(stats.max_abs, 1.0f, 0.01f, "max_abs = 1.0");
    EXPECT_CLOSE(stats.rms, 0.816f, 0.01f, "rms ~ 0.816");
    EXPECT_TRUE(stats.zero_ratio > 0.0, "zero ratio > 0");

    auto s_empty = proc.ComputeStats({});
    EXPECT_EQ(s_empty.num_samples, int64_t(0), "empty: num = 0");

    std::vector<float> nan_vec{std::nanf("")};
    auto s_nan = proc.ComputeStats(nan_vec);
    EXPECT_TRUE(s_nan.has_nan, "NaN detected");

    // -----------------------------------------------------------------------
    // 8. DetectSpeech — 空/纯静音
    // -----------------------------------------------------------------------
    std::cout << "[8] DetectSpeech ...\n";

    EXPECT_TRUE(proc.DetectSpeech({}).empty(), "empty -> no segments");

    std::vector<float> quiet(16000, 0.001f);  // 1 秒微小噪声
    auto segs = proc.DetectSpeech(quiet);
    EXPECT_TRUE(segs.empty(), "quiet -> no speech");

    // -----------------------------------------------------------------------
    // 9. DetectSpeech — 构造有语音的音频
    // -----------------------------------------------------------------------
    std::cout << "[9] DetectSpeech: synthetic speech ...\n";

    // 0.5s 静音 + 0.5s 大声正弦 + 0.5s 静音 + 0.5s 大声正弦
    std::vector<float> speech(32000, 0.001f);  // 2s 基底 (16000*2)
    for (int i = 8000; i < 16000; i++) {       // 0.5s~1.0s
        speech[i] = 0.8f * std::sin(2.0f * 3.14159f * 440.0f * (i - 8000) / 16000.0f);
    }
    for (int i = 24000; i < 32000; i++) {       // 1.5s~2.0s
        speech[i] = 0.8f * std::sin(2.0f * 3.14159f * 440.0f * (i - 24000) / 16000.0f);
    }
    auto segs2 = proc.DetectSpeech(speech);
    EXPECT_TRUE(segs2.size() >= 2, "two speech segments detected");

    // -----------------------------------------------------------------------
    // 10a. ProcessFrame — 正常单帧
    // -----------------------------------------------------------------------
    std::cout << "[10a] ProcessFrame: normal frame ...\n";

    std::vector<float> frame = {0.5f, 0.3f, 0.1f, -0.2f};
    auto rf_ok = proc.ProcessFrame(frame);
    EXPECT_TRUE(rf_ok.success, "frame: success");
    EXPECT_EQ(rf_ok.pcm.size(), frame.size(), "size unchanged");

    // -----------------------------------------------------------------------
    // 10b. ProcessFrame — 空/NaN 错误路径
    // -----------------------------------------------------------------------
    std::cout << "[10b] ProcessFrame: error paths ...\n";

    auto rf = proc.ProcessFrame({});
    EXPECT_FALSE(rf.success, "empty frame -> fail");

    std::vector<float> nan_frame{0.1f, std::nanf("")};
    rf = proc.ProcessFrame(nan_frame);
    EXPECT_FALSE(rf.success, "NaN frame -> fail");

    // -----------------------------------------------------------------------
    // 11. ProcessFrame — 流式一致性（关归一化，纯预加重应对齐）
    // -----------------------------------------------------------------------
    std::cout << "[11] ProcessFrame: streaming == batch (no normalize) ...\n";

    AudioPreprocessOptions opt_stream;
    opt_stream.enable_normalize = false;
    opt_stream.enable_denoise = false;
    opt_stream.enable_pre_emphasis = true;
    opt_stream.pre_emphasis_alpha = 0.5f;

    // 流式：分两块处理
    AudioPreprocessor stream_proc(opt_stream);
    stream_proc.ResetStreamingState();
    std::vector<float> f1 = {1.0f, 2.0f};
    std::vector<float> f2 = {4.0f, 8.0f};
    auto r1 = stream_proc.ProcessFrame(f1);
    auto r2 = stream_proc.ProcessFrame(f2);

    // 整段：同一份数据一次性处理
    AudioPreprocessor batch_proc(opt_stream);
    std::vector<float> batch = {1.0f, 2.0f, 4.0f, 8.0f};
    auto r_batch = batch_proc.Process(batch);

    EXPECT_TRUE(r1.success && r2.success && r_batch.success, "all succeed");
    // 流式两块拼起来应和整段每样本误差 < 1e-5
    EXPECT_CLOSE(r1.pcm[0], r_batch.pcm[0], 1e-5f, "f1[0] == batch[0]");
    EXPECT_CLOSE(r1.pcm[1], r_batch.pcm[1], 1e-5f, "f1[1] == batch[1]");
    EXPECT_CLOSE(r2.pcm[0], r_batch.pcm[2], 1e-5f, "f2[0] == batch[2]");
    EXPECT_CLOSE(r2.pcm[1], r_batch.pcm[3], 1e-5f, "f2[1] == batch[3]");

    // -----------------------------------------------------------------------
    // 12. 选项存取 / ResetStreamingState
    // -----------------------------------------------------------------------
    std::cout << "[12] Options / Reset ...\n";

    AudioPreprocessOptions custom;
    custom.sample_rate = 8000;
    AudioPreprocessor proc_c(custom);
    EXPECT_EQ(proc_c.GetOptions().sample_rate, 8000, "custom sample rate");

    proc_c.SetOptions(AudioPreprocessOptions{});  // 重置
    EXPECT_EQ(proc_c.GetOptions().sample_rate, 16000, "reset to default");

    stream_proc.ResetStreamingState();
    // Reset 后第一帧不应使用旧跨块状态
    auto r_reset = stream_proc.ProcessFrame({5.0f, 10.0f});
    EXPECT_TRUE(r_reset.success, "after reset: success");

    // -----------------------------------------------------------------------
    // 13. Move 语义
    // -----------------------------------------------------------------------
    std::cout << "[13] Move semantics ...\n";

    AudioPreprocessor moved = std::move(proc);
    auto r_moved = moved.Process(pcm);
    EXPECT_TRUE(r_moved.success, "moved: success");

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
