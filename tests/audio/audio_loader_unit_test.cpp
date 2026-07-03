/// @file    audio_loader_unit_test.cpp
/// @brief   AudioLoader 单元测试（当前仅测 Probe + 路径检查 + 状态码）
/// @note    参考 tests/core/image_loader_unit_test.cpp 风格

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "audio/audio_loader.h"

using namespace digital_human::audio;

// ============================================================================
// 让 enum class 可被直接打印
// ============================================================================
inline std::ostream& operator<<(std::ostream& os, AudioLoadStatus s) {
    return os << AudioLoader::StatusToString(s);
}

inline std::ostream& operator<<(std::ostream& os, AudioSampleFormat f) {
    return os << (f == AudioSampleFormat::kInt16 ? "kInt16" : "kFloat32");
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

#define EXPECT_FALSE(cond, msg) \
    EXPECT_TRUE(!(cond), msg)

#define EXPECT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "  FAIL: " << msg \
                      << " (expected=" << (b) << ", actual=" << (a) << ")\n"; \
            g_fail++; \
        } \
    } while(0)

// ============================================================================
int main() {
    std::cout << "=== AudioLoader Unit Tests (Probe) ===\n\n";

    // -----------------------------------------------------------------------
    // 1. StatusToString — 25 个状态码全部返回非空字符串
    // -----------------------------------------------------------------------
    std::cout << "[1] StatusToString (25 codes) ...\n";

    std::vector<AudioLoadStatus> all_status = {
        AudioLoadStatus::kOk,
        AudioLoadStatus::kEmptyPath,
        AudioLoadStatus::kFileNotFound,
        AudioLoadStatus::kPathIsDirectory,
        AudioLoadStatus::kUnsupportedFormat,
        AudioLoadStatus::kInvalidTargetSampleRate,
        AudioLoadStatus::kInvalidTargetChannels,
        AudioLoadStatus::kOpenInputFailed,
        AudioLoadStatus::kFindStreamInfoFailed,
        AudioLoadStatus::kNoAudioStream,
        AudioLoadStatus::kDecoderNotFound,
        AudioLoadStatus::kCodecContextAllocFailed,
        AudioLoadStatus::kCodecParametersFailed,
        AudioLoadStatus::kOpenDecoderFailed,
        AudioLoadStatus::kInvalidSourceAudioParams,
        AudioLoadStatus::kResamplerAllocFailed,
        AudioLoadStatus::kResamplerInitFailed,
        AudioLoadStatus::kPacketAllocFailed,
        AudioLoadStatus::kFrameAllocFailed,
        AudioLoadStatus::kDecodeFailed,
        AudioLoadStatus::kResampleFailed,
        AudioLoadStatus::kFlushFailed,
        AudioLoadStatus::kNoSamplesDecoded,
        AudioLoadStatus::kInvalidPcmData,
        AudioLoadStatus::kFileSystemError,
        AudioLoadStatus::kUnknownError,
    };

    for (auto s : all_status) {
        std::string msg = AudioLoader::StatusToString(s);
        EXPECT_FALSE(msg.empty(), "StatusToString should be non-empty");
    }

    // -----------------------------------------------------------------------
    // 2. GetSupportedExtensions — 返回 6 个扩展名，全小写带点号
    // -----------------------------------------------------------------------
    std::cout << "[2] GetSupportedExtensions ...\n";

    auto exts = AudioLoader::GetSupportedExtensions();
    EXPECT_EQ(exts.size(), size_t(6), "GetSupportedExtensions: size == 6");

    // 检查关键格式都存在
    auto has = [&](const std::string& ext) {
        return std::find(exts.begin(), exts.end(), ext) != exts.end();
    };
    EXPECT_TRUE(has(".wav"), "should contain .wav");
    EXPECT_TRUE(has(".mp3"), "should contain .mp3");
    EXPECT_TRUE(has(".aac"), "should contain .aac");
    EXPECT_TRUE(has(".m4a"), "should contain .m4a");
    EXPECT_TRUE(has(".mp4"), "should contain .mp4");
    EXPECT_TRUE(has(".flac"), "should contain .flac");

    // 验证不含大写字母
    for (const auto& e : exts) {
        EXPECT_TRUE(std::none_of(e.begin(), e.end(),
            [](unsigned char c) { return std::isupper(c); }),
            ("extension should be lowercase: " + e).c_str());
    }

    // -----------------------------------------------------------------------
    // 3. IsSupportedFormat — 大小写不敏感、无扩展名、不支持格式
    // -----------------------------------------------------------------------
    std::cout << "[3] IsSupportedFormat ...\n";

    EXPECT_TRUE(AudioLoader::IsSupportedFormat("test.wav"),    "test.wav");
    EXPECT_TRUE(AudioLoader::IsSupportedFormat("TEST.WAV"),    "TEST.WAV -> true");
    EXPECT_TRUE(AudioLoader::IsSupportedFormat("Test.Mp3"),    "Test.Mp3 -> true");
    EXPECT_TRUE(AudioLoader::IsSupportedFormat("a.M4A"),       "a.M4A -> true");
    EXPECT_FALSE(AudioLoader::IsSupportedFormat("test.txt"),   "test.txt -> false");
    EXPECT_FALSE(AudioLoader::IsSupportedFormat("noext"),      "no ext -> false");
    EXPECT_FALSE(AudioLoader::IsSupportedFormat(""),           "empty -> false");

    // -----------------------------------------------------------------------
    // 4. Probe: 空路径 → kEmptyPath
    // -----------------------------------------------------------------------
    std::cout << "[4] Probe: empty path ...\n";

    AudioLoader loader;
    auto r = loader.Probe("");
    EXPECT_FALSE(r.success, "empty path: success == false");
    EXPECT_EQ(r.status, AudioLoadStatus::kEmptyPath, "empty path -> kEmptyPath");

    // -----------------------------------------------------------------------
    // 5. Probe: 不存在的文件 → kFileNotFound
    // -----------------------------------------------------------------------
    std::cout << "[5] Probe: file not found ...\n";

    r = loader.Probe("__this_file_does_not_exist_9527.wav");
    EXPECT_FALSE(r.success, "not exist: success == false");
    EXPECT_EQ(r.status, AudioLoadStatus::kFileNotFound, "not exist -> kFileNotFound");

    // -----------------------------------------------------------------------
    // 6. Probe: 目录路径 → kPathIsDirectory
    // -----------------------------------------------------------------------
    std::cout << "[6] Probe: directory path ...\n";

    r = loader.Probe(".");   // 当前目录一定存在且是目录
    EXPECT_FALSE(r.success, "directory: success == false");
    EXPECT_EQ(r.status, AudioLoadStatus::kPathIsDirectory, "directory -> kPathIsDirectory");

    // -----------------------------------------------------------------------
    // 7. Probe: 不支持的扩展名 → kUnsupportedFormat
    // -----------------------------------------------------------------------
    std::cout << "[7] Probe: unsupported format ...\n";

    // 创建一个临时 .txt 文件（有扩展名但不支持）
    {
        std::ofstream tmp("__test_tmp.txt");
        tmp << "not an audio file\n";
        tmp.close();
    }
    r = loader.Probe("__test_tmp.txt");
    EXPECT_FALSE(r.success, "unsupported: success == false");
    EXPECT_EQ(r.status, AudioLoadStatus::kUnsupportedFormat, "unsupported -> kUnsupportedFormat");
    std::filesystem::remove("__test_tmp.txt");

    // -----------------------------------------------------------------------
    // 8. Probe: 关闭格式检查时，.txt 不会被格式过滤拦截
    //    （但后续 FFmpeg 打开会失败，返回 kOpenInputFailed）
    // -----------------------------------------------------------------------
    std::cout << "[8] Probe: disable format check ...\n";

    {
        std::ofstream tmp("__test_tmp2.txt");
        tmp << "not audio\n";
        tmp.close();
    }
    AudioLoadOptions opt_no_check;
    opt_no_check.enable_format_check = false;
    r = loader.Probe("__test_tmp2.txt", opt_no_check);
    EXPECT_FALSE(r.success, "no format check: success == false");
    // 预期 kOpenInputFailed（FFmpeg 打不开），不是 kUnsupportedFormat
    EXPECT_EQ(r.status, AudioLoadStatus::kOpenInputFailed,
              "no format check -> kOpenInputFailed (not kUnsupportedFormat)");
    std::filesystem::remove("__test_tmp2.txt");

    // -----------------------------------------------------------------------
    // 9. Probe: 非法目标采样率 → kInvalidTargetSampleRate
    // -----------------------------------------------------------------------
    std::cout << "[9] Probe: invalid target sample rate ...\n";

    AudioLoadOptions opt_bad_rate;
    opt_bad_rate.target_sample_rate = 0;
    r = loader.Probe("test_audio.wav", opt_bad_rate);
    EXPECT_FALSE(r.success, "rate=0: success == false");
    EXPECT_EQ(r.status, AudioLoadStatus::kInvalidTargetSampleRate,
              "rate=0 -> kInvalidTargetSampleRate");

    opt_bad_rate.target_sample_rate = -1;
    r = loader.Probe("test_audio.wav", opt_bad_rate);
    EXPECT_EQ(r.status, AudioLoadStatus::kInvalidTargetSampleRate,
              "rate=-1 -> kInvalidTargetSampleRate");

    // -----------------------------------------------------------------------
    // 10. Probe: 非法目标声道数 → kInvalidTargetChannels
    // -----------------------------------------------------------------------
    std::cout << "[10] Probe: invalid target channels ...\n";

    AudioLoadOptions opt_bad_ch;
    opt_bad_ch.target_channels = 2;   // 当前仅支持 1
    r = loader.Probe("test_audio.wav", opt_bad_ch);
    EXPECT_FALSE(r.success, "channels=2: success == false");
    EXPECT_EQ(r.status, AudioLoadStatus::kInvalidTargetChannels,
              "channels=2 -> kInvalidTargetChannels");

    // -----------------------------------------------------------------------
    // 11. Probe: 正常 WAV 文件 → kOk，源信息完整
    // -----------------------------------------------------------------------
    std::cout << "[11] Probe: valid WAV file ...\n";

    r = loader.Probe("test_audio.wav");
    EXPECT_TRUE(r.success, "valid wav: success == true");
    EXPECT_EQ(r.status, AudioLoadStatus::kOk, "valid wav -> kOk");

    // 源信息必须非空
    const auto& info = r.info;
    EXPECT_FALSE(info.container_format.empty(), "container_format should not be empty");
    EXPECT_FALSE(info.codec_name.empty(),         "codec_name should not be empty");
    EXPECT_TRUE(info.source_sample_rate > 0,      "source_sample_rate > 0");
    EXPECT_TRUE(info.source_channels > 0,         "source_channels > 0");
    EXPECT_FALSE(info.source_sample_format.empty(), "source_sample_format not empty");
    EXPECT_EQ(info.stream_index, 0,               "WAV stream_index == 0");  // WAV 通常流索引为 0

    // Probe 不解码，PCM 必须为空
    EXPECT_TRUE(r.audio.pcm.empty(), "Probe: audio.pcm should be empty");
    EXPECT_EQ(info.sample_count, int64_t(0),      "Probe: sample_count == 0");
    EXPECT_EQ(info.duration_sec, 0.0,             "Probe: duration_sec == 0");

    // 文件路径应该回填
    EXPECT_FALSE(info.file_path.empty(),          "file_path should be recorded");

    // -----------------------------------------------------------------------
    // 12. Probe: 耗时 > 0
    // -----------------------------------------------------------------------
    std::cout << "[12] Probe: timing ...\n";

    EXPECT_TRUE(r.time_ms >= 0.0, "time_ms >= 0");

    // -----------------------------------------------------------------------
    // 13. LoadFromFile — 主接口（float PCM）
    // -----------------------------------------------------------------------
    std::cout << "[13] LoadFromFile (float PCM) ...\n";

    auto lr = loader.LoadFromFile("test_audio.wav");
    EXPECT_TRUE(lr.success, "LoadFromFile should succeed");
    EXPECT_EQ(lr.status, AudioLoadStatus::kOk, "status == kOk");
    EXPECT_FALSE(lr.audio.pcm.empty(), "float pcm should not be empty");
    EXPECT_EQ(lr.audio.sample_rate, 16000, "sample_rate == 16000");
    EXPECT_EQ(lr.audio.channels, 1, "channels == 1");
    EXPECT_EQ(lr.audio.format, AudioSampleFormat::kFloat32, "format == kFloat32");
    // float 范围
    auto [fmin, fmax] = std::minmax_element(lr.audio.pcm.begin(), lr.audio.pcm.end());
    EXPECT_TRUE(*fmin >= -1.0001, "float min >= -1.0");
    EXPECT_TRUE(*fmax <= 1.0001, "float max <= 1.0");
    // int16 和 float 样本数一致
    auto ir_check = loader.LoadInt16FromFile("test_audio.wav");
    EXPECT_EQ(lr.audio.pcm.size(), ir_check.pcm.size(), "float pcm size == int16 pcm size");

    // -----------------------------------------------------------------------
    // 14. ConvertInt16ToFloat — 转换公式验证
    // -----------------------------------------------------------------------
    std::cout << "[14] ConvertInt16ToFloat ...\n";

    // 空输入 → 空输出
    EXPECT_TRUE(AudioLoader::ConvertInt16ToFloat({}).empty(), "empty input -> empty output");

    // 关键值验证：0 / 16384 / -32768
    auto fv = AudioLoader::ConvertInt16ToFloat({0, 16384, -32768});
    EXPECT_EQ(fv.size(), size_t(3), "output size == 3");
    EXPECT_TRUE(std::abs(fv[0] - 0.0f) < 1e-6f,    "0 -> 0.0");
    EXPECT_TRUE(std::abs(fv[1] - 0.5f) < 1e-6f,     "16384 / 32768 -> 0.5");
    EXPECT_TRUE(std::abs(fv[2] - (-1.0f)) < 1e-6f,  "-32768 / 32768 -> -1.0");

    // -----------------------------------------------------------------------
    // 15. GetTargetSampleRate
    // -----------------------------------------------------------------------
    std::cout << "[15] GetTargetSampleRate ...\n";

    EXPECT_EQ(loader.GetTargetSampleRate(), 16000, "default target rate == 16000");

    AudioLoader loader2(22050);
    EXPECT_EQ(loader2.GetTargetSampleRate(), 22050, "custom target rate == 22050");

    // -----------------------------------------------------------------------
    // 16. LoadInt16FromFile: 空路径 → kEmptyPath
    // -----------------------------------------------------------------------
    std::cout << "[16] LoadInt16FromFile: empty path ...\n";

    auto ir = loader.LoadInt16FromFile("");
    EXPECT_FALSE(ir.success, "empty path: success == false");
    EXPECT_EQ(ir.status, AudioLoadStatus::kEmptyPath, "empty path -> kEmptyPath");

    // -----------------------------------------------------------------------
    // 17. LoadInt16FromFile: 不存在的文件 → kFileNotFound
    // -----------------------------------------------------------------------
    std::cout << "[17] LoadInt16FromFile: file not found ...\n";

    ir = loader.LoadInt16FromFile("__no_such_file.wav");
    EXPECT_FALSE(ir.success, "not exist: success == false");
    EXPECT_EQ(ir.status, AudioLoadStatus::kFileNotFound, "not exist -> kFileNotFound");

    // -----------------------------------------------------------------------
    // 18. LoadInt16FromFile: 非法采样率 → kInvalidTargetSampleRate
    // -----------------------------------------------------------------------
    std::cout << "[18] LoadInt16FromFile: invalid sample rate ...\n";

    AudioLoadOptions opt_bad;
    opt_bad.target_sample_rate = 0;
    ir = loader.LoadInt16FromFile("test_audio.wav", opt_bad);
    EXPECT_EQ(ir.status, AudioLoadStatus::kInvalidTargetSampleRate, "rate=0 -> kInvalidTargetSampleRate");

    // -----------------------------------------------------------------------
    // 19. LoadInt16FromFile: 正常 WAV → 解码成功，PCM 非空
    // -----------------------------------------------------------------------
    std::cout << "[19] LoadInt16FromFile: valid WAV decode ...\n";

    ir = loader.LoadInt16FromFile("test_audio.wav");
    EXPECT_TRUE(ir.success, "decode should succeed");
    EXPECT_EQ(ir.status, AudioLoadStatus::kOk, "status == kOk");
    EXPECT_FALSE(ir.pcm.empty(), "pcm should not be empty");

    // -----------------------------------------------------------------------
    // 20. LoadInt16FromFile: 验证输出参数
    // -----------------------------------------------------------------------
    std::cout << "[20] LoadInt16FromFile: verify output params ...\n";

    const auto& i16_info = ir.info;
    // 样本数和输出信息一致
    EXPECT_EQ(i16_info.sample_count, static_cast<int64_t>(ir.pcm.size()),
              "sample_count == pcm.size()");
    // 目标参数
    EXPECT_EQ(i16_info.target_sample_rate, 16000, "target_sample_rate == 16000");
    EXPECT_EQ(i16_info.target_channels, 1, "target_channels == 1");
    EXPECT_EQ(i16_info.target_sample_format, AudioSampleFormat::kInt16,
              "target format == kInt16");
    // 时长计算
    double expected_dur = static_cast<double>(ir.pcm.size()) / 16000.0;
    EXPECT_TRUE(std::abs(i16_info.duration_sec - expected_dur) < 0.01,
                "duration matches sample_count / sample_rate");
    EXPECT_TRUE(i16_info.duration_sec > 0.0, "duration > 0");
    // 源信息也填了（复用 OpenAndFindStream）
    EXPECT_FALSE(i16_info.codec_name.empty(), "source codec_name filled");
    EXPECT_TRUE(i16_info.source_sample_rate > 0, "source_sample_rate > 0");

    // -----------------------------------------------------------------------
    // 21. LoadInt16FromFile: 耗时 > 0
    // -----------------------------------------------------------------------
    std::cout << "[21] LoadInt16FromFile: timing ...\n";

    EXPECT_TRUE(ir.time_ms > 0.0, "decode time_ms > 0");

    // -----------------------------------------------------------------------
    // 22. LoadInt16FromFile: 限制 max_samples
    // -----------------------------------------------------------------------
    std::cout << "[22] LoadInt16FromFile: max_samples limit ...\n";

    AudioLoadOptions opt_limit;
    opt_limit.max_samples = 500;  // 只解码前 500 个采样点
    auto ir_limit = loader.LoadInt16FromFile("test_audio.wav", opt_limit);
    EXPECT_TRUE(ir_limit.success, "max_samples: decode should succeed");
    EXPECT_TRUE(static_cast<int64_t>(ir_limit.pcm.size()) <= 500,
                "pcm.size() <= max_samples");

    // -----------------------------------------------------------------------
    // 23. LoadInt16FromFile: 不同采样率输出
    // -----------------------------------------------------------------------
    std::cout << "[23] LoadInt16FromFile: custom sample rate ...\n";

    AudioLoader loader_8k(8000);
    opt_bad.target_sample_rate = 8000;  // 重置为合法值
    auto ir_8k = loader_8k.LoadInt16FromFile("test_audio.wav", opt_bad);
    EXPECT_TRUE(ir_8k.success, "8kHz decode should succeed");
    EXPECT_EQ(ir_8k.info.target_sample_rate, 8000, "target_sample_rate == 8000");
    // 8kHz 样本数 ≈ 16000 的 1/2
    EXPECT_TRUE(ir_8k.pcm.size() < ir.pcm.size(),
                "8kHz should have fewer samples than 16kHz");

    // -----------------------------------------------------------------------
    // 24. LoadBatch: 正常加载多个文件
    // -----------------------------------------------------------------------
    std::cout << "[24] LoadBatch ...\n";

    auto batch = loader.LoadBatch({"test_audio.wav", "test_audio.wav"});
    EXPECT_EQ(batch.size(), size_t(2), "batch size == 2");
    EXPECT_TRUE(batch[0].success, "batch[0] should succeed");
    EXPECT_TRUE(batch[1].success, "batch[1] should succeed");
    EXPECT_FALSE(batch[0].audio.pcm.empty(), "batch[0] pcm not empty");
    EXPECT_FALSE(batch[1].audio.pcm.empty(), "batch[1] pcm not empty");

    // -----------------------------------------------------------------------
    // 25. LoadBatch: 包含失败文件时下标仍对应，失败项保留
    // -----------------------------------------------------------------------
    std::cout << "[25] LoadBatch: mixed success + failure ...\n";

    auto batch2 = loader.LoadBatch({"test_audio.wav", "__not_exist.wav", "test_audio.wav"});
    EXPECT_EQ(batch2.size(), size_t(3), "batch size == 3");
    EXPECT_TRUE(batch2[0].success,   "[0] should succeed");
    EXPECT_FALSE(batch2[1].success,  "[1] should fail (file not found)");
    EXPECT_TRUE(batch2[2].success,   "[2] should succeed");
    EXPECT_EQ(batch2[1].status, AudioLoadStatus::kFileNotFound, "[1] -> kFileNotFound");

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
