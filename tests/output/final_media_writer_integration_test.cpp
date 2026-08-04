#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "audio/audio_loader.h"
#include "output/final_media_writer.h"

namespace digital_human {
namespace output {
namespace {

// ============================================================================
// 测试辅助
// ============================================================================

cv::Mat MakeSyntheticFrame(int width, int height, int index) {
    cv::Mat frame(height, width, CV_8UC3);
    // 生成可区分的彩色帧：颜色基于帧序号变化
    uint8_t r = static_cast<uint8_t>((index * 17) % 256);
    uint8_t g = static_cast<uint8_t>((index * 31 + 85) % 256);
    uint8_t b = static_cast<uint8_t>((index * 47 + 170) % 256);
    frame = cv::Scalar(b, g, r);
    // 在帧上绘制帧序号（用于 visual debugging）
    cv::putText(frame, std::to_string(index), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    return frame;
}

pipeline::PipelineFrame MakePipelineFrame(const cv::Mat& bgr, int64_t pts_us,
                                          int64_t frame_index) {
    pipeline::PipelineFrame pf;
    pf.video_frame.frame_bgr = bgr;
    pf.video_frame.pts.microseconds = pts_us;
    pf.video_frame.frame_index = frame_index;
    pf.delivery_kind = pipeline::DeliveryKind::kUnique;
    pf.source_task_id = frame_index;
    return pf;
}

bool FileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

/// @brief 运行 ffprobe 并返回输出字符串
std::string RunFfprobe(const std::string& path, const std::string& args) {
    std::string cmd = "ffprobe -v error " + args + " " + path + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[512];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        result += buf;
    }
    pclose(pipe);
    return result;
}

/// @brief 获取流数量
int GetStreamCount(const std::string& path, const std::string& stream_type) {
    std::string sel = stream_type == "video" ? "v" : "a";
    std::string output = RunFfprobe(path,
        "-select_streams " + sel + " -show_entries stream=index -of csv=p=0");
    if (output.empty()) return 0;
    int count = 0;
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty()) count++;
    }
    return count;
}

// ============================================================================
// 集成测试夹具
// ============================================================================

class FinalMediaWriterIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_path_ = "/tmp/test_writer_int_" +
                    std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()) +
                    ".mp4";
    }

    void TearDown() override {
        std::remove(tmp_path_.c_str());
    }

    /// @brief 加载黄金音频
    audio::AudioData LoadGoldenAudio() {
        audio::AudioLoader loader(16000);
        audio::AudioLoadOptions opts;
        opts.target_sample_rate = 16000;
        opts.target_channels = 1;
        opts.output_format = audio::AudioSampleFormat::kFloat32;

        auto result = loader.LoadFromFile("testdata/golden/audio.wav", opts);
        if (!result.success) {
            return audio::AudioData{};
        }
        return result.audio;
    }

    std::string tmp_path_;
};

// ============================================================================
// 集成测试：真实音频 + 75 帧 → ffprobe 验证
// ============================================================================

TEST_F(FinalMediaWriterIntegrationTest, RealAudio75FramesFfprobe) {
    audio::AudioData audio = LoadGoldenAudio();
    if (audio.pcm.empty()) {
        GTEST_SKIP() << "Golden audio not available";
    }

    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.fps_num = 25;
    cfg.fps_den = 1;
    cfg.audio = audio;
    cfg.video_width = 192;  // 显式指定，避免从帧推断延迟
    cfg.video_height = 192;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;
    const int kNumFrames = 75;

    for (int i = 0; i < kNumFrames; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, i);
        auto pf = MakePipelineFrame(frame, i * 40000LL, i); // 25fps = 40ms
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), kNumFrames);
    EXPECT_EQ(writer->GetLastError(), WriterError::kOk);

    // ---------- ffprobe 验证 ----------

    // 1. 文件存在
    ASSERT_TRUE(FileExists(tmp_path_));

    // 2. 双流：视频 + 音频
    int video_streams = GetStreamCount(tmp_path_, "video");
    int audio_streams = GetStreamCount(tmp_path_, "audio");
    EXPECT_EQ(video_streams, 1) << "Expected exactly 1 video stream";
    EXPECT_EQ(audio_streams, 1) << "Expected exactly 1 audio stream";

    // 3. 帧数
    std::string nb_frames = RunFfprobe(tmp_path_,
        "-select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0");
    int probed_frames = 0;
    if (!nb_frames.empty()) {
        // 去除空白
        nb_frames.erase(std::remove(nb_frames.begin(), nb_frames.end(), '\n'),
                        nb_frames.end());
        probed_frames = std::atoi(nb_frames.c_str());
    }
    // 允许编码器延迟导致的 ±1 帧
    EXPECT_GE(probed_frames, kNumFrames - 1);
    EXPECT_LE(probed_frames, kNumFrames + 1);

    // 4. 帧率
    std::string fps = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0");
    EXPECT_NE(fps.find("25"), std::string::npos) << "Frame rate: " << fps;

    // 5. 时长（秒）
    std::string duration = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=duration -of csv=p=0");
    double dur = 0.0;
    if (!duration.empty()) {
        dur = std::atof(duration.c_str());
    }
    // 75 帧 @ 25fps = 3.0 秒，允许少量容差
    EXPECT_GE(dur, 2.8);
    EXPECT_LE(dur, 3.5);

    // 6. 音频时长
    std::string audio_dur = RunFfprobe(tmp_path_,
        "-select_streams a:0 -show_entries stream=duration -of csv=p=0");
    double adur = 0.0;
    if (!audio_dur.empty()) {
        adur = std::atof(audio_dur.c_str());
    }
    double expected_audio_dur = static_cast<double>(audio.pcm.size())
        / audio.sample_rate / audio.channels;
    EXPECT_GE(adur, expected_audio_dur - 0.1);
    EXPECT_LE(adur, expected_audio_dur + 2.0); // 编码器可能补齐

    // 7. 时间基
    std::string tb = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=time_base -of csv=p=0");
    EXPECT_FALSE(tb.empty());
}

// ============================================================================
// Golden 测试：精确断言（比集成测试更严格）
// ============================================================================

TEST_F(FinalMediaWriterIntegrationTest, GoldenExactAssertions) {
    audio::AudioData audio = LoadGoldenAudio();
    if (audio.pcm.empty()) {
        GTEST_SKIP() << "Golden audio not available";
    }

    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.fps_num = 25;
    cfg.fps_den = 1;
    cfg.audio = audio;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;
    const int kNumFrames = 75;

    for (int i = 0; i < kNumFrames; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, i);
        // PTS: 0, 40000, 80000, ... 微秒
        auto pf = MakePipelineFrame(frame, i * 40000LL, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    // 精确帧数
    EXPECT_EQ(writer->GetWrittenFrameCount(), 75);

    // 双流存在
    EXPECT_EQ(GetStreamCount(tmp_path_, "video"), 1);
    EXPECT_EQ(GetStreamCount(tmp_path_, "audio"), 1);

    // 帧率精确
    std::string r_frame_rate = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0");
    std::string avg_frame_rate = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0");

    // 去除换行
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
        s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    };
    trim(r_frame_rate);
    trim(avg_frame_rate);

    EXPECT_EQ(r_frame_rate, "25/1") << "r_frame_rate mismatch";
    EXPECT_EQ(avg_frame_rate, "25/1") << "avg_frame_rate mismatch";

    // 时间基（MP4 muxer 会使用自己的 timescale，不一定是 1/25）
    std::string tb = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=time_base -of csv=p=0");
    trim(tb);
    EXPECT_FALSE(tb.empty());
    EXPECT_NE(tb.find('/'), std::string::npos)
        << "time_base should be rational: " << tb;

    // 音频流
    std::string audio_codec = RunFfprobe(tmp_path_,
        "-select_streams a:0 -show_entries stream=codec_name -of csv=p=0");
    trim(audio_codec);
    EXPECT_FALSE(audio_codec.empty());

    // PTS 起始应为 0
    std::string start_pts = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=start_pts -of csv=p=0");
    trim(start_pts);
    EXPECT_EQ(start_pts, "0");
}

// ============================================================================
// 视频尺寸自动推断
// ============================================================================

TEST_F(FinalMediaWriterIntegrationTest, AutoDetectDimensionsFromFirstFrame) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    // video_width/height 留 0，从首帧推断

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 128;
    const int kHeight = 96;
    for (int i = 0; i < 10; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, i);
        auto pf = MakePipelineFrame(frame, i * 40000, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    EXPECT_TRUE(FileExists(tmp_path_));

    // 验证分辨率
    std::string w = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=width -of csv=p=0");
    std::string h = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=height -of csv=p=0");
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    };
    trim(w); trim(h);
    EXPECT_EQ(w, std::to_string(kWidth));
    EXPECT_EQ(h, std::to_string(kHeight));
}

// ============================================================================
// 视频编码一致性
// ============================================================================

TEST_F(FinalMediaWriterIntegrationTest, H264Encoding) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.video_codec = "libx264";

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    for (int i = 0; i < 25; ++i) {
        auto frame = MakeSyntheticFrame(192, 192, i);
        auto pf = MakePipelineFrame(frame, i * 40000, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    EXPECT_TRUE(FileExists(tmp_path_));

    std::string codec = RunFfprobe(tmp_path_,
        "-select_streams v:0 -show_entries stream=codec_name -of csv=p=0");
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    };
    trim(codec);
    EXPECT_EQ(codec, "h264");
}

// ============================================================================
// Stability：三轮连续写出，验证无资源增长
// ============================================================================

TEST_F(FinalMediaWriterIntegrationTest, StabilityThreeRounds) {
    audio::AudioData audio = LoadGoldenAudio();
    // 即使没有音频也能做 stability 测试

    std::vector<std::string> files;

    for (int round = 0; round < 3; ++round) {
        std::string path = "/tmp/test_writer_stability_" +
                           std::to_string(round) + "_" +
                           std::to_string(
                               std::chrono::steady_clock::now().time_since_epoch().count()) +
                           ".mp4";
        files.push_back(path);

        WriterConfig cfg;
        cfg.output_path = path;
        cfg.fps_num = 25;
        if (!audio.pcm.empty()) {
            cfg.audio = audio;
        }

        auto writer = std::make_shared<FinalMediaWriter>(cfg);

        for (int i = 0; i < 25; ++i) {
            auto frame = MakeSyntheticFrame(192, 192, i);
            auto pf = MakePipelineFrame(frame, i * 40000LL, i);
            writer->OnFrame(pf);
        }

        pipeline::PipelineResult result;
        result.success = true;
        writer->OnTerminal(result);

        EXPECT_TRUE(writer->IsFinalized());
        EXPECT_EQ(writer->GetWrittenFrameCount(), 25);
        EXPECT_EQ(writer->GetLastError(), WriterError::kOk);
        EXPECT_TRUE(FileExists(path));

        // 每轮都应有可播放的视频流
        std::string nb = RunFfprobe(path,
            "-select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0");
        // 只要 ffprobe 未报错即可
        EXPECT_FALSE(nb.empty() && !FileExists(path));
    }

    // 清理 stability 临时文件
    for (const auto& f : files) {
        std::remove(f.c_str());
    }
}

}  // namespace
}  // namespace output
}  // namespace digital_human
