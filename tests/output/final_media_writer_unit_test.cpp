#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/stat.h>
#include <unistd.h>

#include "output/final_media_writer.h"

namespace digital_human {
namespace output {
namespace {

// ============================================================================
// 测试辅助
// ============================================================================

/// @brief 创建合成 BGR 帧（纯色或渐变）
cv::Mat MakeSyntheticFrame(int width, int height, uint8_t r, uint8_t g,
                           uint8_t b) {
    cv::Mat frame(height, width, CV_8UC3);
    frame = cv::Scalar(b, g, r); // OpenCV 用 BGR 顺序
    return frame;
}

/// @brief 创建可变色的合成帧（用于区分帧序号）
cv::Mat MakeIndexedFrame(int width, int height, int index) {
    cv::Mat frame(height, width, CV_8UC3);
    uint8_t val = static_cast<uint8_t>((index * 10) % 256);
    frame = cv::Scalar(val, static_cast<uint8_t>(255 - val),
                       static_cast<uint8_t>((val + 128) % 256));
    return frame;
}

/// @brief 构造 PipelineFrame（简化版，仅用于 writer 测试）
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

/// @brief 生成合成 float PCM（正弦波）
std::vector<float> MakeSineWave(int sample_rate, double freq_hz,
                                double duration_sec) {
    int num_samples = static_cast<int>(sample_rate * duration_sec);
    std::vector<float> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        pcm[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * freq_hz * t));
    }
    return pcm;
}

/// @brief 验证文件存在且大小 > 0
bool FileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    return f.tellg() > 0;
}

/// @brief 检查文件是否可被 ffprobe 解析
bool HasVideoStream(const std::string& path) {
    std::string cmd = "ffprobe -v error -select_streams v:0 -show_entries "
                      "stream=codec_type -of csv=p=0 " + path + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;
    char buf[64] = {0};
    fread(buf, 1, sizeof(buf) - 1, pipe);
    pclose(pipe);
    return std::string(buf).find("video") != std::string::npos;
}

// ============================================================================
// 测试夹具
// ============================================================================

class FinalMediaWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_path_ = "/tmp/test_writer_" +
                    std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()) +
                    ".mp4";
    }

    void TearDown() override {
        std::remove(tmp_path_.c_str());
    }

    std::string tmp_path_;
};

// ============================================================================
// 构造与配置
// ============================================================================

TEST_F(FinalMediaWriterTest, ConfigValidation) {
    // 空路径
    WriterConfig cfg;
    cfg.output_path = "";
    EXPECT_FALSE(cfg.IsValid());

    // 正常配置
    cfg.output_path = tmp_path_;
    EXPECT_TRUE(cfg.IsValid());

    // fps=0
    cfg.fps_num = 0;
    EXPECT_FALSE(cfg.IsValid());
    cfg.fps_num = 25;

    // 负尺寸
    cfg.video_width = -1;
    EXPECT_FALSE(cfg.IsValid());
}

TEST_F(FinalMediaWriterTest, ConstructWithInvalidConfigThrows) {
    WriterConfig cfg;
    cfg.output_path = ""; // invalid
    EXPECT_THROW({ FinalMediaWriter w(cfg); }, std::invalid_argument);
}

TEST_F(FinalMediaWriterTest, ConstructWithValidConfig) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    EXPECT_NO_THROW({ FinalMediaWriter w(cfg); });
}

// ============================================================================
// 基本写出流程
// ============================================================================

TEST_F(FinalMediaWriterTest, BasicWriteNoAudio) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 写 30 帧
    const int kWidth = 192;
    const int kHeight = 192;
    for (int i = 0; i < 30; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, 100, 150, 200);
        auto pf = MakePipelineFrame(frame, i * 40000, i); // 25fps = 40ms/frame
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 30);
    EXPECT_EQ(writer->GetLastError(), WriterError::kOk);

    // 文件应该存在且有内容
    EXPECT_TRUE(FileExists(tmp_path_));
    EXPECT_TRUE(HasVideoStream(tmp_path_));
}

TEST_F(FinalMediaWriterTest, BasicWriteWithAudio) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.audio.pcm = MakeSineWave(16000, 440.0, 2.0); // 440Hz, 2 秒
    cfg.audio.sample_rate = 16000;
    cfg.audio.channels = 1;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;
    for (int i = 0; i < 50; ++i) {
        auto frame = MakeIndexedFrame(kWidth, kHeight, i);
        auto pf = MakePipelineFrame(frame, i * 40000, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 50);

    // 检查双流
    EXPECT_TRUE(FileExists(tmp_path_));
    EXPECT_TRUE(HasVideoStream(tmp_path_));

    // 检查音频流
    std::string cmd = "ffprobe -v error -select_streams a:0 -show_entries "
                      "stream=codec_type -of csv=p=0 " + tmp_path_ +
                      " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr);
    char buf[64] = {0};
    fread(buf, 1, sizeof(buf) - 1, pipe);
    pclose(pipe);
    EXPECT_NE(std::string(buf).find("audio"), std::string::npos);
}

// ============================================================================
// 帧计数与 PTS
// ============================================================================

TEST_F(FinalMediaWriterTest, FrameCountExact) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;
    const int kNumFrames = 75;

    for (int i = 0; i < kNumFrames; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, 0, 0, 0);
        auto pf = MakePipelineFrame(frame, i * 40000, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_EQ(writer->GetWrittenFrameCount(), kNumFrames);

    // ffprobe 验证帧数
    std::string cmd = "ffprobe -v error -select_streams v:0 -count_frames "
                      "-show_entries stream=nb_read_frames -of csv=p=0 " +
                      tmp_path_ + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr);
    char buf[64] = {0};
    fread(buf, 1, sizeof(buf) - 1, pipe);
    pclose(pipe);

    int probed_frames = std::atoi(buf);
    // 允许编码器延迟导致的少量差异（B-frames=0 时应该精确）
    EXPECT_GE(probed_frames, kNumFrames - 2);
    EXPECT_LE(probed_frames, kNumFrames + 2);
}

TEST_F(FinalMediaWriterTest, FrameRateCorrect) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.fps_num = 25;
    cfg.fps_den = 1;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;
    const int kNumFrames = 25;
    // 25fps: 40000 us/frame, exactly divides 1000000
    const int64_t kPtsInterval = 40000LL;
    for (int i = 0; i < kNumFrames; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, 0, 0, 0);
        auto pf = MakePipelineFrame(frame, i * kPtsInterval, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    // 验证时长：25 帧 @ 25fps = 1.0 秒
    std::string cmd = "ffprobe -v error -select_streams v:0 -show_entries "
                      "stream=duration -of csv=p=0 " + tmp_path_ +
                      " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr);
    char buf[64] = {0};
    fread(buf, 1, sizeof(buf) - 1, pipe);
    pclose(pipe);

    double duration = std::atof(buf);
    double expected = static_cast<double>(kNumFrames) / 25.0;
    EXPECT_NEAR(duration, expected, 0.1) << "Duration mismatch for 25fps stream";
}

// ============================================================================
// 空帧与零帧场景
// ============================================================================

TEST_F(FinalMediaWriterTest, NoFramesDirectTerminal) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 直接 OnTerminal，无 OnFrame
    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 0);
    // 无帧不应创建文件
    EXPECT_FALSE(FileExists(tmp_path_));
}

TEST_F(FinalMediaWriterTest, EmptyFrameSkipped) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 空帧
    pipeline::PipelineFrame pf;
    writer->OnFrame(pf);

    EXPECT_EQ(writer->GetLastError(), WriterError::kInvalidFrameDimensions);

    // 后续有效帧应跳过
    auto frame = MakeSyntheticFrame(192, 192, 0, 0, 0);
    auto valid_pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(valid_pf);
    EXPECT_EQ(writer->GetWrittenFrameCount(), 0); // 错误后跳过

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);
    EXPECT_TRUE(writer->IsFinalized());
}

// ============================================================================
// 尺寸变更检测
// ============================================================================

TEST_F(FinalMediaWriterTest, DimensionChangeDetected) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto f1 = MakeSyntheticFrame(192, 192, 0, 0, 0);
    auto pf1 = MakePipelineFrame(f1, 0, 0);
    writer->OnFrame(pf1);
    EXPECT_EQ(writer->GetLastError(), WriterError::kOk);

    // 不同尺寸
    auto f2 = MakeSyntheticFrame(256, 256, 0, 0, 0);
    auto pf2 = MakePipelineFrame(f2, 40000, 1);
    writer->OnFrame(pf2);
    EXPECT_EQ(writer->GetLastError(), WriterError::kInvalidFrameDimensions);

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);
}

// ============================================================================
// 幂等 Finalize
// ============================================================================

TEST_F(FinalMediaWriterTest, DoubleFinalizeIdempotent) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(192, 192, 0, 0, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;

    writer->OnTerminal(result);
    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 1);

    // 第二次 OnTerminal — 应静默 no-op
    writer->OnTerminal(result);
    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 1);

    // 第二次 OnFrame — 应静默 no-op
    auto f2 = MakeSyntheticFrame(192, 192, 0, 0, 0);
    auto pf2 = MakePipelineFrame(f2, 40000, 2);
    writer->OnFrame(pf2);
    EXPECT_EQ(writer->GetWrittenFrameCount(), 1); // 未增加
}

// ============================================================================
// 析构清理
// ============================================================================

TEST_F(FinalMediaWriterTest, DestructorWithoutFinalizeCleansUp) {
    {
        WriterConfig cfg;
        cfg.output_path = tmp_path_;

        auto writer = std::make_shared<FinalMediaWriter>(cfg);

        auto frame = MakeSyntheticFrame(192, 192, 0, 0, 0);
        auto pf = MakePipelineFrame(frame, 0, 0);
        writer->OnFrame(pf);

        // 不调用 OnTerminal，直接析构
    }

    // 析构应已写 trailer，文件存在
    EXPECT_TRUE(FileExists(tmp_path_));
}

TEST_F(FinalMediaWriterTest, ErrorPathNoPseudoSuccessFile) {
    // /dev/null 是字符设备而非目录，在此路径下无法创建文件
    std::string bad_path = "/dev/null/subdir/output.mp4";
    WriterConfig cfg;
    cfg.output_path = bad_path;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(192, 192, 0, 0, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    // 在 /dev/null/ 下无法打开文件，Open 应失败
    EXPECT_NE(writer->GetLastError(), WriterError::kOk)
        << "Writer should fail when output path is under /dev/null, "
        << "last error: " << WriterErrorToString(writer->GetLastError());

    pipeline::PipelineResult result;
    result.success = false;
    writer->OnTerminal(result);

    EXPECT_FALSE(FileExists(bad_path));
}

// ============================================================================
// PTS 单调性校验
// ============================================================================

TEST_F(FinalMediaWriterTest, PtsMonotonic) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.fps_num = 25;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;

    // 非单调 PTS（第 5 帧 PTS 回退）
    for (int i = 0; i < 10; ++i) {
        int64_t pts = i * 40000;
        if (i == 5) pts = 100000; // PTS 回退（正常应为 200000）
        auto frame = MakeIndexedFrame(kWidth, kHeight, i);
        auto pf = MakePipelineFrame(frame, pts, i);
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 10);
    EXPECT_TRUE(FileExists(tmp_path_));
}

// ============================================================================
// 错误码覆盖
// ============================================================================

TEST_F(FinalMediaWriterTest, InvalidOutputPathError) {
    // /dev/null 是字符设备而非目录，在其下无法创建文件
    WriterConfig cfg;
    cfg.output_path = "/dev/null/subdir/output.mp4";

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(64, 64, 0, 0, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    EXPECT_NE(writer->GetLastError(), WriterError::kOk)
        << "Writer should fail on path under /dev/null, "
        << "last error: " << WriterErrorToString(writer->GetLastError());
    EXPECT_FALSE(FileExists(cfg.output_path));

    pipeline::PipelineResult result;
    result.success = false;
    writer->OnTerminal(result);
}

// ============================================================================
// 帧速率变化
// ============================================================================

TEST_F(FinalMediaWriterTest, CustomFrameRate) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.fps_num = 30000;
    cfg.fps_den = 1001; // 29.97fps (NTSC)

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kWidth = 192;
    const int kHeight = 192;
    for (int i = 0; i < 30; ++i) {
        auto frame = MakeSyntheticFrame(kWidth, kHeight, 0, 0, 0);
        auto pf = MakePipelineFrame(frame, i * 33367, i); // ~33.367ms
        writer->OnFrame(pf);
    }

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_TRUE(FileExists(tmp_path_));
    EXPECT_TRUE(HasVideoStream(tmp_path_));
}

// ============================================================================
// 不同分辨率
// ============================================================================

TEST_F(FinalMediaWriterTest, OddDimensions) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 奇数尺寸（如 191x191，需要对齐到偶数）
    const int kWidth = 191;
    const int kHeight = 191;
    auto frame = MakeSyntheticFrame(kWidth, kHeight, 100, 100, 100);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    // 可能失败也可能成功，取决于编码器；至少不能崩溃
    pipeline::PipelineResult result;
    result.success = writer->GetLastError() == WriterError::kOk;
    result.terminal_state = writer->GetLastError() == WriterError::kOk
        ? pipeline::PipelineState::kSucceeded
        : pipeline::PipelineState::kFailed;
    writer->OnTerminal(result);
}

// ============================================================================
// 查询接口
// ============================================================================

TEST_F(FinalMediaWriterTest, QueryBeforeOpen) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    FinalMediaWriter writer(cfg);

    EXPECT_FALSE(writer.IsOpen());
    EXPECT_FALSE(writer.IsFinalized());
    EXPECT_EQ(writer.GetWrittenFrameCount(), 0);
    EXPECT_EQ(writer.GetLastError(), WriterError::kOk);
    EXPECT_EQ(writer.GetOutputPath(), tmp_path_);
}

TEST_F(FinalMediaWriterTest, QueryAfterFirstFrame) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(64, 64, 0, 0, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    EXPECT_TRUE(writer->IsOpen());
    EXPECT_FALSE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 1);

    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    EXPECT_FALSE(writer->IsOpen());
    EXPECT_TRUE(writer->IsFinalized());
}

// ============================================================================
// 边角情况：仅音频无视频帧
// ============================================================================

TEST_F(FinalMediaWriterTest, AudioOnlyNoVideoFrames) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.audio.pcm = MakeSineWave(16000, 880.0, 1.0); // 1 秒 880Hz
    cfg.audio.sample_rate = 16000;
    cfg.audio.channels = 1;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 不发任何视频帧，直接 Terminate
    pipeline::PipelineResult result;
    result.success = true;
    result.terminal_state = pipeline::PipelineState::kSucceeded;
    writer->OnTerminal(result);

    // 无视频流不应创建文件
    EXPECT_FALSE(FileExists(tmp_path_));
}

}  // namespace
}  // namespace output
}  // namespace digital_human
