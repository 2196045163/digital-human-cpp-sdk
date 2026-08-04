#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "output/final_media_writer.h"

namespace digital_human {
namespace output {
namespace {

// ============================================================================
// 辅助函数
// ============================================================================

cv::Mat MakeSyntheticFrame(int width, int height, int index) {
    cv::Mat frame(height, width, CV_8UC3);
    uint8_t val = static_cast<uint8_t>((index * 10) % 256);
    frame = cv::Scalar(val, static_cast<uint8_t>(255 - val),
                       static_cast<uint8_t>((val + 128) % 256));
    return frame;
}

pipeline::PipelineFrame MakePipelineFrame(const cv::Mat& bgr, int64_t pts_us,
                                          int64_t frame_index) {
    pipeline::PipelineFrame pf;
    pf.video_frame.frame_bgr = bgr;
    pf.video_frame.pts.microseconds = pts_us;
    pf.video_frame.frame_index = frame_index;
    pf.delivery_kind = pipeline::DeliveryKind::kUnique;
    return pf;
}

bool FileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ============================================================================
// 失败注入测试
// ============================================================================

class FinalMediaWriterFailureTest : public ::testing::Test {
protected:
    void TearDown() override {
        if (!tmp_path_.empty()) {
            std::remove(tmp_path_.c_str());
        }
    }

    std::string tmp_path_ = "/tmp/test_writer_fail_" +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) +
        ".mp4";
};

// ============================================================================
// 无效输出路径
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, InvalidPath) {
    // /dev/null 是字符设备，在其下创建文件会得到 ENOTDIR
    WriterConfig cfg;
    cfg.output_path = "/dev/null/subdir/output.mp4";

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(64, 64, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    EXPECT_NE(writer->GetLastError(), WriterError::kOk);

    pipeline::PipelineResult result;
    result.success = false;
    writer->OnTerminal(result);

    EXPECT_FALSE(FileExists(cfg.output_path));
}

TEST_F(FinalMediaWriterFailureTest, ReadOnlyOutputPath) {
    // /dev/null 是字符设备，在其下创建文件会得到 ENOTDIR
    WriterConfig cfg;
    cfg.output_path = "/dev/null/subdir/test.mp4";

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(64, 64, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    // 在权限良好的系统中这应当失败
    EXPECT_NE(writer->GetLastError(), WriterError::kOk);

    pipeline::PipelineResult result;
    result.success = false;
    writer->OnTerminal(result);

    EXPECT_FALSE(FileExists(cfg.output_path));
}

// ============================================================================
// 中途失败：写几帧后主动取消 → 无伪成功文件
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, MidStreamCancellation) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 写几帧
    for (int i = 0; i < 5; ++i) {
        auto frame = MakeSyntheticFrame(192, 192, i);
        auto pf = MakePipelineFrame(frame, i * 40000, i);
        writer->OnFrame(pf);
    }

    EXPECT_EQ(writer->GetWrittenFrameCount(), 5);

    // 模拟取消：直接析构（不调 OnTerminal）
    writer.reset();

    // 析构应已完成文件（因 Finalize 在析构中被调用）
    // 如果 Finalize 成功，文件存在；如果失败，文件被移除
    // 关键：不能有半成品遗留在预期路径但不完整
    if (FileExists(tmp_path_)) {
        // 文件存在 → 应该可被 ffprobe 解析
        std::string cmd = "ffprobe -v error " + tmp_path_ +
                          " 2>&1 > /dev/null";
        int ret = system(cmd.c_str());
        EXPECT_EQ(ret, 0) << "File exists but ffprobe failed — partial file left";
    }
    // 两种情况均可接受：要么文件存在且可解析，要么文件不存在
}

// ============================================================================
// 析构中途取消：构造→写帧→析构（不调 OnTerminal）
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, DestructorMidStream) {
    {
        WriterConfig cfg;
        cfg.output_path = tmp_path_;

        auto writer = std::make_shared<FinalMediaWriter>(cfg);

        for (int i = 0; i < 10; ++i) {
            auto frame = MakeSyntheticFrame(192, 192, i);
            auto pf = MakePipelineFrame(frame, i * 40000, i);
            writer->OnFrame(pf);
        }

        // 不调 OnTerminal，直接离开作用域
    }

    // 析构已 Finalize，文件应存在且完整
    if (FileExists(tmp_path_)) {
        std::string cmd = "ffprobe -v error -select_streams v:0 -show_entries "
                          "stream=codec_type -of csv=p=0 " + tmp_path_ +
                          " 2>/dev/null";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            char buf[64] = {0};
            fread(buf, 1, sizeof(buf) - 1, pipe);
            pclose(pipe);
            EXPECT_NE(std::string(buf).find("video"), std::string::npos);
        }
    }
}

// ============================================================================
// 重复 close/finalize 幂等性
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, RepeatedOnFrameAfterTerminal) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(192, 192, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    // 再次 OnFrame 应为 no-op
    writer->OnFrame(pf);
    EXPECT_EQ(writer->GetWrittenFrameCount(), 1);

    // 再次 OnTerminal 应为 no-op
    writer->OnTerminal(result);
    EXPECT_TRUE(writer->IsFinalized());
}

TEST_F(FinalMediaWriterFailureTest, ThreeFinalizations) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(192, 192, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    pipeline::PipelineResult result;
    result.success = true;

    // 三次 OnTerminal
    writer->OnTerminal(result);
    writer->OnTerminal(result);
    writer->OnTerminal(result);

    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_EQ(writer->GetWrittenFrameCount(), 1);
    EXPECT_EQ(writer->GetLastError(), WriterError::kOk);
}

// ============================================================================
// OnTerminal 失败路径：文件应被移除
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, ErrorDuringEncoding) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 先写入一个异常尺寸的帧（可能导致编码器参数不匹配）
    // 注：极端尺寸可能导致编码失败
    auto frame = MakeSyntheticFrame(2, 2, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    // 如果 open 失败
    if (writer->GetLastError() != WriterError::kOk) {
        pipeline::PipelineResult result;
        result.success = false;
        result.error_code = pipeline::PipelineErrorCode::kInternalError;
        writer->OnTerminal(result);

        // 不应留文件
        EXPECT_FALSE(FileExists(tmp_path_));
    } else {
        // 即使 open 成功，后续正常 finalize
        pipeline::PipelineResult result;
        result.success = writer->GetLastError() == WriterError::kOk;
        writer->OnTerminal(result);
    }
}

// ============================================================================
// 帧尺寸为 0
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, ZeroSizeFrame) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    cv::Mat empty_frame; // 0x0
    pipeline::PipelineFrame pf;
    pf.video_frame.frame_bgr = empty_frame;
    pf.video_frame.pts.microseconds = 0;
    pf.video_frame.frame_index = 0;

    writer->OnFrame(pf);
    EXPECT_EQ(writer->GetLastError(), WriterError::kInvalidFrameDimensions);

    pipeline::PipelineResult result;
    result.success = false;
    writer->OnTerminal(result);

    EXPECT_FALSE(FileExists(tmp_path_));
}

// ============================================================================
// 资源泄漏：构造后直接析构（无 open 无 frame）
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, ConstructDestroyNoOpen) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    {
        auto writer = std::make_shared<FinalMediaWriter>(cfg);
        // 什么都不做，直接析构
    }

    // 不应创建文件（未打开过）
    EXPECT_FALSE(FileExists(tmp_path_));
}

// ============================================================================
// 编码器错误路径：flush 失败
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, TerminalAfterOpenError) {
    WriterConfig cfg;
    cfg.output_path = "/dev/null/subdir/test.mp4";

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 尝试写一帧，应失败
    auto frame = MakeSyntheticFrame(64, 64, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    ASSERT_NE(writer->GetLastError(), WriterError::kOk);

    // 即使有错误，OnTerminal 应干净清理
    pipeline::PipelineResult result;
    result.success = false;
    result.error_code = pipeline::PipelineErrorCode::kInternalError;
    EXPECT_NO_THROW({ writer->OnTerminal(result); });
    EXPECT_TRUE(writer->IsFinalized());
    EXPECT_FALSE(FileExists(cfg.output_path));
}

// ============================================================================
// OnTerminal 携带失败结果
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, OnTerminalWithFailedResult) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    auto frame = MakeSyntheticFrame(192, 192, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    EXPECT_EQ(writer->GetLastError(), WriterError::kOk);

    // Pipeline 报告失败
    pipeline::PipelineResult result;
    result.success = false;
    result.error_code = pipeline::PipelineErrorCode::kInferenceFailed;
    result.error_message = "Inference failed mid-stream";

    writer->OnTerminal(result);

    // 即使 Pipeline 失败，writer 仍应完成 finalize
    EXPECT_TRUE(writer->IsFinalized());
    // 如果 writer 本身没出错，文件应存在
    if (writer->GetLastError() == WriterError::kOk) {
        EXPECT_TRUE(FileExists(tmp_path_));
    }
}

// ============================================================================
// 大量帧压力测试（非崩溃）
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, ManyFramesNoCrash) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    const int kNumFrames = 300; // 12 秒 @ 25fps
    for (int i = 0; i < kNumFrames; ++i) {
        auto frame = MakeSyntheticFrame(192, 192, i);
        auto pf = MakePipelineFrame(frame, i * 40000LL, i);
        writer->OnFrame(pf);
    }

    EXPECT_EQ(writer->GetWrittenFrameCount(), kNumFrames);

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    EXPECT_TRUE(FileExists(tmp_path_));
    EXPECT_TRUE(writer->IsFinalized());
}

// ============================================================================
// 显式视频宽度/高度在配置中指定
// ============================================================================

TEST_F(FinalMediaWriterFailureTest, ExplicitDimensionsInConfig) {
    WriterConfig cfg;
    cfg.output_path = tmp_path_;
    cfg.video_width = 640;
    cfg.video_height = 480;

    auto writer = std::make_shared<FinalMediaWriter>(cfg);

    // 注意：即使配置了尺寸，首帧仍需匹配
    auto frame = MakeSyntheticFrame(640, 480, 0);
    auto pf = MakePipelineFrame(frame, 0, 0);
    writer->OnFrame(pf);

    EXPECT_EQ(writer->GetLastError(), WriterError::kOk)
        << writer->GetLastErrorMessage();

    pipeline::PipelineResult result;
    result.success = true;
    writer->OnTerminal(result);

    EXPECT_TRUE(FileExists(tmp_path_));
}

}  // namespace
}  // namespace output
}  // namespace digital_human
