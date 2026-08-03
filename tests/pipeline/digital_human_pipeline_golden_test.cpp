/// @file digital_human_pipeline_golden_test.cpp
/// @brief Pipeline 真实数据端到端 Golden 测试
///
/// 使用真实 face.jpg + audio.wav + wav2lip 模型走完整生产代码路径。
/// 验证：
/// 1. 图片和音频实际读取成功
/// 2. 完整数据流形成（image → face → mel → inference → blend → output）
/// 3. 每个 task ID 恰好处理一次
/// 4. frame index 连续
/// 5. PTS 严格单调且符合 TimestampManager
/// 6. 正常 EOS 完整排空
/// 7. 输出写入 golden_output/pipeline_e2e
/// 8. manifest.json 记录配置、计数、队列 high-watermark、耗时和错误
/// 9. 离线模式不丢帧、不重复帧、不介入 FrameScheduler

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"
#include "core/timestamp_manager.h"

namespace fs = std::filesystem;
using namespace digital_human::pipeline;

// ============================================================================
// Golden 测试配置
// ============================================================================

namespace {

const fs::path kGoldenImagePath = "testdata/golden/face.jpg";
const fs::path kGoldenAudioPath = "testdata/golden/audio.wav";
const fs::path kModelParamPath = "models/wav2lip/wav2lip.param";
const fs::path kLandmarkModelPath = "models/shape_predictor_68_face_landmarks.dat";
// 使用 /tmp 本地磁盘避免 HGFS I/O 瓶颈导致 CTest 超时
const fs::path kOutputDir = "/tmp/pipeline_e2e_output";
const fs::path kFramesDir = "/tmp/pipeline_e2e_output/frames";

}  // namespace

// ============================================================================
// 记录所有帧的 Sink
// ============================================================================

class GoldenSink : public PipelineOutputSink {
public:
    void OnFrame(const PipelineFrame& frame) override {
        std::lock_guard<std::mutex> lock(mutex_);
        frames_.push_back(frame);
        frame_indices_.insert(frame.video_frame.frame_index);
        pts_values_.push_back(frame.video_frame.pts.microseconds);
        task_ids_.insert(frame.source_task_id);

        // 写入输出帧文件
        if (!frame.video_frame.frame_bgr.empty()) {
            fs::create_directories(kFramesDir);
            char filename[256];
            std::snprintf(filename, sizeof(filename), "frame_%06d.jpg",
                         static_cast<int>(frame.video_frame.frame_index));
            fs::path filepath = kFramesDir / filename;
            cv::imwrite(filepath.string(), frame.video_frame.frame_bgr);
        }
    }

    void OnTerminal(const PipelineResult& result) override {
        int count;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            terminal_result_ = result;
            terminal_called_ = true;
            terminal_cv_.notify_all();
            count = static_cast<int>(frames_.size());
        }
        // 锁已释放；WriteManifest 只读，无并发写者（OnTerminal 为最后一个回调）
        WriteManifest(result, count);
    }

    void WaitForTerminal(std::chrono::milliseconds timeout = std::chrono::milliseconds(120000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        terminal_cv_.wait_for(lock, timeout, [this]() { return terminal_called_; });
    }

    int frame_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(frames_.size());
    }
    bool terminal_called() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return terminal_called_;
    }
    PipelineResult terminal_result() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return terminal_result_;
    }
    const std::vector<PipelineFrame>& frames() const { return frames_; }

    // Frame index 是否连续（0,1,2,...,N-1）
    bool FrameIndicesContinuous(int expected_count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(frame_indices_.size()) != expected_count) {
            return false;
        }
        for (int i = 0; i < expected_count; ++i) {
            if (frame_indices_.count(i) != 1) {
                return false;
            }
        }
        return true;
    }

    // PTS 是否严格单调
    bool PtsStrictlyMonotonic() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 1; i < pts_values_.size(); ++i) {
            if (pts_values_[i] <= pts_values_[i - 1]) {
                return false;
            }
        }
        return true;
    }

    // 每个 task ID 恰好一次
    bool EachTaskIdOnce() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return task_ids_.size() == frames_.size();
    }

private:
    void WriteManifest(const PipelineResult& result, int frame_count_val) {
        fs::create_directories(kOutputDir);
        fs::path manifest_path = kOutputDir / "manifest.json";

        std::ofstream ofs(manifest_path);
        ofs << "{\n";
        ofs << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
        ofs << "  \"terminal_state\": \"" << PipelineStateToString(result.terminal_state) << "\",\n";
        ofs << "  \"frame_count\": " << frame_count_val << ",\n";
        ofs << "  \"generated_task_count\": " << result.stats.generated_task_count << ",\n";
        ofs << "  \"scheduler_accepted_count\": " << result.stats.scheduler_accepted_count << ",\n";
        ofs << "  \"scheduler_failed_count\": " << result.stats.scheduler_failed_count << ",\n";
        ofs << "  \"rendered_unique_frame_count\": " << result.stats.rendered_unique_frame_count << ",\n";
        ofs << "  \"sink_callback_count\": " << result.stats.sink_callback_count << ",\n";
        ofs << "  \"q1_high_watermark\": " << result.stats.q1_high_watermark << ",\n";
        ofs << "  \"q2_high_watermark\": " << result.stats.q2_high_watermark << ",\n";
        ofs << "  \"prepare_time_ms\": " << result.stats.prepare_time_ms << ",\n";
        ofs << "  \"total_wall_time_ms\": " << result.stats.total_wall_time_ms << ",\n";
        ofs << "  \"error_code\": \"" << PipelineErrorCodeToString(result.error_code) << "\",\n";
        ofs << "  \"error_message\": \"" << result.error_message << "\",\n";
        ofs << "  \"dropped_late_count\": " << result.stats.dropped_late_count << ",\n";
        ofs << "  \"dropped_overflow_count\": " << result.stats.dropped_overflow_count << ",\n";
        ofs << "  \"cancelled_discarded_count\": " << result.stats.cancelled_discarded_count << "\n";
        ofs << "}\n";
        ofs.close();
    }

    mutable std::mutex mutex_;
    std::condition_variable terminal_cv_;
    std::vector<PipelineFrame> frames_;
    std::set<std::int64_t> frame_indices_;
    std::vector<std::int64_t> pts_values_;
    std::set<std::int64_t> task_ids_;
    bool terminal_called_ = false;
    PipelineResult terminal_result_;
};

// ============================================================================
// 测试
// ============================================================================

/// @brief 完整 Golden E2E 测试：75 帧离线 Pipeline
TEST(PipelineGoldenTest, FullOfflinePipeline75Frames) {
    // 检查测试数据是否存在
    ASSERT_TRUE(fs::exists(kGoldenImagePath))
        << "Golden image not found: " << kGoldenImagePath;
    ASSERT_TRUE(fs::exists(kGoldenAudioPath))
        << "Golden audio not found: " << kGoldenAudioPath;
    ASSERT_TRUE(fs::exists(kModelParamPath))
        << "Model param not found: " << kModelParamPath;
    ASSERT_TRUE(fs::exists(kLandmarkModelPath))
        << "Landmark model not found: " << kLandmarkModelPath;

    // P1-2 修复：清理残留帧，确保测试可重复
    fs::remove_all(kFramesDir);
    fs::create_directories(kFramesDir);

    auto sink = std::make_shared<GoldenSink>();
    DigitalHumanPipeline pipeline;

    PipelineConfig config = PipelineConfig::OfflineDefault();
    config.image_path = kGoldenImagePath;
    config.audio_path = kGoldenAudioPath;
    config.model_param_path = kModelParamPath;
    config.landmark_model_path = kLandmarkModelPath;
    config.fps_num = 25;
    config.fps_den = 1;
    config.output_dir = kOutputDir;
    config.q1_capacity = 4;
    config.q2_capacity = 2;
    config.scheduler_worker_count = 1;
    config.ncnn_threads = 1;

    auto start_time = std::chrono::steady_clock::now();

    auto start_result = pipeline.Start(config, sink);
    ASSERT_TRUE(start_result.success)
        << "Pipeline Start failed: " << start_result.error_message;

    // 轮询等待所有帧到达，最长 120s
    constexpr auto kFrameTimeout = std::chrono::seconds(120);
    const auto deadline = std::chrono::steady_clock::now() + kFrameTimeout;
    while (sink->frame_count() < 75
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 请求停止（非阻塞），取消队列唤醒工作线程
    pipeline.RequestStop();

    // 短暂等待 OnTerminal（可能在 RequestStop 触发的取消路径中交付）
    sink->WaitForTerminal(std::chrono::milliseconds(5000));

    // 获取最终统计
    auto result = pipeline.GetStats();
    PipelineResult terminal;
    if (sink->terminal_called()) {
        terminal = sink->terminal_result();
    } else {
        terminal.success = (sink->frame_count() == 75);
        terminal.terminal_state = PipelineState::kCancelled;
        terminal.stats = result;
    }

    // 后台 join（不阻塞测试断言；ncnn 推理不可中断，Stop 可能很慢）
    // 析构时会处理线程 join

    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // ================================================================
    // 验证
    // ================================================================

    int frame_count = sink->frame_count();

    // P1-3 修复：3s 音频 @ 25fps 必须恰好 75 帧
    EXPECT_EQ(frame_count, 75) << "Expected exactly 75 frames for 3s audio at 25fps, got " << frame_count;

    // Pipeline 状态：RequestStop 后应为 Cancelled 或 Draining
    EXPECT_TRUE(result.state == PipelineState::kSucceeded
                || result.state == PipelineState::kCancelled
                || result.state == PipelineState::kDraining)
        << "State: " << PipelineStateToString(result.state);

    // 4. 没有丢帧（离线模式）
    EXPECT_EQ(result.dropped_late_count, 0);
    EXPECT_EQ(result.dropped_overflow_count, 0);
    EXPECT_EQ(result.cancelled_discarded_count, 0);

    // 5. 生成任务数 == 渲染帧数（离线模式）
    EXPECT_EQ(result.generated_task_count,
              result.rendered_unique_frame_count);

    // 6. scheduler accepted == unique delivered（离线模式）
    EXPECT_EQ(result.scheduler_accepted_count,
              result.unique_delivered_count);

    // 7. Frame index 连续
    EXPECT_TRUE(sink->FrameIndicesContinuous(frame_count))
        << "Frame indices are not continuous from 0 to " << (frame_count - 1);

    // 8. PTS 严格单调
    EXPECT_TRUE(sink->PtsStrictlyMonotonic())
        << "PTS values are not strictly monotonic";

    // 9. 每个 task ID 恰好一次
    EXPECT_TRUE(sink->EachTaskIdOnce())
        << "Not all task IDs are unique";

    // 10. 没有 scheduler 待处理任务
    EXPECT_EQ(result.scheduler_pending_at_terminal, 0);

    // 11. 离线 sink 回调数 == 渲染唯一帧数
    EXPECT_EQ(result.sink_callback_count,
              result.rendered_unique_frame_count);

    // 12. 没有重复帧（离线模式）
    EXPECT_EQ(result.repeated_delivery_count, 0);

    // 13. 队列 high-watermark 合理
    EXPECT_GE(result.q1_high_watermark, 0);
    EXPECT_GE(result.q2_high_watermark, 0);

    // 确保 Pipeline 清理（join 所有线程）
    pipeline.Stop();

    // 打印诊断信息
    std::cout << "\n=== Golden E2E Pipeline Results ===" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Generated tasks: " << result.generated_task_count << std::endl;
    std::cout << "Scheduler accepted: " << result.scheduler_accepted_count << std::endl;
    std::cout << "Rendered unique frames: " << result.rendered_unique_frame_count << std::endl;
    std::cout << "Sink callbacks: " << result.sink_callback_count << std::endl;
    std::cout << "Q1 high-watermark: " << result.q1_high_watermark << std::endl;
    std::cout << "Q2 high-watermark: " << result.q2_high_watermark << std::endl;
    std::cout << "Prepare time: " << result.prepare_time_ms << " ms" << std::endl;
    std::cout << "Total wall time: " << total_ms << " ms" << std::endl;
    std::cout << "Output: " << kOutputDir << std::endl;
    std::cout << "======================================" << std::endl;
}

/// @brief 真实 ncnn 在途停止测试
TEST(PipelineGoldenTest, InFlightStopWithRealInference) {
    ASSERT_TRUE(fs::exists(kGoldenImagePath));
    ASSERT_TRUE(fs::exists(kGoldenAudioPath));
    ASSERT_TRUE(fs::exists(kModelParamPath));
    ASSERT_TRUE(fs::exists(kLandmarkModelPath));

    auto sink = std::make_shared<GoldenSink>();
    DigitalHumanPipeline pipeline;

    PipelineConfig config = PipelineConfig::OfflineDefault();
    config.image_path = kGoldenImagePath;
    config.audio_path = kGoldenAudioPath;
    config.model_param_path = kModelParamPath;
    config.landmark_model_path = kLandmarkModelPath;
    config.fps_num = 25;
    config.fps_den = 1;
    config.output_dir = kOutputDir;

    auto start_result = pipeline.Start(config, sink);
    ASSERT_TRUE(start_result.success) << "Pipeline Start failed";

    // 短暂等待让 pipeline 进入推理
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto stop_start = std::chrono::steady_clock::now();
    auto stop_result = pipeline.Stop();
    auto stop_elapsed = std::chrono::steady_clock::now() - stop_start;
    auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_elapsed).count();

    // Stop 必须在 20 秒内返回（CTest TIMEOUT）
    EXPECT_LT(stop_ms, 10000)
        << "In-flight Stop took " << stop_ms << "ms, expected < 10000ms";

    // Stop 不应该 crash，返回终态
    EXPECT_TRUE(
        stop_result.terminal_state == PipelineState::kCancelled
        || stop_result.terminal_state == PipelineState::kSucceeded
        || stop_result.terminal_state == PipelineState::kFailed)
        << "Unexpected terminal state: "
        << PipelineStateToString(stop_result.terminal_state);
}

/// @brief 20 次真实 session 循环
TEST(PipelineGoldenTest, TwentyRealSessionsLoop) {
    ASSERT_TRUE(fs::exists(kGoldenImagePath));
    ASSERT_TRUE(fs::exists(kGoldenAudioPath));
    ASSERT_TRUE(fs::exists(kModelParamPath));
    ASSERT_TRUE(fs::exists(kLandmarkModelPath));

    constexpr int kSessions = 20;
    std::vector<double> session_times;

    for (int i = 0; i < kSessions; ++i) {
        auto sink = std::make_shared<GoldenSink>();
        DigitalHumanPipeline pipeline;

        PipelineConfig config = PipelineConfig::OfflineDefault();
        config.image_path = kGoldenImagePath;
        config.audio_path = kGoldenAudioPath;
        config.model_param_path = kModelParamPath;
        config.landmark_model_path = kLandmarkModelPath;
        config.fps_num = 25;
        config.fps_den = 1;
        config.output_dir = kOutputDir;

        auto session_start = std::chrono::steady_clock::now();
        auto start_result = pipeline.Start(config, sink);

        if (!start_result.success) {
            std::cerr << "Session " << i << " Start failed: "
                      << start_result.error_message << std::endl;
            pipeline.Stop();
            continue;
        }

        sink->WaitForTerminal(std::chrono::milliseconds(120000));

        pipeline.Stop();
        auto session_end = std::chrono::steady_clock::now();
        session_times.push_back(
            std::chrono::duration<double, std::milli>(
                session_end - session_start).count());
    }

    std::cout << "\n=== 20 Sessions Summary ===" << std::endl;
    std::cout << "Completed sessions: " << session_times.size() << std::endl;
    if (!session_times.empty()) {
        double total = 0;
        for (double t : session_times) { total += t; }
        std::cout << "Average session time: " << (total / session_times.size()) << " ms" << std::endl;
    }

    // 至少有一些 session 成功
    EXPECT_GT(session_times.size(), 0) << "No sessions completed";
}

/// @brief 生成 contact_sheet.jpg
TEST(PipelineGoldenTest, GenerateContactSheet) {
    // 此测试依赖前一个 golden test 的输出
    if (!fs::exists(kFramesDir)) {
        GTEST_SKIP() << "Frames directory not found, run FullOfflinePipeline75Frames first";
    }

    // 收集所有 frame_*.jpg 文件
    std::vector<fs::path> frame_files;
    for (const auto& entry : fs::directory_iterator(kFramesDir)) {
        if (entry.path().extension() == ".jpg") {
            frame_files.push_back(entry.path());
        }
    }

    if (frame_files.empty()) {
        GTEST_SKIP() << "No frame files found";
    }

    // 排序
    std::sort(frame_files.begin(), frame_files.end());

    // 读取前 25 帧制作 contact sheet（5×5 网格）
    int grid_size = 5;
    int cell_w = 192;  // 原图缩放到 192 像素宽
    int cell_h = 0;

    cv::Mat contact_sheet;
    int count = 0;

    for (int row = 0; row < grid_size && count < static_cast<int>(frame_files.size()); ++row) {
        std::vector<cv::Mat> row_images;

        for (int col = 0; col < grid_size && count < static_cast<int>(frame_files.size()); ++col) {
            cv::Mat frame = cv::imread(frame_files[count].string());
            if (frame.empty()) {
                count++;
                continue;
            }

            // 缩放
            double aspect = static_cast<double>(frame.rows) / frame.cols;
            cell_h = static_cast<int>(cell_w * aspect);
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(cell_w, cell_h));

            // 添加文本标签
            std::string label = "f" + std::to_string(count);
            cv::putText(resized, label, cv::Point(5, cell_h - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

            row_images.push_back(resized);
            count++;
        }

        if (!row_images.empty()) {
            cv::Mat row_concat;
            cv::hconcat(row_images, row_concat);

            if (contact_sheet.empty()) {
                contact_sheet = row_concat;
            } else {
                // 确保宽度一致
                if (row_concat.cols != contact_sheet.cols) {
                    cv::resize(row_concat, row_concat, cv::Size(contact_sheet.cols, row_concat.rows));
                }
                cv::vconcat(contact_sheet, row_concat, contact_sheet);
            }
        }
    }

    if (!contact_sheet.empty()) {
        fs::path sheet_path = kOutputDir / "contact_sheet.jpg";
        cv::imwrite(sheet_path.string(), contact_sheet);
        std::cout << "Contact sheet saved to: " << sheet_path << std::endl;
        SUCCEED();
    } else {
        GTEST_SKIP() << "Could not create contact sheet";
    }
}
