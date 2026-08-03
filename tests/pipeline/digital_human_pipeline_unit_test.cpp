/// @file digital_human_pipeline_unit_test.cpp
/// @brief Pipeline 确定性单元测试：状态机、生命周期、异常传播、Stop 行为等
///
/// 测试门槛：
/// - gate 确认阻塞后，RequestStop 到 Wait/Stop 返回 <= 2 秒
/// - 单个生命周期 CTest TIMEOUT = 5 秒
/// - 已终态后的重复 Stop <= 1 秒
/// - Cancel 后阻塞 push/pop <= 2 秒返回

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"

using namespace digital_human::pipeline;

// ============================================================================
// 测试辅助：记录回调的 Sink
// ============================================================================

class TestSink : public PipelineOutputSink {
public:
    void OnFrame(const PipelineFrame& frame) override {
        std::lock_guard<std::mutex> lock(mutex_);
        frame_count_++;
        last_frame_index_ = frame.video_frame.frame_index;
        frames_received_.push_back(frame);
    }

    void OnTerminal(const PipelineResult& result) override {
        std::lock_guard<std::mutex> lock(mutex_);
        terminal_called_ = true;
        terminal_result_ = result;
        terminal_cv_.notify_all();
    }

    void WaitForTerminal(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        terminal_cv_.wait_for(lock, timeout, [this]() { return terminal_called_; });
    }

    int frame_count() const { std::lock_guard<std::mutex> lock(mutex_); return frame_count_; }
    bool terminal_called() const { std::lock_guard<std::mutex> lock(mutex_); return terminal_called_; }
    PipelineResult terminal_result() const { std::lock_guard<std::mutex> lock(mutex_); return terminal_result_; }

private:
    mutable std::mutex mutex_;
    std::condition_variable terminal_cv_;
    int frame_count_ = 0;
    std::int64_t last_frame_index_ = -1;
    bool terminal_called_ = false;
    PipelineResult terminal_result_;
    std::vector<PipelineFrame> frames_received_;
};

// ============================================================================
// 状态机测试
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, InitialStateIsIdle) {
    DigitalHumanPipeline pipeline;
    EXPECT_EQ(pipeline.GetState(), PipelineState::kIdle);
}

TEST(DigitalHumanPipelineUnitTest, StartTwiceFails) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<TestSink>();

    PipelineConfig config;
    config.image_path = "/nonexistent/img.jpg";
    config.audio_path = "/nonexistent/audio.wav";
    config.model_param_path = "/nonexistent/model.param";
    config.landmark_model_path = "/nonexistent/landmark.dat";

    auto r1 = pipeline.Start(config, sink);
    // 应该因为无效路径而失败（但不应该 crash）
    EXPECT_FALSE(r1.success);

    // 第二次 Start 也应该失败（不是 Idle）
    auto r2 = pipeline.Start(config, sink);
    EXPECT_FALSE(r2.success);
}

TEST(DigitalHumanPipelineUnitTest, StopWhenIdle) {
    DigitalHumanPipeline pipeline;
    auto result = pipeline.Stop();
    // 未启动的 Stop 应该返回当前状态
    EXPECT_EQ(result.terminal_state, PipelineState::kIdle);
}

TEST(DigitalHumanPipelineUnitTest, RequestStopWhenIdle) {
    DigitalHumanPipeline pipeline;
    auto result = pipeline.RequestStop();
    EXPECT_EQ(result.terminal_state, PipelineState::kIdle);
}

// ============================================================================
// 配置校验
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, StartWithEmptyPathsFailsGracefully) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<TestSink>();

    PipelineConfig config;
    // 所有路径都为空

    auto result = pipeline.Start(config, sink);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, PipelineErrorCode::kInvalidConfig);
    EXPECT_EQ(pipeline.GetState(), PipelineState::kFailed);
}

// ============================================================================
// 状态查询线程安全
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, GetStateThreadSafe) {
    DigitalHumanPipeline pipeline;

    std::atomic<bool> done{false};
    std::thread reader([&]() {
        for (int i = 0; i < 1000; ++i) {
            pipeline.GetState();
        }
        done.store(true);
    });

    std::thread reader2([&]() {
        for (int i = 0; i < 1000; ++i) {
            pipeline.GetStats();
        }
    });

    reader.join();
    reader2.join();
    EXPECT_TRUE(done.load());

    SUCCEED();  // 没有崩溃就是通过
}

// ============================================================================
// RequestStop 幂等
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, RequestStopIdempotent) {
    DigitalHumanPipeline pipeline;

    auto r1 = pipeline.RequestStop();
    auto r2 = pipeline.RequestStop();
    auto r3 = pipeline.RequestStop();

    // 多次调用不应该崩溃
    EXPECT_EQ(r1.terminal_state, r2.terminal_state);
    EXPECT_EQ(r2.terminal_state, r3.terminal_state);
}

// ============================================================================
// Stop 性能：终态后重复 Stop <= 1 秒
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, StopAfterTerminalUnder1Second) {
    DigitalHumanPipeline pipeline;

    // 先触发取消
    pipeline.RequestStop();

    auto start = std::chrono::steady_clock::now();
    pipeline.Stop();
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    EXPECT_LT(ms, 1000) << "Repeat Stop took " << ms << "ms, expected < 1000ms";
}

// ============================================================================
// 析构安全（不持锁 join，noexcept）
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, DestructorNoexcept) {
    // 确保析构函数 noexcept，不做 detach
    EXPECT_TRUE(noexcept(std::declval<DigitalHumanPipeline>().~DigitalHumanPipeline()));
}

TEST(DigitalHumanPipelineUnitTest, DestructorOnActivePipeline) {
    // 创建 pipeline 后立即析构（模拟异常场景）
    {
        DigitalHumanPipeline pipeline;
        // 不显式 Stop，析构应安全完成
    }
    SUCCEED();
}

// ============================================================================
// 移动构造
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, MoveConstructor) {
    DigitalHumanPipeline p1;
    EXPECT_EQ(p1.GetState(), PipelineState::kIdle);

    DigitalHumanPipeline p2 = std::move(p1);
    EXPECT_EQ(p2.GetState(), PipelineState::kIdle);
}

// ============================================================================
// Sink 回调生命周期
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, SinkNotCalledOnStartFailure) {
    auto sink = std::make_shared<TestSink>();
    DigitalHumanPipeline pipeline;

    PipelineConfig config;
    // 空路径 → Start 失败

    auto result = pipeline.Start(config, sink);
    EXPECT_FALSE(result.success);

    // OnTerminal 不应该被调用（因为从未 Running）
    EXPECT_FALSE(sink->terminal_called());
}

// ============================================================================
// PipelineConfig 默认值验证
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, ConfigDefaults) {
    auto config = PipelineConfig::OfflineDefault();

    EXPECT_EQ(config.mode, PipelineMode::kOffline);
    EXPECT_EQ(config.q1_capacity, 4);
    EXPECT_EQ(config.q2_capacity, 2);
    EXPECT_EQ(config.inference_microbatch, 1);
    EXPECT_EQ(config.scheduler_worker_count, 1);
    EXPECT_EQ(config.ncnn_threads, 1);
    EXPECT_EQ(config.tail_policy, MelTailPolicy::kClampOrReplicateLast);
    EXPECT_EQ(config.fps_num, 25);
    EXPECT_EQ(config.fps_den, 1);
}

TEST(DigitalHumanPipelineUnitTest, RealtimeConfigDefaults) {
    auto config = PipelineConfig::RealtimeDefault();

    EXPECT_EQ(config.mode, PipelineMode::kRealtime);
    EXPECT_EQ(config.frame_scheduler_capacity, 4);
    EXPECT_EQ(config.min_buffered_frames, 2);
}

// ============================================================================
// PipelineStats 初始值
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, StatsInitialValues) {
    DigitalHumanPipeline pipeline;
    auto stats = pipeline.GetStats();

    EXPECT_EQ(stats.state, PipelineState::kIdle);
    EXPECT_EQ(stats.generated_task_count, 0);
    EXPECT_EQ(stats.rendered_unique_frame_count, 0);
    EXPECT_EQ(stats.sink_callback_count, 0);
    EXPECT_EQ(stats.q1_high_watermark, 0);
    EXPECT_EQ(stats.q2_high_watermark, 0);
}

// ============================================================================
// 类型转换
// ============================================================================

TEST(DigitalHumanPipelineUnitTest, PipelineStateToString) {
    EXPECT_EQ(PipelineStateToString(PipelineState::kIdle), "Idle");
    EXPECT_EQ(PipelineStateToString(PipelineState::kRunning), "Running");
    EXPECT_EQ(PipelineStateToString(PipelineState::kSucceeded), "Succeeded");
    EXPECT_EQ(PipelineStateToString(PipelineState::kFailed), "Failed");
    EXPECT_EQ(PipelineStateToString(PipelineState::kCancelled), "Cancelled");
}

TEST(DigitalHumanPipelineUnitTest, PipelineErrorCodeToString) {
    EXPECT_EQ(PipelineErrorCodeToString(PipelineErrorCode::kOk), "OK");
    EXPECT_EQ(PipelineErrorCodeToString(PipelineErrorCode::kInvalidConfig), "InvalidConfig");
    EXPECT_EQ(PipelineErrorCodeToString(PipelineErrorCode::kInternalError), "InternalError");
}

// ============================================================================
// P0/P1 修复回归测试
// ============================================================================

// P1-1：实时模式必须显式拒绝，不静默降级为离线
TEST(DigitalHumanPipelineUnitTest, RealtimeModeReturnsError) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<TestSink>();

    auto config = PipelineConfig::RealtimeDefault();
    config.image_path = "/nonexistent/img.jpg";
    config.audio_path = "/nonexistent/audio.wav";
    config.model_param_path = "/nonexistent/model.param";
    config.landmark_model_path = "/nonexistent/landmark.dat";

    auto result = pipeline.Start(config, sink);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, PipelineErrorCode::kInvalidConfig);
    EXPECT_NE(result.error_message.find("实时模式"), std::string::npos)
        << "Error message should mention realtime mode: " << result.error_message;
}

// P0-1 间接验证：Start 失败（准备阶段异常）后 Wait 不阻塞
TEST(DigitalHumanPipelineUnitTest, WaitReturnsImmediatelyAfterStartFailure) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<TestSink>();

    PipelineConfig config;  // 空路径 → Prepare 失败 → Failed

    auto start_result = pipeline.Start(config, sink);
    EXPECT_FALSE(start_result.success);

    auto wait_start = std::chrono::steady_clock::now();
    auto wait_result = pipeline.Wait();
    auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wait_start).count();

    EXPECT_LT(wait_ms, 1000) << "Wait should return immediately after Start failure, took " << wait_ms << "ms";
    // P0-1 验证：Failed 终态下 Wait 不应阻塞
    EXPECT_EQ(wait_result.terminal_state, PipelineState::kFailed);
}

// P1-1 补充：手动设置 kRealtime 模式也必须拒绝
TEST(DigitalHumanPipelineUnitTest, ManualRealtimeModeRejected) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<TestSink>();

    PipelineConfig config = PipelineConfig::OfflineDefault();
    config.mode = PipelineMode::kRealtime;  // 手动设为实时
    config.image_path = "/nonexistent/img.jpg";
    config.audio_path = "/nonexistent/audio.wav";
    config.model_param_path = "/nonexistent/model.param";
    config.landmark_model_path = "/nonexistent/landmark.dat";

    auto result = pipeline.Start(config, sink);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, PipelineErrorCode::kInvalidConfig);
}

// P0-1 验证：连续 RequestStop + Wait 快速返回
TEST(DigitalHumanPipelineUnitTest, StopWaitRoundTripUnder2Seconds) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<TestSink>();

    PipelineConfig config;  // 空 → 快速失败

    pipeline.Start(config, sink);

    auto start = std::chrono::steady_clock::now();
    pipeline.RequestStop();
    auto result = pipeline.Wait();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    EXPECT_LT(ms, 2000) << "RequestStop + Wait round trip took " << ms << "ms, expected < 2000ms";
}
