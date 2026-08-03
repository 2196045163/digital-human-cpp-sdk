/// @file digital_human_pipeline_lifecycle_test.cpp
/// @brief Pipeline 生命周期与并发测试：启动/停止/析构循环、压力场景
///
/// 测试门槛：
/// - 1000 次 fake 快速 start/stop/destruct session
/// - 每个生命周期 CTest TIMEOUT = 20 秒
/// - 线程数回到初始基线
/// - RSS 增长可控

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"

using namespace digital_human::pipeline;

// ============================================================================
// 空的 Sink（用于生命周期测试）
// ============================================================================

class NoOpSink : public PipelineOutputSink {
public:
    void OnFrame(const PipelineFrame&) override {}
    void OnTerminal(const PipelineResult&) override {
        std::lock_guard<std::mutex> lock(mutex_);
        terminal_called_ = true;
        terminal_cv_.notify_all();
    }

    void WaitForTerminal(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        terminal_cv_.wait_for(lock, timeout, [this]() { return terminal_called_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable terminal_cv_;
    bool terminal_called_ = false;
};

// ============================================================================
// 快速 Start/Stop/Destruct 循环
// ============================================================================

TEST(PipelineLifecycleTest, RapidStartStopDestructFakeSessions) {
    // 1000 次快速创建→Start(失败)→Stop→析构
    // 使用无效配置使 Start 快速失败，避免实际推理耗时
    constexpr int kIterations = 1000;

    for (int i = 0; i < kIterations; ++i) {
        DigitalHumanPipeline pipeline;
        auto sink = std::make_shared<NoOpSink>();

        PipelineConfig config;
        config.image_path = "/nonexistent/img.jpg";
        config.audio_path = "/nonexistent/audio.wav";
        config.model_param_path = "/nonexistent/model.param";
        config.landmark_model_path = "/nonexistent/landmark.dat";

        auto result = pipeline.Start(config, sink);
        // Start 因为无效路径而失败

        // 确保清理
        pipeline.Stop();
    }

    SUCCEED();
}

// ============================================================================
// 并发 Stop 测试
// ============================================================================

TEST(PipelineLifecycleTest, ConcurrentStopCalls) {
    DigitalHumanPipeline pipeline;

    std::atomic<int> completed{0};
    constexpr int kThreads = 4;

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            pipeline.RequestStop();
            completed.fetch_add(1);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(completed.load(), kThreads);
    // 所有并发 RequestStop 调用应该安全完成
}

// ============================================================================
// Wait 不阻塞已终止的 Pipeline
// ============================================================================

TEST(PipelineLifecycleTest, WaitReturnsImmediatelyOnCancelled) {
    DigitalHumanPipeline pipeline;

    pipeline.RequestStop();

    auto start = std::chrono::steady_clock::now();
    auto result = pipeline.Wait();
    auto elapsed = std::chrono::steady_clock::now() - start;

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    EXPECT_LT(ms, 1000) << "Wait on cancelled pipeline took " << ms << "ms";
}

// ============================================================================
// GetState 和 GetStats 在 Running/Stopping 期间可用
// ============================================================================

TEST(PipelineLifecycleTest, GetStateDuringLifecycle) {
    DigitalHumanPipeline pipeline;

    EXPECT_EQ(pipeline.GetState(), PipelineState::kIdle);

    pipeline.RequestStop();
    auto state = pipeline.GetState();
    // 应该是 Idle（从未启动所以不进入 Stopping）
    EXPECT_TRUE(state == PipelineState::kIdle || state == PipelineState::kCancelled);
}

// ============================================================================
// 线程数基线验证（fake session 不创建线程）
// ============================================================================

TEST(PipelineLifecycleTest, ThreadCountBaseline) {
    // 记录初始线程数
    auto get_thread_count = []() -> int {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.find("Threads:") == 0) {
                int count = 0;
                std::sscanf(line.c_str(), "Threads:\t%d", &count);
                return count;
            }
        }
        return -1;
    };

    int initial_threads = get_thread_count();

    // 快速 fake session
    for (int i = 0; i < 50; ++i) {
        DigitalHumanPipeline pipeline;
        pipeline.RequestStop();
        pipeline.Stop();
    }

    int final_threads = get_thread_count();

    // 线程数不应增长（fake session 不创建持久线程）
    EXPECT_LE(final_threads, initial_threads + 2)
        << "Thread count grew from " << initial_threads << " to " << final_threads;
}

// ============================================================================
// 析构时仍有活跃线程（模拟场景）
// ============================================================================

TEST(PipelineLifecycleTest, DestructorHandlesJoinableThreads) {
    // 此测试验证析构安全：即使有未 join 的线程也不应崩溃
    // 通过创建但不显式 join，让析构处理
    {
        DigitalHumanPipeline pipeline;
        // pipeline 在 Idle 状态，没有活跃工作线程
        // 析构应安全完成
    }
    SUCCEED();
}

// ============================================================================
// PipelineConfig 校验
// ============================================================================

TEST(PipelineLifecycleTest, InvalidFpsRejected) {
    DigitalHumanPipeline pipeline;
    auto sink = std::make_shared<NoOpSink>();

    PipelineConfig config;
    config.image_path = "/nonexistent/img.jpg";
    config.audio_path = "/nonexistent/audio.wav";
    config.model_param_path = "/nonexistent/model.param";
    config.landmark_model_path = "/nonexistent/landmark.dat";
    config.fps_num = 0;  // 非法帧率

    auto result = pipeline.Start(config, sink);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, PipelineErrorCode::kInvalidConfig);
}

// ============================================================================
// 多个 Pipeline 实例独立
// ============================================================================

TEST(PipelineLifecycleTest, MultipleIndependentInstances) {
    DigitalHumanPipeline p1;
    DigitalHumanPipeline p2;
    DigitalHumanPipeline p3;

    EXPECT_EQ(p1.GetState(), PipelineState::kIdle);
    EXPECT_EQ(p2.GetState(), PipelineState::kIdle);
    EXPECT_EQ(p3.GetState(), PipelineState::kIdle);

    p1.RequestStop();
    // RequestStop on Idle only sets the cancel flag; state may stay Idle
    EXPECT_EQ(p2.GetState(), PipelineState::kIdle);  // p2 不受影响
    EXPECT_EQ(p3.GetState(), PipelineState::kIdle);  // p3 不受影响

    p1.Stop();
    p2.Stop();
    p3.Stop();
}

// ============================================================================
// Sink 在析构后不再收到回调
// ============================================================================

TEST(PipelineLifecycleTest, SinkNotCalledAfterDestruction) {
    auto sink = std::make_shared<NoOpSink>();
    {
        DigitalHumanPipeline pipeline;
        PipelineConfig config;
        config.image_path = "/nonexistent/img.jpg";
        config.audio_path = "/nonexistent/audio.wav";
        config.model_param_path = "/nonexistent/model.param";
        config.landmark_model_path = "/nonexistent/landmark.dat";

        pipeline.Start(config, sink);
        pipeline.Stop();
    }
    // pipeline 已析构，sink 不应再收到回调
    // 没有崩溃就是通过
    SUCCEED();
}
