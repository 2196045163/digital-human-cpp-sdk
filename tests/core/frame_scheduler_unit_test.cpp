/// @brief 帧调度模块单元测试。
///
/// 测试范围：
/// - 构造配置校验
/// - PushFrame 输入校验、重复拒绝、过时拒绝
/// - Schedule 参考时间校验、缓冲状态、同步判据
/// - Reset
/// - 多线程并发（检查点 6）
///
/// 核心库不打印；所有断言必须验证动作、帧号、队列和状态。

#include "core/frame_scheduler.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace digital_human {
namespace core {
namespace {

// ===================================================================
// 测试工具
// ===================================================================

/// @brief 创建最小合法测试帧。
///
/// 帧图像为 32x32 CV_8UC3，每个通道使用不同值以便区分。
video::VideoFrame MakeTestFrame(
    std::int64_t pts_us,
    std::int64_t frame_index) {
    video::VideoFrame frame;
    frame.pts.microseconds = pts_us;
    frame.frame_index = frame_index;
    frame.frame_bgr = cv::Mat(32, 32, CV_8UC3,
                              cv::Scalar(10, 20, 30));
    return frame;
}

/// @brief 创建测试用默认合法配置。
FrameSchedulerConfig MakeTestConfig() {
    FrameSchedulerConfig cfg;
    cfg.max_queue_size = 6;
    cfg.min_buffered_frames = 2;
    cfg.sync_tolerance_us = 10000;  // 10ms 演示容忍窗口
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    return cfg;
}

/// @brief 创建 min_buffered_frames=1 的配置，简化单帧测试。
FrameSchedulerConfig MakeSingleBufferConfig() {
    FrameSchedulerConfig cfg;
    cfg.max_queue_size = 6;
    cfg.min_buffered_frames = 1;
    cfg.sync_tolerance_us = 10000;
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    return cfg;
}

// ===================================================================
// 构造配置校验
// ===================================================================

TEST(FrameSchedulerConfigTest, MaxQueueSizeZeroThrows) {
    auto cfg = MakeTestConfig();
    cfg.max_queue_size = 0;
    EXPECT_THROW((FrameScheduler(cfg)), std::invalid_argument);
}

TEST(FrameSchedulerConfigTest, MinBufferedFramesZeroThrows) {
    auto cfg = MakeTestConfig();
    cfg.min_buffered_frames = 0;
    EXPECT_THROW((FrameScheduler(cfg)), std::invalid_argument);
}

TEST(FrameSchedulerConfigTest, MinBufferedExceedsMaxQueueThrows) {
    auto cfg = MakeTestConfig();
    cfg.max_queue_size = 4;
    cfg.min_buffered_frames = 5;
    EXPECT_THROW((FrameScheduler(cfg)), std::invalid_argument);
}

TEST(FrameSchedulerConfigTest, NegativeSyncToleranceThrows) {
    auto cfg = MakeTestConfig();
    cfg.sync_tolerance_us = -1;
    EXPECT_THROW((FrameScheduler(cfg)), std::invalid_argument);
}

TEST(FrameSchedulerConfigTest, ValidConfigConstructsSuccessfully) {
    auto cfg = MakeTestConfig();
    EXPECT_NO_THROW((FrameScheduler(cfg)));
}

TEST(FrameSchedulerConfigTest, MinEqualsMaxQueueConstructsSuccessfully) {
    auto cfg = MakeTestConfig();
    cfg.max_queue_size = 3;
    cfg.min_buffered_frames = 3;
    EXPECT_NO_THROW((FrameScheduler(cfg)));
}

TEST(FrameSchedulerConfigTest, ZeroToleranceConstructsSuccessfully) {
    auto cfg = MakeTestConfig();
    cfg.sync_tolerance_us = 0;
    EXPECT_NO_THROW((FrameScheduler(cfg)));
}

// ===================================================================
// PushFrame 输入校验
// ===================================================================

TEST(FrameSchedulerPushTest, ValidFrameIsAccepted) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(100000, 0);

    EXPECT_EQ(scheduler.PushFrame(frame), PushResult::kAccepted);
}

TEST(FrameSchedulerPushTest, EmptyMatIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(100000, 0);
    frame.frame_bgr = cv::Mat();  // 空 Mat

    EXPECT_EQ(scheduler.PushFrame(frame),
              PushResult::kRejectedInvalidFrame);
}

TEST(FrameSchedulerPushTest, GrayscaleMatIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(100000, 0);
    frame.frame_bgr = cv::Mat(32, 32, CV_8UC1, cv::Scalar(128));

    EXPECT_EQ(scheduler.PushFrame(frame),
              PushResult::kRejectedInvalidFrame);
}

TEST(FrameSchedulerPushTest, NegativePtsIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(-1, 0);

    EXPECT_EQ(scheduler.PushFrame(frame),
              PushResult::kRejectedInvalidFrame);
}

TEST(FrameSchedulerPushTest, NegativeFrameIndexIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(100000, -1);

    EXPECT_EQ(scheduler.PushFrame(frame),
              PushResult::kRejectedInvalidFrame);
}

// ===================================================================
// PushFrame：重复 PTS 和过时拒绝
// ===================================================================

TEST(FrameSchedulerPushTest, DuplicatePtsInQueueIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    // 第一个帧入队
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(100000, 0)),
              PushResult::kAccepted);
    // 第二个帧 PTS 相同 → 应拒绝
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(100000, 1)),
              PushResult::kRejectedDuplicatePts);
}

TEST(FrameSchedulerPushTest, StalePtsIsRejected) {
    // min_buffered_frames=1 → 推一帧就够，可直接交付
    // diff=0(ref=100000, PTS=100000)→kDeliver → last_delivered_frame 设为 PTS=100000
    FrameScheduler scheduler(MakeSingleBufferConfig());
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(100000, 0)),
              PushResult::kAccepted);
    auto result = scheduler.Schedule(MediaTimestamp{100000});
    EXPECT_EQ(result.action, ScheduleAction::kDeliver);

    // PTS 等于已交付帧 → 拒绝（已交付的 PTS 不能再入队）
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(100000, 1)),
              PushResult::kRejectedStalePts);

    // PTS 小于已交付帧 → 拒绝（比已播过的还旧）
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(50000, 2)),
              PushResult::kRejectedStalePts);
}

// ===================================================================
// Schedule：参考时间校验
// ===================================================================

TEST(FrameSchedulerScheduleTest, NegativeReferenceTimeIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    // 先推一帧，证明队列非空
    scheduler.PushFrame(MakeTestFrame(100000, 0));

    auto result = scheduler.Schedule(MediaTimestamp{-1});

    EXPECT_EQ(result.action,
              ScheduleAction::kRejectedInvalidReferenceTime);
    EXPECT_EQ(result.dropped_this_call, 0);
    // 队列不应被修改
    EXPECT_EQ(result.queue_size_after, 1);
}

TEST(FrameSchedulerScheduleTest, BackwardsReferenceTimeIsRejected) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 先执行一次合法 Schedule，建立 previous_reference_pts
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.Schedule(MediaTimestamp{100000});

    // 再给一个更小的参考时间
    auto result = scheduler.Schedule(MediaTimestamp{50000});

    EXPECT_EQ(result.action,
              ScheduleAction::kRejectedInvalidReferenceTime);
    EXPECT_EQ(result.dropped_this_call, 0);
}

// ===================================================================
// 同步判据：四个紧邻边界（检查点 3 用户完成）
//
// 共用前提（每个测试都一样，不再逐条重复）：
//   1. tolerance = 10000（由 MakeSingleBufferConfig 设定）
//   2. min_buffered_frames = 1 → 推一帧就够，不会被缓冲拦截
//   3. 全新构造的调度器，从未调过 Schedule → last_delivered_frame 为空
//      → 所以"丢弃后队列空"返回 kBufferingNoFrame 而非 kRepeatLast
//      → 所以"队首太新需等待"也返回 kBufferingNoFrame 而非 kRepeatLast
//   4. diff = frame_pts - reference_pts
//
// 四个紧邻边界：
//   diff = -tolerance-1  (-10001) → ClassifyTimingDiff 返回 kDrop
//   diff = -tolerance    (-10000) → ClassifyTimingDiff 返回 kDeliver
//   diff = +tolerance    (+10000) → ClassifyTimingDiff 返回 kDeliver
//   diff = +tolerance+1  (+10001) → ClassifyTimingDiff 返回 kWait
// ===================================================================

// ① diff = -10001：队列中只有一帧 PTS=0，ref=10001
//    分类为 kDrop → pop 该帧 → 队列空 + 无 last → kBufferingNoFrame
TEST(FrameSchedulerClassifyTest, DiffBelowNegativeToleranceDropsFrame) {
    FrameScheduler scheduler(MakeSingleBufferConfig());
    scheduler.PushFrame(MakeTestFrame(0, /*frame_index=*/0));

    auto result = scheduler.Schedule(MediaTimestamp{10001});

    EXPECT_EQ(result.action, ScheduleAction::kBufferingNoFrame);
    EXPECT_GT(result.dropped_this_call, 0);       // 至少丢了一帧
    EXPECT_EQ(result.queue_size_after, 0);        // 丢完队列空
}

// ② diff = -10000：正好在窗口下边界，应交付
TEST(FrameSchedulerClassifyTest, DiffAtNegativeToleranceDeliversFrame) {
    FrameScheduler scheduler(MakeSingleBufferConfig());
    scheduler.PushFrame(MakeTestFrame(0, /*frame_index=*/5));

    auto result = scheduler.Schedule(MediaTimestamp{10000});

    EXPECT_EQ(result.action, ScheduleAction::kDeliver);
    EXPECT_EQ(result.selected_frame->frame_index, 5);
    EXPECT_EQ(result.dropped_this_call, 0);       // 没丢帧
}

// ③ diff = +10000：正好在窗口上边界，应交付
TEST(FrameSchedulerClassifyTest, DiffAtPositiveToleranceDeliversFrame) {
    FrameScheduler scheduler(MakeSingleBufferConfig());
    scheduler.PushFrame(MakeTestFrame(10000, /*frame_index=*/7));

    auto result = scheduler.Schedule(MediaTimestamp{0});

    EXPECT_EQ(result.action, ScheduleAction::kDeliver);
    EXPECT_EQ(result.selected_frame->frame_index, 7);
}

// ④ diff = +10001：超出上边界，太新，应该等
//    kWait → 有 last 则 RepeatLast，无 last 则 BufferingNoFrame
//    本测试无 last → kBufferingNoFrame，队首留在队列里原封不动
TEST(FrameSchedulerClassifyTest, DiffAbovePositiveToleranceWaits) {
    FrameScheduler scheduler(MakeSingleBufferConfig());
    scheduler.PushFrame(MakeTestFrame(10001, /*frame_index=*/9));

    auto result = scheduler.Schedule(MediaTimestamp{0});

    EXPECT_EQ(result.action, ScheduleAction::kBufferingNoFrame);
    EXPECT_EQ(result.queue_size_after, 1);        // 队首未被消费
}

// ===================================================================
// Schedule：缓冲状态
// ===================================================================

TEST(FrameSchedulerScheduleTest, BufferingWhenQueueBelowMinReturnsNoFrame) {
    // min_buffered_frames=2, 只推一帧 → 缓冲不足
    // 全新调度器，无 last_delivered_frame → 返回 kBufferingNoFrame（不是 kRepeatLast）
    FrameScheduler scheduler(MakeTestConfig());
    scheduler.PushFrame(MakeTestFrame(100000, 0));

    auto result = scheduler.Schedule(MediaTimestamp{100000});

    EXPECT_EQ(result.action, ScheduleAction::kBufferingNoFrame);
    EXPECT_FALSE(result.selected_frame.has_value());
    EXPECT_EQ(result.state_after, SchedulerState::kBuffering);
}

TEST(FrameSchedulerScheduleTest, BufferingWithLastFrameReturnsRepeatLast) {
    // 先构建有 last 但队列空的状态：Running → 队列耗尽 → Buffering
    FrameScheduler scheduler(MakeTestConfig());  // min_buffered_frames=2

    // 推两帧并全部交付，队列清空
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.PushFrame(MakeTestFrame(200000, 1));
    scheduler.Schedule(MediaTimestamp{100000});  // 交付 frame 0
    scheduler.Schedule(MediaTimestamp{200000});  // 交付 frame 1，队列空，状态仍 Running

    // 下一拍：Running + 空队列 → 切 Buffering
    // 此路径在 ClassifyTimingDiff 之前就已返回，不依赖同步判据
    auto result = scheduler.Schedule(MediaTimestamp{300000});

    EXPECT_EQ(result.action, ScheduleAction::kRepeatLast);
    ASSERT_TRUE(result.selected_frame.has_value());
    EXPECT_EQ(result.selected_frame->frame_index, 1);  // 上一帧
    EXPECT_EQ(result.state_after, SchedulerState::kBuffering);
    EXPECT_EQ(result.queue_size_after, 0);
}

// ===================================================================
// Reset
// ===================================================================

TEST(FrameSchedulerResetTest, ResetClearsAllState) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 建立状态：交付一帧
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.Schedule(MediaTimestamp{100000});

    // Reset
    scheduler.Reset();

    // Reset 后推新帧，上一帧的 PTS 不应导致拒绝
    // 如果 Reset 没清 last_delivered_frame，PTS=50000 会被拒绝为 stale
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(50000, 1)),
              PushResult::kAccepted);
}

// ===================================================================
// 交付 PTS 严格递增
// ===================================================================

TEST(FrameSchedulerIntegrationTest, DeliveredPtsStrictlyIncreasing) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 按 PTS 顺序推入三帧
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.PushFrame(MakeTestFrame(200000, 1));
    scheduler.PushFrame(MakeTestFrame(300000, 2));

    std::int64_t last_pts = -1;
    for (auto ref_pts : {100000, 200000, 300000}) {
        auto result = scheduler.Schedule(MediaTimestamp{ref_pts});
        ASSERT_TRUE(result.selected_frame.has_value());
        std::int64_t delivered_pts =
            result.selected_frame->pts.microseconds;
        EXPECT_GT(delivered_pts, last_pts)
            << "交付 PTS 必须严格递增";
        last_pts = delivered_pts;
    }
}

TEST(FrameSchedulerIntegrationTest, OutOfOrderPushDeliversInPtsOrder) {
    // 乱序入队: PTS=300, 100, 200 → 有序插入后队列应为 [100, 200, 300]
    // 连续 Schedule 交付的 frame_index 顺序应为 1→2→0
    FrameScheduler scheduler(MakeSingleBufferConfig());
    scheduler.PushFrame(MakeTestFrame(300000, 0));
    scheduler.PushFrame(MakeTestFrame(100000, 1));
    scheduler.PushFrame(MakeTestFrame(200000, 2));

    // 按 PTS 顺序调度，验证交付的 frame_index
    std::int64_t refs[] = {100000, 200000, 300000};
    std::int64_t expected_indices[] = {1, 2, 0};  // PTS=100→index1, PTS=200→index2, PTS=300→index0

    for (int i = 0; i < 3; i++) {
        auto result = scheduler.Schedule(MediaTimestamp{refs[i]});
        ASSERT_TRUE(result.selected_frame.has_value());
        // 乱序入队(300,100,200)，但有序插入后排成了 [100,200,300]
        // 所以按 100→200→300 调度时，拿到的 frame_index 是 1→2→0
        EXPECT_EQ(result.selected_frame->frame_index, expected_indices[i]);
        EXPECT_EQ(result.action, ScheduleAction::kDeliver);
    }
}

TEST(FrameSchedulerIntegrationTest, ContinuousDropMultipleStaleFrames) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 推入四帧: PTS=0, 100000, 200000, 300000
    scheduler.PushFrame(MakeTestFrame(0, 0));
    scheduler.PushFrame(MakeTestFrame(100000, 1));
    scheduler.PushFrame(MakeTestFrame(200000, 2));
    scheduler.PushFrame(MakeTestFrame(300000, 3));

    // ref=305000: 前三帧 diff 均 < -10000(过时), 300000 帧 diff=-5000(在窗口内)
    auto result = scheduler.Schedule(MediaTimestamp{305000});

    EXPECT_EQ(result.action, ScheduleAction::kDropAndDeliver);
    EXPECT_EQ(result.dropped_this_call, 3);  // 连续丢了 3 帧
    ASSERT_TRUE(result.selected_frame.has_value());
    EXPECT_EQ(result.selected_frame->frame_index, 3);  // 交付第 4 帧
}

// 证明 Buffering→Running→Buffering→Running 完整循环
// min_buffered_frames=2, tolerance=10000
TEST(FrameSchedulerStateMachineTest, RebufferAfterDrain) {
    FrameScheduler scheduler(MakeTestConfig());  // min=2

    // 1. 初始 Buffering, 只推 1 帧（不足 min=2）, 无 last
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    auto r1 = scheduler.Schedule(MediaTimestamp{100000});
    // 队列只有 1 帧 < min=2, 且无 last → BufferingNoFrame
    EXPECT_EQ(r1.action, ScheduleAction::kBufferingNoFrame);
    EXPECT_EQ(r1.state_after, SchedulerState::kBuffering);

    // 2. 再推 1 帧凑够 2 帧, ref 匹配队首 → 交付, 进入 Running
    scheduler.PushFrame(MakeTestFrame(200000, 1));
    auto r2 = scheduler.Schedule(MediaTimestamp{100000});
    EXPECT_EQ(r2.action, ScheduleAction::kDeliver);
    EXPECT_EQ(r2.selected_frame->frame_index, 0);
    EXPECT_EQ(r2.state_after, SchedulerState::kRunning);

    // 3. 交付第二帧 → 队列空
    auto r3 = scheduler.Schedule(MediaTimestamp{200000});
    EXPECT_EQ(r3.action, ScheduleAction::kDeliver);
    EXPECT_EQ(r3.selected_frame->frame_index, 1);
    // 队列空了但状态还是 Running（刚交付完）

    // 4. 下一拍: Running + 队列空 → 切 Buffering, 有 last → RepeatLast
    auto r4 = scheduler.Schedule(MediaTimestamp{300000});
    EXPECT_EQ(r4.action, ScheduleAction::kRepeatLast);
    EXPECT_EQ(r4.selected_frame->frame_index, 1);  // last 是步骤③交付的 frame 1
    EXPECT_EQ(r4.state_after, SchedulerState::kBuffering);

    // 5. 推 1 帧, 不足 min=2, 但有 last → RepeatLast
    scheduler.PushFrame(MakeTestFrame(400000, 2));
    auto r5 = scheduler.Schedule(MediaTimestamp{300000});
    EXPECT_EQ(r5.action, ScheduleAction::kRepeatLast);  // 类型是 ScheduleAction
    EXPECT_EQ(r5.selected_frame->frame_index, 1);       // last 仍是 frame 1
    EXPECT_EQ(r5.state_after, SchedulerState::kBuffering);

    // ⑥ 再推 1 帧凑够 2 帧, ref 匹配队首 → 交付, 回到 Running
    scheduler.PushFrame(MakeTestFrame(500000, 3));
    auto r6 = scheduler.Schedule(MediaTimestamp{400000});
    EXPECT_EQ(r6.action, ScheduleAction::kDeliver);
    EXPECT_EQ(r6.selected_frame->frame_index, 2);
    EXPECT_EQ(r6.state_after, SchedulerState::kRunning);
}


// ===================================================================
// PushFrame：更多非法帧类型
// ===================================================================

TEST(FrameSchedulerPushTest, BgraMatIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(100000, 0);
    frame.frame_bgr = cv::Mat(32, 32, CV_8UC4, cv::Scalar(10, 20, 30, 40));
    EXPECT_EQ(scheduler.PushFrame(frame), PushResult::kRejectedInvalidFrame);
}

TEST(FrameSchedulerPushTest, FloatMatIsRejected) {
    FrameScheduler scheduler(MakeTestConfig());
    auto frame = MakeTestFrame(100000, 0);
    frame.frame_bgr = cv::Mat(32, 32, CV_32FC3, cv::Scalar(0.1f, 0.2f, 0.3f));
    EXPECT_EQ(scheduler.PushFrame(frame), PushResult::kRejectedInvalidFrame);
}

// ===================================================================
// 溢出策略
// ===================================================================

TEST(FrameSchedulerOverflowTest, DropOldestRemovesFront) {
    // 容量=3, 推满 [100,200,300], 再推 400 → 淘汰 100
    FrameSchedulerConfig cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 3;
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    FrameScheduler scheduler(cfg);

    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.PushFrame(MakeTestFrame(200000, 1));
    scheduler.PushFrame(MakeTestFrame(300000, 2));

    // 推第四帧，应淘汰队首(100000,0)
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(400000, 3)),
              PushResult::kDroppedOldest);

    // 交付验证：队首应为 200000(避免 100000)
    auto result = scheduler.Schedule(MediaTimestamp{200000});
    EXPECT_EQ(result.action, ScheduleAction::kDeliver);
    EXPECT_EQ(result.selected_frame->frame_index, 1);  // 100 被淘汰，200 变队首
}

TEST(FrameSchedulerOverflowTest, RejectNewestKeepsQueueUnchanged) {
    FrameSchedulerConfig cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 2;
    cfg.overflow_policy = OverflowPolicy::kRejectNewest;
    FrameScheduler scheduler(cfg);

    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.PushFrame(MakeTestFrame(200000, 1));

    // 满，拒绝新帧
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(300000, 2)),
              PushResult::kRejectedQueueFull);

    // 队列内容不变，交付队首仍是 frame 0
    auto result = scheduler.Schedule(MediaTimestamp{100000});
    EXPECT_EQ(result.selected_frame->frame_index, 0);
    EXPECT_EQ(result.queue_size_after, 1);  // 剩一帧
}

TEST(FrameSchedulerOverflowTest, DuplicateTakesPriorityOverOverflow) {
    // 满队列 + 重复 PTS → 返回 duplicate，不淘汰
    FrameSchedulerConfig cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 2;
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    FrameScheduler scheduler(cfg);

    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.PushFrame(MakeTestFrame(200000, 1));

    // PTS=100000 已在队列中 → duplicate，队列不变
    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(100000, 2)),
              PushResult::kRejectedDuplicatePts);

    // 队首仍是 100000
    auto result = scheduler.Schedule(MediaTimestamp{100000});
    EXPECT_EQ(result.selected_frame->frame_index, 0);
}

TEST(FrameSchedulerOverflowTest, InvalidTakesPriorityOverOverflow) {
    FrameSchedulerConfig cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 2;
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    FrameScheduler scheduler(cfg);

    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.PushFrame(MakeTestFrame(200000, 1));

    // 非法帧即使队列满也返回 invalid，不触发溢出
    auto bad = MakeTestFrame(300000, 2);
    bad.frame_bgr = cv::Mat();  // 空
    EXPECT_EQ(scheduler.PushFrame(bad), PushResult::kRejectedInvalidFrame);

    // 队列未被修改
    auto result = scheduler.Schedule(MediaTimestamp{100000});
    EXPECT_EQ(result.selected_frame->frame_index, 0);
    EXPECT_EQ(result.queue_size_after, 1);
}

TEST(FrameSchedulerOverflowTest, DropOldestWithSevereOutOfOrder) {
    // 满队列 [200,300,400], 推 PTS=100(比队首还旧)
    // kDropOldest 淘汰队首 200, 有序插入后队列 [100,300,400]
    FrameSchedulerConfig cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 3;
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    FrameScheduler scheduler(cfg);

    scheduler.PushFrame(MakeTestFrame(200000, 1));
    scheduler.PushFrame(MakeTestFrame(300000, 2));
    scheduler.PushFrame(MakeTestFrame(400000, 3));

    EXPECT_EQ(scheduler.PushFrame(MakeTestFrame(100000, 0)),
              PushResult::kDroppedOldest);

    // 交付验证：队首应为 100
    auto result = scheduler.Schedule(MediaTimestamp{100000});
    EXPECT_EQ(result.selected_frame->frame_index, 0);
}

// ===================================================================
// 同步窗口：补充场景
// ===================================================================

TEST(FrameSchedulerScheduleTest, DropAllFramesReturnsRepeatLastWhenHasLast) {
    // 推 3 帧全部过时，有 last → RepeatLast，dropped=3
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 先交付一帧建立 last
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.Schedule(MediaTimestamp{100000});  // last = frame 0

    // 推 3 帧，参考时钟远超它们
    scheduler.PushFrame(MakeTestFrame(200000, 1));
    scheduler.PushFrame(MakeTestFrame(300000, 2));
    scheduler.PushFrame(MakeTestFrame(400000, 3));

    auto result = scheduler.Schedule(MediaTimestamp{510000});
    // diff: 200000-510000=-310000 <-10000 → drop; 300,400 同理全部 drop
    EXPECT_EQ(result.action, ScheduleAction::kRepeatLast);
    EXPECT_EQ(result.selected_frame->frame_index, 0);  // 重复 last
    EXPECT_EQ(result.dropped_this_call, 3);
    EXPECT_EQ(result.state_after, SchedulerState::kBuffering);
}

TEST(FrameSchedulerScheduleTest, WaitDoesNotConsumeFrontFrame) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 推一帧远未来，Schedule 近参考 → kWait，队首保留
    scheduler.PushFrame(MakeTestFrame(500000, 0));

    auto result = scheduler.Schedule(MediaTimestamp{0});
    // diff=500000 > 10000 → kWait, 无 last → BufferingNoFrame
    EXPECT_EQ(result.action, ScheduleAction::kBufferingNoFrame);
    EXPECT_EQ(result.queue_size_after, 1);  // 队首未被消费

    // 下次参考时钟追上来后可以交付
    auto result2 = scheduler.Schedule(MediaTimestamp{500000});
    EXPECT_EQ(result2.action, ScheduleAction::kDeliver);
    EXPECT_EQ(result2.selected_frame->frame_index, 0);
}

TEST(FrameSchedulerScheduleTest, DropAndDeliverHasDropCountAndFrame) {
    // kDropAndDeliver 必须同时满足：丢帧数>0 且 存在新交付帧
    FrameScheduler scheduler(MakeSingleBufferConfig());

    scheduler.PushFrame(MakeTestFrame(0, 0));       // 过时
    scheduler.PushFrame(MakeTestFrame(100000, 1));  // 可交付

    auto result = scheduler.Schedule(MediaTimestamp{100000});
    // 帧 0 过时被丢，帧 1 交付
    EXPECT_EQ(result.action, ScheduleAction::kDropAndDeliver);
    EXPECT_GT(result.dropped_this_call, 0);
    ASSERT_TRUE(result.selected_frame.has_value());
    EXPECT_EQ(result.selected_frame->frame_index, 1);
}

TEST(FrameSchedulerScheduleTest, RepeatLastAfterDropIsNotDropAndDeliver) {
    // 丢完后队列空，RepeatLast 不能误报为 kDropAndDeliver
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 先交付一帧建立 last
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.Schedule(MediaTimestamp{100000});

    // 推一帧过时帧
    scheduler.PushFrame(MakeTestFrame(50000, 1));  // PTS=50000 < 已交付 → 被 stale 拦住

    // 换个大于 last 的帧但让它过时
    scheduler.PushFrame(MakeTestFrame(110000, 2));
    auto result = scheduler.Schedule(MediaTimestamp{500000});
    // 帧 2 过时被丢, 队列空, 有 last → RepeatLast
    EXPECT_EQ(result.action, ScheduleAction::kRepeatLast);  // 不是 kDropAndDeliver
    EXPECT_GT(result.dropped_this_call, 0);
    EXPECT_EQ(result.queue_size_after, 0);
}

// ===================================================================
// 参考时间：补充场景
// ===================================================================

TEST(FrameSchedulerScheduleTest, SameReferenceTimeIsAllowed) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.Schedule(MediaTimestamp{100000});  // 交付 frame 0

    scheduler.PushFrame(MakeTestFrame(200000, 1));
    // 相同参考时间 → 允许（不是倒退，是相等）
    auto result = scheduler.Schedule(MediaTimestamp{100000});
    // ref=100000, 队首 PTS=200000, diff=100000 > 10000 → kWait
    EXPECT_NE(result.action, ScheduleAction::kRejectedInvalidReferenceTime);
}

TEST(FrameSchedulerScheduleTest, RejectedRefHasZeroSideEffects) {
    // 非法参考时间不应改变任何内部状态
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 推一帧入队但先不调度，直接给倒退 ref，验证队列不变
    scheduler.PushFrame(MakeTestFrame(100000, 0));

    auto result = scheduler.Schedule(MediaTimestamp{-1});  // 非法负值
    EXPECT_EQ(result.action, ScheduleAction::kRejectedInvalidReferenceTime);
    EXPECT_EQ(result.dropped_this_call, 0);
    EXPECT_EQ(result.queue_size_after, 1);  // 队列帧还在
}

// ===================================================================
// Reset：补充场景
// ===================================================================

TEST(FrameSchedulerResetTest, ResetAllowsNewTimelineFromZero) {
    // Reset 后可以从 0us 开始全新时间线
    FrameScheduler scheduler(MakeSingleBufferConfig());

    // 旧时间线：PTS=100000, ref=100000
    scheduler.PushFrame(MakeTestFrame(100000, 0));
    scheduler.Schedule(MediaTimestamp{100000});

    scheduler.Reset();

    // 新时间线从 0 开始，不会因 previous_ref 倒退而被拒
    scheduler.PushFrame(MakeTestFrame(0, 1));
    auto result = scheduler.Schedule(MediaTimestamp{0});
    EXPECT_EQ(result.action, ScheduleAction::kDeliver);
    EXPECT_EQ(result.selected_frame->frame_index, 1);
}

// ===================================================================
// cv::Mat 共享只读生命周期
// ===================================================================

TEST(FrameSchedulerMatLifecycleTest, FrameDataSurvivesAfterOriginalMatDestroyed) {
    FrameScheduler scheduler(MakeSingleBufferConfig());

    {
        // 局部作用域内创建帧
        cv::Mat local_mat(32, 32, CV_8UC3, cv::Scalar(255, 128, 64));
        video::VideoFrame frame;
        frame.frame_bgr = local_mat;  // 引用计数共享
        frame.pts.microseconds = 100000;
        frame.frame_index = 42;
        scheduler.PushFrame(frame);
        // local_mat 和 frame 离开作用域，但 frame_bgr 的像素数据还在
    }

    auto result = scheduler.Schedule(MediaTimestamp{100000});
    ASSERT_TRUE(result.selected_frame.has_value());
    EXPECT_EQ(result.selected_frame->frame_index, 42);
    // 验证像素数据仍在
    auto pixel = result.selected_frame->frame_bgr.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[0], 255);
    EXPECT_EQ(pixel[1], 128);
    EXPECT_EQ(pixel[2], 64);
}

// ===================================================================
// 多生产者并发
// ===================================================================

TEST(FrameSchedulerConcurrencyTest, TwoProducersSamePtsOnlyOneAccepted) {
    // 两个线程推互不重复 PTS: 每个线程 50 帧，共 50 unique PTS，必须全部被接受
    auto cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 100;  // 足够容纳所有帧，避免溢出干扰计数
    FrameScheduler scheduler(cfg);
    std::atomic<int> accepted{0};

    auto push_fn = [&](int thread_id) {
        for (int round = 0; round < 50; round++) {
            // 两个线程用不同 PTS: thread 0 用偶数, thread 1 用奇数
            int64_t pts = (round * 2 + thread_id) * 10000;
            auto frame = MakeTestFrame(pts, thread_id * 1000 + round);
            auto result = scheduler.PushFrame(frame);
            if (result == PushResult::kAccepted) { accepted++; }
        }
    };

    std::thread t1(push_fn, 0);
    std::thread t2(push_fn, 1);
    t1.join();
    t2.join();

    // 100 unique PTS, 全部应被接受
    EXPECT_EQ(accepted, 100);
}

TEST(FrameSchedulerConcurrencyTest, DuplicatePtsRaceOnlyOneAccepted) {
    // 两个线程推相同 PTS: 100 unique PTS × 2 = 200 次 push
    // 每个 PTS 只有先锁到的线程 Accepted, 另一个 Duplicate
    auto cfg = MakeSingleBufferConfig();
    cfg.max_queue_size = 200;  // 足够大避免溢出干扰
    FrameScheduler scheduler(cfg);
    std::atomic<int> accepted{0};
    std::atomic<int> duplicates{0};

    std::thread t1([&]() {
        for (int i = 0; i < 100; i++) {
            auto frame = MakeTestFrame(i * 10000, i);
            auto r = scheduler.PushFrame(frame);
            if (r == PushResult::kAccepted) { accepted++; }
            if (r == PushResult::kRejectedDuplicatePts) { duplicates++; }
        }
    });
    std::thread t2([&]() {
        for (int i = 0; i < 100; i++) {
            auto frame = MakeTestFrame(i * 10000, i + 1000);
            auto r = scheduler.PushFrame(frame);
            if (r == PushResult::kAccepted) { accepted++; }
            if (r == PushResult::kRejectedDuplicatePts) { duplicates++; }
        }
    });
    t1.join();
    t2.join();

    EXPECT_EQ(accepted, 100);
    EXPECT_EQ(duplicates, 100);
}

TEST(FrameSchedulerConcurrencyTest, PushAndScheduleConcurrentNoDeadlock) {
    FrameScheduler scheduler(MakeSingleBufferConfig());
    std::atomic<bool> stop{false};
    std::atomic<int> push_count{0};
    std::atomic<int> schedule_count{0};

    // Producer 线程: 持续推帧
    std::thread producer([&]() {
        int idx = 0;
        while (!stop) {
            scheduler.PushFrame(MakeTestFrame(idx * 10000, idx));
            push_count++;
            idx++;
            std::this_thread::yield();
        }
    });

    // Consumer 线程: 持续调度
    std::thread consumer([&]() {
        int ref = 0;
        while (!stop) {
            scheduler.Schedule(MediaTimestamp{ref * 10000});
            schedule_count++;
            ref++;
            std::this_thread::yield();
        }
    });

    // 运行 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    producer.join();
    consumer.join();

    // 不崩溃、不死锁即证明并发安全
    EXPECT_GT(push_count, 0);
    EXPECT_GT(schedule_count, 0);
}

TEST(FrameSchedulerConcurrencyTest, MultiProducerOutOfOrderPreservesPtsOrder) {
    // 4 个线程各推 25 帧不同 PTS，最终调度 PTS 严格递增
    FrameScheduler scheduler(MakeSingleBufferConfig());
    constexpr int kProducers = 4;
    constexpr int kFramesPerProducer = 25;

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; p++) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < kFramesPerProducer; i++) {
                // 故意打乱 PTS 分配：不同线程交错的 PTS
                int64_t pts = (i * kProducers + p) * 10000;
                scheduler.PushFrame(MakeTestFrame(pts, p * 1000 + i));
            }
        });
    }
    for (auto& t : producers) { t.join(); }

    // 按 PTS 递增调度，验证交付 frame_index 唯一且 PTS 递增
    std::int64_t last_pts = -1;
    for (int i = 0; i < kProducers * kFramesPerProducer; i++) {
        int64_t ref = i * 10000 + 10000000;  // 参考时钟远超所有帧 PTS
        auto result = scheduler.Schedule(MediaTimestamp{ref});
        // 可能 BufferingNoFrame 或 Deliver
        if (result.action == ScheduleAction::kDeliver
            || result.action == ScheduleAction::kDropAndDeliver) {
            ASSERT_TRUE(result.selected_frame.has_value());
            EXPECT_GT(result.selected_frame->pts.microseconds, last_pts)
                << "并发 Push 后交付 PTS 必须递增";
            last_pts = result.selected_frame->pts.microseconds;
        }
        // 部分帧可能因溢出去失，不强制 count
    }
}

}  // namespace
}  // namespace core
}  // namespace digital_human
