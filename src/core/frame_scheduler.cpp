#include "core/frame_scheduler.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace digital_human {
namespace core {

// ============================================================================
// 内部类型：同步判据的三段分类结果
// ============================================================================

/// @brief 单帧与参考时钟比较后的动作方向。
///
/// 这是 ClassifyTimingDiff 的内部输出，不暴露给调用方。
/// 调用方看到的是 ScheduleAction（含上下文，比如是否丢过帧）。
enum class TimingDecision {
    kDrop,    ///< 帧太旧，应丢弃
    kDeliver, ///< 帧在同步窗口内，应交付
    kWait     ///< 帧太新，应保留并重复上一帧
};

// ============================================================================
// 同步判据
// ============================================================================

/// @brief 将 diff_us 分入三段：Drop / Deliver / Wait。
///
/// 契约：
/// - diff_us = video_pts - reference_pts
/// - 边界 -tolerance_us 和 +tolerance_us 都属于 Deliver 窗口
/// - tolerance_us >= 0 已被构造校验保证
///
/// @param diff_us  视频帧 PTS 减参考时钟 PTS（微秒）
/// @param tolerance_us  调用方配置的对称容忍窗口（微秒）
/// @return TimingDecision 精确分类
static TimingDecision ClassifyTimingDiff(std::int64_t diff_us,
                                         std::int64_t tolerance_us) {

    // 已知:
    //   diff_us = video_pts - reference_pts
    //   tolerance_us >= 0
    // 期望输出:
    //   diff < -tolerance  → kDrop   (视频帧已过时)
    //   diff > +tolerance  → kWait   (视频帧还太早)
    //   其他               → kDeliver (在同步窗口内)
    // 关键: 边界 -tolerance 和 +tolerance 都属于窗口内 ← 用 < 和 >，不用 <= 和 >=
    if (diff_us < -tolerance_us) {
        return TimingDecision::kDrop;
    }
    if (diff_us > tolerance_us) {
        return TimingDecision::kWait;
    }
    return TimingDecision::kDeliver;
}

// ============================================================================
// PImpl：所有共享状态封装在此
// ============================================================================

struct FrameScheduler::Impl {
    // —— 配置（构造后不可变） ——
    const FrameSchedulerConfig config;

    // —— 线程安全 ——
    mutable std::mutex mutex;

    // —— 有序帧队列 ——
    // 不变式：queue[i].pts.microseconds < queue[i+1].pts.microseconds
    std::deque<video::VideoFrame> queue;

    // —— 调度状态 ——
    SchedulerState state{SchedulerState::kBuffering};
    std::optional<video::VideoFrame> last_delivered_frame;
    std::optional<MediaTimestamp> previous_reference_pts;

    explicit Impl(const FrameSchedulerConfig& cfg) : config(cfg) {}
};

// ============================================================================
// 构造与析构
// ============================================================================

FrameScheduler::FrameScheduler(const FrameSchedulerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {

    // 校验固定配置，非法时抛异常——因为非法配置意味着对象无法正确工作
    if (config.max_queue_size == 0) {
        throw std::invalid_argument(
            "FrameSchedulerConfig: max_queue_size must be > 0");
    }
    if (config.min_buffered_frames == 0) {
        throw std::invalid_argument(
            "FrameSchedulerConfig: min_buffered_frames must be >= 1");
    }
    if (config.min_buffered_frames > config.max_queue_size) {
        throw std::invalid_argument(
            "FrameSchedulerConfig: min_buffered_frames must be <= max_queue_size");
    }
    if (config.sync_tolerance_us < 0) {
        throw std::invalid_argument(
            "FrameSchedulerConfig: sync_tolerance_us must be >= 0");
    }
    // OverflowPolicy 枚举值在编译期已受类型系统保护，但显式校验以防未来扩展
    if (config.overflow_policy != OverflowPolicy::kDropOldest
        && config.overflow_policy != OverflowPolicy::kRejectNewest) {
        throw std::invalid_argument(
            "FrameSchedulerConfig: unknown overflow_policy");
    }
}

FrameScheduler::~FrameScheduler() = default;

// ============================================================================
// PushFrame：多生产者并发入队
// ============================================================================

PushResult FrameScheduler::PushFrame(const video::VideoFrame& frame) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // ---- 1. 校验最小显示契约 ----
    // 顺序重要：先校验输入，不能因为队列满而掩盖无效帧
    if (frame.frame_bgr.empty()) {
        return PushResult::kRejectedInvalidFrame;
    }
    if (frame.frame_bgr.type() != CV_8UC3) {
        return PushResult::kRejectedInvalidFrame;
    }
    if (frame.pts.microseconds < 0) {
        return PushResult::kRejectedInvalidFrame;
    }
    if (frame.frame_index < 0) {
        return PushResult::kRejectedInvalidFrame;
    }

    // ---- 2. 拒绝已交付或更旧的 PTS ----
    // 保证交付 PTS 严格递增：新帧 PTS 必须 > 上一已交付帧 PTS
    if (impl_->last_delivered_frame.has_value()) {
        if (frame.pts.microseconds <= impl_->last_delivered_frame->pts.microseconds) {
            return PushResult::kRejectedStalePts;
        }
    }

    // ---- 3. 拒绝队列内重复 PTS ----
    // 保留先到帧，拒绝后到帧
    for (const auto& existing : impl_->queue) {
        if (existing.pts.microseconds == frame.pts.microseconds) {
            return PushResult::kRejectedDuplicatePts;
        }
    }

    // ---- 4. 队列满时执行溢出策略 ----
    bool dropped_for_overflow = false;
    if (impl_->queue.size() >= impl_->config.max_queue_size) {
        if (impl_->config.overflow_policy == OverflowPolicy::kRejectNewest) {
            return PushResult::kRejectedQueueFull;
        }
        // kDropOldest：移除队首，为新帧腾出空间
        impl_->queue.pop_front();
        dropped_for_overflow = true;
    }

    // ---- 5. 按 PTS 有序插入 ----
    // 线性查找第一个 PTS >= 新帧 PTS 的位置，保持队列严格升序
    // 容量小（抖动队列通常几个到十几个元素），线性开销可忽略
    auto insert_pos = impl_->queue.end();
    for (auto it = impl_->queue.begin(); it != impl_->queue.end(); ++it) {
        if (it->pts.microseconds >= frame.pts.microseconds) {
            insert_pos = it;
            break;
        }
    }
    impl_->queue.insert(insert_pos, frame);


    // ---- 6. 返回结果 ----
    if (dropped_for_overflow) {
        return PushResult::kDroppedOldest;
    }
    return PushResult::kAccepted;
}

// ============================================================================
// Schedule：单消费者同步决策
// ============================================================================

ScheduleResult FrameScheduler::Schedule(const MediaTimestamp& reference_pts) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // ---- 1. 校验参考时间 ----
    if (reference_pts.microseconds < 0) {
        return {
            ScheduleAction::kRejectedInvalidReferenceTime,
            std::nullopt, std::nullopt,
            0,
            impl_->queue.size(),
            impl_->state
        };
    }
    if (impl_->previous_reference_pts.has_value()
        && reference_pts.microseconds
               < impl_->previous_reference_pts->microseconds) {
        return {
            ScheduleAction::kRejectedInvalidReferenceTime,
            std::nullopt, std::nullopt,
            0,
            impl_->queue.size(),
            impl_->state
        };
    }
    impl_->previous_reference_pts = reference_pts;

    // ---- 2. 缓冲中且帧数不足 ----
    if (impl_->state == SchedulerState::kBuffering
        && impl_->queue.size() < impl_->config.min_buffered_frames) {
        if (impl_->last_delivered_frame.has_value()) {
            return {
                ScheduleAction::kRepeatLast,
                impl_->last_delivered_frame,
                std::nullopt,
                0,
                impl_->queue.size(),
                SchedulerState::kBuffering
            };
        }
        return {
            ScheduleAction::kBufferingNoFrame,
            std::nullopt, std::nullopt,
            0,
            impl_->queue.size(),
            SchedulerState::kBuffering
        };
    }

    // ---- 3. Running 状态下队列耗尽 → 切回 Buffering ----
    if (impl_->state == SchedulerState::kRunning
        && impl_->queue.empty()) {
        impl_->state = SchedulerState::kBuffering;
        if (impl_->last_delivered_frame.has_value()) {
            return {
                ScheduleAction::kRepeatLast,
                impl_->last_delivered_frame,
                std::nullopt,
                0,
                0,
                SchedulerState::kBuffering
            };
        }
        return {
            ScheduleAction::kBufferingNoFrame,
            std::nullopt, std::nullopt,
            0,
            0,
            SchedulerState::kBuffering
        };
    }

    // ---- 4. 连续丢弃过时帧 ----
    // 队首 PTS 落后参考时钟超过容忍窗口 → 已过时，丢弃并检查下一帧
    std::size_t dropped_this_call = 0;
    while (!impl_->queue.empty()
           && (impl_->queue.front().pts.microseconds - reference_pts.microseconds)
              < -impl_->config.sync_tolerance_us) {
        impl_->queue.pop_front();
        dropped_this_call++;
    }

    // ---- 5. 丢完后队列为空 ----
    if (impl_->queue.empty()) {
        impl_->state = SchedulerState::kBuffering;
        if (impl_->last_delivered_frame.has_value()) {
            return {
                ScheduleAction::kRepeatLast,
                impl_->last_delivered_frame,
                std::nullopt,
                dropped_this_call,
                0,
                SchedulerState::kBuffering
            };
        }
        return {
            ScheduleAction::kBufferingNoFrame,
            std::nullopt, std::nullopt,
            dropped_this_call,
            0,
            SchedulerState::kBuffering
        };
    }

    // ---- 6. 对队首帧做同步分类 ----
    const auto& front = impl_->queue.front();
    std::int64_t diff_us =
        front.pts.microseconds - reference_pts.microseconds;

    TimingDecision decision =
        ClassifyTimingDiff(diff_us, impl_->config.sync_tolerance_us);

    // ---- 7. 根据分类执行动作 ----
    switch (decision) {
    case TimingDecision::kDrop: {
        // 单帧丢弃（连续丢弃由第 4 步的 while 循环处理）
        impl_->queue.pop_front();
        dropped_this_call++;

        if (impl_->queue.empty()) {
            impl_->state = SchedulerState::kBuffering;
            if (impl_->last_delivered_frame.has_value()) {
                return {
                    ScheduleAction::kRepeatLast,
                    impl_->last_delivered_frame,
                    std::nullopt,
                    dropped_this_call,
                    0,
                    SchedulerState::kBuffering
                };
            }
            return {
                ScheduleAction::kBufferingNoFrame,
                std::nullopt, std::nullopt,
                dropped_this_call,
                0,
                SchedulerState::kBuffering
            };
        }
        // 丢完一帧后队列还有帧，但当前实现不递归检查
        // 连续丢弃由while 循环补全
        if (impl_->last_delivered_frame.has_value()) {
            return {
                ScheduleAction::kRepeatLast,
                impl_->last_delivered_frame,
                std::nullopt,
                dropped_this_call,
                impl_->queue.size(),
                impl_->state
            };
        }
        return {
            ScheduleAction::kBufferingNoFrame,
            std::nullopt, std::nullopt,
            dropped_this_call,
            impl_->queue.size(),
            impl_->state
        };
    }

    case TimingDecision::kWait: {
        // 队首太新，不消费，保留在队列中
        if (impl_->last_delivered_frame.has_value()) {
            return {
                ScheduleAction::kRepeatLast,
                impl_->last_delivered_frame,
                diff_us,
                dropped_this_call,
                impl_->queue.size(),
                impl_->state
            };
        }
        return {
            ScheduleAction::kBufferingNoFrame,
            std::nullopt,
            diff_us,
            dropped_this_call,
            impl_->queue.size(),
            impl_->state
        };
    }

    case TimingDecision::kDeliver: {
        // 交付队首帧
        video::VideoFrame delivered = impl_->queue.front();
        impl_->queue.pop_front();

        impl_->last_delivered_frame = delivered;
        impl_->state = SchedulerState::kRunning;

        ScheduleAction action = (dropped_this_call > 0)
            ? ScheduleAction::kDropAndDeliver
            : ScheduleAction::kDeliver;

        return {
            action,
            delivered,
            diff_us,
            dropped_this_call,
            impl_->queue.size(),
            SchedulerState::kRunning
        };
    }
    } // switch

    // 不应到达此处
    return {
        ScheduleAction::kBufferingNoFrame,
        std::nullopt, std::nullopt,
        dropped_this_call,
        impl_->queue.size(),
        impl_->state
    };
}

// ============================================================================
// Reset：清空全部状态
// ============================================================================

void FrameScheduler::Reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    impl_->queue.clear();
    impl_->last_delivered_frame.reset();
    impl_->previous_reference_pts.reset();
    impl_->state = SchedulerState::kBuffering;
}

}  // namespace core
}  // namespace digital_human
