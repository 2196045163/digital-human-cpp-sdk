#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "core/timestamp_manager.h"
#include "video/video_frame.h"

namespace digital_human {
namespace core {

/// @brief 队列满时怎么办——两种策略。
enum class OverflowPolicy {
    /// 扔掉队列里最旧的那帧，给新帧腾位置。适合实时场景，优先保持低延迟。
    kDropOldest,
    /// 拒绝新帧，队列原封不动。适合不能丢帧的离线场景。
    kRejectNewest
};

/// @brief PushFrame() 的返回结果，六种，每种对应一个明确的拒绝原因。
///
/// 调用方根据返回值决定下一步：重试、丢弃、降级还是上报。
enum class PushResult {
    /// 帧成功进入队列。
    kAccepted,
    /// 队列已满，按"淘汰最旧帧"策略扔掉了队首，本帧成功入队。
    kDroppedOldest,
    /// 队列已满，按"拒绝新帧"策略，队列不变，本帧被拒。
    kRejectedQueueFull,
    /// 队列里已经有时间戳完全相同的帧，本帧被拒（保留先到的）。
    kRejectedDuplicatePts,
    /// 本帧的时间戳 <= 已经交付给显示线程的最后一帧，说明它来晚了/重复了，拒绝。
    kRejectedStalePts,
    /// 帧数据有问题：空图像、格式不对、时间戳为负、帧序号为负。
    kRejectedInvalidFrame
};

/// @brief 调度器内部状态。
enum class SchedulerState {
    /// 缓冲中：帧还不够，或者在 Running 时队列耗尽后回到这个状态重新攒帧。
    kBuffering,
    /// 运行中：已经至少成功交付过一帧，队列正常流转。
    kRunning
};

/// @brief Schedule() 告诉调用方"这一拍该干什么"。
///
/// 调用方必须 switch 这个枚举做分支，不能靠 selected_frame 是否为空猜动作。
/// 因为 kBufferingNoFrame 和 kRejectedInvalidReferenceTime 都不带帧，
/// 但一个表示"缓冲中没帧可给"，一个表示"你传的参考时间有问题"——处理方式完全不同。
enum class ScheduleAction {
    /// 队首帧正好在同步窗口里，直接拿去显示。这次没丢任何帧。
    kDeliver,
    /// 丢掉了 N 帧过时数据之后，队列里剩下的队首帧刚好在窗口里，拿去显示。
    kDropAndDeliver,
    /// 没有合适的新帧（队首太早、队列空、缓冲不足），重复显示上一帧凑合一下。
    kRepeatLast,
    /// 正在缓冲阶段，而且从来没有成功交付过任何帧，没东西可重复。
    kBufferingNoFrame,
    /// 参考时间非法（负值或倒退），本次调用被忽略，调度器内部状态没有任何变化。
    kRejectedInvalidReferenceTime
};

/// @brief 帧调度器的固定配置。
///
/// 所有字段必须由调用方显式填写，实现不预设任何通用值。
struct FrameSchedulerConfig {
    std::size_t max_queue_size = 0;
    std::size_t min_buffered_frames = 0;
    std::int64_t sync_tolerance_us = 0;
    OverflowPolicy overflow_policy = OverflowPolicy::kDropOldest;
};

/// @brief Schedule() 的完整返回信息。
///
/// 字段契约：
/// - action: 调用方必须 switch 的动作
/// - selected_frame: Deliver/Repeat 时有效；无帧时为空
/// - timing_diff_us: 交付时为交付帧差值；等待时为队首差值；无候选时为空
/// - dropped_this_call: 本次因过时丢弃的帧数
/// - queue_size_after: 操作完成后的队列长度
/// - state_after: 操作完成后的调度器状态
struct ScheduleResult {
    ScheduleAction action = ScheduleAction::kBufferingNoFrame;
    std::optional<video::VideoFrame> selected_frame;
    std::optional<std::int64_t> timing_diff_us;
    std::size_t dropped_this_call = 0;
    std::size_t queue_size_after = 0;
    SchedulerState state_after = SchedulerState::kBuffering;
};

/// @brief 音频参考时钟驱动的视频帧调度器。
///
/// 职责：
/// - 接收多个推理 worker 乱序产出的已融合帧，按 PTS 维护有序有界队列
/// - 根据调用方提供的参考 PTS，显式决定 Drop / Deliver / Repeat
/// - 管理 Buffering 启动缓冲和耗尽后重新缓冲
///
/// 不负责：产生音频时钟、sleep、渲染、日志或累计监控。
class FrameScheduler {
public:
    /// @brief 使用给定配置构造调度器。
    /// @throws std::invalid_argument 配置非法时抛异常
    explicit FrameScheduler(const FrameSchedulerConfig& config);
    ~FrameScheduler();

    // 不可拷贝、不可移动
    FrameScheduler(const FrameScheduler&) = delete;
    FrameScheduler& operator=(const FrameScheduler&) = delete;
    FrameScheduler(FrameScheduler&&) = delete;
    FrameScheduler& operator=(FrameScheduler&&) = delete;

    /// @brief 将一帧推入调度队列。
    ///
    /// 多线程安全，允许多个生产者并发调用。
    /// 无效帧、重复 PTS、已交付 PTS 会返回明确拒绝结果。
    /// 队列满时行为由 OverflowPolicy 决定。
    ///
    /// @return 精确的 PushResult，调用方据此决定重试、降级或上报
    PushResult PushFrame(const video::VideoFrame& frame);

    /// @brief 根据参考时钟决定本次应显示哪一帧。
    ///
    /// 只允许单个显示线程调用。
    /// 不阻塞、不 sleep——调用方按自己的 tick 周期重复调用。
    ///
    /// @return 完整 ScheduleResult，包含动作、帧、差值、丢帧数和状态
    ScheduleResult Schedule(const MediaTimestamp& reference_pts);

    /// @brief 清空队列和全部内部状态，回到初始 Buffering。
    ///
    /// 调用方必须先暂停所有旧生产者，否则旧帧可能在 Reset 后入队。
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace digital_human
