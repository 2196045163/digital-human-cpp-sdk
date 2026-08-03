#pragma once

#include <memory>

#include "pipeline/pipeline_types.h"

namespace digital_human {
namespace pipeline {

/// @brief Pipeline 输出接收器（Sink）接口。
///
/// 契约：
/// - 同一 session 内所有回调严格串行，绝不并发。
/// - 离线模式：OnFrame 和 OnTerminal 从 render worker 调用。
/// - 实时模式：OnFrame 和 OnTerminal 从输出调度线程调用。
/// - OnTerminal 恰好一次，并且是最后一个回调。
/// - 所有回调均在 Pipeline 内部锁之外执行。
/// - 回调期间允许 RequestStop / GetState / GetStats。
/// - 回调期间禁止 Start / Stop / Wait / 析构 Pipeline。
/// - OnFrame 抛异常：Pipeline 进入 Failed。
/// - OnTerminal 抛异常：记录 secondary diagnostic，不允许异常逃出线程入口。
class PipelineOutputSink {
public:
    virtual ~PipelineOutputSink() = default;

    /// @brief 接收一帧输出帧
    /// @param frame 包含 VideoFrame、交付类型、任务来源等完整信息
    virtual void OnFrame(const PipelineFrame& frame) = 0;

    /// @brief Pipeline 终止回调（恰好一次，最后一个回调）
    /// @param result Pipeline 终止结果，包含状态、统计、错误信息
    virtual void OnTerminal(const PipelineResult& result) = 0;
};

/// @brief Pipeline 持有的 sink 引用类型
using SinkHandle = std::shared_ptr<PipelineOutputSink>;

}  // namespace pipeline
}  // namespace digital_human
