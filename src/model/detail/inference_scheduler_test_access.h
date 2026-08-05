#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "model/inference_scheduler.h"

namespace digital_human {
namespace model {
namespace detail {

/// @brief 只供单元测试使用的内部访问器，不安装为 SDK 公开头文件
///
/// 生产代码仍通过 InferenceScheduler() 使用真实 ModelInference。测试可以
/// 注入 fake executor，以 gate 精确控制任务完成顺序，而不依赖 ncnn 运行快慢。
class InferenceSchedulerTestAccess {
public:
    using Executor = std::function<SingleInferenceResult(
        const ModelRuntimeSnapshot&,
        const NcnnWav2LipInput&,
        const InferenceOptions&)>;
    using EnqueueObserver = std::function<void(std::size_t, bool)>;

    static std::unique_ptr<InferenceScheduler> Create(
        Executor executor,
        EnqueueObserver enqueue_observer = EnqueueObserver{}) {
        return std::unique_ptr<InferenceScheduler>(
            new InferenceScheduler(
                std::move(executor),
                std::move(enqueue_observer)));
    }
};

} // namespace detail
} // namespace model
} // namespace digital_human
