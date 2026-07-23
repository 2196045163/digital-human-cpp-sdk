#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model/model_inference.h"

namespace digital_human {
namespace model {

namespace detail {
class InferenceSchedulerTestAccess;
}

/// @brief Scheduler 一个运行周期内固定不变的配置
struct SchedulerConfig {
    std::size_t worker_count = 0;              ///< 固定 worker 数，必须大于 0
    std::size_t queue_capacity = 0;            ///< 等待队列容量，不包含正在执行的任务，必须大于 0
    std::chrono::milliseconds queue_wait_timeout{0}; ///< 整个 batch 共用的入队等待预算
    InferenceOptions inference_options;        ///< 每个 job 调用 ModelInference 时使用的执行策略
    bool allow_thread_oversubscription = false;///< 是否显式允许 worker × ncnn threads 超过硬件线程数
};

/// @brief 一个 batch 的聚合诊断；逐项真实状态仍以 results[index] 为准
struct BatchSummary {
    std::size_t total_count = 0;               ///< 输入项总数，始终等于 results.size()
    std::size_t accepted_count = 0;            ///< 成功进入 worker pool 的任务数
    std::size_t success_count = 0;             ///< ModelInference 最终成功的任务数
    std::size_t failure_count = 0;             ///< 所有失败项数量
    std::size_t queue_timeout_count = 0;       ///< 因统一 enqueue deadline 超时而未接收的数量
    std::size_t stopping_rejected_count = 0;   ///< 因 Stop 而未接收的数量
    std::size_t task_exception_count = 0;      ///< task 异常被 future 捕获的数量
    std::size_t retry_recovered_count = 0;     ///< 单样本内部经 retry 最终恢复的数量
    double enqueue_wait_ms = 0.0;              ///< 所有输入项 Enqueue 调用耗时之和
    double batch_makespan_ms = 0.0;            ///< InferBatch 从进入到结果完整返回的总耗时
    double throughput_items_per_second = 0.0;  ///< total_count / batch makespan
};

/// @brief 同步 task batch 的完整结果，results 与 inputs 严格按 index 对齐
struct BatchInferenceResult {
    bool success = false;                                      ///< 仅全部单项成功时为 true
    InferenceStatus status = InferenceStatus::kUnknownError;   ///< kOk / kPartialFailure / kAllFailed 等
    std::string error_message;                                 ///< 顶层批次诊断
    std::vector<SingleInferenceResult> results;                ///< 大小与输入相同，不按完成顺序 push_back
    BatchSummary summary;                                      ///< 跨项聚合统计
    std::uint64_t model_generation = 0;                        ///< Start 时固定的模型代次
};

/// @brief Start/Stop 的生命周期控制结果
struct SchedulerControlResult {
    bool success = false;
    InferenceStatus status = InferenceStatus::kUnknownError;
    std::string error_message;
    double stop_wait_ms = 0.0;                ///< Stop 排空并 join worker 的等待耗时
};

/// @brief 有状态推理调度器：固定 snapshot、固定 worker、有界队列、同步 task batch
/// @note  运行中不提供 UpdateModel；切换模型或配置必须 Stop() 后重新 Start()。
class InferenceScheduler {
public:
    InferenceScheduler();
    ~InferenceScheduler();

    InferenceScheduler(const InferenceScheduler&) = delete;
    InferenceScheduler& operator=(const InferenceScheduler&) = delete;
    InferenceScheduler(InferenceScheduler&&) = delete;
    InferenceScheduler& operator=(InferenceScheduler&&) = delete;

    /// @brief 校验并固定 snapshot/config，然后创建 worker pool
    SchedulerControlResult Start(const ModelRuntimeSnapshot& snapshot,
                                 const SchedulerConfig& config);

    /// @brief 同步提交 task batch；任务可乱序完成，结果始终按输入 index 返回
    BatchInferenceResult InferBatch(
        const std::vector<NcnnWav2LipInput>& inputs);

    /// @brief 停止接收新任务、排空已接收任务并 join；可顺序重复调用
    SchedulerControlResult Stop();

private:
    using Executor = std::function<SingleInferenceResult(
        const ModelRuntimeSnapshot&,
        const NcnnWav2LipInput&,
        const InferenceOptions&)>;
    using EnqueueObserver = std::function<void(std::size_t, bool)>;

    /// @brief 仅供内部测试访问器注入可控 executor/入队观察器；不属于 SDK 公开构造方式
    explicit InferenceScheduler(
        Executor executor,
        EnqueueObserver enqueue_observer = EnqueueObserver{});
    friend class detail::InferenceSchedulerTestAccess;

    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace model
} // namespace digital_human
