#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <exception>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "model/inference_scheduler.h"
#include "detail/bounded_worker_pool.h"

namespace digital_human {
namespace model {
namespace {

/// @brief Scheduler 内部生命周期状态，不暴露给 SDK 调用方
enum class SchedulerState {
    kStopped,  ///< 尚未启动或已经完成 Stop；允许再次 Start
    kRunning,  ///< snapshot/config/pool 已就绪，可以接收 batch
    kStopping  ///< 已拒绝新任务，正在排空已接收任务并等待 worker 退出
};

} // namespace

/// @brief InferenceScheduler 跨 Start/InferBatch/Stop 持有的共享状态
struct InferenceScheduler::Impl {
    Impl(Executor executor_in, EnqueueObserver enqueue_observer_in)
        : executor(std::move(executor_in)),
          enqueue_observer(std::move(enqueue_observer_in)) {}

    // 生产构造时保存真实 ModelInference 调用；内部测试构造时保存可控 fake。
    // Scheduler 只负责安排何时执行，不依赖 executor 的具体实现。
    Executor executor;

    // 生产构造时为空。并发测试用它确认某个 index 已被接收或拒绝，避免用
    // sleep 猜测队列状态；观察器不得改变正式调度结果。
    EnqueueObserver enqueue_observer;

    // 共享状态锁，保护 state、snapshot、config 和 pool 的一致性。InferBatch 只在锁内
    // 检查运行状态并复制本次所需快照snapshot/config/pool，随后立刻释放，不能持有此锁执行 Enqueue 或等待 future。
    std::mutex state_mutex;

    // 生命周期操作锁，将 Start/Stop 的完整生命周期操作串行化，避免两个线程同时创建（Start）、停止或
    // join 同一个 pool。若同时需要两把锁，固定先 lifecycle_mutex、后 state_mutex。
    std::mutex lifecycle_mutex;
    SchedulerState state = SchedulerState::kStopped;
    // 言而简之，lifecycle_mutex：保证 Start/Stop 不能互相穿插
    // state_mutex：保证共享字段读写一致，而且只短暂持有

    // Start 成功后固定到本运行周期；换模型或配置必须 Stop 后重新 Start。
    ModelRuntimeSnapshot snapshot;
    SchedulerConfig config;

    // InferBatch 会在 state_mutex 内复制 shared_ptr，再在锁外提交和等待任务。
    // 因此并发 Stop 即使清空 Impl::pool，已进入的 batch 仍不会访问悬空对象。
    std::shared_ptr<detail::BoundedWorkerPool> pool;
};

InferenceScheduler::InferenceScheduler()
    : InferenceScheduler([](const ModelRuntimeSnapshot& snapshot,
                           const NcnnWav2LipInput& input,
                           const InferenceOptions& options) {
          ModelInference model_inference;
          return model_inference.Infer(snapshot, input, options);
      }) {}

InferenceScheduler::InferenceScheduler(
    Executor executor,
    EnqueueObserver enqueue_observer)
    : pImpl_(std::make_unique<Impl>(
          std::move(executor),
          std::move(enqueue_observer))) {}

InferenceScheduler::~InferenceScheduler() {
    // 析构沿用公开 Stop 的排空语义：拒绝新任务、完成已接收任务并 join worker。
    // Stop 自己负责在锁保护下更新 state；析构不能绕过锁再次直接写共享状态。
    Stop();
}

/// @brief 校验并固定 snapshot/config，然后创建 worker pool
SchedulerControlResult InferenceScheduler::Start(const ModelRuntimeSnapshot& snapshot, 
                                                 const SchedulerConfig& config) {
    
                                                    
    // 第一个分支：
    // 1.创建具名 SchedulerControlResult result；
    SchedulerControlResult result;
    // 2.先锁 lifecycle_mutex；
    std::lock_guard<std::mutex> lifecycle_lock(pImpl_->lifecycle_mutex);
    // 3.再锁 state_mutex 检查状态；
    {
        std::lock_guard<std::mutex> state_lock(pImpl_->state_mutex);
        // 4.如果已经是 kRunning，返回 kEngineAlreadyRunning；
        if (pImpl_->state == SchedulerState::kRunning) {
            result.success = false;
            result.status = InferenceStatus::kEngineAlreadyRunning;
            result.error_message = ModelInference::StatusToString(result.status);

            return result;
        }
    }

    // 5.校验 snapshot/config，不创建 pool
    if (!snapshot.IsValid()) {
        result.success = false;
        result.status = InferenceStatus::kModelUnavailable;
        result.error_message = ModelInference::StatusToString(result.status);
        return result;
    }

    if (snapshot.effective_num_threads <= 0) {
        result.success = false;
        result.status = InferenceStatus::kModelUnavailable;
        result.error_message = "模型快照不可用：effective_num_threads 必须大于 0";
        return result;
    }

    if (config.worker_count == 0) {
        result.success = false;
        result.status = InferenceStatus::kInvalidSchedulerConfig;
        result.error_message = "Scheduler 配置无效：worker_count 必须大于 0";
        return result;
    }

    if (config.queue_capacity == 0) {
        result.success = false;
        result.status = InferenceStatus::kInvalidSchedulerConfig;
        result.error_message = "Scheduler 配置无效：queue_capacity 必须大于 0";
        return result;
    }

    if (config.queue_wait_timeout.count() < 0) {
        result.success = false;
        result.status = InferenceStatus::kInvalidSchedulerConfig;
        result.error_message = "Scheduler 配置无效：queue_wait_timeout 不能为负数";
        return result;
    }

    // 6.线程预算计算
    // 请求级并发 worker_count × 每次 ncnn 推理内部线程 effective_num_threads ≈ 可能同时占用的计算线程数
    const unsigned int hardware_threads = std::thread::hardware_concurrency();

    if (!config.allow_thread_oversubscription && hardware_threads > 0) {
        const std::size_t ncnn_threads = static_cast<std::size_t>(snapshot.effective_num_threads);

        // 使用除法比较，避免 worker_count * ncnn_threads 溢出。
        if (config.worker_count > static_cast<std::size_t>(hardware_threads) / ncnn_threads) {
            result.success = false;
            result.status = InferenceStatus::kThreadBudgetExceeded;
            result.error_message = ModelInference::StatusToString(result.status);
            return result;
        }
    }

    // 第二分支
    // 先创建局部候选 pool，成功后再一次性发布到 Impl。这样构造线程失败时，Scheduler 仍保持 kStopped，
    // 不会留下半初始化状态。
    std::shared_ptr<detail::BoundedWorkerPool> candidate_pool;

    try {
        candidate_pool = std::make_shared<detail::BoundedWorkerPool>(config.worker_count, 
                                                                     config.queue_capacity);
    } catch (const std::exception& exception) {
        result.success = false;
        result.status = InferenceStatus::kUnknownError;
        result.error_message = std::string("创建 worker pool 失败：") + exception.what();
        return result;
    } catch (...) {
        result.success = false;
        result.status = InferenceStatus::kUnknownError;
        result.error_message = "创建 worker pool 失败：未知异常";
        return result;
    }

    {
        // lifecycle_mutex 已经持有；按固定顺序再取得 state_mutex。
        std::lock_guard<std::mutex> state_lock(pImpl_->state_mutex);

        // 复制本次所需快照snapshot/config/pool
        pImpl_->snapshot = snapshot;
        pImpl_->config = config;
        pImpl_->pool = std::move(candidate_pool);

        // 所有依赖都发布完成后，最后切换为 Running。
        pImpl_->state = SchedulerState::kRunning;
    }

    result.success = true;
    result.status = InferenceStatus::kOk;
    result.error_message.clear();
    return result;
}

/// @brief 停止接收新任务、排空已接收任务并 join；可顺序重复调用
SchedulerControlResult InferenceScheduler::Stop() {
    const auto stop_start = std::chrono::steady_clock::now();
    SchedulerControlResult result;

    // 1.整个 Stop 持有生命周期锁，防止另一个 Start/Stop 在排空过程中穿插。
    std::lock_guard<std::mutex> lifecycle_lock(pImpl_->lifecycle_mutex);

    // pImpl_->pool ──────┐
    //                    ├──→ 同一个 BoundedWorkerPool
    // pool_to_stop ──────┘
    std::shared_ptr<detail::BoundedWorkerPool> pool_to_stop;

    {
        // 2.只在读取和修改共享字段时短暂持有状态锁。
        std::lock_guard<std::mutex> state_lock(pImpl_->state_mutex);

        // 2.1 已停止已经满足 Stop 的目标，因此幂等返回 kOk，不把它当成错误。
        if (pImpl_->state == SchedulerState::kStopped) {
            result.success = true;
            result.status = InferenceStatus::kOk;
            result.error_message.clear();
            result.stop_wait_ms =std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - stop_start).count();
            return result;
        }

        // 2.2 正常旧状态是 Running。先发布 Stopping，让并发 InferBatch 能立即
        // 拒绝新任务；再复制 shared_ptr，供锁外执行耗时的排空和 join。
        pImpl_->state = SchedulerState::kStopping;
        pool_to_stop = pImpl_->pool;
    }

    // 3. pool_to_stop 只是同一 pool 的局部共享所有权。此调用可能长时间等待，
    // 因此绝不能持有 state_mutex。
    if (pool_to_stop) {
        pool_to_stop->Stop();
    }

    // 4.再取得 state_mutex，清空 pool/snapshot/config
    {
        // lifecycle_mutex 仍然持有；按固定顺序再次取得 state_mutex，清空
        // 本运行周期持有的资源，使模型和 pool 可以在最后一个引用离开时释放。
        std::lock_guard<std::mutex> state_lock(pImpl_->state_mutex);

        pImpl_->pool.reset();
        pImpl_->snapshot = ModelRuntimeSnapshot{};
        pImpl_->config = SchedulerConfig{};
        // 所有依赖都清空后，最后切换为 Stopped。
        pImpl_->state = SchedulerState::kStopped;
    }

    result.success = true;
    result.status = InferenceStatus::kOk;
    result.error_message.clear();
    result.stop_wait_ms =std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - stop_start).count();
    return result;
}

/// @brief 同步提交 task batch；任务可乱序完成，结果始终按输入 index 返回
BatchInferenceResult InferenceScheduler::InferBatch(const std::vector<NcnnWav2LipInput>& inputs) {
    const auto batch_start = std::chrono::steady_clock::now();
    BatchInferenceResult result;

    // 1.取得本次 batch 的稳定运行上下文
    // 先固定大小，worker 后续只写 results[index]，绝不 push_back，避免乱序
    // resize 必须发生在任何任务入队之前，后续 worker 只能修改自己 index
    // 对应的元素，不能再改变 vector 的大小或触发底层存储重新分配。
    result.results.resize(inputs.size());
    result.summary.total_count = inputs.size();

    // 1.2 本批次 batch 的 pool/snapshot/config
    // 这些局部副本共同描述 InferBatch 进入时看到的同一个运行周期。复制完成
    // 后立即释放 state_mutex，后续排队、推理和等待 future 都不占用状态锁。
    SchedulerState scheduler_state = SchedulerState::kStopped;
    ModelRuntimeSnapshot snapshot;
    SchedulerConfig config;
    std::shared_ptr<detail::BoundedWorkerPool> pool;
    Executor executor;
    EnqueueObserver enqueue_observer;
    {
        // 只在复制共享运行上下文时持锁。
        std::lock_guard<std::mutex> state_lock(pImpl_->state_mutex);

        scheduler_state = pImpl_->state;
        snapshot = pImpl_->snapshot;
        config = pImpl_->config;
        pool = pImpl_->pool;
        executor = pImpl_->executor;
        enqueue_observer = pImpl_->enqueue_observer;
    }
    // 1.3 本次批次 batch 的模型代数
    // generation 是模型代次，不随任务完成顺序变化；同一 batch 的每个结果
    // 都使用这次锁内复制得到的 snapshot.generation。
    result.model_generation = snapshot.generation;

    // 2.请求上下文初始化 + Scheduler 状态早退
    // 2.1 每个失败项也必须保留原输入 metadata 和本次 generation。
    // 在任何早退或入队之前先填充公共上下文，保证未运行、停止中、排队超时
    // 和 task 异常等失败结果仍能按 frame_index/PTS 和模型代次追溯。
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        result.results[index].value.metadata = inputs[index].metadata;
        result.results[index].value.model_generation = snapshot.generation;
    }

    // 2.2 Scheduler 状态不是Running，
    // 本批次不能提交任何 job：Stopping 表示停止过程正在拒绝新任务；
    // 其他非 Running 状态按尚未启动或已经停止处理。
    if (scheduler_state != SchedulerState::kRunning) {
        // 顶层和每个输入槽使用同一个生命周期状态。这样调用方既能快速判断
        // 整批失败，也能继续依赖 results.size() == inputs.size() 逐项对齐。
        const InferenceStatus status = scheduler_state == SchedulerState::kStopping
                                                          ? InferenceStatus::kEngineStopping
                                                          : InferenceStatus::kEngineNotRunning;

        result.success = false;
        result.status = status;
        result.error_message = ModelInference::StatusToString(status);

        for (SingleInferenceResult& item : result.results) {
            // 这里只填写统一 Result 外壳；前面已经写入 metadata/generation，
            // 没有任务进入 ModelInference，因此 attempt_count 保持 0、pred 保持空。
            item.success = false;
            item.status = status;
            item.error_message = ModelInference::StatusToString(status);
            // pred 默认保持为空
        }

        // failure_count 统计本次调用中没有成功的输入项；只有 Stopping 才表示
        // 任务是因停止过程拒绝，普通 Stopped 不计入 stopping_rejected_count。
        result.summary.failure_count = inputs.size();

        if (status == InferenceStatus::kEngineStopping) {
            result.summary.stopping_rejected_count = inputs.size();
        }

        // 即使在调度前早退，也记录 InferBatch 从进入到返回的完整 makespan。
        result.summary.batch_makespan_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - batch_start).count();

        return result;
    }

    // 2.3 单独处理运行状态下的空 batch
    // 生命周期检查优先于输入数量检查；只有 Running Scheduler 收到空 inputs
    // 时才返回 kEmptyBatchInput。此时 results/total_count 都自然保持为 0。
    if (inputs.empty()) {
        result.success = false;
        result.status = InferenceStatus::kEmptyBatchInput;
        result.error_message = ModelInference::StatusToString(result.status);
        result.summary.batch_makespan_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - batch_start).count();
        return result;
    }

    // Running 与非空 pool 应由 Start 原子发布。这里仍做防御性检查，避免内部
    // 状态损坏后解引用空指针；所有输入槽保持失败且 pred 为空。
    if (!pool) {
        result.success = false;
        result.status = InferenceStatus::kUnknownError;
        result.error_message = "Scheduler 内部状态错误：Running 时 worker pool 为空";

        for (SingleInferenceResult& item : result.results) {
            item.success = false;
            item.status = InferenceStatus::kUnknownError;
            item.error_message = result.error_message;
        }

        result.summary.failure_count = inputs.size();
        result.summary.batch_makespan_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - batch_start).count();
        return result;
    }

    // 3.准备统一入队期限和已接收任务凭证
    // future 证明对应 task 已经结束，并把 task 异常带回 InferBatch；index 用于
    // 将异常和排队耗时写回原输入位置，而不是按实际完成顺序追加结果。
    struct AcceptedJob {
        std::size_t index = 0;
        std::future<void> completion;
    };

    // 整个 batch 共用一个绝对 deadline；前一项消耗的等待时间不会在后一项
    // 重新获得，避免 N 个输入把总入队预算放大成 N × timeout。
    const auto enqueue_deadline = batch_start + config.queue_wait_timeout;
    std::vector<AcceptedJob> accepted_jobs;
    accepted_jobs.reserve(inputs.size());

    // 提交线程先独立记录排队耗时，等对应 future 完成后再写入 Result，避免它
    // 与 worker 同时修改同一个 results[index]。
    std::vector<double> queue_wait_ms(inputs.size(), 0.0);

    // 4.按输入顺序尝试入队；执行可以乱序，结果始终写回固定 index
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto enqueue_start = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::milliseconds remaining = now < enqueue_deadline
                ? std::chrono::ceil<std::chrono::milliseconds>(enqueue_deadline - now)
                : std::chrono::milliseconds(0);

        // input、snapshot 和 options 按值进入 task；ncnn::Mat 的复制共享引用计数
        // 数据。results 已预分配且每个 job 只写自己的 index，不改变 vector 大小。
        detail::BoundedEnqueueResult enqueue_result;
        try {
            enqueue_result = pool->Enqueue(
                    [batch_results = &result.results,
                     index,
                     snapshot,
                     input = inputs[index],
                     options = config.inference_options,
                     executor]() {
                        (*batch_results)[index] = executor(snapshot, input, options);
                    },
                    remaining);
        } catch (const std::exception& exception) {
            // 当前 task 尚未被 pool 接收；保留之前 accepted_jobs 的 futures，
            // 后面仍会等待它们，避免异常早退后 worker 写入已销毁的 results。
            queue_wait_ms[index] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - enqueue_start).count();
            result.summary.enqueue_wait_ms += queue_wait_ms[index];

            SingleInferenceResult& item = result.results[index];
            item.success = false;
            item.status = InferenceStatus::kUnknownError;
            item.error_message = std::string("任务入队异常：") + exception.what();
            item.value.timing.queue_wait_ms = queue_wait_ms[index];
            continue;
        } catch (...) {
            queue_wait_ms[index] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - enqueue_start).count();
            result.summary.enqueue_wait_ms += queue_wait_ms[index];

            SingleInferenceResult& item = result.results[index];
            item.success = false;
            item.status = InferenceStatus::kUnknownError;
            item.error_message = "任务入队异常：未知异常";
            item.value.timing.queue_wait_ms = queue_wait_ms[index];
            continue;
        }

        queue_wait_ms[index] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enqueue_start).count();
        result.summary.enqueue_wait_ms += queue_wait_ms[index];

        // 测试观察器只报告 Enqueue 已经产生的结果，不参与状态决策。即使测试
        // 回调自身异常，也不能破坏已接收 future 的等待和生产调度语义。
        if (enqueue_observer) {
            try {
                enqueue_observer(
                    index,
                    enqueue_result.status ==
                        detail::BoundedEnqueueStatus::kAccepted);
            } catch (...) {
                // 测试辅助回调不允许改变被测 Scheduler 的业务结果。
            }
        }

        if (enqueue_result.status == detail::BoundedEnqueueStatus::kAccepted) {
            // BoundedWorkerPool 的契约保证 accepted 必定携带 completion future。
            ++result.summary.accepted_count;
            accepted_jobs.push_back({index, std::move(*enqueue_result.completion)});
            continue;
        }

        SingleInferenceResult& item = result.results[index];
        item.success = false;
        item.value.timing.queue_wait_ms = queue_wait_ms[index];

        if (enqueue_result.status == detail::BoundedEnqueueStatus::kTimedOut) {
            item.status = InferenceStatus::kQueueWaitTimeout;
            ++result.summary.queue_timeout_count;
        } else if (enqueue_result.status == detail::BoundedEnqueueStatus::kFatalError) {
            // pool 发生致命错误（worker 兜底 catch），不再接收任何任务
            item.status = InferenceStatus::kUnknownError;
            item.error_message = "BoundedWorkerPool 致命错误：worker 出现无法传播到 future 的异常";
            ++result.summary.stopping_rejected_count;
        } else {
            // 并发 Stop 已让 pool 停止接收；该任务从未进入队列，不属于待排空任务。
            item.status = InferenceStatus::kEngineStopping;
            ++result.summary.stopping_rejected_count;
        }

        item.error_message = ModelInference::StatusToString(item.status);
        // 未接收项没有执行 ModelInference：attempt_count 保持 0，pred 保持空。
    }

    // 5.兑现所有已接收任务：不允许 InferBatch 在 worker 仍引用 results 时返回
    for (AcceptedJob& job : accepted_jobs) {
        SingleInferenceResult& item = result.results[job.index];

        try {
            job.completion.get();
        } catch (const std::exception& exception) {
            // packaged_task 将 task 异常存入 future；这里只让对应输入失败，worker
            // 线程和同 batch 的其他任务仍可继续运行。
            item.success = false;
            item.status = InferenceStatus::kTaskException;
            item.error_message = std::string("推理任务执行异常：") + exception.what();
            item.value.pred = ncnn::Mat();
            item.value.metadata = inputs[job.index].metadata;
            item.value.model_generation = snapshot.generation;
            ++result.summary.task_exception_count;
        } catch (...) {
            item.success = false;
            item.status = InferenceStatus::kTaskException;
            item.error_message = "推理任务执行异常：未知异常";
            item.value.pred = ncnn::Mat();
            item.value.metadata = inputs[job.index].metadata;
            item.value.model_generation = snapshot.generation;
            ++result.summary.task_exception_count;
        }

        // 此时对应 worker 已结束，不会再覆盖 item，可以安全补充 Scheduler 计时。
        item.value.timing.queue_wait_ms = queue_wait_ms[job.index];
    }

    // 6.所有结果都已进入终态，再做只读聚合并决定 batch 顶层状态
    for (const SingleInferenceResult& item : result.results) {
        if (item.success) {
            ++result.summary.success_count;
        }
        if (item.value.attempts.recovered_by_retry) {
            ++result.summary.retry_recovered_count;
        }
    }

    result.summary.failure_count = result.summary.total_count - result.summary.success_count;
    result.summary.batch_makespan_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - batch_start).count();

    if (result.summary.batch_makespan_ms > 0.0) {
        result.summary.throughput_items_per_second =
                static_cast<double>(result.summary.total_count) * 1000.0 /
                result.summary.batch_makespan_ms;
    }

    if (result.summary.success_count == result.summary.total_count) {
        result.success = true;
        result.status = InferenceStatus::kOk;
        result.error_message.clear();
    } else if (result.summary.success_count == 0) {
        result.success = false;
        result.status = InferenceStatus::kAllFailed;
        result.error_message = ModelInference::StatusToString(result.status);
    } else {
        result.success = false;
        result.status = InferenceStatus::kPartialFailure;
        result.error_message = ModelInference::StatusToString(result.status);
    }

    return result;

}

} // namespace model
} // namespace digital_human
