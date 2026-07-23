#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "model/inference_scheduler.h"
#include "model/detail/bounded_worker_pool.h"
#include "model/detail/inference_scheduler_test_access.h"

using namespace digital_human::model;
using namespace std::chrono_literals;

namespace {

/// @brief 测试专用 gate：明确证明 task 已到达阻塞点，再由测试线程释放
/// @note  2 秒只作为失败保险，不用于猜测线程何时到达。
class ManualGate {
public:
    void ArriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        arrived_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this]() { return released_; });
    }

    bool WaitUntilArrived() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(
            lock,
            2s,
            [this]() { return arrived_; });
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool arrived_ = false;
    bool released_ = false;
};

/// @brief 等指定数量的任务全部到达后再统一放行，用来证明多个调用确实并发进入 executor
class CountedGate {
public:
    explicit CountedGate(std::size_t target_count)
        : target_count_(target_count) {}

    void ArriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_count_;
        condition_.notify_all();
        condition_.wait(lock, [this]() { return released_; });
    }

    bool WaitUntilAllArrived() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, 2s, [this]() {
            return arrived_count_ == target_count_;
        });
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t target_count_ = 0;
    std::size_t arrived_count_ = 0;
    bool released_ = false;
};

/// @brief 记录每个输入是否被 worker pool 接收，供测试等待确定的队列状态
class EnqueueProbe {
public:
    explicit EnqueueProbe(std::size_t count)
        : observed_(count, false), accepted_(count, false) {}

    void Record(std::size_t index, bool accepted) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            observed_[index] = true;
            accepted_[index] = accepted;
        }
        condition_.notify_all();
    }

    bool WaitUntilObserved(std::size_t index, bool accepted) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, 2s, [&]() {
            return observed_[index] && accepted_[index] == accepted;
        });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<bool> observed_;
    std::vector<bool> accepted_;
};

SingleInferenceResult MakeSyntheticSuccess(
    const ModelRuntimeSnapshot& snapshot,
    const NcnnWav2LipInput& input,
    float pred_value) {
    SingleInferenceResult result;
    result.success = true;
    result.status = InferenceStatus::kOk;
    result.value.pred = ncnn::Mat(1, sizeof(float));
    result.value.pred.fill(pred_value);
    result.value.metadata = input.metadata;
    result.value.model_generation = snapshot.generation;
    result.value.attempts.attempt_count = 1;
    result.value.attempts.final_attempt_status = InferenceStatus::kOk;
    return result;
}

void ExpectAccepted(const detail::BoundedEnqueueResult& result) {
    EXPECT_EQ(result.status, detail::BoundedEnqueueStatus::kAccepted);
    EXPECT_TRUE(result.completion.has_value());
}

} // namespace

// ============================================================================
// 单 worker 必须按 FIFO 执行已接收任务；Stop 可重复调用并拒绝后续提交
// ============================================================================
TEST(BoundedWorkerPoolTest, ExecutesAcceptedTasksAndRejectsAfterStop) {
    detail::BoundedWorkerPool pool(1, 2);
    std::mutex order_mutex;
    std::vector<int> execution_order;

    auto first = pool.Enqueue(
        [&]() {
            std::lock_guard<std::mutex> lock(order_mutex);
            execution_order.push_back(1);
        },
        1s);
    auto second = pool.Enqueue(
        [&]() {
            std::lock_guard<std::mutex> lock(order_mutex);
            execution_order.push_back(2);
        },
        1s);

    ExpectAccepted(first);
    ExpectAccepted(second);
    ASSERT_TRUE(first.completion.has_value());
    ASSERT_TRUE(second.completion.has_value());
    first.completion->get();
    second.completion->get();

    pool.Stop();
    pool.Stop(); // Stop 必须顺序幂等，不能重复 join 或抛异常。

    ASSERT_EQ(execution_order.size(), 2u);
    EXPECT_EQ(execution_order[0], 1);
    EXPECT_EQ(execution_order[1], 2);

    auto rejected = pool.Enqueue([]() {}, 0ms);
    EXPECT_EQ(rejected.status, detail::BoundedEnqueueStatus::kStopping);
    EXPECT_FALSE(rejected.completion.has_value());
}

// ============================================================================
// worker 被 gate 阻塞且等待队列已满时，第三个任务必须在 deadline 明确超时
// ============================================================================
TEST(BoundedWorkerPoolTest, FullQueueTimesOutWithoutAcceptingTask) {
    detail::BoundedWorkerPool pool(1, 1);
    ManualGate running_gate;
    std::atomic<bool> timed_out_task_ran{false};

    auto running = pool.Enqueue(
        [&]() { running_gate.ArriveAndWait(); },
        1s);
    ExpectAccepted(running);

    const bool worker_arrived = running_gate.WaitUntilArrived();
    if (!worker_arrived) {
        running_gate.Release();
        FAIL() << "worker did not reach running gate";
    }

    // worker 正在执行 running，容量为 1 的等待队列由 queued 占满。
    auto queued = pool.Enqueue([]() {}, 1s);
    ExpectAccepted(queued);

    // 这里的 20ms 是被测 timeout 本身，不是用 sleep 猜线程状态。
    auto timed_out = pool.Enqueue(
        [&]() { timed_out_task_ran.store(true); },
        20ms);
    EXPECT_EQ(timed_out.status, detail::BoundedEnqueueStatus::kTimedOut);
    EXPECT_FALSE(timed_out.completion.has_value());

    running_gate.Release();
    ASSERT_TRUE(running.completion.has_value());
    ASSERT_TRUE(queued.completion.has_value());
    running.completion->get();
    queued.completion->get();
    pool.Stop();

    EXPECT_FALSE(timed_out_task_ran.load());
}

// ============================================================================
// 满队列 producer 在 worker 制造空位后应成功入队，而不是被静默丢弃
// ============================================================================
TEST(BoundedWorkerPoolTest, FullQueueAcceptsWaitingTaskAfterSpaceAppears) {
    detail::BoundedWorkerPool pool(1, 1);
    ManualGate running_gate;
    std::atomic<bool> waiting_task_ran{false};

    auto running = pool.Enqueue(
        [&]() { running_gate.ArriveAndWait(); },
        1s);
    ExpectAccepted(running);

    const bool worker_arrived = running_gate.WaitUntilArrived();
    if (!worker_arrived) {
        running_gate.Release();
        FAIL() << "worker did not reach running gate";
    }

    auto queued = pool.Enqueue([]() {}, 1s);
    ExpectAccepted(queued);

    std::promise<void> producer_started_promise;
    std::future<void> producer_started =
        producer_started_promise.get_future();
    std::promise<detail::BoundedEnqueueStatus> producer_status_promise;
    std::future<detail::BoundedEnqueueStatus> producer_status =
        producer_status_promise.get_future();

    std::thread producer([&]() {
        producer_started_promise.set_value();
        auto waiting = pool.Enqueue(
            [&]() { waiting_task_ran.store(true); },
            2s);
        producer_status_promise.set_value(waiting.status);
    });

    // 只在 producer 已到达提交步骤后释放 A；worker 随后取走 queued 并通知
    // not_full，等待提交的任务最终必须被接受。
    producer_started.get();
    running_gate.Release();
    producer.join();
    pool.Stop();

    EXPECT_EQ(producer_status.get(),
              detail::BoundedEnqueueStatus::kAccepted);
    ASSERT_TRUE(running.completion.has_value());
    ASSERT_TRUE(queued.completion.has_value());
    running.completion->get();
    queued.completion->get();
    EXPECT_TRUE(waiting_task_ran.load());
}

// ============================================================================
// Stop 必须唤醒满队列上的 producer、拒绝未接收任务，并排空 A/B
// ============================================================================
TEST(BoundedWorkerPoolTest, StopUnblocksProducerAndDrainsAcceptedTasks) {
    detail::BoundedWorkerPool pool(1, 1);
    ManualGate running_gate;
    std::atomic<bool> queued_task_ran{false};
    std::atomic<bool> rejected_task_ran{false};

    auto running = pool.Enqueue(
        [&]() { running_gate.ArriveAndWait(); },
        1s);
    ExpectAccepted(running);

    const bool worker_arrived = running_gate.WaitUntilArrived();
    if (!worker_arrived) {
        running_gate.Release();
        FAIL() << "worker did not reach running gate";
    }

    auto queued = pool.Enqueue(
        [&]() { queued_task_ran.store(true); },
        1s);
    ExpectAccepted(queued);

    // 队列保持满时在另一线程调用 Stop：无论当前 Enqueue 先进入等待，还是先
    // 观察到 accepting=false，都必须及时返回 kStopping，绝不能接收新任务。
    std::thread stopper([&]() { pool.Stop(); });
    auto rejected = pool.Enqueue(
        [&]() { rejected_task_ran.store(true); },
        5s);

    // Stop 仍在等待 A/B 排空；先释放 gate，再 join，不能持测试锁等待 worker。
    running_gate.Release();
    stopper.join();

    EXPECT_EQ(rejected.status, detail::BoundedEnqueueStatus::kStopping);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(running.completion.has_value());
    ASSERT_TRUE(queued.completion.has_value());
    running.completion->get();
    queued.completion->get();
    EXPECT_TRUE(queued_task_ran.load());
    EXPECT_FALSE(rejected_task_ran.load());

    auto after_stop = pool.Enqueue([]() {}, 0ms);
    EXPECT_EQ(after_stop.status, detail::BoundedEnqueueStatus::kStopping);
    EXPECT_FALSE(after_stop.completion.has_value());
}

// ============================================================================
// task 异常必须进入自己的 future，不能杀死 worker 或阻止后续任务完成
// ============================================================================
TEST(BoundedWorkerPoolTest, TaskExceptionReachesFutureAndWorkerContinues) {
    detail::BoundedWorkerPool pool(1, 2);
    std::atomic<bool> following_task_ran{false};

    auto throwing = pool.Enqueue(
        []() { throw std::runtime_error("synthetic task failure"); },
        1s);
    auto following = pool.Enqueue(
        [&]() { following_task_ran.store(true); },
        1s);
    ExpectAccepted(throwing);
    ExpectAccepted(following);
    ASSERT_TRUE(throwing.completion.has_value());
    ASSERT_TRUE(following.completion.has_value());

    EXPECT_THROW(throwing.completion->get(), std::runtime_error);
    EXPECT_NO_THROW(following.completion->get());
    EXPECT_TRUE(following_task_ran.load());
    pool.Stop();
}

// ============================================================================
// 析构等价于排空 Stop：调用方忘记显式 Stop 也不能遗留 joinable worker
// ============================================================================
TEST(BoundedWorkerPoolTest, DestructorDrainsAcceptedTasks) {
    std::atomic<int> completed_count{0};

    {
        detail::BoundedWorkerPool pool(1, 2);
        auto first = pool.Enqueue(
            [&]() { completed_count.fetch_add(1); },
            1s);
        auto second = pool.Enqueue(
            [&]() { completed_count.fetch_add(1); },
            1s);
        ExpectAccepted(first);
        ExpectAccepted(second);
        // 离开作用域时析构必须停止接收、排空两个已接收任务并 join。
    }

    EXPECT_EQ(completed_count.load(), 2);
}

// ============================================================================
// 配置哨兵：0 worker 或 0 queue capacity 无法形成可工作的有界池
// ============================================================================
TEST(BoundedWorkerPoolTest, RejectsZeroWorkerOrQueueCapacity) {
    EXPECT_THROW(detail::BoundedWorkerPool(0, 1), std::invalid_argument);
    EXPECT_THROW(detail::BoundedWorkerPool(1, 0), std::invalid_argument);
}

// ============================================================================
// Scheduler 生命周期：参数校验、重复 Start、幂等 Stop 和新 generation Restart
// ============================================================================
TEST(InferenceSchedulerTest, ValidatesLifecycleAndSupportsRestart) {
    InferenceScheduler scheduler;

    SchedulerConfig config;
    config.worker_count = 1;
    config.queue_capacity = 1;
    config.queue_wait_timeout = 100ms;
    // 测试关注生命周期，不依赖执行机器报告的 hardware_concurrency。
    config.allow_thread_oversubscription = true;

    ModelRuntimeSnapshot invalid_snapshot;
    const SchedulerControlResult invalid_snapshot_result =
        scheduler.Start(invalid_snapshot, config);
    EXPECT_FALSE(invalid_snapshot_result.success);
    EXPECT_EQ(invalid_snapshot_result.status, InferenceStatus::kModelUnavailable);

    ModelRuntimeSnapshot first_snapshot;
    first_snapshot.model = std::make_shared<ncnn::Net>();
    first_snapshot.generation = 7;
    first_snapshot.effective_num_threads = 1;

    SchedulerConfig invalid_config = config;
    invalid_config.worker_count = 0;
    const SchedulerControlResult invalid_config_result =
        scheduler.Start(first_snapshot, invalid_config);
    EXPECT_FALSE(invalid_config_result.success);
    EXPECT_EQ(invalid_config_result.status, InferenceStatus::kInvalidSchedulerConfig);

    const SchedulerControlResult first_start = scheduler.Start(first_snapshot, config);
    ASSERT_TRUE(first_start.success) << first_start.error_message;
    EXPECT_EQ(first_start.status, InferenceStatus::kOk);

    const SchedulerControlResult duplicate_start = scheduler.Start(first_snapshot, config);
    EXPECT_FALSE(duplicate_start.success);
    EXPECT_EQ(duplicate_start.status, InferenceStatus::kEngineAlreadyRunning);

    const SchedulerControlResult first_stop = scheduler.Stop();
    EXPECT_TRUE(first_stop.success);
    EXPECT_EQ(first_stop.status, InferenceStatus::kOk);
    EXPECT_GE(first_stop.stop_wait_ms, 0.0);

    // Stop 已经满足目标时仍幂等成功，不能伪造成 kEngineNotRunning 错误。
    const SchedulerControlResult repeated_stop = scheduler.Stop();
    EXPECT_TRUE(repeated_stop.success);
    EXPECT_EQ(repeated_stop.status, InferenceStatus::kOk);

    ModelRuntimeSnapshot second_snapshot = first_snapshot;
    second_snapshot.generation = 8;
    const SchedulerControlResult restart = scheduler.Start(second_snapshot, config);
    ASSERT_TRUE(restart.success) << restart.error_message;
    EXPECT_EQ(restart.status, InferenceStatus::kOk);
    EXPECT_TRUE(scheduler.Stop().success);
}

// ============================================================================
// 正式 InferBatch 必须调用 ModelInference，并按输入 index 保留状态和请求上下文
// ============================================================================
TEST(InferenceSchedulerTest, InferBatchPreservesInputOrderAndContext) {
    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 7;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 2;
    config.queue_capacity = 3;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = true;

    InferenceScheduler scheduler;
    const SchedulerControlResult start_result = scheduler.Start(snapshot, config);
    ASSERT_TRUE(start_result.success) << start_result.error_message;

    std::vector<NcnnWav2LipInput> inputs(3);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].metadata.frame_index = 100 + static_cast<std::int64_t>(index);
        inputs[index].metadata.pts_ms = 1000 + static_cast<std::int64_t>(index);
    }

    // index 0：Mel 为空；index 1：Mel 合法但 Face 为空；
    // index 2：Mel 非空但 width 错误。三种状态用来锁定结果没有按完成顺序错位。
    inputs[1].mel = ncnn::Mat(16, 80, 1, sizeof(float));
    ASSERT_FALSE(inputs[1].mel.empty());
    inputs[1].mel.fill(0.0f);

    inputs[2].mel = ncnn::Mat(15, 80, 1, sizeof(float));
    ASSERT_FALSE(inputs[2].mel.empty());
    inputs[2].mel.fill(0.0f);

    const BatchInferenceResult batch_result = scheduler.InferBatch(inputs);

    EXPECT_FALSE(batch_result.success);
    EXPECT_EQ(batch_result.status, InferenceStatus::kAllFailed);
    ASSERT_EQ(batch_result.results.size(), inputs.size());
    EXPECT_EQ(batch_result.model_generation, snapshot.generation);

    const InferenceStatus expected_statuses[] = {
        InferenceStatus::kEmptyMelInput,
        InferenceStatus::kEmptyFaceInput,
        InferenceStatus::kInvalidMelShape,
    };

    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const SingleInferenceResult& item = batch_result.results[index];
        SCOPED_TRACE(index);
        EXPECT_FALSE(item.success);
        EXPECT_EQ(item.status, expected_statuses[index]);
        EXPECT_TRUE(item.value.pred.empty());
        EXPECT_EQ(item.value.model_generation, snapshot.generation);
        ASSERT_TRUE(item.value.metadata.frame_index.has_value());
        EXPECT_EQ(*item.value.metadata.frame_index,
                  *inputs[index].metadata.frame_index);
        ASSERT_TRUE(item.value.metadata.pts_ms.has_value());
        EXPECT_EQ(*item.value.metadata.pts_ms,
                  *inputs[index].metadata.pts_ms);
        EXPECT_GE(item.value.timing.queue_wait_ms, 0.0);
    }

    EXPECT_EQ(batch_result.summary.total_count, 3u);
    EXPECT_EQ(batch_result.summary.accepted_count, 3u);
    EXPECT_EQ(batch_result.summary.success_count, 0u);
    EXPECT_EQ(batch_result.summary.failure_count, 3u);
    EXPECT_EQ(batch_result.summary.queue_timeout_count, 0u);
    EXPECT_EQ(batch_result.summary.stopping_rejected_count, 0u);
    EXPECT_GE(batch_result.summary.batch_makespan_ms, 0.0);
    EXPECT_GE(batch_result.summary.throughput_items_per_second, 0.0);

    EXPECT_TRUE(scheduler.Stop().success);
}

// ============================================================================
// 未运行与空 batch 都必须明确失败，且仍保持 results 与 inputs 数量对齐
// ============================================================================
TEST(InferenceSchedulerTest, RejectsNotRunningAndEmptyBatchExplicitly) {
    InferenceScheduler scheduler;

    NcnnWav2LipInput input;
    input.metadata.frame_index = 42;
    const BatchInferenceResult not_running = scheduler.InferBatch({input});

    EXPECT_FALSE(not_running.success);
    EXPECT_EQ(not_running.status, InferenceStatus::kEngineNotRunning);
    ASSERT_EQ(not_running.results.size(), 1u);
    EXPECT_EQ(not_running.results[0].status, InferenceStatus::kEngineNotRunning);
    EXPECT_TRUE(not_running.results[0].value.pred.empty());
    ASSERT_TRUE(not_running.results[0].value.metadata.frame_index.has_value());
    EXPECT_EQ(*not_running.results[0].value.metadata.frame_index, 42);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 9;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 1;
    config.queue_capacity = 1;
    config.queue_wait_timeout = 0ms;
    config.allow_thread_oversubscription = true;

    ASSERT_TRUE(scheduler.Start(snapshot, config).success);
    const BatchInferenceResult empty_batch = scheduler.InferBatch({});
    EXPECT_FALSE(empty_batch.success);
    EXPECT_EQ(empty_batch.status, InferenceStatus::kEmptyBatchInput);
    EXPECT_TRUE(empty_batch.results.empty());
    EXPECT_EQ(empty_batch.summary.total_count, 0u);
    EXPECT_EQ(empty_batch.model_generation, snapshot.generation);
    EXPECT_TRUE(scheduler.Stop().success);
}

// ============================================================================
// 真实模型 task batch：两个合法输入都必须经正式 Scheduler 和 ModelInference 成功
// ============================================================================
TEST(InferenceSchedulerTest, RealModelBatchSucceedsWithStableGeneration) {
    ModelLoader loader;
    ModelLoadOptions load_options;
    load_options.backend = ModelBackend::kCpu;
    load_options.num_threads = 1;
    load_options.enable_warmup = false;

    const ModelLoadResult load_result =
        loader.Load("models/wav2lip/wav2lip.param", load_options);
    ASSERT_TRUE(load_result.success) << load_result.error_message;

    const ModelRuntimeSnapshot snapshot = loader.AcquireSnapshot();
    ASSERT_TRUE(snapshot.IsValid());

    SchedulerConfig config;
    config.worker_count = 2;
    config.queue_capacity = 2;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = true;

    InferenceScheduler scheduler;
    const SchedulerControlResult start_result = scheduler.Start(snapshot, config);
    ASSERT_TRUE(start_result.success) << start_result.error_message;

    std::vector<NcnnWav2LipInput> inputs(2);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].mel = ncnn::Mat(16, 80, 1, sizeof(float));
        inputs[index].face = ncnn::Mat(96, 96, 6, sizeof(float));
        ASSERT_FALSE(inputs[index].mel.empty());
        ASSERT_FALSE(inputs[index].face.empty());
        inputs[index].mel.fill(0.0f);
        inputs[index].face.fill(0.0f);
        inputs[index].metadata.frame_index = 200 + static_cast<std::int64_t>(index);
        inputs[index].metadata.pts_ms = 2000 + static_cast<std::int64_t>(index);
    }

    const BatchInferenceResult batch_result = scheduler.InferBatch(inputs);

    ASSERT_TRUE(batch_result.success) << batch_result.error_message;
    EXPECT_EQ(batch_result.status, InferenceStatus::kOk);
    ASSERT_EQ(batch_result.results.size(), inputs.size());
    EXPECT_EQ(batch_result.model_generation, snapshot.generation);
    EXPECT_EQ(batch_result.summary.total_count, 2u);
    EXPECT_EQ(batch_result.summary.accepted_count, 2u);
    EXPECT_EQ(batch_result.summary.success_count, 2u);
    EXPECT_EQ(batch_result.summary.failure_count, 0u);

    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const SingleInferenceResult& item = batch_result.results[index];
        SCOPED_TRACE(index);
        ASSERT_TRUE(item.success) << item.error_message;
        EXPECT_EQ(item.status, InferenceStatus::kOk);
        EXPECT_FALSE(item.value.pred.empty());
        EXPECT_EQ(item.value.pred.w, 96);
        EXPECT_EQ(item.value.pred.h, 96);
        EXPECT_EQ(item.value.pred.c, 3);
        EXPECT_EQ(item.value.model_generation, snapshot.generation);
        ASSERT_TRUE(item.value.metadata.frame_index.has_value());
        EXPECT_EQ(*item.value.metadata.frame_index,
                  *inputs[index].metadata.frame_index);
        EXPECT_GE(item.value.timing.queue_wait_ms, 0.0);
    }

    EXPECT_TRUE(scheduler.Stop().success);
}

// ============================================================================
// fake executor 按 2→0→1 完成时，InferBatch 仍必须按输入 index 返回 0→1→2
// ============================================================================
TEST(InferenceSchedulerTest, PreservesInputOrderWhenTasksCompleteOutOfOrder) {
    std::array<ManualGate, 3> start_gates;
    std::array<ManualGate, 3> completion_gates;
    std::mutex completion_order_mutex;
    std::vector<std::size_t> completion_order;

    // fake executor 不做 ncnn 推理。每个任务先停在自己的 start gate，测试线程
    // 决定放行顺序；记录完成顺序后再停在 completion gate，供测试精确确认。
    auto executor = [&](const ModelRuntimeSnapshot& snapshot,
                        const NcnnWav2LipInput& input,
                        const InferenceOptions&) {
        const std::size_t index =
            static_cast<std::size_t>(*input.metadata.frame_index);
        start_gates[index].ArriveAndWait();

        {
            std::lock_guard<std::mutex> lock(completion_order_mutex);
            completion_order.push_back(index);
        }
        completion_gates[index].ArriveAndWait();

        SingleInferenceResult item;
        item.success = true;
        item.status = InferenceStatus::kOk;
        item.value.pred = ncnn::Mat(1, sizeof(float));
        item.value.pred.fill(static_cast<float>(index) + 0.5f);
        item.value.metadata = input.metadata;
        item.value.model_generation = snapshot.generation;
        item.value.attempts.attempt_count = 1;
        item.value.attempts.final_attempt_status = InferenceStatus::kOk;
        return item;
    };

    std::unique_ptr<InferenceScheduler> scheduler =
        detail::InferenceSchedulerTestAccess::Create(executor);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 11;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 3;
    config.queue_capacity = 3;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = true;
    ASSERT_TRUE(scheduler->Start(snapshot, config).success);

    std::vector<NcnnWav2LipInput> inputs(3);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].metadata.frame_index = static_cast<std::int64_t>(index);
    }

    // InferBatch 是同步接口，因此放到另一个线程；当前测试线程负责控制 gate。
    std::future<BatchInferenceResult> batch_future =
        std::async(std::launch::async, [&]() {
            return scheduler->InferBatch(inputs);
        });

    const auto release_all_gates = [&]() {
        for (ManualGate& gate : start_gates) {
            gate.Release();
        }
        for (ManualGate& gate : completion_gates) {
            gate.Release();
        }
    };

    // 先证明三个 worker 都已到达阻塞点，随后严格按 2→0→1 逐个放行。
    for (ManualGate& gate : start_gates) {
        if (!gate.WaitUntilArrived()) {
            release_all_gates();
            batch_future.wait();
            scheduler->Stop();
            FAIL() << "not all fake inference tasks reached their start gates";
        }
    }

    const std::array<std::size_t, 3> release_order{2, 0, 1};
    for (const std::size_t index : release_order) {
        start_gates[index].Release();
        if (!completion_gates[index].WaitUntilArrived()) {
            release_all_gates();
            batch_future.wait();
            scheduler->Stop();
            FAIL() << "fake inference task did not reach its completion gate";
        }
        completion_gates[index].Release();
    }

    const BatchInferenceResult batch_result = batch_future.get();

    // completion_order 证明实际完成顺序确实是 2→0→1。
    ASSERT_EQ(completion_order.size(), 3u);
    EXPECT_EQ(completion_order[0], 2u);
    EXPECT_EQ(completion_order[1], 0u);
    EXPECT_EQ(completion_order[2], 1u);

    // results 不按完成顺序 push_back，而是每个 job 写回自己的原始 index。
    ASSERT_TRUE(batch_result.success) << batch_result.error_message;
    EXPECT_EQ(batch_result.status, InferenceStatus::kOk);
    ASSERT_EQ(batch_result.results.size(), inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const SingleInferenceResult& item = batch_result.results[index];
        SCOPED_TRACE(index);
        ASSERT_TRUE(item.success) << item.error_message;
        ASSERT_TRUE(item.value.metadata.frame_index.has_value());
        EXPECT_EQ(*item.value.metadata.frame_index,
                  static_cast<std::int64_t>(index));
        EXPECT_EQ(item.value.model_generation, snapshot.generation);
        ASSERT_FALSE(item.value.pred.empty());
        EXPECT_FLOAT_EQ(static_cast<const float*>(item.value.pred)[0],
                        static_cast<float>(index) + 0.5f);
    }

    EXPECT_EQ(batch_result.summary.accepted_count, 3u);
    EXPECT_EQ(batch_result.summary.success_count, 3u);
    EXPECT_EQ(batch_result.summary.failure_count, 0u);
    EXPECT_TRUE(scheduler->Stop().success);
}

// ============================================================================
// A 占用唯一 worker、B 占满等待队列时，C 必须在统一 deadline 超时
// ============================================================================
TEST(InferenceSchedulerTest, QueueFullMapsTimedOutInputWithoutDroppingAcceptedJobs) {
    ManualGate running_gate;
    EnqueueProbe enqueue_probe(3);
    std::array<std::atomic<int>, 3> execution_counts{{0, 0, 0}};

    auto executor = [&](const ModelRuntimeSnapshot& snapshot,
                        const NcnnWav2LipInput& input,
                        const InferenceOptions&) {
        const std::size_t index =
            static_cast<std::size_t>(*input.metadata.frame_index);
        execution_counts[index].fetch_add(1);

        if (index == 0) {
            running_gate.ArriveAndWait();
        }

        return MakeSyntheticSuccess(
            snapshot,
            input,
            static_cast<float>(index) + 0.5f);
    };

    auto observer = [&](std::size_t index, bool accepted) {
        enqueue_probe.Record(index, accepted);
    };
    std::unique_ptr<InferenceScheduler> scheduler =
        detail::InferenceSchedulerTestAccess::Create(executor, observer);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 12;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 1;
    config.queue_capacity = 1;
    config.queue_wait_timeout = 100ms;
    config.allow_thread_oversubscription = true;
    ASSERT_TRUE(scheduler->Start(snapshot, config).success);

    std::vector<NcnnWav2LipInput> inputs(3);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].metadata.frame_index = static_cast<std::int64_t>(index);
    }

    std::future<BatchInferenceResult> batch_future =
        std::async(std::launch::async, [&]() {
            return scheduler->InferBatch(inputs);
        });

    // A 已经占住唯一 worker，B 被确认接收后只能位于容量为 1 的等待队列。
    if (!running_gate.WaitUntilArrived() ||
        !enqueue_probe.WaitUntilObserved(1, true) ||
        !enqueue_probe.WaitUntilObserved(2, false)) {
        running_gate.Release();
        batch_future.wait();
        scheduler->Stop();
        FAIL() << "failed to establish deterministic full-queue state";
    }

    // C 的拒绝结果已被观察到后才释放 A，确保超时不是由 sleep 猜出来的。
    running_gate.Release();
    const BatchInferenceResult batch_result = batch_future.get();

    EXPECT_FALSE(batch_result.success);
    EXPECT_EQ(batch_result.status, InferenceStatus::kPartialFailure);
    ASSERT_EQ(batch_result.results.size(), 3u);
    EXPECT_TRUE(batch_result.results[0].success);
    EXPECT_TRUE(batch_result.results[1].success);
    EXPECT_FALSE(batch_result.results[2].success);
    EXPECT_EQ(batch_result.results[2].status,
              InferenceStatus::kQueueWaitTimeout);
    EXPECT_TRUE(batch_result.results[2].value.pred.empty());
    EXPECT_EQ(batch_result.results[2].value.model_generation,
              snapshot.generation);

    EXPECT_EQ(execution_counts[0].load(), 1);
    EXPECT_EQ(execution_counts[1].load(), 1);
    EXPECT_EQ(execution_counts[2].load(), 0);
    EXPECT_EQ(batch_result.summary.accepted_count, 2u);
    EXPECT_EQ(batch_result.summary.success_count, 2u);
    EXPECT_EQ(batch_result.summary.failure_count, 1u);
    EXPECT_EQ(batch_result.summary.queue_timeout_count, 1u);
    EXPECT_TRUE(scheduler->Stop().success);
}

// ============================================================================
// executor 抛出的异常必须只转换对应输入，其他任务仍然正常完成
// ============================================================================
TEST(InferenceSchedulerTest, MapsExecutorExceptionToOriginalInput) {
    auto executor = [](const ModelRuntimeSnapshot& snapshot,
                       const NcnnWav2LipInput& input,
                       const InferenceOptions&) {
        const std::size_t index =
            static_cast<std::size_t>(*input.metadata.frame_index);
        if (index == 1) {
            throw std::runtime_error("synthetic executor failure");
        }
        return MakeSyntheticSuccess(
            snapshot,
            input,
            static_cast<float>(index) + 0.5f);
    };
    std::unique_ptr<InferenceScheduler> scheduler =
        detail::InferenceSchedulerTestAccess::Create(executor);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 13;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 2;
    config.queue_capacity = 3;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = true;
    ASSERT_TRUE(scheduler->Start(snapshot, config).success);

    std::vector<NcnnWav2LipInput> inputs(3);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].metadata.frame_index = static_cast<std::int64_t>(index);
    }

    const BatchInferenceResult batch_result = scheduler->InferBatch(inputs);

    EXPECT_FALSE(batch_result.success);
    EXPECT_EQ(batch_result.status, InferenceStatus::kPartialFailure);
    ASSERT_EQ(batch_result.results.size(), 3u);
    EXPECT_TRUE(batch_result.results[0].success);
    EXPECT_FALSE(batch_result.results[1].success);
    EXPECT_TRUE(batch_result.results[2].success);
    EXPECT_EQ(batch_result.results[1].status,
              InferenceStatus::kTaskException);
    EXPECT_FALSE(batch_result.results[1].error_message.empty());
    EXPECT_TRUE(batch_result.results[1].value.pred.empty());
    ASSERT_TRUE(
        batch_result.results[1].value.metadata.frame_index.has_value());
    EXPECT_EQ(*batch_result.results[1].value.metadata.frame_index, 1);
    EXPECT_EQ(batch_result.results[1].value.model_generation,
              snapshot.generation);

    EXPECT_EQ(batch_result.summary.accepted_count, 3u);
    EXPECT_EQ(batch_result.summary.success_count, 2u);
    EXPECT_EQ(batch_result.summary.failure_count, 1u);
    EXPECT_EQ(batch_result.summary.task_exception_count, 1u);
    EXPECT_TRUE(scheduler->Stop().success);
}

// ============================================================================
// worker_count=2 时同时进入 executor 的任务数不得超过 2
// ============================================================================
TEST(InferenceSchedulerTest, NeverRunsMoreTasksThanWorkerCount) {
    std::mutex probe_mutex;
    std::condition_variable probe_condition;
    std::size_t started_count = 0;
    std::size_t active_count = 0;
    std::size_t max_active_count = 0;
    bool release_first_wave = false;

    auto executor = [&](const ModelRuntimeSnapshot& snapshot,
                        const NcnnWav2LipInput& input,
                        const InferenceOptions&) {
        {
            std::unique_lock<std::mutex> lock(probe_mutex);
            ++started_count;
            ++active_count;
            max_active_count = std::max(max_active_count, active_count);
            probe_condition.notify_all();
            probe_condition.wait(lock, [&]() {
                return release_first_wave;
            });
        }

        SingleInferenceResult item =
            MakeSyntheticSuccess(snapshot, input, 0.5f);

        {
            std::lock_guard<std::mutex> lock(probe_mutex);
            --active_count;
        }
        probe_condition.notify_all();
        return item;
    };
    std::unique_ptr<InferenceScheduler> scheduler =
        detail::InferenceSchedulerTestAccess::Create(executor);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 14;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 2;
    config.queue_capacity = 4;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = true;
    ASSERT_TRUE(scheduler->Start(snapshot, config).success);

    std::vector<NcnnWav2LipInput> inputs(4);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].metadata.frame_index = static_cast<std::int64_t>(index);
    }

    std::future<BatchInferenceResult> batch_future =
        std::async(std::launch::async, [&]() {
            return scheduler->InferBatch(inputs);
        });

    {
        std::unique_lock<std::mutex> lock(probe_mutex);
        if (!probe_condition.wait_for(lock, 2s, [&]() {
                return started_count == config.worker_count;
            })) {
            release_first_wave = true;
            lock.unlock();
            probe_condition.notify_all();
            batch_future.wait();
            scheduler->Stop();
            FAIL() << "two workers did not reach the controlled first wave";
        }

        EXPECT_EQ(active_count, config.worker_count);
        EXPECT_EQ(max_active_count, config.worker_count);
        release_first_wave = true;
    }
    probe_condition.notify_all();

    const BatchInferenceResult batch_result = batch_future.get();
    ASSERT_TRUE(batch_result.success) << batch_result.error_message;
    EXPECT_EQ(batch_result.summary.accepted_count, 4u);
    EXPECT_EQ(batch_result.summary.success_count, 4u);
    EXPECT_EQ(batch_result.summary.failure_count, 0u);
    EXPECT_EQ(started_count, 4u);
    EXPECT_LE(max_active_count, config.worker_count);
    EXPECT_TRUE(scheduler->Stop().success);
}

// ============================================================================
// 并发 Stop 必须排空已接收 A/B，并让尚未接收的 C 在原 index 明确失败
// ============================================================================
TEST(InferenceSchedulerTest, ConcurrentStopDrainsAcceptedAndRejectsWaitingInput) {
    ManualGate running_gate;
    EnqueueProbe enqueue_probe(3);
    std::array<std::atomic<int>, 3> execution_counts{{0, 0, 0}};

    auto executor = [&](const ModelRuntimeSnapshot& snapshot,
                        const NcnnWav2LipInput& input,
                        const InferenceOptions&) {
        const std::size_t index =
            static_cast<std::size_t>(*input.metadata.frame_index);
        execution_counts[index].fetch_add(1);

        if (index == 0) {
            running_gate.ArriveAndWait();
        }

        return MakeSyntheticSuccess(
            snapshot,
            input,
            static_cast<float>(index) + 0.5f);
    };
    auto observer = [&](std::size_t index, bool accepted) {
        enqueue_probe.Record(index, accepted);
    };
    std::unique_ptr<InferenceScheduler> scheduler =
        detail::InferenceSchedulerTestAccess::Create(executor, observer);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 15;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 1;
    config.queue_capacity = 1;
    config.queue_wait_timeout = 5s;
    config.allow_thread_oversubscription = true;
    ASSERT_TRUE(scheduler->Start(snapshot, config).success);

    std::vector<NcnnWav2LipInput> inputs(3);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index].metadata.frame_index = static_cast<std::int64_t>(index);
    }

    std::future<BatchInferenceResult> batch_future =
        std::async(std::launch::async, [&]() {
            return scheduler->InferBatch(inputs);
        });

    if (!running_gate.WaitUntilArrived() ||
        !enqueue_probe.WaitUntilObserved(1, true)) {
        running_gate.Release();
        batch_future.wait();
        scheduler->Stop();
        FAIL() << "A did not run while B occupied the waiting queue";
    }

    // B 已确认接收后并发调用 Stop。C 无论已进入 Enqueue 等待，还是随后观察
    // 到停止状态，都必须拒绝；Stop 则继续等待 A/B 排空。
    std::future<SchedulerControlResult> stop_future =
        std::async(std::launch::async, [&]() {
            return scheduler->Stop();
        });

    if (!enqueue_probe.WaitUntilObserved(2, false)) {
        running_gate.Release();
        batch_future.wait();
        stop_future.wait();
        FAIL() << "C was not rejected after concurrent Stop";
    }

    running_gate.Release();
    const BatchInferenceResult batch_result = batch_future.get();
    const SchedulerControlResult stop_result = stop_future.get();

    ASSERT_TRUE(stop_result.success) << stop_result.error_message;
    EXPECT_FALSE(batch_result.success);
    EXPECT_EQ(batch_result.status, InferenceStatus::kPartialFailure);
    ASSERT_EQ(batch_result.results.size(), 3u);
    EXPECT_TRUE(batch_result.results[0].success);
    EXPECT_TRUE(batch_result.results[1].success);
    EXPECT_FALSE(batch_result.results[2].success);
    EXPECT_EQ(batch_result.results[2].status,
              InferenceStatus::kEngineStopping);
    EXPECT_TRUE(batch_result.results[2].value.pred.empty());
    EXPECT_EQ(execution_counts[0].load(), 1);
    EXPECT_EQ(execution_counts[1].load(), 1);
    EXPECT_EQ(execution_counts[2].load(), 0);

    EXPECT_EQ(batch_result.summary.accepted_count, 2u);
    EXPECT_EQ(batch_result.summary.success_count, 2u);
    EXPECT_EQ(batch_result.summary.failure_count, 1u);
    EXPECT_EQ(batch_result.summary.stopping_rejected_count, 1u);
    EXPECT_TRUE(scheduler->Stop().success);
}

// ============================================================================
// 两个并发 InferBatch 可以共享同一运行周期，但结果和 metadata 不能互相串位
// ============================================================================
TEST(InferenceSchedulerTest, ConcurrentBatchesKeepIndependentResults) {
    CountedGate executor_gate(2);

    auto executor = [&](const ModelRuntimeSnapshot& snapshot,
                        const NcnnWav2LipInput& input,
                        const InferenceOptions&) {
        executor_gate.ArriveAndWait();
        return MakeSyntheticSuccess(
            snapshot,
            input,
            static_cast<float>(*input.metadata.frame_index));
    };
    std::unique_ptr<InferenceScheduler> scheduler =
        detail::InferenceSchedulerTestAccess::Create(executor);

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 16;
    snapshot.effective_num_threads = 1;

    SchedulerConfig config;
    config.worker_count = 2;
    config.queue_capacity = 2;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = true;
    ASSERT_TRUE(scheduler->Start(snapshot, config).success);

    NcnnWav2LipInput first_input;
    first_input.metadata.frame_index = 101;
    NcnnWav2LipInput second_input;
    second_input.metadata.frame_index = 202;

    std::future<BatchInferenceResult> first_future =
        std::async(std::launch::async, [&]() {
            return scheduler->InferBatch({first_input});
        });
    std::future<BatchInferenceResult> second_future =
        std::async(std::launch::async, [&]() {
            return scheduler->InferBatch({second_input});
        });

    if (!executor_gate.WaitUntilAllArrived()) {
        executor_gate.Release();
        first_future.wait();
        second_future.wait();
        scheduler->Stop();
        FAIL() << "two InferBatch calls did not enter executor concurrently";
    }
    executor_gate.Release();

    const BatchInferenceResult first_result = first_future.get();
    const BatchInferenceResult second_result = second_future.get();

    ASSERT_TRUE(first_result.success) << first_result.error_message;
    ASSERT_TRUE(second_result.success) << second_result.error_message;
    ASSERT_EQ(first_result.results.size(), 1u);
    ASSERT_EQ(second_result.results.size(), 1u);
    ASSERT_TRUE(first_result.results[0].value.metadata.frame_index.has_value());
    ASSERT_TRUE(second_result.results[0].value.metadata.frame_index.has_value());
    EXPECT_EQ(*first_result.results[0].value.metadata.frame_index, 101);
    EXPECT_EQ(*second_result.results[0].value.metadata.frame_index, 202);
    EXPECT_EQ(first_result.model_generation, snapshot.generation);
    EXPECT_EQ(second_result.model_generation, snapshot.generation);
    EXPECT_FLOAT_EQ(
        static_cast<const float*>(first_result.results[0].value.pred)[0],
        101.0f);
    EXPECT_FLOAT_EQ(
        static_cast<const float*>(second_result.results[0].value.pred)[0],
        202.0f);
    EXPECT_TRUE(scheduler->Stop().success);
}

// ============================================================================
// 默认拒绝超出硬件线程预算的配置；调用方显式允许后才可以启动
// ============================================================================
TEST(InferenceSchedulerTest, RequiresExplicitOptInForThreadOversubscription) {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) {
        GTEST_SKIP() << "hardware_concurrency is unknown on this platform";
    }

    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 17;
    snapshot.effective_num_threads =
        static_cast<int>(hardware_threads) + 1;

    SchedulerConfig config;
    config.worker_count = 1;
    config.queue_capacity = 1;
    config.queue_wait_timeout = 1s;
    config.allow_thread_oversubscription = false;

    InferenceScheduler scheduler;
    const SchedulerControlResult rejected =
        scheduler.Start(snapshot, config);
    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected.status, InferenceStatus::kThreadBudgetExceeded);

    // 同一配置只有在调用方明确接受超配风险后才允许启动。
    config.allow_thread_oversubscription = true;
    const SchedulerControlResult accepted =
        scheduler.Start(snapshot, config);
    ASSERT_TRUE(accepted.success) << accepted.error_message;
    EXPECT_EQ(accepted.status, InferenceStatus::kOk);
    EXPECT_TRUE(scheduler.Stop().success);
}
