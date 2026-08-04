#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace digital_human {
namespace model {
namespace detail {

/// @brief 有界队列提交结果；内部 Scheduler 可据此转换为公开状态码
enum class BoundedEnqueueStatus {
    kAccepted,  ///< 任务已进入队列，completion 有值
    kTimedOut,  ///< 队列持续满到 deadline，任务从未被接收
    kStopping,  ///< pool 已停止接收，任务从未被接收
    kFatalError ///< pool 发生致命错误（worker 兜底 catch），不再接收任何任务
};

/// @brief 一次任务提交的最小结果
/// @note  只有 kAccepted 携带 future；task 抛出的异常通过 future::get() 重新抛出。
struct BoundedEnqueueResult {
    BoundedEnqueueStatus status = BoundedEnqueueStatus::kStopping;
    std::optional<std::future<void>> completion;
};

/// @brief 固定 worker + 有界 FIFO 队列；Stop 拒绝新任务并排空已接收任务
/// @note  这是 src/model/detail 内部构件，不包含 ncnn、batch 结果或公开 Scheduler 状态。
//
// ============================================================================
// 并发模型与锁纪律（实现注释）
// ============================================================================
//
// 同步结构：queue_mutex_ 保护 tasks_ / accepting_ / stopping_ 三个成员，
// not_empty_ / not_full_ 是与该锁配对的条件变量。
//
// 有界队列容量与生产者等待语义：
//   - queue_capacity_ 只限制"排队中"的任务数，不包含正在执行的任务——
//     执行中的 worker 不再占用队位，所以容量不等于并发度。
//   - Enqueue 持锁在 not_full_ 上等待"accepting_ 为假（立刻放行去拿
//     kStopping）或队列有空位"，超时返回 kTimedOut（任务从未进入队列）。
//
// stop/drain 行为（Stop 与兜底 catch 共用同一模式）：
//   1) 先置 accepting_ = false：拒绝一切新任务（含正在 not_full_ 上等待的
//      producer，谓词会立刻满足而返回 kStopping）。
//   2) 再置 stopping_ = true 并 notify_all：唤醒等待中的 worker。
//   3) worker 把已接收任务排空后才退出——退出条件 stopping_ && tasks_.empty()
//      （见 WorkerLoop），保证"已接收的任务必然执行"这一契约；Stop 不等候
//      正在执行的 task，也不强制终止可能永久阻塞的用户代码。
//
// 锁纪律（违反即数据竞争或死锁）：
//   - tasks_ / accepting_ / stopping_ 严禁在 queue_mutex_ 外读写。
//   - Stop 的 join 绝不能持 queue_mutex_：worker 退出前要取锁检查
//     tasks_.empty()，Stop 若持锁 join 则两者互相等待。
//   - lifecycle_mutex_ 不保护任何数据，只把并发/重复 Stop 串行化，避免两个
//     调用者同时 join 同一个 std::thread（未定义行为）；Stop 可重复调用。
//
// 致命错误（worker 兜底 catch，正常用户异常已被 packaged_task 吞进 future，
// 能走到这里说明包装本身出了问题）：
//   - fatal_error_ 是 atomic，任意线程可读（Enqueue / HasFatalError）。
//   - fatal_error_message_ 非 atomic，只有 compare_exchange 成功的那一个
//     worker 写一次（唯一写者）；读者必须先确认 HasFatalError() 为 true——
//     此时写入已发生且不会再有第二次写，因此无数据竞争。
//   - 兜底修改 accepting_/stopping_ 时必须持有 queue_mutex_：与 Enqueue 的
//     谓词检查互斥，堵住"检查通过后把任务入队到已停止池"的竞态窗口；
//     通知放在锁外，避免唤醒者立刻阻塞在锁上。
// ============================================================================
class BoundedWorkerPool {
public:
    /// @param worker_count 固定 worker 数，必须大于 0
    /// @param queue_capacity 等待执行的最大任务数，不包含正在执行的任务，必须大于 0
    BoundedWorkerPool(std::size_t worker_count,
                      std::size_t queue_capacity)
        : queue_capacity_(queue_capacity) {
        if (worker_count == 0) {
            throw std::invalid_argument("worker_count must be greater than zero");
        }
        if (queue_capacity == 0) {
            throw std::invalid_argument("queue_capacity must be greater than zero");
        }

        workers_.reserve(worker_count);
        try {
            for (std::size_t index = 0; index < worker_count; ++index) {
                workers_.emplace_back([this]() { WorkerLoop(); });
            }
        } catch (...) {
            // 构造期间只启动了部分线程时，也必须先通知并 join，不能让 joinable
            // thread 随对象构造失败而触发 std::terminate。
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                accepting_ = false;
                stopping_ = true;
            }
            not_empty_.notify_all();
            for (std::thread& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            throw;
        }
    }

    ~BoundedWorkerPool() {
        Stop();
    }

    BoundedWorkerPool(const BoundedWorkerPool&) = delete;
    BoundedWorkerPool& operator=(const BoundedWorkerPool&) = delete;
    BoundedWorkerPool(BoundedWorkerPool&&) = delete;
    BoundedWorkerPool& operator=(BoundedWorkerPool&&) = delete;

    /// @brief 在 timeout 内等待队列空位并提交一个 void task
    /// @tparam Callable 可由 std::packaged_task<void()> 调用的对象
    /// @return accepted 时 future 有值；timeout/stopping/fatal_error 时 task 从未进入队列
    template <typename Callable>
    BoundedEnqueueResult Enqueue(
        Callable&& callable,
        std::chrono::milliseconds timeout) {
        // packaged_task 把用户 task 的返回或异常转换为 future 终态；shared_ptr
        // 使这个 move-only 状态可以安全放入 C++17 std::function 队列。
        auto packaged = std::make_shared<std::packaged_task<void()>>(
            std::forward<Callable>(callable));
        std::future<void> completion = packaged->get_future();

        std::unique_lock<std::mutex> lock(queue_mutex_);
        const bool ready = not_full_.wait_for(
            lock,
            timeout,
            [this]() {
                return !accepting_ || tasks_.size() < queue_capacity_;
            });

        if (!ready) {
            return {BoundedEnqueueStatus::kTimedOut, std::nullopt};
        }
        if (fatal_error_.load(std::memory_order_acquire)) {
            return {BoundedEnqueueStatus::kFatalError, std::nullopt};
        }
        if (!accepting_) {
            return {BoundedEnqueueStatus::kStopping, std::nullopt};
        }

        tasks_.emplace([packaged]() { (*packaged)(); });
        lock.unlock();

        // 入队使 empty -> non-empty 成为可能，唤醒一个等待任务的 worker。
        not_empty_.notify_one();
        return {
            BoundedEnqueueStatus::kAccepted,
            std::optional<std::future<void>>(std::move(completion))
        };
    }

    /// @brief 查询 pool 是否发生致命错误（worker 兜底 catch 触发）
    bool HasFatalError() const { return fatal_error_.load(std::memory_order_acquire); }

    /// @brief 获取致命错误消息（仅在 HasFatalError() 为 true 时有意义）
    std::string FatalErrorMessage() const {
        return fatal_error_message_;
    }

    /// @brief 停止接收新任务，排空已接收任务并等待所有 worker 退出
    /// @note  可顺序重复调用；不会强制终止正在执行且可能永久阻塞的用户 task。
    void Stop() {
        // 生命周期锁把并发/重复 Stop 串行化，避免两个调用者同时 join 同一线程。
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);

        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            accepting_ = false;
            stopping_ = true;
        }

        // worker 可能睡在 empty，producer 可能睡在 full；双方都必须醒来重新
        // 检查 stopping/accepting predicate。
        not_empty_.notify_all();
        not_full_.notify_all();

        // join 绝不能持有 queue_mutex_，否则 worker 无法继续排空队列并退出。
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    using Task = std::function<void()>;

    void WorkerLoop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                not_empty_.wait(
                    lock,
                    [this]() { return stopping_ || !tasks_.empty(); });

                // 排空停止：stopping 后仍先执行队列中的已接收任务；只有队列
                // 真正为空时 worker 才退出。
                if (tasks_.empty()) {
                    if (stopping_) {
                        return;
                    }
                    continue;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            // 出队已经制造空位；先让一个 producer 有机会继续提交。task 必须
            // 在 queue mutex 外执行，避免阻塞 Enqueue、其他 worker 和 Stop。
            not_full_.notify_one();
            try {
                task();
            } catch (const std::exception& e) {
                // 正常用户异常由 packaged_task 写入 future。这里做线程入口兜底：
                // 记录 pool 级致命错误、停止接收新任务并唤醒所有等待者。
                // 如果到达此处，说明 packaged_task 包装本身出现了无法传播到 future 的错误。
                bool expected = false;
                if (fatal_error_.compare_exchange_strong(expected, true)) {
                    fatal_error_message_ = std::string("BoundedWorkerPool worker 兜底异常：")
                                           + e.what();
                }
                // 必须持 queue_mutex_ 修改 accepting_/stopping_：与 Enqueue 的
                // 谓词检查互斥，堵住"检查通过后入队到已停止池"的竞态窗口；
                // 通知放在锁外，避免唤醒者立刻阻塞在锁上。
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    accepting_ = false;
                    stopping_ = true;
                }
                not_empty_.notify_all();
                not_full_.notify_all();
            } catch (...) {
                // 未知异常，同上处理
                bool expected = false;
                if (fatal_error_.compare_exchange_strong(expected, true)) {
                    fatal_error_message_ = "BoundedWorkerPool worker 兜底未知异常";
                }
                // 必须持 queue_mutex_ 修改 accepting_/stopping_：与 Enqueue 的
                // 谓词检查互斥，堵住"检查通过后入队到已停止池"的竞态窗口；
                // 通知放在锁外，避免唤醒者立刻阻塞在锁上。
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    accepting_ = false;
                    stopping_ = true;
                }
                not_empty_.notify_all();
                not_full_.notify_all();
            }
        }
    }

    // 受 queue_mutex_ 保护的成员：tasks_ / accepting_ / stopping_（严禁锁外
    // 访问）。其余成员不需要该锁：queue_capacity_ 构造后只读；fatal_error_
    // 是独立 atomic；workers_ 与 lifecycle_mutex_ 仅 Stop 访问。
    const std::size_t queue_capacity_;
    std::mutex queue_mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<Task> tasks_;
    bool accepting_ = true;
    bool stopping_ = false;

    // 兜底致命错误：worker 的 catch(...) 中记录，供 Scheduler/Pipeline 查询
    std::atomic<bool> fatal_error_{false};
    // 非 atomic：只有 compare_exchange 成功的那一个 worker 写一次（唯一写者）；
    // 读者必须先查 HasFatalError() 再读，保证 happens-before 关系。
    std::string fatal_error_message_;

    // 不保护数据，只串行化 Stop：置标志 + join 同一个 thread 必须互斥
    std::mutex lifecycle_mutex_;
    std::vector<std::thread> workers_;
};

} // namespace detail
} // namespace model
} // namespace digital_human
