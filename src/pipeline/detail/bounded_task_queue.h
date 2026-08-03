#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <utility>

namespace digital_human {
namespace pipeline {
namespace detail {

/// @brief Bounded FIFO queue for move-only tasks with Close/Cancel lifecycle.
///
/// Features:
/// - Bounded FIFO with blocking push on full queue
/// - Move-only task support
/// - Close: normal EOS, rejects new tasks but drains existing
/// - Cancel: error/cancel, wakes all waiters, tracks unprocessed count
/// - High-watermark statistics
/// - Idempotent Close/Cancel
///
/// @tparam T Move-constructible and move-assignable task type
template <typename T>
class BoundedTaskQueue {
public:
    /// @param capacity Maximum number of items in queue (> 0)
    explicit BoundedTaskQueue(std::size_t capacity)
        : capacity_(capacity) {
    }

    ~BoundedTaskQueue() {
        // Ensure no threads are waiting before destruction
        Cancel();
    }

    BoundedTaskQueue(const BoundedTaskQueue&) = delete;
    BoundedTaskQueue& operator=(const BoundedTaskQueue&) = delete;
    BoundedTaskQueue(BoundedTaskQueue&&) = delete;
    BoundedTaskQueue& operator=(BoundedTaskQueue&&) = delete;

    /// @brief Push a task into the queue. Blocks if queue is full.
    /// @return true if task was accepted, false if queue is closed/cancelled
    bool Push(T task) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() {
            return closed_ || cancelled_ || queue_.size() < capacity_;
        });

        if (closed_ || cancelled_) {
            return false;
        }

        queue_.push(std::move(task));
        if (queue_.size() > high_watermark_) {
            high_watermark_ = queue_.size();
        }

        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    /// @brief Try to push with timeout.
    /// @return true if accepted, false if timeout/closed/cancelled
    bool TryPushFor(T task, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool ready = not_full_.wait_for(lock, timeout, [this]() {
            return closed_ || cancelled_ || queue_.size() < capacity_;
        });

        if (!ready || closed_ || cancelled_) {
            return false;
        }

        queue_.push(std::move(task));
        if (queue_.size() > high_watermark_) {
            high_watermark_ = queue_.size();
        }

        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    /// @brief Pop a task from the queue. Blocks if empty.
    /// @return task if available, std::nullopt if closed with empty queue or cancelled
    std::optional<T> Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() {
            return cancelled_ || (closed_ && queue_.empty()) || !queue_.empty();
        });

        if (cancelled_) {
            return std::nullopt;
        }
        if (queue_.empty()) {
            // closed_ && queue_.empty() means normal drain complete
            return std::nullopt;
        }

        T task = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();
        return task;
    }

    /// @brief Non-blocking try-pop.
    /// @return task if available, std::nullopt if empty (but still open)
    std::optional<T> TryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T task = std::move(queue_.front());
        queue_.pop();
        // Wake a potential blocked pusher
        not_full_.notify_one();
        return task;
    }

    /// @brief Close the queue normally: reject new tasks but allow draining existing.
    /// Idempotent.
    void Close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || cancelled_) {
                return;
            }
            closed_ = true;
        }
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    /// @brief Cancel the queue: wake all waiters, track unprocessed count.
    /// Idempotent.
    void Cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cancelled_) {
                return;
            }
            cancelled_ = true;
            unprocessed_count_ = queue_.size();
        }
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    /// @brief Check if queue is closed (normal EOS, no new tasks accepted).
    bool IsClosed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    /// @brief Check if queue is cancelled.
    bool IsCancelled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cancelled_;
    }

    /// @brief Current queue size.
    std::size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /// @brief High-watermark (peak queue depth observed).
    std::size_t HighWatermark() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return high_watermark_;
    }

    /// @brief Number of unprocessed tasks at cancel time.
    std::size_t UnprocessedCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unprocessed_count_;
    }

    /// @brief Queue capacity.
    std::size_t Capacity() const { return capacity_; }

    /// @brief Check if queue is empty.
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    bool closed_ = false;
    bool cancelled_ = false;
    std::size_t high_watermark_ = 0;
    std::size_t unprocessed_count_ = 0;
};

}  // namespace detail
}  // namespace pipeline
}  // namespace digital_human
