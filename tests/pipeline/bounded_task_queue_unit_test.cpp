/// @file bounded_task_queue_unit_test.cpp
/// @brief BoundedTaskQueue 确定性单元测试：FIFO、阻塞、Close、Cancel、high-watermark 等

#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pipeline/detail/bounded_task_queue.h"

using namespace digital_human::pipeline::detail;

// ============================================================================
// move-only 测试任务
// ============================================================================

struct MoveOnlyTask {
    int value = 0;
    std::unique_ptr<int> ptr;

    MoveOnlyTask(int v) : value(v), ptr(std::make_unique<int>(v)) {}

    MoveOnlyTask(MoveOnlyTask&&) = default;
    MoveOnlyTask& operator=(MoveOnlyTask&&) = default;
    MoveOnlyTask(const MoveOnlyTask&) = delete;
    MoveOnlyTask& operator=(const MoveOnlyTask&) = delete;
};

// ============================================================================
// FIFO 基本测试
// ============================================================================

TEST(BoundedTaskQueueTest, FifoOrder) {
    BoundedTaskQueue<int> q(4);

    EXPECT_TRUE(q.Push(1));
    EXPECT_TRUE(q.Push(2));
    EXPECT_TRUE(q.Push(3));

    EXPECT_EQ(q.Size(), 3);

    auto v1 = q.Pop();
    auto v2 = q.Pop();
    auto v3 = q.Pop();

    ASSERT_TRUE(v1.has_value());
    ASSERT_TRUE(v2.has_value());
    ASSERT_TRUE(v3.has_value());
    EXPECT_EQ(*v1, 1);
    EXPECT_EQ(*v2, 2);
    EXPECT_EQ(*v3, 3);
}

// ============================================================================
// Move-only 支持
// ============================================================================

TEST(BoundedTaskQueueTest, MoveOnlyType) {
    BoundedTaskQueue<MoveOnlyTask> q(2);

    EXPECT_TRUE(q.Push(MoveOnlyTask(42)));
    EXPECT_TRUE(q.Push(MoveOnlyTask(99)));

    auto t1 = q.Pop();
    ASSERT_TRUE(t1.has_value());
    EXPECT_EQ(t1->value, 42);
    EXPECT_NE(t1->ptr, nullptr);
    EXPECT_EQ(*t1->ptr, 42);

    auto t2 = q.Pop();
    ASSERT_TRUE(t2.has_value());
    EXPECT_EQ(t2->value, 99);
}

// ============================================================================
// 容量限制：满队列阻塞
// ============================================================================

TEST(BoundedTaskQueueTest, CapacityEnforced) {
    BoundedTaskQueue<int> q(2);

    EXPECT_TRUE(q.Push(1));
    EXPECT_TRUE(q.Push(2));
    EXPECT_EQ(q.Size(), 2);
    EXPECT_EQ(q.HighWatermark(), 2);

    // 容量已满，第三次 push 应阻塞（用异步验证）
    std::atomic<bool> push_done{false};
    std::thread pusher([&]() {
        q.Push(3);
        push_done.store(true, std::memory_order_release);
    });

    // 短暂等待，确认 push 被阻塞
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(push_done.load(std::memory_order_acquire));

    // Pop 释放一个空位
    auto v = q.Pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 1);

    // push 应该完成
    pusher.join();
    EXPECT_TRUE(push_done.load(std::memory_order_acquire));
    EXPECT_EQ(q.Size(), 2);

    // high-watermark 应该是 2（push 3 之后才 pop 1，期间队列达到 2）
    EXPECT_GE(q.HighWatermark(), 2);
}

// ============================================================================
// 空队列阻塞等待
// ============================================================================

TEST(BoundedTaskQueueTest, EmptyQueueBlocksPop) {
    BoundedTaskQueue<int> q(2);

    std::atomic<bool> pop_done{false};
    std::thread popper([&]() {
        auto v = q.Pop();
        pop_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(pop_done.load(std::memory_order_acquire));

    // Push 来唤醒 popper
    q.Push(42);
    popper.join();
    EXPECT_TRUE(pop_done.load(std::memory_order_acquire));
}

// ============================================================================
// Close：拒绝新任务但排空已有任务
// ============================================================================

TEST(BoundedTaskQueueTest, CloseDrainsThenRejects) {
    BoundedTaskQueue<int> q(4);

    q.Push(1);
    q.Push(2);
    q.Close();

    // Close 后不能再 push
    EXPECT_FALSE(q.Push(3));
    EXPECT_TRUE(q.IsClosed());

    // 但可以排空已有的
    auto v1 = q.Pop();
    auto v2 = q.Pop();
    ASSERT_TRUE(v1.has_value());
    ASSERT_TRUE(v2.has_value());

    // 排空后 Pop 返回 nullopt
    auto v3 = q.Pop();
    EXPECT_FALSE(v3.has_value());
}

// ============================================================================
// Cancel：唤醒所有等待线程
// ============================================================================

TEST(BoundedTaskQueueTest, CancelWakesBlockedPop) {
    BoundedTaskQueue<int> q(2);

    std::atomic<bool> popped{false};
    std::thread popper([&]() {
        auto v = q.Pop();  // 阻塞在空队列
        popped.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    q.Cancel();
    popper.join();

    EXPECT_TRUE(popped.load(std::memory_order_acquire));
    EXPECT_TRUE(q.IsCancelled());
}

TEST(BoundedTaskQueueTest, CancelWakesBlockedPush) {
    BoundedTaskQueue<int> q(1);

    q.Push(1);  // 填满队列

    std::atomic<bool> pushed{false};
    std::thread pusher([&]() {
        q.Push(2);  // 阻塞在满队列
        pushed.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    q.Cancel();
    pusher.join();

    EXPECT_TRUE(pushed.load(std::memory_order_acquire));
    EXPECT_EQ(q.UnprocessedCount(), 1);  // 队列中还有 1 个未处理任务
}

// ============================================================================
// 重复 Close/Cancel 幂等
// ============================================================================

TEST(BoundedTaskQueueTest, CloseIdempotent) {
    BoundedTaskQueue<int> q(2);
    q.Close();
    q.Close();  // 不应崩溃
    EXPECT_TRUE(q.IsClosed());
    EXPECT_FALSE(q.IsCancelled());
}

TEST(BoundedTaskQueueTest, CancelIdempotent) {
    BoundedTaskQueue<int> q(2);
    q.Cancel();
    q.Cancel();  // 不应崩溃
    EXPECT_TRUE(q.IsCancelled());
}

TEST(BoundedTaskQueueTest, CancelAfterClose) {
    BoundedTaskQueue<int> q(2);
    q.Close();
    q.Cancel();  // 应该正常处理
    EXPECT_TRUE(q.IsCancelled());
}

// ============================================================================
// High-watermark 统计
// ============================================================================

TEST(BoundedTaskQueueTest, HighWatermark) {
    BoundedTaskQueue<int> q(4);

    EXPECT_EQ(q.HighWatermark(), 0);

    q.Push(1);
    EXPECT_EQ(q.HighWatermark(), 1);

    q.Push(2);
    q.Push(3);
    q.Push(4);
    EXPECT_EQ(q.HighWatermark(), 4);

    q.Pop();
    q.Push(5);
    EXPECT_EQ(q.HighWatermark(), 4);  // 不会下降
}

// ============================================================================
// TryPop 非阻塞
// ============================================================================

TEST(BoundedTaskQueueTest, TryPopNonBlocking) {
    BoundedTaskQueue<int> q(2);

    auto v = q.TryPop();
    EXPECT_FALSE(v.has_value());

    q.Push(42);
    v = q.TryPop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 42);

    v = q.TryPop();
    EXPECT_FALSE(v.has_value());
}

// ============================================================================
// 容量 1 的边界情况
// ============================================================================

TEST(BoundedTaskQueueTest, CapacityOne) {
    BoundedTaskQueue<int> q(1);

    EXPECT_TRUE(q.Push(1));
    EXPECT_EQ(q.Size(), 1);

    // 需要异步验证满队列阻塞
    std::atomic<bool> push_done{false};
    std::thread pusher([&]() {
        q.Push(2);
        push_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(push_done.load());

    auto v = q.Pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 1);

    pusher.join();
    EXPECT_TRUE(push_done.load());
}

// ============================================================================
// Cancel 后 Pop 返回 nullopt（从阻塞中醒来）
// ============================================================================

TEST(BoundedTaskQueueTest, CancelReturnsNulloptFromPop) {
    BoundedTaskQueue<MoveOnlyTask> q(2);

    std::optional<MoveOnlyTask> result;
    std::thread popper([&]() {
        result = q.Pop();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.Cancel();
    popper.join();

    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// 性能：Cancel 后阻塞 push/pop <= 2 秒返回
// ============================================================================

TEST(BoundedTaskQueueTest, CancelReturnsUnder2Seconds) {
    BoundedTaskQueue<int> q(1);
    q.Push(1);  // 填满

    auto start = std::chrono::steady_clock::now();

    std::thread pusher([&]() {
        q.Push(2);  // 阻塞
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    q.Cancel();
    pusher.join();

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    EXPECT_LT(ms, 2000) << "Cancel took " << ms << "ms, expected < 2000ms";
}
