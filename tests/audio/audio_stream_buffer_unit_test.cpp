/// @file    audio_stream_buffer_unit_test.cpp
/// @brief   AudioStreamBuffer 单元测试：构造/读写/溢出/清空/错误路径
/// @note    参考 tests/audio/audio_framer_unit_test.cpp 风格

#include <algorithm>   // std::min
#include <chrono>      // std::chrono::milliseconds
#include <cmath>       // std::isnan, std::isinf
#include <iostream>
#include <string>
#include <thread>      // std::thread
#include <vector>

#include "audio/audio_stream_buffer.h"

using namespace digital_human::audio;

// ============================================================================
// 枚举可打印
// ============================================================================
inline std::ostream& operator<<(std::ostream& os, AudioStreamBufferStatus s) {
    return os << AudioStreamBuffer::StatusToString(s);
}

// ============================================================================
// 简易断言宏
// ============================================================================
static int g_fail = 0;

#define EXPECT_TRUE(cond, msg) \
    do { if (!(cond)) { std::cerr << "  FAIL: " << msg << "\n"; g_fail++; } } while(0)

#define EXPECT_FALSE(cond, msg) EXPECT_TRUE(!(cond), msg)

#define EXPECT_EQ(a, b, msg) \
    do { if ((a) != (b)) { std::cerr << "  FAIL: " << msg \
          << " (expected=" << (b) << ", actual=" << (a) << ")\n"; g_fail++; } } while(0)

// ============================================================================
int main() {
    std::cout << "=== AudioStreamBuffer Unit Tests ===\n\n";

    // -----------------------------------------------------------------------
    // 1. StatusToString — 15 状态码全部非空
    // -----------------------------------------------------------------------
    std::cout << "[1] StatusToString (15 codes) ...\n";

    std::vector<AudioStreamBufferStatus> all_status = {
        AudioStreamBufferStatus::kOk,
        AudioStreamBufferStatus::kInvalidCapacity,
        AudioStreamBufferStatus::kInvalidSampleRate,
        AudioStreamBufferStatus::kInvalidDuration,
        AudioStreamBufferStatus::kEmptyInput,
        AudioStreamBufferStatus::kInvalidPcmData,
        AudioStreamBufferStatus::kInvalidChunk,
        AudioStreamBufferStatus::kInvalidReadSize,
        AudioStreamBufferStatus::kInsufficientData,
        AudioStreamBufferStatus::kInsufficientSpace,
        AudioStreamBufferStatus::kTimeout,
        AudioStreamBufferStatus::kWouldBlock,
        AudioStreamBufferStatus::kClosed,
        AudioStreamBufferStatus::kUnsupportedOperation,
        AudioStreamBufferStatus::kInternalError,
    };
    for (auto s : all_status) {
        EXPECT_FALSE(AudioStreamBuffer::StatusToString(s).empty(), "");
    }

    // -----------------------------------------------------------------------
    // 2. 构造 — 默认容量
    // -----------------------------------------------------------------------
    std::cout << "[2] Constructor ...\n";

    AudioStreamBuffer buf(8);   // 容量 8 样本
    EXPECT_EQ(buf.Capacity(), size_t(8), "capacity = 8");
    EXPECT_EQ(buf.Size(), size_t(0), "init size = 0");
    EXPECT_EQ(buf.FreeSpace(), size_t(8), "free = 8");
    EXPECT_TRUE(buf.Occupancy() < 0.001, "occupancy ≈ 0");

    // FromDuration
    auto opts = AudioStreamBuffer::FromDuration(16000, 500.0);  // 500ms
    EXPECT_EQ(opts.capacity_samples, size_t(8000), "FromDuration: 500ms@16k = 8000");

    // -----------------------------------------------------------------------
    // 3. Push + Pull — 正常读写
    // -----------------------------------------------------------------------
    std::cout << "[3] Push / Pull basic ...\n";

    std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    auto ps = buf.PushSamples(data);
    EXPECT_TRUE(ps.success, "push 5: success");
    EXPECT_EQ(ps.pushed_samples, size_t(5), "pushed 5");
    EXPECT_EQ(buf.Size(), size_t(5), "size = 5");

    auto pl = buf.PullSamples(3);
    EXPECT_TRUE(pl.success, "pull 3: success");
    EXPECT_EQ(pl.pulled_samples, size_t(3), "pulled 3");
    EXPECT_EQ(pl.pcm.size(), size_t(3), "out size = 3");
    EXPECT_TRUE(std::abs(pl.pcm[0] - 0.1f) < 1e-6f, "pcm[0] = 0.1");
    EXPECT_TRUE(std::abs(pl.pcm[2] - 0.3f) < 1e-6f, "pcm[2] = 0.3");
    EXPECT_EQ(buf.Size(), size_t(2), "size = 2 after pull");

    // -----------------------------------------------------------------------
    // 4. Push 填满后 Pull 清空
    // -----------------------------------------------------------------------
    std::cout << "[4] Fill and drain ...\n";

    buf.Clear();
    std::vector<float> fill8 = {1,2,3,4,5,6,7,8};
    ps = buf.PushSamples(fill8);
    EXPECT_TRUE(ps.success, "fill: success");
    EXPECT_EQ(buf.Size(), size_t(8), "full: size = 8");
    EXPECT_EQ(buf.FreeSpace(), size_t(0), "full: free = 0");

    pl = buf.PullSamples(8);
    EXPECT_TRUE(pl.success, "drain: success");
    EXPECT_EQ(buf.Size(), size_t(0), "empty: size = 0");
    EXPECT_EQ(pl.pulled_samples, size_t(8), "pulled 8");

    // -----------------------------------------------------------------------
    // 5. 环形回绕 — 写入触发 wrap
    // -----------------------------------------------------------------------
    std::cout << "[5] Wrap-around ...\n";

    buf.Clear();
    // 先写 5，读 3 → size=2, read=3, write=5
    buf.PushSamples({1,2,3,4,5});
    buf.PullSamples(3);
    // 再写 6 → 会触发回绕（write=5, 只够到 7, 剩余 3 个从 0 开始）
    ps = buf.PushSamples({10,20,30,40,50,60});
    EXPECT_TRUE(ps.success, "wrap push: success");
    EXPECT_EQ(ps.pushed_samples, size_t(6), "wrapped pushed 6");
    EXPECT_EQ(buf.Size(), size_t(8), "size = 8 after wrap");
    // 读出来顺序应对
    pl = buf.PullSamples(2);
    EXPECT_TRUE(std::abs(pl.pcm[0] - 4.0f) < 1e-6f, "wrapped: [0] = 4 (old data)");
    EXPECT_TRUE(std::abs(pl.pcm[1] - 5.0f) < 1e-6f, "wrapped: [1] = 5");

    // -----------------------------------------------------------------------
    // 6. DropNewest — 空间不足，丢弃新数据
    // -----------------------------------------------------------------------
    std::cout << "[6] DropNewest ...\n";

    buf.Clear();
    buf.PushSamples({1,2,3,4,5});  // size=5
    ps = buf.PushSamples({6,7,8,9}, AudioBufferOverflowStrategy::kDropNewest);
    EXPECT_FALSE(ps.success, "drop: fail");
    EXPECT_EQ(ps.dropped_samples, size_t(4), "dropped 4");
    EXPECT_EQ(buf.Size(), size_t(5), "size unchanged");
    EXPECT_EQ(ps.status, AudioStreamBufferStatus::kInsufficientSpace, "kInsufficientSpace");

    // -----------------------------------------------------------------------
    // 7. OverwriteOldest — 覆盖旧数据
    // -----------------------------------------------------------------------
    std::cout << "[7] OverwriteOldest ...\n";

    buf.Clear();
    buf.PushSamples({1,2,3,4,5});  // size=5
    ps = buf.PushSamples({6,7,8,9,10,11,12}, AudioBufferOverflowStrategy::kOverwriteOldest);
    EXPECT_TRUE(ps.success, "overwrite: success");
    // 空闲=3, 要写 7, needed=4, 覆盖 4 个旧数据
    EXPECT_EQ(ps.overwritten_samples, size_t(4), "overwritten 4");
    EXPECT_EQ(buf.Size(), size_t(8), "size = 8 after overwrite");

    // 读出验证：最旧的 4 个已被挤掉，剩余应是 [5,6,7,8,9,10,11,12]
    pl = buf.PullSamples(8);
    EXPECT_TRUE(std::abs(pl.pcm[0] - 5.0f) < 1e-6f,  "overwrite: [0] = 5");
    EXPECT_TRUE(std::abs(pl.pcm[7] - 12.0f) < 1e-6f, "overwrite: [7] = 12");

    // -----------------------------------------------------------------------
    // 8. Block — 非阻塞返回 WouldBlock
    // -----------------------------------------------------------------------
    std::cout << "[8] Block (non-blocking) ...\n";

    buf.Clear();
    buf.PushSamples({1,2,3,4,5,6,7});  // size=7, free=1
    ps = buf.PushSamples({8,9}, AudioBufferOverflowStrategy::kBlock, -1);
    EXPECT_FALSE(ps.success, "block: fail");
    EXPECT_EQ(ps.status, AudioStreamBufferStatus::kWouldBlock, "kWouldBlock");

    // -----------------------------------------------------------------------
    // 9. Pull 数据不足 / 关闭后读
    // -----------------------------------------------------------------------
    std::cout << "[9] Pull: insufficient / closed ...\n";

    buf.Clear();
    buf.PushSamples({1,2,3});
    pl = buf.PullSamples(5, -1);
    EXPECT_FALSE(pl.success, "pull 5 from size=3: fail");
    EXPECT_EQ(pl.status, AudioStreamBufferStatus::kInsufficientData, "kInsufficientData");

    buf.Close();
    pl = buf.PullSamples(2);  // 关闭了但还有数据，能读
    EXPECT_TRUE(pl.success, "closed + has data: ok");
    pl = buf.PullSamples(1);
    EXPECT_TRUE(pl.success, "closed + has last: ok");
    pl = buf.PullSamples(1);  // 已空且关闭
    EXPECT_FALSE(pl.success, "closed + empty: fail");
    EXPECT_EQ(pl.status, AudioStreamBufferStatus::kClosed, "kClosed");

    // -----------------------------------------------------------------------
    // 10. 错误路径 — 空输入 / NaN / 读大小=0
    // -----------------------------------------------------------------------
    std::cout << "[10] Error paths ...\n";

    AudioStreamBuffer buf2(8);
    ps = buf2.PushSamples({});
    EXPECT_FALSE(ps.success, "empty input: fail");
    EXPECT_EQ(ps.status, AudioStreamBufferStatus::kEmptyInput, "kEmptyInput");

    std::vector<float> nan_data{0.1f, std::nanf("")};
    ps = buf2.PushSamples(nan_data);
    EXPECT_FALSE(ps.success, "NaN: fail");
    EXPECT_EQ(ps.status, AudioStreamBufferStatus::kInvalidPcmData, "kInvalidPcmData");

    pl = buf2.PullSamples(0);
    EXPECT_FALSE(pl.success, "read size 0: fail");
    EXPECT_EQ(pl.status, AudioStreamBufferStatus::kInvalidReadSize, "kInvalidReadSize");

    // -----------------------------------------------------------------------
    // 11. Clear / Reset
    // -----------------------------------------------------------------------
    std::cout << "[11] Clear / Reset ...\n";

    buf2.PushSamples({1,2,3,4});
    buf2.Clear();
    EXPECT_EQ(buf2.Size(), size_t(0), "clear: size = 0");

    buf2.PushSamples({5,6,7,8,9,10,11,12});  // 满
    buf2.Reset();
    EXPECT_EQ(buf2.Size(), size_t(0), "reset: size = 0");
    auto stats = buf2.GetStats();
    EXPECT_EQ(stats.total_dropped_samples, int64_t(0), "reset: dropped = 0");
    EXPECT_EQ(stats.total_overwritten_samples, int64_t(0), "reset: overwritten = 0");

    // 已经关闭的 buffer，clear/reset 后还能继续用吗？(clear 不重置 closed 状态)
    // 当前设计 Reset 不重置 closed，需确认是否符合预期

    // -----------------------------------------------------------------------
    // 12. GetStats
    // -----------------------------------------------------------------------
    std::cout << "[12] GetStats ...\n";

    AudioStreamBuffer buf3(100);
    buf3.PushSamples({1,2,3}, AudioBufferOverflowStrategy::kDropNewest);
    stats = buf3.GetStats();
    EXPECT_EQ(stats.capacity_samples, size_t(100), "stats: capacity");
    EXPECT_EQ(stats.size_samples, size_t(3), "stats: size");

    // -----------------------------------------------------------------------
    // 13. Move 语义
    // -----------------------------------------------------------------------
    std::cout << "[13] Move ...\n";

    AudioStreamBuffer src(8);
    src.PushSamples({1,2,3});
    AudioStreamBuffer dst = std::move(src);
    EXPECT_EQ(dst.Size(), size_t(3), "moved: size = 3");
    pl = dst.PullSamples(3);
    EXPECT_TRUE(pl.success, "moved: pull ok");

    // -----------------------------------------------------------------------
    // 14. timeout — PushSamples 超时
    // -----------------------------------------------------------------------
    std::cout << "[14] Timeout (push) ...\n";

    AudioStreamBuffer buf_to(8);
    buf_to.PushSamples({1,2,3,4,5,6,7});  // size=7, free=1
    auto ps_to = buf_to.PushSamples({8,9}, AudioBufferOverflowStrategy::kBlock, 10);
    EXPECT_FALSE(ps_to.success, "push with timeout=10ms: should fail");
    EXPECT_EQ(ps_to.status, AudioStreamBufferStatus::kTimeout, "kTimeout");

    // -----------------------------------------------------------------------
    // 15. timeout — PullSamples 超时
    // -----------------------------------------------------------------------
    std::cout << "[15] Timeout (pull) ...\n";

    AudioStreamBuffer buf_pt(8);
    auto pl_to = buf_pt.PullSamples(5, 10);
    EXPECT_FALSE(pl_to.success, "pull with timeout=10ms: should fail");
    EXPECT_EQ(pl_to.status, AudioStreamBufferStatus::kTimeout, "kTimeout (pull)");

    // -----------------------------------------------------------------------
    // 16. Close 唤醒 Block 中的线程
    // -----------------------------------------------------------------------
    std::cout << "[16] Close wakes Block ...\n";

    AudioStreamBuffer buf_cw(8);
    buf_cw.PushSamples({1,2,3,4,5,6,7});  // size=7, free=1

    // 消费者线程：push 8,9 会被 Block → 主线程 Close 唤醒
    AudioStreamPushResult blocked_result;
    std::thread closer([&]() {
        blocked_result = buf_cw.PushSamples({8,9}, AudioBufferOverflowStrategy::kBlock);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    buf_cw.Close();
    closer.join();
    EXPECT_EQ(blocked_result.status, AudioStreamBufferStatus::kClosed,
              "Block push should wake on Close -> kClosed");

    // -----------------------------------------------------------------------
    // 17. Warning callback 是否触发
    // -----------------------------------------------------------------------
    std::cout << "[17] Warning callback ...\n";

    AudioStreamBuffer buf_wc(8);
    int cb_call_count = 0;
    float cb_occupancy = 0.0f;
    size_t cb_lost = 0;
    buf_wc.SetWarningCallback([&](float occ, size_t lost) {
        cb_call_count++;
        cb_occupancy = occ;
        cb_lost = lost;
    });

    buf_wc.PushSamples({1,2,3,4,5});  // size=5
    buf_wc.PushSamples({6,7,8,9}, AudioBufferOverflowStrategy::kDropNewest);
    EXPECT_EQ(cb_call_count, 1, "DropNewest: callback called once");
    EXPECT_TRUE(cb_occupancy > 0.0f, "DropNewest: occupancy > 0");
    EXPECT_EQ(cb_lost, size_t(4), "DropNewest: lost = 4");

    buf_wc.Clear();
    buf_wc.PushSamples({1,2,3,4,5});
    buf_wc.PushSamples({6,7,8,9,10,11,12}, AudioBufferOverflowStrategy::kOverwriteOldest);
    EXPECT_EQ(cb_call_count, 2, "OverwriteOldest: callback called");

    // =======================================================================
    std::cout << "\n";
    if (g_fail == 0) {
        std::cout << "ALL PASSED.\n";
        return 0;
    } else {
        std::cerr << g_fail << " TEST(S) FAILED.\n";
        return 1;
    }
}
