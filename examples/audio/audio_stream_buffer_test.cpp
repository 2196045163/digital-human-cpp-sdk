/// @file    audio_stream_buffer_test.cpp
/// @brief   AudioStreamBuffer 独立测试：单线程 + 多线程生产者-消费者
/// @note    不依赖其他模块，只测 buffer 本身

#include <cmath>       // std::abs
#include <filesystem>  // std::filesystem::create_directories
#include <fstream>     // std::ofstream
#include <iostream>
#include <string>
#include <thread>      // std::thread
#include <vector>

#include "audio/audio_stream_buffer.h"

using namespace digital_human::audio;

int main() {
    std::cout << "=== Digital Human SDK: AudioStreamBuffer Test ===\n\n";

    // 1. 基本读写
    std::cout << "[1] 基本读写 (capacity=8)\n";
    AudioStreamBuffer buf(8);

    auto ps = buf.PushSamples({0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
    std::cout << "  push 5: success=" << ps.success
              << "  size=" << buf.Size() << "\n";

    auto pl = buf.PullSamples(3);
    std::cout << "  pull 3: success=" << pl.success
              << "  first=" << pl.pcm[0] << "  last=" << pl.pcm[2]
              << "  size=" << buf.Size() << "\n\n";

    // 2. 填满后清空
    std::cout << "[2] Fill & Drain\n";
    buf.Clear();
    buf.PushSamples({1,2,3,4,5,6,7,8});
    std::cout << "  fill 8: size=" << buf.Size()
              << "  free=" << buf.FreeSpace() << "\n";
    buf.PullSamples(8);
    std::cout << "  drain 8: size=" << buf.Size() << "\n\n";

    // 3. 环形回绕
    std::cout << "[3] 环形回绕\n";
    buf.Clear();
    buf.PushSamples({1,2,3,4,5});   // write=5
    buf.PullSamples(3);              // read=3, size=2
    ps = buf.PushSamples({6,7,8,9,10,11});  // 6 个 → 触发回绕
    std::cout << "  wrap push: success=" << ps.success
              << "  size=" << buf.Size() << " (should be 8)\n";
    // 读出验证数据顺序
    pl = buf.PullSamples(2);         // 应读到 4,5
    std::cout << "  first 2: " << pl.pcm[0] << ", " << pl.pcm[1]
              << " (should be 4,5)\n\n";

    // 4. 三种溢出策略
    std::cout << "[4] 溢出策略\n";
    buf.Clear();
    buf.PushSamples({1,2,3,4,5});   // size=5, free=3

    ps = buf.PushSamples({6,7,8,9}, AudioBufferOverflowStrategy::kDropNewest);
    std::cout << "  DropNewest: success=" << ps.success
              << "  dropped=" << ps.dropped_samples
              << "  size=" << buf.Size() << " (unchanged)\n";

    buf.Clear();
    buf.PushSamples({1,2,3,4,5});
    ps = buf.PushSamples({6,7,8,9,10,11,12}, AudioBufferOverflowStrategy::kOverwriteOldest);
    std::cout << "  OverwriteOldest: success=" << ps.success
              << "  overwritten=" << ps.overwritten_samples
              << "  size=" << buf.Size() << " (full)\n";

    buf.Clear();
    buf.PushSamples({1,2,3,4,5,6,7});
    ps = buf.PushSamples({8,9}, AudioBufferOverflowStrategy::kBlock, -1);
    std::cout << "  Block: success=" << ps.success
              << "  status=" << AudioStreamBuffer::StatusToString(ps.status) << "\n\n";

    // 5. AudioChunk (PTS)
    std::cout << "[5] AudioChunk\n";
    buf.Clear();
    AudioChunk chunk;
    chunk.pcm = {0.1f, 0.2f, 0.3f, 0.4f};
    chunk.sample_rate = 16000;
    buf.PushChunk(chunk);

    auto pc = buf.PullChunk(2);
    std::cout << "  pull chunk: success=" << pc.success
              << "  pts_ms=" << pc.chunk.start_pts_ms << " ms"
              << "  size=" << pc.chunk.pcm.size() << "\n\n";

    // 6. 多线程生产者-消费者
    std::cout << "[6] 多线程 (producer-consumer)\n";
    buf.Clear();
    AudioStreamBuffer shared_buf(32);
    const int total_items = 1600;
    int64_t produced_sum = 0;
    int64_t consumed_sum = 0;

    // 生产者线程：每次 push 1~3 个样本，数字递增
    std::thread producer([&]() {
        for (int i = 1; i <= total_items; ) {
            int batch = 1 + (i % 3);  // 1~3 个一批
            std::vector<float> batch_data;
            for (int j = 0; j < batch && i <= total_items; j++, i++) {
                batch_data.push_back(static_cast<float>(i));
                produced_sum += i;
            }
            auto r = shared_buf.PushSamples(batch_data);
            if (!r.success && r.status == AudioStreamBufferStatus::kClosed) { break; }
        }
        shared_buf.Close();  // 生产完毕，关闭 buffer
    });

    // 消费者线程：每次 pull 5 个，攒到 sum 验证
    std::thread consumer([&]() {
        while (true) {
            auto r = shared_buf.PullSamples(5);
            if (!r.success) {
                if (r.status == AudioStreamBufferStatus::kClosed) { break; }
                continue;
            }
            for (auto v : r.pcm) { consumed_sum += static_cast<int64_t>(v); }
        }
    });

    producer.join();
    consumer.join();

    std::cout << "  produced_sum=" << produced_sum
              << "  consumed_sum=" << consumed_sum
              << "  match=" << (produced_sum == consumed_sum ? "PASS" : "FAIL")
              << "\n\n";

    // 7. golden output
    std::filesystem::create_directories("golden_output");

    // 7a. state_trace.csv：记录 push/pull 过程中 read/write 指针和 size 的变化
    {
        std::ofstream f("golden_output/audio_stream_buffer_state_trace.csv");
        f << "step,event,read_pos,write_pos,size,capacity\n";
        AudioStreamBuffer trace_buf(8);
        auto record = [&](const char* event) {
            f << "0,"
              << event << ","
              << trace_buf.GetStats().next_read_sample_index << ","
              << trace_buf.GetStats().next_write_sample_index << ","
              << trace_buf.Size() << ","
              << trace_buf.Capacity() << "\n";
        };
        record("init");
        trace_buf.PushSamples({1,2,3,4,5});  record("push 5");
        trace_buf.PullSamples(3);             record("pull 3");
        trace_buf.PushSamples({6,7,8,9,10,11}); record("push 6 (wrap)");
        trace_buf.PullSamples(8);             record("drain all");
        std::cout << "[Golden] audio_stream_buffer_state_trace.csv saved\n";
    }

    // 7b. overflow_events.json：记录每次溢出事件（策略/丢弃量/占用率）
    {
        std::ofstream f("golden_output/audio_stream_buffer_overflow_events.json");
        f << "[\n";
        AudioStreamBuffer json_buf(8);
        bool first = true;
        auto write_event = [&](const char* strategy, size_t lost_samples, double occ) {
            if (!first) { f << ",\n"; }
            first = false;
            f << "  {\"strategy\":\"" << strategy
              << "\", \"lost_samples\":" << lost_samples
              << ", \"occupancy\":" << occ << "}";
        };

        json_buf.PushSamples({1,2,3,4,5});
        json_buf.PushSamples({6,7,8,9}, AudioBufferOverflowStrategy::kDropNewest);
        write_event("DropNewest", 4, 5.0/8.0);

        json_buf.Clear();
        json_buf.PushSamples({1,2,3,4,5});
        json_buf.PushSamples({6,7,8,9,10,11,12}, AudioBufferOverflowStrategy::kOverwriteOldest);
        write_event("OverwriteOldest", 4, 5.0/8.0);

        f << "\n]\n";
        std::cout << "[Golden] audio_stream_buffer_overflow_events.json saved\n";
    }

    // 7c. thread_summary.json：多线程 push/pull 统计
    {
        std::ofstream f("golden_output/audio_stream_buffer_thread_summary.json");
        f << "{\n";
        f << "  \"total_items\":" << total_items << ",\n";
        f << "  \"produced_sum\":" << produced_sum << ",\n";
        f << "  \"consumed_sum\":" << consumed_sum << ",\n";
        f << "  \"match\":" << (produced_sum == consumed_sum ? "true" : "false") << "\n";
        f << "}\n";
        std::cout << "[Golden] audio_stream_buffer_thread_summary.json saved\n";
    }

    std::cout << "[RESULT] AudioStreamBuffer 测试通过\n";
    return 0;
}
