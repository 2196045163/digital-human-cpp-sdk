/// @file    audio_stream_pipeline_test.cpp
/// @brief   链路测试：AudioLoader → AudioStreamBuffer → AudioPreprocessor
/// @note    多线程模拟流式：一个线程 push，另一个线程 pull 并预处理

#include <cmath>       // std::abs
#include <filesystem>  // std::filesystem::create_directories
#include <fstream>     // std::ofstream
#include <iostream>
#include <string>
#include <thread>      // std::thread
#include <vector>

#include "audio/audio_framer.h"
#include "audio/audio_loader.h"
#include "audio/audio_preprocessor.h"
#include "audio/audio_stream_buffer.h"

using namespace digital_human::audio;

int main(int argc, char* argv[]) {
    std::cout << "=== Digital Human SDK: AudioStreamBuffer Pipeline Test ===\n\n";

    std::string audio_path = (argc >= 2) ? argv[1] : "testdata/golden/audio.wav";

    // 1. AudioLoader 加载整段音频
    AudioLoader loader;
    auto load_res = loader.LoadFromFile(audio_path);
    if (!load_res.success) {
        std::cerr << "[AudioLoader] FAIL: " << load_res.error_message << "\n";
        return -1;
    }
    std::cout << "[AudioLoader] " << load_res.info.codec_name
              << "  " << load_res.audio.pcm.size() << " samples"
              << "  " << load_res.info.source_duration_sec << " sec\n\n";

    // 2. 准备 buffer / preprocessor / framer（buffer 大一点，减少 Block 次数）
    AudioStreamBuffer buf(AudioStreamBuffer::FromDuration(16000, 2000.0));
    AudioPreprocessor preproc;
    AudioFramer framer(AudioFramer::Wav2LipDefault());
    const auto& pcm = load_res.audio.pcm;
    const size_t chunk_size = 320;
    size_t total_push = 0;
    size_t total_pull = 0;
    size_t preproc_count = 0;
    size_t frame_count = 0;

    // 3. 生产者线程：分块 push（Block 策略，满了等消费者）
    std::thread producer([&]() {
        size_t pushed = 0;
        while (pushed < pcm.size()) {
            size_t len = std::min(chunk_size, pcm.size() - pushed);
            std::vector<float> chunk(pcm.begin() + pushed, pcm.begin() + pushed + len);
            auto ps = buf.PushSamples(chunk);
            if (ps.success) {
                total_push += ps.pushed_samples;
                pushed += len;
            }
        }
        buf.Close();
    });

    // 4. 消费者线程：从 buffer pull → preprocessor → framer 流式分帧
    std::thread consumer([&]() {
        while (true) {
            auto pl = buf.PullSamples(320);
            if (!pl.success) {
                if (pl.status == AudioStreamBufferStatus::kClosed) { break; }
                continue;
            }
            total_pull += pl.pulled_samples;

            // 预处理（分块）
            auto pp_res = preproc.ProcessFrame(pl.pcm);
            if (!pp_res.success) { continue; }
            preproc_count++;

            // 流式分帧：把预处理后的 PCM 喂给 framer
            auto new_frames = framer.ProcessFrame(pp_res.pcm);
            frame_count += new_frames.size();
        }
        // 取尾部残留帧
        auto tail = framer.FlushFrames();
        frame_count += tail.size();
    });

    producer.join();
    consumer.join();

    std::cout << "[Producer] pushed: " << total_push << " samples\n";
    std::cout << "[Consumer] pulled: " << total_pull << " samples"
              << "  preprocessed: " << preproc_count << " chunks\n";
    std::cout << "  frames (streaming): " << frame_count << "\n";
    std::cout << "  pushed == pulled: " << (total_push == total_pull ? "PASS" : "FAIL") << "\n\n";

    // golden：pipeline 统计报告
    std::filesystem::create_directories("golden_output");
    {
        std::ofstream f("golden_output/audio_stream_pipeline_report.txt");
        f << "audio_file=" << audio_path << "\n";
        f << "total_samples=" << load_res.audio.pcm.size() << "\n";
        f << "chunk_size=" << chunk_size << "\n";
        f << "total_pushed=" << total_push << "\n";
        f << "total_pulled=" << total_pull << "\n";
        f << "preprocessed_frames=" << preproc_count << "\n";
        f << "pushed_equals_pulled=" << (total_push == total_pull ? "true" : "false") << "\n";
        std::cout << "[Golden] audio_stream_pipeline_report.txt saved\n";
    }

    std::cout << "[RESULT] Pipeline 测试通过\n";
    return 0;
}
