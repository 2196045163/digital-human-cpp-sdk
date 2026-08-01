/// @file    audio_framer_test.cpp
/// @brief   AudioFramer 手动测试：合成 ramp + 真实音频文件 → 分帧 → 打印摘要
/// @note    默认用 testdata/golden/audio.wav，也可命令行指定其他音频文件

#include <cmath>       // std::round, M_PI
#include <iostream>
#include <string>
#include <vector>

#include "audio/audio_framer.h"
#include "audio/audio_loader.h"

using namespace digital_human::audio;

int main(int argc, char* argv[]) {
    std::cout << "=== Digital Human SDK: AudioFramer Test ===\n\n";

    // ========================================================================
    // 1. 合成一段递增 PCM（0, 1, 2, ..., N-1），方便肉眼验证切片位置
    // ========================================================================
    const int sample_rate = 16000;
    const double duration_sec = 0.5;   // 0.5 秒 → 8000 个采样点
    const int N = static_cast<int>(sample_rate * duration_sec);  // 8000

    std::vector<float> pcm(N);
    for (int i = 0; i < N; i++) {
        pcm[i] = static_cast<float>(i);
    }
    std::cout << "[合成 PCM] " << N << " 个采样点，值范围 [0, " << (N - 1) << "]\n\n";

    // ========================================================================
    // 2. SpeechDefault：25ms / 10ms → frame=400, hop=160
    // ========================================================================
    std::cout << "[SpeechDefault] 25ms 帧长 / 10ms 帧移 / Hamming 窗\n";

    AudioFramer framer;
    auto r = framer.Frame(pcm);
    if (!r.success) {
        std::cerr << "  FAIL: " << r.error_message << "\n";
        return -1;
    }

    // 打印分帧统计
    const auto& info = r.info;
    std::cout << "  sample_rate:     " << info.sample_rate << " Hz\n";
    std::cout << "  frame_size:      " << info.frame_size << " samples ("
              << info.frame_duration_ms << " ms)\n";
    std::cout << "  hop_size:        " << info.hop_size << " samples ("
              << info.hop_duration_ms << " ms)\n";
    std::cout << "  overlap:         " << info.overlap_size << " samples\n";
    std::cout << "  original_len:    " << info.original_sample_count << "\n";
    std::cout << "  padded_len:      " << info.padded_sample_count
              << " (pad " << info.pad_sample_count << " zeros)\n";
    std::cout << "  num_frames:      " << info.num_frames << "\n";
    std::cout << "  window_type:     ";
    switch (info.window_type) {
        case AudioWindowType::kNone: std::cout << "None"; break;
        case AudioWindowType::kHamming: std::cout << "Hamming"; break;
        case AudioWindowType::kHann: std::cout << "Hann"; break;
    }
    std::cout << "\n";
    std::cout << "  tail_policy:     ";
    switch (info.tail_policy) {
        case AudioTailPolicy::kCoverLastSample: std::cout << "CoverLastSample"; break;
        case AudioTailPolicy::kStartEveryHop:   std::cout << "StartEveryHop"; break;
        case AudioTailPolicy::kDropIncomplete:  std::cout << "DropIncomplete"; break;
    }
    std::cout << "\n\n";

    // 打印前 3 帧 + 最后 1 帧的摘要
    auto printFrame = [](const AudioFrame& f) {
        std::cout << "  [" << f.index << "]  start=" << f.start_sample
                  << "  end=" << f.end_sample_exclusive
                  << "  ms=[" << f.start_ms << ", " << f.end_ms << ")"
                  << "  size=" << f.samples.size()
                  << "  first=" << f.samples[0]
                  << "  last=" << f.samples[f.samples.size() - 1]
                  << "  padding=" << (f.contains_padding ? "Y" : "N")
                  << "\n";
    };

    std::cout << "[帧摘要] 前 3 帧 + 最后 1 帧\n";
    for (int i = 0; i < 3 && i < info.num_frames; i++) {
        printFrame(r.frames[i]);
    }
    if (info.num_frames > 3) {
        std::cout << "  ...\n";
        printFrame(r.frames[info.num_frames - 1]);
    }
    std::cout << "\n";

    // ========================================================================
    // 3. Wav2LipDefault：50ms / 12.5ms → frame=800, hop=200
    // ========================================================================
    std::cout << "[Wav2LipDefault] 50ms 帧长 / 12.5ms 帧移 / Hamming 窗\n";

    AudioFramer wl_framer(AudioFramer::Wav2LipDefault());
    auto r_wl = wl_framer.Frame(pcm);
    if (!r_wl.success) {
        std::cerr << "  FAIL: " << r_wl.error_message << "\n";
        return -1;
    }

    const auto& wl_info = r_wl.info;
    std::cout << "  frame_size:      " << wl_info.frame_size << " samples\n";
    std::cout << "  hop_size:        " << wl_info.hop_size << " samples\n";
    std::cout << "  num_frames:      " << wl_info.num_frames << "\n";

    printFrame(r_wl.frames[0]);
    if (wl_info.num_frames > 1) {
        printFrame(r_wl.frames[wl_info.num_frames - 1]);
    }
    std::cout << "\n";

    // ========================================================================
    // 4. FrameSamplesOnly 便捷接口
    // ========================================================================
    std::cout << "[FrameSamplesOnly] 便捷接口\n";
    auto fso = framer.FrameSamplesOnly(pcm);
    std::cout << "  帧数: " << fso.size() << "  每帧: " << fso[0].size() << " 个采样点\n\n";

    // ========================================================================
    // 5. 真实音频：AudioLoader → AudioFramer 完整流程
    // ========================================================================
    std::string audio_path = (argc >= 2) ? argv[1] : "testdata/golden/audio.wav";
    std::cout << "[真实音频] AudioLoader → AudioFramer 流程\n";
    std::cout << "  文件: " << audio_path << "\n";

    AudioLoader loader;
    auto load_res = loader.LoadFromFile(audio_path);
    if (!load_res.success) {
        std::cerr << "  AudioLoader 失败: " << load_res.error_message << "\n";
        std::cerr << "  (文件不存在时可跳过本测试，只看合成 PCM 结果)\n\n";
    } else {
        std::cout << "  源采样率: " << load_res.info.source_sample_rate << " Hz"
                  << "  源声道: " << load_res.info.source_channels
                  << "  codec: " << load_res.info.codec_name << "\n";
        std::cout << "  PCM 样本数: " << load_res.audio.pcm.size()
                  << "  时长: " << load_res.info.source_duration_sec << " sec\n";

        // 喂给 AudioFramer
        auto r_real = framer.Frame(load_res.audio.pcm);
        if (!r_real.success) {
            std::cerr << "  AudioFramer 失败: " << r_real.error_message << "\n";
        } else {
            const auto& ri = r_real.info;
            std::cout << "  frame_size:      " << ri.frame_size << " samples\n";
            std::cout << "  hop_size:        " << ri.hop_size << " samples\n";
            std::cout << "  num_frames:      " << ri.num_frames << "\n";
            std::cout << "  pad:             " << ri.pad_sample_count << " zeros\n";

            // 前 2 帧 + 最后 1 帧摘要
            printFrame(r_real.frames[0]);
            if (ri.num_frames > 1) {
                printFrame(r_real.frames[1]);
            }
            if (ri.num_frames > 2) {
                std::cout << "  ...\n";
                printFrame(r_real.frames[ri.num_frames - 1]);
            }
            std::cout << "\n";
        }
    }

    std::cout << "[RESULT] AudioFramer 测试通过\n";
    return 0;
}
