/// @file    audio_preprocessor_test.cpp
/// @brief   AudioPreprocessor 手动测试：AudioLoader → Preprocessor → 统计信息
/// @note    默认用 testdata/golden/audio.wav，也可命令行指定其他音频文件

#include <cmath>       // std::abs
#include <filesystem>  // std::filesystem::create_directories
#include <fstream>     // std::ofstream
#include <iostream>
#include <string>
#include <vector>

#include "audio/audio_loader.h"
#include "audio/audio_preprocessor.h"

using namespace digital_human::audio;

int main(int argc, char* argv[]) {
    std::cout << "=== Digital Human SDK: AudioPreprocessor Test ===\n\n";

    std::string audio_path = (argc >= 2) ? argv[1] : "testdata/golden/audio.wav";

    // 1. AudioLoader：加载音频 → float PCM
    AudioLoader loader;
    auto load_res = loader.LoadFromFile(audio_path);
    if (!load_res.success) {
        std::cerr << "[AudioLoader] FAIL: " << load_res.error_message << "\n";
        return -1;
    }
    std::cout << "[AudioLoader] " << load_res.info.codec_name
              << "  " << load_res.info.source_sample_rate << " Hz"
              << "  " << load_res.audio.pcm.size() << " samples\n\n";

    // 2. AudioPreprocessor：默认选项 → 归一化 + 预加重（不降噪）
    AudioPreprocessor proc;
    auto r = proc.Process(load_res.audio.pcm);
    if (!r.success) {
        std::cerr << "[AudioPreprocessor] FAIL: " << r.error_message << "\n";
        return -1;
    }

    const auto& in  = r.info.input_stats;
    const auto& out = r.info.output_stats;

    std::cout << "[Input Stats]\n";
    std::cout << "  samples:   " << in.num_samples << "\n";
    std::cout << "  max_abs:   " << in.max_abs << "\n";
    std::cout << "  rms:       " << in.rms << "\n";
    std::cout << "  zero:      " << (in.zero_ratio * 100.0) << " %\n\n";

    std::cout << "[Output Stats]\n";
    std::cout << "  max_abs:   " << out.max_abs << "\n";
    std::cout << "  rms:       " << out.rms << "\n";
    std::cout << "  zero:      " << (out.zero_ratio * 100.0) << " %\n";
    std::cout << "  gain:      " << r.info.applied_gain << "\n";
    std::cout << "  normalized:" << (r.info.normalized ? "Y" : "N")
              << "  denoised:" << (r.info.denoised ? "Y" : "N")
              << "  pre_emph: " << (r.info.pre_emphasized ? "Y" : "N") << "\n\n";

    // 3. VAD：检测语音段
    std::cout << "[VAD]\n";
    auto segs = proc.DetectSpeech(load_res.audio.pcm);
    std::cout << "  segments:   " << segs.size() << "\n";
    for (size_t i = 0; i < std::min(segs.size(), size_t(5)); i++) {
        std::cout << "  [" << i << "] "
                  << segs[i].start_ms << "ms - " << segs[i].end_ms << "ms"
                  << "  (" << (segs[i].end_ms - segs[i].start_ms) << "ms)\n";
    }
    if (segs.size() > 5) { std::cout << "  ...\n"; }
    std::cout << "\n";

    // 4. 自定义选项：归一化 + 降噪 + 预加重 全开
    std::cout << "[Custom] 归一化 + 降噪 + 预加重 全开\n";
    AudioPreprocessOptions opt_full;
    opt_full.enable_denoise = true;
    opt_full.denoise_threshold_db = -40.0f;
    AudioPreprocessor proc_full(opt_full);
    auto r_full = proc_full.Process(load_res.audio.pcm);
    if (r_full.success) {
        const auto& o2 = r_full.info.output_stats;
        std::cout << "  max_abs:   " << o2.max_abs << "\n";
        std::cout << "  rms:       " << o2.rms << "\n";
        std::cout << "  gain:      " << r_full.info.applied_gain << "\n";
        std::cout << "  normalized:" << (r_full.info.normalized ? "Y" : "N")
                  << "  denoised:" << (r_full.info.denoised ? "Y" : "N")
                  << "  pre_emph: " << (r_full.info.pre_emphasized ? "Y" : "N") << "\n";
    } else {
        std::cerr << "  FAIL: " << r_full.error_message << "\n";
    }
    std::cout << "\n";

    // 5. golden_output：保存预处理前后统计对比
    std::filesystem::create_directories("golden_output");
    {
        std::ofstream f("golden_output/07_preprocess_report.txt");
        f << "input_max_abs=" << in.max_abs << "\n";
        f << "input_rms=" << in.rms << "\n";
        f << "output_max_abs=" << out.max_abs << "\n";
        f << "output_rms=" << out.rms << "\n";
        f << "applied_gain=" << r.info.applied_gain << "\n";
        f << "normalized=" << r.info.normalized << "\n";
        f << "denoised=" << r.info.denoised << "\n";
        f << "pre_emphasized=" << r.info.pre_emphasized << "\n";
        f << "vad_segments=" << segs.size() << "\n";
        std::cout << "[Golden] 07_preprocess_report.txt saved\n";
    }

    std::cout << "\n[RESULT] AudioPreprocessor 测试通过\n";
    return 0;
}
