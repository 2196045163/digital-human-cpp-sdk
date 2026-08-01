/// @file    audio_mel_feature_extract_test.cpp
/// @brief   MelFeatureExtractor 手动测试：AudioLoader → AudioFramer → Mel → Chunk 完整管线
/// @note    默认用 testdata/golden/audio.wav，也可命令行指定其他音频文件

#include <cmath>       // std::round, M_PI
#include <filesystem>  // std::filesystem::create_directories
#include <fstream>     // std::ofstream
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>  // cv::resize, cv::COLORMAP_*
#include <opencv2/imgcodecs.hpp> // cv::imwrite

#include "audio/audio_loader.h"
#include "audio/audio_framer.h"
#include "audio/audio_mel_feature_extract.h"

using namespace digital_human::audio;

int main(int argc, char* argv[]) {
    std::cout << "=== Digital Human SDK: MelFeatureExtractor Test ===\n\n";

    std::string audio_path = (argc >= 2) ? argv[1] : "testdata/golden/audio.wav";

    // 1. AudioLoader：加载音频文件 → 16kHz / mono / float PCM
    AudioLoader loader;
    auto load_res = loader.LoadFromFile(audio_path);
    if (!load_res.success) {
        std::cerr << "[AudioLoader] FAIL: " << load_res.error_message << "\n";
        return -1;
    }
    std::cout << "[AudioLoader] " << load_res.info.codec_name
              << "  " << load_res.info.source_sample_rate << " Hz"
              << "  " << load_res.info.source_channels << "ch"
              << "  " << load_res.audio.pcm.size() << " samples\n\n";

    // 2. AudioFramer：Wav2Lip 配置 → 50ms/12.5ms → 切帧 + 加窗
    AudioFramer framer(AudioFramer::Wav2LipDefault());
    auto frame_res = framer.Frame(load_res.audio.pcm);
    if (!frame_res.success) {
        std::cerr << "[AudioFramer] FAIL: " << frame_res.error_message << "\n";
        return -1;
    }
    std::cout << "[AudioFramer] frame_size=" << frame_res.info.frame_size
              << "  hop_size=" << frame_res.info.hop_size
              << "  num_frames=" << frame_res.info.num_frames << "\n\n";

    // 3. MelFeatureExtractor：批量提取 Mel 频谱 [T, 80]
    MelFeatureExtractor mel;
    auto fso = framer.FrameSamplesOnly(load_res.audio.pcm);
    auto mel_res = mel.ExtractBatch(fso);
    if (!mel_res.success) {
        std::cerr << "[MelFeatureExtractor] FAIL: " << mel_res.error_message << "\n";
        return -1;
    }
    const auto& mi = mel_res.info;
    std::cout << "[MelFeatureExtractor] n_fft=" << mi.n_fft
              << "  n_fft_bins=" << mi.n_fft_bins
              << "  n_mels=" << mi.n_mels << "\n";
    std::cout << "  mel shape: [" << mi.rows << " x " << mi.cols << "]"
              << "  range: [" << mi.min_value << ", " << mi.max_value << "]"
              << "  NaN/Inf: " << (mi.has_nan_or_inf ? "YES (!)" : "no") << "\n";
    std::cout << "  normalize: " << (mi.normalize_mode == MelNormalizeMode::kWav2LipSymmetric
                                       ? "Wav2Lip [-4,4]" : "other") << "\n\n";

    // 4. BuildWav2LipChunks：切 16 帧 chunk → freq-major 展平 1280 float
    auto chunk_res = mel.BuildWav2LipChunks(mel_res.mel);
    if (!chunk_res.success) {
        std::cerr << "[Chunk] FAIL: " << chunk_res.error_message << "\n";
        return -1;
    }
    std::cout << "[Chunk] chunk_size=" << chunk_res.chunk_size
              << " x n_mels=" << chunk_res.n_mels
              << " = " << (chunk_res.chunk_size * chunk_res.n_mels) << " floats"
              << "  total_chunks=" << chunk_res.chunks.size()
              << "  layout=" << (chunk_res.layout == MelChunkLayout::kFreqMajor80x16
                                  ? "freq-major" : "time-major") << "\n";

    // 打印第一个 chunk 的前 5 个值做肉眼验证
    if (!chunk_res.chunks.empty()) {
        std::cout << "  first chunk values: ";
        for (int i = 0; i < 5 && i < static_cast<int>(chunk_res.chunks[0].size()); i++) {
            std::cout << chunk_res.chunks[0][i] << " ";
        }
        std::cout << "...\n";
    }
    std::cout << "\n";

    // 5. 保存 golden_output：供 pipeline_contract 对照验收录
    std::filesystem::create_directories("golden_output");
    const auto& mel_mat = mel_res.mel;

    // 09_mel_spectrogram.csv：Mel 频谱原始数值
    {
        std::ofstream csv("golden_output/09_mel_spectrogram.csv");
        for (int r = 0; r < mel_mat.rows; r++) {
            for (int c = 0; c < mel_mat.cols; c++) {
                csv << mel_mat.at<float>(r, c);
                if (c < mel_mat.cols - 1) csv << ",";
            }
            csv << "\n";
        }
        std::cout << "[Golden] 09_mel_spectrogram.csv saved  (" << mel_mat.rows
                  << " x " << mel_mat.cols << ")\n";
    }

    // 10_mel_heatmap.png：归一化到 [0,255] 并着色的热力图
    {
        cv::Mat display;
        cv::normalize(mel_mat, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(display, display, cv::COLORMAP_INFERNO);
        cv::imwrite("golden_output/10_mel_heatmap.png", display);
        std::cout << "[Golden] 10_mel_heatmap.png saved  ("
                  << display.cols << " x " << display.rows << ")\n";
    }

    // 11_mel_chunk_000.csv：第一个 chunk 的 1280 个 float
    if (!chunk_res.chunks.empty()) {
        std::ofstream chunk_csv("golden_output/11_mel_chunk_000.csv");
        const auto& c0 = chunk_res.chunks[0];
        for (size_t i = 0; i < c0.size(); i++) {
            chunk_csv << c0[i] << "\n";
        }
        std::cout << "[Golden] 11_mel_chunk_000.csv saved  (" << c0.size() << " values)\n";
    }

    std::cout << "\n[RESULT] MelFeatureExtractor 测试通过\n";
    return 0;
}
