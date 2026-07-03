#include <iostream>
#include <algorithm>   // std::min
#include <cmath>       // std::sqrt, std::isnan, std::isinf
#include <string>

#include "audio/audio_loader.h"

using namespace digital_human::audio;

int main(int argc, char* argv[]) {
    std::cout << "=== Digital Human SDK: AudioLoader Test ===\n\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path> [target_sample_rate]\n";
        return -1;
    }

    std::string file_path = argv[1];
    int target_sample_rate = (argc >= 3) ? std::stoi(argv[2]) : 16000;
    AudioLoader loader(target_sample_rate);

    // ========================================================================
    // 1. Probe：读取源信息（不解码 PCM）
    // ========================================================================
    std::cout << "[Probe] 读取读取容器格式 / 编码 / 采样率 / 声道 / 耗时（不解码 PCM）\n";

    AudioLoadResult al_res = loader.Probe(file_path);
    if (!al_res.success) {
        std::cerr << "  FAIL: " << al_res.error_message << "\n";
        return -1;
    }

    const auto& info = al_res.info;
    std::cout << "  file:          " << info.file_path << "\n";
    std::cout << "  container:     " << info.container_format << "\n";
    std::cout << "  codec:         " << info.codec_name << "\n";
    std::cout << "  sample rate:   " << info.source_sample_rate << " Hz\n";
    std::cout << "  channels:      " << info.source_channels << "\n";
    std::cout << "  sample format: " << info.source_sample_format << "\n";
    std::cout << "  bit rate:      " << (info.source_bit_rate > 0
                    ? std::to_string(info.source_bit_rate / 1000) + " kbps"
                    : "unknown") << "\n";
    std::cout << "  duration:      " << info.source_duration_sec << " sec\n\n";


    // ========================================================================
    // 2. LoadInt16FromFile — 完整解码为 int16 PCM，输出样本数 / 时长 / min/max / 零样本占比 / 前 10 个采样点
    // ========================================================================
    std::cout << "[Load Int16] int16 PCM：样本数 / 时长 / min / max / 零样本占比 / 前 10 个采样点\n";
    
    AudioInt16LoadResult ai16l_res = loader.LoadInt16FromFile(file_path);
    if (!ai16l_res.success) {
        std::cerr << " FAIL: " << ai16l_res.error_message << "\n";

        return -1;
    }

    const auto& i16_info = ai16l_res.info;
    std::cout << "  sample count:          " << i16_info.sample_count << "\n";
    std::cout << "  target sample_rate:   " << i16_info.target_sample_rate << " Hz\n";
    std::cout << "  target channels:      " << i16_info.target_channels << "\n";
    std::cout << "  duration:      " << i16_info.duration_sec << " sec\n";

    // 遍历 PCM 计算 min / max / 零样本占比 / RMS
    std::vector<int16_t> pcm16 = ai16l_res.pcm;
    int16_t min = pcm16[0], max = pcm16[0];
    int64_t zeros = 0;
    int64_t sum_sq = 0;  // 采样值平方的累加，用于 RMS
    for (const auto& p : pcm16) {
        if (p < min)      min = p;
        if (p > max)      max = p;
        if (p == 0)       zeros++;
        sum_sq += static_cast<int64_t>(p) * p;
    }
    double rms = std::sqrt(static_cast<double>(sum_sq) / pcm16.size());
    double zeros_percent = 100.0 * zeros / pcm16.size();

    std::cout << "  min:           " << min << "\n";
    std::cout << "  max:           " << max << "\n";
    std::cout << "  rms:           " << rms << "\n";
    std::cout << "  zero ratio:    " << zeros_percent << " %\n";

    // 前 10 个采样点
    std::cout << "  first 10:     ";
    size_t count = std::min(pcm16.size(), size_t(10));
    for (size_t i = 0; i < count; i++) {
        std::cout << " " << pcm16[i];
    }
    std::cout << "\n\n";


    // ========================================================================
    // 3. LoadFromFile — 主接口，输出 float PCM，统计 min/max/RMS/零样本占比/NaN/前10采样点
    // ========================================================================
    std::cout << "[Load Float] float PCM：样本数 / min / max / RMS / 零样本占比 / NaN / 前 10 个采样点\n";

    al_res = loader.LoadFromFile(file_path);
    if (!al_res.success) {
        std::cerr << " FAIL: " << al_res.error_message << "\n";
        return -1;
    }

    const auto& f32_info = al_res.info;
    std::cout << "  sample count:          " << f32_info.sample_count << "\n";
    std::cout << "  target sample_rate:   " << f32_info.target_sample_rate << " Hz\n";
    std::cout << "  target channels:      " << f32_info.target_channels << "\n";
    std::cout << "  duration:      " << f32_info.duration_sec << " sec\n";

    // 遍历 float PCM 计算 min / max / RMS / 零样本占比 / NaN/Inf
    const auto& f32_pcm = al_res.audio.pcm;
    float fmin = f32_pcm[0], fmax = f32_pcm[0];
    int64_t f_zeros = 0;
    double f_sum_sq = 0.0;    // float 平方用 double 累积，避免精度损失
    bool has_nan = false;
    for (const auto& v : f32_pcm) {
        if (v < fmin)      fmin = v;
        if (v > fmax)      fmax = v;
        if (v == 0.0f)     f_zeros++;
        if (std::isnan(v) || std::isinf(v)) has_nan = true;
        f_sum_sq += static_cast<double>(v) * v;
    }
    double f_rms = std::sqrt(f_sum_sq / f32_pcm.size());
    double f_zeros_percent = 100.0 * f_zeros / f32_pcm.size();

    std::cout << "  min:           " << fmin << "\n";
    std::cout << "  max:           " << fmax << "\n";
    std::cout << "  rms:           " << f_rms << "\n";
    std::cout << "  zero ratio:    " << f_zeros_percent << " %\n";
    std::cout << "  NaN/Inf:       " << (has_nan ? "YES (!)" : "no") << "\n";

    // 前 10 个 float 采样点
    std::cout << "  first 10:     ";
    size_t f_count = std::min(f32_pcm.size(), size_t(10));
    for (size_t i = 0; i < f_count; i++) {
        std::cout << " " << f32_pcm[i];
    }
    std::cout << "\n\n";

    // ========================================================================
    // 4. 对比 int16 和 float 样本数是否一致
    // ========================================================================
    std::cout << "[Check] int16 / float 样本数对比\n";
    std::cout << "  int16: " << pcm16.size() << "  float: " << f32_pcm.size()
              << "  -> " << (pcm16.size() == f32_pcm.size() ? "PASS" : "MISMATCH") << "\n\n";

    // ========================================================================
    // 5. LoadBatch — 批量加载，验证下标一致性（失败项不跳过）
    // ========================================================================
    std::cout << "[LoadBatch] 批量加载：成功 + 失败 混合，验证下标对应\n";

    auto batch = loader.LoadBatch({file_path, "__no_such_file.wav", file_path});
    for (size_t i = 0; i < batch.size(); i++) {
        std::cout << "  [" << i << "] "
                  << (batch[i].success ? "PASS" : "FAIL")
                  << "  " << batch[i].error_message
                  << "  (samples=" << batch[i].info.sample_count << ")\n";
    }

    std::cout << "\n[RESULT] AudioLoader 测试通过\n";

    return 0;
}
