/// @file model_input_test.cpp
/// @brief 演示 Wav2LipInputBuilder + NcnnInputAdapter：合成输入 → 查看转换结果
/// @note  生成 golden_output/15_model_audio_tensor_info.json 和 16_model_image_tensor_info.json

#include "model/input_processor.h"
#include "model/ncnn_input_adapter.h"

#include <opencv2/core.hpp>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>

// 写简单 JSON（不引入第三方库）。
// 返回 bool 是 example 的验收边界：文件没真正落盘时，main 必须返回非零，不能只打印错误后假装成功。
static bool WriteJson(const std::string& path, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);
    if (ec) {
        std::fprintf(stderr, "ERROR: cannot create output directory: %s\n",
                     ec.message().c_str());
        return false;
    }

    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
        return false;
    }
    f << content;
    if (!f) {
        std::fprintf(stderr, "ERROR: failed while writing %s\n", path.c_str());
        return false;
    }
    std::printf("  -> %s\n", path.c_str());
    return true;
}

int main() {
    std::printf("=== Model Input Example ===\n\n");

    // ---- 1. 构造合成人脸 ----
    std::printf("[1] Creating synthetic face (96x96 BGR)...\n");
    cv::Mat face(96, 96, CV_8UC3);
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            face.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 255 / 95),
                static_cast<uchar>(y * 255 / 95),
                static_cast<uchar>(128));
        }
    }

    // ---- 2. 构造合成 Mel ----
    std::printf("[2] Creating bounded synthetic Mel (freq-major, [-4,4])...\n");
    std::vector<float> mel(1280);
    for (int f = 0; f < 80; ++f) {
        for (int t = 0; t < 16; ++t) {
            const int index = f * 16 + t;
            mel[index] = -4.0f + 8.0f * static_cast<float>(index) / 1279.0f;
        }
    }

    // ---- 3. Builder 构建 ----
    std::printf("[3] Building Wav2LipInputData...\n");
    digital_human::model::ModelInputMetadata meta;
    meta.pts_ms = 0;
    meta.frame_index = 0;

    digital_human::model::Wav2LipInputBuilder builder;
    auto r = builder.Build(face, mel, meta);

    if (!r.success) {
        std::fprintf(stderr, "FAILED: %s\n", r.error_message.c_str());
        return 1;
    }

    std::printf("    face: c=%d, range=[%.4f, %.4f]\n",
                r.info.face_channels, r.info.face_min_value, r.info.face_max_value);
    std::printf("    mel:  bins=%d, frames=%d, range=[%.1f, %.1f]\n",
                r.info.mel_bins, r.info.mel_frames,
                r.info.mel_min_value, r.info.mel_max_value);

    // ---- 4. Adapter 映射 ----
    // example 必须真的调用 Adapter，才能证明 Builder 的 float buffer 已按 ncnn w/h/c 落盘。
    std::printf("[4] Adapting float buffers to ncnn::Mat...\n");
    digital_human::model::NcnnInputAdapter adapter;
    auto adapted = adapter.Adapt(r.data);
    if (!adapted.success) {
        std::fprintf(stderr, "FAILED: %s\n", adapted.error_message.c_str());
        return 1;
    }
    if (adapted.input.face.w != 96 || adapted.input.face.h != 96 ||
        adapted.input.face.c != 6 || adapted.input.mel.w != 16 ||
        adapted.input.mel.h != 80 || adapted.input.mel.c != 1) {
        std::fprintf(stderr, "FAILED: unexpected ncnn tensor shape\n");
        return 1;
    }

    // 指定位置值用于验证 freq-major 与 CHW 映射，而不只验证 shape。
    const float expected_mel = mel[7 * 16 + 3];
    if (std::fabs(adapted.input.mel.row(7)[3] - expected_mel) > 1e-6f) {
        std::fprintf(stderr, "FAILED: Mel layout mismatch\n");
        return 1;
    }

    // ---- 5. 写 golden 产物 ----
    std::printf("\n[5] Writing golden outputs...\n");
    bool outputs_written = true;

    {
        // 15_model_audio_tensor_info.json
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\n"
            "  \"description\": \"Wav2Lip audio (Mel) input tensor info\",\n"
            "  \"bins\": %d,\n"
            "  \"frames\": %d,\n"
            "  \"chunk_size\": %d,\n"
            "  \"ncnn_shape\": \"w=%d,h=%d,c=%d\",\n"
            "  \"layout\": \"freq-major\",\n"
            "  \"value_range\": [%.4f, %.4f],\n"
            "  \"has_nan_or_inf\": false\n"
            "}\n",
            r.info.mel_bins, r.info.mel_frames,
            static_cast<int>(r.data.mel_freq_time.size()),
            adapted.input.mel.w, adapted.input.mel.h, adapted.input.mel.c,
            r.info.mel_min_value, r.info.mel_max_value);
        outputs_written =
            WriteJson("golden_output/15_model_audio_tensor_info.json", buf) &&
            outputs_written;
    }

    {
        // 16_model_image_tensor_info.json
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\n"
            "  \"description\": \"Wav2Lip face (image) input tensor info\",\n"
            "  \"channels\": %d,\n"
            "  \"height\": %d,\n"
            "  \"width\": %d,\n"
            "  \"ncnn_shape\": \"w=%d,h=%d,c=%d\",\n"
            "  \"color_order\": \"BGR\",\n"
            "  \"channel_order\": \"masked(0-2) + original(3-5)\",\n"
            "  \"normalization\": \"BGR / 255.0 -> [0, 1]\",\n"
            "  \"value_range\": [%.4f, %.4f],\n"
            "  \"mask_start_row\": %d,\n"
            "  \"has_nan_or_inf\": false\n"
            "}\n",
            r.info.face_channels, r.info.face_height, r.info.face_width,
            adapted.input.face.w, adapted.input.face.h, adapted.input.face.c,
            r.info.face_min_value, r.info.face_max_value,
            r.info.mask_start_row);
        outputs_written =
            WriteJson("golden_output/16_model_image_tensor_info.json", buf) &&
            outputs_written;
    }

    if (!outputs_written) {
        return 1;
    }

    std::printf("\nDone.\n");
    return 0;
}
