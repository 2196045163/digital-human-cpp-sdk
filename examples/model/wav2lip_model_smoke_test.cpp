/// @file wav2lip_model_smoke_test.cpp
/// @brief 全链路 smoke test：真实模型 + 合成输入 → 验证 pred + 写 golden 产物
/// @note  本测试证明模型边界接通，不验证口型视觉质量。

#include "model/model_loader.h"
#include "model/input_processor.h"
#include "model/ncnn_input_adapter.h"

#include <opencv2/core.hpp>
#include <net.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <filesystem>

// 写简单 JSON。Golden 证据写失败时返回 false，让 main 以非零退出。
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

int main(int argc, char* argv[]) {
    const char* model_path = (argc > 1) ? argv[1] : "models/wav2lip/wav2lip.param";
    std::printf("=== Wav2Lip Model Smoke Test ===\n");
    std::printf("Model: %s\n", model_path);

    // ---- 1. 加载模型 ----
    std::printf("[1/5] Loading model...\n");
    digital_human::model::ModelLoader loader;
    digital_human::model::ModelLoadOptions load_opts;
    load_opts.backend = digital_human::model::ModelBackend::kCpu;
    load_opts.enable_warmup = true;

    auto load_r = loader.Load(model_path, load_opts);
    if (!load_r.success) {
        std::fprintf(stderr, "FAILED: %s\n", load_r.error_message.c_str());
        return 1;
    }
    std::printf("      OK (%.2f ms, warmup=%s)\n",
                load_r.time_ms, load_r.info.warmup_performed ? "yes" : "no");
    auto model = loader.AcquireModel();
    if (!model) {
        std::fprintf(stderr, "FAILED: model snapshot is empty after successful load\n");
        return 1;
    }

    // ---- 2. 构造合成输入 ----
    std::printf("[2/5] Building synthetic input...\n");
    cv::Mat face(96, 96, CV_8UC3);
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 96; ++x)
            face.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 255 / 95),
                static_cast<uchar>(y * 255 / 95), 128);

    // 真实模型 smoke 必须使用上游契约内的 [-4,4] 值域。
    // 1000*f+t 仍保留在 Adapter 单元测试中，用于锁定布局，但不应喂给模型。
    std::vector<float> mel(1280);
    for (int f = 0; f < 80; ++f) {
        for (int t = 0; t < 16; ++t) {
            const int index = f * 16 + t;
            mel[index] = -4.0f + 8.0f * static_cast<float>(index) / 1279.0f;
        }
    }

    digital_human::model::ModelInputMetadata meta;
    meta.pts_ms = 0;
    meta.frame_index = 0;

    digital_human::model::Wav2LipInputBuilder builder;
    auto build_r = builder.Build(face, mel, meta);
    if (!build_r.success) {
        std::fprintf(stderr, "FAILED: %s\n", build_r.error_message.c_str());
        return 1;
    }
    if (build_r.info.mel_min_value < -4.0f ||
        build_r.info.mel_max_value > 4.0f) {
        std::fprintf(stderr, "FAILED: Mel values are outside [-4,4]\n");
        return 1;
    }
    std::printf("      OK\n");

    // ---- 3. 适配 ncnn::Mat ----
    std::printf("[3/5] Adapting to ncnn::Mat...\n");
    digital_human::model::NcnnInputAdapter adapter;
    auto adapt_r = adapter.Adapt(build_r.data);
    if (!adapt_r.success) {
        std::fprintf(stderr, "FAILED: %s\n", adapt_r.error_message.c_str());
        return 1;
    }
    std::printf("      OK\n");

    // ---- 4. 推理 ----
    std::printf("[4/5] Running inference...\n");
    ncnn::Extractor ex = model->create_extractor();
    ex.set_light_mode(true);
    if (ex.input("mel", adapt_r.input.mel) != 0 ||
        ex.input("face", adapt_r.input.face) != 0) {
        std::fprintf(stderr, "FAILED: Extractor::input error\n");
        return 1;
    }
    ncnn::Mat pred;
    if (ex.extract("pred", pred) != 0 || pred.empty()) {
        std::fprintf(stderr, "FAILED: Extractor::extract error\n");
        return 1;
    }
    std::printf("      OK (pred w=%d h=%d c=%d)\n", pred.w, pred.h, pred.c);

    // ---- 5. 统计 pred + 写 golden 产物 ----
    std::printf("[5/5] Verifying pred + writing golden outputs...\n");

    float pmin = 1e30f, pmax = -1e30f;
    bool all_finite = true;
    for (int c = 0; c < pred.c && all_finite; ++c) {
        const float* ch = pred.channel(c);
        for (int i = 0; i < pred.w * pred.h; ++i) {
            if (!std::isfinite(ch[i])) { all_finite = false; break; }
            if (ch[i] < pmin) { pmin = ch[i]; }
            if (ch[i] > pmax) { pmax = ch[i]; }
        }
    }

    if (!all_finite) {
        std::fprintf(stderr, "FAILED: pred contains NaN or Inf\n");
        return 1;
    }
    if (pred.w != 96 || pred.h != 96 || pred.c != 3) {
        std::fprintf(stderr, "FAILED: unexpected pred shape\n");
        return 1;
    }

    bool outputs_written = true;

    // 17_model_sync_info.json — 帧同步元数据
    {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\n"
            "  \"description\": \"Frame sync metadata for model input\",\n"
            "  \"pts_ms\": %ld,\n"
            "  \"frame_index\": %ld,\n"
            "  \"face_size\": \"96x96 BGR\",\n"
            "  \"mel_size\": \"80x16 freq-major\"\n"
            "}\n",
            meta.pts_ms.value_or(-1), meta.frame_index.value_or(-1));
        outputs_written =
            WriteJson("golden_output/17_model_sync_info.json", buf) &&
            outputs_written;
    }

    // model_smoke_pred_info.json — pred 验证证据
    {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\n"
            "  \"description\": \"Model smoke test pred verification\",\n"
            "  \"model_file\": \"%s\",\n"
            "  \"backend\": \"CPU\",\n"
            "  \"load_time_ms\": %.2f,\n"
            "  \"warmup_performed\": true,\n"
            "  \"mel_range\": [%.6f, %.6f],\n"
            "  \"pred_shape\": \"%dx%dx%d\",\n"
            "  \"pred_range\": [%.6f, %.6f],\n"
            "  \"all_finite\": true,\n"
            "  \"note\": \"Proves model boundary connectivity, NOT visual quality\"\n"
            "}\n",
            model_path, load_r.time_ms,
            build_r.info.mel_min_value, build_r.info.mel_max_value,
            pred.w, pred.h, pred.c, pmin, pmax);
        outputs_written =
            WriteJson("golden_output/model_smoke_pred_info.json", buf) &&
            outputs_written;
    }

    if (!outputs_written) {
        return 1;
    }

    std::printf("\n=== SMOKE TEST PASSED ===\n");
    return 0;
}
