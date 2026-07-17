/// @file model_loader_test.cpp
/// @brief 演示 ModelLoader：加载模型、AcquireModel、旧模型保留、写 golden 产物

#include "model/model_loader.h"
#include <ncnn/net.h>
#include <cstdio>
#include <fstream>
#include <filesystem>

// Golden 文件是 example 的可复查证据；目录创建或写入失败必须传递给 main。
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
    std::printf("=== ModelLoader Example ===\n");
    std::printf("Model: %s\n\n", model_path);

    digital_human::model::ModelLoader loader;

    // ---- 1. 加载模型 ----
    std::printf("[1/3] Loading model...\n");
    digital_human::model::ModelLoadOptions opts;
    opts.backend = digital_human::model::ModelBackend::kCpu;
    opts.enable_warmup = true;

    auto r = loader.Load(model_path, opts);
    std::printf("    success:  %s\n", r.success ? "true" : "false");
    std::printf("    status:   %s\n",
                digital_human::model::ModelLoader::StatusToString(r.status).c_str());
    std::printf("    time:     %.2f ms\n", r.time_ms);
    std::printf("    warmup:   %s\n", r.info.warmup_performed ? "yes" : "no");
    std::printf("    bin:      %s\n", r.info.bin_path.string().c_str());

    if (!r.success) {
        std::fprintf(stderr, "    error: %s\n", r.error_message.c_str());
        return 1;
    }

    // ---- 2. 验证旧模型保留 ----
    std::printf("\n[2/3] Testing old model preservation...\n");
    auto bad_r = loader.Load("/nonexistent/bad.param");
    std::printf("    bad load: success=%s\n", bad_r.success ? "true" : "false");
    auto model = loader.AcquireModel();
    std::printf("    old model: %s\n", model ? "still available" : "LOST!");
    if (bad_r.success || !model) {
        std::fprintf(stderr, "FAILED: bad reload did not preserve the old model\n");
        return 1;
    }

    // ---- 3. 写 golden 产物 ----
    std::printf("\n[3/3] Writing golden output...\n");
    {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\n"
            "  \"description\": \"ModelLoader CPU load result\",\n"
            "  \"model_path\": \"%s\",\n"
            "  \"success\": true,\n"
            "  \"time_ms\": %.2f,\n"
            "  \"warmup_performed\": %s,\n"
            "  \"old_model_preserved_after_bad_load\": true\n"
            "}\n",
            model_path, r.time_ms,
            r.info.warmup_performed ? "true" : "false");
        if (!WriteJson("golden_output/model_load_info.json", buf)) {
            return 1;
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
