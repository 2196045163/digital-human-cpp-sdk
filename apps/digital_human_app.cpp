/// @file digital_human_app.cpp
/// @brief M03 离线 CLI 入口：参数映射 → Pipeline + FinalMediaWriter → 机器可读 JSON
///
/// 用法：
///   digital_human_app --image <path> --audio <path> --model-param <path> \
///       --landmark <path> --output <path> [--model-bin <path>] [--fps <int>] \
///       [--mode offline|realtime]
///
/// 职责边界（仅编排，不复刻 Pipeline/推理/渲染/音频处理逻辑）：
/// - 解析 CLI 参数并校验
/// - 加载音频 PCM 供 FinalMediaWriter 编码
/// - 创建 PipelineConfig + FinalMediaWriter
/// - 启动 Pipeline 并等待终态
/// - 输出单个机器可读 JSON，退出码与 JSON status 一致

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "audio/audio_loader.h"
#include "output/final_media_writer.h"
#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_types.h"

namespace fs = std::filesystem;
using namespace digital_human;

// ============================================================================
// JSON 输出辅助
// ============================================================================

namespace {

/// @brief 最小 JSON 字符串转义（仅处理 " \ 和控制字符）
std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

/// @brief CLI 输出结构，最终序列化为单个 JSON 对象
struct CliResult {
    int exit_code = 0;
    std::string status;          // "success" | "error" | "realtime_not_implemented"
    std::string output_path;
    int64_t frame_count = 0;
    double duration_ms = 0.0;
    bool has_stats = false;
    pipeline::PipelineStats stats;
    std::string error_code;
    std::string error_message;
};

/// @brief 将 CliResult 输出为 pretty-printed JSON 到 stdout
void EmitJson(const CliResult& r) {
    // 使用 printf 系列以保证 flush 行为确定
    std::printf("{\n");
    std::printf("  \"schema_version\": 1,\n");
    std::printf("  \"status\": \"%s\",\n", JsonEscape(r.status).c_str());
    std::printf("  \"output_path\": \"%s\",\n", JsonEscape(r.output_path).c_str());
    std::printf("  \"frame_count\": %ld,\n", static_cast<long>(r.frame_count));
    std::printf("  \"duration_ms\": %.2f,\n", r.duration_ms);

    if (r.has_stats) {
        std::printf("  \"stats\": {\n");
        std::printf("    \"generated_task_count\": %ld,\n",
                    static_cast<long>(r.stats.generated_task_count));
        std::printf("    \"scheduler_accepted_count\": %ld,\n",
                    static_cast<long>(r.stats.scheduler_accepted_count));
        std::printf("    \"rendered_unique_frame_count\": %ld,\n",
                    static_cast<long>(r.stats.rendered_unique_frame_count));
        std::printf("    \"unique_delivered_count\": %ld,\n",
                    static_cast<long>(r.stats.unique_delivered_count));
        std::printf("    \"prepare_time_ms\": %.2f,\n",
                    r.stats.prepare_time_ms);
        std::printf("    \"audio_process_time_ms\": %.2f,\n",
                    r.stats.audio_process_time_ms);
        std::printf("    \"inference_total_time_ms\": %.2f,\n",
                    r.stats.inference_total_time_ms);
        std::printf("    \"render_total_time_ms\": %.2f,\n",
                    r.stats.render_total_time_ms);
        std::printf("    \"total_wall_time_ms\": %.2f\n",
                    r.stats.total_wall_time_ms);
        std::printf("  },\n");
    } else {
        std::printf("  \"stats\": null,\n");
    }

    if (!r.error_code.empty() || !r.error_message.empty()) {
        std::printf("  \"error\": {\n");
        std::printf("    \"code\": \"%s\",\n", JsonEscape(r.error_code).c_str());
        std::printf("    \"message\": \"%s\"\n", JsonEscape(r.error_message).c_str());
        std::printf("  }\n");
    } else {
        std::printf("  \"error\": null\n");
    }

    std::printf("}\n");
    std::fflush(stdout);
}

}  // namespace

// ============================================================================
// --help 输出
// ============================================================================

static void PrintHelp(const char* prog) {
    std::printf("Usage: %s [OPTIONS]\n\n", prog);
    std::printf("Offline digital human video generation CLI.\n");
    std::printf("Maps image, audio, and model parameters to the existing Pipeline\n");
    std::printf("and FinalMediaWriter; outputs a single machine-readable JSON result.\n\n");
    std::printf("Required:\n");
    std::printf("  --image <path>        Input face image (JPEG/PNG)\n");
    std::printf("  --audio <path>        Input audio file (WAV/MP3/FLAC/...)\n");
    std::printf("  --model-param <path>  wav2lip model param file (.param)\n");
    std::printf("  --landmark <path>     dlib 68-point landmark model (.dat)\n");
    std::printf("  --output <path>       Output MP4 file path\n\n");
    std::printf("Optional:\n");
    std::printf("  --model-bin <path>    wav2lip model bin file (.bin).\n");
    std::printf("                        Derived from --model-param if omitted\n");
    std::printf("                        (replaces .param with .bin).\n");
    std::printf("  --fps <int>           Output frame rate (1-120, default: 25)\n");
    std::printf("  --mode <mode>         Pipeline mode: 'offline' (default) or 'realtime'.\n");
    std::printf("                        'realtime' is explicitly rejected with exit code 3.\n");
    std::printf("  --help                Show this help message and exit\n\n");
    std::printf("Exit codes:\n");
    std::printf("  0  Success — MP4 written, JSON status \"success\"\n");
    std::printf("  1  Pipeline execution error — JSON status \"error\"\n");
    std::printf("  2  Invalid arguments / missing required parameters — JSON status \"error\"\n");
    std::printf("  3  Real-time mode requested — JSON status \"realtime_not_implemented\"\n");
    std::fflush(stdout);
}

// ============================================================================
// 参数解析
// ============================================================================

struct ParsedArgs {
    std::string image_path;
    std::string audio_path;
    std::string model_param;
    std::string model_bin;
    std::string landmark_path;
    std::string output_path;
    std::string mode = "offline";
    int fps = 25;
    bool show_help = false;
    std::string error;  // 解析阶段的错误
};

static ParsedArgs ParseArgs(int argc, char* argv[]) {
    ParsedArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
            // --help 优先级最高：即使同时有其它参数也先展示帮助
        } else if (arg == "--image" && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (arg == "--audio" && i + 1 < argc) {
            args.audio_path = argv[++i];
        } else if (arg == "--model-param" && i + 1 < argc) {
            args.model_param = argv[++i];
        } else if (arg == "--model-bin" && i + 1 < argc) {
            args.model_bin = argv[++i];
        } else if (arg == "--landmark" && i + 1 < argc) {
            args.landmark_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            args.fps = std::atoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (arg.rfind("--", 0) == 0) {
            // 以 -- 开头但不认识 → 报错
            args.error = "Unknown argument: " + arg;
            return args;
        } else {
            args.error = "Unknown argument: " + arg;
            return args;
        }
    }

    return args;
}

// ============================================================================
// 校验
// ============================================================================

/// @return 空字符串表示通过；非空字符串为错误描述
static std::string ValidateArgs(const ParsedArgs& args) {
    // 必填参数
    if (args.image_path.empty())  return "Missing required argument: --image";
    if (args.audio_path.empty())  return "Missing required argument: --audio";
    if (args.model_param.empty()) return "Missing required argument: --model-param";
    if (args.landmark_path.empty()) return "Missing required argument: --landmark";
    if (args.output_path.empty()) return "Missing required argument: --output";

    // FPS 范围
    if (args.fps <= 0 || args.fps > 120)
        return "FPS must be between 1 and 120, got: " + std::to_string(args.fps);

    // mode
    if (args.mode != "offline" && args.mode != "realtime")
        return "Invalid --mode value: '" + args.mode + "'. Valid: 'offline', 'realtime'";

    return "";
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    // ---- 1. 解析参数 ----
    auto args = ParseArgs(argc, argv);

    if (!args.error.empty()) {
        CliResult r;
        r.exit_code = 2;
        r.status = "error";
        r.error_code = "InvalidArgument";
        r.error_message = args.error;
        EmitJson(r);
        return r.exit_code;
    }

    if (args.show_help) {
        PrintHelp(argv[0]);
        return 0;
    }

    // ---- 2. 业务校验 ----
    std::string validation_error = ValidateArgs(args);
    if (!validation_error.empty()) {
        CliResult r;
        r.exit_code = 2;
        r.status = "error";
        r.error_code = "InvalidArgument";
        r.error_message = validation_error;
        EmitJson(r);
        return r.exit_code;
    }

    // ---- 3. 实时模式显式拒绝 ----
    if (args.mode == "realtime") {
        CliResult r;
        r.exit_code = 3;
        r.status = "realtime_not_implemented";
        r.error_code = "RealtimeNotImplemented";
        r.error_message = "Real-time mode is not yet implemented in this version. "
                          "Use 'offline' mode or omit --mode.";
        EmitJson(r);
        return r.exit_code;
    }

    // ---- 4. 检查输入文件存在性 ----
    {
        std::vector<std::pair<std::string, std::string>> input_files = {
            {"image", args.image_path},
            {"audio", args.audio_path},
            {"model-param", args.model_param},
            {"landmark", args.landmark_path},
        };
        if (!args.model_bin.empty()) {
            input_files.push_back({"model-bin", args.model_bin});
        }

        for (const auto& [label, path] : input_files) {
            if (!fs::exists(path)) {
                CliResult r;
                r.exit_code = 1;
                r.status = "error";
                r.error_code = "FileNotFound";
                r.error_message = label + " file not found: " + path;
                EmitJson(r);
                return r.exit_code;
            }
        }
    }

    // ---- 5. 推导 model_bin（若未提供） ----
    std::string resolved_model_bin = args.model_bin;
    if (resolved_model_bin.empty()) {
        resolved_model_bin = args.model_param;
        auto pos = resolved_model_bin.rfind(".param");
        if (pos != std::string::npos) {
            resolved_model_bin.replace(pos, 6, ".bin");
        }
    }

    // ---- 6. 加载音频 PCM（供 FinalMediaWriter 编码） ----
    audio::AudioLoader audio_loader(16000);
    auto audio_result = audio_loader.LoadFromFile(args.audio_path);
    if (!audio_result.success) {
        CliResult r;
        r.exit_code = 1;
        r.status = "error";
        r.error_code = "AudioLoadFailed";
        r.error_message = audio_result.error_message;
        EmitJson(r);
        return r.exit_code;
    }

    // ---- 7. 确保输出目录存在 ----
    {
        fs::path out_path(args.output_path);
        fs::path parent = out_path.parent_path();
        if (!parent.empty() && !fs::exists(parent)) {
            std::error_code ec;
            fs::create_directories(parent, ec);
            if (ec) {
                CliResult r;
                r.exit_code = 1;
                r.status = "error";
                r.error_code = "OutputPathInvalid";
                r.error_message = "Cannot create output directory: " + parent.string()
                                  + " (" + ec.message() + ")";
                EmitJson(r);
                return r.exit_code;
            }
        }
    }

    // ---- 8. 配置 FinalMediaWriter（作为 Pipeline Sink） ----
    output::WriterConfig writer_cfg;
    writer_cfg.output_path = args.output_path;
    writer_cfg.fps_num = args.fps;
    writer_cfg.fps_den = 1;
    writer_cfg.audio = audio_result.audio;  // PCM 直接传给 writer 编码

    std::shared_ptr<output::FinalMediaWriter> writer;
    try {
        writer = std::make_shared<output::FinalMediaWriter>(writer_cfg);
    } catch (const std::exception& e) {
        CliResult r;
        r.exit_code = 1;
        r.status = "error";
        r.error_code = "WriterConfigInvalid";
        r.error_message = std::string("Failed to create FinalMediaWriter: ") + e.what();
        EmitJson(r);
        return r.exit_code;
    }

    // ---- 9. 配置 Pipeline ----
    pipeline::PipelineConfig pipeline_cfg = pipeline::PipelineConfig::OfflineDefault();
    pipeline_cfg.image_path = args.image_path;
    pipeline_cfg.audio_path = args.audio_path;
    pipeline_cfg.model_param_path = args.model_param;
    pipeline_cfg.model_bin_path = resolved_model_bin;
    pipeline_cfg.landmark_model_path = args.landmark_path;
    pipeline_cfg.fps_num = args.fps;

    // ---- 10. 启动 Pipeline 并等待终态 ----
    auto t_start = std::chrono::steady_clock::now();

    pipeline::DigitalHumanPipeline pipeline;
    auto start_result = pipeline.Start(pipeline_cfg, writer);

    if (!start_result.success) {
        auto t_end = std::chrono::steady_clock::now();
        CliResult r;
        r.exit_code = 1;
        r.status = "error";
        r.error_code = pipeline::PipelineErrorCodeToString(start_result.error_code);
        r.error_message = start_result.error_message;
        r.duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        EmitJson(r);
        return r.exit_code;
    }

    auto wait_result = pipeline.Wait();

    auto t_end = std::chrono::steady_clock::now();
    double total_duration_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ---- 11. 组装 JSON 结果 ----
    CliResult r;
    r.output_path = fs::absolute(args.output_path).string();
    r.frame_count = writer->GetWrittenFrameCount();
    r.duration_ms = total_duration_ms;
    r.has_stats = true;
    r.stats = wait_result.stats;

    if (wait_result.success) {
        r.exit_code = 0;
        r.status = "success";
        // error_code / error_message 留空 → JSON error: null
    } else {
        r.exit_code = 1;
        r.status = "error";
        r.error_code = pipeline::PipelineErrorCodeToString(wait_result.error_code);
        r.error_message = wait_result.error_message;
    }

    EmitJson(r);
    return r.exit_code;
}
