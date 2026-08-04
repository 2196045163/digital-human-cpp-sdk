/// @file pipeline_performance_benchmark.cpp
/// @brief M05 正式 Pipeline 性能基准：多轮全链路采集 PipelineStats/RSS/吞吐
///
/// 用法：
///   ./pipeline_performance_benchmark [iterations] [output_dir]
///
/// 采集指标：
///   - PipelineStats（prepare/audio/inference/render/total 阶段耗时）
///   - 队列高水位（Q1/Q2）
///   - 峰值 RSS（/proc/self/status 轮询线程，~100ms）
///   - 吞吐（帧/秒）
///   - 系统环境信息（CPU、编译器、内核）
///   - 输入文件 SHA256 哈希
///
/// 输出：
///   - <output_dir>/pipeline_performance_benchmark.json — 汇总 JSON
///   - <output_dir>/pipeline_benchmark_run_N.json — 单轮详细统计
///
/// @note 性能结果仅对本次机器、构建和参数有效，不构成正确性阈值。

#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace digital_human::pipeline;

// ============================================================================
// 配置
// ============================================================================

namespace {

const fs::path kGoldenImagePath = "testdata/golden/face.jpg";
const fs::path kGoldenAudioPath = "testdata/golden/audio.wav";
const fs::path kModelParamPath = "models/wav2lip/wav2lip.param";
const fs::path kModelBinPath = "models/wav2lip/wav2lip.bin";
const fs::path kLandmarkModelPath = "models/shape_predictor_68_face_landmarks.dat";
constexpr int kDefaultIterations = 3;
constexpr std::size_t kRssPollIntervalMs = 100;
constexpr std::size_t kQueueCapacityQ1 = 4;
constexpr std::size_t kQueueCapacityQ2 = 2;
constexpr std::size_t kSchedulerWorkers = 1;
constexpr int kNcnnThreads = 1;
constexpr int kFps = 25;

}  // namespace

// ============================================================================
// RSS 轮询器
// ============================================================================

class RssPoller {
public:
    RssPoller() = default;

    void Start() {
        running_ = true;
        peak_kb_ = -1;
        poller_ = std::thread([this]() {
            while (running_) {
                long rss = ReadLinuxRssKb();
                if (rss >= 0) {
                    long current = peak_kb_.load();
                    while (rss > current &&
                           !peak_kb_.compare_exchange_weak(current, rss)) {
                    }
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kRssPollIntervalMs));
            }
        });
    }

    void Stop() {
        running_ = false;
        if (poller_.joinable()) {
            poller_.join();
        }
        // 最后再读一次
        long rss = ReadLinuxRssKb();
        if (rss >= 0) {
            long current = peak_kb_.load();
            while (rss > current &&
                   !peak_kb_.compare_exchange_weak(current, rss)) {
            }
        }
    }

    long PeakKb() const { return peak_kb_.load(); }

private:
    static long ReadLinuxRssKb() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("VmRSS:", 0) != 0) {
                continue;
            }
            std::istringstream parser(line.substr(6));
            long rss_kb = -1;
            parser >> rss_kb;
            return parser ? rss_kb : -1;
        }
        return -1;
    }

    std::atomic<bool> running_{false};
    std::atomic<long> peak_kb_{-1};
    std::thread poller_;
};

// ============================================================================
// 基准 Sink（计数 + 不写帧文件以消除 I/O 干扰）
// ============================================================================

class BenchmarkSink : public PipelineOutputSink {
public:
    void OnFrame(const PipelineFrame& frame) override {
        std::lock_guard<std::mutex> lock(mutex_);
        frame_count_++;
        frame_indices_.push_back(frame.video_frame.frame_index);
        pts_values_.push_back(frame.video_frame.pts.microseconds);
    }

    void OnTerminal(const PipelineResult& result) override {
        std::lock_guard<std::mutex> lock(mutex_);
        terminal_result_ = result;
        terminal_called_ = true;
        terminal_cv_.notify_all();
    }

    void WaitForTerminal(std::chrono::milliseconds timeout =
                             std::chrono::milliseconds(120000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        terminal_cv_.wait_for(lock, timeout,
                              [this]() { return terminal_called_; });
    }

    int frame_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frame_count_;
    }
    bool terminal_called() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return terminal_called_;
    }
    PipelineResult terminal_result() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return terminal_result_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable terminal_cv_;
    int frame_count_ = 0;
    std::vector<std::int64_t> frame_indices_;
    std::vector<std::int64_t> pts_values_;
    bool terminal_called_ = false;
    PipelineResult terminal_result_;
};

// ============================================================================
// 统计辅助
// ============================================================================

struct PerRunRecord {
    int iteration = 0;
    bool success = false;
    long rss_peak_kb = -1;
    double total_wall_time_ms = 0.0;
    PipelineStats stats;
    int frame_count = 0;
    std::string terminal_state;
    std::string error_message;
};

struct AggregateStats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
};

double NearestRankPercentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    double rank = std::ceil(p * static_cast<double>(sorted.size()));
    std::size_t idx = static_cast<std::size_t>(std::max(1.0, rank) - 1.0);
    return sorted[std::min(idx, sorted.size() - 1)];
}

AggregateStats Aggregate(std::vector<double> vals) {
    AggregateStats a;
    if (vals.empty()) return a;
    a.mean = std::accumulate(vals.begin(), vals.end(), 0.0) /
             static_cast<double>(vals.size());
    std::sort(vals.begin(), vals.end());
    a.p50 = NearestRankPercentile(vals, 0.50);
    a.p95 = NearestRankPercentile(vals, 0.95);
    a.min_val = vals.front();
    a.max_val = vals.back();
    return a;
}

// ============================================================================
// SHA256 计算（通过 popen 调用 sha256sum）
// ============================================================================

std::string ComputeSha256(const fs::path& path) {
    std::string cmd = "sha256sum " + path.string() + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "ERROR:popen";
    char buf[128];
    std::string result;
    if (fgets(buf, sizeof(buf), pipe)) {
        result = buf;
        // 只取前 64 个 hex 字符
        std::size_t space = result.find(' ');
        if (space != std::string::npos) {
            result = result.substr(0, space);
        }
    }
    pclose(pipe);
    return result.empty() ? "ERROR:sha256sum" : result;
}

// ============================================================================
// 系统信息
// ============================================================================

std::string GetCpuModel() {
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    while (std::getline(info, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 1);
                // trim leading spaces
                std::size_t start = model.find_first_not_of(" \t");
                return (start != std::string::npos) ? model.substr(start) : model;
            }
        }
    }
    return "Unknown";
}

std::string GetKernelVersion() {
    std::ifstream info("/proc/version");
    std::string line;
    if (std::getline(info, line)) return line;
    return "Unknown";
}

std::string GetCompilerInfo() {
#ifdef __GNUC__
    return std::string("GCC ") + std::to_string(__GNUC__) + "." +
           std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "Unknown";
#endif
}

std::string BuildMode() {
#ifdef NDEBUG
    return "Release-like (NDEBUG)";
#else
    return "Debug-like";
#endif
}

// ============================================================================
// JSON 输出
// ============================================================================

std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

void AppendAggregateJson(std::ostringstream& json, const char* name,
                         const AggregateStats& a, bool trailing_comma) {
    json << "      \"" << name << "\": {"
         << "\"mean\": " << a.mean << ", "
         << "\"p50\": " << a.p50 << ", "
         << "\"p95\": " << a.p95 << ", "
         << "\"min\": " << a.min_val << ", "
         << "\"max\": " << a.max_val << "}"
         << (trailing_comma ? "," : "") << "\n";
}

void WriteRunJson(const fs::path& path, const PerRunRecord& run) {
    std::ofstream ofs(path);
    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"iteration\": " << run.iteration << ",\n";
    ofs << "  \"success\": " << (run.success ? "true" : "false") << ",\n";
    ofs << "  \"rss_peak_kb\": " << run.rss_peak_kb << ",\n";
    ofs << "  \"total_wall_time_ms\": " << run.total_wall_time_ms << ",\n";
    ofs << "  \"frame_count\": " << run.frame_count << ",\n";
    ofs << "  \"terminal_state\": \"" << JsonEscape(run.terminal_state)
        << "\",\n";
    if (!run.error_message.empty()) {
        ofs << "  \"error_message\": \""
            << JsonEscape(run.error_message) << "\",\n";
    }
    ofs << "  \"stats\": {\n";
    ofs << "    \"generated_task_count\": "
        << run.stats.generated_task_count << ",\n";
    ofs << "    \"scheduler_accepted_count\": "
        << run.stats.scheduler_accepted_count << ",\n";
    ofs << "    \"rendered_unique_frame_count\": "
        << run.stats.rendered_unique_frame_count << ",\n";
    ofs << "    \"unique_delivered_count\": "
        << run.stats.unique_delivered_count << ",\n";
    ofs << "    \"prepare_time_ms\": " << run.stats.prepare_time_ms << ",\n";
    ofs << "    \"audio_process_time_ms\": "
        << run.stats.audio_process_time_ms << ",\n";
    ofs << "    \"inference_total_time_ms\": "
        << run.stats.inference_total_time_ms << ",\n";
    ofs << "    \"render_total_time_ms\": "
        << run.stats.render_total_time_ms << ",\n";
    ofs << "    \"total_wall_time_ms\": "
        << run.stats.total_wall_time_ms << ",\n";
    ofs << "    \"q1_high_watermark\": "
        << run.stats.q1_high_watermark << ",\n";
    ofs << "    \"q2_high_watermark\": "
        << run.stats.q2_high_watermark << ",\n";
    ofs << "    \"dropped_late_count\": "
        << run.stats.dropped_late_count << ",\n";
    ofs << "    \"dropped_overflow_count\": "
        << run.stats.dropped_overflow_count << ",\n";
    ofs << "    \"repeated_delivery_count\": "
        << run.stats.repeated_delivery_count << "\n";
    ofs << "  }\n";
    ofs << "}\n";
    ofs.close();
}

std::string BuildBenchmarkJson(const std::vector<PerRunRecord>& runs) {
    // 提取各字段
    std::vector<double> total_wall, prepare, audio, inference, render;
    std::vector<double> throughput, rss_vals;
    std::vector<double> q1_hwm, q2_hwm, gen_tasks, rendered_frames;

    for (const auto& r : runs) {
        if (!r.success) continue;
        total_wall.push_back(r.total_wall_time_ms);
        prepare.push_back(r.stats.prepare_time_ms);
        audio.push_back(r.stats.audio_process_time_ms);
        inference.push_back(r.stats.inference_total_time_ms);
        render.push_back(r.stats.render_total_time_ms);
        rss_vals.push_back(static_cast<double>(r.rss_peak_kb));
        q1_hwm.push_back(static_cast<double>(r.stats.q1_high_watermark));
        q2_hwm.push_back(static_cast<double>(r.stats.q2_high_watermark));
        gen_tasks.push_back(
            static_cast<double>(r.stats.generated_task_count));
        rendered_frames.push_back(
            static_cast<double>(r.stats.rendered_unique_frame_count));
        if (r.total_wall_time_ms > 0.0) {
            throughput.push_back(
                1000.0 * static_cast<double>(r.frame_count) /
                r.total_wall_time_ms);
        }
    }

    // 系统信息
    unsigned int hw_threads = std::thread::hardware_concurrency();

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\n";
    json << "  \"description\": "
            "\"M05 Pipeline performance benchmark with golden inputs\",\n";
    json << "  \"build_mode\": \"" << BuildMode() << "\",\n";
    json << "  \"environment\": {\n";
    json << "    \"cpu_model\": \""
         << JsonEscape(GetCpuModel()) << "\",\n";
    json << "    \"cpu_cores\": " << hw_threads << ",\n";
    json << "    \"compiler\": \""
         << JsonEscape(GetCompilerInfo()) << "\",\n";
    json << "    \"kernel\": \""
         << JsonEscape(GetKernelVersion()) << "\",\n";
    json << "    \"hostname\": \""
         << JsonEscape([]() -> std::string {
                std::ifstream h("/proc/sys/kernel/hostname");
                std::string hn;
                std::getline(h, hn);
                return hn.empty() ? "Unknown" : hn;
            }())
         << "\"\n  },\n";

    // 输入哈希
    json << "  \"inputs\": {\n";
    json << "    \"image_path\": \""
         << JsonEscape(kGoldenImagePath.string()) << "\",\n";
    json << "    \"image_sha256\": \""
         << ComputeSha256(kGoldenImagePath) << "\",\n";
    json << "    \"audio_path\": \""
         << JsonEscape(kGoldenAudioPath.string()) << "\",\n";
    json << "    \"audio_sha256\": \""
         << ComputeSha256(kGoldenAudioPath) << "\",\n";
    json << "    \"model_param_path\": \""
         << JsonEscape(kModelParamPath.string()) << "\",\n";
    json << "    \"model_param_sha256\": \""
         << ComputeSha256(kModelParamPath) << "\",\n";
    json << "    \"model_bin_path\": \""
         << JsonEscape(kModelBinPath.string()) << "\",\n";
    json << "    \"model_bin_sha256\": \""
         << ComputeSha256(kModelBinPath) << "\",\n";
    json << "    \"landmark_model_path\": \""
         << JsonEscape(kLandmarkModelPath.string()) << "\",\n";
    json << "    \"landmark_model_sha256\": \""
         << ComputeSha256(kLandmarkModelPath) << "\"\n  },\n";

    // Pipeline 配置
    json << "  \"pipeline_config\": {\n";
    json << "    \"fps\": " << kFps << ",\n";
    json << "    \"q1_capacity\": " << kQueueCapacityQ1 << ",\n";
    json << "    \"q2_capacity\": " << kQueueCapacityQ2 << ",\n";
    json << "    \"scheduler_worker_count\": "
         << kSchedulerWorkers << ",\n";
    json << "    \"ncnn_threads\": " << kNcnnThreads << "\n  },\n";

    json << "  \"iterations\": " << runs.size() << ",\n";
    json << "  \"successful_runs\": "
         << std::count_if(runs.begin(), runs.end(),
                          [](const PerRunRecord& r) { return r.success; })
         << ",\n";
    json << "  \"percentile_method\": \"nearest-rank\",\n";

    // 逐轮结果索引
    json << "  \"run_files\": [\n";
    for (std::size_t i = 0; i < runs.size(); ++i) {
        json << "    \"pipeline_benchmark_run_" << runs[i].iteration << ".json\""
             << (i + 1 < runs.size() ? "," : "") << "\n";
    }
    json << "  ],\n";

    // 聚合统计
    json << "  \"aggregates\": {\n";
    AppendAggregateJson(json, "total_wall_time_ms",
                        Aggregate(total_wall), true);
    AppendAggregateJson(json, "prepare_time_ms",
                        Aggregate(prepare), true);
    AppendAggregateJson(json, "audio_process_time_ms",
                        Aggregate(audio), true);
    AppendAggregateJson(json, "inference_total_time_ms",
                        Aggregate(inference), true);
    AppendAggregateJson(json, "render_total_time_ms",
                        Aggregate(render), true);
    AppendAggregateJson(json, "rss_peak_kb",
                        Aggregate(rss_vals), true);
    AppendAggregateJson(json, "throughput_frames_per_second",
                        Aggregate(throughput), true);
    AppendAggregateJson(json, "q1_high_watermark",
                        Aggregate(q1_hwm), true);
    AppendAggregateJson(json, "q2_high_watermark",
                        Aggregate(q2_hwm), true);
    AppendAggregateJson(json, "generated_task_count",
                        Aggregate(gen_tasks), true);
    AppendAggregateJson(json, "rendered_unique_frame_count",
                        Aggregate(rendered_frames), false);
    json << "  },\n";

    json << "  \"notes\": [\n";
    json << "    \"Performance values are environment-specific, "
            "not correctness thresholds\",\n";
    json << "    \"RSS sampled from /proc/self/status via polling "
            "thread at ~" << kRssPollIntervalMs << "ms intervals\",\n";
    json << "    \"Runs executed sequentially in single process; "
            "RSS may include allocator/cache retained from earlier runs\",\n";
    json << "    \"Golden inputs verify correctness; benchmark "
            "captures timing and resource evidence\",\n";
    json << "    \"No business algorithm, Pipeline thread/queue "
            "parameters, or models were modified for benchmarking\",\n";
    json << "    \"Stage timers not populated by Pipeline implementation: "
            "audio_process_time_ms, inference_total_time_ms, "
            "render_total_time_ms — declared as 未测量. "
            "Only prepare_time_ms is instrumented in "
            "src/pipeline/digital_human_pipeline.cpp:341. "
            "The other three stages run inside worker threads without "
            "per-stage wall-clock accumulation in PipelineStats\",\n";
    json << "    \"wav2lip.bin SHA256 collected for full model "
            "traceability; model bin is ~138 MB ncnn weights file\"\n";
    json << "  ]\n";
    json << "}\n";
    return json.str();
}

// ============================================================================
// 单轮运行
// ============================================================================

PerRunRecord RunOneIteration(int iter, const fs::path& output_dir) {
    PerRunRecord record;
    record.iteration = iter;

    // 刷新文件系统缓存？不做——这是环境特征，不应隐藏
    auto sink = std::make_shared<BenchmarkSink>();
    DigitalHumanPipeline pipeline;

    PipelineConfig config = PipelineConfig::OfflineDefault();
    config.image_path = kGoldenImagePath;
    config.audio_path = kGoldenAudioPath;
    config.model_param_path = kModelParamPath;
    config.landmark_model_path = kLandmarkModelPath;
    config.fps_num = kFps;
    config.fps_den = 1;
    config.output_dir = output_dir;
    config.q1_capacity = kQueueCapacityQ1;
    config.q2_capacity = kQueueCapacityQ2;
    config.scheduler_worker_count = kSchedulerWorkers;
    config.ncnn_threads = kNcnnThreads;

    // 启动 RSS 轮询
    RssPoller rss_poller;
    rss_poller.Start();

    auto t_start = std::chrono::steady_clock::now();

    auto start_result = pipeline.Start(config, sink);
    if (!start_result.success) {
        rss_poller.Stop();
        record.success = false;
        record.error_message = start_result.error_message;
        record.rss_peak_kb = rss_poller.PeakKb();
        return record;
    }

    // 等待终端
    sink->WaitForTerminal(std::chrono::milliseconds(180000));

    auto t_end = std::chrono::steady_clock::now();
    record.total_wall_time_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Stop pipeline（确保 join）
    pipeline.Stop();

    rss_poller.Stop();
    record.rss_peak_kb = rss_poller.PeakKb();

    if (sink->terminal_called()) {
        auto terminal = sink->terminal_result();
        record.success = terminal.success;
        record.stats = terminal.stats;
        record.terminal_state =
            PipelineStateToString(terminal.terminal_state);
        if (!terminal.success) {
            record.error_message = terminal.error_message;
        }
    } else {
        record.success = false;
        record.error_message = "OnTerminal not called within timeout";
        record.terminal_state = "Timeout";
        record.stats = pipeline.GetStats();
    }

    record.frame_count = sink->frame_count();

    // 使用 PipelineStats 中的 total_wall_time_ms（内部时钟）
    // 作为更精确的总耗时参考（但保留外部测量供交叉验证）
    // record.total_wall_time_ms 保留外部测量供参考

    return record;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    const int iterations =
        (argc > 1) ? std::atoi(argv[1]) : kDefaultIterations;
    const fs::path output_dir =
        (argc > 2) ? argv[2] : "golden_output";

    if (iterations < 1) {
        std::cerr << "Error: iterations must be >= 1, got "
                  << iterations << "\n";
        return 1;
    }

    // 检查输入
    for (const auto& p : {kGoldenImagePath, kGoldenAudioPath,
                          kModelParamPath, kModelBinPath, kLandmarkModelPath}) {
        if (!fs::exists(p)) {
            std::cerr << "Error: file not found: " << p << "\n";
            return 1;
        }
    }

    fs::create_directories(output_dir);

    std::cout << "=== M05 Pipeline Performance Benchmark ===\n";
    std::cout << "Build mode: " << BuildMode() << "\n";
    std::cout << "CPU: " << GetCpuModel() << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Output: " << output_dir << "\n\n";

    std::vector<PerRunRecord> runs;
    runs.reserve(static_cast<std::size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        std::cout << "--- Iteration " << (i + 1) << "/" << iterations
                  << " ---" << std::endl;

        auto record = RunOneIteration(i, output_dir);
        runs.push_back(record);

        // 写单轮 JSON
        char run_filename[64];
        std::snprintf(run_filename, sizeof(run_filename),
                     "pipeline_benchmark_run_%d.json", i);
        WriteRunJson(output_dir / run_filename, record);

        std::cout << "  Success: " << (record.success ? "true" : "false")
                  << "\n";
        if (record.success) {
            std::cout << "  Frames: " << record.frame_count << "\n";
            std::cout << "  Total wall time: " << std::fixed
                      << std::setprecision(2)
                      << record.total_wall_time_ms << " ms\n";
            std::cout << "  RSS peak: " << record.rss_peak_kb << " kB\n";
            if (record.total_wall_time_ms > 0.0) {
                double fps = 1000.0 *
                             static_cast<double>(record.frame_count) /
                             record.total_wall_time_ms;
                std::cout << "  Throughput: " << std::fixed
                          << std::setprecision(2) << fps << " fps\n";
            }
            std::cout << "  Prepare: " << std::fixed
                      << std::setprecision(2)
                      << record.stats.prepare_time_ms << " ms\n";
            std::cout << "  Audio: " << std::fixed
                      << std::setprecision(2)
                      << record.stats.audio_process_time_ms << " ms\n";
            std::cout << "  Inference: " << std::fixed
                      << std::setprecision(2)
                      << record.stats.inference_total_time_ms << " ms\n";
            std::cout << "  Render: " << std::fixed
                      << std::setprecision(2)
                      << record.stats.render_total_time_ms << " ms\n";
            std::cout << "  Q1 HWM: " << record.stats.q1_high_watermark
                      << "\n";
            std::cout << "  Q2 HWM: " << record.stats.q2_high_watermark
                      << "\n";
        } else {
            std::cout << "  Error: " << record.error_message << "\n";
            // 非成功轮次：继续下一轮
        }

        std::cout << std::endl;
    }

    // 写汇总 JSON
    std::string summary_json = BuildBenchmarkJson(runs);
    fs::path summary_path =
        output_dir / "pipeline_performance_benchmark.json";
    {
        std::ofstream ofs(summary_path);
        ofs << summary_json;
        ofs.close();
    }
    std::cout << "Summary written to: " << summary_path << "\n";

    // 检查：至少一轮成功
    bool any_success = false;
    for (const auto& r : runs) {
        if (r.success) {
            any_success = true;
            break;
        }
    }

    if (!any_success) {
        std::cerr << "ERROR: All " << iterations
                  << " iterations failed.\n";
        return 1;
    }

    std::cout << "\n=== BENCHMARK COMPLETED ===\n";
    return 0;
}
