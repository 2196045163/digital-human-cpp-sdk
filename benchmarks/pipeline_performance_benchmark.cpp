/// @file pipeline_performance_benchmark.cpp
/// @brief M05 正式 Pipeline 性能基准：多轮全链路采集 PipelineStats/RSS/吞吐
///
/// 本基准测量【真实端到端路径】：
///   Pipeline → FinalMediaWriter → libx264 视频编码 → AAC 音频编码 → mux →
///   flush/trailer → 磁盘 MP4 文件
///
/// 用法：
///   ./pipeline_performance_benchmark [iterations] [output_dir] [raw_log_dir]
///
/// 采集指标：
///   - PipelineStats（prepare/audio/inference/render/total 阶段耗时）
///   - 队列高水位（Q1/Q2）
///   - 峰值 RSS（/proc/self/status 轮询线程，~100ms）
///   - 吞吐（帧/秒）
///   - 输出 MP4 路径 / 文件大小（磁盘字节）
///   - ffprobe 验证（流数量、视频帧数、编解码器、尺寸、时长）
///   - 系统环境信息（CPU、编译器、内核）
///   - 输入文件 SHA256 哈希
///
/// 输出：
///   - <output_dir>/pipeline_performance_benchmark.json — 汇总 JSON
///   - <output_dir>/pipeline_benchmark_run_N.json — 单轮详细统计（兼容）
///   - <raw_log_dir>/pipeline_benchmark_run_raw_N.json — 单轮原始数据
///     （含帧索引/PTS 完整轨迹），默认 raw_log_dir = output_dir
///
/// 每轮运行后：
///   - 记录输出 MP4 路径与文件大小
///   - 运行 ffprobe 验证文件有效性（帧数、流）
///   - 验证完成后删除 MP4，避免磁盘累积（下一轮重新生成）
///
/// @note 性能结果仅对本次机器、构建和参数有效，不构成正确性阈值。

#include "audio/audio_loader.h"
#include "output/final_media_writer.h"
#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"

#include <sys/wait.h>

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
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace digital_human::pipeline;

// FinalMediaWriter / AudioLoader 显式引入（audio/output 命名空间不做
// using-directive，避免与 pipeline 命名空间符号混淆）
using digital_human::audio::AudioData;
using digital_human::audio::AudioLoader;
using digital_human::audio::AudioLoadOptions;
using digital_human::audio::AudioSampleFormat;
using digital_human::output::FinalMediaWriter;
using digital_human::output::WriterConfig;
using digital_human::output::WriterError;
using digital_human::output::WriterErrorToString;

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
constexpr int kAudioSampleRate = 16000;

// FinalMediaWriter 编码配置（与正式离线输出链路一致）
constexpr std::int64_t kVideoBitRate = 2000000;   // 2 Mbps (libx264)
constexpr std::int64_t kAudioBitRate = 128000;    // 128 kbps (AAC)
constexpr int kGopSize = 250;
constexpr int kMaxBFrames = 0;
const char* kX264Preset = "medium";

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
// FinalMediaWriter 基准 Sink
// ============================================================================
//
// 真实端到端路径：Pipeline → FinalMediaWriter → 编码 → mux → flush/trailer →
// 磁盘 MP4。本包装类在转发回调给 FinalMediaWriter 的同时：
//   - 提供 WaitForTerminal（OnTerminal 在 writer 完成 flush/trailer 之后
//     才通知，因此等待返回即代表 MP4 已完整落盘）
//   - 记录交付帧索引/PTS 轨迹（原始数据）
//   - 透出 writer 的查询接口（写帧数、错误码）

class FinalMediaWriterSink : public PipelineOutputSink {
public:
    explicit FinalMediaWriterSink(
        std::shared_ptr<FinalMediaWriter> writer)
        : writer_(std::move(writer)) {}

    void OnFrame(const PipelineFrame& frame) override {
        // 真实编码路径：BGR → YUV420P → libx264 编码 → 交错写入 MP4
        writer_->OnFrame(frame);
        std::lock_guard<std::mutex> lock(mutex_);
        frame_count_++;
        frame_indices_.push_back(frame.video_frame.frame_index);
        pts_values_.push_back(frame.video_frame.pts.microseconds);
    }

    void OnTerminal(const PipelineResult& result) override {
        // writer 的 OnTerminal 同步执行：flush 视频编码器 → 编码全部音频 →
        // flush 音频编码器 → 写 trailer → 关闭文件。全部完成后才通知等待者，
        // 保证 WaitForTerminal 返回时 MP4 已完整存在于磁盘。
        writer_->OnTerminal(result);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            terminal_result_ = result;
            terminal_called_ = true;
        }
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
    std::vector<std::int64_t> FrameIndicesSnapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frame_indices_;
    }
    std::vector<std::int64_t> PtsValuesSnapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pts_values_;
    }

    const FinalMediaWriter& writer() const { return *writer_; }

private:
    std::shared_ptr<FinalMediaWriter> writer_;
    mutable std::mutex mutex_;
    std::condition_variable terminal_cv_;
    int frame_count_ = 0;
    std::vector<std::int64_t> frame_indices_;
    std::vector<std::int64_t> pts_values_;
    bool terminal_called_ = false;
    PipelineResult terminal_result_;
};

// ============================================================================
// ffprobe 验证
// ============================================================================

/// @brief ffprobe 对输出 MP4 的验证结果
struct FfprobeReport {
    bool ok = false;                       ///< 验证通过（exit 0 且有流）
    int exit_code = -1;                    ///< ffprobe 退出码
    int stream_count = 0;                  ///< 流数量
    std::vector<std::string> stream_types; ///< 流类型列表（video/audio/...）
    std::int64_t video_frame_count = -1;   ///< 实测解码视频帧数
    std::string video_codec;               ///< 视频编码器名（如 h264）
    int video_width = 0;                   ///< 视频宽度
    int video_height = 0;                  ///< 视频高度
    double duration_sec = 0.0;             ///< 容器时长（秒）
    std::string probe_output;              ///< ffprobe 原始输出（含错误）
};

std::string TrimString(std::string s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    std::size_t pos = s.find_last_not_of(ws);
    if (pos != std::string::npos) {
        s.erase(pos + 1);
    } else {
        s.clear();
    }
    return s;
}

/// @brief 执行 shell 命令并捕获 stdout+stderr，返回 (退出码, 输出)
std::pair<int, std::string> RunCommandCapture(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {-1, "ERROR:popen"};
    std::string out;
    char buf[512];
    std::size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        out.append(buf, n);
    }
    int status = pclose(pipe);
    int code = -1;
    if (status != -1 && WIFEXITED(status)) {
        code = WEXITSTATUS(status);
    }
    return {code, out};
}

/// @brief 验证输出 MP4：流数量、实测帧数、编解码器、尺寸、时长
FfprobeReport ProbeMp4(const fs::path& path) {
    FfprobeReport report;
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        report.probe_output = "MP4 file not found: " + path.string();
        return report;
    }

    int code = -1;
    std::string out;

    // 1. 流类型列表
    std::tie(code, out) = RunCommandCapture(
        "ffprobe -v error -show_entries stream=codec_type -of csv=p=0 \""
        + path.string() + "\"");
    report.exit_code = code;
    report.probe_output = out;
    if (code != 0) return report;
    {
        std::istringstream iss(out);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                report.stream_types.push_back(line);
                ++report.stream_count;
            }
        }
    }

    // 2. 实测解码视频帧数（-count_frames）
    std::tie(code, out) = RunCommandCapture(
        "ffprobe -v error -count_frames -select_streams v:0 "
        "-show_entries stream=nb_read_frames "
        "-of default=noprint_wrappers=1:nokey=1 \""
        + path.string() + "\"");
    if (code == 0) {
        try {
            report.video_frame_count = std::stoll(TrimString(out));
        } catch (...) {
        }
    } else if (!out.empty()) {
        report.probe_output += "\nframe_count_error: " + out;
    }

    // 3. 视频编解码器 / 尺寸
    std::tie(code, out) = RunCommandCapture(
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=codec_name,width,height -of csv=p=0 \""
        + path.string() + "\"");
    if (code == 0) {
        std::istringstream iss(out);
        std::string name, w, h;
        if (std::getline(iss, name, ',') && std::getline(iss, w, ',') &&
            std::getline(iss, h)) {
            report.video_codec = name;
            try { report.video_width = std::stoi(w); } catch (...) {}
            try { report.video_height = std::stoi(h); } catch (...) {}
        }
    }

    // 4. 容器时长
    std::tie(code, out) = RunCommandCapture(
        "ffprobe -v error -show_entries format=duration "
        "-of default=noprint_wrappers=1:nokey=1 \""
        + path.string() + "\"");
    if (code == 0) {
        try {
            report.duration_sec = std::stod(TrimString(out));
        } catch (...) {
        }
    }

    report.ok = (report.stream_count > 0);
    return report;
}

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

    // ---- FinalMediaWriter 端到端输出 ----
    std::string output_mp4_path;              ///< 输出 MP4 路径
    bool output_file_exists = false;          ///< MP4 是否存在于磁盘
    std::int64_t output_file_size_bytes = -1; ///< MP4 文件大小（字节）
    std::int64_t writer_written_frame_count = -1; ///< writer 写入帧数
    std::string writer_error;                 ///< writer 错误码字符串
    std::string writer_error_message;         ///< writer 错误描述
    FfprobeReport ffprobe;                    ///< ffprobe 验证结果

    // ---- 原始数据（帧轨迹） ----
    std::vector<std::int64_t> frame_indices;
    std::vector<std::int64_t> pts_values;
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

/// @brief 写单轮 JSON。
/// @param include_raw 为 true 时额外包含原始帧轨迹（frame_indices / pts）
void WriteRunJson(const fs::path& path, const PerRunRecord& run,
                  bool include_raw) {
    std::ofstream ofs(path);
    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"iteration\": " << run.iteration << ",\n";
    ofs << "  \"success\": " << (run.success ? "true" : "false") << ",\n";
    ofs << "  \"rss_peak_kb\": " << run.rss_peak_kb << ",\n";
    ofs << "  \"total_wall_time_ms\": " << run.total_wall_time_ms << ",\n";
    ofs << "  \"frame_count\": " << run.frame_count << ",\n";
    {
        double fps = (run.total_wall_time_ms > 0.0)
            ? 1000.0 * static_cast<double>(run.frame_count) /
                  run.total_wall_time_ms
            : 0.0;
        ofs << "  \"throughput_frames_per_second\": " << fps << ",\n";
    }
    ofs << "  \"terminal_state\": \"" << JsonEscape(run.terminal_state)
        << "\",\n";
    if (!run.error_message.empty()) {
        ofs << "  \"error_message\": \""
            << JsonEscape(run.error_message) << "\",\n";
    }
    // ---- FinalMediaWriter 端到端输出 ----
    ofs << "  \"output_mp4_path\": \""
        << JsonEscape(run.output_mp4_path) << "\",\n";
    ofs << "  \"output_file_exists\": "
        << (run.output_file_exists ? "true" : "false") << ",\n";
    ofs << "  \"output_file_size_bytes\": " << run.output_file_size_bytes
        << ",\n";
    ofs << "  \"writer_written_frame_count\": "
        << run.writer_written_frame_count << ",\n";
    ofs << "  \"writer_error\": \"" << JsonEscape(run.writer_error)
        << "\",\n";
    ofs << "  \"writer_error_message\": \""
        << JsonEscape(run.writer_error_message) << "\",\n";
    // ---- ffprobe 验证 ----
    ofs << "  \"ffprobe\": {\n";
    ofs << "    \"ok\": " << (run.ffprobe.ok ? "true" : "false") << ",\n";
    ofs << "    \"exit_code\": " << run.ffprobe.exit_code << ",\n";
    ofs << "    \"stream_count\": " << run.ffprobe.stream_count << ",\n";
    ofs << "    \"stream_types\": [";
    for (std::size_t i = 0; i < run.ffprobe.stream_types.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << "\"" << JsonEscape(run.ffprobe.stream_types[i]) << "\"";
    }
    ofs << "],\n";
    ofs << "    \"video_frame_count\": " << run.ffprobe.video_frame_count
        << ",\n";
    ofs << "    \"video_codec\": \"" << JsonEscape(run.ffprobe.video_codec)
        << "\",\n";
    ofs << "    \"video_width\": " << run.ffprobe.video_width << ",\n";
    ofs << "    \"video_height\": " << run.ffprobe.video_height << ",\n";
    ofs << "    \"duration_sec\": " << run.ffprobe.duration_sec << ",\n";
    ofs << "    \"probe_output\": \""
        << JsonEscape(run.ffprobe.probe_output) << "\"\n";
    ofs << "  },\n";
    // ---- PipelineStats ----
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
    ofs << "  }";
    // ---- 原始帧轨迹（raw 模式） ----
    if (include_raw) {
        ofs << ",\n  \"raw\": {\n";
        ofs << "    \"frame_indices\": [";
        for (std::size_t i = 0; i < run.frame_indices.size(); ++i) {
            if (i > 0) ofs << ", ";
            ofs << run.frame_indices[i];
        }
        ofs << "],\n";
        ofs << "    \"pts_values_us\": [";
        for (std::size_t i = 0; i < run.pts_values.size(); ++i) {
            if (i > 0) ofs << ", ";
            ofs << run.pts_values[i];
        }
        ofs << "]\n";
        ofs << "  }\n";
    } else {
        ofs << "\n";
    }
    ofs << "}\n";
    ofs.close();
}

std::string BuildBenchmarkJson(const std::vector<PerRunRecord>& runs) {
    // 提取各字段
    std::vector<double> total_wall, prepare, audio, inference, render;
    std::vector<double> throughput, rss_vals;
    std::vector<double> q1_hwm, q2_hwm, gen_tasks, rendered_frames;
    std::vector<double> output_sizes, writer_frames;

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
        output_sizes.push_back(
            static_cast<double>(r.output_file_size_bytes));
        writer_frames.push_back(
            static_cast<double>(r.writer_written_frame_count));
    }

    // 系统信息
    unsigned int hw_threads = std::thread::hardware_concurrency();

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\n";
    json << "  \"description\": "
            "\"M05 end-to-end pipeline performance benchmark with golden "
            "inputs: Pipeline -> FinalMediaWriter (libx264 video + AAC audio "
            "mux) -> MP4 on disk\",\n";
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

    // FinalMediaWriter 编码配置
    json << "  \"writer_config\": {\n";
    json << "    \"video_codec\": \"libx264\",\n";
    json << "    \"audio_codec\": \"aac\",\n";
    json << "    \"video_bit_rate\": " << kVideoBitRate << ",\n";
    json << "    \"audio_bit_rate\": " << kAudioBitRate << ",\n";
    json << "    \"x264_preset\": \"" << kX264Preset << "\",\n";
    json << "    \"gop_size\": " << kGopSize << ",\n";
    json << "    \"max_b_frames\": " << kMaxBFrames << ",\n";
    json << "    \"audio_sample_rate\": " << kAudioSampleRate << ",\n";
    json << "    \"audio_channels\": 1\n  },\n";

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
                        Aggregate(rendered_frames), true);
    AppendAggregateJson(json, "output_file_size_bytes",
                        Aggregate(output_sizes), true);
    AppendAggregateJson(json, "writer_written_frame_count",
                        Aggregate(writer_frames), false);
    json << "  },\n";

    json << "  \"notes\": [\n";
    json << "    \"End-to-end path measured: Pipeline -> FinalMediaWriter -> "
            "libx264 video encode -> AAC audio encode -> mux -> "
            "flush/trailer -> MP4 on disk\",\n";
    json << "    \"total_wall_time_ms includes writer OnTerminal "
            "(encoder flush + audio encode + trailer write)\",\n";
    json << "    \"Each run verified with ffprobe: stream count, decoded "
            "video frame count, codec, dimensions, duration\",\n";
    json << "    \"Output MP4 deleted after verification to avoid disk "
            "accumulation; next run regenerates it\",\n";
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

PerRunRecord RunOneIteration(int iter, const fs::path& output_dir,
                             const fs::path& mp4_path,
                             const AudioData& audio_data) {
    PerRunRecord record;
    record.iteration = iter;
    record.output_mp4_path = mp4_path.string();

    // 防御性清理：不允许上一轮残留影响本轮
    std::error_code ec;
    fs::remove(mp4_path, ec);

    // ============ FinalMediaWriter（真实端到端输出路径） ============
    WriterConfig wcfg;
    wcfg.output_path = mp4_path.string();
    wcfg.fps_num = kFps;
    wcfg.fps_den = 1;
    wcfg.video_width = 0;      // 0 = 首帧自动推断
    wcfg.video_height = 0;
    wcfg.video_codec = "libx264";
    wcfg.audio_codec = "aac";
    wcfg.video_bit_rate = kVideoBitRate;
    wcfg.audio_bit_rate = kAudioBitRate;
    wcfg.gop_size = kGopSize;
    wcfg.max_b_frames = kMaxBFrames;
    wcfg.x264_preset = kX264Preset;
    wcfg.audio = audio_data;   // golden 音频随视频帧一起封装进 MP4

    std::shared_ptr<FinalMediaWriter> writer;
    try {
        writer = std::make_shared<FinalMediaWriter>(wcfg);
    } catch (const std::exception& ex) {
        record.success = false;
        record.error_message =
            std::string("FinalMediaWriter constructor: ") + ex.what();
        return record;
    }
    auto sink = std::make_shared<FinalMediaWriterSink>(writer);

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

    // 等待终端（返回即代表 FinalMediaWriter 已 flush/trailer，MP4 完整落盘）
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
    record.frame_indices = sink->FrameIndicesSnapshot();
    record.pts_values = sink->PtsValuesSnapshot();

    // ============ FinalMediaWriter 状态 ============
    record.writer_written_frame_count = writer->GetWrittenFrameCount();
    record.writer_error =
        WriterErrorToString(writer->GetLastError());
    record.writer_error_message = writer->GetLastErrorMessage();
    if (record.success &&
        writer->GetLastError() != WriterError::kOk) {
        record.success = false;
        record.error_message =
            "FinalMediaWriter error: " + record.writer_error + " (" +
            record.writer_error_message + ")";
    }

    // ============ 输出 MP4 文件信息 ============
    std::error_code stat_ec;
    record.output_file_exists = fs::exists(mp4_path, stat_ec) && !stat_ec;
    if (record.output_file_exists) {
        std::error_code sz_ec;
        std::uintmax_t sz = fs::file_size(mp4_path, sz_ec);
        record.output_file_size_bytes =
            sz_ec ? -1 : static_cast<std::int64_t>(sz);
    }

    // ============ ffprobe 验证 ============
    record.ffprobe = ProbeMp4(mp4_path);
    if (record.success && !record.ffprobe.ok) {
        record.success = false;
        record.error_message =
            "ffprobe verification failed: " + record.ffprobe.probe_output;
    }

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
    // 原始数据目录：logs/performance_benchmark/<timestamp>/
    const fs::path raw_log_dir =
        (argc > 3) ? argv[3] : output_dir;

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

    // 加载 golden 音频（FinalMediaWriter 需要 PCM 以封装音轨）
    AudioLoader audio_loader(kAudioSampleRate);
    AudioLoadOptions audio_opts;
    audio_opts.target_sample_rate = kAudioSampleRate;
    audio_opts.target_channels = 1;
    audio_opts.output_format = AudioSampleFormat::kFloat32;
    auto audio_result =
        audio_loader.LoadFromFile(kGoldenAudioPath.string(), audio_opts);
    if (!audio_result.success) {
        std::cerr << "Error: failed to load golden audio: "
                  << audio_result.error_message << "\n";
        return 1;
    }
    const AudioData audio_data = audio_result.audio;

    fs::create_directories(output_dir);
    fs::create_directories(raw_log_dir);

    std::cout << "=== M05 Pipeline Performance Benchmark ===\n";
    std::cout << "Build mode: " << BuildMode() << "\n";
    std::cout << "CPU: " << GetCpuModel() << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Output: " << output_dir << "\n";
    std::cout << "Raw log dir: " << raw_log_dir << "\n";
    std::cout << "Path: Pipeline -> FinalMediaWriter -> MP4 "
                 "(libx264 video + AAC audio)\n\n";

    std::vector<PerRunRecord> runs;
    runs.reserve(static_cast<std::size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        std::cout << "--- Iteration " << (i + 1) << "/" << iterations
                  << " ---" << std::endl;

        fs::path mp4_path = output_dir /
            ("pipeline_benchmark_run_" + std::to_string(i) + ".mp4");

        auto record = RunOneIteration(i, output_dir, mp4_path, audio_data);
        runs.push_back(record);

        // 写单轮 JSON（兼容位置：输出目录）
        char run_filename[64];
        std::snprintf(run_filename, sizeof(run_filename),
                     "pipeline_benchmark_run_%d.json", i);
        WriteRunJson(output_dir / run_filename, record,
                     /*include_raw=*/false);

        // 写单轮原始 JSON（logs/performance_benchmark/<timestamp>/）
        char raw_filename[64];
        std::snprintf(raw_filename, sizeof(raw_filename),
                     "pipeline_benchmark_run_raw_%d.json", i);
        WriteRunJson(raw_log_dir / raw_filename, record,
                     /*include_raw=*/true);

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
            std::cout << "  Writer frames: "
                      << record.writer_written_frame_count << "\n";
            std::cout << "  Output MP4: " << record.output_mp4_path
                      << (record.output_file_exists
                              ? " (" +
                                    std::to_string(
                                        record.output_file_size_bytes) +
                                    " bytes)"
                              : " (MISSING)")
                      << "\n";
            std::cout << "  ffprobe: " << (record.ffprobe.ok ? "OK" : "FAILED");
            if (record.ffprobe.ok) {
                std::cout << " streams=" << record.ffprobe.stream_count
                          << " video_frames="
                          << record.ffprobe.video_frame_count
                          << " codec=" << record.ffprobe.video_codec
                          << " " << record.ffprobe.video_width << "x"
                          << record.ffprobe.video_height
                          << " duration=" << std::fixed
                          << std::setprecision(2)
                          << record.ffprobe.duration_sec << "s";
            }
            std::cout << "\n";
        } else {
            std::cout << "  Error: " << record.error_message << "\n";
            // 非成功轮次：继续下一轮
        }

        // 清理：验证完成后删除 MP4，避免磁盘累积（下一轮重新生成）
        std::error_code ec;
        fs::remove(mp4_path, ec);

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
