#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

extern "C" {
#include <libavutil/avutil.h>
}

#include "audio/audio_loader.h"
#include "output/final_media_writer.h"
#include "pipeline/digital_human_pipeline.h"

#ifndef DIGITAL_HUMAN_BENCHMARK_SOURCE_DIR
#define DIGITAL_HUMAN_BENCHMARK_SOURCE_DIR "."
#endif

#ifndef DIGITAL_HUMAN_BENCHMARK_MODEL_PARAM
#define DIGITAL_HUMAN_BENCHMARK_MODEL_PARAM ""
#endif

#ifndef DIGITAL_HUMAN_BENCHMARK_LANDMARK_MODEL
#define DIGITAL_HUMAN_BENCHMARK_LANDMARK_MODEL ""
#endif

#ifndef DIGITAL_HUMAN_BENCHMARK_COMPILER
#define DIGITAL_HUMAN_BENCHMARK_COMPILER __VERSION__
#endif

#ifndef DIGITAL_HUMAN_BENCHMARK_BUILD_TYPE
#define DIGITAL_HUMAN_BENCHMARK_BUILD_TYPE ""
#endif

namespace fs = std::filesystem;

namespace {

constexpr int kDefaultOutputFps = 25;
constexpr int kOutputFpsDen = 1;
constexpr int kDefaultNcnnThreads = 1;
constexpr std::size_t kErrorCapacity = 1024;

struct BenchmarkPaths {
    fs::path image;
    fs::path audio;
    fs::path model_param;
    fs::path model_bin;
    fs::path landmark_model;
};

struct RunMetrics {
    int run_index = 0;
    std::int64_t process_id = 0;
    bool success = false;
    std::string error;
    std::int64_t total_frames = 0;
    std::int64_t inference_frames = 0;
    double total_wall_time_ms = 0.0;
    double inference_total_time_ms = 0.0;
    double average_inference_time_ms = 0.0;
    double inference_fps = 0.0;
    double peak_rss_mb = 0.0;
};

struct RunWireResult {
    std::int32_t run_index = 0;
    std::int64_t process_id = 0;
    std::int32_t success = 0;
    std::int64_t total_frames = 0;
    std::int64_t inference_frames = 0;
    double total_wall_time_ms = 0.0;
    double inference_total_time_ms = 0.0;
    double average_inference_time_ms = 0.0;
    double inference_fps = 0.0;
    double peak_rss_mb = 0.0;
    char error[kErrorCapacity]{};
};

static_assert(std::is_trivially_copyable<RunWireResult>::value,
              "RunWireResult must be pipe-safe");

struct MetricSummary {
    double average = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

std::string Trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n\"");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n\"");
    return value.substr(begin, end - begin + 1);
}

std::string ReadOsDescription() {
    std::ifstream input("/etc/os-release");
    std::string line;
    while (std::getline(input, line)) {
        constexpr const char* kPrefix = "PRETTY_NAME=";
        if (line.rfind(kPrefix, 0) == 0) {
            return Trim(line.substr(std::char_traits<char>::length(kPrefix)));
        }
    }
    return "unavailable";
}

std::string ReadCpuModel() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const auto delimiter = line.find(':');
        if (delimiter != std::string::npos
            && Trim(line.substr(0, delimiter)) == "model name") {
            return Trim(line.substr(delimiter + 1));
        }
    }
    return "unavailable";
}

std::string KernelDescription(bool& is_wsl) {
    utsname info{};
    if (uname(&info) != 0) {
        is_wsl = false;
        return "unavailable";
    }

    std::string release(info.release);
    std::string version(info.version);
    std::string combined = release + " " + version;
    std::string lower = combined;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    is_wsl = lower.find("microsoft") != std::string::npos
          || lower.find("wsl") != std::string::npos;
    return combined;
}

double ReadTotalMemoryMb() {
    struct sysinfo info {};
    if (sysinfo(&info) != 0) {
        return 0.0;
    }
    const auto total_bytes = static_cast<long double>(info.totalram)
                           * static_cast<long double>(info.mem_unit);
    return static_cast<double>(total_bytes / (1024.0L * 1024.0L));
}

long ReadLogicalCpuCount() {
    const long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? count : 0;
}

bool ParsePositiveInt(const char* text, int& value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }

    const char* end = text + std::char_traits<char>::length(text);
    const auto parsed = std::from_chars(text, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end && value > 0;
}

BenchmarkPaths DefaultPaths() {
    const fs::path source_dir(DIGITAL_HUMAN_BENCHMARK_SOURCE_DIR);
    const fs::path model_param(DIGITAL_HUMAN_BENCHMARK_MODEL_PARAM);

    BenchmarkPaths paths;
    paths.image = source_dir / "testdata/input/face.jpg";
    paths.audio = source_dir / "testdata/input/real_voice_test.wav";
    paths.model_param = model_param;
    paths.model_bin = model_param;
    paths.model_bin.replace_extension(".bin");
    paths.landmark_model = fs::path(DIGITAL_HUMAN_BENCHMARK_LANDMARK_MODEL);
    return paths;
}

bool ValidatePaths(const BenchmarkPaths& paths, std::string& error) {
    const std::pair<const char*, const fs::path*> required[] = {
        {"image", &paths.image},
        {"audio", &paths.audio},
        {"model param", &paths.model_param},
        {"model bin", &paths.model_bin},
        {"landmark model", &paths.landmark_model},
    };

    for (const auto& item : required) {
        if (item.second->empty() || !fs::is_regular_file(*item.second)) {
            error = std::string("Missing ") + item.first + ": " + item.second->string();
            return false;
        }
    }
    return true;
}

void PrintEnvironment(const BenchmarkPaths& paths, int run_count, int output_fps,
                      int ncnn_threads) {
    bool is_wsl = false;
    const std::string kernel = KernelDescription(is_wsl);
    const std::string build_type = Trim(DIGITAL_HUMAN_BENCHMARK_BUILD_TYPE);
    const cv::Mat source_image = cv::imread(paths.image.string(), cv::IMREAD_COLOR);
    const std::string resolution = source_image.empty()
        ? "unavailable"
        : std::to_string(source_image.cols) + "x" + std::to_string(source_image.rows);

    std::cout << "Benchmark environment\n"
              << "  operating_system: " << ReadOsDescription()
              << (is_wsl ? " (WSL)" : "") << '\n'
              << "  linux_kernel: " << kernel << '\n'
              << "  cpu_model: " << ReadCpuModel() << '\n'
              << "  logical_cpu_count: " << ReadLogicalCpuCount() << '\n'
              << "  total_memory_mb: " << ReadTotalMemoryMb() << '\n'
              << "  compiler: " << DIGITAL_HUMAN_BENCHMARK_COMPILER << '\n'
              << "  cmake_build_type: "
              << (build_type.empty() ? "not set" : build_type) << '\n'
              << "  inference_backend: ncnn CPU\n"
              << "  ncnn_threads: " << ncnn_threads << '\n'
              << "  ffmpeg_version: " << av_version_info() << '\n'
              << "  image_path: " << paths.image.string() << '\n'
              << "  audio_path: " << paths.audio.string() << '\n'
              << "  output_resolution: " << resolution << '\n'
              << "  output_fps: " << output_fps << '/' << kOutputFpsDen << '\n'
              << "  runs: " << run_count << '\n'
              << "Peak memory measurement\n"
              << "  metric: process Peak RSS\n"
              << "  api: getrusage(RUSAGE_SELF).ru_maxrss\n"
              << "  platform: Linux / WSL\n"
              << "  raw_unit: KB\n"
              << "  output_unit: MB\n"
              << "  conversion: ru_maxrss / 1024.0\n"
              << "  scope: highest resident set size during one complete benchmark child process\n"
              << "  per_run_isolation: one independent child process per run\n"
              << "  meaning: CPU process resident-memory peak; not heap-only and not GPU memory\n"
              << std::flush;
}

bool ReadPeakRssMb(double& peak_rss_mb, std::string& error) {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        error = "getrusage(RUSAGE_SELF) failed";
        return false;
    }

    // Linux/WSL reports ru_maxrss in KiB.
    peak_rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
    return true;
}

RunMetrics RunOnce(int run_index, int output_fps, int ncnn_threads,
                   const BenchmarkPaths& paths) {
    using digital_human::audio::AudioLoader;
    using digital_human::output::FinalMediaWriter;
    using digital_human::output::WriterConfig;
    using digital_human::output::WriterError;
    using digital_human::pipeline::DigitalHumanPipeline;
    using digital_human::pipeline::PipelineConfig;
    using digital_human::pipeline::PipelineErrorCodeToString;

    RunMetrics metrics;
    metrics.run_index = run_index;
    metrics.process_id = static_cast<std::int64_t>(getpid());

    if (!ValidatePaths(paths, metrics.error)) {
        ReadPeakRssMb(metrics.peak_rss_mb, metrics.error);
        return metrics;
    }

    const std::string output_name = output_fps == kDefaultOutputFps
        ? "digital_human_cpu_benchmark_run_" + std::to_string(run_index) + ".mp4"
        : "digital_human_cpu_benchmark_" + std::to_string(output_fps)
            + "fps_run_" + std::to_string(run_index) + ".mp4";
    const fs::path output_path = fs::path("/tmp") / output_name;
    std::error_code remove_error;
    fs::remove(output_path, remove_error);
    if (remove_error) {
        metrics.error = "Cannot remove previous output: " + output_path.string()
                      + ": " + remove_error.message();
        ReadPeakRssMb(metrics.peak_rss_mb, metrics.error);
        return metrics;
    }

    AudioLoader audio_loader;
    auto audio_result = audio_loader.LoadFromFile(paths.audio.string());
    if (!audio_result.success) {
        metrics.error = "Writer audio load failed: " + audio_result.error_message;
        ReadPeakRssMb(metrics.peak_rss_mb, metrics.error);
        return metrics;
    }

    WriterConfig writer_config;
    writer_config.output_path = output_path.string();
    writer_config.fps_num = output_fps;
    writer_config.fps_den = kOutputFpsDen;
    writer_config.audio = audio_result.audio;

    auto writer = std::make_shared<FinalMediaWriter>(writer_config);

    PipelineConfig pipeline_config = PipelineConfig::OfflineDefault();
    pipeline_config.image_path = paths.image;
    pipeline_config.audio_path = paths.audio;
    pipeline_config.model_param_path = paths.model_param;
    pipeline_config.model_bin_path = paths.model_bin;
    pipeline_config.landmark_model_path = paths.landmark_model;
    pipeline_config.fps_num = output_fps;
    pipeline_config.fps_den = kOutputFpsDen;
    pipeline_config.scheduler_worker_count = 1;
    pipeline_config.ncnn_threads = ncnn_threads;

    DigitalHumanPipeline pipeline;
    auto pipeline_result = pipeline.Start(pipeline_config, writer);
    if (pipeline_result.success) {
        pipeline_result = pipeline.Wait();
    }

    metrics.total_frames = writer->GetWrittenFrameCount();
    metrics.inference_frames = pipeline_result.stats.scheduler_accepted_count;
    metrics.total_wall_time_ms = pipeline_result.stats.total_wall_time_ms;
    metrics.inference_total_time_ms = pipeline_result.stats.inference_total_time_ms;

    if (metrics.inference_frames > 0 && metrics.inference_total_time_ms > 0.0) {
        metrics.average_inference_time_ms =
            metrics.inference_total_time_ms / static_cast<double>(metrics.inference_frames);
        metrics.inference_fps =
            static_cast<double>(metrics.inference_frames) * 1000.0 /
            metrics.inference_total_time_ms;
    }

    std::string rss_error;
    if (!ReadPeakRssMb(metrics.peak_rss_mb, rss_error)) {
        metrics.error = rss_error;
        return metrics;
    }

    const bool writer_ok = writer->IsFinalized()
                        && writer->GetLastError() == WriterError::kOk;
    std::error_code file_error;
    const bool output_ok = fs::is_regular_file(output_path, file_error)
                        && !file_error
                        && fs::file_size(output_path, file_error) > 0
                        && !file_error;
    const bool counts_ok = metrics.total_frames > 0
                        && metrics.inference_frames == metrics.total_frames;
    const bool timings_ok = metrics.total_wall_time_ms > 0.0
                         && metrics.inference_total_time_ms > 0.0;

    metrics.success = pipeline_result.success && writer_ok && output_ok
                   && counts_ok && timings_ok;
    if (!metrics.success) {
        if (!pipeline_result.success) {
            metrics.error = PipelineErrorCodeToString(pipeline_result.error_code)
                          + ": " + pipeline_result.error_message;
        } else if (!writer_ok) {
            metrics.error = "Writer failed: " + writer->GetLastErrorMessage();
        } else if (!output_ok) {
            metrics.error = "Output MP4 missing or empty: " + output_path.string();
        } else if (!counts_ok) {
            metrics.error = "Frame count mismatch: inference="
                          + std::to_string(metrics.inference_frames)
                          + ", written=" + std::to_string(metrics.total_frames);
        } else {
            metrics.error = "Performance timing was not populated";
        }
    }

    return metrics;
}

RunWireResult ToWireResult(const RunMetrics& metrics) {
    RunWireResult wire;
    wire.run_index = metrics.run_index;
    wire.process_id = metrics.process_id;
    wire.success = metrics.success ? 1 : 0;
    wire.total_frames = metrics.total_frames;
    wire.inference_frames = metrics.inference_frames;
    wire.total_wall_time_ms = metrics.total_wall_time_ms;
    wire.inference_total_time_ms = metrics.inference_total_time_ms;
    wire.average_inference_time_ms = metrics.average_inference_time_ms;
    wire.inference_fps = metrics.inference_fps;
    wire.peak_rss_mb = metrics.peak_rss_mb;
    std::snprintf(wire.error, sizeof(wire.error), "%s", metrics.error.c_str());
    return wire;
}

RunMetrics FromWireResult(const RunWireResult& wire) {
    RunMetrics metrics;
    metrics.run_index = wire.run_index;
    metrics.process_id = wire.process_id;
    metrics.success = wire.success != 0;
    metrics.error = wire.error;
    metrics.total_frames = wire.total_frames;
    metrics.inference_frames = wire.inference_frames;
    metrics.total_wall_time_ms = wire.total_wall_time_ms;
    metrics.inference_total_time_ms = wire.inference_total_time_ms;
    metrics.average_inference_time_ms = wire.average_inference_time_ms;
    metrics.inference_fps = wire.inference_fps;
    metrics.peak_rss_mb = wire.peak_rss_mb;
    return metrics;
}

bool WriteAll(int file_descriptor, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::size_t written = 0;
    while (written < size) {
        const ssize_t count = write(file_descriptor, bytes + written, size - written);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        written += static_cast<std::size_t>(count);
    }
    return true;
}

bool ReadAll(int file_descriptor, void* data, std::size_t size) {
    auto* bytes = static_cast<unsigned char*>(data);
    std::size_t received = 0;
    while (received < size) {
        const ssize_t count = read(file_descriptor, bytes + received, size - received);
        if (count == 0) {
            return false;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        received += static_cast<std::size_t>(count);
    }
    return true;
}

RunMetrics RunInIndependentProcess(
    int run_index, int output_fps, int ncnn_threads,
    const BenchmarkPaths& paths) {
    RunMetrics failure;
    failure.run_index = run_index;

    int pipe_descriptors[2] = {-1, -1};
    if (pipe(pipe_descriptors) != 0) {
        failure.error = std::string("pipe failed: ") + std::strerror(errno);
        return failure;
    }

    const pid_t child_pid = fork();
    if (child_pid < 0) {
        failure.error = std::string("fork failed: ") + std::strerror(errno);
        close(pipe_descriptors[0]);
        close(pipe_descriptors[1]);
        return failure;
    }

    if (child_pid == 0) {
        close(pipe_descriptors[0]);
        const RunMetrics metrics = RunOnce(
            run_index, output_fps, ncnn_threads, paths);
        const RunWireResult wire = ToWireResult(metrics);
        const bool sent = WriteAll(pipe_descriptors[1], &wire, sizeof(wire));
        close(pipe_descriptors[1]);
        _exit(sent && metrics.success ? 0 : 1);
    }

    close(pipe_descriptors[1]);
    RunWireResult wire;
    const bool received = ReadAll(pipe_descriptors[0], &wire, sizeof(wire));
    close(pipe_descriptors[0]);

    int child_status = 0;
    pid_t waited = -1;
    do {
        waited = waitpid(child_pid, &child_status, 0);
    } while (waited < 0 && errno == EINTR);

    if (!received) {
        failure.process_id = static_cast<std::int64_t>(child_pid);
        failure.error = "child process exited without a complete benchmark result";
        return failure;
    }

    RunMetrics metrics = FromWireResult(wire);
    if (waited < 0) {
        metrics.success = false;
        metrics.error = std::string("waitpid failed: ") + std::strerror(errno);
    } else if (!WIFEXITED(child_status)) {
        metrics.success = false;
        metrics.error = "benchmark child terminated abnormally";
    } else if (WEXITSTATUS(child_status) != 0 && metrics.success) {
        metrics.success = false;
        metrics.error = "benchmark child returned non-zero status";
    }
    return metrics;
}

void PrintRun(const RunMetrics& metrics, int total_runs) {
    std::cout << "Run " << metrics.run_index << "/" << total_runs << '\n'
              << "  process_id: " << metrics.process_id << '\n'
              << "  success: " << (metrics.success ? "true" : "false") << '\n'
              << "  total_frames: " << metrics.total_frames << '\n'
              << "  inference_frames: " << metrics.inference_frames << '\n'
              << "  total_wall_time_ms: " << metrics.total_wall_time_ms << '\n'
              << "  inference_total_time_ms: " << metrics.inference_total_time_ms << '\n'
              << "  average_inference_time_ms: "
              << metrics.average_inference_time_ms << '\n'
              << "  inference_fps: " << metrics.inference_fps << '\n'
              << "  peak_rss_mb: " << metrics.peak_rss_mb << '\n';
    if (!metrics.success) {
        std::cout << "  error: " << metrics.error << '\n';
    }
    std::cout << std::flush;
}

template <typename Getter>
MetricSummary Summarize(const std::vector<RunMetrics>& runs, Getter getter) {
    std::vector<double> values;
    values.reserve(runs.size());
    for (const auto& run : runs) {
        if (run.success) {
            values.push_back(getter(run));
        }
    }

    MetricSummary summary;
    if (values.empty()) {
        return summary;
    }

    const auto bounds = std::minmax_element(values.begin(), values.end());
    summary.average = std::accumulate(values.begin(), values.end(), 0.0)
                    / static_cast<double>(values.size());
    summary.minimum = *bounds.first;
    summary.maximum = *bounds.second;
    return summary;
}

void PrintSummaryMetric(const char* name, const MetricSummary& summary) {
    std::cout << "  " << name
              << ": avg=" << summary.average
              << ", min=" << summary.minimum
              << ", max=" << summary.maximum << '\n';
}

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program
              << " [--runs N] [--fps N] [--threads N]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    int run_count = 5;
    int output_fps = kDefaultOutputFps;
    int ncnn_threads = kDefaultNcnnThreads;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--runs") {
            if (i + 1 >= argc || !ParsePositiveInt(argv[++i], run_count)) {
                std::cerr << "--runs requires a positive integer\n";
                return 2;
            }
        } else if (arg == "--fps") {
            if (i + 1 >= argc || !ParsePositiveInt(argv[++i], output_fps)) {
                std::cerr << "--fps requires a positive integer\n";
                return 2;
            }
        } else if (arg == "--threads") {
            if (i + 1 >= argc || !ParsePositiveInt(argv[++i], ncnn_threads)) {
                std::cerr << "--threads requires a positive integer\n";
                return 2;
            }
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            PrintUsage(argv[0]);
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(3);
    const BenchmarkPaths paths = DefaultPaths();
    PrintEnvironment(paths, run_count, output_fps, ncnn_threads);

    std::string path_error;
    if (!ValidatePaths(paths, path_error)) {
        std::cerr << "Benchmark input validation failed: " << path_error << '\n';
        return 1;
    }

    std::vector<RunMetrics> runs;
    runs.reserve(static_cast<std::size_t>(run_count));

    bool all_success = true;
    for (int run_index = 1; run_index <= run_count; ++run_index) {
        runs.push_back(RunInIndependentProcess(
            run_index, output_fps, ncnn_threads, paths));
        PrintRun(runs.back(), run_count);
        all_success = all_success && runs.back().success;
    }

    const auto total_wall = Summarize(
        runs, [](const RunMetrics& run) { return run.total_wall_time_ms; });
    const auto average_inference = Summarize(
        runs, [](const RunMetrics& run) { return run.average_inference_time_ms; });
    const auto inference_fps = Summarize(
        runs, [](const RunMetrics& run) { return run.inference_fps; });
    const auto peak_rss = Summarize(
        runs, [](const RunMetrics& run) { return run.peak_rss_mb; });

    std::cout << "Summary (successful runs only)\n";
    PrintSummaryMetric("total_wall_time_ms", total_wall);
    PrintSummaryMetric("average_inference_time_ms", average_inference);
    PrintSummaryMetric("inference_fps", inference_fps);
    PrintSummaryMetric("peak_rss_mb", peak_rss);
    std::cout << std::flush;

    return all_success ? 0 : 1;
}
