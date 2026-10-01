#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
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
#include <thread>
#include <type_traits>
#include <vector>

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#ifdef DIGITAL_HUMAN_BENCHMARK_NVML
#include <nvml.h>
#endif

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

enum class BenchmarkBackend {
    kNcnnCpu,
    kLibTorchCuda
};

struct BenchmarkPaths {
    fs::path image;
    fs::path audio;
    fs::path model_param;
    fs::path model_bin;
    fs::path torchscript_model;
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
    double e2e_fps = 0.0;
    double inference_total_time_ms = 0.0;
    double average_inference_time_ms = 0.0;
    double inference_fps = 0.0;
    double peak_rss_mb = 0.0;
    double h2d_time_ms = 0.0;
    double cuda_forward_time_ms = 0.0;
    double d2h_time_ms = 0.0;
    double gpu_backend_total_time_ms = 0.0;
    double gpu_peak_memory_mb = 0.0;
    double gpu_memory_peak_usage_mb = 0.0;
    double gpu_inference_memory_delta_mb = 0.0;
};

struct RunWireResult {
    std::int32_t run_index = 0;
    std::int64_t process_id = 0;
    std::int32_t success = 0;
    std::int64_t total_frames = 0;
    std::int64_t inference_frames = 0;
    double total_wall_time_ms = 0.0;
    double e2e_fps = 0.0;
    double inference_total_time_ms = 0.0;
    double average_inference_time_ms = 0.0;
    double inference_fps = 0.0;
    double peak_rss_mb = 0.0;
    double h2d_time_ms = 0.0;
    double cuda_forward_time_ms = 0.0;
    double d2h_time_ms = 0.0;
    double gpu_backend_total_time_ms = 0.0;
    double gpu_peak_memory_mb = 0.0;
    double gpu_memory_peak_usage_mb = 0.0;
    double gpu_inference_memory_delta_mb = 0.0;
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

bool ValidatePaths(const BenchmarkPaths& paths, BenchmarkBackend backend,
                   std::string& error) {
    const auto require_file = [&error](const char* name, const fs::path& path) {
        if (!path.empty() && fs::is_regular_file(path)) {
            return true;
        }
        error = std::string("Missing ") + name + ": " + path.string();
        return false;
    };

    const std::pair<const char*, const fs::path*> common_required[] = {
        {"image", &paths.image},
        {"audio", &paths.audio},
        {"landmark model", &paths.landmark_model},
    };

    for (const auto& item : common_required) {
        if (!require_file(item.first, *item.second)) {
            return false;
        }
    }

    if (backend == BenchmarkBackend::kNcnnCpu) {
        return require_file("model param", paths.model_param)
            && require_file("model bin", paths.model_bin);
    }
    return require_file("TorchScript model", paths.torchscript_model);
}

void PrintEnvironment(const BenchmarkPaths& paths, int run_count, int output_fps,
                      int ncnn_threads, BenchmarkBackend backend) {
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
              << "  inference_backend: "
              << (backend == BenchmarkBackend::kNcnnCpu
                      ? "ncnn CPU" : "LibTorch CUDA") << '\n';
    if (backend == BenchmarkBackend::kNcnnCpu) {
        std::cout << "  ncnn_threads: " << ncnn_threads << '\n';
    } else {
        std::cout << "  torchscript_model_path: "
                  << paths.torchscript_model.string() << '\n';
    }
    std::cout
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

#ifdef DIGITAL_HUMAN_BENCHMARK_NVML
class NvmlTaskMemorySampler {
public:
    ~NvmlTaskMemorySampler() {
        double ignored_peak_delta_mb = 0.0;
        std::string ignored_error;
        Stop(ignored_peak_delta_mb, ignored_error);
    }

    bool Start(unsigned int device_index, std::string& error) {
        const nvmlReturn_t init_status = nvmlInit_v2();
        if (init_status != NVML_SUCCESS) {
            error = std::string("nvmlInit_v2 failed: ") + nvmlErrorString(init_status);
            return false;
        }
        initialized_ = true;

        const nvmlReturn_t device_status =
            nvmlDeviceGetHandleByIndex_v2(device_index, &device_);
        if (device_status != NVML_SUCCESS) {
            error = std::string("nvmlDeviceGetHandleByIndex_v2 failed: ")
                  + nvmlErrorString(device_status);
            nvmlShutdown();
            initialized_ = false;
            return false;
        }

        if (!Sample(baseline_used_mb_, error)) {
            nvmlShutdown();
            initialized_ = false;
            return false;
        }
        peak_used_mb_ = baseline_used_mb_;
        stop_requested_.store(false, std::memory_order_release);

        try {
            sampler_thread_ = std::thread([this]() {
                while (!stop_requested_.load(std::memory_order_acquire)) {
                    double used_mb = 0.0;
                    std::string error;
                    if (!Sample(used_mb, error)) {
                        sample_error_ = std::move(error);
                        break;
                    }
                    peak_used_mb_ = std::max(peak_used_mb_, used_mb);
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            });
            started_ = true;
            return true;
        } catch (const std::exception& exception) {
            error = std::string("NVML sampler thread creation failed: ")
                  + exception.what();
            nvmlShutdown();
            initialized_ = false;
            return false;
        }
    }

    bool Stop(double& peak_delta_mb, std::string& error) {
        if (!initialized_) {
            return true;
        }

        if (started_) {
            stop_requested_.store(true, std::memory_order_release);
            if (sampler_thread_.joinable()) {
                sampler_thread_.join();
            }
            started_ = false;
        }

        double final_used_mb = 0.0;
        std::string final_sample_error;
        if (Sample(final_used_mb, final_sample_error)) {
            peak_used_mb_ = std::max(peak_used_mb_, final_used_mb);
        } else if (sample_error_.empty()) {
            sample_error_ = std::move(final_sample_error);
        }

        peak_delta_mb = std::max(0.0, peak_used_mb_ - baseline_used_mb_);
        const nvmlReturn_t shutdown_status = nvmlShutdown();
        initialized_ = false;

        if (!sample_error_.empty()) {
            error = sample_error_;
            return false;
        }
        if (shutdown_status != NVML_SUCCESS) {
            error = std::string("nvmlShutdown failed: ")
                  + nvmlErrorString(shutdown_status);
            return false;
        }
        return true;
    }

private:
    bool Sample(double& used_mb, std::string& error) const {
        nvmlMemory_t memory{};
        const nvmlReturn_t status = nvmlDeviceGetMemoryInfo(device_, &memory);
        if (status != NVML_SUCCESS) {
            error = std::string("nvmlDeviceGetMemoryInfo failed: ")
                  + nvmlErrorString(status);
            return false;
        }
        used_mb = static_cast<double>(memory.used) / (1024.0 * 1024.0);
        return true;
    }

    nvmlDevice_t device_ = nullptr;
    bool initialized_ = false;
    bool started_ = false;
    std::atomic<bool> stop_requested_{false};
    std::thread sampler_thread_;
    double baseline_used_mb_ = 0.0;
    double peak_used_mb_ = 0.0;
    std::string sample_error_;
};
#endif

RunMetrics RunOnce(int run_index, int output_fps, int ncnn_threads,
                   BenchmarkBackend backend, const BenchmarkPaths& paths) {
    using digital_human::audio::AudioLoader;
    using digital_human::output::FinalMediaWriter;
    using digital_human::output::WriterConfig;
    using digital_human::output::WriterError;
    using digital_human::pipeline::DigitalHumanPipeline;
    using digital_human::pipeline::InferenceBackend;
    using digital_human::pipeline::PipelineConfig;
    using digital_human::pipeline::PipelineErrorCodeToString;

    RunMetrics metrics;
    metrics.run_index = run_index;
    metrics.process_id = static_cast<std::int64_t>(getpid());

#ifdef DIGITAL_HUMAN_BENCHMARK_NVML
    NvmlTaskMemorySampler task_memory_sampler;
    if (backend == BenchmarkBackend::kLibTorchCuda
        && !task_memory_sampler.Start(0, metrics.error)) {
        ReadPeakRssMb(metrics.peak_rss_mb, metrics.error);
        return metrics;
    }
#else
    if (backend == BenchmarkBackend::kLibTorchCuda) {
        metrics.error = "GPU benchmark requires an NVML-enabled build";
        ReadPeakRssMb(metrics.peak_rss_mb, metrics.error);
        return metrics;
    }
#endif

    if (!ValidatePaths(paths, backend, metrics.error)) {
        ReadPeakRssMb(metrics.peak_rss_mb, metrics.error);
        return metrics;
    }

    const std::string backend_name =
        backend == BenchmarkBackend::kNcnnCpu ? "cpu" : "gpu";
    const std::string output_name = output_fps == kDefaultOutputFps
        ? "digital_human_" + backend_name + "_benchmark_run_"
            + std::to_string(run_index) + ".mp4"
        : "digital_human_" + backend_name + "_benchmark_"
            + std::to_string(output_fps) + "fps_run_"
            + std::to_string(run_index) + ".mp4";
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
    pipeline_config.landmark_model_path = paths.landmark_model;
    pipeline_config.fps_num = output_fps;
    pipeline_config.fps_den = kOutputFpsDen;
    if (backend == BenchmarkBackend::kNcnnCpu) {
        pipeline_config.inference_backend = InferenceBackend::kNcnnCpu;
        pipeline_config.model_param_path = paths.model_param;
        pipeline_config.model_bin_path = paths.model_bin;
        pipeline_config.scheduler_worker_count = 1;
        pipeline_config.ncnn_threads = ncnn_threads;
    } else {
        pipeline_config.inference_backend = InferenceBackend::kLibTorchCuda;
        pipeline_config.torchscript_model_path = paths.torchscript_model;
    }

    DigitalHumanPipeline pipeline;
    auto pipeline_result = pipeline.Start(pipeline_config, writer);
    if (pipeline_result.success) {
        pipeline_result = pipeline.Wait();
    }

    metrics.total_frames = writer->GetWrittenFrameCount();
    metrics.inference_frames = backend == BenchmarkBackend::kNcnnCpu
        ? pipeline_result.stats.scheduler_accepted_count
        : pipeline_result.stats.rendered_unique_frame_count;
    metrics.total_wall_time_ms = pipeline_result.stats.total_wall_time_ms;
    metrics.inference_total_time_ms = pipeline_result.stats.inference_total_time_ms;
    metrics.h2d_time_ms = pipeline_result.stats.h2d_time_ms;
    metrics.cuda_forward_time_ms = pipeline_result.stats.cuda_forward_time_ms;
    metrics.d2h_time_ms = pipeline_result.stats.d2h_time_ms;
    metrics.gpu_backend_total_time_ms =
        pipeline_result.stats.gpu_backend_total_time_ms;
    metrics.gpu_peak_memory_mb = pipeline_result.stats.gpu_peak_memory_mb;
    metrics.gpu_inference_memory_delta_mb =
        pipeline_result.stats.gpu_inference_memory_delta_mb;

    if (metrics.total_frames > 0 && metrics.total_wall_time_ms > 0.0) {
        metrics.e2e_fps = static_cast<double>(metrics.total_frames) * 1000.0 /
            metrics.total_wall_time_ms;
    }

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

#ifdef DIGITAL_HUMAN_BENCHMARK_NVML
    if (backend == BenchmarkBackend::kLibTorchCuda
        && !task_memory_sampler.Stop(metrics.gpu_memory_peak_usage_mb, metrics.error)) {
        return metrics;
    }
#endif

    const bool writer_ok = writer->IsFinalized()
                        && writer->GetLastError() == WriterError::kOk;
    std::error_code file_error;
    const bool output_ok = fs::is_regular_file(output_path, file_error)
                        && !file_error
                        && fs::file_size(output_path, file_error) > 0
                        && !file_error;
    const bool counts_ok = metrics.total_frames > 0
                        && metrics.inference_frames == metrics.total_frames;
    const bool common_timings_ok = metrics.total_wall_time_ms > 0.0
                                && metrics.inference_total_time_ms > 0.0;
    const bool gpu_timings_ok = backend == BenchmarkBackend::kNcnnCpu
        || (metrics.h2d_time_ms > 0.0
            && metrics.cuda_forward_time_ms > 0.0
            && metrics.d2h_time_ms > 0.0
            && metrics.gpu_backend_total_time_ms > 0.0
            && metrics.gpu_peak_memory_mb > 0.0
            && metrics.gpu_memory_peak_usage_mb > 0.0
            && metrics.gpu_inference_memory_delta_mb > 0.0);
    const bool timings_ok = common_timings_ok && gpu_timings_ok;

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
    wire.e2e_fps = metrics.e2e_fps;
    wire.inference_total_time_ms = metrics.inference_total_time_ms;
    wire.average_inference_time_ms = metrics.average_inference_time_ms;
    wire.inference_fps = metrics.inference_fps;
    wire.peak_rss_mb = metrics.peak_rss_mb;
    wire.h2d_time_ms = metrics.h2d_time_ms;
    wire.cuda_forward_time_ms = metrics.cuda_forward_time_ms;
    wire.d2h_time_ms = metrics.d2h_time_ms;
    wire.gpu_backend_total_time_ms = metrics.gpu_backend_total_time_ms;
    wire.gpu_peak_memory_mb = metrics.gpu_peak_memory_mb;
    wire.gpu_memory_peak_usage_mb = metrics.gpu_memory_peak_usage_mb;
    wire.gpu_inference_memory_delta_mb = metrics.gpu_inference_memory_delta_mb;
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
    metrics.e2e_fps = wire.e2e_fps;
    metrics.inference_total_time_ms = wire.inference_total_time_ms;
    metrics.average_inference_time_ms = wire.average_inference_time_ms;
    metrics.inference_fps = wire.inference_fps;
    metrics.peak_rss_mb = wire.peak_rss_mb;
    metrics.h2d_time_ms = wire.h2d_time_ms;
    metrics.cuda_forward_time_ms = wire.cuda_forward_time_ms;
    metrics.d2h_time_ms = wire.d2h_time_ms;
    metrics.gpu_backend_total_time_ms = wire.gpu_backend_total_time_ms;
    metrics.gpu_peak_memory_mb = wire.gpu_peak_memory_mb;
    metrics.gpu_memory_peak_usage_mb = wire.gpu_memory_peak_usage_mb;
    metrics.gpu_inference_memory_delta_mb = wire.gpu_inference_memory_delta_mb;
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
    BenchmarkBackend backend, const BenchmarkPaths& paths) {
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
            run_index, output_fps, ncnn_threads, backend, paths);
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

void PrintRun(const RunMetrics& metrics, int total_runs,
              BenchmarkBackend backend) {
    std::cout << "Run " << metrics.run_index << "/" << total_runs << '\n'
              << "  process_id: " << metrics.process_id << '\n'
              << "  success: " << (metrics.success ? "true" : "false") << '\n'
              << "  total_frames: " << metrics.total_frames << '\n'
              << "  inference_frames: " << metrics.inference_frames << '\n'
              << "  total_wall_time_ms: " << metrics.total_wall_time_ms << '\n'
              << "  e2e_fps: " << metrics.e2e_fps << '\n'
              << "  inference_total_time_ms: " << metrics.inference_total_time_ms << '\n'
              << "  average_inference_time_ms: "
              << metrics.average_inference_time_ms << '\n'
              << "  inference_fps: " << metrics.inference_fps << '\n'
              << "  peak_rss_mb: " << metrics.peak_rss_mb << '\n';
    if (backend == BenchmarkBackend::kLibTorchCuda) {
        std::cout << "  h2d_time_ms: " << metrics.h2d_time_ms << '\n'
                  << "  cuda_forward_time_ms: " << metrics.cuda_forward_time_ms << '\n'
                  << "  d2h_time_ms: " << metrics.d2h_time_ms << '\n'
                  << "  gpu_backend_total_time_ms: "
                  << metrics.gpu_backend_total_time_ms << '\n'
                  << "  torch_allocated_memory_peak_mb: "
                  << metrics.gpu_peak_memory_mb << '\n'
                  << "  gpu_memory_peak_usage_mb: "
                  << metrics.gpu_memory_peak_usage_mb << '\n'
                  << "  gpu_inference_memory_delta_mb: "
                  << metrics.gpu_inference_memory_delta_mb << '\n';
    }
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
              << " [--runs N] [--fps N] [--threads N]"
                 " [--backend ncnn|libtorch-cuda]"
                 " [--torchscript-model PATH]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    int run_count = 5;
    int output_fps = kDefaultOutputFps;
    int ncnn_threads = kDefaultNcnnThreads;
    BenchmarkBackend backend = BenchmarkBackend::kNcnnCpu;
    fs::path torchscript_model;
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
        } else if (arg == "--backend") {
            if (i + 1 >= argc) {
                std::cerr << "--backend requires ncnn or libtorch-cuda\n";
                return 2;
            }
            const std::string value(argv[++i]);
            if (value == "ncnn") {
                backend = BenchmarkBackend::kNcnnCpu;
            } else if (value == "libtorch-cuda") {
                backend = BenchmarkBackend::kLibTorchCuda;
            } else {
                std::cerr << "--backend requires ncnn or libtorch-cuda\n";
                return 2;
            }
        } else if (arg == "--torchscript-model") {
            if (i + 1 >= argc || argv[i + 1][0] == '\0') {
                std::cerr << "--torchscript-model requires a path\n";
                return 2;
            }
            torchscript_model = argv[++i];
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
    BenchmarkPaths paths = DefaultPaths();
    paths.torchscript_model = std::move(torchscript_model);
    PrintEnvironment(paths, run_count, output_fps, ncnn_threads, backend);

    std::string path_error;
    if (!ValidatePaths(paths, backend, path_error)) {
        std::cerr << "Benchmark input validation failed: " << path_error << '\n';
        return 1;
    }

    std::vector<RunMetrics> runs;
    runs.reserve(static_cast<std::size_t>(run_count));

    bool all_success = true;
    for (int run_index = 1; run_index <= run_count; ++run_index) {
        runs.push_back(RunInIndependentProcess(
            run_index, output_fps, ncnn_threads, backend, paths));
        PrintRun(runs.back(), run_count, backend);
        all_success = all_success && runs.back().success;
    }

    const auto total_wall = Summarize(
        runs, [](const RunMetrics& run) { return run.total_wall_time_ms; });
    const auto e2e_fps = Summarize(
        runs, [](const RunMetrics& run) { return run.e2e_fps; });
    const auto average_inference = Summarize(
        runs, [](const RunMetrics& run) { return run.average_inference_time_ms; });
    const auto inference_fps = Summarize(
        runs, [](const RunMetrics& run) { return run.inference_fps; });
    const auto peak_rss = Summarize(
        runs, [](const RunMetrics& run) { return run.peak_rss_mb; });

    std::cout << "Summary (successful runs only)\n";
    PrintSummaryMetric("total_wall_time_ms", total_wall);
    PrintSummaryMetric("e2e_fps", e2e_fps);
    PrintSummaryMetric("average_inference_time_ms", average_inference);
    PrintSummaryMetric("inference_fps", inference_fps);
    PrintSummaryMetric("peak_rss_mb", peak_rss);
    if (backend == BenchmarkBackend::kLibTorchCuda) {
        PrintSummaryMetric("h2d_time_ms", Summarize(
            runs, [](const RunMetrics& run) { return run.h2d_time_ms; }));
        PrintSummaryMetric("cuda_forward_time_ms", Summarize(
            runs, [](const RunMetrics& run) { return run.cuda_forward_time_ms; }));
        PrintSummaryMetric("d2h_time_ms", Summarize(
            runs, [](const RunMetrics& run) { return run.d2h_time_ms; }));
        PrintSummaryMetric("gpu_backend_total_time_ms", Summarize(
            runs, [](const RunMetrics& run) { return run.gpu_backend_total_time_ms; }));
        PrintSummaryMetric("torch_allocated_memory_peak_mb", Summarize(
            runs, [](const RunMetrics& run) { return run.gpu_peak_memory_mb; }));
        PrintSummaryMetric("gpu_memory_peak_usage_mb", Summarize(
            runs, [](const RunMetrics& run) { return run.gpu_memory_peak_usage_mb; }));
        PrintSummaryMetric("gpu_inference_memory_delta_mb", Summarize(
            runs, [](const RunMetrics& run) {
                return run.gpu_inference_memory_delta_mb;
            }));
    }
    std::cout << std::flush;

    return all_success ? 0 : 1;
}
