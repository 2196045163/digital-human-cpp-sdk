/// @file model_inference_benchmark.cpp
/// @brief 正式 InferenceScheduler batch benchmark，输出 P50/P95/吞吐和 RSS
/// @note 性能结果只适用于本次机器、构建和参数，不属于正确性阈值。

#include "model/inference_scheduler.h"
#include "model/model_loader.h"
#include "model_example_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace dhm = digital_human::model;

struct BenchmarkCase {
    const char* name = "";
    std::size_t worker_count = 0;
    int ncnn_threads = 0;
};

struct IterationSample {
    double batch_makespan_ms = 0.0;
    double throughput_items_per_second = 0.0;
    double average_queue_wait_ms = 0.0;
    double average_compute_ms = 0.0;
    double average_item_total_ms = 0.0;
};

struct MetricSummary {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

struct CaseResult {
    BenchmarkCase config;
    std::uint64_t model_generation = 0;
    double model_load_time_ms = 0.0;
    bool oversubscription_opt_in = false;
    long rss_before_kb = -1;
    long rss_peak_kb = -1;
    long rss_after_kb = -1;
    MetricSummary batch_makespan_ms;
    MetricSummary throughput_items_per_second;
    MetricSummary average_queue_wait_ms;
    MetricSummary average_compute_ms;
    MetricSummary average_item_total_ms;
};

std::size_t ParsePositiveSize(
    const char* text,
    std::size_t fallback) {
    if (text == nullptr) {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0) {
        return fallback;
    }
    return static_cast<std::size_t>(value);
}

long ReadLinuxRssKb() {
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

double NearestRankPercentile(
    const std::vector<double>& sorted_values,
    double percentile) {
    if (sorted_values.empty()) {
        return 0.0;
    }

    const double rank =
        std::ceil(percentile * static_cast<double>(sorted_values.size()));
    const std::size_t index = static_cast<std::size_t>(
        std::max(1.0, rank) - 1.0);
    return sorted_values[std::min(index, sorted_values.size() - 1)];
}

MetricSummary Summarize(std::vector<double> values) {
    MetricSummary summary;
    if (values.empty()) {
        return summary;
    }

    summary.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    summary.p50 = NearestRankPercentile(values, 0.50);
    summary.p95 = NearestRankPercentile(values, 0.95);
    summary.maximum = values.back();
    return summary;
}

MetricSummary SummarizeField(
    const std::vector<IterationSample>& samples,
    double IterationSample::* field) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const IterationSample& sample : samples) {
        values.push_back(sample.*field);
    }
    return Summarize(std::move(values));
}

IterationSample BuildIterationSample(
    const dhm::BatchInferenceResult& batch_result) {
    IterationSample sample;
    sample.batch_makespan_ms =
        batch_result.summary.batch_makespan_ms;
    sample.throughput_items_per_second =
        batch_result.summary.throughput_items_per_second;

    for (const dhm::SingleInferenceResult& item : batch_result.results) {
        sample.average_queue_wait_ms +=
            item.value.timing.queue_wait_ms;
        sample.average_compute_ms +=
            model_example::AttemptComputeMs(item);
        sample.average_item_total_ms +=
            item.value.timing.total_latency_ms +
            item.value.timing.queue_wait_ms;
    }

    const double item_count =
        static_cast<double>(batch_result.results.size());
    if (item_count > 0.0) {
        sample.average_queue_wait_ms /= item_count;
        sample.average_compute_ms /= item_count;
        sample.average_item_total_ms /= item_count;
    }
    return sample;
}

void AppendMetricJson(
    std::ostringstream& json,
    const char* name,
    const MetricSummary& summary,
    bool trailing_comma) {
    json << "      \"" << name << "\": {"
         << "\"mean\": " << summary.mean
         << ", \"p50\": " << summary.p50
         << ", \"p95\": " << summary.p95
         << ", \"max\": " << summary.maximum << "}"
         << (trailing_comma ? "," : "") << "\n";
}

void AppendRssJsonValue(std::ostringstream& json, long rss_kb) {
    if (rss_kb < 0) {
        json << "null";
    } else {
        json << rss_kb;
    }
}

std::string BuildBatchInfoJson(
    const BenchmarkCase& config,
    const dhm::ModelRuntimeSnapshot& snapshot,
    const dhm::BatchInferenceResult& result) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\n"
         << "  \"description\": "
            "\"Formal InferenceScheduler batch result\",\n"
         << "  \"build_mode\": \"" << model_example::BuildMode() << "\",\n"
         << "  \"worker_count\": " << config.worker_count << ",\n"
         << "  \"effective_ncnn_threads\": "
         << snapshot.effective_num_threads << ",\n"
         << "  \"model_generation\": " << result.model_generation << ",\n"
         << "  \"success\": " << (result.success ? "true" : "false") << ",\n"
         << "  \"status\": \""
         << model_example::JsonEscape(
                dhm::ModelInference::StatusToString(result.status))
         << "\",\n"
         << "  \"total_count\": " << result.summary.total_count << ",\n"
         << "  \"accepted_count\": "
         << result.summary.accepted_count << ",\n"
         << "  \"success_count\": "
         << result.summary.success_count << ",\n"
         << "  \"failure_count\": "
         << result.summary.failure_count << ",\n"
         << "  \"queue_timeout_count\": "
         << result.summary.queue_timeout_count << ",\n"
         << "  \"retry_recovered_count\": "
         << result.summary.retry_recovered_count << ",\n"
         << "  \"enqueue_wait_ms\": "
         << result.summary.enqueue_wait_ms << ",\n"
         << "  \"batch_makespan_ms\": "
         << result.summary.batch_makespan_ms << ",\n"
         << "  \"throughput_items_per_second\": "
         << result.summary.throughput_items_per_second << ",\n"
         << "  \"note\": "
            "\"Synthetic batch proves scheduler/result connectivity; "
            "NOT visual quality\"\n"
         << "}\n";
    return json.str();
}

std::string BuildBenchmarkJson(
    const std::filesystem::path& model_path,
    std::size_t batch_size,
    std::size_t warmup_iterations,
    std::size_t measured_iterations,
    unsigned int hardware_threads,
    const std::vector<CaseResult>& results) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\n"
         << "  \"description\": "
            "\"InferenceScheduler CPU benchmark\",\n"
         << "  \"model_path\": \""
         << model_example::JsonEscape(model_path.string()) << "\",\n"
         << "  \"build_mode\": \"" << model_example::BuildMode() << "\",\n"
         << "  \"hardware_concurrency\": " << hardware_threads << ",\n"
         << "  \"batch_size\": " << batch_size << ",\n"
         << "  \"warmup_iterations\": " << warmup_iterations << ",\n"
         << "  \"measured_iterations\": " << measured_iterations << ",\n"
         << "  \"percentile_method\": \"nearest-rank\",\n"
         << "  \"cases\": [\n";

    for (std::size_t index = 0; index < results.size(); ++index) {
        const CaseResult& result = results[index];
        json << "    {\n"
             << "      \"name\": \"" << result.config.name << "\",\n"
             << "      \"worker_count\": "
             << result.config.worker_count << ",\n"
             << "      \"effective_ncnn_threads\": "
             << result.config.ncnn_threads << ",\n"
             << "      \"model_generation\": "
             << result.model_generation << ",\n"
             << "      \"model_load_time_ms\": "
             << result.model_load_time_ms << ",\n"
             << "      \"oversubscription_opt_in\": "
             << (result.oversubscription_opt_in ? "true" : "false")
             << ",\n"
             << "      \"rss_before_kb\": ";
        AppendRssJsonValue(json, result.rss_before_kb);
        json << ",\n      \"rss_peak_kb\": ";
        AppendRssJsonValue(json, result.rss_peak_kb);
        json << ",\n      \"rss_after_kb\": ";
        AppendRssJsonValue(json, result.rss_after_kb);
        json << ",\n";
        AppendMetricJson(
            json,
            "batch_makespan_ms",
            result.batch_makespan_ms,
            true);
        AppendMetricJson(
            json,
            "throughput_items_per_second",
            result.throughput_items_per_second,
            true);
        AppendMetricJson(
            json,
            "average_queue_wait_ms",
            result.average_queue_wait_ms,
            true);
        AppendMetricJson(
            json,
            "average_compute_ms",
            result.average_compute_ms,
            true);
        AppendMetricJson(
            json,
            "average_item_total_ms",
            result.average_item_total_ms,
            false);
        json << "    }" << (index + 1 < results.size() ? "," : "") << "\n";
    }

    json << "  ],\n"
         << "  \"notes\": [\n"
         << "    \"Correctness is covered by tests; benchmark values are "
            "environment-specific\",\n"
         << "    \"RSS is sampled from /proc/self/status when available, "
            "not a guaranteed process peak\",\n"
         << "    \"Cases run sequentially in one process; RSS includes "
            "allocator/cache retained from earlier cases\",\n"
         << "    \"Synthetic tensors do not prove visual quality\"\n"
         << "  ]\n"
         << "}\n";
    return json.str();
}

} // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path model_path =
        argc > 1 ? argv[1] : "models/wav2lip/wav2lip.param";
    const std::size_t measured_iterations =
        ParsePositiveSize(argc > 2 ? argv[2] : nullptr, 10);
    const std::size_t batch_size =
        ParsePositiveSize(argc > 3 ? argv[3] : nullptr, 8);
    constexpr std::size_t kWarmupIterations = 2;
    constexpr auto kQueueWaitTimeout = std::chrono::seconds(5);

    const unsigned int hardware_threads =
        std::thread::hardware_concurrency();
    const std::vector<BenchmarkCase> benchmark_cases = {
        {"worker1_ncnn1", 1, 1},
        {"worker1_ncnn2", 1, 2},
        {"worker2_ncnn1", 2, 1},
    };

    std::cout << "=== InferenceScheduler CPU Benchmark ===\n"
              << "Model: " << model_path << '\n'
              << "Build: " << model_example::BuildMode() << '\n'
              << "Batch / measured iterations: "
              << batch_size << " / " << measured_iterations << '\n';

    dhm::ModelLoader loader;
    std::vector<CaseResult> case_results;
    case_results.reserve(benchmark_cases.size());
    std::string first_batch_json;

    for (const BenchmarkCase& benchmark_case : benchmark_cases) {
        dhm::ModelLoadOptions load_options;
        load_options.backend = dhm::ModelBackend::kCpu;
        load_options.num_threads = benchmark_case.ncnn_threads;
        load_options.enable_warmup = true;

        const dhm::ModelLoadResult load_result =
            loader.Load(model_path, load_options);
        if (!load_result.success) {
            std::cerr << "FAILED to load " << benchmark_case.name << ": "
                      << load_result.error_message << '\n';
            return 1;
        }

        const dhm::ModelRuntimeSnapshot snapshot =
            loader.AcquireSnapshot();
        dhm::SchedulerConfig scheduler_config;
        scheduler_config.worker_count = benchmark_case.worker_count;
        scheduler_config.queue_capacity = batch_size;
        scheduler_config.queue_wait_timeout = kQueueWaitTimeout;

        const std::size_t requested_threads =
            benchmark_case.worker_count *
            static_cast<std::size_t>(benchmark_case.ncnn_threads);
        const bool must_opt_in =
            hardware_threads > 0 &&
            requested_threads > static_cast<std::size_t>(hardware_threads);
        scheduler_config.allow_thread_oversubscription = must_opt_in;

        dhm::InferenceScheduler scheduler;
        const dhm::SchedulerControlResult start_result =
            scheduler.Start(snapshot, scheduler_config);
        if (!start_result.success) {
            std::cerr << "FAILED to start " << benchmark_case.name << ": "
                      << start_result.error_message << '\n';
            return 1;
        }

        std::vector<dhm::NcnnWav2LipInput> inputs;
        inputs.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index) {
            inputs.push_back(model_example::MakeSyntheticInput(
                static_cast<std::int64_t>(index),
                static_cast<std::int64_t>(index * 40)));
        }

        for (std::size_t warmup = 0;
             warmup < kWarmupIterations;
             ++warmup) {
            const dhm::BatchInferenceResult warmup_result =
                scheduler.InferBatch(inputs);
            if (!warmup_result.success) {
                std::cerr << "FAILED scheduler warmup for "
                          << benchmark_case.name << ": "
                          << warmup_result.error_message << '\n';
                return 1;
            }
        }

        CaseResult case_result;
        case_result.config = benchmark_case;
        case_result.model_generation = snapshot.generation;
        case_result.model_load_time_ms = load_result.time_ms;
        case_result.oversubscription_opt_in = must_opt_in;
        case_result.rss_before_kb = ReadLinuxRssKb();
        case_result.rss_peak_kb = case_result.rss_before_kb;

        std::vector<IterationSample> samples;
        samples.reserve(measured_iterations);
        for (std::size_t iteration = 0;
             iteration < measured_iterations;
             ++iteration) {
            const dhm::BatchInferenceResult batch_result =
                scheduler.InferBatch(inputs);
            if (!batch_result.success) {
                std::cerr << "FAILED measured batch for "
                          << benchmark_case.name << ": "
                          << batch_result.error_message << '\n';
                return 1;
            }

            samples.push_back(BuildIterationSample(batch_result));
            const long current_rss_kb = ReadLinuxRssKb();
            if (current_rss_kb >= 0) {
                case_result.rss_peak_kb =
                    std::max(case_result.rss_peak_kb, current_rss_kb);
            }

            if (first_batch_json.empty()) {
                first_batch_json = BuildBatchInfoJson(
                    benchmark_case,
                    snapshot,
                    batch_result);
            }
        }

        const dhm::SchedulerControlResult stop_result = scheduler.Stop();
        if (!stop_result.success) {
            std::cerr << "FAILED to stop " << benchmark_case.name << ": "
                      << stop_result.error_message << '\n';
            return 1;
        }

        case_result.rss_after_kb = ReadLinuxRssKb();
        case_result.batch_makespan_ms = SummarizeField(
            samples,
            &IterationSample::batch_makespan_ms);
        case_result.throughput_items_per_second = SummarizeField(
            samples,
            &IterationSample::throughput_items_per_second);
        case_result.average_queue_wait_ms = SummarizeField(
            samples,
            &IterationSample::average_queue_wait_ms);
        case_result.average_compute_ms = SummarizeField(
            samples,
            &IterationSample::average_compute_ms);
        case_result.average_item_total_ms = SummarizeField(
            samples,
            &IterationSample::average_item_total_ms);
        case_results.push_back(case_result);

        std::cout << std::fixed << std::setprecision(3)
                  << benchmark_case.name
                  << ": makespan p50/p95="
                  << case_result.batch_makespan_ms.p50 << "/"
                  << case_result.batch_makespan_ms.p95
                  << " ms, throughput mean="
                  << case_result.throughput_items_per_second.mean
                  << " items/s\n";
    }

    const bool batch_written = model_example::WriteTextFile(
        "golden_output/model_inference_batch_info.json",
        first_batch_json);
    const bool benchmark_written = model_example::WriteTextFile(
        "golden_output/model_inference_benchmark.json",
        BuildBenchmarkJson(
            model_path,
            batch_size,
            kWarmupIterations,
            measured_iterations,
            hardware_threads,
            case_results));

    if (!batch_written || !benchmark_written) {
        return 1;
    }

    std::cout << "=== BENCHMARK COMPLETED ===\n";
    return 0;
}
