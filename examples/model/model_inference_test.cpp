/// @file model_inference_test.cpp
/// @brief 正式 ModelInference 单样本示例：Snapshot + 合法 tensor -> 守门后的 pred
/// @note 合成输入只证明正式 API、诊断和输出契约，不证明真实口型视觉质量。

#include "model/model_inference.h"
#include "model/model_loader.h"
#include "model_example_utils.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

namespace dhm = digital_human::model;

std::string OptionalStatusJson(
    const std::optional<dhm::InferenceStatus>& status) {
    if (!status.has_value()) {
        return "null";
    }

    return "\"" + model_example::JsonEscape(
        dhm::ModelInference::StatusToString(*status)) + "\"";
}

std::string BuildResultJson(
    const std::filesystem::path& model_path,
    const dhm::ModelLoadResult& load_result,
    const dhm::ModelRuntimeSnapshot& snapshot,
    const dhm::SingleInferenceResult& result) {
    const dhm::InferenceOutput& value = result.value;
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\n"
         << "  \"description\": "
            "\"Formal ModelInference single-sample result\",\n"
         << "  \"build_mode\": \"" << model_example::BuildMode() << "\",\n"
         << "  \"model_path\": \""
         << model_example::JsonEscape(model_path.string()) << "\",\n"
         << "  \"backend\": \"CPU\",\n"
         << "  \"model_load_time_ms\": " << load_result.time_ms << ",\n"
         << "  \"model_generation\": " << snapshot.generation << ",\n"
         << "  \"effective_ncnn_threads\": "
         << snapshot.effective_num_threads << ",\n"
         << "  \"input_mel_shape\": \"16x80x1 unpacked FP32\",\n"
         << "  \"input_face_shape\": \"96x96x6 unpacked FP32\",\n"
         << "  \"success\": " << (result.success ? "true" : "false") << ",\n"
         << "  \"status\": \""
         << model_example::JsonEscape(
                dhm::ModelInference::StatusToString(result.status))
         << "\",\n"
         << "  \"error_message\": \""
         << model_example::JsonEscape(result.error_message) << "\",\n"
         << "  \"pred_empty\": "
         << (value.pred.empty() ? "true" : "false") << ",\n"
         << "  \"pred_shape\": \""
         << value.output_info.width << "x"
         << value.output_info.height << "x"
         << value.output_info.channels << "\",\n"
         << "  \"pred_range\": ["
         << value.output_info.min_value << ", "
         << value.output_info.max_value << "],\n"
         << "  \"pred_all_finite\": "
         << (value.output_info.all_finite ? "true" : "false") << ",\n"
         << "  \"pred_within_expected_range\": "
         << (value.output_info.within_expected_range ? "true" : "false")
         << ",\n"
         << "  \"attempt_count\": "
         << value.attempts.attempt_count << ",\n"
         << "  \"first_failure_status\": "
         << OptionalStatusJson(value.attempts.first_failure_status) << ",\n"
         << "  \"final_attempt_status\": \""
         << model_example::JsonEscape(
                dhm::ModelInference::StatusToString(
                    value.attempts.final_attempt_status))
         << "\",\n"
         << "  \"recovered_by_retry\": "
         << (value.attempts.recovered_by_retry ? "true" : "false") << ",\n"
         << "  \"timing_ms\": {\n"
         << "    \"input_validation\": "
         << value.timing.input_validation_ms << ",\n"
         << "    \"first_attempt\": "
         << value.timing.first_attempt_ms << ",\n"
         << "    \"retry_attempts\": "
         << value.timing.retry_attempts_ms << ",\n"
         << "    \"output_validation\": "
         << value.timing.output_validation_ms << ",\n"
         << "    \"total_latency\": "
         << value.timing.total_latency_ms << "\n"
         << "  },\n"
         << "  \"metadata\": {\n"
         << "    \"frame_index\": "
         << value.metadata.frame_index.value_or(-1) << ",\n"
         << "    \"pts_ms\": "
         << value.metadata.pts_ms.value_or(-1) << "\n"
         << "  },\n"
         << "  \"note\": "
            "\"Synthetic legal tensors prove API and guard connectivity; "
            "NOT visual quality\"\n"
         << "}\n";
    return json.str();
}

} // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path model_path =
        argc > 1 ? argv[1] : "models/wav2lip/wav2lip.param";
    const std::filesystem::path output_path =
        argc > 2
            ? argv[2]
            : "golden_output/model_inference_single_info.json";

    std::cout << "=== Formal ModelInference Single-Sample Example ===\n";
    std::cout << "Model: " << model_path << '\n';

    dhm::ModelLoader loader;
    dhm::ModelLoadOptions load_options;
    load_options.backend = dhm::ModelBackend::kCpu;
    load_options.num_threads = 1;
    load_options.enable_warmup = true;

    const dhm::ModelLoadResult load_result =
        loader.Load(model_path, load_options);
    if (!load_result.success) {
        std::cerr << "FAILED to load model: "
                  << load_result.error_message << '\n';
        return 1;
    }

    const dhm::ModelRuntimeSnapshot snapshot = loader.AcquireSnapshot();
    if (!snapshot.IsValid()) {
        std::cerr << "FAILED: ModelLoader returned an invalid snapshot\n";
        return 1;
    }

    const dhm::NcnnWav2LipInput input =
        model_example::MakeSyntheticInput(0, 0);
    dhm::InferenceOptions inference_options;
    inference_options.retry.max_retries = 1;
    inference_options.light_mode = true;

    const dhm::ModelInference inference;
    const dhm::SingleInferenceResult result =
        inference.Infer(snapshot, input, inference_options);

    std::cout << "Status: "
              << dhm::ModelInference::StatusToString(result.status) << '\n';
    std::cout << "Generation / ncnn threads: "
              << result.value.model_generation << " / "
              << snapshot.effective_num_threads << '\n';
    std::cout << "Pred shape: "
              << result.value.output_info.width << "x"
              << result.value.output_info.height << "x"
              << result.value.output_info.channels << '\n';
    std::cout << "Attempts / recovered: "
              << result.value.attempts.attempt_count << " / "
              << (result.value.attempts.recovered_by_retry ? "true" : "false")
              << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "First attempt / total: "
              << result.value.timing.first_attempt_ms << " / "
              << result.value.timing.total_latency_ms << " ms\n";

    if (!model_example::WriteTextFile(
            output_path,
            BuildResultJson(
                model_path,
                load_result,
                snapshot,
                result))) {
        return 1;
    }

    if (!result.success) {
        std::cerr << "FAILED inference: " << result.error_message << '\n';
        return 1;
    }

    std::cout << "=== MODEL INFERENCE EXAMPLE PASSED ===\n";
    return 0;
}
