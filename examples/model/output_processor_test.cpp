#include "model/output_processor.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "model/detail/wav2lip_model_spec.h"

namespace {

digital_human::model::InferenceOutput MakeSyntheticInferenceOutput() {
    using digital_human::model::detail::Wav2LipModelSpec;

    digital_human::model::InferenceOutput input;
    input.pred = ncnn::Mat(
        Wav2LipModelSpec::kPredWidth,
        Wav2LipModelSpec::kPredHeight,
        Wav2LipModelSpec::kPredChannels);
    float* blue_plane = input.pred.channel(Wav2LipModelSpec::kBluePlaneIndex);
    float* green_plane = input.pred.channel(Wav2LipModelSpec::kGreenPlaneIndex);
    float* red_plane = input.pred.channel(Wav2LipModelSpec::kRedPlaneIndex);

    for (int row_index = 0; row_index < Wav2LipModelSpec::kPredHeight; ++row_index) {
        for (int column_index = 0;
             column_index < Wav2LipModelSpec::kPredWidth;
             ++column_index) {
            const int pixel_index = row_index * Wav2LipModelSpec::kPredWidth + column_index;
            blue_plane[pixel_index] = static_cast<float>(column_index) /
                static_cast<float>(Wav2LipModelSpec::kPredWidth - 1);
            green_plane[pixel_index] = static_cast<float>(row_index) /
                static_cast<float>(Wav2LipModelSpec::kPredHeight - 1);
            red_plane[pixel_index] = static_cast<float>(row_index + column_index) /
                static_cast<float>(Wav2LipModelSpec::kPredWidth + Wav2LipModelSpec::kPredHeight - 2);
        }
    }

    input.metadata.pts_ms = 1234;
    input.metadata.frame_index = 42;
    input.model_generation = 7;
    return input;
}

bool WriteInfoJson(const digital_human::model::OutputProcessResult& result,
                   const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream output(path);
    if (!output.is_open()) {
        return false;
    }

    const auto& value = result.value;
    output << std::boolalpha
           << "{\n"
           << "  \"success\": " << result.success << ",\n"
           << "  \"status\": \""
           << digital_human::model::OutputProcessor::StatusToString(result.status) << "\",\n"
           << "  \"image\": \"96x96 CV_8UC3 BGR\",\n"
           << "  \"corrected_value_count\": "
           << value.conversion_info.corrected_value_count << ",\n"
           << "  \"sharpness_score\": " << std::fixed << std::setprecision(3)
           << value.conversion_info.sharpness_score << ",\n"
           << "  \"sharpness_threshold_applied\": "
           << value.conversion_info.sharpness_threshold_applied << ",\n"
           << "  \"sharpness_passed\": "
           << value.conversion_info.sharpness_passed << ",\n"
           << "  \"conversion_ms\": " << value.conversion_info.conversion_ms << ",\n"
           << "  \"pts_ms\": " << value.metadata.pts_ms.value_or(-1) << ",\n"
           << "  \"frame_index\": " << value.metadata.frame_index.value_or(-1) << ",\n"
           << "  \"model_generation\": " << value.model_generation << "\n"
           << "}\n";
    return output.good();
}

} // namespace

int main() {
    const digital_human::model::InferenceOutput input = MakeSyntheticInferenceOutput();
    digital_human::model::OutputProcessor processor;
    const digital_human::model::OutputProcessResult result = processor.Convert(input);
    if (!result.success) {
        std::cerr << "OutputProcessor failed: " << result.error_message << '\n';
        return 1;
    }

    const std::string image_path = "golden_output/model_output_synthetic.png";
    const std::string info_path = "golden_output/model_output_example_info.json";
    std::filesystem::create_directories("golden_output");
    if (!cv::imwrite(image_path, result.value.generated_face_bgr) ||
        !WriteInfoJson(result, info_path)) {
        std::cerr << "Failed to save OutputProcessor golden output\n";
        return 1;
    }

    const auto& info = result.value.conversion_info;
    std::cout << std::boolalpha
              << "output_process success=" << result.success
              << " status=" << digital_human::model::OutputProcessor::StatusToString(result.status)
              << "\nimage=96x96 CV_8UC3 BGR"
              << " corrected_count=" << info.corrected_value_count
              << "\nsharpness_score=" << info.sharpness_score
              << " threshold_passed=" << info.sharpness_passed
              << "\npts_ms=" << result.value.metadata.pts_ms.value_or(-1)
              << " frame_index=" << result.value.metadata.frame_index.value_or(-1)
              << " generation=" << result.value.model_generation
              << "\nconversion_ms=" << info.conversion_ms
              << "\nsaved=" << image_path << " and " << info_path << '\n';
    return 0;
}
