#include "model/libtorch_cuda_wav2lip_runtime.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>

namespace {

constexpr int kFaceChannels = 6;
constexpr int kFaceHeight = 96;
constexpr int kFaceWidth = 96;
constexpr int kMelBins = 80;
constexpr int kMelFrames = 16;

digital_human::model::Wav2LipInputData MakeInput() {
    digital_human::model::Wav2LipInputData input;
    input.face_chw.resize(
        static_cast<std::size_t>(kFaceChannels) * kFaceHeight * kFaceWidth);
    input.mel_freq_time.resize(static_cast<std::size_t>(kMelBins) * kMelFrames);
    input.metadata.pts_ms = 4000;
    input.metadata.frame_index = 100;

    const std::size_t plane_size =
        static_cast<std::size_t>(kFaceHeight) * kFaceWidth;
    for (int channel = 0; channel < 3; ++channel) {
        const float channel_value = 0.35f + static_cast<float>(channel) * 0.1f;
        for (int row = 0; row < kFaceHeight; ++row) {
            for (int column = 0; column < kFaceWidth; ++column) {
                const std::size_t pixel_index =
                    static_cast<std::size_t>(row) * kFaceWidth + column;
                input.face_chw[static_cast<std::size_t>(channel) * plane_size + pixel_index] =
                    row < kFaceHeight / 2 ? channel_value : 0.0f;
                input.face_chw[static_cast<std::size_t>(channel + 3) * plane_size + pixel_index] =
                    channel_value;
            }
        }
    }

    for (int frequency = 0; frequency < kMelBins; ++frequency) {
        for (int frame = 0; frame < kMelFrames; ++frame) {
            input.mel_freq_time[static_cast<std::size_t>(frequency) * kMelFrames + frame] =
                -4.0f + 2.0f * static_cast<float>(frame) /
                    static_cast<float>(kMelFrames - 1);
        }
    }
    return input;
}

std::string FormatShape(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            text += ",";
        }
        text += std::to_string(shape[index]);
    }
    text += "]";
    return text;
}

} // namespace

int main() {
    using digital_human::model::LibTorchCudaWav2LipRuntime;

    const bool cuda_available = LibTorchCudaWav2LipRuntime::IsCudaAvailable();
    std::cout << "cuda_available=" << (cuda_available ? "true" : "false") << '\n';
    if (!cuda_available) {
        std::cerr << "CUDA is unavailable\n";
        return 1;
    }

    LibTorchCudaWav2LipRuntime runtime;
    const auto load_result = runtime.Load(DIGITAL_HUMAN_LIBTORCH_WAV2LIP_MODEL);
    if (!load_result.success) {
        std::cerr << "model_load_error=" << load_result.error_message << '\n';
        return 2;
    }

    const auto inference_result = runtime.Infer(MakeInput());
    std::cout << "output_shape=" << FormatShape(inference_result.info.output_shape) << '\n';
    std::cout << "output_was_cuda="
              << (inference_result.info.output_was_cuda ? "true" : "false") << '\n';
    std::cout << "h2d_time_ms=" << inference_result.timing.h2d_time_ms << '\n';
    std::cout << "cuda_forward_time_ms="
              << inference_result.timing.cuda_forward_time_ms << '\n';
    std::cout << "d2h_time_ms=" << inference_result.timing.d2h_time_ms << '\n';
    std::cout << "gpu_backend_total_time_ms="
              << inference_result.timing.gpu_backend_total_time_ms << '\n';
    if (!inference_result.success) {
        std::cerr << "inference_error=" << inference_result.error_message << '\n';
        return 3;
    }

    const auto& image = inference_result.output.generated_face_bgr;
    if (inference_result.info.output_shape != std::vector<std::int64_t>{1, 3, 96, 96} ||
        !inference_result.info.output_was_cuda || image.rows != 96 || image.cols != 96 ||
        image.type() != CV_8UC3 || inference_result.timing.h2d_time_ms <= 0.0 ||
        inference_result.timing.cuda_forward_time_ms <= 0.0 ||
        inference_result.timing.d2h_time_ms <= 0.0 ||
        inference_result.timing.gpu_backend_total_time_ms <= 0.0) {
        std::cerr << "output contract validation failed\n";
        return 4;
    }

    std::cout << "processed_output=96x96 CV_8UC3 BGR\n";
    std::cout << "status=success\n";
    return 0;
}
