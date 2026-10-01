#include "model/libtorch_cuda_wav2lip_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <c10/cuda/CUDACachingAllocator.h>
#include <nvml.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include "detail/wav2lip_model_spec.h"

namespace digital_human {
namespace model {

namespace {

using Spec = detail::Wav2LipModelSpec;

bool AllFinite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

double CalculateLaplacianVariance(const cv::Mat& generated_face_bgr) {
    cv::Mat grayscale;
    cv::Mat laplacian;
    cv::cvtColor(generated_face_bgr, grayscale, cv::COLOR_BGR2GRAY);
    cv::Laplacian(grayscale, laplacian, CV_64F);

    cv::Scalar mean;
    cv::Scalar standard_deviation;
    cv::meanStdDev(laplacian, mean, standard_deviation);
    return standard_deviation[0] * standard_deviation[0];
}

double ReadNvmlUsedMemoryMb(nvmlDevice_t device) {
    nvmlMemory_t memory{};
    const nvmlReturn_t status = nvmlDeviceGetMemoryInfo(device, &memory);
    if (status != NVML_SUCCESS) {
        throw std::runtime_error(
            std::string("nvmlDeviceGetMemoryInfo failed: ") + nvmlErrorString(status));
    }
    return static_cast<double>(memory.used) / (1024.0 * 1024.0);
}

LibTorchCudaInferenceResult MakeInferenceError(
    LibTorchCudaStatus status,
    std::string error_message,
    const Wav2LipInputData& input) {
    LibTorchCudaInferenceResult result;
    result.status = status;
    result.error_message = std::move(error_message);
    result.output.metadata = input.metadata;
    return result;
}

} // namespace

struct LibTorchCudaWav2LipRuntime::Impl {
    torch::Device device{torch::kCUDA, 0};
    std::unique_ptr<torch::jit::script::Module> module;
    std::uint64_t generation = 0;
    nvmlDevice_t nvml_device = nullptr;
    bool nvml_initialized = false;

    ~Impl() {
        if (nvml_initialized) {
            nvmlShutdown();
        }
    }
};

LibTorchCudaWav2LipRuntime::LibTorchCudaWav2LipRuntime()
    : pImpl_(std::make_unique<Impl>()) {}

LibTorchCudaWav2LipRuntime::~LibTorchCudaWav2LipRuntime() = default;

LibTorchCudaWav2LipRuntime::LibTorchCudaWav2LipRuntime(
    LibTorchCudaWav2LipRuntime&&) noexcept = default;

LibTorchCudaWav2LipRuntime& LibTorchCudaWav2LipRuntime::operator=(
    LibTorchCudaWav2LipRuntime&&) noexcept = default;

bool LibTorchCudaWav2LipRuntime::IsCudaAvailable() {
    return torch::cuda::is_available();
}

LibTorchCudaLoadResult LibTorchCudaWav2LipRuntime::Load(
    const std::filesystem::path& model_path) {
    LibTorchCudaLoadResult result;

    if (!IsCudaAvailable()) {
        result.status = LibTorchCudaStatus::kCudaUnavailable;
        result.error_message = "LibTorch reports that CUDA is unavailable";
        return result;
    }
    if (model_path.empty() || !std::filesystem::is_regular_file(model_path)) {
        result.status = LibTorchCudaStatus::kModelFileNotFound;
        result.error_message = "TorchScript model file does not exist: " + model_path.string();
        return result;
    }

    if (!pImpl_->nvml_initialized) {
        const nvmlReturn_t init_status = nvmlInit_v2();
        if (init_status != NVML_SUCCESS) {
            result.status = LibTorchCudaStatus::kModelLoadFailed;
            result.error_message =
                std::string("nvmlInit_v2 failed: ") + nvmlErrorString(init_status);
            return result;
        }
        pImpl_->nvml_initialized = true;

        const nvmlReturn_t device_status = nvmlDeviceGetHandleByIndex_v2(
            static_cast<unsigned int>(pImpl_->device.index()),
            &pImpl_->nvml_device);
        if (device_status != NVML_SUCCESS) {
            result.status = LibTorchCudaStatus::kModelLoadFailed;
            result.error_message = std::string("nvmlDeviceGetHandleByIndex_v2 failed: ")
                + nvmlErrorString(device_status);
            nvmlShutdown();
            pImpl_->nvml_initialized = false;
            pImpl_->nvml_device = nullptr;
            return result;
        }
    }

    try {
        auto candidate = torch::jit::load(model_path.string(), torch::Device(torch::kCPU));
        candidate.eval();
        candidate.to(pImpl_->device);
        pImpl_->module = std::make_unique<torch::jit::script::Module>(
            std::move(candidate));
        ++pImpl_->generation;

        result.success = true;
        result.status = LibTorchCudaStatus::kOk;
        return result;
    } catch (const std::exception& exception) {
        result.status = LibTorchCudaStatus::kModelLoadFailed;
        result.error_message = exception.what();
        return result;
    }
}

bool LibTorchCudaWav2LipRuntime::IsReady() const {
    return pImpl_ != nullptr && pImpl_->module != nullptr;
}

LibTorchCudaInferenceResult LibTorchCudaWav2LipRuntime::Infer(
    const Wav2LipInputData& input) const {
    const auto backend_start = std::chrono::steady_clock::now();

    if (!IsReady()) {
        return MakeInferenceError(
            LibTorchCudaStatus::kModelNotReady,
            "TorchScript model is not loaded",
            input);
    }

    constexpr std::size_t kExpectedFaceElements =
        static_cast<std::size_t>(Spec::kFaceChannels) *
        Spec::kFaceHeight * Spec::kFaceWidth;
    constexpr std::size_t kExpectedMelElements =
        static_cast<std::size_t>(Spec::kMelBins) * Spec::kMelFrames;

    if (input.face_chw.size() != kExpectedFaceElements ||
        input.mel_freq_time.size() != kExpectedMelElements ||
        !AllFinite(input.face_chw) || !AllFinite(input.mel_freq_time)) {
        return MakeInferenceError(
            LibTorchCudaStatus::kInvalidInput,
            "Wav2LipInputData shape or finite-value contract is invalid",
            input);
    }

    try {
        torch::NoGradGuard no_grad;
        const c10::DeviceIndex device_index = pImpl_->device.index();
        c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);
        double nvml_used_memory_peak_mb = 0.0;
        double nvml_query_time_ms = 0.0;
        const auto sample_nvml_used_memory = [&]() {
            const auto query_start = std::chrono::steady_clock::now();
            const double used_mb = ReadNvmlUsedMemoryMb(pImpl_->nvml_device);
            nvml_used_memory_peak_mb = std::max(
                nvml_used_memory_peak_mb, used_mb);
            nvml_query_time_ms += ElapsedMs(query_start);
            return used_mb;
        };
        const double baseline_gpu_memory_used_mb = sample_nvml_used_memory();

        const auto float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU);

        // from_blob 不拥有外部 vector；立即 clone 后再传入 CUDA，runtime 不保存外部引用。
        torch::Tensor mel_cpu = torch::from_blob(
            const_cast<float*>(input.mel_freq_time.data()),
            {1, 1, Spec::kMelBins, Spec::kMelFrames},
            float_options).clone();
        torch::Tensor face_cpu = torch::from_blob(
            const_cast<float*>(input.face_chw.data()),
            {1, Spec::kFaceChannels, Spec::kFaceHeight, Spec::kFaceWidth},
            float_options).clone();

        // CUDA 操作异步提交；分段计时前后同步，确保记录的是实际完成耗时。
        torch::cuda::synchronize();
        const auto h2d_start = std::chrono::steady_clock::now();
        torch::Tensor mel_cuda = mel_cpu.to(pImpl_->device, torch::kFloat32);
        torch::Tensor face_cuda = face_cpu.to(pImpl_->device, torch::kFloat32);
        torch::cuda::synchronize();
        const double h2d_time_ms = ElapsedMs(h2d_start);
        sample_nvml_used_memory();

        const auto forward_start = std::chrono::steady_clock::now();
        torch::IValue forward_value = pImpl_->module->forward({mel_cuda, face_cuda});
        torch::cuda::synchronize();
        const double cuda_forward_time_ms = ElapsedMs(forward_start);
        sample_nvml_used_memory();
        if (!forward_value.isTensor()) {
            return MakeInferenceError(
                LibTorchCudaStatus::kInvalidOutput,
                "TorchScript forward did not return a tensor",
                input);
        }

        torch::Tensor prediction = forward_value.toTensor();
        LibTorchCudaInferenceResult result;
        result.info.output_shape = prediction.sizes().vec();
        result.info.output_was_cuda = prediction.is_cuda();
        result.timing.h2d_time_ms = h2d_time_ms;
        result.timing.cuda_forward_time_ms = cuda_forward_time_ms;
        result.output.metadata = input.metadata;
        result.output.model_generation = pImpl_->generation;

        const bool valid_shape = prediction.dim() == 4 &&
            prediction.size(0) == 1 &&
            prediction.size(1) == Spec::kPredChannels &&
            prediction.size(2) == Spec::kPredHeight &&
            prediction.size(3) == Spec::kPredWidth;
        if (!prediction.is_cuda() || prediction.scalar_type() != torch::kFloat32 ||
            !valid_shape) {
            result.status = LibTorchCudaStatus::kInvalidOutput;
            result.error_message =
                "TorchScript output must be CUDA float32 [1,3,96,96]";
            return result;
        }
        if (!torch::isfinite(prediction).all().item<bool>()) {
            result.status = LibTorchCudaStatus::kInvalidOutput;
            result.error_message = "TorchScript output contains NaN or Inf";
            return result;
        }

        const float min_value = prediction.min().item<float>();
        const float max_value = prediction.max().item<float>();
        if (min_value < Spec::kPredValueMin - Spec::kPredRangeTolerance ||
            max_value > Spec::kPredValueMax + Spec::kPredRangeTolerance) {
            result.status = LibTorchCudaStatus::kInvalidOutput;
            result.error_message = "TorchScript output exceeds the permitted [0,1] range";
            return result;
        }
        sample_nvml_used_memory();

        torch::cuda::synchronize();
        const auto d2h_start = std::chrono::steady_clock::now();
        torch::Tensor prediction_cpu = prediction.detach()
            .to(torch::Device(torch::kCPU), torch::kFloat32)
            .contiguous();
        torch::cuda::synchronize();
        result.timing.d2h_time_ms = ElapsedMs(d2h_start);
        result.timing.gpu_backend_total_time_ms = std::max(
            0.0, ElapsedMs(backend_start) - nvml_query_time_ms);
        sample_nvml_used_memory();

        const auto conversion_start = std::chrono::steady_clock::now();
        const float* prediction_data = prediction_cpu.data_ptr<float>();

        cv::Mat generated_face_bgr(Spec::kPredHeight, Spec::kPredWidth, CV_8UC3);
        std::size_t corrected_value_count = 0;
        constexpr std::size_t kPlaneElements =
            static_cast<std::size_t>(Spec::kPredHeight) * Spec::kPredWidth;

        for (int row = 0; row < Spec::kPredHeight; ++row) {
            cv::Vec3b* output_row = generated_face_bgr.ptr<cv::Vec3b>(row);
            for (int column = 0; column < Spec::kPredWidth; ++column) {
                const std::size_t pixel_index =
                    static_cast<std::size_t>(row) * Spec::kPredWidth + column;
                cv::Vec3b pixel;
                for (int channel = 0; channel < Spec::kPredChannels; ++channel) {
                    const float value = prediction_data[
                        static_cast<std::size_t>(channel) * kPlaneElements + pixel_index];
                    const float bounded_value = std::clamp(
                        value, Spec::kPredValueMin, Spec::kPredValueMax);
                    if (bounded_value != value) {
                        ++corrected_value_count;
                    }
                    pixel[channel] = static_cast<unsigned char>(
                        std::lround(bounded_value * Spec::kImageQuantizationScale));
                }
                output_row[column] = pixel;
            }
        }

        result.output.generated_face_bgr = std::move(generated_face_bgr);
        result.output.conversion_info.corrected_value_count = corrected_value_count;
        result.output.conversion_info.sharpness_score =
            CalculateLaplacianVariance(result.output.generated_face_bgr);
        result.output.conversion_info.sharpness_threshold_applied = false;
        result.output.conversion_info.sharpness_passed = true;
        result.output.conversion_info.conversion_ms = ElapsedMs(conversion_start);

        constexpr std::size_t kAggregateStat = static_cast<std::size_t>(
            c10::CachingAllocator::StatType::AGGREGATE);
        const auto cuda_memory_stats =
            c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
        const auto peak_allocated_bytes =
            cuda_memory_stats.allocated_bytes[kAggregateStat].peak;
        result.info.gpu_peak_memory_mb =
            static_cast<double>(peak_allocated_bytes) / (1024.0 * 1024.0);
        result.info.gpu_inference_memory_delta_mb = std::max(
            0.0, nvml_used_memory_peak_mb - baseline_gpu_memory_used_mb);

        result.success = true;
        result.status = LibTorchCudaStatus::kOk;
        return result;
    } catch (const cv::Exception& exception) {
        return MakeInferenceError(
            LibTorchCudaStatus::kOutputConversionFailed,
            exception.what(),
            input);
    } catch (const std::exception& exception) {
        return MakeInferenceError(
            LibTorchCudaStatus::kForwardFailed,
            exception.what(),
            input);
    }
}

} // namespace model
} // namespace digital_human
