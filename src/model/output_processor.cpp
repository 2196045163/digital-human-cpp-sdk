#include "model/output_processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "detail/wav2lip_model_spec.h"

namespace digital_human {
namespace model {

namespace {

// 转换耗时只覆盖本模块，不把上游模型推理时间混进来。
double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

OutputProcessResult MakeErrorResult(const InferenceOutput& input,
                                    OutputProcessStatus status,
                                    std::string error_message,
                                    const std::chrono::steady_clock::time_point& start) {
    // 失败时不交付半成品图像，但时间戳和模型代数仍可用于定位问题。
    OutputProcessResult result;
    result.status = status;
    result.error_message = std::move(error_message);
    result.value.metadata = input.metadata;
    result.value.model_generation = input.model_generation;
    result.value.conversion_info.conversion_ms = ElapsedMs(start);
    return result;
}

std::string FormatPredictionShape(const ncnn::Mat& prediction) {
    // 形状信息放进错误消息，调用方不必额外打印底层 ncnn::Mat。
    std::ostringstream stream;
    stream << "dims=" << prediction.dims
           << ", w=" << prediction.w
           << ", h=" << prediction.h
           << ", c=" << prediction.c
           << ", elempack=" << prediction.elempack
           << ", elemsize=" << prediction.elemsize;
    return stream.str();
}

bool HasExpectedPredictionShape(const ncnn::Mat& prediction) {
    // shape 表示“有多大、几个通道”，不等同于内存中的元素类型和打包方式。
    return prediction.dims == detail::Wav2LipModelSpec::kTensorDims &&
        prediction.w == detail::Wav2LipModelSpec::kPredWidth &&
        prediction.h == detail::Wav2LipModelSpec::kPredHeight &&
        prediction.c == detail::Wav2LipModelSpec::kPredChannels;
}

bool HasSupportedPredictionLayout(const ncnn::Mat& prediction) {
    // 当前转换器只读取每元素一个 float 的未打包 FP32 平面张量。
    return prediction.elempack == detail::Wav2LipModelSpec::kUnpackedElementPack &&
        prediction.elemsize == detail::Wav2LipModelSpec::kUnpackedFp32ElementSize;
}

double CalculateLaplacianVariance(const cv::Mat& generated_face_bgr) {
    // 清晰度诊断只看亮度边缘：先转灰度，避免把同一条边缘在三个颜色平面重复计分。
    cv::Mat grayscale;
    cv::Mat laplacian;
    cv::cvtColor(generated_face_bgr, grayscale, cv::COLOR_BGR2GRAY);
    cv::Laplacian(grayscale, laplacian, CV_64F);

    cv::Scalar mean;
    cv::Scalar standard_deviation;
    cv::meanStdDev(laplacian, mean, standard_deviation);
    return standard_deviation[0] * standard_deviation[0];
}

} // namespace

OutputProcessResult OutputProcessor::Convert(
    const InferenceOutput& input,
    const OutputProcessOptions& options) const {
    const auto start = std::chrono::steady_clock::now();

    // 先阻止无意义的阈值进入主流程；阈值只影响诊断结论，不影响转换成功与否。
    if (options.minimum_sharpness.has_value() &&
        (!std::isfinite(*options.minimum_sharpness) ||
         *options.minimum_sharpness < 0.0)) {
        return MakeErrorResult(
            input,
            OutputProcessStatus::kInvalidSharpnessThreshold,
            "minimum_sharpness must be finite and non-negative",
            start);
    }

    if (input.pred.empty()) {
        return MakeErrorResult(
            input,
            OutputProcessStatus::kEmptyPrediction,
            "prediction tensor is empty",
            start);
    }

    // 即使 ModelInference 已做过守门，这个模块仍在自己的公开边界复核契约。
    if (!HasExpectedPredictionShape(input.pred)) {
        return MakeErrorResult(
            input,
            OutputProcessStatus::kInvalidPredictionShape,
            "prediction shape does not match Wav2Lip contract: " +
                FormatPredictionShape(input.pred),
            start);
    }

    if (!HasSupportedPredictionLayout(input.pred)) {
        return MakeErrorResult(
            input,
            OutputProcessStatus::kUnsupportedPredictionLayout,
            "prediction layout is not unpacked FP32: " +
                FormatPredictionShape(input.pred),
            start);
    }

    try {
        // ncnn 的三通道数据按三个平面存放。A/B 实验证实当前权重的
        // plane 0/1/2 分别是 B/G/R；这里仅把平面改成交织 OpenCV 像素。
        const ncnn::Mat blue_plane = input.pred.channel(
            detail::Wav2LipModelSpec::kBluePlaneIndex);
        const ncnn::Mat green_plane = input.pred.channel(
            detail::Wav2LipModelSpec::kGreenPlaneIndex);
        const ncnn::Mat red_plane = input.pred.channel(
            detail::Wav2LipModelSpec::kRedPlaneIndex);
        // 先写局部图，全部像素成功后才交给下游，避免返回半张有效图。
        cv::Mat generated_face_bgr(
            detail::Wav2LipModelSpec::kPredHeight,
            detail::Wav2LipModelSpec::kPredWidth,
            CV_8UC3);
        std::size_t corrected_value_count = 0;

        for (int row_index = 0;
             row_index < detail::Wav2LipModelSpec::kPredHeight;
             ++row_index) {
            const float* blue_row = blue_plane.row(row_index);
            const float* green_row = green_plane.row(row_index);
            const float* red_row = red_plane.row(row_index);
            cv::Vec3b* output_row = generated_face_bgr.ptr<cv::Vec3b>(row_index);

            for (int column_index = 0;
                 column_index < detail::Wav2LipModelSpec::kPredWidth;
                 ++column_index) {
                const float source_values[] = {
                    blue_row[column_index],
                    green_row[column_index],
                    red_row[column_index]
                };
                // OpenCV 的 CV_8UC3 像素顺序是 B、G、R，与 source_values 保持一致。
                unsigned char output_values[3] = {};

                for (int channel_index = 0; channel_index < 3; ++channel_index) {
                    const float value = source_values[channel_index];
                    if (!std::isfinite(value)) {
                        return MakeErrorResult(
                            input,
                            OutputProcessStatus::kNonFinitePrediction,
                            "prediction contains NaN or Inf",
                            start);
                    }

                    if (value < detail::Wav2LipModelSpec::kPredValueMin -
                            detail::Wav2LipModelSpec::kPredRangeTolerance ||
                        value > detail::Wav2LipModelSpec::kPredValueMax +
                            detail::Wav2LipModelSpec::kPredRangeTolerance) {
                        return MakeErrorResult(
                            input, OutputProcessStatus::kRangeViolation,
                            "prediction value exceeds the permitted range",
                            start);
                    }

                    // 极小的浮点误差允许夹回 [0,1]；明显越界已在上面作为失败处理。
                    const float bounded_value = std::clamp(
                        value,
                        detail::Wav2LipModelSpec::kPredValueMin,
                        detail::Wav2LipModelSpec::kPredValueMax);
                    if (bounded_value != value) {
                        ++corrected_value_count;
                    }
                    output_values[channel_index] = static_cast<unsigned char>(
                        std::lround(bounded_value * detail::Wav2LipModelSpec::kImageQuantizationScale));
                }

                output_row[column_index] = cv::Vec3b(output_values[0], output_values[1], output_values[2]);
            }
        }

        OutputProcessResult result;
        result.success = true;
        result.status = OutputProcessStatus::kOk;
        result.value.generated_face_bgr = generated_face_bgr;
        result.value.metadata = input.metadata;
        result.value.model_generation = input.model_generation;
        result.value.conversion_info.corrected_value_count = corrected_value_count;
        result.value.conversion_info.sharpness_score =
            CalculateLaplacianVariance(generated_face_bgr);
        // 低清晰度代表模型产物可能偏模糊，不代表 BGR 转换失败。
        result.value.conversion_info.sharpness_threshold_applied =
            options.minimum_sharpness.has_value();
        result.value.conversion_info.sharpness_passed =
            !options.minimum_sharpness.has_value() ||
            result.value.conversion_info.sharpness_score >= *options.minimum_sharpness;
        result.value.conversion_info.conversion_ms = ElapsedMs(start);
        return result;
    } catch (const std::bad_alloc&) {
        // 分配失败也用 Result 报告，调用方不需要猜测空 Mat 的原因。
        return MakeErrorResult(
            input,
            OutputProcessStatus::kAllocationFailed,
            "failed to allocate output image",
            start);
    } catch (const cv::Exception& exception) {
        return MakeErrorResult(
            input,
            OutputProcessStatus::kOpenCvError,
            exception.what(),
            start);
    } catch (const std::exception& exception) {
        return MakeErrorResult(
            input,
            OutputProcessStatus::kUnknownError,
            exception.what(),
            start);
    }
}

std::string OutputProcessor::StatusToString(OutputProcessStatus status) {
    // 给 UI、测试或上层日志使用的稳定中文摘要；底层细节保留在 error_message。
    switch (status) {
    case OutputProcessStatus::kOk:                          return "输出转换成功";
    case OutputProcessStatus::kEmptyPrediction:             return "预测张量为空";
    case OutputProcessStatus::kInvalidPredictionShape:      return "预测张量 shape 不符合契约";
    case OutputProcessStatus::kUnsupportedPredictionLayout: return "预测张量不是未打包 FP32";
    case OutputProcessStatus::kNonFinitePrediction:         return "预测张量包含 NaN 或 Inf";
    case OutputProcessStatus::kRangeViolation:              return "预测张量值域超出允许范围";
    case OutputProcessStatus::kInvalidSharpnessThreshold:   return "清晰度阈值必须是非负有限值";
    case OutputProcessStatus::kAllocationFailed:            return "输出图像分配失败";
    case OutputProcessStatus::kOpenCvError:                 return "OpenCV 输出处理失败";
    case OutputProcessStatus::kUnknownError:                return "未知输出处理错误";
    }

    return "未知输出处理错误";
}

} // namespace model
} // namespace digital_human
