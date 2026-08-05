
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "model/model_inference.h"
#include "detail/inference_retry_runner.h"
#include "detail/wav2lip_model_spec.h"

namespace digital_human {
namespace model {

namespace {

struct FloatRange {
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
};

/// @brief 计算从指定起点到当前时刻的单调时钟耗时（毫秒）
double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

/// @brief 检查三维、未打包 FP32 Mat （4 字节float）的所有逻辑元素是否为有限值
/// @note  调用方应先完成 tensor shape 校验；逐 channel/row 遍历会跳过 cstep padding。
bool AllFiniteUnpackedFp32(const ncnn::Mat& mat) {
    if (mat.empty() ||
        mat.dims != detail::Wav2LipModelSpec::kTensorDims ||
        mat.elempack != detail::Wav2LipModelSpec::kUnpackedElementPack ||
        mat.elemsize != detail::Wav2LipModelSpec::kUnpackedFp32ElementSize) {
        return false;
    }

    for (int channel_index = 0; channel_index < mat.c; ++channel_index) {
        const ncnn::Mat channel = mat.channel(channel_index);
        for (int row_index = 0; row_index < mat.h; ++row_index) {
            const float* row = channel.row(row_index);
            for (int column_index = 0; column_index < mat.w; ++column_index) {
                if (!std::isfinite(row[column_index])) {
                    return false;
                }
            }
        }
    }

    return true;
}

/// @brief 统计三维、未打包 FP32 Mat 所有逻辑元素的最小值和最大值
/// @note  仅在 shape 与 finite 守门通过后调用；逐 channel/row 遍历会跳过 cstep padding
FloatRange MeasureUnpackedFp32Range(const ncnn::Mat& mat) {
    FloatRange range;

    for (int channel_index = 0; channel_index < mat.c; ++channel_index) {
        const ncnn::Mat channel = mat.channel(channel_index);
        for (int row_index = 0; row_index < mat.h; ++row_index) {
            const float* row = channel.row(row_index);
            for (int column_index = 0; column_index < mat.w; ++column_index) {
                const float value = row[column_index];
                if (value < range.min_value) {
                    range.min_value = value;
                }
                if (value > range.max_value) {
                    range.max_value = value;
                }
            }
        }
    }

    return range;
}

} // namespace

    /// @brief 同步单样本推理
    /// @note  每次调用创建局部 Extractor，不跨请求共享
    SingleInferenceResult ModelInference::Infer(const ModelRuntimeSnapshot& snapshot,
                                const NcnnWav2LipInput& input,
                                const InferenceOptions& options) const {

        SingleInferenceResult result;
        const auto total_start = std::chrono::steady_clock::now();

        // 无论成功失败，都保留推理请求上下文。
        result.value.metadata = input.metadata;
        result.value.model_generation = snapshot.generation;

        // 1.Validate snapshot/input 验证模型快照和推理输入
        // 1.1 snapshot
        if (!snapshot.IsValid()) {
            result.success = false;
            result.status = InferenceStatus::kModelUnavailable;
            result.error_message = StatusToString(InferenceStatus::kModelUnavailable);
            // snapshot 失败不属于 input shape/finite 校验。
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }

        const auto input_validation_start = std::chrono::steady_clock::now();
        // 1.2 mel input empty shape nan/inf 校验
        if (input.mel.empty()) {
            result.success = false;
            result.status = InferenceStatus::kEmptyMelInput;
            result.error_message = StatusToString(InferenceStatus::kEmptyMelInput);
            result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        // dims == 3：必须是显式三维 tensor。
        // w == 16：16 个时间帧。
        // h == 80：80 个 Mel 频率 bin。
        // c == 1：单通道。
        // elempack == 1：没有打包多个标量。
        // elemsize == sizeof(float)：每个元素是一个 FP32。
        // 不检查 cstep == w*h，因为 channel 间允许有合法 padding。
        const bool mel_shape_valid =
            input.mel.dims == detail::Wav2LipModelSpec::kTensorDims &&
            input.mel.w == detail::Wav2LipModelSpec::kMelFrames &&
            input.mel.h == detail::Wav2LipModelSpec::kMelBins &&
            input.mel.c == 1 &&
            input.mel.elempack == detail::Wav2LipModelSpec::kUnpackedElementPack &&
            input.mel.elemsize == detail::Wav2LipModelSpec::kUnpackedFp32ElementSize;
        if (!mel_shape_valid) {
            result.success = false;
            result.status = InferenceStatus::kInvalidMelShape;
            result.error_message = StatusToString(InferenceStatus::kInvalidMelShape);
            result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        if (!AllFiniteUnpackedFp32(input.mel)) {
            result.success = false;
            result.status = InferenceStatus::kNonFiniteMelInput;
            result.error_message = StatusToString(InferenceStatus::kNonFiniteMelInput);
            result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }

        if (input.face.empty()) {
            result.success = false;
            result.status = InferenceStatus::kEmptyFaceInput;
            result.error_message = StatusToString(InferenceStatus::kEmptyFaceInput);
            result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        const bool face_shape_valid =
            input.face.dims == detail::Wav2LipModelSpec::kTensorDims &&
            input.face.w == detail::Wav2LipModelSpec::kFaceWidth &&
            input.face.h == detail::Wav2LipModelSpec::kFaceHeight &&
            input.face.c == detail::Wav2LipModelSpec::kFaceChannels &&
            input.face.elempack == detail::Wav2LipModelSpec::kUnpackedElementPack &&
            input.face.elemsize == detail::Wav2LipModelSpec::kUnpackedFp32ElementSize;
        if (!face_shape_valid) {
            result.success = false;
            result.status = InferenceStatus::kInvalidFaceShape;
            result.error_message = StatusToString(InferenceStatus::kInvalidFaceShape);
            result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        if (!AllFiniteUnpackedFp32(input.face)) {
            result.success = false;
            result.status = InferenceStatus::kNonFiniteFaceInput;
            result.error_message = StatusToString(InferenceStatus::kNonFiniteFaceInput);
            result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        // 1.3 输入全部通过后，结束并记录 input_validation_ms
        result.value.timing.input_validation_ms = ElapsedMs(input_validation_start);

        // 2.定义“一次独立推理尝试”的 callable。
        // runner 每调用一次 inference_attempt，就代表一次完整 attempt；lambda 按引用捕获
        // 已校验的 snapshot/input/options，但不会修改它们，也不会在两次调用之间保存状态。
        // 每次调用都重新创建局部 Extractor，确保失败 attempt 的执行上下文和中间 blob
        // 不会泄漏到下一次尝试。
        auto inference_attempt = [&]() {
            // outcome 只描述本次 attempt：语义状态、原始 ncnn code，以及成功时的
            // 候选 pred。公开 Result 的 metadata、generation 和总计时不在这里组装。
            detail::InferenceAttemptOutcome outcome;

            // 2.1 每次 attempt 都从只读 Net 创建全新的 Extractor。
            ncnn::Extractor extractor = snapshot.model->create_extractor();
            // light_mode 允许 ncnn 在中间 blob 用完后尽早释放或复用内存。
            extractor.set_light_mode(options.light_mode);

            // 2.2 按模型 blob 名绑定 Mel 输入。
            // input() 只操作本次局部 Extractor；非零返回码表示绑定失败。
            // kInputMelFailed 属于确定性错误，runner 收到后会立即停止，不会重试。
            const int mel_input_code = extractor.input(
                detail::Wav2LipModelSpec::kInputMel, input.mel);
            if (mel_input_code != 0) {
                outcome.status = InferenceStatus::kInputMelFailed;
                outcome.ncnn_error_code = mel_input_code;
                return outcome;
            }

            // 2.3 Mel 绑定成功后再绑定 Face；失败时保留本次真实 ncnn 返回码。
            // kInputFaceFailed 同样不在可重试白名单内。
            const int face_input_code = extractor.input(
                detail::Wav2LipModelSpec::kInputFace, input.face);
            if (face_input_code != 0) {
                outcome.status = InferenceStatus::kInputFaceFailed;
                outcome.ncnn_error_code = face_input_code;
                return outcome;
            }

            // 2.4 extract("pred") 真正触发所需的前向计算并提取输出。
            // 非零表示 ncnn 执行失败；kExtractFailed 属于可能恢复的执行错误，
            // runner 可在预算允许时新建另一个 Extractor 再尝试。
            ncnn::Mat candidate_pred;
            const int extract_code = extractor.extract(
                detail::Wav2LipModelSpec::kOutputPred, candidate_pred);
            if (extract_code != 0) {
                outcome.status = InferenceStatus::kExtractFailed;
                outcome.ncnn_error_code = extract_code;
                return outcome;
            }

            // 2.5 extract 返回 0 只说明调用成功，不保证实际生成了输出。
            // 空 pred 使用 kOutputEmpty，并保留 ncnn code=0；它同样允许有限重试。
            if (candidate_pred.empty()) {
                outcome.status = InferenceStatus::kOutputEmpty;
                outcome.ncnn_error_code = 0;
                return outcome;
            }

            // 2.6 本次执行阶段成功：只把非空候选 pred 交给 runner。
            // shape、finite 和 range 尚未校验，因此这里仍不能写入公开 Result::value.pred。
            outcome.status = InferenceStatus::kOk;
            outcome.ncnn_error_code = 0;
            outcome.candidate_pred = candidate_pred;
            return outcome;
        };

        // 3.运行分类重试控制器。
        // max_retries 是首次 attempt 之外允许的额外次数；runner 始终至少调用一次
        // inference_attempt，并且只会为 kExtractFailed/kOutputEmpty 消耗额外预算。
        const detail::InferenceRetryRunResult retry_result =
              detail::RunWithRetry(options.retry.max_retries, inference_attempt);

        // 3.1 runner 统一汇总首末 attempt 诊断：调用次数、首次失败、最终状态、
        // 首末 ncnn code 和是否经重试恢复。Infer 将摘要原样装入公开 Result。
        result.value.attempts = retry_result.attempts;

        // 3.2 两个计时字段采用相同边界：create/config/input/extract/empty-check。
        // first_attempt_ms 只记录首次；retry_attempts_ms 是所有额外 attempt 的总和。
        result.value.timing.first_attempt_ms = retry_result.first_attempt_ms;
        result.value.timing.retry_attempts_ms = retry_result.retry_attempts_ms;

        // 3.3 runner 失败时直接返回：status 可能是不可重试的具体错误、
        // max_retries=0 时的首次原始错误，或实际重试后仍失败的 kRetryExhausted。
        // retry_result 在失败时不携带可交付 candidate_pred，因此公开 pred 保持为空。
        if (!retry_result.success) {
            result.success = false;
            result.status = retry_result.status;
            result.error_message = StatusToString(retry_result.status);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);
            return result;
        }

        // 4.只有最终成功 attempt 的非空候选输出能越过 runner，继续接受 shape、
        // finite 和 range 三道守门。ncnn::Mat 赋值共享其引用计数数据，不复制整张 tensor；
        // 此时 candidate_pred 仍是内部候选，尚未写入公开 pred。
        const ncnn::Mat candidate_pred = retry_result.candidate_pred;

        result.value.output_info.width = candidate_pred.w; // width
        result.value.output_info.height = candidate_pred.h; // height
        result.value.output_info.channels = candidate_pred.c; // channels

        // shape 校验
        const auto output_validation_start = std::chrono::steady_clock::now();
        const bool output_shape_valid =
            candidate_pred.dims == detail::Wav2LipModelSpec::kTensorDims &&
            candidate_pred.w == detail::Wav2LipModelSpec::kPredWidth &&
            candidate_pred.h == detail::Wav2LipModelSpec::kPredHeight &&
            candidate_pred.c == detail::Wav2LipModelSpec::kPredChannels &&
            candidate_pred.elempack == detail::Wav2LipModelSpec::kUnpackedElementPack &&
            candidate_pred.elemsize == detail::Wav2LipModelSpec::kUnpackedFp32ElementSize;
        if (!output_shape_valid) {
            result.success = false;
            result.status = InferenceStatus::kInvalidOutputShape;
            result.error_message = StatusToString(InferenceStatus::kInvalidOutputShape);

            // 若执行阶段此前已失败并靠重试取得候选输出，保留最初失败原因；
            // 否则当前守门错误就是本请求的首次失败。
            if (!result.value.attempts.first_failure_status.has_value()) {
                result.value.attempts.first_failure_status =
                    InferenceStatus::kInvalidOutputShape;
            }
            result.value.attempts.final_attempt_status = InferenceStatus::kInvalidOutputShape;
            result.value.attempts.recovered_by_retry = false;

            result.value.timing.output_validation_ms = ElapsedMs(output_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        // Nan/Inf校验
        if (!AllFiniteUnpackedFp32(candidate_pred)) {
            result.success = false;
            result.status = InferenceStatus::kNonFiniteOutput;
            result.error_message = StatusToString(InferenceStatus::kNonFiniteOutput);

            if (!result.value.attempts.first_failure_status.has_value()) {
                result.value.attempts.first_failure_status =
                    InferenceStatus::kNonFiniteOutput;
            }
            result.value.attempts.final_attempt_status = InferenceStatus::kNonFiniteOutput;
            result.value.attempts.recovered_by_retry = false;

            result.value.timing.output_validation_ms = ElapsedMs(output_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        result.value.output_info.all_finite = true;

        // 5.验证输出的值域范围
        const FloatRange output_range = MeasureUnpackedFp32Range(candidate_pred);

        result.value.output_info.min_value = output_range.min_value;
        result.value.output_info.max_value = output_range.max_value;

        const bool within_expected_range =
            output_range.min_value >= detail::Wav2LipModelSpec::kPredValueMin -
                detail::Wav2LipModelSpec::kPredRangeTolerance &&
            output_range.max_value <= detail::Wav2LipModelSpec::kPredValueMax +
                detail::Wav2LipModelSpec::kPredRangeTolerance;
        if (!within_expected_range) {
            result.success = false;
            result.status = InferenceStatus::kOutputRangeViolation;
            result.error_message = StatusToString(InferenceStatus::kOutputRangeViolation);

            if (!result.value.attempts.first_failure_status.has_value()) {
                result.value.attempts.first_failure_status =
                    InferenceStatus::kOutputRangeViolation;
            }
            result.value.attempts.final_attempt_status = InferenceStatus::kOutputRangeViolation;
            result.value.attempts.recovered_by_retry = false;

            result.value.timing.output_validation_ms = ElapsedMs(output_validation_start);
            result.value.timing.total_latency_ms = ElapsedMs(total_start);

            return result;
        }
        result.value.output_info.within_expected_range = within_expected_range;

        // 6.全部通过后组装成功结果
        result.value.timing.output_validation_ms = ElapsedMs(output_validation_start);

        // 交付 合格输出 candidate_pred
        result.value.pred = candidate_pred;
        // runner 已记录最终成功 attempt；若发生过重试，首次失败留痕也会保留。

        result.success = true;
        result.status = InferenceStatus::kOk;
        result.error_message.clear();
        result.value.timing.total_latency_ms = ElapsedMs(total_start);

        return result;
    }

    /// @brief 状态码 → 人类可读字符串
    std::string ModelInference::StatusToString(InferenceStatus status) {
        switch (status) {
        case InferenceStatus::kOk:                       return "推理成功";
        case InferenceStatus::kModelUnavailable:         return "模型快照不可用";
        case InferenceStatus::kEmptyMelInput:            return "Mel 输入为空";
        case InferenceStatus::kInvalidMelShape:          return "Mel tensor 规格错误";
        case InferenceStatus::kNonFiniteMelInput:        return "Mel 输入包含 NaN 或 Inf";
        case InferenceStatus::kEmptyFaceInput:           return "Face 输入为空";
        case InferenceStatus::kInvalidFaceShape:         return "Face tensor 规格错误";
        case InferenceStatus::kNonFiniteFaceInput:       return "Face 输入包含 NaN 或 Inf";
        case InferenceStatus::kInputMelFailed:           return "ncnn input(\"mel\") 失败";
        case InferenceStatus::kInputFaceFailed:          return "ncnn input(\"face\") 失败";
        case InferenceStatus::kExtractFailed:            return "ncnn extract(\"pred\") 失败";
        case InferenceStatus::kOutputEmpty:              return "ncnn 输出 pred 为空";
        case InferenceStatus::kInvalidOutputShape:       return "pred tensor 规格错误";
        case InferenceStatus::kNonFiniteOutput:          return "pred 包含 NaN 或 Inf";
        case InferenceStatus::kOutputRangeViolation:     return "pred 值域超出允许范围";
        case InferenceStatus::kRetryExhausted:           return "推理重试次数已耗尽";
        case InferenceStatus::kEngineAlreadyRunning:     return "推理调度器已经运行";
        case InferenceStatus::kEngineNotRunning:         return "推理调度器未运行";
        case InferenceStatus::kEngineStopping:           return "推理调度器正在停止";
        case InferenceStatus::kInvalidSchedulerConfig:   return "推理调度器配置无效";
        case InferenceStatus::kThreadBudgetExceeded:     return "推理线程预算超限";
        case InferenceStatus::kEmptyBatchInput:          return "Batch 输入为空";
        case InferenceStatus::kQueueWaitTimeout:         return "任务等待入队超时";
        case InferenceStatus::kTaskException:            return "推理任务执行异常";
        case InferenceStatus::kPartialFailure:           return "Batch 部分任务失败";
        case InferenceStatus::kAllFailed:                return "Batch 全部任务失败";
        case InferenceStatus::kUnknownError:             return "未知推理错误";
        }

        return "未知推理错误";
    }

} // namespace model
} // namespace digital_human
