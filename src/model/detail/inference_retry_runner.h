#pragma once

#include <chrono>
#include <cstddef>

#include "model/model_inference.h"

namespace digital_human {
namespace model {
namespace detail {

/// @brief 一次独立推理 attempt 的最小结果
/// @note  callable 每被调用一次，都必须创建新的局部 Extractor；本结构不保存请求上下文或总计时。
struct InferenceAttemptOutcome {
    InferenceStatus status = InferenceStatus::kUnknownError; ///< 本次尝试的语义状态
    int ncnn_error_code = 0;                                 ///< 本次 input/extract 的原始 ncnn 返回码
    ncnn::Mat candidate_pred;                                ///< 仅 kOk 时携带待输出守门的候选 pred
};

/// @brief retry runner 对一次请求内所有 attempt 的汇总结果
/// @note  这里只汇总重试相关信息；metadata、generation、输出守门和 total latency 仍由 Infer 负责。
struct InferenceRetryRunResult {
    bool success = false;                                    ///< 是否最终得到一次成功 attempt
    InferenceStatus status = InferenceStatus::kUnknownError; ///< 顶层状态：kOk、具体错误或 kRetryExhausted
    ncnn::Mat candidate_pred;                                ///< 仅 success=true 时交回 Infer 继续输出守门
    InferenceAttemptInfo attempts;                           ///< 次数、首次/末次状态和原始错误码
    double first_attempt_ms = 0.0;                           ///< 首次 callable 调用耗时
    double retry_attempts_ms = 0.0;                          ///< 所有额外 callable 调用耗时之和
};

/// @brief 判断一次推理失败是否属于允许有限重试的瞬时错误
/// @param status 本次 attempt 返回的语义状态码
/// @return 仅 kExtractFailed 或 kOutputEmpty 返回 true，其余状态（包括 kOk）返回 false
/// @note  本 helper 只做无副作用的状态分类，不创建 Extractor、不计数，也不执行重试
inline bool IsRetryableStatus(InferenceStatus status) {
    switch (status) {
    case InferenceStatus::kExtractFailed:
    case InferenceStatus::kOutputEmpty:
        return true;

    default:
        return false;
    }
}

/// @brief 按策略执行一次首次 attempt 和有限次额外 retry，并汇总可观察诊断
/// @tparam AttemptCallable 可像函数一样调用、且每次返回 InferenceAttemptOutcome 的内部对象
/// @param max_retries 首次 attempt 之外最多允许的额外调用次数
/// @param attempt 执行一次独立推理尝试的 callable；生产实现每次调用都必须新建 Extractor
/// @return 重试过程汇总；失败时 candidate_pred 为空，成功时交回 Infer 继续输出守门
/// @note  max_retries=0 时从不返回 kRetryExhausted，而是保留首次 attempt 的原始失败状态
template <typename AttemptCallable>
InferenceRetryRunResult RunWithRetry(std::size_t max_retries,
                                     AttemptCallable&& attempt) {
    InferenceRetryRunResult result;
    std::size_t retry_index = 0;

    while (true) {
        const auto attempt_start = std::chrono::steady_clock::now();
        InferenceAttemptOutcome outcome = attempt();
        const double attempt_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - attempt_start)
                .count();

        ++result.attempts.attempt_count;
        const bool is_first_attempt = result.attempts.attempt_count == 1;

        if (is_first_attempt) {
            result.first_attempt_ms = attempt_ms;
            result.attempts.first_ncnn_error_code = outcome.ncnn_error_code;
            if (outcome.status != InferenceStatus::kOk) {
                result.attempts.first_failure_status = outcome.status;
            }
        } else {
            result.retry_attempts_ms += attempt_ms;
        }

        result.attempts.final_attempt_status = outcome.status;
        result.attempts.final_ncnn_error_code = outcome.ncnn_error_code;

        if (outcome.status == InferenceStatus::kOk) {
            result.success = true;
            result.status = InferenceStatus::kOk;
            result.candidate_pred = outcome.candidate_pred;
            result.attempts.recovered_by_retry =
                result.attempts.attempt_count > 1;
            return result;
        }

        if (!IsRetryableStatus(outcome.status)) {
            result.status = outcome.status;
            return result;
        }

        if (retry_index >= max_retries) {
            result.status = retry_index == 0
                ? outcome.status
                : InferenceStatus::kRetryExhausted;
            return result;
        }

        ++retry_index;
    }
}

} // namespace detail
} // namespace model
} // namespace digital_human
