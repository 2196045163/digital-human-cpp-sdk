#pragma once

#include <cstdint>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <ncnn/mat.h>

#include "model/model_loader.h"           // ModelRuntimeSnapshot
#include "model/ncnn_input_adapter.h"     // NcnnWav2LipInput, ModelInputMetadata

namespace digital_human {
namespace model {

// ============================================================================
// 状态码
// ============================================================================

/// @brief 推理状态码，每个失败点对应明确枚举，调用方可按组 switch 做不同处理
/// @note  ncnn 原始返回码放入 diagnostics，不把公开枚举扩张成无规律数字表
enum class InferenceStatus {
    // ---- 成功 ----
    kOk,                        ///< 推理成功

    // ---- 生命周期 ----
    kModelUnavailable,          ///< 模型快照无效（未加载或加载失败）

    // ---- 输入校验（确定性错误，不重试） ----
    kEmptyMelInput,             ///< Mel 输入为空
    kInvalidMelShape,           ///< Mel tensor 规格错误（w/h/c、elempack 或 elemsize 不匹配）
    kNonFiniteMelInput,         ///< Mel 包含 NaN 或 Inf
    kEmptyFaceInput,            ///< Face 输入为空
    kInvalidFaceShape,          ///< Face tensor 规格错误（w/h/c、elempack 或 elemsize 不匹配）
    kNonFiniteFaceInput,        ///< Face 包含 NaN 或 Inf

    // ---- ncnn 执行 ----
    kInputMelFailed,            ///< input("mel") 返回非 0（blob 名不匹配）
    kInputFaceFailed,           ///< input("face") 返回非 0
    kExtractFailed,             ///< extract("pred") 返回非 0（可重试）
    kOutputEmpty,               ///< extract 成功但 pred 为空（可重试）

    // ---- 输出守门（确定性错误，不重试） ----
    kInvalidOutputShape,        ///< pred tensor 规格错误（w/h/c、elempack 或 elemsize 不匹配）
    kNonFiniteOutput,           ///< pred 包含 NaN 或 Inf
    kOutputRangeViolation,      ///< pred 值域明显超出 [0,1]

    // ---- 重试 ----
    kRetryExhausted,            ///< 至少执行过一次额外重试，但所有 attempt 仍失败（保留首次/末次原因）

    // ---- Scheduler 生命周期与配置 ----
    kEngineAlreadyRunning,      ///< Scheduler 已运行，不能重复 Start
    kEngineNotRunning,          ///< Scheduler 尚未 Start 或已经 Stop
    kEngineStopping,            ///< Scheduler 正在停止，不再接收新任务
    kInvalidSchedulerConfig,    ///< worker、queue 或 timeout 配置无效
    kThreadBudgetExceeded,      ///< worker_count × ncnn threads 超出当前机器线程预算

    // ---- Scheduler batch ----
    kEmptyBatchInput,           ///< batch 输入为空
    kQueueWaitTimeout,          ///< 在统一 enqueue deadline 前未获得队列空位，任务从未被接收
    kTaskException,             ///< 已接收 task 抛出异常，worker 仍可继续运行
    kPartialFailure,            ///< batch 中同时存在成功项和失败项
    kAllFailed,                 ///< batch 中没有任何成功项

    kUnknownError               ///< 未知错误（兜底）
};

// ============================================================================
// 配置结构
// ============================================================================

/// @brief 重试策略
struct RetryPolicy {
    std::size_t max_retries = 1;    ///< 首次尝试之外最多再试次数；为 0 时失败保留原状态，不返回 kRetryExhausted
};

/// @brief 单次推理选项
struct InferenceOptions {
    RetryPolicy retry;          ///< 重试策略
    bool light_mode = true;     ///< Extractor 轻量模式：中间 blob 用完即回收，降低单次推理峰值内存
};


// ============================================================================
// 诊断与输出结构
// ============================================================================

/// @brief 分阶段计时，用于定位瓶颈：用户总延迟 ≠ 计算时间
struct InferenceTiming {
    double queue_wait_ms = 0.0;              ///< Scheduler 入队等待耗时；直接调用 ModelInference 时为 0
    double input_validation_ms = 0.0;       ///< 输入 shape/finite 校验耗时
    double first_attempt_ms = 0.0;          ///< 首次 create/config/input/extract/empty-check 总耗时
    double retry_attempts_ms = 0.0;         ///< 所有额外重试的同口径 attempt 总耗时
    double output_validation_ms = 0.0;      ///< 输出 shape/finite/range 校验耗时
    double total_latency_ms = 0.0;          ///< 端到端总耗时
};

/// @brief 输出 tensor 的诊断信息，供 example/test 使用，不再重复推测 shape
struct InferenceOutputInfo {
    int width = 0;                  ///< pred 实际宽度
    int height = 0;                 ///< pred 实际高度
    int channels = 0;               ///< pred 实际通道数
    float min_value = 0.0f;         ///< pred 最小值
    float max_value = 0.0f;         ///< pred 最大值
    bool all_finite = false;        ///< pred 所有值均为有限值
    bool within_expected_range = false; ///< pred 值域在 [0,1] 容差内
};

/// @brief 每次推理尝试的重试信息
/// @note  首次失败用 std::optional<InferenceStatus>：首次成功时 first_failure_status = std::nullopt
struct InferenceAttemptInfo {
    std::size_t attempt_count = 0;                       ///< 总尝试次数（≥1）
    std::optional<InferenceStatus> first_failure_status; ///< 首次失败原因（成功则为 std::nullopt）
    InferenceStatus final_attempt_status = InferenceStatus::kUnknownError; ///< 最终状态
    int first_ncnn_error_code = 0;                       ///< ncnn 首次返回码
    int final_ncnn_error_code = 0;                       ///< ncnn 末次返回码
    bool recovered_by_retry = false;                     ///< 是否靠重试恢复（首次失败→重试成功）
};

/// @brief 单样本推理输出
struct InferenceOutput {
    ncnn::Mat pred;                          ///< 原始输出 tensor（成功时非空）
    ModelInputMetadata metadata;             ///< 请求时间元数据；失败也透传，用于按帧号/PTS 追踪
    InferenceTiming timing;                  ///< 分阶段耗时
    InferenceOutputInfo output_info;         ///< 输出 tensor 诊断
    InferenceAttemptInfo attempts;           ///< 重试记录
    std::uint64_t model_generation = 0;      ///< 请求携带的模型代次；失败也保留传入值用于追踪
};

// ============================================================================
// 统一 Result 模板
// ============================================================================

/// @brief 统一推理 Result 外壳：所有调用都先看 success，再按 status 分支，成功数据在 value
/// @note  不提供 operator bool()，初学阶段显式写 result.success 更清楚
template <typename T>
struct InferenceResult {
    bool success = false;                                    ///< 是否成功（默认失败）
    InferenceStatus status = InferenceStatus::kUnknownError; ///< 语义状态码
    std::string error_message;                               ///< 人类可读错误描述
    T value;                                                 ///< 成功时的业务数据
};

/// @name 类型别名
using SingleInferenceResult  = InferenceResult<InferenceOutput>;

// ============================================================================
// 模型推理模块
// ============================================================================

/// @brief 模型推理模块（Model Inference）
///
/// 在只读 Wav2Lip 模型快照上，安全执行单样本 ncnn 前向计算，
/// 返回经过 shape、有限性和值域守门的原始 pred 及完整计时、重试和代次诊断。
///
/// 核心设计：
/// - 每次推理创建局部 ncnn::Extractor（不跨请求共享，并发隔离）
/// - 只读模型快照：推理时不修改 Net::opt，generation 保证批次可复现
/// - 无状态单样本推理内核。每次调用显式传入模型快照和推理选项。
/// - 核心：局部 Extractor + 输入/输出守门 + 分类重试。
/// - 分类重试：extract 失败/空输出可重试，确定性问题不可重试
/// - 不持有模型、不管理线程池、不做任务调度。
///
/// 职责边界：
/// - 不做输入构造 / resize / 归一化 / Mel 提取（上游模块的事）
/// - 不做 pred 转 cv::Mat / 反归一化 / 贴回原图（输出处理模块的事）
/// - 不做模型文件加载（ModelLoader 的事）
/// - 不做异步 callback / future API（本轮只做同步）
/// - 不强杀正在执行的 ncnn（C++ 无法安全终止第三方库线程）
///
/// 一个公开动作形成完整而不臃肿的接口：Infer
class ModelInference {
public:
    /// @brief 同步单样本推理
    /// @param input mel + face ncnn::Mat + metadata
    /// @return 成功时 pred 非空且经守门校验；失败时 status 区分输入/执行/输出错误
    /// @note  每次调用创建局部 Extractor，不跨请求共享
    SingleInferenceResult Infer(const ModelRuntimeSnapshot& snapshot,
                                const NcnnWav2LipInput& input,
                                const InferenceOptions& options) const;

    /// @brief 状态码 → 人类可读字符串
    static std::string StatusToString(InferenceStatus status);
};

} // namespace model
} // namespace digital_human
