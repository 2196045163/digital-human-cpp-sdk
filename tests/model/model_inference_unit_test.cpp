
#include <gtest/gtest.h>
#include <memory>

#include "model/model_inference.h"
#include "model/detail/inference_retry_runner.h"

using namespace digital_human::model;

// ============================================================================
// 无效模型快照必须在创建 Extractor 前失败，同时保留请求上下文
// ============================================================================
// 准备输入（Arrange）→ 调用接口（Act）→ 检查结果（Assert）
TEST(ModelInferenceTest, InvalidSnapshotPreservesContextAndReturnsEmptyPred) {
    // Arrange：构造异常 snapshot。
    ModelRuntimeSnapshot snapshot;
    snapshot.generation = 7;
    // model 默认是 nullptr，不需要额外赋值。

    // Arrange：构造带追踪信息的请求。
    NcnnWav2LipInput input;
    input.metadata.pts_ms = 1234;
    input.metadata.frame_index = 56;

    ModelInference model_inference;
    InferenceOptions options;

    // Act：实际调用被测接口。
    const SingleInferenceResult result = model_inference.Infer(snapshot, input, options);

    // success == false
    EXPECT_FALSE(result.success);
    // status == kModelUnavailable
    EXPECT_EQ(result.status, InferenceStatus::kModelUnavailable);
    // error_message 非空
    EXPECT_FALSE(result.error_message.empty());
    // value.pred.empty() == true
    EXPECT_TRUE(result.value.pred.empty());
    // value.model_generation == 7
    EXPECT_EQ(result.value.model_generation, 7u);
    // metadata 必须原样保留即 pts_ms 和 frame_index 原样保留
    ASSERT_TRUE(result.value.metadata.pts_ms.has_value());
    EXPECT_EQ(*result.value.metadata.pts_ms, 1234);

    ASSERT_TRUE(result.value.metadata.frame_index.has_value());
    EXPECT_EQ(*result.value.metadata.frame_index, 56);

    // attempt_count == 0，因为尚未创建 Extractor
    EXPECT_EQ(result.value.attempts.attempt_count, 0u);
    // input_validation_ms == 0.0，因为 snapshot 失败不属于输入校验
    EXPECT_DOUBLE_EQ(result.value.timing.input_validation_ms, 0.0);
}


// ============================================================================
// 有效 snapshot 下的空 Mel 必须在创建 Extractor 前失败
// ============================================================================
// 准备输入（Arrange）→ 调用接口（Act）→ 检查结果（Assert）
TEST(ModelInferenceTest, EmptyMelFailsBeforeExtractorCreation) {

    // Arrange：构造契约有效的 sentinel snapshot；本测试不会真正使用该 Net 执行推理。
    ModelRuntimeSnapshot snapshot;
    snapshot.model = std::make_shared<ncnn::Net>();
    snapshot.generation = 7;

    NcnnWav2LipInput input;
    input.metadata.pts_ms = 1234;
    input.metadata.frame_index = 56;

    ModelInference model_inference;
    InferenceOptions options;

    // 该 Net 没有加载网络，但 snapshot 在接口契约上有效。由于 mel 为空，
    // Infer() 必须在使用 Net 创建 Extractor 之前返回。
    const SingleInferenceResult result = model_inference.Infer(snapshot, input, options);

    // success == false
    EXPECT_FALSE(result.success);
    // status == kEmptyMelInput
    EXPECT_EQ(result.status, InferenceStatus::kEmptyMelInput);
    // error_message 非空
    EXPECT_FALSE(result.error_message.empty());
    // value.pred.empty() == true
    EXPECT_TRUE(result.value.pred.empty());
    // value.model_generation == 7
    EXPECT_EQ(result.value.model_generation, 7u);

    // metadata 必须原样保留即 pts_ms 和 frame_index 原样保留
    ASSERT_TRUE(result.value.metadata.pts_ms.has_value());
    EXPECT_EQ(*result.value.metadata.pts_ms, 1234);

    ASSERT_TRUE(result.value.metadata.frame_index.has_value());
    EXPECT_EQ(*result.value.metadata.frame_index, 56);

    // attempt_count == 0，因为尚未创建 Extractor
    EXPECT_EQ(result.value.attempts.attempt_count, 0u);

    // first_attempt_ms == 0.0
    EXPECT_DOUBLE_EQ(result.value.timing.first_attempt_ms, 0.0);

    // total_latency_ms >= input_validation_ms
    EXPECT_GE(result.value.timing.total_latency_ms, result.value.timing.input_validation_ms);

}

// ============================================================================
// 真实模型 + 合成合法 tensor 必须通过正式 ModelInference API
// ============================================================================
TEST(ModelInferenceTest, RealModelSingleSampleSucceeds) {
    // Arrange：加载真实 CPU 模型。本测试关闭 loader warmup，避免把 warmup 成功误当成
    // ModelInference 成功；下面的 Infer() 才是本测试要观察的正式执行路径。
    ModelLoader loader;
    ModelLoadOptions load_options;
    load_options.backend = ModelBackend::kCpu;
    load_options.enable_warmup = false;

    const ModelLoadResult load_result = loader.Load("models/wav2lip/wav2lip.param", load_options);
    ASSERT_TRUE(load_result.success) << load_result.error_message;

    const ModelRuntimeSnapshot snapshot = loader.AcquireSnapshot();
    ASSERT_TRUE(snapshot.IsValid());

    // 使用与 Wav2Lip 契约一致的三维、unpacked FP32 tensor。
    NcnnWav2LipInput input;
    input.mel = ncnn::Mat(16, 80, 1, sizeof(float));
    input.face = ncnn::Mat(96, 96, 6, sizeof(float));
    ASSERT_FALSE(input.mel.empty());
    ASSERT_FALSE(input.face.empty());
    input.mel.fill(0.0f);
    input.face.fill(0.0f);
    input.metadata.pts_ms = 4321;
    input.metadata.frame_index = 65;

    ModelInference model_inference;
    InferenceOptions options;

    // Act：正式 API 执行一次真实 ncnn 前向计算。
    const SingleInferenceResult result = model_inference.Infer(snapshot, input, options);

    // 断言成功状态、pred/output_info、attempt、generation/metadata 和计时。
    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.status, InferenceStatus::kOk);
    EXPECT_TRUE(result.error_message.empty());

    ASSERT_FALSE(result.value.pred.empty());
    EXPECT_EQ(result.value.pred.w, 96);
    EXPECT_EQ(result.value.pred.h, 96);
    EXPECT_EQ(result.value.pred.c, 3);

    // output_info 不能是默认值或硬编码值：记录的宽、高、通道数必须与最终交付的
    // pred 完全一致，证明调用方看到的诊断确实来自本次输出。
    EXPECT_EQ(result.value.output_info.width, result.value.pred.w);
    EXPECT_EQ(result.value.output_info.height, result.value.pred.h);
    EXPECT_EQ(result.value.output_info.channels, result.value.pred.c);

    // 两个布尔标志证明 finite 与 range 守门均已执行并通过；min <= max 锁定统计
    // 顺序正确，后两条断言验证当前严格 [0,1] 契约，而不是只相信布尔标志。
    EXPECT_TRUE(result.value.output_info.all_finite);
    EXPECT_TRUE(result.value.output_info.within_expected_range);
    EXPECT_LE(result.value.output_info.min_value,
              result.value.output_info.max_value);
    EXPECT_GE(result.value.output_info.min_value, 0.0f);
    EXPECT_LE(result.value.output_info.max_value, 1.0f);

    // 本测试首次尝试即成功，因此只发生一次 attempt，也不存在首次失败记录。
    // final_attempt_status=kOk 和两个 ncnn code=0 证明 input/extract 没有底层错误；
    // recovered_by_retry=false 则防止一次成功被错误标记成“重试恢复”。
    EXPECT_EQ(result.value.attempts.attempt_count, 1u);
    EXPECT_FALSE(result.value.attempts.first_failure_status.has_value());
    EXPECT_EQ(result.value.attempts.final_attempt_status,
              InferenceStatus::kOk);
    EXPECT_EQ(result.value.attempts.first_ncnn_error_code, 0);
    EXPECT_EQ(result.value.attempts.final_ncnn_error_code, 0);
    EXPECT_FALSE(result.value.attempts.recovered_by_retry);

    // 结果中的模型代次必须来自本次传入的 snapshot；PTS 和帧号也必须从 input
    // 原样透传，证明成功路径没有丢失或重新计算请求关联信息。
    EXPECT_EQ(result.value.model_generation, snapshot.generation);
    ASSERT_TRUE(result.value.metadata.pts_ms.has_value());
    EXPECT_EQ(*result.value.metadata.pts_ms, 4321);
    ASSERT_TRUE(result.value.metadata.frame_index.has_value());
    EXPECT_EQ(*result.value.metadata.frame_index, 65);

    // 正确性测试只验证计时字段的阶段语义，不约束具体速度：校验耗时允许因时钟
    // 分辨率而等于 0；真实前向 attempt 应有正耗时；检查点 4 未重试，所以
    // retry_attempts_ms 必须精确为 0；端到端耗时不能小于其中的首次 attempt。
    EXPECT_GE(result.value.timing.input_validation_ms, 0.0);
    EXPECT_GT(result.value.timing.first_attempt_ms, 0.0);
    EXPECT_DOUBLE_EQ(result.value.timing.retry_attempts_ms, 0.0);
    EXPECT_GE(result.value.timing.output_validation_ms, 0.0);
    EXPECT_GE(result.value.timing.total_latency_ms,
              result.value.timing.first_attempt_ms);
}

// ============================================================================
// Scripted fake：首次 extract 失败，额外重试一次后成功
// ============================================================================
TEST(InferenceRetryRunnerTest, ExtractFailureThenSuccess) {
    std::size_t fake_call_count = 0;
    ncnn::Mat expected_pred(1, sizeof(float));
    ASSERT_FALSE(expected_pred.empty());
    expected_pred.fill(0.5f);

    auto scripted_attempt = [&]() {
        detail::InferenceAttemptOutcome outcome;
        ++fake_call_count;

        if (fake_call_count == 1) {
            outcome.status = InferenceStatus::kExtractFailed;
            outcome.ncnn_error_code = -7;
            return outcome;
        }

        outcome.status = InferenceStatus::kOk;
        outcome.ncnn_error_code = 0;
        outcome.candidate_pred = expected_pred;
        return outcome;
    };

    const detail::InferenceRetryRunResult result = detail::RunWithRetry(1, scripted_attempt);

    // max_retries=1 允许首次失败后再尝试一次，因此 callable 共被调用两次。
    EXPECT_EQ(fake_call_count, 2u);

    // 第二次 attempt 成功，所以 runner 的顶层结果必须成功并返回 kOk。
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.status, InferenceStatus::kOk);

    // 摘要必须同时保留首次失败原因和最终成功状态。
    EXPECT_EQ(result.attempts.attempt_count, 2u);
    ASSERT_TRUE(result.attempts.first_failure_status.has_value());
    EXPECT_EQ(*result.attempts.first_failure_status,
              InferenceStatus::kExtractFailed);
    EXPECT_EQ(result.attempts.final_attempt_status,
              InferenceStatus::kOk);

    // 原始 ncnn 错误码也必须分别对应第一次失败和最后一次成功。
    EXPECT_EQ(result.attempts.first_ncnn_error_code, -7);
    EXPECT_EQ(result.attempts.final_ncnn_error_code, 0);
    EXPECT_TRUE(result.attempts.recovered_by_retry);

    // runner 只交付最终成功 attempt 的候选 pred，失败 attempt 不得污染它。
    ASSERT_FALSE(result.candidate_pred.empty());
    const float* pred_data = result.candidate_pred;
    ASSERT_NE(pred_data, nullptr);
    EXPECT_FLOAT_EQ(pred_data[0], 0.5f);

    // 计时仅检查字段已按阶段记录且非负，不给单元测试设置性能阈值。
    EXPECT_GE(result.first_attempt_ms, 0.0);
    EXPECT_GE(result.retry_attempts_ms, 0.0);

}

// ============================================================================
// Scripted fake：首次 extract 成功但输出为空，额外重试后得到可交付候选 pred
// ============================================================================
TEST(InferenceRetryRunnerTest, OutputEmptyThenSuccess) {
    std::size_t fake_call_count = 0;
    ncnn::Mat expected_pred(1, sizeof(float));
    ASSERT_FALSE(expected_pred.empty());
    expected_pred.fill(0.25f);

    auto scripted_attempt = [&]() {
        detail::InferenceAttemptOutcome outcome;
        ++fake_call_count;

        if (fake_call_count == 1) {
            // ncnn extract 本身返回 0，但没有生成 pred，因此原始 code 仍为 0。
            outcome.status = InferenceStatus::kOutputEmpty;
            outcome.ncnn_error_code = 0;
            return outcome;
        }

        outcome.status = InferenceStatus::kOk;
        outcome.ncnn_error_code = 0;
        outcome.candidate_pred = expected_pred;
        return outcome;
    };

    const detail::InferenceRetryRunResult result =
        detail::RunWithRetry(1, scripted_attempt);

    // 一次额外预算允许在空输出后重新创建执行上下文，共调用两次。
    EXPECT_EQ(fake_call_count, 2u);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.status, InferenceStatus::kOk);

    // 首次失败必须保留为 kOutputEmpty，末次状态则描述第二次已经成功。
    EXPECT_EQ(result.attempts.attempt_count, 2u);
    ASSERT_TRUE(result.attempts.first_failure_status.has_value());
    EXPECT_EQ(*result.attempts.first_failure_status,
              InferenceStatus::kOutputEmpty);
    EXPECT_EQ(result.attempts.final_attempt_status,
              InferenceStatus::kOk);
    EXPECT_EQ(result.attempts.first_ncnn_error_code, 0);
    EXPECT_EQ(result.attempts.final_ncnn_error_code, 0);
    EXPECT_TRUE(result.attempts.recovered_by_retry);

    // runner 只能交付第二次成功 attempt 的候选输出。
    ASSERT_FALSE(result.candidate_pred.empty());
    const float* pred_data = result.candidate_pred;
    ASSERT_NE(pred_data, nullptr);
    EXPECT_FLOAT_EQ(pred_data[0], 0.25f);

    // 首次与额外 attempt 使用相同计时边界；这里只验证字段非负。
    EXPECT_GE(result.first_attempt_ms, 0.0);
    EXPECT_GE(result.retry_attempts_ms, 0.0);
}

// ============================================================================
// Scripted fake：首次得到空输出，额外重试仍失败并耗尽预算
// ============================================================================
TEST(InferenceRetryRunnerTest, OutputEmptyThenExtractFailureExhaustsRetry) {
    std::size_t fake_call_count = 0;

    auto scripted_attempt = [&]() {
        detail::InferenceAttemptOutcome outcome;
        ++fake_call_count;

        if (fake_call_count == 1) {
            // extract 返回成功，但没有产生输出。
            outcome.status = InferenceStatus::kOutputEmpty;
            outcome.ncnn_error_code = 0;
            return outcome;
        }

        // 第二次尝试直接在 extract 阶段失败。
        outcome.status = InferenceStatus::kExtractFailed;
        outcome.ncnn_error_code = -7;
        return outcome;
    };

    const detail::InferenceRetryRunResult result = detail::RunWithRetry(1, scripted_attempt);

    // max_retries=1 允许首次失败后额外重试一次，因此共调用两次。
    EXPECT_EQ(fake_call_count, 2u);

    // 两次 attempt 均失败，runner 必须报告重试耗尽，而不是伪装成成功。
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.status, InferenceStatus::kRetryExhausted);

    // 摘要同时保留最初触发重试的原因和最后一次尝试的实际结果。
    EXPECT_EQ(result.attempts.attempt_count, 2u);
    ASSERT_TRUE(result.attempts.first_failure_status.has_value());
    EXPECT_EQ(*result.attempts.first_failure_status,
              InferenceStatus::kOutputEmpty);
    EXPECT_EQ(result.attempts.final_attempt_status,
              InferenceStatus::kExtractFailed);

    // 空输出发生在 extract 返回 0 之后；末次 extract 失败则保留其非零错误码。
    EXPECT_EQ(result.attempts.first_ncnn_error_code, 0);
    EXPECT_EQ(result.attempts.final_ncnn_error_code, -7);

    // 所有尝试均失败，因此不能标记为“经重试恢复”，也不能交付候选输出。
    EXPECT_FALSE(result.attempts.recovered_by_retry);
    EXPECT_TRUE(result.candidate_pred.empty());

    // 首次和额外尝试分别计时；只验证阶段字段非负，不设置性能阈值。
    EXPECT_GE(result.first_attempt_ms, 0.0);
    EXPECT_GE(result.retry_attempts_ms, 0.0);

}

// ============================================================================
// Scripted fake：max_retries=0 时保留首次可重试失败，不伪造“重试耗尽”
// ============================================================================
TEST(InferenceRetryRunnerTest, ZeroRetriesPreservesOriginalFailure) {
    std::size_t fake_call_count = 0;

    auto scripted_attempt = [&]() {
        detail::InferenceAttemptOutcome outcome;
        ++fake_call_count;
        outcome.status = InferenceStatus::kOutputEmpty;
        outcome.ncnn_error_code = 0;
        return outcome;
    };

    const detail::InferenceRetryRunResult result =
        detail::RunWithRetry(0, scripted_attempt);

    // max_retries 表示首次 attempt 之外的额外次数；取 0 时只能调用一次。
    EXPECT_EQ(fake_call_count, 1u);
    EXPECT_FALSE(result.success);

    // 没有真正执行过 retry，因此顶层必须保留 kOutputEmpty，不能返回
    // kRetryExhausted；后者只描述“至少重试过一次但仍全部失败”。
    EXPECT_EQ(result.status, InferenceStatus::kOutputEmpty);
    EXPECT_EQ(result.attempts.attempt_count, 1u);
    ASSERT_TRUE(result.attempts.first_failure_status.has_value());
    EXPECT_EQ(*result.attempts.first_failure_status,
              InferenceStatus::kOutputEmpty);
    EXPECT_EQ(result.attempts.final_attempt_status,
              InferenceStatus::kOutputEmpty);

    // 唯一一次 attempt 同时是首次和末次，其 ncnn code 为 0；没有成功恢复，
    // 也没有可交付的候选 pred。
    EXPECT_EQ(result.attempts.first_ncnn_error_code, 0);
    EXPECT_EQ(result.attempts.final_ncnn_error_code, 0);
    EXPECT_FALSE(result.attempts.recovered_by_retry);
    EXPECT_TRUE(result.candidate_pred.empty());

    // 首次计时允许为 0；由于没有额外 attempt，retry 耗时必须精确为 0。
    EXPECT_GE(result.first_attempt_ms, 0.0);
    EXPECT_DOUBLE_EQ(result.retry_attempts_ms, 0.0);
}

// ============================================================================
// Scripted fake：input 绑定和输出守门等确定性错误即使有预算也禁止重试
// ============================================================================
TEST(InferenceRetryRunnerTest, DeterministicFailuresDoNotRetry) {
    const InferenceStatus deterministic_statuses[] = {
        InferenceStatus::kInputMelFailed,
        InferenceStatus::kInputFaceFailed,
        InferenceStatus::kInvalidOutputShape,
        InferenceStatus::kNonFiniteOutput,
        InferenceStatus::kOutputRangeViolation,
    };

    for (const InferenceStatus expected_status : deterministic_statuses) {
        // SCOPED_TRACE 会在某个状态失败时打印枚举值，便于定位是哪一项策略回归。
        SCOPED_TRACE(static_cast<int>(expected_status));
        std::size_t fake_call_count = 0;
        const bool is_input_binding_failure =
            expected_status == InferenceStatus::kInputMelFailed ||
            expected_status == InferenceStatus::kInputFaceFailed;
        const int expected_ncnn_code = is_input_binding_failure ? -9 : 0;

        auto scripted_attempt = [&]() {
            detail::InferenceAttemptOutcome outcome;
            ++fake_call_count;
            outcome.status = expected_status;
            outcome.ncnn_error_code = expected_ncnn_code;
            return outcome;
        };

        // 故意提供多次 retry 预算，证明停止由错误分类决定，而不是预算耗尽。
        const detail::InferenceRetryRunResult result =
            detail::RunWithRetry(3, scripted_attempt);

        // 输入绑定或输出契约错误不会因重复相同 attempt 而自行修复，因此只调用一次，
        // 顶层保留具体原始状态，不能改写成 kRetryExhausted。
        EXPECT_EQ(fake_call_count, 1u);
        EXPECT_FALSE(result.success);
        EXPECT_EQ(result.status, expected_status);
        EXPECT_EQ(result.attempts.attempt_count, 1u);
        ASSERT_TRUE(result.attempts.first_failure_status.has_value());
        EXPECT_EQ(*result.attempts.first_failure_status, expected_status);
        EXPECT_EQ(result.attempts.final_attempt_status, expected_status);
        EXPECT_EQ(result.attempts.first_ncnn_error_code,
                  expected_ncnn_code);
        EXPECT_EQ(result.attempts.final_ncnn_error_code,
                  expected_ncnn_code);

        // 没有执行 retry，也就不可能“经重试恢复”或交付候选 pred。
        EXPECT_FALSE(result.attempts.recovered_by_retry);
        EXPECT_TRUE(result.candidate_pred.empty());
        EXPECT_GE(result.first_attempt_ms, 0.0);
        EXPECT_DOUBLE_EQ(result.retry_attempts_ms, 0.0);
    }


}
