/// @file golden_upstream_integration_test.cpp
/// @brief 第 4 层：真实 golden face/audio → 上游模块 → Builder → Adapter → 模型 → pred
///
/// 和合成 smoke 的区别：
///   人脸不是手写渐变图，是 face.jpg → ImageLoader → FaceDetector → FaceAligner 真实处理
///   Mel 不是 1000*f+t，是 audio.wav → AudioLoader → Preprocessor → Framer → Mel 真实提取
///
/// 验证的是：上游模块真实产出能否被下游"消费"——type/shape/range 都能过 Builder 的校验。
/// 不验证口型视觉质量。
///
/// 审计已在 2026-07-18 完成：状态码断言全覆盖、失败路径有 independent 用例、JSON 0.00 字段已标注。

#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/image_loader.h"
#include "audio/audio_loader.h"
#include "audio/audio_preprocessor.h"
#include "audio/audio_framer.h"
#include "audio/audio_mel_feature_extract.h"
#include "model/model_loader.h"
#include "model/model_inference.h"
#include "model/input_processor.h"
#include "model/ncnn_input_adapter.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <ncnn/net.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

using namespace digital_human;

static void WriteJson(const std::string& path, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);
    std::ofstream f(path);
    ASSERT_TRUE(f.is_open()) << "cannot write " << path;
    f << content;
}

struct PredStats {
    bool all_finite = true;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
};

static PredStats InspectPred(const ncnn::Mat& pred) {
    PredStats stats;
    for (int channel_index = 0;
         channel_index < pred.c && stats.all_finite;
         ++channel_index) {
        const float* channel_data = pred.channel(channel_index);
        const int elements_per_channel = pred.w * pred.h;
        for (int element_index = 0;
             element_index < elements_per_channel;
             ++element_index) {
            const float value = channel_data[element_index];
            if (!std::isfinite(value)) {
                stats.all_finite = false;
                break;
            }
            stats.min_value = std::min(stats.min_value, value);
            stats.max_value = std::max(stats.max_value, value);
        }
    }
    return stats;
}

static std::vector<std::size_t> SelectRepresentativeChunkIndices(
    std::size_t chunk_count) {
    if (chunk_count == 0) {
        return {};
    }

    std::vector<std::size_t> indices = {
        0,
        chunk_count / 4,
        chunk_count / 2,
        (chunk_count * 3) / 4,
        chunk_count - 1
    };
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

static std::string JoinIndices(const std::vector<std::size_t>& indices) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << indices[index];
    }
    return stream.str();
}

// ============================================================================
// 成功路径：golden face.jpg + audio.wav → pred
// ============================================================================
TEST(GoldenUpstreamIntegration, FaceAndAudioToPred) {
    // ---- 0. 加载模型 ----
    model::ModelLoader loader;
    model::ModelLoadOptions load_opts;
    load_opts.backend = model::ModelBackend::kCpu;
    load_opts.enable_warmup = true;

    auto load_r = loader.Load("models/wav2lip/wav2lip.param", load_opts);
    ASSERT_TRUE(load_r.success)
        << model::ModelLoader::StatusToString(load_r.status);
    // 正式推理 API 接收完整模型快照，而不是只有 Net 指针。generation、backend
    // 和实际线程配置共同描述这组 sampled chunk 使用的模型运行上下文。
    const model::ModelRuntimeSnapshot snapshot = loader.AcquireSnapshot();
    ASSERT_TRUE(snapshot.IsValid());

    // ModelInference 自身无状态，可以对同一只读 snapshot 连续执行多个 chunk；
    // 每次 Infer 内部仍会为每个 attempt 创建独立 Extractor。
    model::ModelInference model_inference;
    model::InferenceOptions inference_options;

    // ================================================================
    // 人脸链路
    // ================================================================

    // ---- 1a. ImageLoader ----
    core::ImageLoader img_loader;
    auto img_r = img_loader.LoadFromFile("testdata/golden/face.jpg");
    ASSERT_TRUE(img_r.success)
        << core::ImageLoader::StatusToString(img_r.status);
    ASSERT_EQ(img_r.status, core::ImageLoadStatus::kOk);
    const cv::Mat& original = img_r.image;
    ASSERT_FALSE(original.empty());
    ASSERT_EQ(original.type(), CV_8UC3);

    // ---- 1b. FaceDetector ----
    core::FaceDetector detector;
    auto load_model_r = detector.LoadLandmarkModel(
        "models/shape_predictor_68_face_landmarks.dat");
    ASSERT_TRUE(load_model_r.success) << load_model_r.error_message;

    core::FaceDetectOptions det_opts;
    auto det_r = detector.DetectAndLandmark(original, det_opts);
    ASSERT_TRUE(det_r.success)
        << core::FaceDetector::StatusToString(det_r.status);
    ASSERT_EQ(det_r.status, core::FaceDetectStatus::kOk);
    ASSERT_FALSE(det_r.detection.faces.empty());
    ASSERT_FALSE(det_r.landmarks.empty());

    cv::Rect face_rect = det_r.detection.faces[0].rect;
    std::vector<cv::Point> landmarks = det_r.landmarks[0].landmarks;

    // ---- 1c. FaceAligner ----
    core::FaceAligner aligner;
    core::FaceAlignmentOptions align_opts;
    align_opts.target_size = 96;

    auto align_r = aligner.Align(original, face_rect, landmarks, align_opts);
    ASSERT_TRUE(align_r.success)
        << core::FaceAligner::StatusToString(align_r.status);
    ASSERT_EQ(align_r.status, core::ImagePreprocessStatus::kOk);

    const cv::Mat& aligned = align_r.aligned_face;
    ASSERT_EQ(aligned.rows, 96);
    ASSERT_EQ(aligned.cols, 96);
    ASSERT_EQ(aligned.type(), CV_8UC3);

    // ================================================================
    // 音频链路
    // ================================================================

    // ---- 2a. AudioLoader ----
    audio::AudioLoader audio_loader(16000);
    auto audio_r = audio_loader.LoadFromFile("testdata/golden/audio.wav");
    ASSERT_TRUE(audio_r.success)
        << audio::AudioLoader::StatusToString(audio_r.status);
    ASSERT_EQ(audio_r.status, audio::AudioLoadStatus::kOk);
    ASSERT_FALSE(audio_r.audio.pcm.empty());
    ASSERT_EQ(audio_r.audio.sample_rate, 16000);

    // ---- 2b. AudioPreprocessor ----
    audio::AudioPreprocessor preproc;
    auto preproc_r = preproc.Process(audio_r.audio.pcm);
    ASSERT_TRUE(preproc_r.success)
        << audio::AudioPreprocessor::StatusToString(preproc_r.status);
    ASSERT_EQ(preproc_r.status, audio::AudioPreprocessStatus::kOk);
    ASSERT_FALSE(preproc_r.pcm.empty());

    // ---- 2c. AudioFramer (流式接口) ----
    audio::AudioFramer framer;
    auto frames = framer.ProcessFrame(preproc_r.pcm);
    auto flushed = framer.FlushFrames();
    frames.insert(frames.end(), flushed.begin(), flushed.end());
    ASSERT_FALSE(frames.empty());

    // ---- 2d. MelFeatureExtractor ----
    std::vector<std::vector<float>> frame_samples;
    frame_samples.reserve(frames.size());
    for (const auto& f : frames) {
        frame_samples.push_back(f.samples);
    }
    audio::MelFeatureExtractor mel_ext;
    auto mel_r = mel_ext.ExtractBatch(frame_samples);
    ASSERT_TRUE(mel_r.success)
        << audio::MelFeatureExtractor::StatusToString(mel_r.status);
    ASSERT_EQ(mel_r.status, audio::MelFeatureStatus::kOk);
    ASSERT_FALSE(mel_r.mel.empty());
    ASSERT_GT(mel_r.mel.rows, 16) << "need at least 16 mel frames for one chunk";

    // ---- 2e. BuildWav2LipChunks（按时间位置选择代表性 chunk） ----
    auto chunk_r = mel_ext.BuildWav2LipChunks(mel_r.mel);
    ASSERT_TRUE(chunk_r.success)
        << audio::MelFeatureExtractor::StatusToString(chunk_r.status);
    ASSERT_FALSE(chunk_r.chunks.empty());

    const std::vector<std::size_t> sampled_chunk_indices =
        SelectRepresentativeChunkIndices(chunk_r.chunks.size());
    ASSERT_FALSE(sampled_chunk_indices.empty());

    const std::vector<float>& mel_chunk =
        chunk_r.chunks[sampled_chunk_indices.front()];
    ASSERT_EQ(mel_chunk.size(), 1280u);

    auto [mel_min_it, mel_max_it] = std::minmax_element(mel_chunk.begin(), mel_chunk.end());
    float mel_min = *mel_min_it;
    float mel_max = *mel_max_it;
    EXPECT_GE(mel_min, -4.5f) << "Mel min outside expected range";
    EXPECT_LE(mel_max, 4.5f)  << "Mel max outside expected range";

    // ================================================================
    // 汇合：Builder → Adapter → 推理
    // ================================================================

    // ---- 3. Builder ----
    model::ModelInputMetadata meta;
    meta.pts_ms = 0;
    meta.frame_index = 0;

    model::Wav2LipInputBuilder builder;
    auto build_r = builder.Build(aligned, mel_chunk, meta);
    ASSERT_TRUE(build_r.success)
        << model::Wav2LipInputBuilder::StatusToString(build_r.status);
    ASSERT_EQ(build_r.status, model::ModelInputStatus::kOk);

    EXPECT_EQ(build_r.info.face_channels, 6);
    EXPECT_EQ(build_r.info.face_height, 96);
    EXPECT_EQ(build_r.info.face_width, 96);
    EXPECT_EQ(build_r.info.mel_bins, 80);
    EXPECT_EQ(build_r.info.mel_frames, 16);
    EXPECT_FALSE(build_r.info.has_nan_or_inf);

    // ---- 4. Adapter ----
    model::NcnnInputAdapter adapter;
    auto adapt_r = adapter.Adapt(build_r.data);
    ASSERT_TRUE(adapt_r.success) << adapt_r.error_message;
    EXPECT_EQ(adapt_r.input.face.c, 6);
    EXPECT_EQ(adapt_r.input.mel.c, 1);

    // ---- 5. 正式 ModelInference API ----
    // 第一条 representative chunk 先做完整断言；公开 pred 只有通过 snapshot/input、
    // 分类重试和输出三道数值守门后才会出现在 SingleInferenceResult 中。
    const model::SingleInferenceResult first_inference_r =
        model_inference.Infer(snapshot, adapt_r.input, inference_options);
    ASSERT_TRUE(first_inference_r.success) << first_inference_r.error_message;
    ASSERT_EQ(first_inference_r.status, model::InferenceStatus::kOk);
    ASSERT_TRUE(first_inference_r.error_message.empty());

    const ncnn::Mat& pred = first_inference_r.value.pred;
    ASSERT_FALSE(pred.empty());
    ASSERT_EQ(first_inference_r.value.output_info.width, pred.w);
    ASSERT_EQ(first_inference_r.value.output_info.height, pred.h);
    ASSERT_EQ(first_inference_r.value.output_info.channels, pred.c);
    ASSERT_EQ(pred.w, 96);
    ASSERT_EQ(pred.h, 96);
    ASSERT_EQ(pred.c, 3);
    ASSERT_EQ(pred.elempack, 1);
    ASSERT_EQ(pred.elemsize, sizeof(float));
    ASSERT_TRUE(first_inference_r.value.output_info.all_finite);
    ASSERT_TRUE(first_inference_r.value.output_info.within_expected_range);
    ASSERT_GE(first_inference_r.value.attempts.attempt_count, 1u);
    ASSERT_EQ(first_inference_r.value.attempts.final_attempt_status,
              model::InferenceStatus::kOk);
    EXPECT_EQ(first_inference_r.value.attempts.recovered_by_retry,
              first_inference_r.value.attempts.attempt_count > 1u);
    EXPECT_GE(first_inference_r.value.timing.input_validation_ms, 0.0);
    EXPECT_GE(first_inference_r.value.timing.first_attempt_ms, 0.0);
    EXPECT_GE(first_inference_r.value.timing.retry_attempts_ms, 0.0);
    EXPECT_GE(first_inference_r.value.timing.output_validation_ms, 0.0);
    EXPECT_GE(first_inference_r.value.timing.total_latency_ms,
              first_inference_r.value.timing.first_attempt_ms);

    // 请求上下文和模型身份必须由正式 Result 透传；同一 sampled 集合后续都要
    // 与这里记录的 generation 一致。
    ASSERT_EQ(first_inference_r.value.model_generation, snapshot.generation);
    ASSERT_TRUE(first_inference_r.value.metadata.frame_index.has_value());
    EXPECT_EQ(*first_inference_r.value.metadata.frame_index,
              static_cast<int64_t>(sampled_chunk_indices.front()));
    ASSERT_TRUE(first_inference_r.value.metadata.pts_ms.has_value());
    EXPECT_EQ(*first_inference_r.value.metadata.pts_ms, 0);

    // InspectPred 保留为独立交叉检查：JSON 的正式诊断取自 output_info，但测试
    // 仍自行扫描 Mat，防止 Result 中的统计字段只是默认值或错误赋值。
    const PredStats first_pred_stats = InspectPred(pred);
    ASSERT_TRUE(first_pred_stats.all_finite) << "pred contains NaN/Inf";
    EXPECT_FLOAT_EQ(first_pred_stats.min_value,
                    first_inference_r.value.output_info.min_value);
    EXPECT_FLOAT_EQ(first_pred_stats.max_value,
                    first_inference_r.value.output_info.max_value);
    const bool non_zero =
        std::fabs(first_pred_stats.min_value) > 1e-10f ||
        std::fabs(first_pred_stats.max_value) > 1e-10f;
    EXPECT_TRUE(non_zero) << "pred is all zeros";

    // 单个 chunk 的权威字段来自 Result；以下变量只负责聚合 5 个 Result。
    float sampled_pred_min = first_inference_r.value.output_info.min_value;
    float sampled_pred_max = first_inference_r.value.output_info.max_value;
    bool sampled_pred_all_finite =
        first_inference_r.value.output_info.all_finite;
    bool sampled_pred_all_within_range =
        first_inference_r.value.output_info.within_expected_range;
    std::size_t sampled_total_attempt_count =
        first_inference_r.value.attempts.attempt_count;
    std::size_t sampled_max_attempt_count =
        first_inference_r.value.attempts.attempt_count;
    std::size_t sampled_retry_recovered_count =
        first_inference_r.value.attempts.recovered_by_retry ? 1u : 0u;
    double sampled_input_validation_ms_sum =
        first_inference_r.value.timing.input_validation_ms;
    double sampled_first_attempt_ms_sum =
        first_inference_r.value.timing.first_attempt_ms;
    double sampled_retry_attempts_ms_sum =
        first_inference_r.value.timing.retry_attempts_ms;
    double sampled_output_validation_ms_sum =
        first_inference_r.value.timing.output_validation_ms;
    double sampled_total_latency_ms_sum =
        first_inference_r.value.timing.total_latency_ms;

    // 第一个样本沿用上面的完整断言；其余样本只补充跨时间位置的 pred 极值证据。
    for (std::size_t sample_offset = 1;
         sample_offset < sampled_chunk_indices.size();
         ++sample_offset) {
        const std::size_t chunk_index = sampled_chunk_indices[sample_offset];
        SCOPED_TRACE("sampled chunk index=" + std::to_string(chunk_index));

        model::ModelInputMetadata sampled_meta;
        sampled_meta.frame_index = static_cast<int64_t>(chunk_index);

        auto sampled_build_r =
            builder.Build(aligned, chunk_r.chunks[chunk_index], sampled_meta);
        ASSERT_TRUE(sampled_build_r.success) << sampled_build_r.error_message;

        auto sampled_adapt_r = adapter.Adapt(sampled_build_r.data);
        ASSERT_TRUE(sampled_adapt_r.success) << sampled_adapt_r.error_message;

        const model::SingleInferenceResult sampled_inference_r =
            model_inference.Infer(
                snapshot, sampled_adapt_r.input, inference_options);
        ASSERT_TRUE(sampled_inference_r.success)
            << sampled_inference_r.error_message;
        ASSERT_EQ(sampled_inference_r.status, model::InferenceStatus::kOk);

        const ncnn::Mat& sampled_pred = sampled_inference_r.value.pred;
        ASSERT_FALSE(sampled_pred.empty());
        ASSERT_EQ(sampled_inference_r.value.output_info.width, sampled_pred.w);
        ASSERT_EQ(sampled_inference_r.value.output_info.height, sampled_pred.h);
        ASSERT_EQ(sampled_inference_r.value.output_info.channels, sampled_pred.c);
        ASSERT_EQ(sampled_pred.w, pred.w);
        ASSERT_EQ(sampled_pred.h, pred.h);
        ASSERT_EQ(sampled_pred.c, pred.c);
        ASSERT_EQ(sampled_pred.elempack, 1);
        ASSERT_EQ(sampled_pred.elemsize, sizeof(float));
        ASSERT_TRUE(sampled_inference_r.value.output_info.all_finite);
        ASSERT_TRUE(sampled_inference_r.value.output_info.within_expected_range);
        ASSERT_GE(sampled_inference_r.value.attempts.attempt_count, 1u);
        ASSERT_EQ(sampled_inference_r.value.attempts.final_attempt_status,
                  model::InferenceStatus::kOk);
        EXPECT_EQ(sampled_inference_r.value.attempts.recovered_by_retry,
                  sampled_inference_r.value.attempts.attempt_count > 1u);
        EXPECT_GE(sampled_inference_r.value.timing.input_validation_ms, 0.0);
        EXPECT_GE(sampled_inference_r.value.timing.first_attempt_ms, 0.0);
        EXPECT_GE(sampled_inference_r.value.timing.retry_attempts_ms, 0.0);
        EXPECT_GE(sampled_inference_r.value.timing.output_validation_ms, 0.0);
        EXPECT_GE(sampled_inference_r.value.timing.total_latency_ms,
                  sampled_inference_r.value.timing.first_attempt_ms);

        // 所有 sampled chunks 必须使用入口处固定的同一 snapshot；generation
        // 不一致会破坏聚合结果的可比较性和可复现性，应立即失败。
        ASSERT_EQ(sampled_inference_r.value.model_generation,
                  first_inference_r.value.model_generation);
        ASSERT_TRUE(sampled_inference_r.value.metadata.frame_index.has_value());
        EXPECT_EQ(*sampled_inference_r.value.metadata.frame_index,
                  static_cast<int64_t>(chunk_index));

        const PredStats sampled_stats = InspectPred(sampled_pred);
        ASSERT_TRUE(sampled_stats.all_finite) << "pred contains NaN/Inf";
        EXPECT_FLOAT_EQ(sampled_stats.min_value,
                        sampled_inference_r.value.output_info.min_value);
        EXPECT_FLOAT_EQ(sampled_stats.max_value,
                        sampled_inference_r.value.output_info.max_value);

        // 聚合时读取每个正式 Result 的诊断，不绕过 API 重复定义主证据。
        sampled_pred_min = std::min(
            sampled_pred_min,
            sampled_inference_r.value.output_info.min_value);
        sampled_pred_max = std::max(
            sampled_pred_max,
            sampled_inference_r.value.output_info.max_value);
        sampled_pred_all_finite =
            sampled_pred_all_finite &&
            sampled_inference_r.value.output_info.all_finite;
        sampled_pred_all_within_range =
            sampled_pred_all_within_range &&
            sampled_inference_r.value.output_info.within_expected_range;
        sampled_total_attempt_count +=
            sampled_inference_r.value.attempts.attempt_count;
        sampled_max_attempt_count = std::max(
            sampled_max_attempt_count,
            sampled_inference_r.value.attempts.attempt_count);
        if (sampled_inference_r.value.attempts.recovered_by_retry) {
            ++sampled_retry_recovered_count;
        }
        sampled_input_validation_ms_sum +=
            sampled_inference_r.value.timing.input_validation_ms;
        sampled_first_attempt_ms_sum +=
            sampled_inference_r.value.timing.first_attempt_ms;
        sampled_retry_attempts_ms_sum +=
            sampled_inference_r.value.timing.retry_attempts_ms;
        sampled_output_validation_ms_sum +=
            sampled_inference_r.value.timing.output_validation_ms;
        sampled_total_latency_ms_sum +=
            sampled_inference_r.value.timing.total_latency_ms;
    }

    ASSERT_TRUE(sampled_pred_all_finite);
    ASSERT_TRUE(sampled_pred_all_within_range);
    const double sampled_total_latency_ms_avg =
        sampled_total_latency_ms_sum /
        static_cast<double>(sampled_chunk_indices.size());

    const std::string sampled_indices_text =
        JoinIndices(sampled_chunk_indices);

    // ================================================================
    // Golden JSON 证据
    // ================================================================
    {
        char buf[2048];
        const int json_size = std::snprintf(buf, sizeof(buf),
            "{\n"
            "  \"description\": \"Golden upstream integration: real face.jpg + audio.wav\",\n"
            "  \"image_loader_time_ms\": \"%s\",\n"
            "  \"face_detect_time_ms\": %.2f,\n"
            "  \"face_align_time_ms\": %.2f,\n"
            "  \"audio_load_time_ms\": %.2f,\n"
            "  \"audio_preprocess_time_ms\": \"%s\",\n"
            "  \"audio_frame_time_ms\": \"%s\",\n"
            "  \"mel_extract_time_ms\": %.2f,\n"
            "  \"model_load_time_ms\": %.2f,\n"
            "  \"builder_time_ms\": %.2f,\n"
            "  \"adapter_time_ms\": %.2f,\n"
            "  \"model_generation\": %llu,\n"
            "  \"face_shape\": \"96x96 BGR CV_8UC3\",\n"
            "  \"mel_chunk_shape\": \"80x16 freq-major 1280 floats\",\n"
            "  \"mel_range\": [%.4f, %.4f],\n"
            "  \"face_input_range\": [%.4f, %.4f],\n"
            "  \"pred_shape\": \"%dx%dx%d\",\n"
            "  \"pred_sampled_chunk_count\": %zu,\n"
            "  \"pred_sampled_indices\": \"%s\",\n"
            "  \"pred_range\": [%.6f, %.6f],\n"
            "  \"pred_all_finite\": %s,\n"
            "  \"pred_all_within_expected_range\": %s,\n"
            "  \"inference_total_attempt_count\": %zu,\n"
            "  \"inference_max_attempt_count\": %zu,\n"
            "  \"inference_retry_recovered_count\": %zu,\n"
            "  \"inference_input_validation_ms_sum\": %.3f,\n"
            "  \"inference_first_attempt_ms_sum\": %.3f,\n"
            "  \"inference_retry_attempts_ms_sum\": %.3f,\n"
            "  \"inference_output_validation_ms_sum\": %.3f,\n"
            "  \"inference_total_latency_ms_sum\": %.3f,\n"
            "  \"inference_total_latency_ms_avg\": %.3f,\n"
            "  \"note\": \"Proves golden upstream connectivity; NOT visual quality\"\n"
            "}\n",
            "ImageLoader tracks time in info, not result struct",
            det_r.time_ms, align_r.time_ms,
            audio_r.time_ms,
            "timing in info.process_time_ms, zero under O(1) preprocessing",
            "streaming interface: per-chunk timing not aggregated here",
            mel_r.time_ms,
            load_r.time_ms, build_r.time_ms, adapt_r.time_ms,
            static_cast<unsigned long long>(
                first_inference_r.value.model_generation),
            mel_min, mel_max,
            build_r.info.face_min_value, build_r.info.face_max_value,
            first_inference_r.value.output_info.width,
            first_inference_r.value.output_info.height,
            first_inference_r.value.output_info.channels,
            sampled_chunk_indices.size(), sampled_indices_text.c_str(),
            sampled_pred_min, sampled_pred_max,
            sampled_pred_all_finite ? "true" : "false",
            sampled_pred_all_within_range ? "true" : "false",
            sampled_total_attempt_count,
            sampled_max_attempt_count,
            sampled_retry_recovered_count,
            sampled_input_validation_ms_sum,
            sampled_first_attempt_ms_sum,
            sampled_retry_attempts_ms_sum,
            sampled_output_validation_ms_sum,
            sampled_total_latency_ms_sum,
            sampled_total_latency_ms_avg);
        ASSERT_GT(json_size, 0);
        ASSERT_LT(static_cast<std::size_t>(json_size), sizeof(buf))
            << "golden JSON buffer is too small";
        WriteJson("golden_output/golden_upstream_integration_info.json", buf);
    }
}

// ============================================================================
// 失败路径 1：空图 → FaceDetector 返回精确状态码，不留下半成品
// ============================================================================
TEST(GoldenUpstreamIntegration, FaceDetectorRejectsEmptyImage) {
    core::FaceDetector detector;
    auto load_model_r = detector.LoadLandmarkModel(
        "models/shape_predictor_68_face_landmarks.dat");
    ASSERT_TRUE(load_model_r.success) << load_model_r.error_message;

    cv::Mat empty;
    auto det_r = detector.DetectAndLandmark(empty, core::FaceDetectOptions{});
    EXPECT_FALSE(det_r.success);
    EXPECT_EQ(det_r.status, core::FaceDetectStatus::kEmptyImage)
        << "actual status: "
        << core::FaceDetector::StatusToString(det_r.status);
    EXPECT_FALSE(det_r.error_message.empty());
    // 验证不留下半成品：faces 和 landmarks 应为空
    EXPECT_TRUE(det_r.detection.faces.empty());
    EXPECT_TRUE(det_r.landmarks.empty());
}

// ============================================================================
// 失败路径 2：Builder 拒绝错误尺寸/类型的人脸和错误长度的 Mel
// ============================================================================
TEST(GoldenUpstreamIntegration, BuilderRejectsInvalidInput) {
    std::vector<float> valid_mel(1280, 0.0f);

    model::Wav2LipInputBuilder builder;

    // 错误尺寸（95×96）：绝不 resize
    {
        cv::Mat wrong_size(95, 96, CV_8UC3, cv::Scalar(128));
        auto r = builder.Build(wrong_size, valid_mel);
        EXPECT_FALSE(r.success);
        EXPECT_EQ(r.status, model::ModelInputStatus::kInvalidFaceSize);
    }

    // 错误类型（灰度图 CV_8UC1）：绝不 convert
    {
        cv::Mat gray(96, 96, CV_8UC1, cv::Scalar(128));
        auto r = builder.Build(gray, valid_mel);
        EXPECT_FALSE(r.success);
        EXPECT_EQ(r.status, model::ModelInputStatus::kInvalidFaceType);
    }

    // 空人脸
    {
        cv::Mat empty;
        auto r = builder.Build(empty, valid_mel);
        EXPECT_FALSE(r.success);
        EXPECT_EQ(r.status, model::ModelInputStatus::kEmptyAlignedFace);
    }

    // 错误 Mel 长度
    {
        cv::Mat face(96, 96, CV_8UC3, cv::Scalar(128));
        std::vector<float> bad_mel(100);
        auto r = builder.Build(face, bad_mel);
        EXPECT_FALSE(r.success);
        EXPECT_EQ(r.status, model::ModelInputStatus::kInvalidMelChunkSize);
    }
}
