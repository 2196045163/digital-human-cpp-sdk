#include "pipeline/digital_human_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "audio/audio_framer.h"
#include "audio/audio_loader.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_preprocessor.h"
#include "core/face_blender.h"
#include "core/face_detector.h"
#include "core/face_mask_generator.h"
#include "core/image_loader.h"
#include "core/timestamp_manager.h"
#include "model/inference_scheduler.h"
#include "model/input_processor.h"
#include "model/model_inference.h"
#include "model/model_loader.h"
#include "model/ncnn_input_adapter.h"
#include "model/output_processor.h"
#include "video/video_frame.h"

#include "detail/bounded_task_queue.h"
#include "detail/pipeline_components.h"

namespace digital_human {
namespace pipeline {
// ============================================================================
// 前向声明（在 digital_human::pipeline 命名空间中）
// ============================================================================

void AudioWorkerLoop(detail::SharedState& shared);
void InferenceCoordinatorLoop(detail::SharedState& shared);
void RenderWorkerLoop(detail::SharedState& shared);

// ============================================================================
// Pipeline::Impl
// ============================================================================

struct DigitalHumanPipeline::Impl {
    detail::SharedState shared_;

    // 生命周期同步
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
    bool terminal_notified_ = false;

    // Pipeline 实例指针（供静态线程函数回调 CheckTerminalAndNotify）
    DigitalHumanPipeline* owner_ = nullptr;
};

// ============================================================================
// 构造/析构/移动
// ============================================================================

DigitalHumanPipeline::DigitalHumanPipeline()
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->owner_ = this;
}

DigitalHumanPipeline::~DigitalHumanPipeline() noexcept {
    if (!pImpl_) { return; }
    try {
        // 析构等价于 RequestStop + 完整 join
        RequestStop();
        JoinAllThreads();
        if (pImpl_->shared_.scheduler) {
            pImpl_->shared_.scheduler->Stop();
        }
    } catch (...) {
        // 析构 noexcept，吞掉所有异常
    }
}

DigitalHumanPipeline::DigitalHumanPipeline(DigitalHumanPipeline&& other) noexcept
    : pImpl_(std::move(other.pImpl_)) {
    if (pImpl_) {
        pImpl_->owner_ = this;
    }
}

DigitalHumanPipeline& DigitalHumanPipeline::operator=(DigitalHumanPipeline&& other) noexcept {
    if (this != &other) {
        // 停止旧状态
        if (pImpl_) {
            RequestStop();
            JoinAllThreads();
        }
        pImpl_ = std::move(other.pImpl_);
        if (pImpl_) {
            pImpl_->owner_ = this;
        }
    }
    return *this;
}

// ============================================================================
// 内部辅助：状态转移
// ============================================================================

namespace {

/// @brief 设置状态（无条件），通知等待者
void SetSharedState(detail::SharedState& shared, PipelineState state) {
    std::lock_guard<std::mutex> lock(shared.state_mutex);
    shared.state = state;
    shared.stats.state = state;
}

/// @brief 获取当前状态（需要可变引用以加锁）
PipelineState GetSharedState(detail::SharedState& shared) {
    std::lock_guard<std::mutex> lock(shared.state_mutex);
    return shared.state;
}

/// @brief 确定终止原因（第一个获胜）
PipelineTermination DetermineTermination(const detail::SharedState& shared) {
    if (shared.internal_error.load(std::memory_order_acquire)) {
        return PipelineTermination::kInternalError;
    }
    if (shared.user_cancelled.load(std::memory_order_acquire)) {
        return PipelineTermination::kUserCancel;
    }
    if (shared.input_eos.load(std::memory_order_acquire)) {
        return PipelineTermination::kNormalEos;
    }
    return PipelineTermination::kNone;
}

/// @brief 根据终止原因确定终态
PipelineState TerminalStateForReason(PipelineTermination term) {
    switch (term) {
        case PipelineTermination::kNormalEos:    return PipelineState::kSucceeded;
        case PipelineTermination::kUserCancel:   return PipelineState::kCancelled;
        case PipelineTermination::kInternalError: return PipelineState::kFailed;
        default:                                  return PipelineState::kFailed;
    }
}

/// @brief 同步准备阶段：加载所有资源，在工作线程启动前完成
PipelineResult DoPrepare(detail::SharedState& shared, const PipelineConfig& config) {
    PipelineResult result;
    const auto prepare_start = std::chrono::steady_clock::now();
    auto prepare_ctx = std::make_shared<detail::PrepareContext>();

    // ---- 1. 配置基础校验 ----
    if (config.image_path.empty() || config.audio_path.empty()
        || config.model_param_path.empty() || config.landmark_model_path.empty()) {
        result.error_code = PipelineErrorCode::kInvalidConfig;
        result.error_message = "配置无效：图片路径、音频路径、模型路径、关键点模型路径均不能为空";
        return result;
    }
    if (config.fps_num <= 0 || config.fps_den <= 0) {
        result.error_code = PipelineErrorCode::kInvalidConfig;
        result.error_message = "配置无效：帧率参数必须大于 0";
        return result;
    }

    // ---- 2. 模型加载 ----
    model::ModelLoader model_loader;
    model::ModelLoadOptions load_opts;
    load_opts.num_threads = config.ncnn_threads;
    auto load_result = model_loader.Load(config.model_param_path, load_opts);
    if (!load_result.success) {
        result.error_code = PipelineErrorCode::kInvalidConfig;
        result.error_message = "模型加载失败：" + load_result.error_message;
        return result;
    }
    prepare_ctx->model_snapshot = model_loader.AcquireSnapshot();

    // ---- 3. 图片加载 ----
    core::ImageLoader image_loader;
    auto img_result = image_loader.LoadFromFile(config.image_path.string());
    if (!img_result.success) {
        result.error_code = PipelineErrorCode::kImageLoadFailed;
        result.error_message = "图片加载失败：" + img_result.error_message;
        return result;
    }
    cv::Mat source_bgr = img_result.image;
    if (source_bgr.type() != CV_8UC3) {
        result.error_code = PipelineErrorCode::kImageLoadFailed;
        result.error_message = "图片格式不符合预期：需要 CV_8UC3 BGR";
        return result;
    }

    // ---- 4. 人脸检测和关键点 ----
    core::FaceDetector face_detector;
    auto landmark_load = face_detector.LoadLandmarkModel(config.landmark_model_path.string());
    if (!landmark_load.success) {
        result.error_code = PipelineErrorCode::kFaceDetectFailed;
        result.error_message = "人脸关键点模型加载失败：" + landmark_load.error_message;
        return result;
    }
    auto analyze_result = face_detector.DetectAndLandmark(source_bgr);
    if (!analyze_result.success) {
        result.error_code = PipelineErrorCode::kFaceDetectFailed;
        result.error_message = "人脸检测失败：" + analyze_result.error_message;
        return result;
    }
    if (analyze_result.landmarks.empty()) {
        result.error_code = PipelineErrorCode::kFaceDetectFailed;
        result.error_message = "未检测到人脸";
        return result;
    }

    const auto& best_landmark = analyze_result.landmarks[0];
    if (best_landmark.landmarks.size() != 68) {
        result.error_code = PipelineErrorCode::kFaceDetectFailed;
        result.error_message = "关键点数量不符合 68 点契约";
        return result;
    }

    // ---- 5. PrepareFace（使用 Wav2LipInputBuilder） ----
    model::Wav2LipInputBuilder input_builder;
    auto face_prepare = input_builder.PrepareFace(
        source_bgr,
        best_landmark.face_rect,
        best_landmark.landmarks);
    if (!face_prepare.success) {
        result.error_code = PipelineErrorCode::kFacePrepareFailed;
        result.error_message = "PrepareFace 失败：" + face_prepare.error_message;
        return result;
    }

    // ---- 6. 生成 96×96 mask ----
    core::FaceMaskGenerator mask_gen;
    auto mask_result = mask_gen.GenerateAlignedMouthMask(
        cv::Size(96, 96),
        face_prepare.value.landmarks_96);
    if (!mask_result.success) {
        result.error_code = PipelineErrorCode::kMaskGenerateFailed;
        result.error_message = "Mask 生成失败：" + mask_result.error_message;
        return result;
    }

    // ---- 7. 构建不可变人脸上下文 ----
    auto face_ctx = std::make_shared<PreparedFaceContext>();
    face_ctx->source_bgr = source_bgr;
    face_ctx->prepared_face_bgr = face_prepare.value.face_bgr;
    face_ctx->mask = mask_result.alpha_mask;
    face_ctx->source_crop_rect = face_prepare.value.source_crop_rect;
    face_ctx->transform = face_prepare.value.transform;
    face_ctx->inverse_transform = face_prepare.value.inverse_transform;
    face_ctx->landmarks_96 = face_prepare.value.landmarks_96;
    face_ctx->original_size = source_bgr.size();
    face_ctx->original_landmarks = best_landmark.landmarks;
    prepare_ctx->face_ctx = face_ctx;

    // ---- 8. 音频加载 ----
    audio::AudioLoader audio_loader;
    auto audio_result = audio_loader.LoadFromFile(config.audio_path.string());
    if (!audio_result.success) {
        result.error_code = PipelineErrorCode::kAudioLoadFailed;
        result.error_message = "音频加载失败：" + audio_result.error_message;
        return result;
    }

    // ---- 9. 音频预处理 ----
    audio::AudioPreprocessor preprocessor;
    auto preprocess_result = preprocessor.Process(audio_result.audio.pcm);
    if (!preprocess_result.success) {
        result.error_code = PipelineErrorCode::kAudioProcessFailed;
        result.error_message = "音频预处理失败：" + preprocess_result.error_message;
        return result;
    }
    prepare_ctx->audio_pcm = std::move(preprocess_result.pcm);
    prepare_ctx->audio_sample_rate = audio_result.audio.sample_rate;
    prepare_ctx->audio_sample_count = static_cast<std::int64_t>(prepare_ctx->audio_pcm.size());
    prepare_ctx->audio_duration_sec = static_cast<double>(prepare_ctx->audio_sample_count)
                                      / static_cast<double>(prepare_ctx->audio_sample_rate);

    // ---- 10. 音频分帧 + Mel 提取 ----
    audio::AudioFramer framer(audio::AudioFramer::Wav2LipDefault());
    auto frame_result = framer.Frame(prepare_ctx->audio_pcm);
    if (!frame_result.success) {
        result.error_code = PipelineErrorCode::kAudioProcessFailed;
        result.error_message = "音频分帧失败：" + frame_result.error_message;
        return result;
    }

    // 提取每帧的 PCM 数据
    std::vector<std::vector<float>> frame_samples;
    frame_samples.reserve(frame_result.frames.size());
    for (const auto& frame : frame_result.frames) {
        frame_samples.push_back(frame.samples);
    }

    audio::MelFeatureExtractor mel_extractor(audio::MelFeatureExtractor::Wav2LipDefault());
    auto mel_result = mel_extractor.ExtractBatch(frame_samples);
    if (!mel_result.success) {
        result.error_code = PipelineErrorCode::kMelExtractFailed;
        result.error_message = "Mel 提取失败：" + mel_result.error_message;
        return result;
    }
    prepare_ctx->mel_spectrogram = mel_result.mel;
    prepare_ctx->mel_frame_count = mel_result.mel.rows;

    // ---- 11. 计算帧数 ----
    std::int64_t frame_count = detail::ComputeFrameCount(
        prepare_ctx->audio_sample_count,
        prepare_ctx->audio_sample_rate,
        config.fps_num, config.fps_den);

    shared.frame_count = frame_count;

    // ---- 12. 创建 InferenceScheduler ----
    model::SchedulerConfig sched_config;
    sched_config.worker_count = config.scheduler_worker_count;
    sched_config.queue_capacity = config.scheduler_worker_count * 2;
    sched_config.queue_wait_timeout = config.inference_timeout;
    sched_config.inference_options.light_mode = true;
    sched_config.allow_thread_oversubscription = false;

    auto scheduler = std::make_shared<model::InferenceScheduler>();
    auto sched_start = scheduler->Start(prepare_ctx->model_snapshot, sched_config);
    if (!sched_start.success) {
        result.error_code = PipelineErrorCode::kInternalError;
        result.error_message = "InferenceScheduler 启动失败：" + sched_start.error_message;
        return result;
    }
    shared.scheduler = scheduler;
    shared.prepare_ctx = prepare_ctx;

    auto prepare_end = std::chrono::steady_clock::now();
    shared.stats.prepare_time_ms = std::chrono::duration<double, std::milli>(
        prepare_end - prepare_start).count();

    result.success = true;
    result.stats = shared.stats;
    return result;
}

}  // namespace (anonymous)

// ============================================================================
// 工作线程入口（在同一匿名命名空间中）
// ============================================================================

/// @brief Audio Worker：音频预处理 → 分帧 → Mel → 生成 AudioFeatureTask → 提交 Q1
void AudioWorkerLoop(detail::SharedState& shared) {
    try {
        const auto& config = shared.config;
        const auto& prepare = *shared.prepare_ctx;
        const std::int64_t frame_count = shared.frame_count;

        for (std::int64_t i = 0; i < frame_count; ++i) {
            // 检查取消或错误
            if (shared.user_cancelled.load(std::memory_order_acquire)
                || shared.internal_error.load(std::memory_order_acquire)) {
                break;
            }

            // 计算 PTS（微秒）
            auto pts_result = core::TimestampManager::FromVideoFrameIndex(
                i, core::FrameRate{config.fps_num, config.fps_den});
            if (!pts_result.success) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kInternalError,
                    "PTS 计算失败：" + pts_result.error_message,
                    i, "audio_worker");
                shared.internal_error.store(true, std::memory_order_release);
                break;
            }

            // 计算 Mel 起始帧（整数运算）
            std::int64_t mel_start = detail::ComputeMelStart(
                i, config.fps_num, config.fps_den, 80);

            // 从 Mel 频谱取 16 帧 chunk（kClampOrReplicateLast 策略）
            std::vector<float> mel_chunk;
            bool chunk_ok = detail::BuildWav2LipChunkAt(
                prepare.mel_spectrogram,
                mel_start,
                prepare.mel_frame_count,
                mel_chunk);
            if (!chunk_ok) {
                // T == 0 → kEmptyMelInput → Pipeline Failed
                shared.first_error.TryRecord(
                    PipelineErrorCode::kMelExtractFailed,
                    "Mel chunk 提取失败：kEmptyMelInput（T==0）",
                    i, "audio_worker");
                shared.internal_error.store(true, std::memory_order_release);
                break;
            }

            // 构建 AudioFeatureTask
            AudioFeatureTask task;
            task.task_id = shared.next_task_id.fetch_add(1, std::memory_order_relaxed);
            task.frame_index = i;
            task.pts_us = pts_result.value.microseconds;
            task.mel_start = mel_start;
            task.mel_chunk = std::move(mel_chunk);
            task.face_ctx = prepare.face_ctx;

            // 提交到 Q1
            if (!shared.q1->Push(std::move(task))) {
                // 队列已关闭或取消
                break;
            }

            {
                std::lock_guard<std::mutex> lk(shared.stats_mutex);
                shared.stats.generated_task_count++;
            }
        }

        // 正常 EOS：关闭 Q1
        shared.q1->Close();
        shared.input_eos.store(true, std::memory_order_release);
    } catch (const std::exception& e) {
        shared.first_error.TryRecord(
            PipelineErrorCode::kInternalError,
            std::string("Audio worker 异常：") + e.what(),
            -1, "audio_worker");
        shared.internal_error.store(true, std::memory_order_release);
        if (shared.q1) { shared.q1->Cancel(); }
    } catch (...) {
        shared.first_error.TryRecord(
            PipelineErrorCode::kInternalError,
            "Audio worker 未知异常",
            -1, "audio_worker");
        shared.internal_error.store(true, std::memory_order_release);
        if (shared.q1) { shared.q1->Cancel(); }
    }
}

/// @brief Inference Coordinator：消费 Q1 → 构建输入 → 调用 InferenceScheduler → 提交 Q2
void InferenceCoordinatorLoop(detail::SharedState& shared) {
    try {
        model::NcnnInputAdapter ncnn_adapter;

        while (true) {
            // 检查取消或错误
            if (shared.user_cancelled.load(std::memory_order_acquire)
                || shared.internal_error.load(std::memory_order_acquire)) {
                break;
            }

            auto task_opt = shared.q1->Pop();
            if (!task_opt.has_value()) {
                // Q1 已关闭且排空 → 正常 EOS
                break;
            }

            auto& task = *task_opt;

            // 构建模型语义输入（Wav2LipInputBuilder）
            model::Wav2LipInputBuilder input_builder;
            auto build_result = input_builder.Build(
                task.face_ctx->prepared_face_bgr,
                task.mel_chunk,
                model::ModelInputMetadata{
                    std::optional<std::int64_t>(task.pts_us / 1000),
                    std::optional<std::int64_t>(task.frame_index)
                });
            if (!build_result.success) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kInferenceFailed,
                    "模型输入构建失败：" + build_result.error_message,
                    task.task_id, "inference_coordinator");
                shared.internal_error.store(true, std::memory_order_release);
                if (shared.q1) { shared.q1->Cancel(); }
                if (shared.q2) { shared.q2->Cancel(); }
                break;
            }

            // 适配为 ncnn 格式
            auto adapt_result = ncnn_adapter.Adapt(build_result.data);
            if (!adapt_result.success) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kInferenceFailed,
                    "ncnn 适配失败：" + adapt_result.error_message,
                    task.task_id, "inference_coordinator");
                shared.internal_error.store(true, std::memory_order_release);
                if (shared.q1) { shared.q1->Cancel(); }
                if (shared.q2) { shared.q2->Cancel(); }
                break;
            }

            // 调用 InferenceScheduler（microbatch=1 时单样本）
            std::vector<model::NcnnWav2LipInput> batch_inputs;
            batch_inputs.push_back(std::move(adapt_result.input));

            auto batch_result = shared.scheduler->InferBatch(batch_inputs);
            {
                std::lock_guard<std::mutex> lk(shared.stats_mutex);
                shared.stats.scheduler_accepted_count += batch_result.summary.accepted_count;
                shared.stats.scheduler_failed_count += batch_result.summary.failure_count;
            }

            // 检查 batch 结果：任一失败 → Pipeline 失败
            if (!batch_result.success || batch_result.results.empty()) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kInferenceFailed,
                    "推理 batch 失败：" + batch_result.error_message,
                    task.task_id, "inference_coordinator");
                shared.internal_error.store(true, std::memory_order_release);
                if (shared.q1) { shared.q1->Cancel(); }
                if (shared.q2) { shared.q2->Cancel(); }
                break;
            }

            const auto& single_result = batch_result.results[0];
            if (!single_result.success) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kInferenceFailed,
                    "推理失败：" + single_result.error_message,
                    task.task_id, "inference_coordinator");
                shared.internal_error.store(true, std::memory_order_release);
                if (shared.q1) { shared.q1->Cancel(); }
                if (shared.q2) { shared.q2->Cancel(); }
                break;
            }

            // 构建 InferenceFrameTask
            InferenceFrameTask frame_task;
            frame_task.task_id = task.task_id;
            frame_task.frame_index = task.frame_index;
            frame_task.pts_us = task.pts_us;
            frame_task.face_ctx = task.face_ctx;
            frame_task.inference_output = std::move(single_result.value);
            frame_task.attempt_count = single_result.value.attempts.attempt_count;
            frame_task.model_generation = batch_result.model_generation;

            // 提交到 Q2
            if (!shared.q2->Push(std::move(frame_task))) {
                break;
            }
        }

        // 排空 Q1 完成 → 关闭 Q2
        if (shared.q2) {
            shared.q2->Close();
        }
    } catch (const std::exception& e) {
        shared.first_error.TryRecord(
            PipelineErrorCode::kInferenceFailed,
            std::string("Inference coordinator 异常：") + e.what(),
            -1, "inference_coordinator");
        shared.internal_error.store(true, std::memory_order_release);
        if (shared.q1) { shared.q1->Cancel(); }
        if (shared.q2) { shared.q2->Cancel(); }
    } catch (...) {
        shared.first_error.TryRecord(
            PipelineErrorCode::kInferenceFailed,
            "Inference coordinator 未知异常",
            -1, "inference_coordinator");
        shared.internal_error.store(true, std::memory_order_release);
        if (shared.q1) { shared.q1->Cancel(); }
        if (shared.q2) { shared.q2->Cancel(); }
    }
}

/// @brief Render Worker：消费 Q2 → OutputProcessor → FaceBlender → VideoFrame → sink
void RenderWorkerLoop(detail::SharedState& shared) {
    try {
        model::OutputProcessor output_processor;
        core::FaceBlender face_blender;

        while (true) {
            // 检查取消或错误
            if (shared.user_cancelled.load(std::memory_order_acquire)
                || shared.internal_error.load(std::memory_order_acquire)) {
                break;
            }

            auto task_opt = shared.q2->Pop();
            if (!task_opt.has_value()) {
                // Q2 已关闭且排空 → 正常 EOS
                break;
            }

            auto& task = *task_opt;

            // 步骤 1：OutputProcessor — pred ncnn::Mat → CV_8UC3 96×96 BGR
            auto convert_result = output_processor.Convert(task.inference_output);
            if (!convert_result.success) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kOutputProcessFailed,
                    "输出处理失败：" + convert_result.error_message,
                    task.task_id, "render_worker");
                shared.internal_error.store(true, std::memory_order_release);
                if (shared.q2) { shared.q2->Cancel(); }
                break;
            }

            // 步骤 2：FaceBlender — 96×96 BGR + mask → 原图尺寸 BGR
            auto blend_result = face_blender.BlendMouthToOriginal(
                task.face_ctx->source_bgr,
                convert_result.value.generated_face_bgr,
                task.face_ctx->mask,
                task.face_ctx->inverse_transform);
            if (!blend_result.success) {
                shared.first_error.TryRecord(
                    PipelineErrorCode::kFaceBlendFailed,
                    "人脸融合失败：" + blend_result.error_message,
                    task.task_id, "render_worker");
                shared.internal_error.store(true, std::memory_order_release);
                if (shared.q2) { shared.q2->Cancel(); }
                break;
            }

            // 步骤 3：构建 VideoFrame
            video::VideoFrame video_frame;
            video_frame.frame_bgr = blend_result.final_bgr;
            video_frame.pts = core::MediaTimestamp{task.pts_us};
            video_frame.frame_index = task.frame_index;

            // 步骤 4：构建 PipelineFrame
            PipelineFrame pipeline_frame;
            pipeline_frame.video_frame = std::move(video_frame);
            pipeline_frame.delivery_kind = DeliveryKind::kUnique;
            pipeline_frame.source_task_id = task.task_id;
            pipeline_frame.schedule_action = PipelineScheduleAction::kDeliver;

            // 步骤 5：离线模式直接从 render worker 调用 sink
            auto sink = shared.sink;
            if (sink) {
                try {
                    sink->OnFrame(pipeline_frame);
                } catch (const std::exception& e) {
                    shared.first_error.TryRecord(
                        PipelineErrorCode::kSinkCallbackError,
                        std::string("OnFrame 回调异常：") + e.what(),
                        task.task_id, "render_worker");
                    shared.internal_error.store(true, std::memory_order_release);
                    if (shared.q2) { shared.q2->Cancel(); }
                    break;
                } catch (...) {
                    shared.first_error.TryRecord(
                        PipelineErrorCode::kSinkCallbackError,
                        "OnFrame 回调未知异常",
                        task.task_id, "render_worker");
                    shared.internal_error.store(true, std::memory_order_release);
                    if (shared.q2) { shared.q2->Cancel(); }
                    break;
                }
            }

            {
                std::lock_guard<std::mutex> lk(shared.stats_mutex);
                shared.stats.rendered_unique_frame_count++;
                shared.stats.unique_delivered_count++;
                shared.stats.sink_callback_count++;
            }
        }
    } catch (const std::exception& e) {
        shared.first_error.TryRecord(
            PipelineErrorCode::kInternalError,
            std::string("Render worker 异常：") + e.what(),
            -1, "render_worker");
        shared.internal_error.store(true, std::memory_order_release);
        if (shared.q2) { shared.q2->Cancel(); }
    } catch (...) {
        shared.first_error.TryRecord(
            PipelineErrorCode::kInternalError,
            "Render worker 未知异常",
            -1, "render_worker");
        shared.internal_error.store(true, std::memory_order_release);
        if (shared.q2) { shared.q2->Cancel(); }
    }
}

// ============================================================================
// StatsToResult — 将内部统计包装为 PipelineResult
// ============================================================================

PipelineResult DigitalHumanPipeline::StatsToResult() const {
    PipelineResult result;
    PipelineState current;
    {
        std::lock_guard<std::mutex> state_lock(pImpl_->shared_.state_mutex);
        current = pImpl_->shared_.state;
        result.stats = pImpl_->shared_.stats;
        result.stats.state = current;
        result.stats.termination = pImpl_->shared_.stats.termination;
    }

    result.success = (current == PipelineState::kSucceeded);
    result.terminal_state = current;

    if (pImpl_->shared_.q1) {
        result.stats.q1_high_watermark = pImpl_->shared_.q1->HighWatermark();
    }
    if (pImpl_->shared_.q2) {
        result.stats.q2_high_watermark = pImpl_->shared_.q2->HighWatermark();
    }

    if (pImpl_->shared_.first_error.recorded.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> err_lock(pImpl_->shared_.first_error.mutex);
        result.error_code = pImpl_->shared_.first_error.code;
        result.error_message = pImpl_->shared_.first_error.message;
    }
    if (result.error_code == PipelineErrorCode::kOk
        && pImpl_->shared_.stats.first_error_code != PipelineErrorCode::kOk) {
        result.error_code = pImpl_->shared_.stats.first_error_code;
        result.error_message = pImpl_->shared_.stats.first_error_message;
    }
    return result;
}

// ============================================================================
// CheckTerminalAndNotify — 检查是否达到终态
// ============================================================================

void DigitalHumanPipeline::CheckTerminalAndNotify() {
    auto& shared = pImpl_->shared_;

    // 幂等：OnTerminal 只能调用一次。使用 CAS 确保只有一个线程进入交付路径。
    bool expected = false;
    if (!shared.terminal_delivered.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        // 另一个线程已经在交付或已交付
        return;
    }

    auto term = DetermineTermination(shared);
    if (term == PipelineTermination::kNone) {
        // 尚未达到终态，重置标志以便后续线程尝试
        shared.terminal_delivered.store(false, std::memory_order_release);
        return;
    }

    PipelineState current = GetSharedState(shared);

    if (current == PipelineState::kRunning && shared.input_eos.load(std::memory_order_acquire)) {
        SetSharedState(shared, PipelineState::kDraining);
        current = PipelineState::kDraining;
    }

    bool q2_drained = shared.q2
        && (shared.q2->IsClosed() || shared.q2->IsCancelled())
        && shared.q2->Empty();

    // P0-1 修复：内部错误发生时，若 Q2 已排空，转入 Failed 终态。
    // 否则工作线程异常退出后，Pipeline 状态仍为 kRunning，Wait() 永久阻塞。
    if (shared.internal_error.load(std::memory_order_acquire) && q2_drained) {
        if (current != PipelineState::kFailed && current != PipelineState::kStopping) {
            SetSharedState(shared, PipelineState::kFailed);
            current = PipelineState::kFailed;
        }
    }

    bool is_terminal = (current == PipelineState::kDraining && q2_drained)
                       || current == PipelineState::kStopping
                       || current == PipelineState::kFailed;

    if (!is_terminal) {
        // 记录为何未交付（用于诊断）
        {
            std::lock_guard<std::mutex> stats_lock(shared.stats_mutex);
            shared.stats.terminal_diagnostics =
                std::string("not_terminal: current=") + PipelineStateToString(current)
                + " q2_drained=" + (q2_drained ? "true" : "false")
                + " term=" + (term == PipelineTermination::kNormalEos ? "EOS" :
                              term == PipelineTermination::kUserCancel ? "Cancel" :
                              term == PipelineTermination::kInternalError ? "Error" : "None");
            shared.terminal_delivered.store(false, std::memory_order_release);
        }
        return;
    }

    PipelineState final_state = TerminalStateForReason(term);

    // 回调 OnTerminal（在锁外）
    auto sink = shared.sink;
    if (sink) {
        // 在 stats_mutex 下取一致性快照，OnTerminal 调用期间不持有任何锁
        PipelineStats stats_snapshot;
        {
            std::lock_guard<std::mutex> stats_lock(shared.stats_mutex);
            stats_snapshot = shared.stats;
        }

        PipelineResult term_result;
        term_result.success = (final_state == PipelineState::kSucceeded);
        term_result.terminal_state = final_state;
        term_result.stats = stats_snapshot;
        term_result.stats.state = final_state;
        term_result.stats.termination = term;

        if (shared.q1) {
            term_result.stats.q1_high_watermark = shared.q1->HighWatermark();
        }
        if (shared.q2) {
            term_result.stats.q2_high_watermark = shared.q2->HighWatermark();
        }

        if (shared.first_error.recorded.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> err_lock(shared.first_error.mutex);
            term_result.error_code = shared.first_error.code;
            term_result.error_message = shared.first_error.message;
        }

        try {
            sink->OnTerminal(term_result);
        } catch (...) {
            std::lock_guard<std::mutex> stats_lock(shared.stats_mutex);
            shared.stats.secondary_diagnostics = "OnTerminal 回调异常（已忽略）";
        }
    }

    {
        std::lock_guard<std::mutex> stats_lock(shared.stats_mutex);
        shared.stats.state = final_state;
        shared.stats.termination = term;
    }
    shared.cleanup_complete.store(true, std::memory_order_release);
    SetSharedState(shared, final_state);

    {
        std::lock_guard<std::mutex> lock(pImpl_->wait_mutex_);
        pImpl_->terminal_notified_ = true;
    }
    pImpl_->wait_cv_.notify_all();
}

// ============================================================================
// Start — 同步准备 + 创建线程
// ============================================================================

PipelineResult DigitalHumanPipeline::Start(const PipelineConfig& config,
                                            SinkHandle sink) {
    std::lock_guard<std::mutex> lifecycle_lock(pImpl_->shared_.lifecycle_mutex);

    PipelineResult result;

    // P1-1 修复：实时模式尚未实现，显式拒绝
    if (config.mode == PipelineMode::kRealtime) {
        result.error_code = PipelineErrorCode::kInvalidConfig;
        result.error_message = "实时模式尚未实现，当前仅支持 PipelineMode::kOffline";
        result.stats.state = PipelineState::kIdle;
        return result;
    }

    // 状态检查：只能从 Idle 启动
    {
        PipelineState current = GetSharedState(pImpl_->shared_);
        if (current != PipelineState::kIdle) {
            result.error_code = PipelineErrorCode::kInvalidConfig;
            result.error_message = "Pipeline 不处于 Idle 状态，当前："
                                   + PipelineStateToString(current);
            result.stats.state = current;
            return result;
        }
    }

    // 保存配置和 sink
    pImpl_->shared_.config = config;
    pImpl_->shared_.sink = std::move(sink);
    pImpl_->shared_.stats = PipelineStats{};

    // 进入 Starting 状态
    SetSharedState(pImpl_->shared_, PipelineState::kStarting);

    // 同步准备
    auto prepare_result = DoPrepare(pImpl_->shared_, config);
    if (!prepare_result.success) {
        SetSharedState(pImpl_->shared_, PipelineState::kFailed);
        pImpl_->shared_.stats = prepare_result.stats;
        pImpl_->shared_.stats.state = PipelineState::kFailed;
        pImpl_->shared_.stats.termination = PipelineTermination::kInternalError;
        pImpl_->shared_.stats.first_error_code = prepare_result.error_code;
        pImpl_->shared_.stats.first_error_message = prepare_result.error_message;
        pImpl_->shared_.cleanup_complete.store(true, std::memory_order_release);

        // 通知等待者
        {
            std::lock_guard<std::mutex> lock(pImpl_->wait_mutex_);
            pImpl_->terminal_notified_ = true;
        }
        pImpl_->wait_cv_.notify_all();

        return StatsToResult();
    }

    pImpl_->shared_.stats = prepare_result.stats;

    // 检查 Starting 期间是否收到停止
    if (pImpl_->shared_.user_cancelled.load(std::memory_order_acquire)) {
        SetSharedState(pImpl_->shared_, PipelineState::kCancelled);
        pImpl_->shared_.stats.state = PipelineState::kCancelled;
        pImpl_->shared_.stats.termination = PipelineTermination::kUserCancel;
        pImpl_->shared_.cleanup_complete.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(pImpl_->wait_mutex_);
            pImpl_->terminal_notified_ = true;
        }
        pImpl_->wait_cv_.notify_all();
        return StatsToResult();
    }

    // 创建阶段队列
    pImpl_->shared_.q1 = std::make_unique<detail::BoundedTaskQueue<AudioFeatureTask>>(
        config.q1_capacity);
    pImpl_->shared_.q2 = std::make_unique<detail::BoundedTaskQueue<InferenceFrameTask>>(
        config.q2_capacity);

    // 重置计数器和标志位
    pImpl_->shared_.next_task_id.store(0, std::memory_order_relaxed);
    pImpl_->shared_.input_eos.store(false, std::memory_order_release);
    pImpl_->shared_.user_cancelled.store(false, std::memory_order_release);
    pImpl_->shared_.internal_error.store(false, std::memory_order_release);
    pImpl_->shared_.cleanup_complete.store(false, std::memory_order_release);
    pImpl_->shared_.first_error.recorded.store(false, std::memory_order_release);
    pImpl_->terminal_notified_ = false;

    // 创建工作线程，每个线程退出时递减计数器
    pImpl_->shared_.running_workers.store(3, std::memory_order_release);

    // 创建线程（下游先启动，上游后启动）
    try {
        // render worker 先启动（消费者先准备好）
        // 注意：fetch_sub 必须在 CheckTerminalAndNotify 之前，确保 Wait() 的
        // running_workers==0 兜底能在 OnTerminal 被阻塞时仍触发终态交付。
        pImpl_->shared_.render_worker = std::thread([this]() {
            RenderWorkerLoop(pImpl_->shared_);
            pImpl_->shared_.running_workers.fetch_sub(1, std::memory_order_release);
            this->CheckTerminalAndNotify();
        });

        // inference coordinator — 只递减计数器，不参与终态判断（防 OnTerminal/OnFrame 并发）
        pImpl_->shared_.inference_coordinator = std::thread([this]() {
            InferenceCoordinatorLoop(pImpl_->shared_);
            pImpl_->shared_.running_workers.fetch_sub(1, std::memory_order_release);
        });

        // audio worker — 只递减计数器，不参与终态判断（防 OnTerminal/OnFrame 并发）
        pImpl_->shared_.audio_worker = std::thread([this]() {
            AudioWorkerLoop(pImpl_->shared_);
            pImpl_->shared_.running_workers.fetch_sub(1, std::memory_order_release);
        });
    } catch (const std::exception& e) {
        // 线程创建失败 → Failed
        pImpl_->shared_.first_error.TryRecord(
            PipelineErrorCode::kThreadCreateFailed,
            std::string("线程创建失败：") + e.what());
        pImpl_->shared_.internal_error.store(true, std::memory_order_release);

        // 取消队列，唤醒所有等待线程
        if (pImpl_->shared_.q1) { pImpl_->shared_.q1->Cancel(); }
        if (pImpl_->shared_.q2) { pImpl_->shared_.q2->Cancel(); }

        // join 已创建的线程
        JoinAllThreads();

        // 停止 scheduler
        if (pImpl_->shared_.scheduler) {
            pImpl_->shared_.scheduler->Stop();
        }

        SetSharedState(pImpl_->shared_, PipelineState::kFailed);
        pImpl_->shared_.stats.state = PipelineState::kFailed;
        pImpl_->shared_.stats.termination = PipelineTermination::kInternalError;
        pImpl_->shared_.stats.first_error_code = PipelineErrorCode::kThreadCreateFailed;
        pImpl_->shared_.stats.first_error_message = "线程创建失败：" + std::string(e.what());
        pImpl_->shared_.cleanup_complete.store(true, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lock(pImpl_->wait_mutex_);
            pImpl_->terminal_notified_ = true;
        }
        pImpl_->wait_cv_.notify_all();

        return StatsToResult();
    }

    // 全部线程创建成功 → Running
    SetSharedState(pImpl_->shared_, PipelineState::kRunning);

    result.success = true;
    result.terminal_state = PipelineState::kRunning;
    result.stats = pImpl_->shared_.stats;
    result.stats.state = PipelineState::kRunning;
    return result;
}

// ============================================================================
// Wait — 等待 Pipeline 进入终态
// ============================================================================

PipelineResult DigitalHumanPipeline::Wait() {
    if (!pImpl_) { return PipelineResult{}; }

    // 在工作线程运行期间周期性检查终态，防止 CheckTerminalAndNotify 遗漏
    const auto wait_start = std::chrono::steady_clock::now();
    constexpr auto kMaxWait = std::chrono::seconds(300);
    bool force_delivered = false;

    while (true) {
        // 硬超时兜底：超过最大等待时间后强制交付
        if (std::chrono::steady_clock::now() - wait_start > kMaxWait) {
            break;
        }
        CheckTerminalAndNotify();

        {
            std::unique_lock<std::mutex> lock(pImpl_->wait_mutex_);
            if (pImpl_->terminal_notified_
                || pImpl_->shared_.cleanup_complete.load(std::memory_order_acquire)) {
                break;
            }
            // 每 500ms 重试一次，避免忙轮询
            pImpl_->wait_cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
                return pImpl_->terminal_notified_
                    || pImpl_->shared_.cleanup_complete.load(std::memory_order_acquire);
            });
            if (pImpl_->terminal_notified_
                || pImpl_->shared_.cleanup_complete.load(std::memory_order_acquire)) {
                break;
            }
        }

        // 双保险：如果所有工作线程函数已退出，跳出循环强制交付
        if (pImpl_->shared_.running_workers.load(std::memory_order_acquire) == 0) {
            break;
        }
    }

    // 最终交付：工作线程已全部退出（running_workers == 0），绕过 is_terminal 检查
    // 直接确定终止原因并调用 OnTerminal。这是 running_workers 兜底的正确语义——
    // 所有生产者/消费者已退出，Pipeline 必然已到达终态。
    {
        auto& shared = pImpl_->shared_;

        bool expected = false;
        if (shared.terminal_delivered.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel, std::memory_order_acquire)) {

            auto term = DetermineTermination(shared);
            if (term == PipelineTermination::kNone) {
                // 无明确终止原因时，按当前状态推断
                PipelineState current = GetSharedState(shared);
                if (current == PipelineState::kDraining) {
                    term = PipelineTermination::kNormalEos;
                } else if (current == PipelineState::kStopping) {
                    term = PipelineTermination::kUserCancel;
                } else {
                    term = PipelineTermination::kInternalError;
                }
            }

            PipelineState final_state = TerminalStateForReason(term);

            auto sink = shared.sink;
            if (sink) {
                PipelineResult term_result;
                term_result.success = (final_state == PipelineState::kSucceeded);
                term_result.terminal_state = final_state;
                term_result.stats = shared.stats;
                term_result.stats.state = final_state;
                term_result.stats.termination = term;

                if (shared.q1) { term_result.stats.q1_high_watermark = shared.q1->HighWatermark(); }
                if (shared.q2) { term_result.stats.q2_high_watermark = shared.q2->HighWatermark(); }

                if (shared.first_error.recorded.load(std::memory_order_acquire)) {
                    std::lock_guard<std::mutex> err_lock(shared.first_error.mutex);
                    term_result.error_code = shared.first_error.code;
                    term_result.error_message = shared.first_error.message;
                }

                try {
                    sink->OnTerminal(term_result);
                } catch (...) {
                    shared.stats.secondary_diagnostics = "OnTerminal 回调异常（force-deliver 路径，已忽略）";
                }
            }

            shared.stats.state = final_state;
            shared.stats.termination = term;
            shared.cleanup_complete.store(true, std::memory_order_release);
            SetSharedState(shared, final_state);
        }

        // 通知所有等待者
        {
            std::lock_guard<std::mutex> lock(pImpl_->wait_mutex_);
            pImpl_->terminal_notified_ = true;
        }
        pImpl_->wait_cv_.notify_all();
    }

    // join 所有线程确保完全退出
    JoinAllThreads();

    // 停止 scheduler
    if (pImpl_->shared_.scheduler) {
        pImpl_->shared_.scheduler->Stop();
    }

    PipelineResult result;
    {
        std::lock_guard<std::mutex> state_lock(pImpl_->shared_.state_mutex);
        PipelineState current = pImpl_->shared_.state;
        result.success = (current == PipelineState::kSucceeded);
        result.terminal_state = current;
        result.stats = pImpl_->shared_.stats;
        result.stats.state = current;
        result.stats.termination = pImpl_->shared_.stats.termination;

        if (pImpl_->shared_.q1) {
            result.stats.q1_high_watermark = pImpl_->shared_.q1->HighWatermark();
        }
        if (pImpl_->shared_.q2) {
            result.stats.q2_high_watermark = pImpl_->shared_.q2->HighWatermark();
        }

        if (pImpl_->shared_.first_error.recorded.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> err_lock(pImpl_->shared_.first_error.mutex);
            result.error_code = pImpl_->shared_.first_error.code;
            result.error_message = pImpl_->shared_.first_error.message;
        }
    }

    return result;
}

// ============================================================================
// RequestStop — 请求停止（幂等）
// ============================================================================

PipelineResult DigitalHumanPipeline::RequestStop() {
    if (!pImpl_) { return PipelineResult{}; }
    // 幂等设置取消标志
    bool already = pImpl_->shared_.user_cancelled.exchange(true, std::memory_order_acq_rel);
    if (already) {
        return StatsToResult();
    }

    // 记录取消原因
    pImpl_->shared_.first_error.TryRecord(
        PipelineErrorCode::kOk,
        "用户主动取消");

    // 设置状态为 Stopping（如果当前是 Running/Draining/Starting）
    {
        std::lock_guard<std::mutex> state_lock(pImpl_->shared_.state_mutex);
        auto current = pImpl_->shared_.state;
        if (current == PipelineState::kRunning
            || current == PipelineState::kDraining
            || current == PipelineState::kStarting) {
            pImpl_->shared_.state = PipelineState::kStopping;
            pImpl_->shared_.stats.state = PipelineState::kStopping;
        }
    }

    // 取消两条阶段队列，唤醒所有等待线程
    if (pImpl_->shared_.q1) {
        pImpl_->shared_.q1->Cancel();
    }
    if (pImpl_->shared_.q2) {
        pImpl_->shared_.q2->Cancel();
    }

    // 统计被取消的未处理任务
    {
        std::lock_guard<std::mutex> stats_lock(pImpl_->shared_.stats_mutex);
        if (pImpl_->shared_.q1) {
            pImpl_->shared_.stats.cancelled_discarded_count +=
                static_cast<std::int64_t>(pImpl_->shared_.q1->UnprocessedCount());
        }
        if (pImpl_->shared_.q2) {
            pImpl_->shared_.stats.cancelled_discarded_count +=
                static_cast<std::int64_t>(pImpl_->shared_.q2->UnprocessedCount());
        }
    }

    // 通知等待者
    {
        std::lock_guard<std::mutex> lock(pImpl_->wait_mutex_);
        pImpl_->terminal_notified_ = true;
    }
    pImpl_->wait_cv_.notify_all();

    return StatsToResult();
}

// ============================================================================
// Stop — RequestStop + join
// ============================================================================

PipelineResult DigitalHumanPipeline::Stop() {
    if (!pImpl_) { return PipelineResult{}; }
    auto stop_start = std::chrono::steady_clock::now();

    RequestStop();

    // 确保 OnTerminal 交付（工作线程可能在 RequestStop 取消队列后已退出，
    // 但 CheckTerminalAndNotify 可能因 is_terminal 判断与取消之间的竞态而遗漏）
    CheckTerminalAndNotify();

    // join 所有线程
    JoinAllThreads();

    // 停止 scheduler
    if (pImpl_->shared_.scheduler) {
        pImpl_->shared_.scheduler->Stop();
    }

    pImpl_->shared_.cleanup_complete.store(true, std::memory_order_release);

    auto stop_end = std::chrono::steady_clock::now();
    pImpl_->shared_.stats.total_wall_time_ms = std::chrono::duration<double, std::milli>(
        stop_end - stop_start).count();

    PipelineResult result;
    {
        std::lock_guard<std::mutex> state_lock(pImpl_->shared_.state_mutex);
        PipelineState current = pImpl_->shared_.state;

        // 如果是 Stopping，转为 Cancelled
        if (current == PipelineState::kStopping) {
            pImpl_->shared_.state = PipelineState::kCancelled;
            current = PipelineState::kCancelled;
        }

        result.success = (current == PipelineState::kSucceeded);
        result.terminal_state = current;
        result.stats = pImpl_->shared_.stats;
        result.stats.state = current;

        if (pImpl_->shared_.q1) {
            result.stats.q1_high_watermark = pImpl_->shared_.q1->HighWatermark();
        }
        if (pImpl_->shared_.q2) {
            result.stats.q2_high_watermark = pImpl_->shared_.q2->HighWatermark();
        }

        if (pImpl_->shared_.first_error.recorded.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> err_lock(pImpl_->shared_.first_error.mutex);
            result.error_code = pImpl_->shared_.first_error.code;
            result.error_message = pImpl_->shared_.first_error.message;
        }
    }

    return result;
}

// ============================================================================
// JoinAllThreads — join 所有工作线程
// ============================================================================

void DigitalHumanPipeline::JoinAllThreads() {
    // join 顺序：audio → inference → render → output scheduler
    // 不能持锁 join
    if (pImpl_->shared_.audio_worker.joinable()) {
        pImpl_->shared_.audio_worker.join();
    }
    if (pImpl_->shared_.inference_coordinator.joinable()) {
        pImpl_->shared_.inference_coordinator.join();
    }
    if (pImpl_->shared_.render_worker.joinable()) {
        pImpl_->shared_.render_worker.join();
    }
    if (pImpl_->shared_.output_scheduler_thread.joinable()) {
        pImpl_->shared_.output_scheduler_thread.join();
    }
}

// ============================================================================
// GetState / GetStats
// ============================================================================

PipelineState DigitalHumanPipeline::GetState() const {
    if (!pImpl_) { return PipelineState::kIdle; }
    return GetSharedState(pImpl_->shared_);
}

PipelineStats DigitalHumanPipeline::GetStats() const {
    if (!pImpl_) { return PipelineStats{}; }
    // 分别持锁读取状态和统计，避免 stats_mutex 和 state_mutex 交叉死锁
    PipelineStats stats;
    {
        std::lock_guard<std::mutex> lock(pImpl_->shared_.stats_mutex);
        stats = pImpl_->shared_.stats;
    }
    {
        std::lock_guard<std::mutex> lock(pImpl_->shared_.state_mutex);
        stats.state = pImpl_->shared_.state;
    }
    // 从原子标志同步
    stats.input_eos = pImpl_->shared_.input_eos.load(std::memory_order_acquire);
    stats.internal_error = pImpl_->shared_.internal_error.load(std::memory_order_acquire);
    stats.user_cancelled = pImpl_->shared_.user_cancelled.load(std::memory_order_acquire);
    stats.cleanup_complete = pImpl_->shared_.cleanup_complete.load(std::memory_order_acquire);

    if (pImpl_->shared_.q1) {
        stats.q1_high_watermark = pImpl_->shared_.q1->HighWatermark();
    }
    if (pImpl_->shared_.q2) {
        stats.q2_high_watermark = pImpl_->shared_.q2->HighWatermark();
    }

    return stats;
}

}  // namespace pipeline
}  // namespace digital_human
