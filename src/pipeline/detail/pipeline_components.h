#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio/audio_framer.h"
#include "audio/audio_loader.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_preprocessor.h"
#include "core/face_blender.h"
#include "core/face_mask_generator.h"
#include "core/timestamp_manager.h"
#include "model/inference_scheduler.h"
#include "model/input_processor.h"
#include "model/model_inference.h"
#include "model/model_loader.h"
#include "model/ncnn_input_adapter.h"
#include "model/output_processor.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"
#include "video/video_frame.h"

#include "bounded_task_queue.h"

namespace digital_human {
namespace pipeline {
namespace detail {

// ============================================================================
// 内部错误记录
// ============================================================================

/// @brief 第一个错误获胜模式下的错误记录
struct FirstError {
    std::atomic<bool> recorded{false};
    std::mutex mutex;
    PipelineErrorCode code = PipelineErrorCode::kOk;
    std::string message;
    std::int64_t task_id = -1;
    std::string stage;

    /// @brief 尝试记录第一个错误（只有第一次成功）
    bool TryRecord(PipelineErrorCode c, std::string msg,
                   std::int64_t tid = -1, std::string stg = "") {
        bool expected = false;
        if (!recorded.compare_exchange_strong(expected, true)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        code = c;
        message = std::move(msg);
        task_id = tid;
        stage = std::move(stg);
        return true;
    }
};

// ============================================================================
// 同步准备阶段产物
// ============================================================================

/// @brief 同步准备阶段产出的所有只读上下文
struct PrepareContext {
    std::shared_ptr<const PreparedFaceContext> face_ctx;
    std::vector<float> audio_pcm;              ///< 预处理后的 PCM
    int audio_sample_rate = 0;
    std::int64_t audio_sample_count = 0;
    double audio_duration_sec = 0.0;
    cv::Mat mel_spectrogram;                   ///< [T_mel, 80] Mel 频谱图
    int mel_frame_count = 0;                   ///< Mel 总时间帧数
    model::ModelRuntimeSnapshot model_snapshot;///< 模型快照
    core::TimestampManager timestamp_manager;  ///< （仅作命名空间引用）
};

// ============================================================================
// Pipeline 共享状态
// ============================================================================

/// @brief Pipeline 内部共享状态，受 state_mutex 保护
struct SharedState {
    // ---- 控制状态 ----
    PipelineState state = PipelineState::kIdle;
    PipelineTermination termination = PipelineTermination::kNone;
    std::atomic<bool> input_eos{false};
    std::atomic<bool> user_cancelled{false};
    std::atomic<bool> internal_error{false};
    std::atomic<bool> cleanup_complete{false};
    std::atomic<bool> terminal_delivered{false};  ///< 防止 OnTerminal 重复调用
    std::atomic<int> running_workers{0};          ///< 尚未退出的工作线程数

    // ---- 错误记录 ----
    FirstError first_error;

    // ---- Sink 引用 ----
    SinkHandle sink;

    // ---- 配置快照 ----
    PipelineConfig config;

    // ---- 准备阶段上下文 ----
    std::shared_ptr<const PrepareContext> prepare_ctx;

    // ---- 推理调度器 ----
    std::shared_ptr<model::InferenceScheduler> scheduler;

    // ---- 阶段队列 ----
    std::unique_ptr<BoundedTaskQueue<AudioFeatureTask>> q1;
    std::unique_ptr<BoundedTaskQueue<InferenceFrameTask>> q2;

    // ---- 统计 ----
    PipelineStats stats;

    // ---- 帧计数器 ----
    std::atomic<std::int64_t> next_task_id{0};
    std::atomic<std::int64_t> total_frames{0};
    std::int64_t frame_count = 0;

    // ---- 同步原语 ----
    std::mutex state_mutex;
    std::condition_variable state_cv;
    std::mutex lifecycle_mutex;  ///< 保护 Start/Stop/Wait 互斥
    mutable std::mutex stats_mutex;  ///< 保护 PipelineStats 读写（#3 数据竞争修复）

    // ---- 线程句柄 ----
    std::thread audio_worker;
    std::thread inference_coordinator;
    std::thread render_worker;
    std::thread output_scheduler_thread;  ///< 仅实时模式

    // ---- 实时模式组件（由实现管理） ----
    bool realtime_components_created = false;
};

// ============================================================================
// Mel 辅助函数
// ============================================================================

/// @brief 根据视频帧索引计算 Mel 起始帧位置
/// @note  使用整数运算避免浮点漂移
inline std::int64_t ComputeMelStart(std::int64_t frame_index,
                                     int fps_num, int fps_den,
                                     int mel_frames_per_sec = 80) {
    // mel_start(i) = floor(i × mel_frames_per_sec × fps_den / fps_num)
    // 25 fps 时：0, 3, 6, 9, 12, 16...
    return (frame_index * mel_frames_per_sec * fps_den) / fps_num;
}

/// @brief 从音频采样数和帧率计算视频帧数
inline std::int64_t ComputeFrameCount(std::int64_t audio_sample_count,
                                       int sample_rate,
                                       int fps_num, int fps_den) {
    // frame_count = ceil(audio_sample_count × fps_num / (sample_rate × fps_den))
    std::int64_t num = audio_sample_count * fps_num;
    std::int64_t den = static_cast<std::int64_t>(sample_rate) * fps_den;
    return (num + den - 1) / den;
}

/// @brief 从 Mel 频谱中按 clamp-or-replicate 策略取 16 帧 chunk
/// @param mel_spectrogram [T, 80] 形状的 Mel 频谱图
/// @param mel_start 起始帧索引
/// @param mel_frame_count Mel 总时间帧数
/// @param out_chunk 输出的 1280 float chunk
/// @return 成功时返回 true；T==0 时返回 false（kEmptyMelInput）
inline bool BuildWav2LipChunkAt(const cv::Mat& mel_spectrogram,
                                 std::int64_t mel_start,
                                 std::int64_t mel_frame_count,
                                 std::vector<float>& out_chunk) {
    constexpr int kMelBins = 80;
    constexpr int kMelFrames = 16;
    constexpr int kChunkSize = kMelBins * kMelFrames;  // 1280

    out_chunk.resize(kChunkSize, 0.0f);

    if (mel_frame_count == 0) {
        // T == 0：kEmptyMelInput
        return false;
    }

    const int mel_bins = mel_spectrogram.cols;
    if (mel_bins != kMelBins) {
        return false;
    }

    if (mel_frame_count >= kMelFrames) {
        // 正常情况：T >= 16，取 [mel_start, mel_start+16)
        std::int64_t start = mel_start;
        if (start < 0) {
            start = 0;
        }
        if (start + kMelFrames > mel_frame_count) {
            // 尾部越界：夹到 T-16
            start = mel_frame_count - kMelFrames;
            if (start < 0) {
                start = 0;
            }
        }

        for (int t = 0; t < kMelFrames; ++t) {
            const float* row = mel_spectrogram.ptr<float>(static_cast<int>(start + t));
            for (int f = 0; f < kMelBins; ++f) {
                // freq-major: chunk[f * kMelFrames + t] = row[f]
                out_chunk[f * kMelFrames + t] = row[f];
            }
        }
        return true;
    }

    // 1 <= T < 16：复制最后一个有效 Mel 时间帧，补足到 16 帧
    // 先填满已有的帧
    for (int t = 0; t < static_cast<int>(mel_frame_count); ++t) {
        const float* row = mel_spectrogram.ptr<float>(t);
        for (int f = 0; f < kMelBins; ++f) {
            out_chunk[f * kMelFrames + t] = row[f];
        }
    }
    // 复制最后一帧补足
    const float* last_row = mel_spectrogram.ptr<float>(static_cast<int>(mel_frame_count - 1));
    for (int t = static_cast<int>(mel_frame_count); t < kMelFrames; ++t) {
        for (int f = 0; f < kMelBins; ++f) {
            out_chunk[f * kMelFrames + t] = last_row[f];
        }
    }
    return true;
}

}  // namespace detail
}  // namespace pipeline
}  // namespace digital_human
