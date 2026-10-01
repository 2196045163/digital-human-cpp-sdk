#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "model/input_processor.h"
#include "model/output_processor.h"
#include "video/video_frame.h"

namespace digital_human {
namespace pipeline {

// ============================================================================
// PreparedFaceContext — 预处理阶段不可变人脸上下文
// ============================================================================

/// @brief 预处理阶段一次性准备好的人脸上下文，发布后不得原地修改其中的 cv::Mat。
/// @note  通过 shared_ptr<const PreparedFaceContext> 在任务间共享，避免深拷贝。
struct PreparedFaceContext {
    cv::Mat source_bgr;                       ///< 原始 BGR 图（CV_8UC3，原图尺寸）
    cv::Mat prepared_face_bgr;                ///< 96×96 CV_8UC3 BGR [0,255] 模型输入脸
    cv::Mat mask;                             ///< 96×96 alpha mask（CV_32FC1，0~1）
    cv::Rect source_crop_rect;                ///< 原图中实际使用的裁剪框
    cv::Mat transform;                        ///< 原图到 96×96 的 2×3 矩阵
    cv::Mat inverse_transform;                ///< 96×96 回到原图的 2×3 矩阵
    std::vector<cv::Point2f> landmarks_96;    ///< 与 prepared_face_bgr 同坐标系的 68 点
    cv::Size original_size;                   ///< 原图尺寸（用于回贴）
    std::vector<cv::Point> original_landmarks;///< 原图坐标系的 68 点（用于 mask 生成）

    /// @brief 验证上下文是否完整可用
    bool IsValid() const {
        return !source_bgr.empty()
            && !prepared_face_bgr.empty()
            && !mask.empty()
            && prepared_face_bgr.cols == 96
            && prepared_face_bgr.rows == 96
            && prepared_face_bgr.type() == CV_8UC3
            && original_size.width > 0
            && original_size.height > 0;
    }
};

// ============================================================================
// AudioFeatureTask — 音频 worker 产出、inference coordinator 消费的任务
// ============================================================================

/// @brief 音频特征任务：携带单帧所需的完整 Mel 和人脸上下文。
/// @note  move-only，包含 1280 float freq-major Mel 和只读人脸上下文。
struct AudioFeatureTask {
    std::int64_t task_id = 0;                            ///< 全局唯一任务 ID
    std::int64_t frame_index = 0;                        ///< 视频帧序号（0-based）
    std::int64_t pts_us = 0;                             ///< 微秒 PTS
    std::int64_t mel_start = 0;                          ///< Mel 时间帧起始索引
    std::vector<float> mel_chunk;                        ///< 1280 float freq-major Mel
    std::shared_ptr<const PreparedFaceContext> face_ctx; ///< 只读人脸上下文

    AudioFeatureTask() = default;

    // move-only
    AudioFeatureTask(AudioFeatureTask&&) = default;
    AudioFeatureTask& operator=(AudioFeatureTask&&) = default;
    AudioFeatureTask(const AudioFeatureTask&) = delete;
    AudioFeatureTask& operator=(const AudioFeatureTask&) = delete;
};

// ============================================================================
// InferenceFrameTask — inference coordinator 产出、render worker 消费的任务
// ============================================================================

/// @brief 推理帧任务：携带已转换的模型输出和人脸上下文，供 render worker 生成最终帧。
struct InferenceFrameTask {
    std::int64_t task_id = 0;                            ///< 全局唯一任务 ID
    std::int64_t frame_index = 0;                        ///< 视频帧序号
    std::int64_t pts_us = 0;                             ///< 微秒 PTS
    std::shared_ptr<const PreparedFaceContext> face_ctx; ///< 只读人脸上下文
    model::ProcessedModelOutput processed_output;        ///< 96×96 CV_8UC3 BGR 模型输出
    std::size_t attempt_count = 1;                       ///< 推理尝试次数
    std::uint64_t model_generation = 0;                  ///< 模型代次

    InferenceFrameTask() = default;

    // 队列任务保持 move-only
    InferenceFrameTask(InferenceFrameTask&&) = default;
    InferenceFrameTask& operator=(InferenceFrameTask&&) = default;
    InferenceFrameTask(const InferenceFrameTask&) = delete;
    InferenceFrameTask& operator=(const InferenceFrameTask&) = delete;
};

// ============================================================================
// DeliveryKind — 帧交付类型
// ============================================================================

/// @brief 帧交付类型：区分唯一帧和重复帧
enum class DeliveryKind {
    kUnique,    ///< 正常生成的唯一帧
    kRepeated   ///< 因实时调度重复显示的帧
};

// ============================================================================
// ScheduleAction — 调度动作（实时模式透传 FrameScheduler）
// ============================================================================

/// @brief 调度动作：实时模式下从 FrameScheduler 透传的动作类型
enum class PipelineScheduleAction {
    kDeliver,               ///< 正常交付
    kDropAndDeliver,        ///< 丢弃过期帧后交付
    kRepeatLast,            ///< 重复上一帧
    kBufferingNoFrame,      ///< 缓冲中无帧可交付
    kDroppedLate,           ///< 帧因过期被丢弃
    kDroppedOverflow        ///< 帧因队列满被丢弃
};

// ============================================================================
// PipelineFrame — sink 回调接收的帧
// ============================================================================

/// @brief Pipeline 输出帧：除 VideoFrame 外包含交付类型、任务 ID、调度动作等。
struct PipelineFrame {
    video::VideoFrame video_frame;                     ///< 最终视频帧（CV_8UC3 BGR，原图尺寸）
    DeliveryKind delivery_kind = DeliveryKind::kUnique;///< 交付类型
    std::int64_t source_task_id = 0;                   ///< 来源任务 ID
    PipelineScheduleAction schedule_action = PipelineScheduleAction::kDeliver; ///< 调度动作
    std::int64_t audio_reference_pts_us = 0;           ///< 实时模式的音频参考 PTS
};

// ============================================================================
// PipelineResult — Start/Wait/Stop 返回的统一结果
// ============================================================================

/// @brief Pipeline 控制状态
enum class PipelineState {
    kIdle,        ///< 尚未启动
    kStarting,    ///< 正在启动（同步准备 + 创建线程）
    kRunning,     ///< 正常运行
    kDraining,    ///< 正常 EOS 排空中
    kSucceeded,   ///< 正常完成
    kFailed,      ///< 内部错误
    kStopping,    ///< 正在停止
    kCancelled    ///< 已取消
};

/// @brief Pipeline 终止原因：第一个终止原因获胜
enum class PipelineTermination {
    kNone,        ///< 未终止
    kNormalEos,   ///< 正常 EOS
    kUserCancel,  ///< 用户主动取消
    kInternalError///< 内部错误
};

/// @brief Pipeline 错误码
enum class PipelineErrorCode {
    kOk,
    kInvalidConfig,           ///< 配置无效
    kImageLoadFailed,         ///< 图片加载失败
    kFaceDetectFailed,        ///< 人脸检测失败
    kFacePrepareFailed,       ///< 人脸准备失败
    kMaskGenerateFailed,      ///< mask 生成失败
    kAudioLoadFailed,         ///< 音频加载失败
    kAudioProcessFailed,      ///< 音频处理失败
    kMelExtractFailed,        ///< Mel 提取失败（含 kEmptyMelInput）
    kInferenceFailed,         ///< 推理失败
    kOutputProcessFailed,     ///< 输出处理失败
    kFaceBlendFailed,         ///< 人脸融合失败
    kQueueError,              ///< 队列错误
    kThreadCreateFailed,      ///< 线程创建失败
    kSinkCallbackError,       ///< sink 回调异常
    kInternalError            ///< 内部未知错误
};

/// @brief Pipeline 统计信息
struct PipelineStats {
    // 生命周期
    PipelineState state = PipelineState::kIdle;
    PipelineTermination termination = PipelineTermination::kNone;
    bool cleanup_complete = false;
    bool input_eos = false;
    bool user_cancelled = false;
    bool internal_error = false;

    // 任务计数（worker 线程在 stats_mutex 下写入；GetStats/StatsToResult 在 stats_mutex 下读取）
    std::int64_t generated_task_count = 0;       ///< audio worker 生成的任务总数
    std::int64_t scheduler_accepted_count = 0;   ///< scheduler 接受的任务数
    std::int64_t scheduler_failed_count = 0;     ///< scheduler 失败的任务数
    std::int64_t rendered_unique_frame_count = 0;///< 成功渲染的唯一帧数
    std::int64_t unique_delivered_count = 0;     ///< 交付的唯一帧数
    std::int64_t repeated_delivery_count = 0;    ///< 重复交付的帧数
    std::int64_t dropped_late_count = 0;         ///< 因过期丢弃的帧数
    std::int64_t dropped_overflow_count = 0;     ///< 因队列满丢弃的帧数
    std::int64_t cancelled_discarded_count = 0;  ///< 因取消丢弃的等待任务数
    std::int64_t sink_callback_count = 0;        ///< sink 回调总次数

    // 队列统计
    std::size_t q1_high_watermark = 0;           ///< audio→inference 队列峰值深度
    std::size_t q2_high_watermark = 0;           ///< inference→render 队列峰值深度
    std::size_t scheduler_pending_at_terminal = 0;///< 终止时 scheduler 待处理任务数

    // 时间统计
    double total_wall_time_ms = 0.0;             ///< Start 到 sink OnTerminal 完成的总耗时
    double prepare_time_ms = 0.0;                ///< 同步准备耗时
    double audio_process_time_ms = 0.0;          ///< 音频处理总耗时
    double inference_total_time_ms = 0.0;        ///< 模型前向推理累计耗时（GPU 为同步 CUDA forward）
    double h2d_time_ms = 0.0;                    ///< LibTorch CPU tensor→CUDA 累计耗时
    double cuda_forward_time_ms = 0.0;           ///< LibTorch 同步 CUDA 前向累计耗时
    double d2h_time_ms = 0.0;                    ///< LibTorch CUDA output→CPU 累计耗时
    double gpu_backend_total_time_ms = 0.0;      ///< LibTorch backend 输入到 CPU output 可读取累计耗时
    double gpu_peak_memory_mb = 0.0;             ///< LibTorch allocator 单帧峰值的全任务最大值
    double gpu_inference_memory_delta_mb = 0.0;  ///< NVML 单帧推理 used peak-baseline 的全任务最大值
    double render_total_time_ms = 0.0;           ///< 渲染总耗时

    // 错误信息
    PipelineErrorCode first_error_code = PipelineErrorCode::kOk;
    std::string first_error_message;
    std::string secondary_diagnostics;           ///< 次要诊断信息（清理阶段的额外错误）
    std::string terminal_diagnostics;            ///< CheckTerminalAndNotify 诊断（为何未交付）
};

/// @brief Pipeline 统一返回结果
struct PipelineResult {
    bool success = false;
    PipelineState terminal_state = PipelineState::kIdle;
    PipelineErrorCode error_code = PipelineErrorCode::kOk;
    std::string error_message;
    PipelineStats stats;
};

/// @brief 将状态转为可读字符串
inline std::string PipelineStateToString(PipelineState state) {
    switch (state) {
        case PipelineState::kIdle:       return "Idle";
        case PipelineState::kStarting:   return "Starting";
        case PipelineState::kRunning:    return "Running";
        case PipelineState::kDraining:   return "Draining";
        case PipelineState::kSucceeded:  return "Succeeded";
        case PipelineState::kFailed:     return "Failed";
        case PipelineState::kStopping:   return "Stopping";
        case PipelineState::kCancelled:  return "Cancelled";
        default:                         return "Unknown";
    }
}

inline std::string PipelineErrorCodeToString(PipelineErrorCode code) {
    switch (code) {
        case PipelineErrorCode::kOk:                  return "OK";
        case PipelineErrorCode::kInvalidConfig:        return "InvalidConfig";
        case PipelineErrorCode::kImageLoadFailed:      return "ImageLoadFailed";
        case PipelineErrorCode::kFaceDetectFailed:     return "FaceDetectFailed";
        case PipelineErrorCode::kFacePrepareFailed:    return "FacePrepareFailed";
        case PipelineErrorCode::kMaskGenerateFailed:   return "MaskGenerateFailed";
        case PipelineErrorCode::kAudioLoadFailed:      return "AudioLoadFailed";
        case PipelineErrorCode::kAudioProcessFailed:   return "AudioProcessFailed";
        case PipelineErrorCode::kMelExtractFailed:     return "MelExtractFailed";
        case PipelineErrorCode::kInferenceFailed:      return "InferenceFailed";
        case PipelineErrorCode::kOutputProcessFailed:  return "OutputProcessFailed";
        case PipelineErrorCode::kFaceBlendFailed:      return "FaceBlendFailed";
        case PipelineErrorCode::kQueueError:           return "QueueError";
        case PipelineErrorCode::kThreadCreateFailed:   return "ThreadCreateFailed";
        case PipelineErrorCode::kSinkCallbackError:    return "SinkCallbackError";
        case PipelineErrorCode::kInternalError:        return "InternalError";
        default:                                        return "Unknown";
    }
}

}  // namespace pipeline
}  // namespace digital_human
