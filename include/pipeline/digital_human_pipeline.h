#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"

namespace digital_human {
namespace pipeline {

// ============================================================================
// PipelineConfig — Pipeline 集中配置
// ============================================================================

/// @brief Mel 尾部策略：短音频默认冻结为钳位或复制最后一帧
enum class MelTailPolicy {
    kClampOrReplicateLast  ///< T>=16 正常取；1<=T<16 复制最后一帧补足；T==0 报 kEmptyMelInput
};

/// @brief Pipeline 运行模式
enum class PipelineMode {
    kOffline,   ///< 离线模式：不使用 FrameScheduler，render worker 直接调用 sink
    kRealtime   ///< 实时模式：使用 FrameScheduler + AudioVideoSynchronous
};

/// @brief Pipeline 全部可配置项，所有字段都有默认值
struct PipelineConfig {
    // ---- 输入路径 ----
    std::filesystem::path image_path;               ///< 输入图片路径
    std::filesystem::path audio_path;               ///< 输入音频路径

    // ---- 模型路径 ----
    std::filesystem::path model_param_path;         ///< wav2lip.param 路径
    std::filesystem::path model_bin_path;            ///< 推导出；留空则自动推导
    std::filesystem::path landmark_model_path;       ///< dlib 68 点模型路径

    // ---- 视频参数 ----
    int fps_num = 25;                                ///< 帧率分子
    int fps_den = 1;                                 ///< 帧率分母

    // ---- 运行模式 ----
    PipelineMode mode = PipelineMode::kOffline;      ///< 离线/实时模式

    // ---- 队列容量 ----
    std::size_t q1_capacity = 4;                     ///< audio→inference 队列容量
    std::size_t q2_capacity = 2;                     ///< inference→render 队列容量

    // ---- 推理配置 ----
    std::size_t inference_microbatch = 1;            ///< 推理微批次大小
    std::size_t scheduler_worker_count = 1;           ///< scheduler worker 数
    int ncnn_threads = 1;                            ///< 每 worker 的 ncnn 线程数
    std::chrono::milliseconds inference_timeout{30000};///< 单次推理超时

    // ---- Mel 配置 ----
    MelTailPolicy tail_policy = MelTailPolicy::kClampOrReplicateLast; ///< 尾部策略

    // ---- 实时模式配置 ----
    std::size_t frame_scheduler_capacity = 4;        ///< FrameScheduler 队列容量
    std::size_t min_buffered_frames = 2;             ///< 启动播放前最小缓冲帧数
    std::int64_t sync_tolerance_us = 20000;          ///< 同步容忍窗口（微秒）

    // ---- 输出目录（离线模式） ----
    std::filesystem::path output_dir;                ///< 离线输出帧目录

    /// @brief 创建默认离线配置的工厂方法
    static PipelineConfig OfflineDefault() {
        PipelineConfig config;
        config.mode = PipelineMode::kOffline;
        return config;
    }

    /// @brief 创建默认实时配置的工厂方法
    static PipelineConfig RealtimeDefault() {
        PipelineConfig config;
        config.mode = PipelineMode::kRealtime;
        return config;
    }
};

// ============================================================================
// DigitalHumanPipeline — 数字人多线程 Pipeline
// ============================================================================

/// @brief 数字人 SDK 多线程 Pipeline 处理架构。
///
/// 线程模型：
/// 1. audio worker — 音频预处理、分帧、Mel、按视频帧率生成任务
/// 2. inference coordinator — 消费 AudioFeatureTask → InferenceScheduler → OutputProcessor
/// 3. render worker — FaceBlender → VideoFrame → sink
/// 4. InferenceScheduler worker — 执行实际 ncnn 推理
/// 5. 实时输出调度线程 — 仅实时模式创建
///
/// 同步准备阶段在工作线程启动前完成：
/// 1. 配置、路径和容量校验
/// 2. 模型加载和 ModelRuntimeSnapshot 固定
/// 3. 原图读取、人脸检测、PrepareFace、mask 和逆变换
/// 4. 音频读取和格式验证
/// 5. 实时模式准备 PortAudio（不立即开始播放）
///
/// 使用 PImpl 模式隐藏实现细节。禁止拷贝，允许移动。
class DigitalHumanPipeline {
public:
    DigitalHumanPipeline();
    ~DigitalHumanPipeline() noexcept;

    DigitalHumanPipeline(const DigitalHumanPipeline&) = delete;
    DigitalHumanPipeline& operator=(const DigitalHumanPipeline&) = delete;
    DigitalHumanPipeline(DigitalHumanPipeline&&) noexcept;
    DigitalHumanPipeline& operator=(DigitalHumanPipeline&&) noexcept;

    /// @brief 启动 Pipeline：同步准备 + 创建所有工作线程
    /// @param config Pipeline 配置
    /// @param sink 输出接收器（shared_ptr，生命周期至少到 OnTerminal 返回）
    /// @return PipelineResult，成功时 state=kRunning
    PipelineResult Start(const PipelineConfig& config,
                         SinkHandle sink);

    /// @brief 等待 Pipeline 进入终态（阻塞直到 Succeeded/Failed/Cancelled）
    /// @return PipelineResult，含终态状态和完整统计
    PipelineResult Wait();

    /// @brief 请求停止 Pipeline（幂等）
    /// @return PipelineResult，含当前状态
    PipelineResult RequestStop();

    /// @brief 停止并等待（等价于 RequestStop + Wait）
    /// @return PipelineResult，含终态状态和完整统计
    PipelineResult Stop();

    /// @brief 获取当前 Pipeline 状态
    PipelineState GetState() const;

    /// @brief 获取当前统计快照
    PipelineStats GetStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;

    /// @brief join 所有工作线程（按 audio→inference→render→scheduler 顺序）
    void JoinAllThreads();

    /// @brief 检查并更新终止状态（由工作线程退出时调用）
    void CheckTerminalAndNotify();

    /// @brief 将 PipelineStats 包装为 PipelineResult 返回值
    PipelineResult StatsToResult() const;
};

}  // namespace pipeline
}  // namespace digital_human
