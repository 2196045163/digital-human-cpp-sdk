#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "audio/audio_loader.h"
#include "pipeline/pipeline_output_sink.h"

namespace digital_human {
namespace output {

// ============================================================================
// WriterError — 写出器专用错误码
// ============================================================================

/// @brief 最终媒体写出器错误码，每个失败点对应一个明确的枚举值。
enum class WriterError {
    kOk = 0,
    kAlreadyClosed,          ///< 已关闭，拒绝操作
    kNotOpen,                ///< 尚未打开（OnFrame 在 open 前调用）
    kInvalidConfig,          ///< 配置无效（路径为空 / 尺寸非法 / fps 非法）
    kOutputPathInvalid,      ///< 输出路径无法创建
    kFormatContextFailed,    ///< avformat_alloc_output_context2 失败
    kStreamCreationFailed,   ///< 创建视频或音频流失败
    kEncoderOpenFailed,      ///< 打开编码器失败
    kCodecContextAllocFailed,///< avcodec_alloc_context3 失败
    kFrameAllocFailed,       ///< av_frame_alloc 失败
    kPacketAllocFailed,      ///< av_packet_alloc 失败
    kSwsContextFailed,       ///< sws_getContext 失败
    kSwrContextFailed,       ///< swr_alloc_set_opts / swr_init 失败
    kHeaderWriteFailed,      ///< avformat_write_header 失败
    kFrameConvertFailed,     ///< BGR→YUV 转换失败
    kVideoEncodeFailed,      ///< 视频编码失败
    kAudioEncodeFailed,      ///< 音频编码失败
    kInterleavedWriteFailed, ///< av_interleaved_write_frame 失败
    kTrailerWriteFailed,     ///< av_write_trailer 失败
    kFlushFailed,            ///< 编码器 flush 失败
    kInvalidFrameDimensions, ///< 帧尺寸无效（0 或与首帧不一致）
    kPtsOverflow,            ///< PTS 溢出
    kInternalError,          ///< 内部未知错误
};

/// @brief 将错误码转为可读字符串
inline std::string WriterErrorToString(WriterError e) {
    switch (e) {
        case WriterError::kOk:                     return "OK";
        case WriterError::kAlreadyClosed:          return "AlreadyClosed";
        case WriterError::kNotOpen:                return "NotOpen";
        case WriterError::kInvalidConfig:          return "InvalidConfig";
        case WriterError::kOutputPathInvalid:      return "OutputPathInvalid";
        case WriterError::kFormatContextFailed:     return "FormatContextFailed";
        case WriterError::kStreamCreationFailed:    return "StreamCreationFailed";
        case WriterError::kEncoderOpenFailed:       return "EncoderOpenFailed";
        case WriterError::kCodecContextAllocFailed: return "CodecContextAllocFailed";
        case WriterError::kFrameAllocFailed:        return "FrameAllocFailed";
        case WriterError::kPacketAllocFailed:       return "PacketAllocFailed";
        case WriterError::kSwsContextFailed:        return "SwsContextFailed";
        case WriterError::kSwrContextFailed:        return "SwrContextFailed";
        case WriterError::kHeaderWriteFailed:       return "HeaderWriteFailed";
        case WriterError::kFrameConvertFailed:      return "FrameConvertFailed";
        case WriterError::kVideoEncodeFailed:       return "VideoEncodeFailed";
        case WriterError::kAudioEncodeFailed:       return "AudioEncodeFailed";
        case WriterError::kInterleavedWriteFailed:  return "InterleavedWriteFailed";
        case WriterError::kTrailerWriteFailed:      return "TrailerWriteFailed";
        case WriterError::kFlushFailed:             return "FlushFailed";
        case WriterError::kInvalidFrameDimensions:  return "InvalidFrameDimensions";
        case WriterError::kPtsOverflow:             return "PtsOverflow";
        case WriterError::kInternalError:           return "InternalError";
        default:                                     return "Unknown";
    }
}

// ============================================================================
// WriterConfig — 写出器配置
// ============================================================================

/// @brief 最终媒体写出器配置。
///
/// 所有字段都有默认值，调用方只需设置关心的部分。
/// audio 字段可选：若 pcm 为空则生成纯视频 MP4。
struct WriterConfig {
    /// @brief 输出 MP4 文件路径（必填）
    std::string output_path;

    /// @brief 视频帧率分子/分母（默认 25/1 = 25 fps）
    int fps_num = 25;
    int fps_den = 1;

    /// @brief 视频宽度/高度。0 表示从首帧自动推断。
    int video_width = 0;
    int video_height = 0;

    /// @brief 视频编码器名称（默认 "libx264"）
    std::string video_codec = "libx264";

    /// @brief 音频编码器名称（默认 "aac"）
    std::string audio_codec = "aac";

    /// @brief 视频比特率（bps），默认 2 Mbps
    int64_t video_bit_rate = 2000000;

    /// @brief 音频比特率（bps），默认 128 kbps
    int64_t audio_bit_rate = 128000;

    /// @brief 音频 PCM 数据（float，16kHz mono）。若 pcm 为空则纯视频。
    audio::AudioData audio;

    /// @brief GOP 大小（关键帧间隔），默认 250
    int gop_size = 250;

    /// @brief 最大 B 帧数，默认 0（baseline，最小延迟）
    int max_b_frames = 0;

    /// @brief x264 preset（默认 "medium"）
    std::string x264_preset = "medium";

    /// @brief 验证配置合法性
    /// @return true 表示 output_path 非空、fps>0 且尺寸合法（为 0 或正数）
    bool IsValid() const {
        return !output_path.empty()
            && fps_num > 0
            && fps_den > 0
            && video_width >= 0
            && video_height >= 0
            && video_bit_rate > 0
            && audio_bit_rate > 0
            && gop_size > 0;
    }
};

// ============================================================================
// FinalMediaWriter — 最终媒体写出器
// ============================================================================

/// @brief 离线 MP4 写出器，基于 FFmpeg C API 实现。
///
/// 实现 PipelineOutputSink 接口，将 Pipeline 输出的 BGR 视频帧
/// 与输入音频按明确 time base 编码和封装为 MP4 文件。
///
/// @par 线程/回调限制
/// - 构造函数和析构函数可在任意线程调用。
/// - OnFrame / OnTerminal 遵循 PipelineOutputSink 契约：串行、互不重叠。
/// - OnTerminal 恰好一次，为最后一个回调。
/// - 关闭后所有回调静默变为 no-op。
///
/// @par 所有权
/// - FinalMediaWriter 独占输出文件；不与其他 writer 共享。
/// - WriterConfig::audio.pcm 在构造时拷贝，调用方可在构造后释放。
///
/// @par 错误语义
/// - 打开/编码失败：OnFrame 内部失败后标记错误，OnTerminal 写 trailer 失败
///   或跳过并清理半成品文件。
/// - 析构函数：若尚未清理则强制关闭并移除半成品文件，
///   记录 secondary diagnostics（类似 Pipeline 策略）。
///
/// @par 使用示例
/// @code
/// WriterConfig cfg;
/// cfg.output_path = "output.mp4";
/// cfg.audio = audio_data;  // 可选
/// auto writer = std::make_shared<FinalMediaWriter>(cfg);
/// pipeline->Start(face_img, writer);
/// @endcode
class FinalMediaWriter : public pipeline::PipelineOutputSink {
public:
    /// @brief 构造写出器，验证配置但不打开文件。
    /// @param config 写出器配置（output_path 必填）
    /// @throws std::invalid_argument 若 !config.IsValid()
    explicit FinalMediaWriter(const WriterConfig& config);

    /// @brief 析构：若尚未 Finalize 则强制关闭并移除半成品文件。
    ~FinalMediaWriter() override;

    // 禁止拷贝和移动（内部持有 FFmpeg 资源不可共享）
    FinalMediaWriter(const FinalMediaWriter&) = delete;
    FinalMediaWriter& operator=(const FinalMediaWriter&) = delete;
    FinalMediaWriter(FinalMediaWriter&&) = delete;
    FinalMediaWriter& operator=(FinalMediaWriter&&) = delete;

    // =================== PipelineOutputSink 实现 ===================

    /// @brief 接收一帧输出帧。
    ///
    /// 首帧触发打开 MP4 文件、创建编码器和写文件头。后续帧编码并写入。
    /// 内部失败时标记错误状态，后续 OnFrame 静默忽略，OnTerminal 清理。
    ///
    /// @param frame 包含 VideoFrame、交付类型等。frame.video_frame.frame_bgr
    ///              必须是 CV_8UC3 BGR 格式。
    void OnFrame(const pipeline::PipelineFrame& frame) override;

    /// @brief Pipeline 终止回调（恰好一次，最后一个回调）。
    ///
    /// 执行：flush 视频编码器 → 编码全部音频 PCM → flush 音频编码器 →
    /// 写文件 trailer → 关闭并释放所有 FFmpeg 资源。
    ///
    /// 仅当 result.terminal_state == kSucceeded、writer 无内部错误且
    /// header 已写入时才执行 Finalize；否则（Pipeline 取消/失败/排空，
    /// 或 writer 出错）清理资源并移除输出文件，不留下半成品。
    /// 若已 Finalize 过，静默返回（幂等）。
    ///
    /// @param result Pipeline 终止结果
    void OnTerminal(const pipeline::PipelineResult& result) override;

    // =================== 查询接口 ===================

    /// @brief 是否已打开（avformat 已分配且 header 已写入）
    bool IsOpen() const;

    /// @brief 是否已 Finalize（trailer 已写，资源已释放）
    bool IsFinalized() const;

    /// @brief 已写入的视频帧数（不含 flush 帧）
    int64_t GetWrittenFrameCount() const;

    /// @brief 最后一次错误码
    WriterError GetLastError() const;

    /// @brief 最后一次错误描述
    std::string GetLastErrorMessage() const;

    /// @brief 输出文件路径
    const std::string& GetOutputPath() const;

private:
    struct Impl;                       ///< PImpl，隐藏 FFmpeg 头文件
    std::unique_ptr<Impl> impl_;
};

}  // namespace output
}  // namespace digital_human
