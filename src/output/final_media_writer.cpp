extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <cstdio>
#include <stdexcept>

#include "output/final_media_writer.h"

namespace digital_human {
namespace output {

// ============================================================================
// FinalMediaWriter（实现见下方 PImpl）— 设计要点
// ============================================================================
//
// 资源所有权与生命周期：
//   - fmt_ctx（AVFormatContext）是 muxer 与 AVIO（输出文件句柄）的唯一所有者，
//     并拥有全部 AVStream；video_enc_ctx / audio_enc_ctx 是独立分配的编码器
//     上下文，必须先于 fmt_ctx 释放（编码器内部缓冲可能持有流相关状态）。
//   - stream 指针归 fmt_ctx 所有，不单独释放，只在清理时置空防悬垂。
//   - 所有 FFmpeg 对象仅经 SafeFree* 辅助释放；Cleanup 顺序固定：
//     Sws/Swr → AVFrame → AVPacket → 编码器 → fmt_ctx，保证任一路径
//     都无泄漏、无双释放。
//
// PTS 换算约定（统一换算到"编码器 time_base"后再写 AVFrame）：
//   - 视频：Pipeline 交付微秒级时间戳（固定帧率下与帧序号一一对应，
//     frame_index = pts_us * fps / 1e6）。OnFrame 中 av_rescale_q 换算到
//     video_enc_ctx->time_base（fps_den/fps_num），并经 next_video_pts 钳制
//     单调不后退——编码器不接受 PTS 回退，回退会污染容器时间戳。
//   - 音频：PCM 是连续采样流，AVFrame::pts 直接取"已消费采样数"
//     （sample_offset），其单位恰好是 audio_enc_ctx->time_base
//     （1/sample_rate），天然无需换算。
//   - 包写盘时 av_packet_rescale_ts 再做"编码器 time_base → 流 time_base"
//     换算；视频/音频共用 ReceiveAndWritePackets 交错写帧，保证 mux 顺序。
//
// Finalize 顺序（flush → 音频 → trailer → 关闭）：
//   先 flush 视频编码器（avcodec_send_frame(nullptr) 排出延迟帧），再编码
//   全部音频并 flush 音频编码器，最后 av_write_trailer 写 moov 等收尾 atom
//   并关闭文件。顺序不可颠倒：编码器滞留帧不先排出，容器内帧数就不足。
//
// OnTerminal 处理路径（区分 Pipeline 成功/失败/取消）：
//   - Pipeline kSucceeded 且 writer 无错且 header 已写 → Finalize，产物完整。
//   - 其余一律（kDraining/kCancelled/kFailed、writer 自身错误、从未成功打开
//     输出）→ 只 Cleanup、不写 trailer、删除半成品文件。
//
// 失败时删除不完整 MP4 的原因：未写 trailer 的 MP4 无法被播放器/ffprobe 正常
// 解析，是明确的坏文件。让文件"不存在"而非"存在但损坏"，可使上层（CLI、
// 测试）以退出码/JSON/文件存在性为唯一真相源，避免"产物存在但不完整"的
// 歧义状态被误判为成功。

// ============================================================================
// 内部 RAII 辅助
// ============================================================================

namespace {

/// @brief 安全释放 AVFormatContext 并可选关闭文件
void SafeFreeFormatContext(AVFormatContext** ctx) {
    if (ctx && *ctx) {
        // 如果已 write_header 但未 write_trailer，关闭 AVIO 防止泄漏
        if ((*ctx)->pb) {
            avio_closep(&(*ctx)->pb);
        }
        avformat_free_context(*ctx);
        *ctx = nullptr;
    }
}

/// @brief 安全释放 AVCodecContext
void SafeFreeCodecContext(AVCodecContext** ctx) {
    if (ctx && *ctx) {
        avcodec_free_context(ctx);
    }
}

/// @brief 安全释放 AVFrame
void SafeFreeFrame(AVFrame** frame) {
    if (frame && *frame) {
        av_frame_free(frame);
    }
}

/// @brief 安全释放 AVPacket
void SafeFreePacket(AVPacket** pkt) {
    if (pkt && *pkt) {
        av_packet_free(pkt);
    }
}

/// @brief 安全释放 SwsContext
void SafeFreeSwsContext(SwsContext** ctx) {
    if (ctx && *ctx) {
        sws_freeContext(*ctx);
        *ctx = nullptr;
    }
}

/// @brief 安全释放 SwrContext
void SafeFreeSwrContext(SwrContext** ctx) {
    if (ctx && *ctx) {
        swr_free(ctx);
        // swr_free sets pointer to nullptr internally
    }
}

}  // anonymous namespace

// ============================================================================
// PImpl
// ============================================================================

struct FinalMediaWriter::Impl {
    // ---- 配置 ----
    WriterConfig config;
    bool finalized = false;
    bool header_written = false;
    WriterError last_error = WriterError::kOk;
    std::string last_error_msg;
    int64_t written_frame_count = 0;

    // ---- FFmpeg 资源 ----
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* video_enc_ctx = nullptr;
    AVCodecContext* audio_enc_ctx = nullptr;
    AVStream* video_stream = nullptr;
    AVStream* audio_stream = nullptr;
    AVFrame* video_avframe = nullptr;
    AVFrame* audio_avframe = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws_ctx = nullptr;
    SwrContext* swr_ctx = nullptr;

    // ---- 状态 ----
    int first_width = 0;
    int first_height = 0;
    int64_t next_video_pts = 0;        // 以 video_enc_ctx->time_base 为单位
    bool has_video_stream = false;
    bool has_audio_stream = false;
    int audio_frame_size = 0;          // 编码器期望的每帧采样数
    int64_t audio_samples_written = 0; // 已编码的音频采样数

    // ---- 临时缓冲区 ----
    uint8_t* audio_src_data[1] = {nullptr}; // swr 输入指针
    int audio_src_linesize = 0;

    explicit Impl(const WriterConfig& cfg) : config(cfg) {}

    ~Impl() {
        Cleanup();
        // 若未 Finalize 且文件存在，移除半成品
        if (!finalized && !config.output_path.empty()) {
            std::remove(config.output_path.c_str());
        }
    }

    // ---- 错误标记 ----
    void SetError(WriterError code, const std::string& msg) {
        if (last_error == WriterError::kOk) {
            last_error = code;
            last_error_msg = msg;
        }
    }

    // ---- 资源清理 ----
    void Cleanup() {
        // 顺序：先释放需要编码器的，再释放格式上下文
        SafeFreeSwsContext(&sws_ctx);
        SafeFreeSwrContext(&swr_ctx);
        SafeFreeFrame(&video_avframe);
        SafeFreeFrame(&audio_avframe);
        SafeFreePacket(&pkt);
        if (audio_enc_ctx) {
            avcodec_free_context(&audio_enc_ctx);
        }
        if (video_enc_ctx) {
            avcodec_free_context(&video_enc_ctx);
        }
        if (fmt_ctx) {
            // 如有 AVIO 未关闭，关闭之
            if (fmt_ctx->pb) {
                avio_closep(&fmt_ctx->pb);
            }
            avformat_free_context(fmt_ctx);
            fmt_ctx = nullptr;
        }
        video_stream = nullptr;
        audio_stream = nullptr;
    }

    // ---- 打开输出文件 ----
    WriterError Open(int width, int height) {
        first_width = width;
        first_height = height;

        // 1. 分配输出格式上下文
        int ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, "mp4",
                                                  config.output_path.c_str());
        if (ret < 0 || !fmt_ctx) {
            SetError(WriterError::kFormatContextFailed,
                     "Failed to allocate output format context");
            return WriterError::kFormatContextFailed;
        }

        // 2. 创建视频流
        if (!CreateVideoStream(width, height)) {
            return last_error;
        }

        // 3. 创建音频流（若有音频数据）
        if (!config.audio.pcm.empty()) {
            if (!CreateAudioStream()) {
                return last_error;
            }
        }

        // 4. 打开输出文件（AVIO）
        if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            ret = avio_open(&fmt_ctx->pb, config.output_path.c_str(),
                            AVIO_FLAG_WRITE);
            if (ret < 0) {
                char err_buf[128];
                av_strerror(ret, err_buf, sizeof(err_buf));
                SetError(WriterError::kOutputPathInvalid,
                         std::string("avio_open failed: ") + err_buf);
                return WriterError::kOutputPathInvalid;
            }
        }

        // 5. 写文件头
        ret = avformat_write_header(fmt_ctx, nullptr);
        if (ret < 0) {
            char err_buf[128];
            av_strerror(ret, err_buf, sizeof(err_buf));
            SetError(WriterError::kHeaderWriteFailed,
                     std::string("avformat_write_header failed: ") + err_buf);
            return WriterError::kHeaderWriteFailed;
        }
        header_written = true;

        // 6. 分配临时资源
        pkt = av_packet_alloc();
        if (!pkt) {
            SetError(WriterError::kPacketAllocFailed,
                     "Failed to allocate AVPacket");
            return WriterError::kPacketAllocFailed;
        }

        return WriterError::kOk;
    }

    // ---- 创建视频流 ----
    bool CreateVideoStream(int width, int height) {
        const AVCodec* codec = avcodec_find_encoder_by_name(
            config.video_codec.c_str());
        if (!codec) {
            // fallback to mpeg4
            codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
            if (!codec) {
                SetError(WriterError::kEncoderOpenFailed,
                         "Video codec not found: " + config.video_codec);
                return false;
            }
        }

        video_stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!video_stream) {
            SetError(WriterError::kStreamCreationFailed,
                     "Failed to create video stream");
            return false;
        }

        video_enc_ctx = avcodec_alloc_context3(codec);
        if (!video_enc_ctx) {
            SetError(WriterError::kCodecContextAllocFailed,
                     "Failed to alloc video codec context");
            return false;
        }

        // 设置视频编码参数
        video_enc_ctx->width = width;
        video_enc_ctx->height = height;
        video_enc_ctx->time_base = AVRational{config.fps_den, config.fps_num};
        video_enc_ctx->framerate = AVRational{config.fps_num, config.fps_den};
        video_enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        video_enc_ctx->bit_rate = config.video_bit_rate;
        video_enc_ctx->gop_size = config.gop_size;
        video_enc_ctx->max_b_frames = config.max_b_frames;

        // x264 preset
        if (config.video_codec == "libx264" ||
            codec->id == AV_CODEC_ID_H264) {
            av_opt_set(video_enc_ctx->priv_data, "preset",
                       config.x264_preset.c_str(), 0);
        }

        // 全局 header（MP4 需要）
        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            video_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        // 打开编码器
        int ret = avcodec_open2(video_enc_ctx, codec, nullptr);
        if (ret < 0) {
            SetError(WriterError::kEncoderOpenFailed,
                     "Failed to open video encoder");
            return false;
        }

        // 复制参数到流
        ret = avcodec_parameters_from_context(video_stream->codecpar,
                                              video_enc_ctx);
        if (ret < 0) {
            SetError(WriterError::kCodecContextAllocFailed,
                     "Failed to copy video codec params to stream");
            return false;
        }

        video_stream->time_base = video_enc_ctx->time_base;
        video_stream->avg_frame_rate = video_enc_ctx->framerate;
        video_stream->r_frame_rate = video_enc_ctx->framerate;
        has_video_stream = true;

        // 分配视频 AVFrame
        video_avframe = av_frame_alloc();
        if (!video_avframe) {
            SetError(WriterError::kFrameAllocFailed,
                     "Failed to alloc video AVFrame");
            return false;
        }
        video_avframe->format = video_enc_ctx->pix_fmt;
        video_avframe->width = width;
        video_avframe->height = height;
        ret = av_frame_get_buffer(video_avframe, 0);
        if (ret < 0) {
            SetError(WriterError::kFrameAllocFailed,
                     "Failed to alloc video frame buffer");
            return false;
        }

        // 创建 SwsContext: BGR24 → YUV420P
        sws_ctx = sws_getContext(
            width, height, AV_PIX_FMT_BGR24,
            width, height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) {
            SetError(WriterError::kSwsContextFailed,
                     "Failed to create SwsContext");
            return false;
        }

        return true;
    }

    // ---- 创建音频流 ----
    bool CreateAudioStream() {
        const AVCodec* codec = avcodec_find_encoder_by_name(
            config.audio_codec.c_str());
        if (!codec) {
            // fallback to default AAC
            codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
            if (!codec) {
                SetError(WriterError::kEncoderOpenFailed,
                         "Audio codec not found: " + config.audio_codec);
                return false;
            }
        }

        audio_stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!audio_stream) {
            SetError(WriterError::kStreamCreationFailed,
                     "Failed to create audio stream");
            return false;
        }

        audio_enc_ctx = avcodec_alloc_context3(codec);
        if (!audio_enc_ctx) {
            SetError(WriterError::kCodecContextAllocFailed,
                     "Failed to alloc audio codec context");
            return false;
        }

        // 设置音频编码参数
        audio_enc_ctx->sample_rate = config.audio.sample_rate;
        audio_enc_ctx->channels = config.audio.channels;
        audio_enc_ctx->channel_layout = config.audio.channels == 1
            ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;
        audio_enc_ctx->sample_fmt = codec->sample_fmts
            ? codec->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
        audio_enc_ctx->bit_rate = config.audio_bit_rate;
        audio_enc_ctx->time_base = AVRational{1, config.audio.sample_rate};

        // 全局 header
        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        // 打开编码器
        int ret = avcodec_open2(audio_enc_ctx, codec, nullptr);
        if (ret < 0) {
            SetError(WriterError::kEncoderOpenFailed,
                     "Failed to open audio encoder");
            return false;
        }

        // 复制参数到流
        ret = avcodec_parameters_from_context(audio_stream->codecpar,
                                              audio_enc_ctx);
        if (ret < 0) {
            SetError(WriterError::kCodecContextAllocFailed,
                     "Failed to copy audio codec params to stream");
            return false;
        }

        audio_stream->time_base = audio_enc_ctx->time_base;
        has_audio_stream = true;

        // 编码器期望的帧大小
        audio_frame_size = audio_enc_ctx->frame_size;
        if (audio_frame_size <= 0) {
            audio_frame_size = 1024; // AAC 默认
        }

        // 分配音频 AVFrame
        audio_avframe = av_frame_alloc();
        if (!audio_avframe) {
            SetError(WriterError::kFrameAllocFailed,
                     "Failed to alloc audio AVFrame");
            return false;
        }
        audio_avframe->format = audio_enc_ctx->sample_fmt;
        audio_avframe->channels = audio_enc_ctx->channels;
        audio_avframe->channel_layout = audio_enc_ctx->channel_layout;
        audio_avframe->nb_samples = audio_frame_size;
        audio_avframe->sample_rate = audio_enc_ctx->sample_rate;
        ret = av_frame_get_buffer(audio_avframe, 0);
        if (ret < 0) {
            SetError(WriterError::kFrameAllocFailed,
                     "Failed to alloc audio frame buffer");
            return false;
        }

        // 创建 SwrContext: float planar → encoder sample format
        swr_ctx = swr_alloc_set_opts(
            nullptr,
            audio_enc_ctx->channel_layout,
            audio_enc_ctx->sample_fmt,
            audio_enc_ctx->sample_rate,
            audio_enc_ctx->channel_layout,
            AV_SAMPLE_FMT_FLT,  // input: float non-planar (interleaved)
            audio_enc_ctx->sample_rate,
            0, nullptr);
        if (!swr_ctx) {
            SetError(WriterError::kSwrContextFailed,
                     "Failed to alloc SwrContext");
            return false;
        }
        ret = swr_init(swr_ctx);
        if (ret < 0) {
            SetError(WriterError::kSwrContextFailed,
                     "Failed to init SwrContext");
            return false;
        }

        return true;
    }

    // ---- 编码并写入一帧视频 ----
    bool WriteVideoFrame(const uint8_t* bgr_data, int bgr_stride,
                         int64_t pts_in_stream_units) {
        // 确保 AVFrame 可写
        int ret = av_frame_make_writable(video_avframe);
        if (ret < 0) {
            SetError(WriterError::kVideoEncodeFailed,
                     "Failed to make video frame writable");
            return false;
        }

        // BGR → YUV420P
        const uint8_t* src_slice[1] = {bgr_data};
        int src_stride[1] = {bgr_stride};
        sws_scale(sws_ctx, src_slice, src_stride, 0, first_height,
                  video_avframe->data, video_avframe->linesize);

        video_avframe->pts = pts_in_stream_units;

        // 发送到编码器
        ret = avcodec_send_frame(video_enc_ctx, video_avframe);
        if (ret < 0) {
            SetError(WriterError::kVideoEncodeFailed,
                     "avcodec_send_frame failed");
            return false;
        }

        // 接收编码后的包
        return ReceiveAndWritePackets(video_enc_ctx, video_stream);
    }

    // ---- 从编码器接收包并交错写入 ----
    bool ReceiveAndWritePackets(AVCodecContext* enc_ctx, AVStream* stream) {
        int ret = 0;
        while (ret >= 0) {
            av_packet_unref(pkt);
            ret = avcodec_receive_packet(enc_ctx, pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return true;
            }
            if (ret < 0) {
                SetError(WriterError::kVideoEncodeFailed,
                         "avcodec_receive_packet failed");
                return false;
            }

            // 将包 PTS/DTS 从编码器时间基转换为流时间基
            av_packet_rescale_ts(pkt, enc_ctx->time_base, stream->time_base);
            pkt->stream_index = stream->index;

            // 交错写入
            ret = av_interleaved_write_frame(fmt_ctx, pkt);
            if (ret < 0) {
                SetError(WriterError::kInterleavedWriteFailed,
                         "av_interleaved_write_frame failed");
                return false;
            }
        }
        return true;
    }

    // ---- Flush 视频编码器 ----
    bool FlushVideoEncoder() {
        int ret = avcodec_send_frame(video_enc_ctx, nullptr);
        if (ret < 0 && ret != AVERROR_EOF) {
            SetError(WriterError::kFlushFailed,
                     "Failed to flush video encoder (send)");
            return false;
        }
        return ReceiveAndWritePackets(video_enc_ctx, video_stream);
    }

    // ---- 编码全部音频并写入 ----
    bool WriteAllAudio() {
        if (!has_audio_stream || config.audio.pcm.empty()) {
            return true;
        }

        const auto& pcm = config.audio.pcm;
        int64_t total_samples = static_cast<int64_t>(pcm.size())
                                / config.audio.channels;
        int sample_rate = config.audio.sample_rate;

        // 将 float PCM 转为适合 swr 的输入指针数组
        // float interleaved: src_data[0] points to pcm.data()
        const uint8_t* src_buf = reinterpret_cast<const uint8_t*>(pcm.data());

        int64_t sample_offset = 0;
        while (sample_offset < total_samples) {
            int remaining = static_cast<int>(total_samples - sample_offset);
            int nb_samples = (remaining < audio_frame_size)
                ? remaining : audio_frame_size;

            // 确保音频 AVFrame 可写
            int ret = av_frame_make_writable(audio_avframe);
            if (ret < 0) {
                SetError(WriterError::kAudioEncodeFailed,
                         "Failed to make audio frame writable");
                return false;
            }

            // 重采样: float interleaved → encoder sample format
            // 对于短帧（最后不足 audio_frame_size），调整 AVFrame
            if (nb_samples < audio_frame_size) {
                audio_avframe->nb_samples = nb_samples;
            } else {
                audio_avframe->nb_samples = audio_frame_size;
            }

            const uint8_t* src_ptr = src_buf + sample_offset
                * config.audio.channels * sizeof(float);
            ret = swr_convert(
                swr_ctx,
                audio_avframe->data, nb_samples,
                &src_ptr, nb_samples);
            if (ret < 0) {
                SetError(WriterError::kAudioEncodeFailed,
                         "swr_convert failed");
                return false;
            }

            audio_avframe->pts = sample_offset;

            // 发送到编码器
            ret = avcodec_send_frame(audio_enc_ctx, audio_avframe);
            if (ret < 0) {
                SetError(WriterError::kAudioEncodeFailed,
                         "audio avcodec_send_frame failed");
                return false;
            }

            // 接收并写入
            if (!ReceiveAndWritePackets(audio_enc_ctx, audio_stream)) {
                return false;
            }

            sample_offset += nb_samples;
        }

        audio_samples_written = sample_offset;

        // Flush 音频编码器
        return FlushAudioEncoder();
    }

    // ---- Flush 音频编码器 ----
    bool FlushAudioEncoder() {
        int ret = avcodec_send_frame(audio_enc_ctx, nullptr);
        if (ret < 0 && ret != AVERROR_EOF) {
            SetError(WriterError::kFlushFailed,
                     "Failed to flush audio encoder (send)");
            return false;
        }
        return ReceiveAndWritePackets(audio_enc_ctx, audio_stream);
    }

    // ---- Finalize：最终化输出 ----
    bool Finalize() {
        if (finalized) return true; // 幂等

        bool success = true;

        // Flush 视频编码器
        if (has_video_stream && video_enc_ctx) {
            if (!FlushVideoEncoder()) success = false;
        }

        // 编码全部音频
        if (has_audio_stream && audio_enc_ctx) {
            if (!WriteAllAudio()) success = false;
        }

        // 写 trailer
        if (fmt_ctx && header_written) {
            int ret = av_write_trailer(fmt_ctx);
            if (ret < 0) {
                SetError(WriterError::kTrailerWriteFailed,
                         "Failed to write trailer");
                success = false;
            }
        }

        // 清理资源
        Cleanup();
        finalized = true;

        // 失败则删除半成品文件
        if (!success && !config.output_path.empty()) {
            std::remove(config.output_path.c_str());
        }

        return success;
    }
};

// ============================================================================
// FinalMediaWriter 公共接口
// ============================================================================

FinalMediaWriter::FinalMediaWriter(const WriterConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
    if (!config.IsValid()) {
        throw std::invalid_argument("FinalMediaWriter: invalid config");
    }
}

FinalMediaWriter::~FinalMediaWriter() {
    // 若尚未 Finalize，强制清理（析构函数不抛异常）
    if (impl_ && !impl_->finalized) {
        impl_->Finalize();
    }
}

void FinalMediaWriter::OnFrame(const pipeline::PipelineFrame& frame) {
    if (impl_->finalized) return;
    if (impl_->last_error != WriterError::kOk) return;

    const auto& vf = frame.video_frame.frame_bgr;
    if (vf.empty()) {
        impl_->SetError(WriterError::kInvalidFrameDimensions,
                        "Empty frame received");
        return;
    }

    int w = vf.cols;
    int h = vf.rows;

    // 首帧：打开输出
    if (!impl_->header_written) {
        WriterError err = impl_->Open(w, h);
        if (err != WriterError::kOk) {
            return; // Open 内部已 SetError
        }
    }

    // 校验尺寸一致
    if (w != impl_->first_width || h != impl_->first_height) {
        impl_->SetError(WriterError::kInvalidFrameDimensions,
                        "Frame dimensions changed");
        return;
    }

    // 将 PTS 从微秒转换为编码器时间基单位（AVFrame::pts 使用编码器时间基）
    int64_t pts_us = frame.video_frame.pts.microseconds;
    int64_t pts_in_tb = av_rescale_q(
        pts_us, AVRational{1, 1000000},
        impl_->video_enc_ctx->time_base);

    // 确保 PTS 不会回退
    if (pts_in_tb < impl_->next_video_pts && pts_in_tb >= 0) {
        // 允许通过，但使用 next_video_pts 保持单调
        pts_in_tb = impl_->next_video_pts;
    }
    if (pts_in_tb < 0) {
        pts_in_tb = impl_->next_video_pts;
    }

    // 编码并写入
    if (!impl_->WriteVideoFrame(vf.data, static_cast<int>(vf.step[0]),
                                pts_in_tb)) {
        return; // WriteVideoFrame 内部已 SetError
    }

    impl_->written_frame_count++;
    impl_->next_video_pts = pts_in_tb + 1;
}

void FinalMediaWriter::OnTerminal(const pipeline::PipelineResult& result) {
    if (impl_->finalized) return; // 幂等

    // Pipeline 未正常完成（取消/失败/排空等），或 writer 自身出错，
    // 或从未成功打开输出：清理资源并移除半成品文件，不写 trailer。
    if (result.terminal_state != pipeline::PipelineState::kSucceeded
        || impl_->last_error != WriterError::kOk
        || !impl_->header_written) {
        impl_->Cleanup();
        impl_->finalized = true;
        if (!impl_->config.output_path.empty()) {
            std::remove(impl_->config.output_path.c_str());
        }
        return;
    }

    // 仅当 Pipeline 成功完成、writer 无错误且 header 已写入时才 Finalize
    // （flush + trailer），确保输出文件完整可解析。
    impl_->Finalize();
}

bool FinalMediaWriter::IsOpen() const {
    return impl_->header_written && !impl_->finalized;
}

bool FinalMediaWriter::IsFinalized() const {
    return impl_->finalized;
}

int64_t FinalMediaWriter::GetWrittenFrameCount() const {
    return impl_->written_frame_count;
}

WriterError FinalMediaWriter::GetLastError() const {
    return impl_->last_error;
}

std::string FinalMediaWriter::GetLastErrorMessage() const {
    return impl_->last_error_msg;
}

const std::string& FinalMediaWriter::GetOutputPath() const {
    return impl_->config.output_path;
}

}  // namespace output
}  // namespace digital_human
