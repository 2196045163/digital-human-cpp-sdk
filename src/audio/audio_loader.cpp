

#include <algorithm>    // std::transform
#include <cctype>       // std::tolower
#include <filesystem>  // std::filesystem::exists / is_regular_file
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "audio/audio_loader.h"

// FFmpeg 是 C 库，不加extern "C"会导致链接时符号找不到
extern "C" {
    #include <libavformat/avformat.h>   // AVFormatContext, avformat_open_input 等
    #include <libavcodec/avcodec.h>       // AVCodec, avcodec_find_decoder 等
    #include <libswresample/swresample.h> // SwrContext, swr_* 等
    #include <libavutil/avutil.h>         // av_get_sample_fmt_name 等
}


namespace digital_human {
namespace audio {
namespace {

    static AudioLoadStatus ValidatePathAndOptions(const std::string& file_path, 
        const AudioLoadOptions& options) {
        
        // 1.空路径检查
        if (file_path.empty()) {
            return AudioLoadStatus::kEmptyPath;
        }

        // 2.采样率、声道数检查
        if (options.target_sample_rate <= 0) {
            return AudioLoadStatus::kInvalidTargetSampleRate;
        }
        if (options.target_channels != 1) {
            return AudioLoadStatus::kInvalidTargetChannels;
        }

        // 3.文件存在性检查（必须在格式检查之前，否则 "." 等目录路径会被格式拦截）
        if (!std::filesystem::exists(file_path)) {
            return AudioLoadStatus::kFileNotFound;
        }
        if (std::filesystem::is_directory(file_path)) {
            return AudioLoadStatus::kPathIsDirectory;
        }

        // 4.格式检查（放在文件 I/O 之后，确保目录路径先被拦截）
        if (options.enable_format_check && !AudioLoader::IsSupportedFormat(file_path)) {
            return AudioLoadStatus::kUnsupportedFormat;
        }

        return AudioLoadStatus::kOk;
    }

    /// @brief 从 FFmpeg 对象中提取源音频信息，填入 AudioInfo 的源字段（目标字段不填，保持默认值）
    /// @param info       [out] 输出：AudioInfo 的源信息字段将被填充
    /// @param file_path  原始文件路径
    /// @param fmt_ctx    已打开并已调用 avformat_find_stream_info 的容器上下文
    /// @param stream_idx av_find_best_stream 返回的音频流索引
    /// @note  调用方需确保 fmt_ctx 已成功打开，否则内部访问空指针会崩溃
    static void FillSourceInfo(AudioInfo& info, const std::string& file_path, AVFormatContext* fmt_ctx,
                               int stream_idx) {
        AVStream* stream = fmt_ctx->streams[stream_idx];
        AVCodecParameters* codecpar = stream->codecpar;

        // -- 文件与容器 --
        info.file_path        = file_path;                 // 原始路径，直接拷贝
        info.container_format = fmt_ctx->iformat->name;    // 如 "wav", "mp3", "mov,mp4,m4a"

        // -- 音频流索引 --
        info.stream_index = stream_idx;                    // 在容器中的第几个流（0-based）

        // -- 解码方式 --
        const AVCodec* dec = avcodec_find_decoder(codecpar->codec_id);
        info.codec_name = dec ? dec->name : "unknown";     // 如 "pcm_s16le", "aac"；解码器不存在时填 unknown

        // -- 源信号参数 --
        info.source_sample_rate   = codecpar->sample_rate; // 源采样率（Hz），如 44100、48000
        info.source_channels      = codecpar->channels;     // 源声道数，如 1（mono）、2（stereo）
        info.source_sample_format = av_get_sample_fmt_name(static_cast<AVSampleFormat>(codecpar->format)); // 源采样格式名
        info.source_bit_rate      = codecpar->bit_rate;     // 源比特率（bps），可能为 0 表示容器未记录

        // -- 源时长 --
        // fmt_ctx->duration 单位是微秒；AV_NOPTS_VALUE 表示未知
        if (fmt_ctx->duration != AV_NOPTS_VALUE) {
            info.source_duration_sec = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE; // 微秒 → 秒
        }
    }


    /// @brief 打开音频文件、找到音频流并填充源信息（不关闭 fmt_ctx，留给调用方继续使用）
    /// @param file_path  音频文件路径
    /// @param fmt_ctx    [out] 成功时指向已打开的容器上下文，失败时为 nullptr
    /// @param stream_idx [out] 音频流在容器中的索引
    /// @param info       [out] 填充源信息字段（file_path / container / codec / 采样率 / 声道等）
    /// @return kOk 表示成功，此时 fmt_ctx 已打开且由调用方负责关闭；
    ///         其它值表示失败，fmt_ctx 已由本函数内部关闭
    /// @note   本函数整合了 Probe 和 DecodeToInt16 共用的前半段逻辑，避免重复代码。
    ///         成功时不关 fmt_ctx 是刻意设计的——Probe 下一步要关，DecodeToInt16 要继续创建解码器。
    static AudioLoadStatus OpenAndFindStream(const std::string& file_path, AVFormatContext*& fmt_ctx,
        int& stream_idx, AudioInfo& info) {
        
        fmt_ctx = nullptr;
        stream_idx = -1;

        // ① 打开容器：avformat_open_input 失败时自动清理，不需手动 close
        if (avformat_open_input(&fmt_ctx, file_path.c_str(), nullptr, nullptr) < 0) {
            return AudioLoadStatus::kOpenInputFailed;
        }

        // ② 补全流信息
        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            avformat_close_input(&fmt_ctx);
            return AudioLoadStatus::kFindStreamInfoFailed;
        }

        // ③ 找音频流
        stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (stream_idx < 0) {
            avformat_close_input(&fmt_ctx);
            return AudioLoadStatus::kNoAudioStream;
        }

        // ④ 填源信息
        FillSourceInfo(info, file_path, fmt_ctx, stream_idx);

        return AudioLoadStatus::kOk;
    }

} // 匿名空间

    // ====== Impl 定义 ======
    struct AudioLoader::Impl {
        int target_sample_rate; // 目标采样率

        // ========================= Impl 转发的接口 ===============================
        
        /// @brief 探测模式：只读取音频基础信息（采样率/声道/编码/时长），不解码完整 PCM
        /// @note  适合先快速扫描文件信息，再决定是否完整解码
        AudioLoadResult Probe(const std::string& file_path, const AudioLoadOptions& options) {
            AudioLoadResult al_res;

            // 1. 路径和参数校验
            AudioLoadStatus status = ValidatePathAndOptions(file_path, options);
            if (status != AudioLoadStatus::kOk) {
                al_res.success = false;
                al_res.status = status;
                al_res.error_message = AudioLoader::StatusToString(status);
                return al_res;
            }

            // 2. 打开文件 → 找音频流 → 填源信息（OpenAndFindStream 失败时已自动关闭 fmt_ctx）
            AVFormatContext* fmt_ctx = nullptr;
            int audio_stream_idx = -1;
            status = OpenAndFindStream(file_path, fmt_ctx, audio_stream_idx, al_res.info);
            if (status != AudioLoadStatus::kOk) {
                al_res.success = false;
                al_res.status = status;
                al_res.error_message = AudioLoader::StatusToString(status);
                return al_res;
            }

            // 3. Probe 只需信息不需解码，立即释放
            avformat_close_input(&fmt_ctx);

            // 4. 返回成功（audio.pcm 保持空）
            al_res.success = true;
            al_res.status = AudioLoadStatus::kOk;
            return al_res;
        }


        /// @brief 加载为 int16 PCM [-32768, 32767]，用于调试和 WAV 对照
        AudioInt16LoadResult LoadInt16FromFile(const std::string& file_path, 
                                               const AudioLoadOptions& options) {
            AudioInt16LoadResult al_res;

            auto start = std::chrono::high_resolution_clock::now();
            // 1.调用 DecodeToInt16
            AudioLoadStatus status = DecodeToInt16(file_path, options, al_res.pcm, al_res.info);

            // 2.判断成功/失败
            if (status != AudioLoadStatus::kOk) {
                al_res.success = false;
                al_res.status = status;
                al_res.error_message = AudioLoader::StatusToString(status);

                return al_res;
            }

            // 3.计算耗时返回结果
            al_res.success = true;
            al_res.status = status;
            auto end = std::chrono::high_resolution_clock::now();
            al_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            return al_res;
        }


        /// @brief 主加载接口，返回 float PCM [-1.0, 1.0]，正式算法链路使用
        AudioLoadResult LoadFromFile(const std::string& file_path, 
                                     const AudioLoadOptions& options) {
            AudioLoadResult al_res;
            
            auto start = std::chrono::high_resolution_clock::now();
            // 1.LoadInt16FromFile 获取 int16 PCM
            AudioInt16LoadResult ai16l_res = LoadInt16FromFile(file_path, options);
            // 如果失败，把 AudioInt16LoadResult 的错误信息转到 AudioLoadResult 返回
            if (!ai16l_res.success) {
                al_res.success = false;
                al_res.status = ai16l_res.status;
                al_res.error_message = ai16l_res.error_message;

                return al_res;
            }

            // 2.ConvertInt16ToFloat 把 int16 → float，并填上info
            // 即 ai16l_res 中的 int16的pcm 转 float，放到 al_res 的audio的pcm中
            al_res.audio.pcm = AudioLoader::ConvertInt16ToFloat(ai16l_res.pcm);
            al_res.info = ai16l_res.info;

            // 3.填 AudioData
            al_res.audio.sample_rate = options.target_sample_rate;
            al_res.audio.channels = options.target_channels;
            al_res.audio.format = options.output_format;

            // 4.计算 time_ms，填好返回的结构体
            al_res.success = true;
            al_res.status = AudioLoadStatus::kOk;

            auto end = std::chrono::high_resolution_clock::now();
            al_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            return al_res;
        }

        /// @brief 批量加载，对每个路径调用 LoadFromFile
        std::vector<AudioLoadResult> LoadBatch(const std::vector<std::string>& file_paths,
                                               const AudioLoadOptions& options) {
            
            std::vector<AudioLoadResult> als_res(file_paths.size());
            for (size_t i = 0; i < file_paths.size(); i++) {
                als_res[i] = LoadFromFile(file_paths[i], options);
            }

            return als_res;
        }

        // =========================================================================


        /// @brief 完整解码链路：解封装 → 解码 → 重采样 → 输出 int16 PCM
        /// @param file_path 音频文件路径
        /// @param options   加载选项（采样率/声道/max_samples 等）
        /// @param out_pcm   [out] 输出 int16 PCM，范围 [-32768, 32767]
        /// @param out_info  [out] 输出音频完整信息（源+目标参数、时长、样本数）
        /// @return kOk 成功；其它值表示失败，此时 out_pcm 为空、out_info 保留已填的源信息部分
        /// @note   流程：校验 → OpenAndFindStream → 创建解码器 → 创建重采样器
        ///         → while(read → send → while(receive → swr_convert))
        ///         → flush 解码器 → flush 重采样器 → 统一释放 → 返回结果
        AudioLoadStatus DecodeToInt16(const std::string& file_path, const AudioLoadOptions& options,
            std::vector<int16_t>& out_pcm, AudioInfo& out_info) {
            
            // 1.路径校验
            auto status = ValidatePathAndOptions(file_path, options);
            if (status != AudioLoadStatus::kOk) {
                return status;
            }


            // 2.打开文件 + 找音频流 + 填源信息
            AVFormatContext* fmt_ctx = nullptr;
            int audio_stream_idx = -1;
            status = OpenAndFindStream(file_path, fmt_ctx, audio_stream_idx, out_info);
            if (status != AudioLoadStatus::kOk) {
                return status;
            }
            // 成功后 fmt_ctx 是打开的，所以后续失败路径全要 avformat_close_input
            // 失败时 OpenAndFindStream 内部是关闭了 fmt_ctx 的


            // 3.创建并打开解码器
            AVCodecParameters* codecpar = fmt_ctx->streams[audio_stream_idx]->codecpar;

            // 3.1根据音频流的编码信息codecpar，找到解码器
            const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
            if (!codec) {
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kDecoderNotFound;
            }

            // 3.2 给解码器分配上下文
            AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
            if (!codec_ctx) {
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kCodecContextAllocFailed;
            }

            // 3.3 复制编码参数信息 —— 来自 codecpar = fmt_ctx->streams[audio_stream_idx]->codecpar
            if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
                // 按照依赖关系依次关闭解码器和释放容器上下文
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kCodecParametersFailed;
            }

            // 3.4 打开解码器
            int ret = avcodec_open2(codec_ctx, codec, nullptr);
            if (ret < 0) {
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kOpenDecoderFailed;
            }

            // 3.5 校验源参数 —— 主要是避免有损文件让采样率和声道数参数值为0
            if (codec_ctx->sample_rate <= 0 || codec_ctx->channels <= 0) {
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kInvalidSourceAudioParams;
            }


            // 4.创建重采样转换器
            // 4.1 获取输入声道布局
            // 输入声道布局如果为 0，根据声道数自己算一个默认布局
            // 声道布局是指 "这些声道分别是什么角色"（一个位掩码）
            int64_t in_ch_layout = codec_ctx->channel_layout;
            if (in_ch_layout == 0) {
                in_ch_layout = av_get_default_channel_layout(codec_ctx->channels);
            }

            // 4.2 设置转换器参数
            SwrContext* swr_ctx = swr_alloc_set_opts(
                nullptr,                    // 1 已有的 SwrContext（nullptr = 新建）
                AV_CH_LAYOUT_MONO,          // 2 输出声道布局
                AV_SAMPLE_FMT_S16,          // 3 输出采样格式
                options.target_sample_rate, // 4 输出采样率
                in_ch_layout,               // 5 输入声道布局
                codec_ctx->sample_fmt,      // 6 输入采样格式
                codec_ctx->sample_rate,     // 7 输入采样率
                0,                          // 8 额外选项
                nullptr                     // 9 日志上下文
            );
            if (!swr_ctx) {
                // 释放解码器、容器上下文 —— 因为 swr_ctx 就是nullptr 不用手动释放了
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kResamplerAllocFailed;
            }
            // 4.3 完全初始化转换器
            ret = swr_init(swr_ctx);
            if (ret < 0) {
                // 释放转换器、解码器、容器上下文
                swr_free(&swr_ctx);
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kResamplerInitFailed;
            }


            // 5.分配压缩数据包 packet 和原始数据帧 frame 缓冲区
            AVPacket* packet = av_packet_alloc();
            if (!packet) {
                swr_free(&swr_ctx);
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kPacketAllocFailed;
            }

            AVFrame* frame = av_frame_alloc();
            if (!frame) {
                // 前面packet 已经分配了，这里也要释放
                av_packet_free(&packet); 

                swr_free(&swr_ctx);
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                return AudioLoadStatus::kFrameAllocFailed;
            }

            
            // 6.主解码循环
            out_pcm.clear();
            while (av_read_frame(fmt_ctx, packet) >= 0) {
                // 6.1 只处理音频流，跳过视频/字幕包
                if (packet->stream_index != audio_stream_idx) {
                    av_packet_unref(packet);
                    continue;
                }

                // 6.2 送压缩包进解码器（EAGAIN 正常，其它负值才算错误）
                ret = avcodec_send_packet(codec_ctx, packet);
                av_packet_unref(packet);  // 已送入解码器，立即释放 packet 引用
                if (ret < 0 && ret != AVERROR(EAGAIN)) {
                    break;  // 解码器内部出错
                }

                // 6.3 取原始音频帧（while：一个 packet 可能产出多帧）
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    // 6.3.1 计算重采样后的输出样本数（含管线延迟）
                    int output_samples = av_rescale_rnd(
                        swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
                        options.target_sample_rate, codec_ctx->sample_rate,
                        AV_ROUND_UP);

                    // 6.3.2 分配临时缓冲区，执行重采样
                    std::vector<int16_t> output_buf(output_samples);
                    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output_buf.data());
                    int converted = swr_convert(swr_ctx, &out_ptr, output_samples,
                        (const uint8_t**)frame->data, frame->nb_samples);

                    // 6.3.3 追加有效数据到结果集
                    if (converted > 0) {
                        out_pcm.insert(out_pcm.end(), output_buf.begin(), output_buf.begin() + converted);

                        // 达到 max_samples 上限：截断后跳出双重循环
                        if (options.max_samples > 0 &&
                            static_cast<int64_t>(out_pcm.size()) >= options.max_samples) {
                            out_pcm.resize(static_cast<size_t>(options.max_samples));
                            av_frame_unref(frame);
                            goto cleanup;
                        }
                    }

                    // 6.3.4 释放 frame 内部引用，准备接收下一帧
                    av_frame_unref(frame);
                }
            }


            // 7. 清理管线残留数据（flush）
            // 7.1 flush 解码器：通知解码器没有更多数据，取出最后残留帧
            avcodec_send_packet(codec_ctx, nullptr);
            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                int output_samples = av_rescale_rnd(
                    swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
                    options.target_sample_rate, codec_ctx->sample_rate,
                    AV_ROUND_UP);

                std::vector<int16_t> output_buf(output_samples);
                uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output_buf.data());
                int converted = swr_convert(swr_ctx, &out_ptr, output_samples,
                    (const uint8_t**)frame->data, frame->nb_samples);

                if (converted > 0) {
                    out_pcm.insert(out_pcm.end(), output_buf.begin(), output_buf.begin() + converted);
                }
                av_frame_unref(frame);
            }

            // 7.2 flush 重采样器：取出 FIR 滤波器管线里残留的最后几个采样点
            {
                int delayed;
                do {
                    std::vector<int16_t> flush_buf(4096);
                    uint8_t* flush_ptr = reinterpret_cast<uint8_t*>(flush_buf.data());
                    delayed = swr_convert(swr_ctx, &flush_ptr, 4096, nullptr, 0);
                    if (delayed > 0) {
                        out_pcm.insert(out_pcm.end(), flush_buf.begin(), flush_buf.begin() + delayed);
                    }
                } while (delayed > 0);
            }

            // 7.3 统一释放资源（goto cleanup 也跳到这里）
        cleanup:
            av_frame_free(&frame);
            av_packet_free(&packet);
            swr_free(&swr_ctx);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);


            // 8. 填输出信息并返回
            out_info.target_sample_rate = options.target_sample_rate;
            out_info.target_channels = options.target_channels;
            out_info.target_sample_format = AudioSampleFormat::kInt16;
            out_info.sample_count = out_pcm.size();
            out_info.duration_sec = static_cast<double>(out_pcm.size()) / options.target_sample_rate;

            if (out_pcm.empty()) {
                return AudioLoadStatus::kNoSamplesDecoded;
            }
            return AudioLoadStatus::kOk;
        }

    };

    AudioLoader::AudioLoader(int target_sample_rate) : pImpl_(std::make_unique<Impl>()) {
        pImpl_->target_sample_rate = target_sample_rate;
    }

    AudioLoader::~AudioLoader() = default;

    AudioLoader::AudioLoader(AudioLoader&&) noexcept = default;
    AudioLoader& AudioLoader::operator=(AudioLoader&&) noexcept = default;

    // =================== 主要接口 =========================

    /// @brief 探测模式：只读取音频基础信息（采样率/声道/编码/时长），不解码完整 PCM
    /// @note  适合先快速扫描文件信息，再决定是否完整解码
    AudioLoadResult AudioLoader::Probe(const std::string& file_path,
        const AudioLoadOptions& options) {
        return pImpl_->Probe(file_path, options);
    }

    /// @brief 主加载接口，返回 float PCM [-1.0, 1.0]，正式算法链路使用
    AudioLoadResult AudioLoader::LoadFromFile(const std::string& file_path,
        const AudioLoadOptions& options) {
        return pImpl_->LoadFromFile(file_path, options);
    }

    /// @brief 加载为 int16 PCM [-32768, 32767]，用于调试和 WAV 对照
    AudioInt16LoadResult AudioLoader::LoadInt16FromFile(const std::string& file_path,
        const AudioLoadOptions& options) {
        return pImpl_->LoadInt16FromFile(file_path, options);
    }

    /// @brief 批量加载，对每个路径调用 LoadFromFile
    std::vector<AudioLoadResult> AudioLoader::LoadBatch(
        const std::vector<std::string>& file_paths,
        const AudioLoadOptions& options) {
        return pImpl_->LoadBatch(file_paths, options);
    }

    /// @brief 将 int16 PCM 转换为 float PCM
    /// @note  空输入返回空 vector；不做范围 clamp
    std::vector<float> AudioLoader::ConvertInt16ToFloat(
        const std::vector<int16_t> &pcm_int16) {

        std::vector<float> output(pcm_int16.size());
        for (size_t i = 0; i < pcm_int16.size(); i++)
        {
            output[i] = pcm_int16[i] / 32768.0f;
        }

        return output;
    }
    // ===================================================


    /// @brief 获取构造时设置的目标采样率
    int AudioLoader::GetTargetSampleRate() const {
        return pImpl_->target_sample_rate;
    }

    /// @brief 返回支持的扩展名列表
    /// @return 小写含点号的扩展名列表，如 {".wav", ".mp3", ".aac", ".m4a", ".mp4", ".flac"}
    std::vector<std::string> AudioLoader::GetSupportedExtensions() {
        return std::vector<std::string>{".wav", ".mp3", ".aac", ".m4a", ".mp4", ".flac"};
    }

    /// @brief 通过扩展名判断是否为支持的音频格式（不区分大小写）
    /// @return true 表示扩展名在支持列表中
    bool AudioLoader::IsSupportedFormat(const std::string& file_path) {
        // 1.提取扩展名
        std::string extension = std::filesystem::path(file_path).extension().string();
        if (extension.empty()) {
            return false;
        }

        // 2.统一转换成小写
        // 这里使用lambda表示可调用对象是为了和某些编译器兼容（如GCC），::tolower 编译器可能报错
        // 这样提高代码的平台兼容性
        std::transform(extension.begin(), extension.end(), extension.begin(), 
            [](unsigned char c) { return std::tolower(c); });

        // 3.查表
        auto supported = GetSupportedExtensions();
        return std::find(supported.begin(), supported.end(), extension) != supported.end();
    }


    /// @brief 将状态码转为人类可读字符串，用于日志和 example 打印
    /// @return 非空字符串（覆盖所有状态码）
    std::string AudioLoader::StatusToString(AudioLoadStatus status) {
        // ai给的状态码太多了，这里改用映射表 unordered_map
        static const std::unordered_map<AudioLoadStatus, std::string> k_map = {
            {AudioLoadStatus::kOk,"成功"},
            {AudioLoadStatus::kEmptyPath,               "路径为空"},
            {AudioLoadStatus::kFileNotFound,             "文件不存在"},
            {AudioLoadStatus::kPathIsDirectory,           "路径为目录"},
            {AudioLoadStatus::kUnsupportedFormat,         "不支持的音频格式"},
            {AudioLoadStatus::kInvalidTargetSampleRate,   "无效的目标采样率"},
            {AudioLoadStatus::kInvalidTargetChannels,     "无效的目标声道数"},
            {AudioLoadStatus::kOpenInputFailed,           "打开文件失败"},
            {AudioLoadStatus::kFindStreamInfoFailed,      "读取流信息失败"},
            {AudioLoadStatus::kNoAudioStream,             "未找到音频流"},
            {AudioLoadStatus::kDecoderNotFound,           "未找到解码器"},
            {AudioLoadStatus::kCodecContextAllocFailed,    "解码器上下文分配失败"},
            {AudioLoadStatus::kCodecParametersFailed,      "解码器参数复制失败"},
            {AudioLoadStatus::kOpenDecoderFailed,          "打开解码器失败"},
            {AudioLoadStatus::kInvalidSourceAudioParams,   "源音频参数无效"},
            {AudioLoadStatus::kResamplerAllocFailed,       "重采样器分配失败"},
            {AudioLoadStatus::kResamplerInitFailed,        "重采样器初始化失败"},
            {AudioLoadStatus::kPacketAllocFailed,          "数据包分配失败"},
            {AudioLoadStatus::kFrameAllocFailed,           "音频帧分配失败"},
            {AudioLoadStatus::kDecodeFailed,              "解码失败"},
            {AudioLoadStatus::kResampleFailed,            "重采样失败"},
            {AudioLoadStatus::kFlushFailed,               "刷新失败"},
            {AudioLoadStatus::kNoSamplesDecoded,           "未解码到任何样本"},
            {AudioLoadStatus::kInvalidPcmData,             "PCM 数据包含异常值"},
            {AudioLoadStatus::kFileSystemError,            "文件系统错误"},
            {AudioLoadStatus::kUnknownError,             "未定义错误"},
        };

        auto it = k_map.find(status);
        if (it != k_map.end()) {
            return it->second;
        }

        return "未定义错误";
    }

} // namespace audio
} // namespace digital_human