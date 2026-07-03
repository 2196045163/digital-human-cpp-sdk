# 学习复盘卡

### 模块：audio_loader

1. 这个模块叫什么？代码类名叫什么？
    这个模块是音频加载模块，类名是 AudioLoader

2. 解决什么问题？
    把用户提供的 wav / mp3 / mp4 / aac / m4a / flac 等任意格式的音频文件，
    通过 FFmpeg 完成解封装 → 解码 → 重采样 → 声道转换 → 格式转换，
    统一输出 16kHz / mono / float PCM，为后续 AudioFramer / MelFeatureExtractor /
    模型推理提供标准化的音频输入。

3. 输入是什么？输出是什么？
    输入：本地音频文件路径（std::string）+ 加载选项（AudioLoadOptions，含目标采样率/声道/格式/max_samples 等）
    输出：
    - AudioLoadResult（float PCM 主接口）：success/status/error_message + AudioData（pcm 数组 + 采样率 + 声道 + 格式）+ AudioInfo（源+目标参数、样本数、时长、统计信息）+ time_ms
    - AudioInt16LoadResult（int16 PCM 调试接口）：同上，PCM 为 std::vector<int16_t>
    - Probe 模式下 pcm 为空，仅返回源信息

4. 核心数据结构有哪些？
    FFmpeg C 对象（6 个）：AVFormatContext / AVStream / AVCodecContext / AVPacket / AVFrame / SwrContext
    C++ 结构体（8 个）：AudioLoadStatus（25 状态码）/ AudioSampleFormat / AudioLoadOptions / AudioStats /
    AudioInfo / AudioData / AudioLoadResult / AudioInt16LoadResult
    AudioLoader 类（PImpl 模式，9 个公开方法 + 私有 Impl）

5. 核心流程是什么？
    ① ValidatePathAndOptions 路径/参数校验
    ② OpenAndFindStream 打开容器→找音频流→填源信息（不关 fmt_ctx）
    ③ avcodec_find_decoder→alloc_context3→parameters_to_context→open2 创建并打开解码器
    ④ swr_alloc_set_opts→swr_init 创建重采样器（采样率/声道/格式三维统一）
    ⑤ av_packet_alloc + av_frame_alloc 分配缓冲区
    ⑥ while(av_read_frame) → send_packet → while(avcodec_receive_frame) → swr_convert → insert
    ⑦ flush 解码器：send_packet(nullptr) + while(receive_frame) + swr_convert
    ⑧ flush 重采样器：while(swr_convert(nullptr,0)) 吐出管线残留
    ⑨ 统一释放（frame→packet→swr→codec→fmt） + 填输出信息 + 结果判断

6. 最容易失败在哪？
    i. 6 个 FFmpeg 对象内存未及时释放导致内存泄漏
    ii. 解码循环结束后忘 flush 解码器和重采样器，丢失末尾数据
    iii. av_rescale_rnd 算输出缓冲区时忘加 swr_get_delay，缓冲区偏小
    iv. av_frame_unref 忘记调用，下次 receive 数据混乱
    v. PImpl 成功路径忘设 result.success = true，永远返回"失败"
    vi. ConvertInt16ToFloat 的 static 关键字在 .cpp 定义时重复，GCC 报错

7. 怎么验证？
    25 个单元测试（ctest 全绿）覆盖：
    - Probe：7 个错误场景 + 1 个成功验证（源信息完整、PCM 为空）
    - LoadInt16FromFile：6 个错误 + 6 个成功验证（样本数/目标参数/时长/源信息/max_samples/自定义采样率）
    - LoadFromFile：float 范围检查 + int16 一致性对比
    - ConvertInt16ToFloat：公式精度验证（0→0.0、16384→0.5、-32768→-1.0）
    - LoadBatch：正常批量 + 失败项下标一致性
    - StatusToString/IsSupportedFormat/GetSupportedExtensions 全覆盖
    example 程序输出：源信息 / int16 统计（min/max/RMS/零样本占比/前 10 样本）/
    float 统计（min/max/RMS/NaN/前 10 样本）/ 样本数对比 / LoadBatch 批量验证

8. 后续哪个模块会调用它？
    AudioFramer 音频分帧模块——需要本模块输出的 16kHz / mono / float PCM；
    AudioPreprocessor 音频预处理模块——复用 float PCM 做预加重/归一化/VAD；
    MelFeatureExtractor 梅尔特征提取——依赖固定采样率的 PCM 做 FFT 和 Mel 滤波器组计算

9. 面试 2 分钟怎么讲？
    我在数字人项目里实现的音频加载模块，是音频处理链路的统一入口。
    上游是用户提供的 wav、mp3、mp4、aac、m4a、flac 等任意格式的音频文件，
    下游是 AudioFramer、MelFeatureExtractor 和 ncnn/Wav2Lip 模型推理，它们要求
    固定 16kHz / mono / float PCM。我的工作就是在中间把各种格式统一成标准格式。

    具体流程：avformat_open_input 打开容器 → av_find_best_stream 定位音频流 →
    根据流编码参数创建解码器 avcodec_open2 → 用 swr_alloc_set_opts 配置重采样器，
    把采样率/声道/格式三维统一到 16kHz/mono/int16 →
    主循环 av_read_frame→send_packet→while(receive_frame)→swr_convert 逐帧转换 →
    flush 解码器和重采样器吐出管线残留 → 释放全部资源。

    工程上我用 PImpl 模式把 FFmpeg 头文件全藏在 .cpp 里，外部只需 LoadFromFile 一行。
    错误处理设计了 25 个状态码，从路径为空到 PCM 含 NaN 都有明确区分。
    同时输出 int16 和 float 两种 PCM：int16 用于和 FFmpeg 原生输出对照验证，
    float 用于正式算法链路。测试上写了 25 个单元测试覆盖所有可达路径，
    还有 example 程序打印源信息/统计/样本对比做肉眼验证。

---

## 待深入学习：FFmpeg 解码原理

> 记录于音频加载模块完成时。

当前对 FFmpeg 的理解停留在"知道 6 个对象分别干什么、怎么串起来、怎么调 API"的工程层面。
但 FFmpeg 内部的具体原理—— MP3/AAC 解码器如何从频域重建时域 PCM、
心理声学模型如何决定丢弃哪些频率分量、SwrContext 的 FIR 多相滤波器如何做采样率转换、
packet 和 frame 在不同编码器下为何不是 1:1 ——才是这个模块真正需要吃透的核心知识。

计划在后续音频模块（分帧/Mel/预处理）的开发中，随着对这些概念的反复接触逐步深入：
- MP3/AAC 编码原理：时域→频域变换、量化、心理声学模型、比特率控制
- 重采样数学：奈奎斯特采样定理、抗混叠滤波、多相滤波器组
- FFmpeg 内部架构：libavformat/libavcodec/libswresample 的模块划分和交互

这部分原理理解到位后，面试时可以从"会调 FFmpeg API"升级到"理解 FFmpeg 内部如何工作"。
