# AudioLoader 音频加载模块文档

## 一句话定位

音频处理链路的统一输入入口。把 wav / mp3 / aac / m4a / mp4 / flac 等格式的音频文件，
统一转换成 16kHz / mono / float PCM，供后续 AudioFramer / MelFeatureExtractor / 模型推理使用。

---

## 数据流（从用户视角）

```
用户提供文件（任意格式/采样率/声道）
         │
         ▼
     AudioLoader
    ┌────────────────────────────────────────────┐
    │  1. 解封装 (demux)    → 从容器拆出音频流     │
    │  2. 解码 (decode)     → 压缩包 → 原始 PCM    │
    │  3. 重采样 (resample)  → 任意采样率 → 16kHz   │
    │  4. 声道转换           → stereo/5.1 → mono   │
    │  5. 格式转换           → int16 → float       │
    └────────────────────────────────────────────┘
         │
         ▼
  16kHz / mono / float PCM 数组 + AudioInfo
         │
         ▼
     AudioFramer / MelFeatureExtractor
```

---

## 核心接口

| 方法 | 用途 | 返回 |
|------|------|------|
| `LoadFromFile` | **主接口**：加载为 float PCM | `AudioLoadResult` |
| `LoadInt16FromFile` | 加载为 int16 PCM（调试/WAV对照） | `AudioInt16LoadResult` |
| `Probe` | 只读源信息，不解码 PCM | `AudioLoadResult`（pcm 为空） |
| `LoadBatch` | 批量加载，失败不跳过 | `vector<AudioLoadResult>` |
| `ConvertInt16ToFloat` | int16 → float 转换 | `vector<float>` |

### 使用示例

```cpp
#include "audio/audio_loader.h"
using namespace digital_human::audio;

AudioLoader loader;  // 默认 16000 Hz
auto result = loader.LoadFromFile("speech.wav");
if (result.success) {
    // result.audio.pcm   → std::vector<float>，范围 [-1, 1]
    // result.audio.sample_rate → 16000
    // result.info.source_sample_rate → 原始采样率（如 44100）
}
```

---

## FFmpeg 对象关系

```
AVFormatContext（容器上下文）
  → AVStream（音频流描述）
    → AVCodecContext（解码器实例）
      ← AVPacket（压缩包）→ 解码 → AVFrame（PCM 帧）
                                      → SwrContext（重采样/声道/格式统一）
                                        → std::vector<int16_t>
                                          → std::vector<float>
```

| 对象 | 职责 | 释放函数 |
|------|------|---------|
| `AVFormatContext` | 打开媒体文件，管理全部流 | `avformat_close_input` |
| `AVStream` | 描述一个流的编码参数 | 随 `AVFormatContext` 释放 |
| `AVCodecContext` | 解码器工作状态 | `avcodec_free_context` |
| `AVPacket` | 承载压缩数据 | `av_packet_free`（每次用完 `av_packet_unref`） |
| `AVFrame` | 承载解码后 PCM | `av_frame_free`（每次用完 `av_frame_unref`） |
| `SwrContext` | 重采样/声道/格式转换 | `swr_free` |

---

## 内部实现架构（PImpl + helper）

```
公开类 AudioLoader（头文件不暴露 FFmpeg）
  → pImpl_ → Impl::Probe / DecodeToInt16 / LoadFromFile / ...

匿名空间 helpers：
  ValidatePathAndOptions   — 路径 / 参数合法性校验
  FillSourceInfo           — 从 FFmpeg 对象填 AudioInfo 源字段
  OpenAndFindStream        — 打开文件 + 找音频流 + 填源信息（不关闭）
```

---

## 状态码与错误处理

25 个状态码覆盖从"路径为空"到"PCM 含 NaN"的全链路失败点。
所有公开方法返回结果结构体（含 `success` / `status` / `error_message`），不抛异常。

---

## 采样率转换公式

```
输出样本数 = (swr_get_delay + frame->nb_samples) × 目标采样率 / 源采样率（向上取整）
```

`SwrContext` 的 FIR 滤波器有管线延迟，`swr_get_delay` 记录上一帧卡在滤波器中的残留样本数。

---

## int16 ↔ float 转换

```
float = int16 / 32768.0f
int16 = float × 32768.0f（然后 clamp）
```

除数是 32768（不是 32767），保证 `-32768 → -1.0` 正好映射。

---

## 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| 漏 `#include <libswresample/swresample.h>` | SwrContext 来自 libswresample | 加上 |
| `av_packet_unref` 忘记调用 | 长音频内存持续增长 | 循环内每次用完即 unref |
| `av_frame_unref` 忘记调用 | 下次 receive 数据混乱 | receive 后立即 unref |
| flush 解码器/重采样器遗漏 | 丢失末尾几十毫秒音频 | 循环后 send_packet(nullptr) + swr_convert(nullptr,0) |
| 成功路径忘设 `result.success = true` | 结构体默认 false | 返回前显式赋值 |
| `av_rescale_rnd` 忘加 `swr_get_delay` | 输出缓冲区偏小 | 加上 delay |

---

## 预留：分块加载（任务 9）

当前一次性加载全部 PCM 到 `vector`，适合短音频。后续长音频 / 实时流可扩展：

```cpp
using AudioChunkCallback = std::function<bool(const std::vector<float>&, double pts_ms)>;
AudioLoadStatus DecodeFileByChunks(file, options, callback);
```

解码循环内每攒够一个 chunk 就回调一次，`pts_ms = total_samples / sample_rate × 1000`。
chunk 边界和分帧重叠属于 AudioFramer，不混入本模块。

---

## 面试 2 分钟表述

> 我在数字人项目里实现的音频加载模块，是音频处理链路的统一入口。它的上游是用户提供的 wav、mp3、mp4、aac、m4a、flac 等格式的音频文件——这些文件来自不同来源，采样率、声道数、编码格式各不相同。下游是 AudioFramer 音频分帧、MelFeatureExtractor 梅尔特征提取、以及 ncnn/Wav2Lip 模型推理，它们要求固定 16kHz / mono / float PCM 作为输入。
>
> 我的工作就是在这个中间环节，把各种格式的音频统一成标准格式。具体流程：先 avformat_open_input 打开媒体容器、av_find_best_stream 定位音频流，然后根据流里的编码参数 avcodec_find_decoder 找到对应解码器、avcodec_open2 打开解码器。同时用 swr_alloc_set_opts 配置一个重采样转换器——把源音频的采样率、声道、采样格式三个维度统一到 16kHz / mono / int16，再转成后续算法需要的 float PCM。主循环里 av_read_frame 逐包读取 → avcodec_send_packet 送入解码器 → while(avcodec_receive_frame) 取出解码帧 → swr_convert 重采样转换 → 存入结果数组。循环结束后还要 flush 解码器和重采样器，把管线里残留的末尾数据吐干净，否则会丢最后几十毫秒音频。
>
> 工程上我用 PImpl 模式把 FFmpeg 的头文件全部隐藏在 .cpp 里，外部使用者只需要 include 一个头文件和调用 LoadFromFile 一行代码就能拿到标准 PCM，不需要关心 FFmpeg。错误处理上我没有只返回 bool，而是设计了 25 个状态码，从"路径为空"到"PCM 含 NaN"，每个失败点都有明确的状态码和错误描述，调试和测试都能立刻定位。同时输出了 int16 和 float 两种 PCM——int16 用于和 FFmpeg 原生输出逐字节对照验证，float 用于正式算法链路。
>
> 测试上写了 25 个单元测试，覆盖了 Probe 的错误路径和成功验证、LoadInt16FromFile 的完整解码链路和输出参数校验、max_samples 限制、LoadFromFile 的 float 范围检查、LoadBatch 批量加载失败项下标一致性、ConvertInt16ToFloat 的转换精度。另外还写了一个 example 程序，跑真实 wav 文件，打出源信息、int16 和 float 各自的样本数/min/max/RMS/零样本占比/前 10 个采样点，肉眼确认解码结果合理。还写了完整的模块文档，包括 FFmpeg 对象关系图、常见错误排查表、采样率转换公式和分块加载的设计思路。
