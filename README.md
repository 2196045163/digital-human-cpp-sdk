# Digital Human SDK（C++）

这是一个基于 C++ 的语音驱动人脸动画推理 SDK，核心模型为 Wav2Lip。SDK 输入一张静态人脸图片和一段音频，通过完全离线的处理流程生成口型与语音同步的 MP4 视频。

## 架构概览

项目按照分层的 C++17 库进行组织，目录结构如下：

```text
include/          公共 API 头文件（8 个模块，共 28 个头文件）
src/              核心实现（24 个 C++ 源文件）
apps/             CLI 应用程序入口
tests/            单元测试（GoogleTest，共 21 个测试文件）
docs/             架构与设计文档
```

更详细的 Pipeline 流程图请参见 [docs/architecture.md](docs/architecture.md)。

## 核心模块

| 模块 | 目录 | 功能 |
|---|---|---|
| **Audio** | `include/audio/`, `src/audio/` | 音频加载（FFmpeg）、重采样到 16 kHz、PCM 分帧、Mel 频谱提取 |
| **Core** | `include/core/`, `src/core/` | 人脸检测（dlib HOG + SVM）、dlib 68 点关键点定位、人脸对齐、图像 I/O、Mask 生成、嘴部区域恢复与融合 |
| **Model** | `include/model/`, `src/model/` | ncnn 模型加载、Wav2Lip 输入构造、批量推理调度、输出后处理 |
| **Pipeline** | `include/pipeline/`, `src/pipeline/` | 端到端流程调度、有界任务队列、多线程生产者-消费者 Pipeline |
| **Output** | `include/output/`, `src/output/` | 基于 FFmpeg 的 H.264/AAC 编码与 MP4 封装（FinalMediaWriter） |
| **Sync** | `include/sync/`, `src/sync/` | 音视频同步、PortAudio 播放、音频主时钟 |

## 完整处理流程

1. **音频解码与重采样** —— 将输入音频（FFmpeg 支持的格式）解码并统一重采样为 16 kHz 单声道 PCM。
2. **Mel 频谱提取** —— 使用与 Wav2Lip 兼容的参数（`n_fft=800`、`fmin=55 Hz`、`fmax=7600 Hz`）计算 80 维 Mel 频谱，并切分为连续 16 个时间步的 Mel chunk。
3. **人脸检测与关键点定位** —— 使用 dlib HOG + SVM 正面人脸检测器定位人脸，并通过 dlib `shape_predictor` 提取 68 点人脸关键点。
4. **图像预处理** —— 按人脸检测框直接裁剪并在底部保留 10 像素 padding，再缩放为 `96×96`；对下半脸区域施加 Mask，用于构造 Wav2Lip 所需的六通道人脸输入。OpenCV 用于图像加载、缩放、颜色转换和矩阵运算。
5. **ncnn Wav2Lip 推理** —— 将六通道人脸输入与 Mel chunk 一同送入 ncnn 推理引擎。
6. **嘴部恢复与图像融合** —— 将生成的 `96×96` 人脸结果恢复到原始图像坐标，并在 Mask 区域内进行局部颜色匹配、Alpha 融合，同时可选地恢复原图中的轻量高频细节。
7. **FFmpeg H.264/AAC 封装** —— 将视频帧编码为 H.264、音频编码为 AAC，并封装为 MP4 文件。

## 技术栈

| 组件 | 用途 |
|---|---|
| C++17 | 核心开发语言 |
| CMake 3.16+ | 构建系统 |
| [ncnn](https://github.com/Tencent/ncnn) | 神经网络推理 |
| [FFmpeg](https://ffmpeg.org/) | 音频解码、音视频编码、MP4 封装 |
| [OpenCV](https://opencv.org/) | 图像加载、缩放、颜色转换、矩阵运算、Mask 生成、嘴部融合 |
| [dlib](http://dlib.net/) | 人脸检测（HOG + SVM）和 68 点关键点定位（shape_predictor） |
| [PortAudio](http://www.portaudio.com/) | 音频播放（实验性组件） |
| [GoogleTest](https://github.com/google/googletest) | 单元测试 |
| OpenMP | CPU 并行计算 |

## 支持的输入 / 输出格式

**输入：**

- 图像：JPEG、PNG、BMP（单张静态人脸图片，推荐 `512×512`）
- 音频：WAV、MP3、AAC、M4A、MP4、FLAC（通过文件扩展名判断格式，由 FFmpeg 完成解码）

**输出：**

- 容器：MP4
- 视频：H.264，25 FPS
- 音频：AAC，16 kHz 单声道
- 分辨率：可配置，默认与输入图片尺寸一致

## 构建依赖

以下为 Ubuntu 22.04 下所需的系统依赖示例：

```bash
sudo apt install -y \
    build-essential cmake pkg-config \
    libopencv-dev \
    libdlib-dev \
    portaudio19-dev \
    libavformat-dev libavcodec-dev libavutil-dev \
    libswresample-dev libswscale-dev \
    libgtest-dev
```

ncnn 需要从源码构建或通过包管理器安装：

```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) && sudo make install
```

如果需要指定自定义 ncnn 安装路径：

```bash
cmake .. -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
# 或：
cmake .. -DCMAKE_PREFIX_PATH=/path/to/ncnn/install
```

## 模型文件

**本仓库不提供模型权重。** 以下文件需要由用户自行获取：

| 文件 | 说明 | 来源 |
|---|---|---|
| `wav2lip.param` / `wav2lip.bin` | ncnn 格式的 Wav2Lip 模型 | 由原始 Wav2Lip PyTorch 权重转换 |
| `shape_predictor_68_face_landmarks.dat` | dlib 68 点人脸关键点模型 | [dlib 模型仓库](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |

关于 Wav2Lip 模型的许可信息，请参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 基础构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

构建选项：

- `BUILD_TESTS=ON/OFF` —— 启用 / 禁用单元测试，默认开启
- `BUILD_EXAMPLES=ON/OFF` —— 启用 / 禁用示例程序，默认开启
- `USE_MARCH_NATIVE=ON/OFF` —— 为当前主机启用 `-march=native` 指令集优化，默认关闭
- `ENABLE_ASAN=ON` —— 启用 AddressSanitizer
- `ENABLE_TSAN=ON` —— 启用 ThreadSanitizer

## CLI 使用方法

CLI 当前仅支持离线模式。实时模式会被明确拒绝，并返回退出码 `3`。

```bash
./build/bin/digital_human_app \
    --image        path/to/face.jpg \
    --audio        path/to/speech.wav \
    --model-param  path/to/wav2lip.param \
    --model-bin    path/to/wav2lip.bin \
    --landmark     path/to/shape_predictor_68_face_landmarks.dat \
    --output       output.mp4 \
    --fps 25
```

CLI 完成后会检查以下四个成功条件：

1. Pipeline 正常完成且没有错误；
2. Media Writer 完成 finalize；
3. 没有记录 Writer 错误；
4. 输出文件存在且非空。

执行成功后，程序会向标准输出打印包含 Pipeline 统计信息的 JSON，并以退出码 `0` 结束。

## 性能

### 正式测试方法

当前正式性能对比统一使用 `512×512`、30 FPS 和固定真实输入。模型权重与测试素材不包含在公开仓库快照中。

| 项目 | 配置 |
|---|---|
| OS | Ubuntu 26.04 LTS / WSL2 |
| CPU | AMD Ryzen 5 9600X 6-Core Processor |
| CPU 核心 | 6 physical cores / 12 logical CPUs |
| Compiler | GNU 15.2.0 |
| Build Type | Release |
| Backend | ncnn CPU |
| FFmpeg | 4.4.2 |
| 输入 | 固定真实 `testdata/input/face.jpg` + `testdata/input/real_voice_test.wav` |
| 输出 | `512×512`、30 FPS、1816 帧 |
| 测试轮数 | 每组 5 轮 |

每轮 benchmark 使用独立子进程，重新创建 Pipeline/runtime、重新加载模型，并完整执行图像与音频读取、1816 帧生成、MP4 编码及收尾。端到端耗时覆盖一次完整任务开始到 MP4 写完；模型推理耗时只累计 Wav2Lip 模型前向推理，不包含人脸检测、Mel 处理、融合和视频编码。

Peak RSS 使用 Linux/WSL 的 `getrusage(RUSAGE_SELF).ru_maxrss` 测量，原始 KB 值除以 1024 换算为 MB。每轮独立子进程统计自己的 Peak RSS，该指标表示整个进程的峰值常驻内存。

正式测试命令：

```bash
./build/bin/cpu_benchmark --runs 5 --fps 30 --threads 1
./build/bin/cpu_benchmark --runs 5 --fps 30 --threads 6
```

### CPU 正式结果

两组测试均为 5/5 成功，每轮 `total_frames=1816`、`inference_frames=1816`。

| Metric | 1 Thread | 6 Threads | Change |
|---|---:|---:|---:|
| End-to-end time | 64.475 s | 36.703 s | -43.1% |
| Avg model inference time | 33.231 ms/frame | 18.089 ms/frame | -45.6% |
| Model inference throughput | 30.227 FPS | 55.316 FPS | +83.0% |
| End-to-end throughput | 28.17 FPS | 49.48 FPS | +75.7% |
| Peak RSS | 630.131 MB | 640.250 MB | +1.6% |

两组结果均为 5 轮平均，且均为 `success=5/5`。30 FPS 对应约 33.33 ms/frame：单线程平均模型推理为 33.231 ms/frame，处于实时临界线附近，完整链路约 28.17 FPS，未达到 30 FPS 端到端实时；6 线程平均模型推理降至 18.089 ms/frame，完整链路约 49.48 FPS，明显超过 30 FPS 目标。

### CPU 多线程优化范围

- 6 线程版本只调整 ncnn 模型内部推理线程数；scheduler worker 仍为 1，没有多帧并行，也没有创建多个模型实例。
- Wav2Lip 中卷积等计算密集型算子可以利用多个 CPU 核心并行执行；`ncnn_threads=6` 获得的是**单帧模型内部并行**收益。
- 加速未达到 6 倍，主要受串行部分、线程调度与同步开销，以及缓存和内存带宽限制。
- 模型推理变快后，前处理、融合和编码等非模型阶段在端到端耗时中的占比会上升。

### 当前状态

- CPU correctness baseline：完成
- Wav2Lip 真实输入预处理问题：已修复（模型音频路径设置 `enable_normalize=false`；检测框直接裁剪并保留 bottom padding=10）
- 30 FPS CPU single-thread baseline：完成
- 30 FPS CPU 6-thread optimization：完成
- CUDA GPU backend：后续工作，尚未完成

### 历史性能数据（旧口径）

以下为 README 原有的 Ubuntu 22.04、2 个逻辑 CPU 核心、25 FPS 基准，测试环境、输入和统计口径均不同，仅作为历史记录，不与当前正式 30 FPS 数据直接比较。

以下数据来自完整开发仓库中的固定输入基准测试。模型权重和测试素材未包含在当前公开仓库快照中。

| 指标 | 数值 |
|---|---|
| 输入图像 | `512×512` 静态人像 |
| 输入音频 | 8.136 秒，16 kHz 单声道 |
| 输出 | `512×512`、25 FPS、204 帧、H.264/AAC MP4 |
| 平均端到端耗时 | 16.51 秒 |
| 平均处理速度 | 12.36 FPS |
| 峰值内存占用 | 约 666 MB |

> 性能会受到 CPU 核心数量、ncnn 线程配置、输入分辨率和运行环境等因素的显著影响。当前正式对比基准为上文所列的 512×512 / 30 FPS 条件。

## 测试

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

测试集包含音频处理、核心图像操作、模型输入构造、媒体输出、Pipeline 调度和同步组件等模块的单元测试。

依赖模型权重或测试数据文件的测试未包含在公开仓库中，详见 [docs/testing.md](docs/testing.md)。

## 已知限制

- **依赖模型文件**：Pipeline 需要 ncnn 格式的 Wav2Lip 模型。本仓库不包含模型权重，需要用户自行获取。
- **仅支持静态人像**：当前实现处理单张静态人脸图片，不支持背景变化的视频输入，也不支持多人脸处理。
- **仅支持离线处理**：CLI 仅支持离线批处理。代码库包含实验性的音视频同步和 PortAudio 播放组件，但公开 CLI 会明确拒绝实时模式。
- **单一模型架构**：Pipeline 基于 Wav2Lip 的输入 / 输出契约构建。如需使用其他口型同步模型，需要修改模型规格和输入构造逻辑。
- **平台支持**：项目主要在 Linux（Ubuntu 22.04）下开发和测试。Windows 和 macOS 可能需要调整构建配置以及 FFmpeg / PortAudio 集成方式。
- **人脸检测限制**：dlib HOG + SVM 正面人脸检测器对正面人脸效果较好，但在大角度侧脸、光照较差等情况下可能漏检，也可能对非人脸区域产生误检。

## 第三方许可证

本项目集成了多个第三方组件。各组件的许可证及使用说明请参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**重要说明：** Wav2Lip 模型及其预训练权重仅允许用于个人、研究和非商业用途。用户必须遵守 Wav2Lip 原项目的许可证条款。

## 项目结构

```text
digital-human-cpp-sdk/
├── CMakeLists.txt              根构建配置
├── README.md
├── THIRD_PARTY_NOTICES.md
├── include/                    公共 API 头文件
│   ├── audio/                  音频处理 API
│   ├── core/                   图像与人脸处理 API
│   ├── model/                  模型推理 API
│   ├── output/                 媒体输出 API
│   ├── pipeline/               Pipeline 调度 API
│   ├── sync/                   同步与播放 API
│   ├── utils/                  工具组件
│   └── video/                  视频帧类型
├── src/                        核心实现
│   ├── audio/
│   ├── core/
│   ├── model/detail/           模型内部规格与辅助组件
│   ├── output/
│   ├── pipeline/detail/        Pipeline 内部组件
│   └── sync/
├── apps/                       CLI 应用程序
├── tests/                      单元测试（不依赖模型的子集）
└── docs/                       文档
    ├── architecture.md         Pipeline 架构图
    └── testing.md              测试套件说明
```

## 项目定位

这是一个基于 C++17 的离线静态人像口型同步工程示例，展示了以下完整技术链路：

- 使用 FFmpeg 完成音频解码和重采样；
- 使用 OpenCV / dlib 完成人脸与图像处理；
- 使用 ncnn 集成 Wav2Lip 神经网络推理；
- 使用 FFmpeg 完成 H.264/AAC 编码和 MP4 封装。

本仓库不分发模型权重和测试媒体文件，也不接受模型权重提交。
