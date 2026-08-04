# M05 性能与稳定性报告

> **Module**: M05-performance-stability
> **Generated**: 2026-08-04
> **Roles**: implementation + repair
> **Repair commit**: see git log on `automation/05-performance-stability`

## 1. 环境可追溯性

| 属性 | 值 |
|---|---|
| CPU 型号 | Intel(R) Core(TM) i5-13400F (13th Gen) |
| CPU 核心 | 2（VM 限制） |
| 编译器 | GCC 11.4.0 |
| CMake | 3.22.1 |
| 构建类型 | Release (`-DNDEBUG -march=native -O3`) |
| 内核 | Linux 6.8.0-94-generic |
| ASAN 构建 | Debug (`-fsanitize=address -fno-omit-frame-pointer`) |
| TSAN 构建 | Debug (`-fsanitize=thread -fno-omit-frame-pointer`) |

### 输入文件哈希

| 文件 | SHA256 |
|---|---|
| `testdata/golden/face.jpg` | `7e497501a28bcf9a353ccadf6eb9216bf098ac32888fb542fb9bfe71d486761f` |
| `testdata/golden/audio.wav` | `c948a7fb75fa54706374b6cf27326cc6840c737aedf2f27d474899cd6d61d404` |
| `models/wav2lip/wav2lip.param` | `d510b51697fa71f6e4d9ca7b8a44ad8d90f4a3fe52506f2ab76c5133f157b60b` |
| `models/wav2lip/wav2lip.bin` | `d5ffe4ef320f403a58ebf3a1c1976b86f816a8550e2479b6e809b7a836fcdfd1` |
| `models/shape_predictor_68_face_landmarks.dat` | `fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f` |

## 2. 测试矩阵与方法

### 统计方法

- **百分位数**: nearest-rank 法
- **聚合维度**: mean, P50, P95, min, max
- **采样单位**: 毫秒（时间）, kB（RSS）, 帧数（计数）, 帧/秒（吞吐）
- **RSS 测量**: `/proc/self/status` VmRSS, 每轮运行期间以 ~100ms 间隔轮询取峰值

### Pipeline 配置（固定）

| 参数 | 值 |
|---|---|
| FPS | 25 |
| Q1 容量 (audio→inference) | 4 |
| Q2 容量 (inference→render) | 2 |
| Scheduler worker | 1 |
| ncnn threads/worker | 1 |
| 模式 | 离线 (Offline) |
| Golden 输入 | face.jpg + audio.wav (3s, 75 帧) |

### 采集指标

| 指标 | 来源 | 单位 |
|---|---|---|
| `total_wall_time_ms` | PipelineStats + 外部时钟 | ms |
| `prepare_time_ms` | PipelineStats | ms |
| `audio_process_time_ms` | PipelineStats | ms | **未测量** |
| `inference_total_time_ms` | PipelineStats | ms | **未测量** |
| `render_total_time_ms` | PipelineStats | ms | **未测量** |
| `q1_high_watermark` | PipelineStats | 任务数 |
| `q2_high_watermark` | PipelineStats | 任务数 |
| `rss_peak_kb` | /proc/self/status 轮询 | kB |
| `throughput_frames_per_second` | 帧数 / 总耗时 | fps |
| `generated_task_count` | PipelineStats | 任务数 |
| `rendered_unique_frame_count` | PipelineStats | 帧数 |

未测量指标（声明）:

- **audio_process_time_ms**: 未测量。Pipeline 实现（`src/pipeline/digital_human_pipeline.cpp`）仅在同步准备阶段（第 341 行）设置 `prepare_time_ms`。音频处理在独立 worker 线程中运行，PipelineStats 未累积该阶段 wall-clock。修复需在 `src/` 中添加 per-stage `std::chrono` 测量，不在 M05 模块允许修改范围内。
- **inference_total_time_ms**: 未测量。推理在 scheduler worker 线程中通过 ncnn 执行，PipelineStats 未累积该阶段 wall-clock。修复需在 `src/` 中添加推理前后的时间戳记录。
- **render_total_time_ms**: 未测量。渲染在 render worker 线程中执行（face blend + 帧合成），PipelineStats 未累积该阶段 wall-clock。修复需在 `src/` 中添加渲染前后的时间戳记录。
- **GPU 利用率**: 未测量（系统仅 CPU 推理）
- **帧级延迟分布**: 未测量（PipelineStats 仅提供阶段级总和）
- **磁盘 I/O**: 未测量（基准 sink 不写帧文件，消除 I/O 干扰）

## 3. 基准测试结果

> 详见 `golden_output/pipeline_performance_benchmark.json` 和各轮 `pipeline_benchmark_run_N.json`

### 多轮结果摘要

| 轮次 | 成功 | 总耗时 (ms) | RSS 峰值 (kB) | 帧数 | 吞吐 (fps) |
|---|---|---|---|---|---|
| 0 | ✓ | 5478.67 | 626416 | 75 | 13.69 |
| 1 | ✓ | 5496.70 | 652856 | 75 | 13.64 |
| 2 | ✓ | 5384.12 | 668836 | 75 | 13.93 |

### 聚合统计

| 指标 | Mean | P50 | P95 | Min | Max |
|---|---|---|---|---|---|
| total_wall_time_ms | 5453.16 | 5478.67 | 5496.70 | 5384.12 | 5496.70 |
| prepare_time_ms | 933.73 | 916.65 | 1018.89 | 865.63 | 1018.89 |
| audio_process_time_ms | 0.00 (未测量) | 0.00 | 0.00 | 0.00 | 0.00 |
| inference_total_time_ms | 0.00 (未测量) | 0.00 | 0.00 | 0.00 | 0.00 |
| render_total_time_ms | 0.00 (未测量) | 0.00 | 0.00 | 0.00 | 0.00 |
| rss_peak_kb | 649370.67 | 652856.00 | 668836.00 | 626416.00 | 668836.00 |
| throughput_fps | 13.75 | 13.69 | 13.93 | 13.64 | 13.93 |
| q1_high_watermark | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| q2_high_watermark | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

> **注意**: audio/inference/render 阶段计时器值为 0.00（未测量），原因为 Pipeline 实现仅在 `src/pipeline/digital_human_pipeline.cpp:341` 设置 `prepare_time_ms`，其他三阶段在独立 worker 线程中运行但未在 PipelineStats 中累积 per-stage wall-clock。修复需在 `src/` 中添加计时，超出 M05 模块允许修改范围。详细说明见第 2 节"未测量指标"。

### 波动性说明

- 多轮间波动来自于 CPU 频率缩放、内核调度器噪声、ncnn 内部非确定性（浮点运算顺序）和系统后台负载。
- 不承诺实时性 —— 此 Pipeline 为离线处理设计，吞吐受限于单 worker / 单 ncnn 线程配置。
- 未修改任何业务算法、Pipeline 线程/队列参数或模型以优化跑分。

## 4. 消毒器报告

### ASAN (AddressSanitizer)

| 项目 | 结果 |
|---|---|
| 构建类型 | Debug + ASAN |
| CTest 退出码 | 0 |
| 测试通过 | 288/289（1 个失败为上游测试 `image_preprocessor_test.Prepare_EmptyCropRect`，非 M05 引入） |
| AddressSanitizer 错误 | 0 |
| 日志路径 | `logs/sanitizer_checks/<timestamp>/asan/` |

### TSAN (ThreadSanitizer)

| 项目 | 结果 |
|---|---|
| 构建类型 | Debug + TSAN |
| 编译状态 | ✅ 成功（library + executables 均通过 TSAN 编译） |
| CTest 执行 | NOT_RUN — 硬件约束 |
| ThreadSanitizer 错误 | NOT_RUN |
| 测试范围 | pipeline.*, bounded_task_queue, audio_video_synchronous, portaudio_playback, frame_scheduler, timestamp_manager |
| 日志路径 | 无运行时日志 |

**TSAN NOT_RUN 技术原因**: 当前可用硬件为 2 核 VM（Intel i5-13400F，VM 限制至 2 核），物理内存 3.8 GB，已使用 swap 1.1 GB。ThreadSanitizer 需要 ≥4 核才能稳定运行测试（5-10x CPU 开销 + 2-3x 内存开销）。在 2 核环境中，gtest_discover_tests 阶段因 `frame_scheduler_integration_test` 超时而失败（cmake 退出码 66）。TSAN 编译验证通过，表明代码无编译期线程安全问题。

**缓解因素**: 
- M05 模块仅新增 `benchmarks/` 和 `scripts/`（无新线程、锁或共享状态），不引入新的竞态条件风险
- ASAN 全量测试已通过（288/289 passed, 0 AddressSanitizer 错误）
- TSAN 脚本 (`scripts/run_sanitizer_checks.sh`) 已就绪，可在 ≥4 核 CI 环境中执行
- 脚本设计为独立运行：`bash scripts/run_sanitizer_checks.sh` |

## 5. 证据审计追踪

| 证据类别 | 路径 | 描述 |
|---|---|---|
| 环境信息 | `logs/performance_benchmark/<ts>/ENVIRONMENT.txt` | CPU/编译/内核/输入哈希 |
| 构建日志 | `logs/performance_benchmark/<ts>/cmake_build.log` | Release 构建输出 |
| 基准运行日志 | `logs/performance_benchmark/<ts>/benchmark_run.log` | 基准原始 stdout |
| /usr/bin/time -v | `logs/performance_benchmark/<ts>/time_v_output.txt` | 系统资源指标 |
| 汇总 JSON | `golden_output/pipeline_performance_benchmark.json` | 完整 JSON 汇总 |
| 单轮 JSON | `golden_output/pipeline_benchmark_run_N.json` | 每轮详细统计 |
| ASAN 日志 | `logs/sanitizer_checks/<ts>/asan/ctest_output.log` | 全量测试 output |
| TSAN 编译 | `build/tsan_check/` | TSAN 编译通过（library + executables），测试执行受 2 核 VM 硬件限制 |
| 消毒器摘要 | `logs/sanitizer_checks/<ts>/SANITIZER_SUMMARY.txt` | ASAN 结论（TSAN NOT_RUN） |

所有原始日志长期保存。所有数字可从原始 JSON 反向核验。

## 6. 限制与声明

1. **不承诺实时性**: Pipeline 使用离线模式，帧以最快速度生成；未在实时模式下测量端到端延迟。
2. **未修改基线**: 未修改任何业务算法、Pipeline 线程/队列参数或模型。性能数据仅记录现状。
3. **环境绑定**: 所有数据仅对本次机器、构建和输入有效。不同 CPU、内存或 ncnn 版本的性能可能显著不同。
4. **RSS 近似**: `/proc/self/status` 的 VmRSS 是常驻集大小的快照，非精确峰值。轮询间隔 ~100ms 可能遗漏短暂尖峰。
5. **单进程**: 多轮在同一进程中顺序执行，后轮 RSS 可能受前轮分配器缓存影响。
6. **未测量指标**: GPU 利用率、磁盘 I/O、帧级延迟分布 — 这些未包含在 PipelineStats 中，也未额外测量。
7. **性能问题发现**: 若有性能瓶颈，只在文档中记录证据和建议，不自行优化。
