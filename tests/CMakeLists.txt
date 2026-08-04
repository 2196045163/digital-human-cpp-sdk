# gtest_discover_tests 来自 CMake 的 GoogleTest 模块；只 find_package(GTest) 不会自动导入该命令。
include(GoogleTest)

# 图像加载模块单元测试
add_executable(image_loader_unit_test core/image_loader_unit_test.cpp)
target_link_libraries(image_loader_unit_test PRIVATE digital_human_core)
add_test(NAME image_loader_unit_test COMMAND image_loader_unit_test)

# 人脸检测模块单元测试
add_executable(face_detector_unit_test core/face_detector_unit_test.cpp)
target_link_libraries(face_detector_unit_test PRIVATE digital_human_core)
add_test(NAME face_detector_unit_test COMMAND face_detector_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸对齐模块单元测试
add_executable(face_aligner_unit_test core/face_aligner_unit_test.cpp)
target_link_libraries(face_aligner_unit_test PRIVATE digital_human_core)
add_test(NAME face_aligner_unit_test COMMAND face_aligner_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸对齐图的像素预处理模块单元测试
add_executable(image_preprocessor_unit_test core/image_preprocessor_unit_test.cpp)
target_link_libraries(image_preprocessor_unit_test PRIVATE digital_human_core)
add_test(NAME image_preprocessor_unit_test COMMAND image_preprocessor_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸掩码生成模块单元测试
add_executable(face_mask_generator_unit_test core/face_mask_generator_unit_test.cpp)
target_link_libraries(face_mask_generator_unit_test PRIVATE digital_human_core)
add_test(NAME face_mask_generator_unit_test COMMAND face_mask_generator_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸融合模块单元测试
add_executable(face_blender_unit_test core/face_blender_unit_test.cpp)
target_link_libraries(face_blender_unit_test PRIVATE digital_human_core)
add_test(NAME face_blender_unit_test COMMAND face_blender_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 音频加载模块单元测试
add_executable(audio_loader_unit_test audio/audio_loader_unit_test.cpp)
target_link_libraries(audio_loader_unit_test PRIVATE digital_human_core)
add_test(NAME audio_loader_unit_test COMMAND audio_loader_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 音频分帧模块单元测试
add_executable(audio_framer_unit_test audio/audio_framer_unit_test.cpp)
target_link_libraries(audio_framer_unit_test PRIVATE digital_human_core)
add_test(NAME audio_framer_unit_test COMMAND audio_framer_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 梅尔频谱提取模块单元测试
add_executable(audio_mel_feature_extract_unit_test audio/audio_mel_feature_extract_unit_test.cpp)
target_link_libraries(audio_mel_feature_extract_unit_test PRIVATE digital_human_core)
add_test(NAME audio_mel_feature_extract_unit_test COMMAND audio_mel_feature_extract_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 音频预处理模块单元测试
add_executable(audio_preprocessor_unit_test audio/audio_preprocessor_unit_test.cpp)
target_link_libraries(audio_preprocessor_unit_test PRIVATE digital_human_core)
add_test(NAME audio_preprocessor_unit_test COMMAND audio_preprocessor_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 音频流缓冲模块单元测试
add_executable(audio_stream_buffer_unit_test audio/audio_stream_buffer_unit_test.cpp)
target_link_libraries(audio_stream_buffer_unit_test PRIVATE digital_human_core)
add_test(NAME audio_stream_buffer_unit_test COMMAND audio_stream_buffer_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 帧调度模块单元测试 (GTest)
add_executable(frame_scheduler_unit_test core/frame_scheduler_unit_test.cpp)
target_link_libraries(frame_scheduler_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(frame_scheduler_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "frame_scheduler.")

# 帧调度窄集成测试 (GTest, TimestampManager + FaceBlender + FrameScheduler)
add_executable(frame_scheduler_integration_test core/frame_scheduler_integration_test.cpp)
target_link_libraries(frame_scheduler_integration_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(frame_scheduler_integration_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "frame_scheduler.")

# 时间戳管理模块单元测试 (GTest)
add_executable(timestamp_manager_unit_test core/timestamp_manager_unit_test.cpp)
target_link_libraries(timestamp_manager_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(timestamp_manager_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "timestamp_manager.")

# PortAudio callback/playback contract tests (hardware-independent paths only)
add_executable(portaudio_playback_unit_test
    sync/portaudio_playback_unit_test.cpp)
target_link_libraries(portaudio_playback_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(portaudio_playback_unit_test
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(portaudio_playback_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "portaudio_playback.")

# Audio clock -> FrameScheduler coordinator tests
add_executable(audio_video_synchronous_unit_test
    sync/audio_video_synchronous_unit_test.cpp)
target_link_libraries(audio_video_synchronous_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(audio_video_synchronous_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "audio_video_synchronous.")

# 模型加载模块单元测试 (GTest)
add_executable(model_loader_unit_test model/model_loader_unit_test.cpp)
target_link_libraries(model_loader_unit_test PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(model_loader_unit_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 模型输入构建器单元测试 (GTest)
add_executable(model_input_unit_test model/model_input_unit_test.cpp)
target_link_libraries(model_input_unit_test PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(model_input_unit_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ncnn 适配器单元测试 (GTest)
add_executable(ncnn_input_adapter_unit_test model/ncnn_input_adapter_unit_test.cpp)
target_link_libraries(ncnn_input_adapter_unit_test PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(ncnn_input_adapter_unit_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 无状态单样本模型推理单元测试 (GTest)
add_executable(model_inference_unit_test model/model_inference_unit_test.cpp)
target_link_libraries(model_inference_unit_test PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(model_inference_unit_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(model_inference_unit_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 模型输出处理单元测试 (GTest)
add_executable(output_processor_unit_test model/output_processor_unit_test.cpp)
target_link_libraries(output_processor_unit_test PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(output_processor_unit_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(output_processor_unit_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Scheduler 前置：不含 ncnn 的有界 worker pool 并发语义测试 (GTest)
add_executable(inference_scheduler_unit_test model/inference_scheduler_unit_test.cpp)
target_link_libraries(inference_scheduler_unit_test PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(inference_scheduler_unit_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(inference_scheduler_unit_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Wav2Lip 全链路 smoke test (GTest, 与 example 同名但不同 target)
add_executable(wav2lip_model_smoke_ctest model/wav2lip_model_smoke_test.cpp)
target_link_libraries(wav2lip_model_smoke_ctest PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(wav2lip_model_smoke_ctest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Golden 上游集成测试 (GTest, 第 4 层证据)
add_executable(golden_upstream_integration_ctest model/golden_upstream_integration_test.cpp)
target_link_libraries(golden_upstream_integration_ctest PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(golden_upstream_integration_ctest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 模型视觉质量诊断：同一真实素材的输入几何 / Mel 配置受控变量实验
add_executable(model_visual_quality_diagnostic_ctest model/model_visual_quality_diagnostic_test.cpp)
target_link_libraries(model_visual_quality_diagnostic_ctest PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(model_visual_quality_diagnostic_ctest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ============================================================================
# Pipeline 测试
# ============================================================================

# Pipeline 有界队列单元测试
add_executable(bounded_task_queue_unit_test
    pipeline/bounded_task_queue_unit_test.cpp)
target_link_libraries(bounded_task_queue_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(bounded_task_queue_unit_test
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(bounded_task_queue_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "pipeline.queue.")

# Pipeline 单元测试（确定性测试）
add_executable(digital_human_pipeline_unit_test
    pipeline/digital_human_pipeline_unit_test.cpp)
target_link_libraries(digital_human_pipeline_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(digital_human_pipeline_unit_test
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(digital_human_pipeline_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "pipeline.unit."
    PROPERTIES TIMEOUT 5)

# Pipeline 生命周期测试
add_executable(digital_human_pipeline_lifecycle_test
    pipeline/digital_human_pipeline_lifecycle_test.cpp)
target_link_libraries(digital_human_pipeline_lifecycle_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(digital_human_pipeline_lifecycle_test
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(digital_human_pipeline_lifecycle_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "pipeline.lifecycle."
    PROPERTIES TIMEOUT 20)

# Pipeline Golden E2E 测试（真实数据端到端）
add_executable(digital_human_pipeline_golden_test
    pipeline/digital_human_pipeline_golden_test.cpp)
target_link_libraries(digital_human_pipeline_golden_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(digital_human_pipeline_golden_test
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
# TIMEOUT 480 的原因：本套件含 TwentyRealSessionsLoop（20 轮真实 ncnn 推理 +
# 编码），ASAN 构建下执行时间约为普通 Debug 的数倍，120s 在 ASAN 下实测
# 出现过超时；480s 给 sanitizer 构建留足余量，同时仍能暴露挂起/死锁。
gtest_discover_tests(digital_human_pipeline_golden_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "pipeline.golden."
    PROPERTIES TIMEOUT 480)

# ============================================================================
# FinalMediaWriter 测试
# ============================================================================

# Unit tests: synthetic frames/PCM, timebase, PTS, flush, error codes
add_executable(final_media_writer_unit_test
    output/final_media_writer_unit_test.cpp)
target_link_libraries(final_media_writer_unit_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(final_media_writer_unit_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "final_media_writer.unit."
    PROPERTIES TIMEOUT 30)

# Integration tests: real audio + 75 frames, ffprobe dual stream, frame count, duration
add_executable(final_media_writer_integration_test
    output/final_media_writer_integration_test.cpp)
target_link_libraries(final_media_writer_integration_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(final_media_writer_integration_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "final_media_writer.integration."
    PROPERTIES TIMEOUT 60)

# Failure injection tests: invalid path, mid-failure, cancellation, idempotence
add_executable(final_media_writer_failure_test
    output/final_media_writer_failure_test.cpp)
target_link_libraries(final_media_writer_failure_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
gtest_discover_tests(final_media_writer_failure_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "final_media_writer.failure."
    PROPERTIES TIMEOUT 60)

# ============================================================================
# M03 CLI 契约测试
# ============================================================================

# CLI contract tests: --help, unknown args, missing inputs, realtime rejection,
# integration smoke with golden data, JSON schema validation.
add_executable(digital_human_cli_contract_test
    cli/digital_human_cli_contract_test.cpp)
target_link_libraries(digital_human_cli_contract_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_compile_definitions(digital_human_cli_contract_test PRIVATE
    CLI_BINARY_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# TIMEOUT 360 的原因：CLI 契约测试以子进程方式多次启动 digital_human_app，
# 每次都是完整推理 + 编码 + MP4 落盘（单次在 ASAN 下可能接近 2 分钟），
# 多个用例叠加后单测超时需要放宽到 6 分钟。
gtest_discover_tests(digital_human_cli_contract_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "cli.contract."
    PROPERTIES TIMEOUT 360)

# CLI 失败注入测试：以 Fake MediaWriter（实现 output::MediaWriter 接口）注入
# RunCli，在真实 Pipeline 跑完的前提下验证四条件成功校验的逐项失败路径：
# 编码/mux/flush/trailer 失败、未 Finalize、输出文件缺失/空文件。
# 复用 digital_human_app.cpp（以 DIGITAL_HUMAN_APP_NO_MAIN 宏编译，仅取 RunCli）。
add_executable(cli_failure_injection_test
    cli/cli_failure_injection_test.cpp
    ${PROJECT_SOURCE_DIR}/apps/digital_human_app.cpp)
target_link_libraries(cli_failure_injection_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_include_directories(cli_failure_injection_test
    PRIVATE ${PROJECT_SOURCE_DIR}/apps)
target_compile_definitions(cli_failure_injection_test PRIVATE
    DIGITAL_HUMAN_APP_NO_MAIN=1)
# TIMEOUT 300 的原因：每个用例都完整跑一遍 Golden 推理 + 编码（Release 下
# 单次约 5-10s，Debug 下实测约 9s），7 个用例叠加留足余量，同时仍能暴露挂起。
gtest_discover_tests(cli_failure_injection_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "cli.failure_injection."
    PROPERTIES TIMEOUT 300)

# ============================================================================
# M04 E2E 验收测试
# ============================================================================

# E2E acceptance tests: full user path CLI → MP4 with Golden inputs,
# ffprobe validation, artifact consistency, failure injection, stability.
add_executable(digital_human_e2e_acceptance_test
    e2e/digital_human_e2e_acceptance_test.cpp)
target_link_libraries(digital_human_e2e_acceptance_test
    PRIVATE digital_human_core GTest::GTest GTest::Main)
target_compile_definitions(digital_human_e2e_acceptance_test PRIVATE
    CLI_BINARY_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# TIMEOUT 600 的原因：E2E 验收套件包含 10+ 个用例，多数都要启动完整 CLI
# （推理 + 编码 + MP4），另有稳定性测试要连续跑 3 轮；ASAN 下全套耗时
# 显著拉长，300s 实测不足，600s 保证稳定通过且仍能暴露挂起。
gtest_discover_tests(digital_human_e2e_acceptance_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TEST_PREFIX "e2e.acceptance."
    PROPERTIES TIMEOUT 600)

# add_test(NAME audio_preprocessor_unit_test_dup COMMAND dummy_placeholder_test
#          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
