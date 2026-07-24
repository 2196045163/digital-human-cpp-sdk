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

# add_test(NAME audio_preprocessor_unit_test_dup COMMAND dummy_placeholder_test
#          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
