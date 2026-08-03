# 1. 构建 DemoApp（基础演示）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp")
    add_executable(demo_app demo.cpp)
    target_link_libraries(demo_app PRIVATE digital_human_core)
endif()

# 2. 构建 OpenCV 测试程序（验证图像功能）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv_test.cpp")
    add_executable(opencv_test opencv_test.cpp)
    target_link_libraries(opencv_test PRIVATE digital_human_core)
endif()

# 3. 构建 ncnn 推理引擎测试程序（验证 ncnn 集成与模型推理性能）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn_test.cpp")
    add_executable(ncnn_test ncnn_test.cpp)
    target_link_libraries(ncnn_test PRIVATE digital_human_core)
endif()

# 4. 构建 FFmpeg 测试程序 (验证音频功能)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_audio_test.cpp")
    add_executable(ffmpeg_audio_test ffmpeg_audio_test.cpp)
    target_link_libraries(ffmpeg_audio_test PRIVATE digital_human_core)
endif()

# 5. 构建 dlib 测试程序 (验证人脸检测和关键点提取)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dlib_test.cpp")
    add_executable(dlib_test dlib_test.cpp)
    target_link_libraries(dlib_test PRIVATE digital_human_core)
endif()

# 6. 构建 ImageLoader 测试程序（验证图像加载模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/image_loader_test.cpp")
    add_executable(image_loader_test core/image_loader_test.cpp)
    target_link_libraries(image_loader_test PRIVATE digital_human_core)
endif()

# 7. 构建 FaceDetector 测试程序（验证人脸检测与关键点）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/face_detector_test.cpp")
    add_executable(face_detector_test core/face_detector_test.cpp)
    target_link_libraries(face_detector_test PRIVATE digital_human_core)
endif()

# 8. 构建 FaceAligner 测试程序（验证人脸对齐）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/face_aligner_test.cpp")
    add_executable(face_aligner_test core/face_aligner_test.cpp)
    target_link_libraries(face_aligner_test PRIVATE digital_human_core)
endif()

# 9. 构建 ImagePreprocessor 测试程序（验证图像预处理）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/image_preprocessor_test.cpp")
    add_executable(image_preprocessor_test core/image_preprocessor_test.cpp)
    target_link_libraries(image_preprocessor_test PRIVATE digital_human_core)
endif()

# 10. 构建 FacemaskGenerator 测试程序（验证人脸掩码生成）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/face_mask_generator_test.cpp")
    add_executable(face_mask_generator_test core/face_mask_generator_test.cpp)
    target_link_libraries(face_mask_generator_test PRIVATE digital_human_core)
endif()

# 11. 构建 FaceBlender 测试程序（验证人脸融合模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/face_blender_test.cpp")
    add_executable(face_blender_test core/face_blender_test.cpp)
    target_link_libraries(face_blender_test PRIVATE digital_human_core)
endif()

# 12. 构建 AudioLoader 测试程序（验证音频加载模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio/audio_loader_test.cpp")
    add_executable(audio_loader_test audio/audio_loader_test.cpp)
    target_link_libraries(audio_loader_test PRIVATE digital_human_core)
endif()

# 13. 构建 AudioFramer 测试程序（验证音频分帧模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio/audio_framer_test.cpp")
    add_executable(audio_framer_test audio/audio_framer_test.cpp)
    target_link_libraries(audio_framer_test PRIVATE digital_human_core)
endif()

# 14. 构建 MelFeatureExtractor 测试程序（验证梅尔频谱提取模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio/audio_mel_feature_extract_test.cpp")
    add_executable(audio_mel_feature_extract_test audio/audio_mel_feature_extract_test.cpp)
    target_link_libraries(audio_mel_feature_extract_test PRIVATE digital_human_core)
endif()

# 15. 构建 AudioPreprocessor 测试程序（验证音频预处理模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio/audio_preprocessor_test.cpp")
    add_executable(audio_preprocessor_test audio/audio_preprocessor_test.cpp)
    target_link_libraries(audio_preprocessor_test PRIVATE digital_human_core)
endif()

# 16. 构建 AudioStreamBuffer 测试程序（验证环形缓冲区）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio/audio_stream_buffer_test.cpp")
    add_executable(audio_stream_buffer_test audio/audio_stream_buffer_test.cpp)
    target_link_libraries(audio_stream_buffer_test PRIVATE digital_human_core)
endif()

# 17. 构建 AudioStreamPipeline 测试程序（验证 Loader→Buffer→Preprocessor 链路）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio/audio_stream_pipeline_test.cpp")
    add_executable(audio_stream_pipeline_test audio/audio_stream_pipeline_test.cpp)
    target_link_libraries(audio_stream_pipeline_test PRIVATE digital_human_core)
endif()

# 18. 构建 ModelLoader 测试程序（验证模型加载模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/model_loader_test.cpp")
    add_executable(model_loader_test model/model_loader_test.cpp)
    target_link_libraries(model_loader_test PRIVATE digital_human_core)
endif()

# 19. 构建 ModelInput 测试程序（验证模型输入构建 + Adapter）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/model_input_test.cpp")
    add_executable(model_input_test model/model_input_test.cpp)
    target_link_libraries(model_input_test PRIVATE digital_human_core)
endif()

# 20. 构建 Wav2Lip 全链路 smoke test（验证模型边界接通）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/wav2lip_model_smoke_test.cpp")
    add_executable(wav2lip_model_smoke_test model/wav2lip_model_smoke_test.cpp)
    target_link_libraries(wav2lip_model_smoke_test PRIVATE digital_human_core)
endif()

# 21. 构建正式 ModelInference 单样本示例（输出 Result 诊断）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/model_inference_test.cpp")
    add_executable(model_inference_test model/model_inference_test.cpp)
    target_link_libraries(model_inference_test PRIVATE digital_human_core)
endif()

# 22. 构建正式 InferenceScheduler batch benchmark（输出 P50/P95/吞吐）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/model_inference_benchmark.cpp")
    add_executable(model_inference_benchmark model/model_inference_benchmark.cpp)
    target_link_libraries(model_inference_benchmark PRIVATE digital_human_core)
endif()

# 23. 构建 OutputProcessor 合成示例（输出 BGR 图像和转换诊断）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/output_processor_test.cpp")
    add_executable(output_processor_test model/output_processor_test.cpp)
    target_link_libraries(output_processor_test PRIVATE digital_human_core)
    target_include_directories(output_processor_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

# 24. 构建时间戳示例（音频/视频索引 → 微秒时间轴 + golden 文本）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/timestamp_manager_test.cpp")
    add_executable(timestamp_manager_test core/timestamp_manager_test.cpp)
    target_link_libraries(timestamp_manager_test PRIVATE digital_human_core)
endif()

# 25. 构建帧调度示例（合成帧 + 确定性 trace）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/frame_scheduler_test.cpp")
    add_executable(frame_scheduler_test core/frame_scheduler_test.cpp)
    target_link_libraries(frame_scheduler_test PRIVATE digital_human_core)
endif()

# 26. 音频主时钟 + FrameScheduler 确定性 trace；可选 PortAudio 设备 smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sync/audio_video_synchronous_test.cpp")
    add_executable(audio_video_synchronous_test
        sync/audio_video_synchronous_test.cpp)
    target_link_libraries(audio_video_synchronous_test PRIVATE digital_human_core)
endif()

# 27. Pipeline Demo（完整 Pipeline 示例）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/digital_human_pipeline_demo.cpp")
    add_executable(digital_human_pipeline_demo
        pipeline/digital_human_pipeline_demo.cpp)
    target_link_libraries(digital_human_pipeline_demo PRIVATE digital_human_core)
endif()

message(STATUS "Configured example targets: demo_app, opencv_test, "
        "ncnn_test, ffmpeg_audio_test, dlib_test, image_loader_test, "
        "face_detector_test, face_aligner_test, image_preprocessor_test, "
        "face_mask_generator_test, face_blender_test, audio_loader_test, "
        "audio_framer_test, audio_mel_feature_extract_test, "
        "audio_preprocessor_test, audio_stream_buffer_test, "
        "audio_stream_pipeline_test, model_loader_test, "
        "model_input_test, wav2lip_model_smoke_test, "
        "model_inference_test, model_inference_benchmark, "
        "output_processor_test, timestamp_manager_test, "
        "frame_scheduler_test, audio_video_synchronous_test")
