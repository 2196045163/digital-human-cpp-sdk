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

message(STATUS "Configured example targets: demo_app, opencv_test, "
        "ncnn_test, ffmpeg_audio_test, dlib_test, image_loader_test, "
        "face_detector_test, face_aligner_test, image_preprocessor_test, "
        "face_mask_generator_test, face_blender_test, audio_loader_test")
