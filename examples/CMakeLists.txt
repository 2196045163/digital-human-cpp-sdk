# 1. 构建 DemoApp（基础演示）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp")
    add_executable(demo_app demo.cpp)
    target_link_libraries(demo_app PRIVATE digital_human_core)
endif()

# 2. 构建 OpenCV 测试程序（验证图像功能）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv_test.cpp")
    add_executable(opencv_test opencv_test.cpp)
    # digital_human_core 已 PUBLIC 链接 OpenCV，此处无需重复
    target_link_libraries(opencv_test PRIVATE digital_human_core)
endif()

# 3. 构建 ncnn 推理引擎测试程序（验证 ncnn 集成与模型推理性能）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn_test.cpp")
    add_executable(ncnn_test ncnn_test.cpp)
    # digital_human_core 已 PUBLIC 链接 ncnn + OpenCV，此处无需重复
    target_link_libraries(ncnn_test PRIVATE digital_human_core)
endif()

# 4. 构建 FFmpeg 测试程序 (验证音频功能)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_audio_test.cpp")
    add_executable(ffmpeg_audio_test ffmpeg_audio_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(ffmpeg_audio_test PRIVATE digital_human_core)
endif()

# 5. 构建 dlib 测试程序 (验证人脸检测和关键点提取)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dlib_test.cpp")
    add_executable(dlib_test dlib_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(dlib_test PRIVATE digital_human_core)
endif()


# 6. 构建 ImageLoader 测试程序（验证图像加载模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image_loader_test.cpp")
    add_executable(image_loader_test image_loader_test.cpp)
    target_link_libraries(image_loader_test PRIVATE digital_human_core)
endif()

# 7. 构建 FaceDetector 测试程序（验证人脸检测与关键点）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/face_detector_test.cpp")
    add_executable(face_detector_test face_detector_test.cpp)
    target_link_libraries(face_detector_test PRIVATE digital_human_core)
endif()

message(STATUS "Configured example targets: demo_app, opencv_test, ncnn_test, image_loader_test, face_detector_test")