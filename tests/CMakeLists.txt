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
