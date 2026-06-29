# 图像加载模块单元测试
add_executable(image_loader_unit_test image_loader_unit_test.cpp)
target_link_libraries(image_loader_unit_test PRIVATE digital_human_core)
add_test(NAME image_loader_unit_test COMMAND image_loader_unit_test)

# 人脸检测模块单元测试
add_executable(face_detector_unit_test face_detector_unit_test.cpp)
target_link_libraries(face_detector_unit_test PRIVATE digital_human_core)
add_test(NAME face_detector_unit_test COMMAND face_detector_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸对齐模块单元测试
add_executable(face_aligner_unit_test face_aligner_unit_test.cpp)
target_link_libraries(face_aligner_unit_test PRIVATE digital_human_core)
add_test(NAME face_aligner_unit_test COMMAND face_aligner_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸对齐图的像素预处理模块单元测试
add_executable(image_preprocessor_unit_test image_preprocessor_unit_test.cpp)
target_link_libraries(image_preprocessor_unit_test PRIVATE digital_human_core)
add_test(NAME image_preprocessor_unit_test COMMAND image_preprocessor_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸掩码生成模块单元测试
add_executable(face_mask_generator_unit_test face_mask_generator_unit_test.cpp)
target_link_libraries(face_mask_generator_unit_test PRIVATE digital_human_core)
add_test(NAME face_mask_generator_unit_test COMMAND face_mask_generator_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 人脸融合模块单元测试
add_executable(face_blender_unit_test face_blender_unit_test.cpp)
target_link_libraries(face_blender_unit_test PRIVATE digital_human_core)
add_test(NAME face_blender_unit_test COMMAND face_blender_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})