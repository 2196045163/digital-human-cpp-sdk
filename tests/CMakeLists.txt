# 图像加载模块单元测试
add_executable(image_loader_unit_test image_loader_unit_test.cpp)
target_link_libraries(image_loader_unit_test PRIVATE digital_human_core)
add_test(NAME image_loader_unit_test COMMAND image_loader_unit_test)

# 人脸检测模块单元测试
add_executable(face_detector_unit_test face_detector_unit_test.cpp)
target_link_libraries(face_detector_unit_test PRIVATE digital_human_core)
add_test(NAME face_detector_unit_test COMMAND face_detector_unit_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
