# 图像加载模块单元测试
add_executable(image_loader_unit_test image_loader_unit_test.cpp)
target_link_libraries(image_loader_unit_test PRIVATE digital_human_core)

# 注册到 CTest
add_test(NAME image_loader_unit_test COMMAND image_loader_unit_test)
