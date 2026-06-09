cmake_minimum_required(VERSION 3.16)

project(digital_hunam_sdk VERSION 0.1.0 LANGUAGES CXX)

# 全局配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认开启优化
# add_compile_options(-O3)

#启用本机指令集优化(如AVx2/SSE)
add_compile_options(-march=native)
add_definitions(-DNDEBUG)

#常用警告
add_compile_options(-Wall -Wextra)

# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# ncnn
find_package(ncnn REQUIRED)
message(STATUS ">> ncnn Found")

# FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswresample
    libswscale
)

message(STATUS ">> FFMPEG Found: ${FFMPEG_VERSION}")
message(STATUS ">> FFmpeg Include: ${FFMPEG_INCLUDE_DIRS}")
include_directories(${FFMPEG_INCLUDE_DIRS})

# 输出目录
Set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
Set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
Set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 全局头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 第三方依赖

# 项目的第三方依赖
# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 打印以下信息，便于调试
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# 将 OpenCV 的头文件路径天骄到全局的包含路径中
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# dlib
find_package(dlib REQUIRED)
message(STATUS ">> dlib Found: ${dlib_VERSION}")
include_directories(${dlib_INCLUDE_DIRS})

# CMakeLists 子目录 这些路径下再搞一个 CMakeLists，编译 SDK 的核心源码，比如生成一个库
# 如 src/CMakeLists.txt 负责源码库；examples/CMakeLists.txt 负责示例程序
add_subdirectory(src)
add_subdirectory(examples)

# 测试支持
enable_testing()
add_subdirectory(tests)

# 库和可执行文件分别在 src/CMakeLists.txt 和 examples/CMakeLists.txt 中定义