cmake_minimum_required(VERSION 3.16)

project(digital_human_sdk VERSION 0.1.0 LANGUAGES CXX)

# 全局配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ──────────────────────────────────────────────
# 构建选项
# ──────────────────────────────────────────────
option(USE_MARCH_NATIVE "Compile with -march=native for host-specific instruction sets" OFF)
# 本机指令集优化，默认关闭以保证生成物可移植性
option(BUILD_TESTS       "Build unit and integration tests (requires GTest)" ON)
option(ENABLE_ASAN       "Enable AddressSanitizer (mutually exclusive with TSAN)" OFF)
option(ENABLE_TSAN       "Enable ThreadSanitizer (mutually exclusive with ASAN)" OFF)

if(USE_MARCH_NATIVE)
    add_compile_options(-march=native)
    message(STATUS ">> march=native enabled")
endif()
#常用警告
add_compile_options(-Wall -Wextra)

# OpenMP
find_package(OpenMP REQUIRED)

# ncnn — 通过标准 CMake 方式查找；可通过以下方式指定安装路径：
#   cmake .. -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/ncnn/install
find_package(ncnn CONFIG REQUIRED)
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

# PortAudio（音频播放与音频主时钟）
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

message(STATUS ">> FFMPEG Found: ${FFMPEG_VERSION}")
message(STATUS ">> FFmpeg Include: ${FFMPEG_INCLUDE_DIRS}")
message(STATUS ">> PortAudio Found: ${PORTAUDIO_VERSION}")
include_directories(${FFMPEG_INCLUDE_DIRS})

# 输出目录
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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

# 将 OpenCV 的头文件路径添加到全局的包含路径中
include_directories(${OpenCV_INCLUDE_DIRS})

# dlib
find_package(dlib REQUIRED)
message(STATUS ">> dlib Found: ${dlib_VERSION}")
include_directories(${dlib_INCLUDE_DIRS})

# Threads（Pipeline 多线程支持）
find_package(Threads REQUIRED)
message(STATUS ">> Threads Found")

# ASAN/TSAN 互斥检查与编译器标志注入
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASAN and TSAN are mutually exclusive. Enable only one.")
endif()

if(ENABLE_ASAN)
    message(STATUS ">> ASAN enabled")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_TSAN)
    message(STATUS ">> TSAN enabled")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

# 子目录
add_subdirectory(src)
add_subdirectory(apps)

# Benchmarks 依赖未公开的模型和测试数据，未包含在公开副本中

# 测试支持
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS ">> Tests enabled")
else()
    message(STATUS ">> Tests disabled (BUILD_TESTS=OFF)")
endif()
