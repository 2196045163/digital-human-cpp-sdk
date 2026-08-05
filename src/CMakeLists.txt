# 递归搜集 src 下所有 .cpp 源文件
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/video/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/output/*.cpp
)

# 若尚无源文件，编译一个占位文件避免 CMake 报错
if(NOT SRC_FILES)
    set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/placeholder.cpp)
endif()

# 将所有源文件编译成 digital_human_core 共享库
add_library(digital_human_core SHARED ${SRC_FILES})

# 动态库
set_target_properties(digital_human_core PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
)

target_compile_options(digital_human_core PRIVATE $<$<CONFIG:Debug>:-g>)
#配置库的头文件搜索路径
target_include_directories(digital_human_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 配置一些第三方库等
# 链接 OpenCV ncnn 库
target_link_libraries(digital_human_core PUBLIC
    ${OpenCV_LIBS}
    ncnn
    PkgConfig::FFMPEG
    dlib::dlib
)

# PortAudio is hidden behind PImpl and is not part of the public header ABI.
target_link_libraries(digital_human_core PRIVATE PkgConfig::PORTAUDIO)

# Pipeline 多线程支持
target_link_libraries(digital_human_core PRIVATE Threads::Threads)

# OpenMP
target_link_libraries(digital_human_core PRIVATE OpenMP::OpenMP_CXX)

message("OpenCV libraries: ${OpenCV_LIBS}")
