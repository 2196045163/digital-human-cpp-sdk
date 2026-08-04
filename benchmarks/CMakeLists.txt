# M05 Pipeline 性能基准可执行文件
add_executable(pipeline_performance_benchmark
    pipeline_performance_benchmark.cpp)
target_link_libraries(pipeline_performance_benchmark
    PRIVATE digital_human_core)
target_include_directories(pipeline_performance_benchmark
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
