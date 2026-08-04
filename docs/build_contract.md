# Build Contract — digital_human_sdk v0.1.0

> **Module**: M01-build-contract-foundation  
> **Generated**: 2026-08-04  
> **Commit baseline**: `504391d90f8057178a4cc59a935407a14b5c4df2`

## CMake Options

All options default to **ON** for backward compatibility.

| Option | Default | Description |
|---|---|---|
| `USE_MARCH_NATIVE` | ON | Compile with `-march=native` for host-specific instruction sets (AVX2, SSE, etc.) |
| `BUILD_TESTS` | ON | Build unit and integration tests (requires GTest). When OFF, skips `find_package(GTest)`, `enable_testing()`, and `add_subdirectory(tests)`. |
| `BUILD_EXAMPLES` | ON | Build example programs. When OFF, skips `add_subdirectory(examples)`. |
| `ENABLE_ASAN` | OFF | Enable AddressSanitizer (`-fsanitize=address -fno-omit-frame-pointer`). **Mutually exclusive with TSAN.** |
| `ENABLE_TSAN` | OFF | Enable ThreadSanitizer (`-fsanitize=thread -fno-omit-frame-pointer`). **Mutually exclusive with ASAN.** |

### Usage Examples

```sh
# Default release build (all options ON, -march=native)
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel $(nproc)

# Disable host-specific optimizations (portable binary)
cmake -B build/portable -DCMAKE_BUILD_TYPE=Release -DUSE_MARCH_NATIVE=OFF

# Library-only build (no tests, no examples)
cmake -B build/libonly -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF

# AddressSanitizer build
cmake -B build/asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON

# ThreadSanitizer build
cmake -B build/tsan -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON

# ASAN+TSAN together → FATAL_ERROR (mutually exclusive)
cmake -B build/broken -DENABLE_ASAN=ON -DENABLE_TSAN=ON  # FAILS
```

## Required Dependencies

| Dependency | Version (tested) | CMake Module |
|---|---|---|
| C++17 compiler | GCC 11.4.0 | `CMAKE_CXX_STANDARD 17` |
| OpenMP | 4.5 | `find_package(OpenMP)` |
| ncnn | 20260801 | `find_package(ncnn)` |
| FFmpeg (avformat, avcodec, avutil, swresample, swscale) | 58.76.100 | `pkg_check_modules` |
| PortAudio | 19 | `pkg_check_modules(portaudio-2.0)` |
| OpenCV | 4.5.4 | `find_package(OpenCV)` |
| dlib | 19.10.0 | `find_package(dlib)` |
| GTest | 1.11.0 | `find_package(GTest)` — only when `BUILD_TESTS=ON` |
| Threads | — | `find_package(Threads)` |
| PkgConfig | 0.29.2 | `find_package(PkgConfig)` |

## Compiler Flags (Default)

| Flag | Condition | Purpose |
|---|---|---|
| `-std=c++17` | Always | C++17 standard |
| `-march=native` | `USE_MARCH_NATIVE=ON` | Host-specific instruction set |
| `-DNDEBUG` | Always | Disable assertions in Release |
| `-Wall -Wextra` | Always | Common warnings |
| `-g` | Always (via `src/CMakeLists.txt`) | Debug symbols for GDB |
| `-fsanitize=address -fno-omit-frame-pointer` | `ENABLE_ASAN=ON` | AddressSanitizer |
| `-fsanitize=thread -fno-omit-frame-pointer` | `ENABLE_TSAN=ON` | ThreadSanitizer |

## Build Targets

| Target | Type | Description |
|---|---|---|
| `digital_human_core` | SHARED library | Core SDK library |
| `demo_app` | Executable | Basic demo (if `demo.cpp` exists) |
| `opencv_test` | Executable | OpenCV integration test |
| `ncnn_test` | Executable | ncnn integration test |
| `ffmpeg_audio_test` | Executable | FFmpeg audio test |
| `dlib_test` | Executable | dlib face detection test |
| Various `*_unit_test` | Executable + CTest | Unit tests (when `BUILD_TESTS=ON`) |
| Various `*_ctest` | Executable + CTest | Integration/smoke tests (when `BUILD_TESTS=ON`) |

## Output Directories

| Artifact | Path |
|---|---|
| Shared library | `${CMAKE_BINARY_DIR}/lib/libdigital_human_core.so` |
| Static library | `${CMAKE_BINARY_DIR}/lib/` |
| Executables | `${CMAKE_BINARY_DIR}/bin/` |

## Sanitizer Contract

1. `ENABLE_ASAN` and `ENABLE_TSAN` are **mutually exclusive** — setting both produces `FATAL_ERROR`.
2. Each sanitizer must be configured in its **own build directory** to avoid cached flag conflicts.
3. Both sanitizers add `-fno-omit-frame-pointer` for readable stack traces.
4. ASAN uses `-fsanitize=address` for both compile and link.
5. TSAN uses `-fsanitize=thread` for both compile and link.

## CTest Structure

Tests use `enable_testing()` + `add_subdirectory(tests)` (when `BUILD_TESTS=ON`).

| Test category | Prefix | Framework |
|---|---|---|
| Frame scheduler | `frame_scheduler.` | `gtest_discover_tests` |
| Timestamp manager | `timestamp_manager.` | `gtest_discover_tests` |
| PortAudio playback | `portaudio_playback.` | `gtest_discover_tests` |
| Audio-video sync | `audio_video_synchronous.` | `gtest_discover_tests` |
| Pipeline (unit/lifecycle/golden) | `pipeline.` | `gtest_discover_tests` |
| Legacy unit tests | (no prefix) | `add_test` |

Run with: `cd build/release && ctest --output-on-failure -j$(nproc)`

## Data Contracts (Key Types)

See formal module documentation:
- **Pipeline types**: `include/pipeline/pipeline_types.h` — `PreparedFaceContext` (96×96 CV_8UC3 BGR), `AudioFeatureTask` (1280-float freq-major Mel), `InferenceFrameTask`
- **Model inference**: `include/model/model_inference.h` — `InferenceOutput` (ncnn::Mat pred), `InferenceStatus`, retry policy
- **Audio clock**: `include/sync/audio_clock.h` — `AudioClockResult`, monotonic PTS timeline
- **Timestamp**: `include/core/timestamp_manager.h` — `MediaTimestamp` (pts_us, frame_index, sample_offset)
