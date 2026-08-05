# Digital Human SDK (C++)

A C++ inference pipeline for speech-driven facial animation, built around the
Wav2Lip model. The SDK takes a static face image and an audio clip, and produces
a lip-synchronized MP4 video through a fully offline processing pipeline.

## Architecture Overview

The system is organized as a layered C++17 library with the following structure:

```
include/          Public API headers (28 headers across 8 modules)
src/              Core implementation (24 C++ source files)
apps/             CLI application entry point
tests/            Unit tests (GoogleTest, 21 test files)
docs/             Architecture and design documentation
```

For a detailed pipeline diagram, see [docs/architecture.md](docs/architecture.md).

## Core Modules

| Module | Directory | Description |
|---|---|---|
| **Audio** | `include/audio/`, `src/audio/` | Audio loading (FFmpeg), resampling to 16kHz, PCM framing, Mel spectrogram extraction |
| **Core** | `include/core/`, `src/core/` | Face detection (dlib HOG + SVM), dlib 68-point landmarks, face alignment, image I/O, mask generation, mouth restoration and blending |
| **Model** | `include/model/`, `src/model/` | ncnn model loading, Wav2Lip input construction, batched inference scheduling, output post-processing |
| **Pipeline** | `include/pipeline/`, `src/pipeline/` | End-to-end orchestration, bounded task queues, multi-threaded producer-consumer pipeline |
| **Output** | `include/output/`, `src/output/` | FFmpeg-based H.264/AAC MP4 encoding and muxing (FinalMediaWriter) |
| **Sync** | `include/sync/`, `src/sync/` | Audio-video synchronization, PortAudio playback, audio master clock |

## Complete Processing Pipeline

1. **Audio decoding & resampling** — Input audio (any FFmpeg-supported format) is decoded and resampled to 16 kHz mono PCM.
2. **Mel spectrogram extraction** — 80-bin Mel spectrogram computed with Wav2Lip-compatible parameters (n_fft=800, fmin=55 Hz, fmax=7600 Hz), sliced into 16-frame chunks.
3. **Face detection & landmarks** — dlib HOG + SVM frontal face detector locates the face; dlib shape_predictor extracts 68-point facial landmarks.
4. **Image preprocessing** — Face is aligned and cropped to 96×96 using similarity transform; lower-half mask is applied for Wav2Lip's six-channel input. OpenCV is used for image loading, resize, color conversion, and matrix operations.
5. **ncnn Wav2Lip inference** — The six-channel face input and Mel spectrogram chunk are fed to the ncnn inference engine.
6. **Mouth restoration & blending** — The 96×96 generated face is restored to the original image coordinates through local color matching within the mask region, alpha blending, and optional lightweight high-frequency detail recovery from the original image.
7. **FFmpeg H.264/AAC muxing** — Video frames and audio are encoded and muxed into an MP4 container.

## Technology Stack

| Component | Role |
|---|---|
| C++17 | Core language |
| CMake 3.16+ | Build system |
| [ncnn](https://github.com/Tencent/ncnn) | Neural network inference |
| [FFmpeg](https://ffmpeg.org/) | Audio decode, video/audio encode, MP4 mux |
| [OpenCV](https://opencv.org/) | Image loading, resize, color conversion, matrix operations, mask generation, mouth blending |
| [dlib](http://dlib.net/) | Face detection (HOG + SVM) and landmark detection (68-point shape_predictor) |
| [PortAudio](http://www.portaudio.com/) | Audio playback (experimental component) |
| [GoogleTest](https://github.com/google/googletest) | Unit testing |
| OpenMP | CPU parallelism |

## Supported Input / Output Formats

**Input:**
- Image: JPEG, PNG, BMP (single static face portrait, recommended 512×512)
- Audio: WAV, MP3, AAC, M4A, MP4, FLAC (format detection is extension-based; decoding is performed by FFmpeg)

**Output:**
- Container: MP4
- Video: H.264, 25 FPS
- Audio: AAC, 16 kHz mono
- Resolution: Configurable (default matches input image dimensions)

## Build Dependencies

Required system packages (Ubuntu 22.04 example):

```bash
sudo apt install -y \
    build-essential cmake pkg-config \
    libopencv-dev \
    libdlib-dev \
    portaudio19-dev \
    libavformat-dev libavcodec-dev libavutil-dev \
    libswresample-dev libswscale-dev \
    libgtest-dev
```

ncnn must be built from source or installed via package manager:
```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) && sudo make install
```

To specify a custom ncnn installation path:

```bash
cmake .. -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
# or:
cmake .. -DCMAKE_PREFIX_PATH=/path/to/ncnn/install
```

## Model Files

**Model weights are not provided in this repository.** The following files must
be obtained separately by the user:

| File | Description | Source |
|---|---|---|
| `wav2lip.param` / `wav2lip.bin` | Wav2Lip model in ncnn format | Convert from original Wav2Lip PyTorch weights |
| `shape_predictor_68_face_landmarks.dat` | dlib 68-point landmark model | [dlib model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing information
regarding the Wav2Lip model.

## Basic Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Build options:
- `BUILD_TESTS=ON/OFF` — Enable/disable unit tests (default: ON)
- `BUILD_EXAMPLES=ON/OFF` — Enable/disable example programs (default: ON)
- `USE_MARCH_NATIVE=ON/OFF` — Enable `-march=native` for host-specific instruction sets (default: OFF)
- `ENABLE_ASAN=ON` — Enable AddressSanitizer
- `ENABLE_TSAN=ON` — Enable ThreadSanitizer

## CLI Usage

The CLI supports offline mode only. Real-time mode is explicitly rejected
with exit code 3.

```bash
./build/bin/digital_human_app \
    --image      path/to/face.jpg \
    --audio      path/to/speech.wav \
    --model-param  path/to/wav2lip.param \
    --model-bin    path/to/wav2lip.bin \
    --landmark     path/to/shape_predictor_68_face_landmarks.dat \
    --output     output.mp4 \
    --fps 25
```

The CLI performs a four-condition success check on completion:
1. Pipeline completes without error
2. Media writer is finalized
3. No writer error recorded
4. Output file exists and is non-empty

On success, a JSON result with pipeline statistics is printed to stdout and the
process exits with code 0.

## Performance

*Measured on a 2-logical-core Ubuntu 22.04 virtual machine.*

These figures were collected from the full development repository using fixed
input benchmarks; model weights and test assets are not provided in this
public snapshot.

| Metric | Value |
|---|---|
| Input image | 512×512 static portrait |
| Input audio | 8.136 seconds, 16 kHz mono |
| Output | 512×512, 25 FPS, 204 frames, H.264/AAC MP4 |
| Average total wall time | 16.51 seconds |
| Average processing speed | 12.36 FPS |
| Peak memory usage | ~666 MB |

*Note: Performance varies significantly with CPU core count, ncnn optimization
flags, and input dimensions.*

## Testing

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

The test suite includes unit tests for audio processing, core image operations,
model input construction, output writing, pipeline orchestration, and
synchronization components. Tests that require model weights or test data files
are excluded from this public repository (see [docs/testing.md](docs/testing.md)).

## Known Limitations

- **Model-dependent**: The pipeline requires a Wav2Lip model in ncnn format. Model weights are not included and must be obtained by the user.
- **Static portrait only**: The current implementation processes a single static face image; it does not support video input with changing backgrounds or multiple faces.
- **Offline processing**: The CLI supports offline batch processing only. The codebase includes experimental audio-video synchronization and PortAudio playback components, but the public CLI explicitly rejects real-time mode.
- **Single model architecture**: The pipeline is built around the Wav2Lip model input/output contract; using other lip-sync models requires modifications to the model spec and input builder.
- **Platform support**: Developed and tested on Linux (Ubuntu 22.04). Windows and macOS may require adjustments to the build configuration and FFmpeg/PortAudio integration.
- **Face detection**: The dlib HOG + SVM frontal face detector works well for frontal faces but may miss faces at extreme angles or under poor lighting, and may produce false positives on non-face objects.

## Third-Party Licenses

This project integrates multiple third-party components. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for detailed information about
each component's license and usage.

**Important**: The Wav2Lip model and pretrained weights are released for
personal, research, and non-commercial use only. Users must comply with the
original Wav2Lip license terms.

## Project Structure

```
digital-human-cpp-sdk/
├── CMakeLists.txt              Root build configuration
├── README.md
├── THIRD_PARTY_NOTICES.md
├── include/                    Public API headers
│   ├── audio/                  Audio processing API
│   ├── core/                   Image and face processing API
│   ├── model/                  Model inference API
│   ├── output/                 Media writer API
│   ├── pipeline/               Pipeline orchestration API
│   ├── sync/                   Synchronization and playback API
│   ├── utils/                  Utilities
│   └── video/                  Video frame types
├── src/                        Implementation
│   ├── audio/
│   ├── core/
│   ├── model/detail/           Internal model specs and helpers
│   ├── output/
│   ├── pipeline/detail/        Internal pipeline components
│   └── sync/
├── apps/                       CLI application
├── tests/                      Unit tests (model-independent subset)
└── docs/                       Documentation
    ├── architecture.md         Pipeline architecture diagram
    └── testing.md              Test suite overview
```

## Project Scope

This is a C++17 engineering example for offline static-portrait lip-sync
generation. It demonstrates media processing (FFmpeg audio decode and
resampling), image processing (OpenCV-based face alignment, mask generation,
mouth-region blending), neural network inference integration (ncnn), and
audio/video encoding and muxing (FFmpeg H.264/AAC MP4).

Model weights and test media are not distributed with this repository.
This public repository does not accept model weight submissions.
