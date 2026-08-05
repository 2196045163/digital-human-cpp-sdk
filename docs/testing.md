# Testing

This document describes the test files retained in this public repository,
which test types are excluded and why, and the dependencies required to run
the retained tests.

---

## Retained Tests (21 files)

### tests/audio/
| File | Description |
|---|---|
| `audio_framer_unit_test.cpp` | Unit test — synthesizes PCM in memory, no external data |
| `audio_mel_feature_extract_unit_test.cpp` | Unit test — synthetic audio, no file dependency |
| `audio_preprocessor_unit_test.cpp` | Unit test — synthetic signal, no file dependency |
| `audio_stream_buffer_unit_test.cpp` | Unit test — in-memory buffer operations |

### tests/core/
| File | Description |
|---|---|
| `face_aligner_unit_test.cpp` | Unit test — synthetic image geometry |
| `face_blender_unit_test.cpp` | Unit test — synthetic image operations |
| `face_mask_generator_unit_test.cpp` | Unit test — synthetic mask generation |
| `frame_scheduler_unit_test.cpp` | Unit test — scheduling logic with synthetic data |
| `image_loader_unit_test.cpp` | Self-contained — creates and cleans up its own temp files |
| `image_preprocessor_unit_test.cpp` | Unit test — synthetic image transforms |
| `timestamp_manager_unit_test.cpp` | Unit test — timestamp arithmetic |

### tests/model/
| File | Description |
|---|---|
| `model_input_unit_test.cpp` | Unit test — builds synthetic tensors in memory |
| `ncnn_input_adapter_unit_test.cpp` | Unit test — adapter logic, no model loading |
| `output_processor_unit_test.cpp` | Unit test — post-processing on synthetic data |

### tests/output/
| File | Description |
|---|---|
| `final_media_writer_unit_test.cpp` | Unit test — writer API and config validation |
| `final_media_writer_failure_test.cpp` | Unit test — error-path coverage (uses standard /dev/null paths) |

### tests/pipeline/
| File | Description |
|---|---|
| `bounded_task_queue_unit_test.cpp` | Unit test — concurrent queue correctness |
| `digital_human_pipeline_unit_test.cpp` | Unit test — config validation and error handling with /nonexistent paths |
| `digital_human_pipeline_lifecycle_test.cpp` | Unit test — Start/Stop/Cancel lifecycle error handling |

### tests/sync/
| File | Description |
|---|---|
| `audio_video_synchronous_unit_test.cpp` | Unit test — sync logic with synthetic data |
| `portaudio_playback_unit_test.cpp` | Unit test — callback state machine logic |

---

## Excluded Test Types

The following categories of tests are not included in this public repository
because they depend on model weights or test data that are not publicly distributed:

| Category | Reason |
|---|---|
| E2E acceptance tests | Require wav2lip model, landmark model, and test audio/image files |
| CLI integration tests | Require wav2lip model, landmark model, and test audio/image files |
| Golden / upstream integration tests | Require model files and test data; write golden output artifacts |
| Model loader / inference unit tests | Reference wav2lip model files for loading and inference |
| Face detector unit tests | Reference landmark model file and test images |
| Frame scheduler integration tests | Reference test image files |
| Audio loader unit tests | Reference test audio files |
| Media writer integration tests | Reference test audio files |

---

## Test Dependencies

To build and run the retained tests:

```bash
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

Required system packages:
- GoogleTest (`libgtest-dev`)
- All dependencies of `digital_human_core` (ncnn, OpenCV, dlib, FFmpeg, PortAudio)

Tests marked as "pure unit test" use only in-memory synthetic data and
do not require any external files to run.
