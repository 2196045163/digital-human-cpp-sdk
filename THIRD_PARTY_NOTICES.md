# Third-Party Notices

This project integrates and depends on several third-party components.
Their usage in this project is described below.

---

## Wav2Lip

- **Official source**: https://github.com/Rudrabha/Wav2Lip
- **Paper**: "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild" (Prajwal et al., ACM Multimedia 2020)
- **Usage in this project**: This project implements a C++ inference pipeline that consumes a Wav2Lip-compatible ncnn model. The pipeline handles audio preprocessing (16 kHz resampling, Mel spectrogram extraction with Wav2Lip parameters), face detection and alignment, six-channel face input construction, ncnn inference, and mouth-region restoration and blending.
- **Model weights**: The Wav2Lip model weights (`.param` / `.bin` for ncnn) are **not included** in this repository. Users must obtain or convert model weights from legitimate sources themselves.
- **License note**: The Wav2Lip model and its associated weights are intended for **personal, research, and non-commercial use only**. Users are responsible for complying with the original Wav2Lip license terms. This project does not grant any additional rights to the Wav2Lip model or its derivatives.

---

## ncnn

- **Official source**: https://github.com/Tencent/ncnn
- **License**: BSD 3-Clause License (see ncnn repository for full text)
- **Usage in this project**: ncnn is the neural network inference framework used to run the Wav2Lip model. The project loads ncnn `.param` and `.bin` model files, constructs input tensors (Mel spectrogram + face image), performs forward inference, and retrieves the generated face prediction.

---

## FFmpeg

- **Official source**: https://ffmpeg.org/
- **License**: LGPL 2.1+ / GPL 2.0+ (depending on build configuration)
- **Usage in this project**: FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`, `libswresample`, `libswscale`) are used for:
  - Audio decoding and resampling (input audio to 16 kHz mono PCM)
  - H.264 video encoding
  - AAC audio encoding
  - MP4 muxing (final output container)

---

## OpenCV

- **Official source**: https://opencv.org/
- **License**: Apache 2.0 License
- **Usage in this project**: OpenCV is used for:
  - Image loading, decoding, and writing
  - Image resize, color conversion, and matrix operations
  - Face crop and image preprocessing
  - Mask generation
  - Mouth region restoration: local color matching and alpha blending

---

## dlib

- **Official source**: http://dlib.net/
- **License**: Boost Software License
- **Usage in this project**: dlib is used for face detection (HOG + SVM frontal face detector) and face landmark detection (68-point shape_predictor). The landmark model file (`shape_predictor_68_face_landmarks.dat`) is **not included** in this repository. Users must obtain it from legitimate sources.

---

## PortAudio

- **Official source**: http://www.portaudio.com/
- **License**: MIT License (PortAudio-specific)
- **Usage in this project**: PortAudio is an experimental component used for audio playback and as the audio master clock for audio-video synchronization. The public CLI does not expose real-time mode.

---

## GoogleTest

- **Official source**: https://github.com/google/googletest
- **License**: BSD 3-Clause License
- **Usage in this project**: GoogleTest is the unit testing framework used for all C++ tests in the `tests/` directory.

---

## Summary

| Component | License | Model/Weight Included? |
|---|---|---|
| Wav2Lip | Personal / Research / Non-commercial | **No** — user must obtain separately |
| ncnn | BSD 3-Clause | **No** — user must build/install separately |
| FFmpeg | LGPL 2.1+ / GPL 2.0+ | **No** — user must install separately |
| OpenCV | Apache 2.0 | **No** — user must install separately |
| dlib | Boost Software License | **No** — landmark model not included |
| PortAudio | MIT (PortAudio) | **No** — user must install separately |
| GoogleTest | BSD 3-Clause | **No** — user must install separately |

---

## Important Notes

1. **No model weights or landmark data files are distributed with this repository.**
   All `.param`, `.bin`, `.dat` files referenced in source code are expected to be
   provided by the user from legitimate sources.

2. **Wav2Lip usage restriction**: The Wav2Lip model and pretrained weights are
   released for personal, research, and non-commercial use. Users of this
   project must respect the original Wav2Lip license terms.

3. **This document does not re-license any third-party component.** Each
   component is governed by its own license. Users are responsible for
   understanding and complying with all applicable licenses.

4. **No LICENSE file is provided in this repository at this time.** The
   appropriate open-source license for the original code in this project
   is to be determined separately.
