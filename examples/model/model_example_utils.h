#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "model/model_inference.h"

namespace model_example {

constexpr int kMelFrames = 16;
constexpr int kMelBins = 80;
constexpr int kMelChannels = 1;
constexpr int kFaceWidth = 96;
constexpr int kFaceHeight = 96;
constexpr int kFaceChannels = 6;

inline digital_human::model::NcnnWav2LipInput MakeSyntheticInput(
    std::int64_t frame_index,
    std::int64_t pts_ms) {
    digital_human::model::NcnnWav2LipInput input;
    input.mel = ncnn::Mat(
        kMelFrames,
        kMelBins,
        kMelChannels,
        sizeof(float));
    input.face = ncnn::Mat(
        kFaceWidth,
        kFaceHeight,
        kFaceChannels,
        sizeof(float));

    if (!input.mel.empty()) {
        input.mel.fill(0.0f);
    }
    if (!input.face.empty()) {
        input.face.fill(0.0f);
    }

    input.metadata.frame_index = frame_index;
    input.metadata.pts_ms = pts_ms;
    return input;
}

inline bool WriteTextFile(
    const std::filesystem::path& path,
    const std::string& content) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        std::cerr << "ERROR: cannot create " << path.parent_path()
                  << ": " << error.message() << '\n';
        return false;
    }

    std::ofstream output(path);
    if (!output) {
        std::cerr << "ERROR: cannot write " << path << '\n';
        return false;
    }

    output << content;
    if (!output) {
        std::cerr << "ERROR: failed while writing " << path << '\n';
        return false;
    }

    std::cout << "  -> " << path << '\n';
    return true;
}

inline std::string JsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());

    for (const char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }

    return escaped;
}

inline const char* BuildMode() {
#ifdef NDEBUG
    return "Release-like (NDEBUG)";
#else
    return "Debug-like";
#endif
}

inline double AttemptComputeMs(
    const digital_human::model::SingleInferenceResult& result) {
    return result.value.timing.first_attempt_ms +
           result.value.timing.retry_attempts_ms;
}

} // namespace model_example
