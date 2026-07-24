/// @file model_visual_quality_diagnostic_test.cpp
/// @brief 用同一份真实素材做受控变量实验，定位嘴部异常来自输入几何、Mel 语义还是模型音频通路。
///
/// 这不是“看起来不错就通过”的视觉质量测试。它只负责生成可比较的证据：
/// - 同一时刻、同一模型、只替换一个变量；
/// - 数值报告说明两个候选是否真的不同；
/// - 图片用于人工 A/B，不把主观观感伪装成自动断言。

#include "audio/audio_framer.h"
#include "audio/audio_loader.h"
#include "audio/audio_mel_feature_extract.h"
#include "audio/audio_preprocessor.h"
#include "core/face_aligner.h"
#include "core/face_detector.h"
#include "core/image_loader.h"
#include "model/input_processor.h"
#include "model/model_inference.h"
#include "model/model_loader.h"
#include "model/ncnn_input_adapter.h"
#include "model/output_processor.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace digital_human;

namespace {

constexpr double kDiagnosticTimeMs = 1000.0;

struct MelVariant {
    std::string name;
    double frame_duration_ms = 0.0;
    double hop_duration_ms = 0.0;
    audio::AudioWindowType window_type = audio::AudioWindowType::kNone;
    std::size_t chunk_index = 0;
    std::vector<float> chunk;
};

struct InferenceImage {
    bool success = false;
    std::string error_message;
    cv::Mat generated_face_bgr;
};

void WriteImage(const std::string& path, const cv::Mat& image) {
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);
    ASSERT_FALSE(ec) << "cannot create parent directory for " << path;
    ASSERT_TRUE(cv::imwrite(path, image)) << "cannot write " << path;
}

void WriteText(const std::string& path, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);
    ASSERT_FALSE(ec) << "cannot create parent directory for " << path;
    std::ofstream output(path);
    ASSERT_TRUE(output.is_open()) << "cannot write " << path;
    output << content;
}

/// @brief 构造官方 inference.py 风格的人脸输入：检测框下方补 10 像素后直接缩放，不旋转双眼。
cv::Mat BuildDetectorCrop96(const cv::Mat& image, const cv::Rect& detected_face) {
    const cv::Rect image_bounds(0, 0, image.cols, image.rows);
    cv::Rect padded = detected_face;
    padded.height += 10;
    padded &= image_bounds;
    if (padded.width <= 0 || padded.height <= 0) {
        return {};
    }

    cv::Mat resized;
    cv::resize(image(padded), resized, cv::Size(96, 96), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

MelVariant BuildMelVariant(
    const std::vector<float>& preprocessed_pcm,
    const std::string& name,
    const audio::AudioFrameOptions& frame_options) {
    MelVariant variant;
    variant.name = name;
    variant.frame_duration_ms = frame_options.frame_duration_ms;
    variant.hop_duration_ms = frame_options.hop_duration_ms;
    variant.window_type = frame_options.window_type;

    audio::AudioFramer framer(frame_options);
    const std::vector<std::vector<float>> frames =
        framer.FrameSamplesOnly(preprocessed_pcm);
    if (frames.empty()) {
        return variant;
    }

    audio::MelFeatureExtractor extractor(
        audio::MelFeatureExtractor::Wav2LipDefault());
    const audio::MelFeatureResult mel_result = extractor.ExtractBatch(frames);
    if (!mel_result.success) {
        return variant;
    }

    const audio::MelChunkResult chunks =
        extractor.BuildWav2LipChunks(mel_result.mel);
    if (!chunks.success || chunks.chunks.empty()) {
        return variant;
    }

    const double mel_frames_per_ms = 1.0 / frame_options.hop_duration_ms;
    const std::size_t requested_index = static_cast<std::size_t>(
        std::llround(kDiagnosticTimeMs * mel_frames_per_ms));
    variant.chunk_index = std::min(requested_index, chunks.chunks.size() - 1);
    variant.chunk = chunks.chunks[variant.chunk_index];
    return variant;
}

InferenceImage RunInference(
    const model::ModelRuntimeSnapshot& snapshot,
    const cv::Mat& face_bgr,
    const std::vector<float>& mel_chunk) {
    InferenceImage result;

    model::Wav2LipInputBuilder builder;
    const model::Wav2LipInputResult build_result =
        builder.Build(face_bgr, mel_chunk);
    if (!build_result.success) {
        result.error_message = "builder: " + build_result.error_message;
        return result;
    }

    model::NcnnInputAdapter adapter;
    const model::NcnnInputResult adapter_result =
        adapter.Adapt(build_result.data);
    if (!adapter_result.success) {
        result.error_message = "adapter: " + adapter_result.error_message;
        return result;
    }

    model::ModelInference inference;
    model::InferenceOptions inference_options;
    const model::SingleInferenceResult inference_result =
        inference.Infer(snapshot, adapter_result.input, inference_options);
    if (!inference_result.success) {
        result.error_message = "inference: " + inference_result.error_message;
        return result;
    }

    model::OutputProcessor output_processor;
    const model::OutputProcessResult output_result =
        output_processor.Convert(inference_result.value);
    if (!output_result.success) {
        result.error_message = "output: " + output_result.error_message;
        return result;
    }

    result.success = true;
    result.generated_face_bgr = output_result.value.generated_face_bgr;
    return result;
}

double MeanAbsoluteDifference(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size() || left.empty()) {
        return -1.0;
    }

    double absolute_sum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        absolute_sum += std::abs(
            static_cast<double>(left[index]) - static_cast<double>(right[index]));
    }
    return absolute_sum / static_cast<double>(left.size());
}

/// @brief 计算两张同尺寸、同类型图像所有通道的平均绝对像素差。
/// @return 合法时返回 [0,255]；尺寸/类型不一致或图像为空时返回 -1。
double MeanAbsoluteImageDifference(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty() ||
        left.size() != right.size() ||
        left.type() != right.type()) {
        return -1.0;
    }

    cv::Mat absolute_difference;
    cv::absdiff(left, right, absolute_difference);
    const cv::Scalar channel_means = cv::mean(absolute_difference);

    double mean_sum = 0.0;
    for (int channel_index = 0;
         channel_index < absolute_difference.channels();
         ++channel_index) {
        mean_sum += channel_means[channel_index];
    }
    return mean_sum / absolute_difference.channels();
}

const char* WindowName(audio::AudioWindowType window_type) {
    switch (window_type) {
        case audio::AudioWindowType::kNone: return "none";
        case audio::AudioWindowType::kHamming: return "hamming";
        case audio::AudioWindowType::kHann: return "hann";
        default: return "unknown";
    }
}

}  // namespace

TEST(ModelVisualQualityDiagnostic, ControlledInputAblations) {
    model::ModelLoader loader;
    model::ModelLoadOptions load_options;
    load_options.backend = model::ModelBackend::kCpu;
    load_options.enable_warmup = true;
    const model::ModelLoadResult load_result =
        loader.Load("models/wav2lip/wav2lip.param", load_options);
    ASSERT_TRUE(load_result.success) << load_result.error_message;
    const model::ModelRuntimeSnapshot snapshot = loader.AcquireSnapshot();
    ASSERT_TRUE(snapshot.IsValid());

    core::ImageLoader image_loader;
    const core::LoadResult image_result =
        image_loader.LoadFromFile("testdata/golden/face.jpg");
    ASSERT_TRUE(image_result.success) << image_result.error_message;
    const cv::Mat& original = image_result.image;

    core::FaceDetector detector;
    const core::ModelLoadResult landmark_load =
        detector.LoadLandmarkModel("models/shape_predictor_68_face_landmarks.dat");
    ASSERT_TRUE(landmark_load.success) << landmark_load.error_message;
    const core::FaceAnalyzeResult face_result =
        detector.DetectAndLandmark(original);
    ASSERT_TRUE(face_result.success) << face_result.error_message;
    ASSERT_FALSE(face_result.detection.faces.empty());
    ASSERT_FALSE(face_result.landmarks.empty());

    const cv::Rect detected_face = face_result.detection.faces.front().rect;
    const std::vector<cv::Point>& landmarks =
        face_result.landmarks.front().landmarks;

    core::FaceAligner aligner;
    core::FaceAlignmentOptions align_options;
    align_options.target_size = 96;
    const core::FaceAlignmentResult aligned_result =
        aligner.Align(original, detected_face, landmarks, align_options);
    ASSERT_TRUE(aligned_result.success) << aligned_result.error_message;
    const cv::Mat eye_aligned_96 = aligned_result.aligned_face;
    model::Wav2LipInputBuilder face_builder;
    const model::Wav2LipFacePrepareResult prepared_face_result =
        face_builder.PrepareFace(original, detected_face, landmarks);
    ASSERT_TRUE(prepared_face_result.success)
        << prepared_face_result.error_message;
    const cv::Mat detector_crop_96 =
        prepared_face_result.value.face_bgr;
    const cv::Mat direct_detector_crop_96 =
        BuildDetectorCrop96(original, detected_face);
    ASSERT_FALSE(eye_aligned_96.empty());
    ASSERT_FALSE(detector_crop_96.empty());
    ASSERT_EQ(
        MeanAbsoluteImageDifference(detector_crop_96, direct_detector_crop_96),
        0.0);

    audio::AudioLoader audio_loader(16000);
    const audio::AudioLoadResult audio_result =
        audio_loader.LoadFromFile("testdata/golden/audio.wav");
    ASSERT_TRUE(audio_result.success) << audio_result.error_message;

    audio::AudioPreprocessor current_preprocessor;
    const audio::AudioPreprocessResult current_preprocess_result =
        current_preprocessor.Process(audio_result.audio.pcm);
    ASSERT_TRUE(current_preprocess_result.success)
        << current_preprocess_result.error_message;

    audio::AudioPreprocessOptions wav2lip_preprocess_options;
    wav2lip_preprocess_options.enable_normalize = false;
    wav2lip_preprocess_options.enable_pre_emphasis = true;
    audio::AudioPreprocessor wav2lip_preprocessor(wav2lip_preprocess_options);
    const audio::AudioPreprocessResult wav2lip_preprocess_result =
        wav2lip_preprocessor.Process(audio_result.audio.pcm);
    ASSERT_TRUE(wav2lip_preprocess_result.success)
        << wav2lip_preprocess_result.error_message;

    const audio::AudioFrameOptions speech_options =
        audio::AudioFramer::SpeechDefault();
    audio::AudioFrameOptions wav2lip_options =
        audio::AudioFramer::Wav2LipDefault();
    const MelVariant current_mel = BuildMelVariant(
        current_preprocess_result.pcm, "current_speech_default", speech_options);
    const MelVariant wav2lip_framer_mel = BuildMelVariant(
        wav2lip_preprocess_result.pcm,
        "wav2lip_input_contract",
        wav2lip_options);
    ASSERT_EQ(current_mel.chunk.size(), 1280u);
    ASSERT_EQ(wav2lip_framer_mel.chunk.size(), 1280u);

    const std::vector<float> zero_mel(1280, 0.0f);
    const InferenceImage current_all = RunInference(
        snapshot, eye_aligned_96, current_mel.chunk);
    const InferenceImage geometry_only = RunInference(
        snapshot, detector_crop_96, current_mel.chunk);
    const InferenceImage mel_only = RunInference(
        snapshot, eye_aligned_96, wav2lip_framer_mel.chunk);
    const InferenceImage candidate_all = RunInference(
        snapshot, detector_crop_96, wav2lip_framer_mel.chunk);
    const InferenceImage zero_audio = RunInference(
        snapshot, detector_crop_96, zero_mel);

    ASSERT_TRUE(current_all.success) << current_all.error_message;
    ASSERT_TRUE(geometry_only.success) << geometry_only.error_message;
    ASSERT_TRUE(mel_only.success) << mel_only.error_message;
    ASSERT_TRUE(candidate_all.success) << candidate_all.error_message;
    ASSERT_TRUE(zero_audio.success) << zero_audio.error_message;

    WriteImage("golden_output/visual_diagnostic_01_eye_aligned_input.png",
               eye_aligned_96);
    WriteImage("golden_output/visual_diagnostic_02_detector_crop_input.png",
               detector_crop_96);
    WriteImage("golden_output/visual_diagnostic_03_current_all.png",
               current_all.generated_face_bgr);
    WriteImage("golden_output/visual_diagnostic_04_geometry_only.png",
               geometry_only.generated_face_bgr);
    WriteImage("golden_output/visual_diagnostic_05_mel_only.png",
               mel_only.generated_face_bgr);
    WriteImage("golden_output/visual_diagnostic_06_candidate_all.png",
               candidate_all.generated_face_bgr);
    WriteImage("golden_output/visual_diagnostic_07_zero_audio.png",
               zero_audio.generated_face_bgr);

    const double face_input_mad =
        MeanAbsoluteImageDifference(eye_aligned_96, detector_crop_96);
    const double mel_input_mad =
        MeanAbsoluteDifference(current_mel.chunk, wav2lip_framer_mel.chunk);
    const double geometry_output_mad = MeanAbsoluteImageDifference(
        current_all.generated_face_bgr, geometry_only.generated_face_bgr);
    const double mel_output_mad = MeanAbsoluteImageDifference(
        current_all.generated_face_bgr, mel_only.generated_face_bgr);
    const double combined_output_mad = MeanAbsoluteImageDifference(
        current_all.generated_face_bgr, candidate_all.generated_face_bgr);
    const double audio_sensitivity_mad = MeanAbsoluteImageDifference(
        candidate_all.generated_face_bgr, zero_audio.generated_face_bgr);

    // 这些断言只证明受控变量确实改变了输入/输出，不判断哪张图主观上更好。
    EXPECT_GT(face_input_mad, 0.0);
    EXPECT_GT(mel_input_mad, 0.0);
    EXPECT_GT(geometry_output_mad, 0.0);
    EXPECT_GT(mel_output_mad, 0.0);
    EXPECT_GT(combined_output_mad, 0.0);
    EXPECT_GT(audio_sensitivity_mad, 0.0);

    std::ostringstream report;
    report << std::boolalpha << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"diagnostic_time_ms\": " << kDiagnosticTimeMs << ",\n"
           << "  \"model_generation\": " << snapshot.generation << ",\n"
           << "  \"current_mel\": {\"frame_ms\": "
           << current_mel.frame_duration_ms << ", \"hop_ms\": "
           << current_mel.hop_duration_ms << ", \"window\": \""
           << WindowName(current_mel.window_type) << "\", \"chunk_index\": "
           << current_mel.chunk_index << "},\n"
           << "  \"wav2lip_framer_candidate_mel\": {\"frame_ms\": "
           << wav2lip_framer_mel.frame_duration_ms << ", \"hop_ms\": "
           << wav2lip_framer_mel.hop_duration_ms << ", \"window\": \""
           << WindowName(wav2lip_framer_mel.window_type)
           << "\", \"chunk_index\": " << wav2lip_framer_mel.chunk_index << "},\n"
           << "  \"mean_absolute_differences\": {\n"
           << "    \"face_input_pixels\": " << face_input_mad << ",\n"
           << "    \"mel_input_values\": " << mel_input_mad << ",\n"
           << "    \"geometry_only_output_pixels\": " << geometry_output_mad << ",\n"
           << "    \"mel_only_output_pixels\": " << mel_output_mad << ",\n"
           << "    \"combined_output_pixels\": " << combined_output_mad << ",\n"
           << "    \"candidate_vs_zero_audio_output_pixels\": "
           << audio_sensitivity_mad << "\n"
           << "  },\n"
           << "  \"evidence_boundary\": \"Differences prove sensitivity, not visual quality. "
              "The candidate Mel still lacks a numerical librosa oracle.\"\n"
           << "}\n";
    WriteText("golden_output/model_visual_quality_diagnostic.json", report.str());
}
