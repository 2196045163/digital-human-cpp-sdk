#include "model/output_processor.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "model/detail/wav2lip_model_spec.h"

namespace digital_human::model {
namespace {

InferenceOutput MakeValidInferenceOutput() {
    InferenceOutput input;
    input.pred = ncnn::Mat(
        detail::Wav2LipModelSpec::kPredWidth,
        detail::Wav2LipModelSpec::kPredHeight,
        detail::Wav2LipModelSpec::kPredChannels);
    input.pred.fill(0.25f);
    input.metadata.pts_ms = 120;
    input.metadata.frame_index = 3;
    input.model_generation = 9;
    return input;
}

InferenceOutput MakeInferenceOutputFromBgr(const cv::Mat& image_bgr) {
    EXPECT_EQ(image_bgr.type(), CV_8UC3);
    EXPECT_EQ(image_bgr.cols, detail::Wav2LipModelSpec::kPredWidth);
    EXPECT_EQ(image_bgr.rows, detail::Wav2LipModelSpec::kPredHeight);

    InferenceOutput input = MakeValidInferenceOutput();
    float* blue_plane = input.pred.channel(detail::Wav2LipModelSpec::kBluePlaneIndex);
    float* green_plane = input.pred.channel(detail::Wav2LipModelSpec::kGreenPlaneIndex);
    float* red_plane = input.pred.channel(detail::Wav2LipModelSpec::kRedPlaneIndex);
    for (int row_index = 0; row_index < image_bgr.rows; ++row_index) {
        for (int column_index = 0; column_index < image_bgr.cols; ++column_index) {
            const int pixel_index = row_index * image_bgr.cols + column_index;
            const cv::Vec3b pixel = image_bgr.at<cv::Vec3b>(row_index, column_index);
            blue_plane[pixel_index] = static_cast<float>(pixel[0]) / 255.0f;
            green_plane[pixel_index] = static_cast<float>(pixel[1]) / 255.0f;
            red_plane[pixel_index] = static_cast<float>(pixel[2]) / 255.0f;
        }
    }

    return input;
}

cv::Mat MakeCheckerboardImage() {
    cv::Mat image(
        detail::Wav2LipModelSpec::kPredHeight,
        detail::Wav2LipModelSpec::kPredWidth,
        CV_8UC3);

    for (int row_index = 0; row_index < image.rows; ++row_index) {
        for (int column_index = 0; column_index < image.cols; ++column_index) {
            const unsigned char value = (row_index + column_index) % 2 == 0 ? 0 : 255;
            image.at<cv::Vec3b>(row_index, column_index) = cv::Vec3b(value, value, value);
        }
    }

    return image;
}

void ExpectPreservedIdentity(const OutputProcessResult& result) {
    ASSERT_TRUE(result.value.metadata.pts_ms.has_value());
    ASSERT_TRUE(result.value.metadata.frame_index.has_value());
    EXPECT_EQ(*result.value.metadata.pts_ms, 120);
    EXPECT_EQ(*result.value.metadata.frame_index, 3);
    EXPECT_EQ(result.value.model_generation, 9u);
}

void ExpectStatus(const OutputProcessResult& result, OutputProcessStatus expected_status) {
    EXPECT_FALSE(result.success);
    EXPECT_EQ(static_cast<int>(result.status), static_cast<int>(expected_status));
    EXPECT_FALSE(result.error_message.empty());
    EXPECT_TRUE(result.value.generated_face_bgr.empty());
    ExpectPreservedIdentity(result);
}

} // namespace

// 该测试锁定已由真实 A/B 证据确认的映射；它本身不能证明模型原始通道语义。
TEST(OutputProcessorTest, ConvertsConfirmedPlanesWithRoundToNearestQuantization) {
    InferenceOutput input = MakeValidInferenceOutput();
    float* blue_plane = input.pred.channel(detail::Wav2LipModelSpec::kBluePlaneIndex);
    float* green_plane = input.pred.channel(detail::Wav2LipModelSpec::kGreenPlaneIndex);
    float* red_plane = input.pred.channel(detail::Wav2LipModelSpec::kRedPlaneIndex);
    blue_plane[0] = 0.0f;
    green_plane[0] = 0.5f;
    red_plane[0] = 1.0f;

    OutputProcessor processor;
    const OutputProcessResult result = processor.Convert(input);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(static_cast<int>(result.status),
              static_cast<int>(OutputProcessStatus::kOk));
    ASSERT_EQ(result.value.generated_face_bgr.type(), CV_8UC3);
    ASSERT_EQ(result.value.generated_face_bgr.rows, 96);
    ASSERT_EQ(result.value.generated_face_bgr.cols, 96);
    const cv::Vec3b pixel = result.value.generated_face_bgr.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[0], 0);
    EXPECT_EQ(pixel[1], 128);
    EXPECT_EQ(pixel[2], 255);
    EXPECT_EQ(result.value.conversion_info.corrected_value_count, 0u);
    ExpectPreservedIdentity(result);
}

TEST(OutputProcessorTest, ClampsValuesInsideSharedToleranceAndReportsCorrections) {
    InferenceOutput input = MakeValidInferenceOutput();
    float* blue_plane = input.pred.channel(detail::Wav2LipModelSpec::kBluePlaneIndex);
    float* red_plane = input.pred.channel(detail::Wav2LipModelSpec::kRedPlaneIndex);
    blue_plane[0] = detail::Wav2LipModelSpec::kPredValueMin -
        detail::Wav2LipModelSpec::kPredRangeTolerance * 0.5f;
    red_plane[0] = detail::Wav2LipModelSpec::kPredValueMax +
        detail::Wav2LipModelSpec::kPredRangeTolerance * 0.5f;

    OutputProcessor processor;
    const OutputProcessResult result = processor.Convert(input);

    ASSERT_TRUE(result.success) << result.error_message;
    const cv::Vec3b pixel = result.value.generated_face_bgr.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[0], 0);
    EXPECT_EQ(pixel[2], 255);
    EXPECT_EQ(result.value.conversion_info.corrected_value_count, 2u);
}

TEST(OutputProcessorTest, RejectsInvalidTensorAndNeverReturnsPartialImage) {
    InferenceOutput input = MakeValidInferenceOutput();
    input.pred = ncnn::Mat(95, 96, 3);

    OutputProcessor processor;
    const OutputProcessResult result = processor.Convert(input);

    ExpectStatus(result, OutputProcessStatus::kInvalidPredictionShape);
}

TEST(OutputProcessorTest, RejectsNaNAndOutOfToleranceRange) {
    OutputProcessor processor;

    InferenceOutput non_finite_input = MakeValidInferenceOutput();
    float* blue_plane = non_finite_input.pred.channel(
        detail::Wav2LipModelSpec::kBluePlaneIndex);
    blue_plane[0] = std::numeric_limits<float>::quiet_NaN();
    ExpectStatus(
        processor.Convert(non_finite_input),
        OutputProcessStatus::kNonFinitePrediction);

    InferenceOutput out_of_range_input = MakeValidInferenceOutput();
    float* red_plane = out_of_range_input.pred.channel(
        detail::Wav2LipModelSpec::kRedPlaneIndex);
    red_plane[0] = detail::Wav2LipModelSpec::kPredValueMax +
        detail::Wav2LipModelSpec::kPredRangeTolerance * 2.0f;
    ExpectStatus(
        processor.Convert(out_of_range_input),
        OutputProcessStatus::kRangeViolation);
}

TEST(OutputProcessorTest, RejectsInvalidSharpnessThresholdBeforeImageAllocation) {
    InferenceOutput input = MakeValidInferenceOutput();
    OutputProcessOptions options;
    options.minimum_sharpness = -1.0;

    OutputProcessor processor;
    const OutputProcessResult result = processor.Convert(input, options);

    ExpectStatus(result, OutputProcessStatus::kInvalidSharpnessThreshold);
}

TEST(OutputProcessorTest, RecordsFiniteSharpnessWithoutThreshold) {
    OutputProcessor processor;
    const OutputProcessResult result = processor.Convert(MakeValidInferenceOutput());

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_TRUE(std::isfinite(result.value.conversion_info.sharpness_score));
    EXPECT_GE(result.value.conversion_info.sharpness_score, 0.0);
    EXPECT_FALSE(result.value.conversion_info.sharpness_threshold_applied);
    EXPECT_TRUE(result.value.conversion_info.sharpness_passed);
}

TEST(OutputProcessorTest, ReportsLowerSharpnessForBlurredImageAndKeepsConversionSuccessful) {
    const cv::Mat high_frequency_image = MakeCheckerboardImage();
    cv::Mat blurred_image;
    cv::GaussianBlur(high_frequency_image, blurred_image, cv::Size(9, 9), 2.0);

    OutputProcessor processor;
    const OutputProcessResult sharp_result = processor.Convert(
        MakeInferenceOutputFromBgr(high_frequency_image));
    const OutputProcessResult blurred_result = processor.Convert(
        MakeInferenceOutputFromBgr(blurred_image));

    ASSERT_TRUE(sharp_result.success) << sharp_result.error_message;
    ASSERT_TRUE(blurred_result.success) << blurred_result.error_message;
    ASSERT_GT(sharp_result.value.conversion_info.sharpness_score, 0.0);
    EXPECT_GT(
        sharp_result.value.conversion_info.sharpness_score,
        blurred_result.value.conversion_info.sharpness_score * 10.0);

    OutputProcessOptions too_high_threshold;
    too_high_threshold.minimum_sharpness =
        sharp_result.value.conversion_info.sharpness_score + 1.0;
    const OutputProcessResult below_threshold_result = processor.Convert(
        MakeInferenceOutputFromBgr(high_frequency_image), too_high_threshold);

    ASSERT_TRUE(below_threshold_result.success) << below_threshold_result.error_message;
    EXPECT_TRUE(below_threshold_result.value.conversion_info.sharpness_threshold_applied);
    EXPECT_FALSE(below_threshold_result.value.conversion_info.sharpness_passed);

    OutputProcessOptions passing_threshold;
    passing_threshold.minimum_sharpness = 0.0;
    const OutputProcessResult above_threshold_result = processor.Convert(
        MakeInferenceOutputFromBgr(high_frequency_image), passing_threshold);

    ASSERT_TRUE(above_threshold_result.success) << above_threshold_result.error_message;
    EXPECT_TRUE(above_threshold_result.value.conversion_info.sharpness_threshold_applied);
    EXPECT_TRUE(above_threshold_result.value.conversion_info.sharpness_passed);
}

} // namespace digital_human::model
