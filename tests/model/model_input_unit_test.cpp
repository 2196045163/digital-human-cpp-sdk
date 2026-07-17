/// @file model_input_unit_test.cpp
/// @brief 验证 Wav2LipInputBuilder：六通道构建、下半脸 mask、[0,1] 归一化、Fail Fast
///
/// 测试范围：
///   - 正常构建（shape、范围、metadata 透传、Mel 透传）
///   - 归一化范围 [0,1]（非 [-1,1]）—— 用纯黑/纯白像素验证
///   - 下半脸 mask（上半脸保留、下半脸通道 0-2 置零、通道 3-5 保留）
///   - 错误路径（空图、尺寸错误、类型错误、Mel 长度错误、NaN）
///
/// 不测试：
///   - ncnn::Extractor 推理（Builder 不依赖 ncnn）
///   - NcnnInputAdapter 的 w/h/c 映射（那是 Adapter 单元测试的范围）
///   - 真实人脸对齐效果（需要 FaceAligner + golden face.jpg）

#include "model/input_processor.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <limits>

using namespace digital_human::model;

// ============================================================================
// 正常构建：验证 shape、范围、metadata 透传
// ============================================================================
TEST(Wav2LipInputBuilderTest, NormalBuild) {
    // 全白 96×96 BGR 图
    cv::Mat white(96, 96, CV_8UC3, cv::Scalar(255, 255, 255));

    // Mel：freq-major, 值 = 1000*f + t（可推导，用于验证布局）
    std::vector<float> mel(1280);
    for (int f = 0; f < 80; ++f) {
        for (int t = 0; t < 16; ++t) {
            mel[f * 16 + t] = 1000.0f * f + t;
        }
    }

    ModelInputMetadata meta;
    meta.pts_ms = 100;
    meta.frame_index = 5;

    Wav2LipInputBuilder builder;
    auto r = builder.Build(white, mel, meta);

    ASSERT_TRUE(r.success) << r.error_message;

    // Shape 正确
    EXPECT_EQ(r.info.face_channels, 6);
    EXPECT_EQ(r.info.face_height, 96);
    EXPECT_EQ(r.info.face_width, 96);
    EXPECT_EQ(r.info.mel_bins, 80);
    EXPECT_EQ(r.info.mel_frames, 16);
    EXPECT_EQ(r.info.mask_start_row, 48);
    EXPECT_FALSE(r.info.has_nan_or_inf);

    // 全白图归一化后 max=1.0，但下半脸 mask 置零所以 min=0.0
    EXPECT_FLOAT_EQ(r.info.face_max_value, 1.0f);
    EXPECT_FLOAT_EQ(r.info.face_min_value, 0.0f);

    // metadata 原样透传
    ASSERT_TRUE(r.data.metadata.pts_ms.has_value());
    EXPECT_EQ(r.data.metadata.pts_ms.value(), 100);
    ASSERT_TRUE(r.data.metadata.frame_index.has_value());
    EXPECT_EQ(r.data.metadata.frame_index.value(), 5);

    // Mel 原样透传
    EXPECT_FLOAT_EQ(r.data.mel_freq_time[7 * 16 + 3], 7003.0f);
}

// ============================================================================
// 黑色像素验证归一化 [0,1] 而非 [-1,1]
// ============================================================================
TEST(Wav2LipInputBuilderTest, NormalizationIsZeroToOne) {
    cv::Mat black(96, 96, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<float> mel(1280, 0.0f);

    Wav2LipInputBuilder builder;
    auto r = builder.Build(black, mel);

    ASSERT_TRUE(r.success) << r.error_message;
    // 如果是 [-1,1] 公式，黑色会变成 -1.0
    EXPECT_FLOAT_EQ(r.info.face_min_value, 0.0f);
    EXPECT_FLOAT_EQ(r.info.face_max_value, 0.0f);
}

TEST(Wav2LipInputBuilderTest, BgrOrderAndMaskBoundary) {
    // B/G/R 使用互不相等的值，才能真正捕获 BGR↔RGB 或通道索引写反。
    cv::Mat face(96, 96, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<float> mel(1280, 0.0f);

    Wav2LipInputBuilder builder;
    auto r = builder.Build(face, mel);

    ASSERT_TRUE(r.success) << r.error_message;
    const int plane_size = 96 * 96;
    const int x = 11;
    const int last_unmasked_row = 47;
    const int first_masked_row = 48;
    const float b = 10.0f / 255.0f;
    const float g = 20.0f / 255.0f;
    const float red = 30.0f / 255.0f;

    // y=47 仍在上半脸：masked 与 original 都应保留 B、G、R。
    const int upper_offset = last_unmasked_row * 96 + x;
    EXPECT_FLOAT_EQ(r.data.face_chw[0 * plane_size + upper_offset], b);
    EXPECT_FLOAT_EQ(r.data.face_chw[1 * plane_size + upper_offset], g);
    EXPECT_FLOAT_EQ(r.data.face_chw[2 * plane_size + upper_offset], red);
    EXPECT_FLOAT_EQ(r.data.face_chw[3 * plane_size + upper_offset], b);
    EXPECT_FLOAT_EQ(r.data.face_chw[4 * plane_size + upper_offset], g);
    EXPECT_FLOAT_EQ(r.data.face_chw[5 * plane_size + upper_offset], red);

    // y=48 是遮挡第一行：前 3 通道清零，后 3 通道仍保留完整原图。
    const int lower_offset = first_masked_row * 96 + x;
    EXPECT_FLOAT_EQ(r.data.face_chw[0 * plane_size + lower_offset], 0.0f);
    EXPECT_FLOAT_EQ(r.data.face_chw[1 * plane_size + lower_offset], 0.0f);
    EXPECT_FLOAT_EQ(r.data.face_chw[2 * plane_size + lower_offset], 0.0f);
    EXPECT_FLOAT_EQ(r.data.face_chw[3 * plane_size + lower_offset], b);
    EXPECT_FLOAT_EQ(r.data.face_chw[4 * plane_size + lower_offset], g);
    EXPECT_FLOAT_EQ(r.data.face_chw[5 * plane_size + lower_offset], red);
}

// ============================================================================
// 下半脸 mask：上半脸保留、下半脸通道 0-2 置零、通道 3-5 保留
// ============================================================================
TEST(Wav2LipInputBuilderTest, LowerHalfMask) {
    cv::Mat white(96, 96, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<float> mel(1280, 0.0f);

    Wav2LipInputBuilder builder;
    auto r = builder.Build(white, mel);

    ASSERT_TRUE(r.success) << r.error_message;

    const int HW = 96 * 96;
    const float kOne = 1.0f;

    // 上半脸 (y=10)：masked 和 original 值相同
    int y_upper = 10, x = 30;
    EXPECT_FLOAT_EQ(r.data.face_chw[0 * HW + y_upper * 96 + x], kOne);
    EXPECT_FLOAT_EQ(r.data.face_chw[3 * HW + y_upper * 96 + x], kOne);

    // 下半脸 (y=80)：masked(0-2)=0, original(3-5)=1.0
    int y_lower = 80;
    EXPECT_FLOAT_EQ(r.data.face_chw[0 * HW + y_lower * 96 + x], 0.0f);
    EXPECT_FLOAT_EQ(r.data.face_chw[3 * HW + y_lower * 96 + x], kOne);
    EXPECT_FLOAT_EQ(r.data.face_chw[1 * HW + y_lower * 96 + x], 0.0f);
    EXPECT_FLOAT_EQ(r.data.face_chw[2 * HW + y_lower * 96 + x], 0.0f);
}

// ============================================================================
// 错误路径
// ============================================================================
TEST(Wav2LipInputBuilderTest, EmptyAlignedFace) {
    Wav2LipInputBuilder builder;
    std::vector<float> mel(1280, 0.0f);
    cv::Mat empty;
    auto r = builder.Build(empty, mel);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelInputStatus::kEmptyAlignedFace);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_TRUE(r.data.face_chw.empty());
    EXPECT_TRUE(r.data.mel_freq_time.empty());
}

TEST(Wav2LipInputBuilderTest, WrongFaceSize) {
    Wav2LipInputBuilder builder;
    std::vector<float> mel(1280, 0.0f);

    cv::Mat wrong(95, 96, CV_8UC3, cv::Scalar(0));
    auto wrong_height = builder.Build(wrong, mel);
    EXPECT_FALSE(wrong_height.success);
    EXPECT_EQ(wrong_height.status, ModelInputStatus::kInvalidFaceSize);
    EXPECT_FALSE(wrong_height.error_message.empty());

    cv::Mat wrong2(96, 95, CV_8UC3, cv::Scalar(0));
    auto wrong_width = builder.Build(wrong2, mel);
    EXPECT_FALSE(wrong_width.success);
    EXPECT_EQ(wrong_width.status, ModelInputStatus::kInvalidFaceSize);
}

TEST(Wav2LipInputBuilderTest, WrongFaceType) {
    Wav2LipInputBuilder builder;
    std::vector<float> mel(1280, 0.0f);
    cv::Mat gray(96, 96, CV_8UC1, cv::Scalar(128));
    auto gray_result = builder.Build(gray, mel);
    EXPECT_FALSE(gray_result.success);
    EXPECT_EQ(gray_result.status, ModelInputStatus::kInvalidFaceType);
    EXPECT_FALSE(gray_result.error_message.empty());

    // BGRA 与 float 三通道也必须拒绝；Builder 不在边界内偷偷 convert。
    cv::Mat bgra(96, 96, CV_8UC4, cv::Scalar(0, 0, 0, 255));
    EXPECT_EQ(builder.Build(bgra, mel).status, ModelInputStatus::kInvalidFaceType);

    cv::Mat float_bgr(96, 96, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    EXPECT_EQ(builder.Build(float_bgr, mel).status, ModelInputStatus::kInvalidFaceType);
}

TEST(Wav2LipInputBuilderTest, WrongMelSize) {
    Wav2LipInputBuilder builder;
    cv::Mat face(96, 96, CV_8UC3, cv::Scalar(0));
    std::vector<float> too_short(1279, 0.0f);
    auto short_result = builder.Build(face, too_short);
    EXPECT_FALSE(short_result.success);
    EXPECT_EQ(short_result.status, ModelInputStatus::kInvalidMelChunkSize);
    EXPECT_FALSE(short_result.error_message.empty());

    std::vector<float> too_long(1281, 0.0f);
    EXPECT_EQ(builder.Build(face, too_long).status,
              ModelInputStatus::kInvalidMelChunkSize);
}

TEST(Wav2LipInputBuilderTest, NonFiniteMelIsRejected) {
    Wav2LipInputBuilder builder;
    cv::Mat face(96, 96, CV_8UC3, cv::Scalar(0));
    std::vector<float> nan_mel(1280, 0.0f);
    nan_mel[500] = std::nanf("");
    auto nan_result = builder.Build(face, nan_mel);
    EXPECT_FALSE(nan_result.success);
    EXPECT_EQ(nan_result.status, ModelInputStatus::kNonFiniteMelValue);
    EXPECT_FALSE(nan_result.error_message.empty());

    std::vector<float> inf_mel(1280, 0.0f);
    inf_mel[501] = std::numeric_limits<float>::infinity();
    auto inf_result = builder.Build(face, inf_mel);
    EXPECT_FALSE(inf_result.success);
    EXPECT_EQ(inf_result.status, ModelInputStatus::kNonFiniteMelValue);
}
