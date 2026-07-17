/// @file ncnn_input_adapter_unit_test.cpp
/// @brief 验证 NcnnInputAdapter：float buffer → ncnn::Mat 的 w/h/c 映射
///
/// 测试范围：
///   - Mel 布局验证（1000*f+t 可推导值锁定 freq-major）
///   - 人脸六通道映射验证（不对称 BGR 像素锁定通道顺序）
///   - 错误路径验证（空 buffer、错误尺寸）
///   - metadata 透传验证
///
/// 不测试：
///   - ncnn::Extractor 推理（Adapter 不做推理）
///   - Wav2LipInputBuilder 的 mask/归一化（那是 Builder 单元测试的范围）

#include "model/ncnn_input_adapter.h"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace digital_human::model;

// ============================================================================
// Mel 布局：1000*f + t 锁定 freq-major
// ============================================================================
// 如果 Adapter 误将 time-major 数据按 freq-major 索引写入 ncnn::Mat，
// row(7)[3] 会是 3007 而非 7003 —— 本测试必须抓到这种错误。
TEST(NcnnInputAdapterTest, MelLayoutFreqMajor) {
    Wav2LipInputData data;
    data.mel_freq_time.resize(1280);
    for (int f = 0; f < 80; ++f) {
        for (int t = 0; t < 16; ++t) {
            data.mel_freq_time[f * 16 + t] = 1000.0f * f + t;
        }
    }
    data.face_chw.resize(6 * 96 * 96, 0.5f);

    NcnnInputAdapter adapter;
    auto r = adapter.Adapt(data);

    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_EQ(r.input.mel.w, 16);
    ASSERT_EQ(r.input.mel.h, 80);
    ASSERT_EQ(r.input.mel.c, 1);

    // 三个频率的首、中、尾帧逐一验证
    EXPECT_FLOAT_EQ(r.input.mel.row(0)[0],   0.0f);
    EXPECT_FLOAT_EQ(r.input.mel.row(0)[15],  15.0f);
    EXPECT_FLOAT_EQ(r.input.mel.row(7)[3],   7003.0f);
    EXPECT_FLOAT_EQ(r.input.mel.row(7)[8],   7008.0f);
    EXPECT_FLOAT_EQ(r.input.mel.row(79)[0],  79000.0f);
    EXPECT_FLOAT_EQ(r.input.mel.row(79)[15], 79015.0f);
}

// ============================================================================
// 人脸通道映射：不对称 BGR 像素锁定每个通道的值
// ============================================================================
// 为 6 个通道分别设置唯一 sentinel，并同时检查首、中、尾三个位置。
// 这样既能捕获通道交换，也能捕获 cstep/复制长度错误；本测试只验证 Adapter 原样映射，
// masked/original 与 BGR 的语义正确性由 Builder 的独立测试负责。
TEST(NcnnInputAdapterTest, FaceChannelMapping) {
    const int HW = 96 * 96;
    std::vector<float> face_chw(6 * HW, 0.0f);

    const int middle = 20 * 96 + 10;
    const int last = HW - 1;
    for (int channel = 0; channel < 6; ++channel) {
        // 六个通道和三个位置使用不同 sentinel。
        // 若 Adapter 交换 masked/original、错用 cstep 或漏拷贝尾部，至少一个断言会失败。
        face_chw[channel * HW] = 0.01f * static_cast<float>(channel + 1);
        face_chw[channel * HW + middle] = 0.10f * static_cast<float>(channel + 1);
        face_chw[channel * HW + last] = 1.00f + 0.10f * static_cast<float>(channel + 1);
    }

    Wav2LipInputData data;
    data.face_chw = face_chw;
    data.mel_freq_time.resize(1280, 0.0f);

    NcnnInputAdapter adapter;
    auto r = adapter.Adapt(data);

    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_EQ(r.input.face.w, 96);
    ASSERT_EQ(r.input.face.h, 96);
    ASSERT_EQ(r.input.face.c, 6);

    for (int channel = 0; channel < 6; ++channel) {
        const float* output = r.input.face.channel(channel);
        EXPECT_FLOAT_EQ(output[0], 0.01f * static_cast<float>(channel + 1));
        EXPECT_FLOAT_EQ(output[middle], 0.10f * static_cast<float>(channel + 1));
        EXPECT_FLOAT_EQ(output[last], 1.00f + 0.10f * static_cast<float>(channel + 1));
    }
}

// ============================================================================
// 错误路径
// ============================================================================
TEST(NcnnInputAdapterTest, EmptyFaceBuffer) {
    NcnnInputAdapter adapter;
    Wav2LipInputData data;
    data.mel_freq_time.resize(1280, 0.0f);
    // face_chw 为空

    auto r = adapter.Adapt(data);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelInputStatus::kInvalidFaceSize);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_TRUE(r.input.face.empty());
    EXPECT_TRUE(r.input.mel.empty());
}

TEST(NcnnInputAdapterTest, WrongMelSize) {
    NcnnInputAdapter adapter;
    Wav2LipInputData data;
    data.face_chw.resize(6 * 96 * 96, 0.0f);
    data.mel_freq_time.resize(100);  // 不是 1280

    auto r = adapter.Adapt(data);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelInputStatus::kInvalidMelChunkSize);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_TRUE(r.input.face.empty());
    EXPECT_TRUE(r.input.mel.empty());
}

TEST(NcnnInputAdapterTest, NonFiniteFaceIsRejected) {
    Wav2LipInputData data;
    data.face_chw.resize(6 * 96 * 96, 0.0f);
    data.mel_freq_time.resize(1280, 0.0f);
    data.face_chw[123] = std::numeric_limits<float>::infinity();

    NcnnInputAdapter adapter;
    auto r = adapter.Adapt(data);

    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelInputStatus::kNonFiniteFaceValue);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_TRUE(r.input.face.empty());
    EXPECT_TRUE(r.input.mel.empty());
}

TEST(NcnnInputAdapterTest, NonFiniteMelIsRejected) {
    Wav2LipInputData data;
    data.face_chw.resize(6 * 96 * 96, 0.0f);
    data.mel_freq_time.resize(1280, 0.0f);
    data.mel_freq_time[456] = std::nanf("");

    NcnnInputAdapter adapter;
    auto r = adapter.Adapt(data);

    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelInputStatus::kNonFiniteMelValue);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_TRUE(r.input.face.empty());
    EXPECT_TRUE(r.input.mel.empty());
}

// ============================================================================
// metadata 透传
// ============================================================================
TEST(NcnnInputAdapterTest, MetadataPassthrough) {
    Wav2LipInputData data;
    data.face_chw.resize(6 * 96 * 96, 0.0f);
    data.mel_freq_time.resize(1280, 0.0f);
    data.metadata.pts_ms = 12345;
    data.metadata.frame_index = 42;

    NcnnInputAdapter adapter;
    auto r = adapter.Adapt(data);

    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_TRUE(r.input.metadata.pts_ms.has_value());
    EXPECT_EQ(r.input.metadata.pts_ms.value(), 12345);
    ASSERT_TRUE(r.input.metadata.frame_index.has_value());
    EXPECT_EQ(r.input.metadata.frame_index.value(), 42);
}
