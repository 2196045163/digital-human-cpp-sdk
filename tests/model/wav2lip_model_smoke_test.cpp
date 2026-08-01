/// @file wav2lip_model_smoke_test.cpp
/// @brief 全链路 smoke test：真实 Wav2Lip 模型 + 合成输入 → 验证 pred 正确性
///
/// 测试范围：
///   - ModelLoader 加载真实 .param/.bin → shared_ptr<const ncnn::Net>
///   - Wav2LipInputBuilder 构建合成人脸 + Mel → Wav2LipInputData
///   - NcnnInputAdapter 适配为 ncnn::Mat
///   - ncnn::Extractor 真实推理 → pred
///   - 验证 pred 非空、shape (96,96,3)、所有值 finite
///
/// 不测试：
///   - 口型视觉质量（需要真实人脸/音频 + 视频后处理 + 人眼/指标评估）
///   - 推理性能（需要 benchmark）
///   - 音画同步（需要完整 pipeline + PTS 管理）
///
/// 本测试是"三层证据"的最后一层：证明模型边界接通。
/// 通过不代表口型正确，但不过一定代表链路哪里断了。

#include "model/model_loader.h"
#include "model/input_processor.h"
#include "model/ncnn_input_adapter.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <net.h>
#include <cmath>

using namespace digital_human::model;

TEST(Wav2LipModelSmoke, FullPipelineToPred) {
    // ---- 1. 加载模型 ----
    ModelLoader loader;
    ModelLoadOptions load_opts;
    load_opts.backend = ModelBackend::kCpu;
    load_opts.enable_warmup = true;

    auto load_r = loader.Load("models/wav2lip/wav2lip.param", load_opts);
    ASSERT_TRUE(load_r.success) << load_r.error_message;
    ASSERT_TRUE(load_r.info.warmup_performed);

    auto model = loader.AcquireModel();
    ASSERT_NE(model, nullptr);

    // ---- 2. 构造合成输入 ----
    // 人脸：水平+垂直渐变 BGR，非均匀值比纯色更能暴露通道交换 bug
    cv::Mat face(96, 96, CV_8UC3);
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            face.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 255 / 95),   // B: 左→右渐变
                static_cast<uchar>(y * 255 / 95),   // G: 上→下渐变
                static_cast<uchar>(128));            // R: 固定中灰
        }
    }

    // Mel 使用 [-4,4] 内的确定性渐变，符合上游 MelFeatureExtractor 的数值契约。
    // 1000*f+t 只适合布局单测；把它喂给真实模型会让 smoke 在错误值域上“假通过”。
    std::vector<float> mel(1280);
    for (int f = 0; f < 80; ++f) {
        for (int t = 0; t < 16; ++t) {
            const int index = f * 16 + t;
            mel[index] = -4.0f + 8.0f * static_cast<float>(index) / 1279.0f;
        }
    }

    Wav2LipInputBuilder builder;
    auto build_r = builder.Build(face, mel);
    ASSERT_TRUE(build_r.success) << build_r.error_message;
    EXPECT_GE(build_r.info.mel_min_value, -4.0f);
    EXPECT_LE(build_r.info.mel_max_value, 4.0f);

    // ---- 3. 适配为 ncnn::Mat ----
    NcnnInputAdapter adapter;
    auto adapt_r = adapter.Adapt(build_r.data);
    ASSERT_TRUE(adapt_r.success) << adapt_r.error_message;
    EXPECT_EQ(adapt_r.input.face.c, 6);
    EXPECT_EQ(adapt_r.input.face.h, 96);
    EXPECT_EQ(adapt_r.input.face.w, 96);
    EXPECT_EQ(adapt_r.input.mel.c, 1);
    EXPECT_EQ(adapt_r.input.mel.h, 80);
    EXPECT_EQ(adapt_r.input.mel.w, 16);

    // ---- 4. 真实推理 ----
    ncnn::Extractor ex = model->create_extractor();
    ex.set_light_mode(true);

    ASSERT_EQ(ex.input("mel",  adapt_r.input.mel),  0);
    ASSERT_EQ(ex.input("face", adapt_r.input.face), 0);

    ncnn::Mat pred;
    ASSERT_EQ(ex.extract("pred", pred), 0);

    // ---- 5. 三层递进验证 ----
    // 5a. 非空 — extract 返回 0 不等于拿到了数据
    ASSERT_FALSE(pred.empty());

    // 5b. shape 正确 — 不是 (96,96,3) 说明输入维度不匹配或模型内部出错
    EXPECT_EQ(pred.w, 96);
    EXPECT_EQ(pred.h, 96);
    EXPECT_EQ(pred.c, 3);

    // 5c. 值全部有限 — NaN/Inf 说明输入范围不对或模型内部除零
    bool all_finite = true;
    float pmin = 1e30f, pmax = -1e30f;
    for (int c = 0; c < pred.c && all_finite; ++c) {
        const float* ch = pred.channel(c);
        for (int i = 0; i < pred.w * pred.h; ++i) {
            if (!std::isfinite(ch[i])) { all_finite = false; break; }
            if (ch[i] < pmin) { pmin = ch[i]; }
            if (ch[i] > pmax) { pmax = ch[i]; }
        }
    }
    EXPECT_TRUE(all_finite) << "pred contains NaN or Inf";

    // 5d. 模型确实产生了输出（不为全零，验证模型在真实工作）
    // 注意：此断言不验证口型正确性，只验证模型产生了非平凡输出
    bool non_zero = (std::fabs(pmin) > 1e-10f || std::fabs(pmax) > 1e-10f);
    EXPECT_TRUE(non_zero) << "pred is all zeros — model may not be computing correctly";
}
