#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // cv::GaussianBlur, cv::warpAffine

#include "core/face_blender.h"

using namespace digital_human::core;

// ==========================================================================
// 让枚举可打印
// ==========================================================================
inline std::ostream& operator<<(std::ostream& os, FaceBlendStatus s) {
    return os << FaceBlender::StatusToString(s);
}

// ==========================================================================
// 简易断言宏（照搬 face_mask_generator_unit_test.cpp 风格）
// ==========================================================================
static int g_fail = 0;

#define EXPECT_TRUE(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; g_fail++; } } while(0)

#define EXPECT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "FAIL: " << msg \
                      << " (expected=" << (b) << ", actual=" << (a) << ")\n"; \
            g_fail++; \
        } \
    } while(0)

#define EXPECT_NEAR(a, b, tol, msg) \
    do { \
        if (std::fabs((a) - (b)) > (tol)) { \
            std::cerr << "FAIL: " << msg \
                      << " (expected≈" << (b) << ", actual=" << (a) << ")\n"; \
            g_fail++; \
        } \
    } while(0)

// ==========================================================================
// 辅助函数：构造合成测试数据（不依赖真实图片、dlib 模型）
// ==========================================================================

/// @brief 构造一张纯色 BGR 图（模拟原图）
/// @param w 宽度
/// @param h 高度
/// @param color BGR 三通道颜色
static cv::Mat MakeColorBgr(int w, int h, const cv::Scalar& color = cv::Scalar(100, 120, 150)) {
    return cv::Mat(h, w, CV_8UC3, color);
}

/// @brief 构造 96×96 合成生成图（模拟模型输出的嘴部图）
/// @param color BGR 颜色，默认与 base 不同的灰色，便于区分
static cv::Mat MakeFakeGenerated96(const cv::Scalar& color = cv::Scalar(200, 180, 160)) {
    return cv::Mat(96, 96, CV_8UC3, color);
}

/// @brief 构造 96×96 合成 mask（CV_32FC1，值范围 0~1）
/// @param val 填充值（0.0=全保留原图，1.0=全替换为生成图）
static cv::Mat MakeFakeMask96(float val = 1.0f) {
    return cv::Mat(96, 96, CV_32FC1, cv::Scalar(val));
}

/// @brief 构造 96×96 uint8 mask（CV_8UC1，0~255）
static cv::Mat MakeFakeMask96U8(int val = 255) {
    return cv::Mat(96, 96, CV_8UC1, cv::Scalar(val));
}

/// @brief 构造单位仿射矩阵（2×3 CV_64F）
///        单位矩阵意味着 warpAffine 是"原样复制"——x、y 不变
///        左上角 96×96 区域在输出图中保持不变，不会被旋转/缩放/平移
static cv::Mat MakeIdentityTransform() {
    return (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0);
}

/// @brief 检查两张图像素级近似相等（允许 ±2 的 uint8 四舍五入误差）
static bool MatApproxEqual(const cv::Mat& a, const cv::Mat& b, int tolerance = 2) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    double max_val;
    cv::minMaxLoc(diff.reshape(1), nullptr, &max_val);
    return max_val <= tolerance;
}

// ==========================================================================
int main() {
    std::cout << "=== FaceBlender Unit Tests ===\n";

    FaceBlender blender;

    // ------------------------------------------------------------------
    // 测试 1: StatusToString — 全部 14 个枚举值都返回非空字符串
    // ------------------------------------------------------------------
    std::cout << "[1] StatusToString ...\n";
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kOk).empty(),
                "kOk non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kEmptyBaseImage).empty(),
                "kEmptyBaseImage non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kEmptyGeneratedImage).empty(),
                "kEmptyGeneratedImage non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kEmptyMask).empty(),
                "kEmptyMask non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kInvalidGeneratedSize).empty(),
                "kInvalidGeneratedSize non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kInvalidBaseImageType).empty(),
                "kInvalidBaseImageType non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kInvalidGeneratedType).empty(),
                "kInvalidGeneratedType non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kInvalidMaskType).empty(),
                "kInvalidMaskType non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kInvalidInverseTransform).empty(),
                "kInvalidInverseTransform non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kInvalidOutputSize).empty(),
                "kInvalidOutputSize non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kSizeMismatch).empty(),
                "kSizeMismatch non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kWarpFailed).empty(),
                "kWarpFailed non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kBlendFailed).empty(),
                "kBlendFailed non-empty");
    EXPECT_TRUE(!FaceBlender::StatusToString(FaceBlendStatus::kOpenCvError).empty(),
                "kOpenCvError non-empty");

    // ------------------------------------------------------------------
    // 测试 2: Sharpen96 — 空图 → 返回空 Mat，不崩溃
    // ------------------------------------------------------------------
    std::cout << "[2] Sharpen96 empty ...\n";
    cv::Mat empty96 = blender.Sharpen96(cv::Mat());
    EXPECT_TRUE(empty96.empty(), "Sharpen96(empty) should return empty");

    // ------------------------------------------------------------------
    // 测试 3: Sharpen96 — 正常 96×96 CV_8UC3 → 输出尺寸和类型不变
    // ------------------------------------------------------------------
    std::cout << "[3] Sharpen96 normal ...\n";
    cv::Mat gen96 = MakeFakeGenerated96();
    cv::Mat sharp = blender.Sharpen96(gen96);
    EXPECT_TRUE(!sharp.empty(), "Sharpen96 result non-empty");
    EXPECT_EQ(sharp.cols, 96, "Sharpen96 cols == 96");
    EXPECT_EQ(sharp.rows, 96, "Sharpen96 rows == 96");
    EXPECT_EQ(sharp.type(), CV_8UC3, "Sharpen96 type == CV_8UC3");

    // ------------------------------------------------------------------
    // 测试 4: Sharpen96 — 关闭锐化 → 输出 ≈ 输入（像素不变）
    // ------------------------------------------------------------------
    std::cout << "[4] Sharpen96 disabled ...\n";
    FaceBlendOptions opt4;
    opt4.enable_sharpen = false;
    cv::Mat sharp_disabled = blender.Sharpen96(gen96, opt4);
    EXPECT_TRUE(MatApproxEqual(sharp_disabled, gen96, 0),
                "Sharpen96 disabled: output == input (pixel exact)");

    // ------------------------------------------------------------------
    // 测试 5: RestoreToOriginal — 单位矩阵 + 96×96 → 输出 512×512
    //          左上角 96×96 区域应与输入一致，其余区域黑色
    // ------------------------------------------------------------------
    std::cout << "[5] RestoreToOriginal identity ...\n";
    cv::Mat identity = MakeIdentityTransform();
    cv::Size out_sz(512, 512);
    cv::Mat restored = blender.RestoreToOriginal(gen96, identity, out_sz);
    EXPECT_TRUE(!restored.empty(), "RestoreToOriginal non-empty");
    EXPECT_EQ(restored.cols, 512, "RestoreToOriginal cols == 512");
    EXPECT_EQ(restored.rows, 512, "RestoreToOriginal rows == 512");
    EXPECT_EQ(restored.type(), CV_8UC3, "RestoreToOriginal type == CV_8UC3");

    // 验证左上角 96×96 = 输入图
    cv::Rect top_left(0, 0, 96, 96);
    cv::Mat roi = restored(top_left);
    EXPECT_TRUE(MatApproxEqual(roi, gen96, 0),
                "RestoreToOriginal: top-left 96x96 == input");

    // 验证右下角最后一行全部为 0（黑色）
    cv::Mat bottom_row = restored.row(511);
    double bmin, bmax;
    cv::minMaxLoc(bottom_row.reshape(1), &bmin, &bmax);
    EXPECT_NEAR(bmax, 0.0, 1.0, "RestoreToOriginal: bottom row ≈ 0 (black)");

    // ------------------------------------------------------------------
    // 测试 6: RestoreToOriginal — 空矩阵 → 返回空 Mat
    // ------------------------------------------------------------------
    std::cout << "[6] RestoreToOriginal empty M_inv ...\n";
    cv::Mat restored_empty = blender.RestoreToOriginal(gen96, cv::Mat(), out_sz);
    EXPECT_TRUE(restored_empty.empty(), "RestoreToOriginal with empty M_inv → empty");

    // ------------------------------------------------------------------
    // 测试 7: RestoreMaskToOriginal — CV_32FC1 输入 → CV_32FC3 输出，0~1
    // ------------------------------------------------------------------
    std::cout << "[7] RestoreMaskToOriginal CV_32FC1 ...\n";
    cv::Mat mask96_f32 = MakeFakeMask96(0.8f);
    cv::Mat restored_mask = blender.RestoreMaskToOriginal(mask96_f32, identity, out_sz);
    EXPECT_TRUE(!restored_mask.empty(), "RestoreMaskToOriginal CV_32FC1 non-empty");
    EXPECT_EQ(restored_mask.cols, 512, "RestoreMask cols == 512");
    EXPECT_EQ(restored_mask.rows, 512, "RestoreMask rows == 512");
    EXPECT_EQ(restored_mask.type(), CV_32FC3, "RestoreMask type == CV_32FC3");

    // mask 值应在 0~1 范围
    std::vector<cv::Mat> chs;
    cv::split(restored_mask, chs);
    double mm_min, mm_max;
    cv::minMaxLoc(chs[0], &mm_min, &mm_max);
    EXPECT_NEAR(mm_min, 0.0, 0.01, "RestoreMask min >= 0");
    // mask 填的 0.8，warpAffine 后值应近似为 0.8，且在 0~1 范围内
    EXPECT_NEAR(mm_max, 0.8, 0.01, "RestoreMask max ≈ 0.8");
    EXPECT_NEAR(mm_min, 0.0, 0.01, "RestoreMask min >= 0");

    // ------------------------------------------------------------------
    // 测试 8: RestoreMaskToOriginal — CV_8UC1 输入（全 255）→ 非零区域 ≈ 1.0
    // ------------------------------------------------------------------
    std::cout << "[8] RestoreMaskToOriginal CV_8UC1 ...\n";
    cv::Mat mask96_u8 = MakeFakeMask96U8(255);   // 全白 uint8
    cv::Mat restored_mask_u8 = blender.RestoreMaskToOriginal(mask96_u8, identity, out_sz);
    EXPECT_TRUE(!restored_mask_u8.empty(), "RestoreMask CV_8UC1 non-empty");
    EXPECT_EQ(restored_mask_u8.type(), CV_32FC3, "RestoreMask CV_8UC1 type == CV_32FC3");

    // 输出图中 96×96 区域（mask=255 归一化后 ≈ 1.0 的区域）的均值应接近 1.0
    cv::Mat roi_mask = restored_mask_u8(cv::Rect(0, 0, 96, 96));
    double mean_val = cv::mean(roi_mask)[0];
    // 96×96 区域的均值应接近 1.0（填充值 255→1.0）
    // 容差设为 0.02：边缘像素与黑色边框(0)做线性插值，会被轻微拉低
    EXPECT_NEAR(mean_val, 1.0, 0.02, "RestoreMask CV_8UC1: mean ≈ 1.0 in 96x96 region");

    // ------------------------------------------------------------------
    // 测试 9: BlendWithDetail — mask 全 0 → 输出 ≈ base（原图不变）
    // ------------------------------------------------------------------
    std::cout << "[9] BlendWithDetail mask=0 ...\n";
    cv::Mat base512 = MakeColorBgr(512, 512, cv::Scalar(100, 100, 100));
    cv::Mat gen512 = MakeColorBgr(512, 512, cv::Scalar(200, 200, 200));
    cv::Mat mask_zero(512, 512, CV_32FC1, cv::Scalar(0.0f));

    FaceBlendOptions opt9;
    opt9.enable_detail_restore = false;  // 关掉细节恢复和颜色匹配，纯测 alpha 公式
    opt9.enable_color_match = false;
    FaceBlendResult r9 = blender.BlendWithDetail(base512, gen512, mask_zero, opt9);
    EXPECT_TRUE(r9.success, "mask=0: success == true");
    EXPECT_EQ(r9.status, FaceBlendStatus::kOk, "mask=0: kOk");
    // mask=0 时 final 应非常接近 base（允许 1~2 的 uint8 截断误差）
    EXPECT_TRUE(MatApproxEqual(r9.final_bgr, base512, 2),
                "mask=0: final ≈ base");

    // ------------------------------------------------------------------
    // 测试 10: BlendWithDetail — mask 全 1 → 输出 ≈ generated
    // ------------------------------------------------------------------
    std::cout << "[10] BlendWithDetail mask=1 ...\n";
    cv::Mat mask_one(512, 512, CV_32FC1, cv::Scalar(1.0f));
    FaceBlendResult r10 = blender.BlendWithDetail(base512, gen512, mask_one, opt9);
    EXPECT_TRUE(r10.success, "mask=1: success == true");
    EXPECT_EQ(r10.status, FaceBlendStatus::kOk, "mask=1: kOk");
    EXPECT_TRUE(MatApproxEqual(r10.final_bgr, gen512, 2),
                "mask=1: final ≈ generated");

    // ------------------------------------------------------------------
    // 测试 11: BlendWithDetail — 单像素验证 alpha 公式
    //          base=100, generated=200, mask=0.25 → output ≈ 125
    // ------------------------------------------------------------------
    std::cout << "[11] BlendWithDetail single pixel ...\n";
    cv::Mat base1x1(1, 1, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat gen1x1(1, 1, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat mask1x1(1, 1, CV_32FC1, cv::Scalar(0.25f));

    FaceBlendResult r11 = blender.BlendWithDetail(base1x1, gen1x1, mask1x1, opt9);
    EXPECT_TRUE(r11.success, "1x1: success == true");

    // output = 200*0.25 + 100*0.75 = 50 + 75 = 125
    cv::Vec3b pixel = r11.final_bgr.at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(static_cast<double>(pixel[0]), 125.0, 2.0, "1x1 B channel ≈ 125");
    EXPECT_NEAR(static_cast<double>(pixel[1]), 125.0, 2.0, "1x1 G channel ≈ 125");
    EXPECT_NEAR(static_cast<double>(pixel[2]), 125.0, 2.0, "1x1 R channel ≈ 125");

    // ------------------------------------------------------------------
    // 测试 12: BlendMouthToOriginal — 全部合法合成输入 → 正常完成
    // ------------------------------------------------------------------
    std::cout << "[12] BlendMouthToOriginal normal ...\n";
    cv::Mat base_512 = MakeColorBgr(512, 512, cv::Scalar(100, 120, 150));
    cv::Mat gen_96 = MakeFakeGenerated96(cv::Scalar(200, 180, 160));
    cv::Mat mask_96 = MakeFakeMask96(0.5f);
    cv::Mat M_unit = MakeIdentityTransform();

    FaceBlendResult r12 = blender.BlendMouthToOriginal(base_512, gen_96, mask_96, M_unit);
    EXPECT_TRUE(r12.success, "normal: success == true");
    EXPECT_EQ(r12.status, FaceBlendStatus::kOk, "normal: kOk");
    EXPECT_TRUE(!r12.final_bgr.empty(), "normal: final_bgr non-empty");
    EXPECT_EQ(r12.final_bgr.cols, 512, "normal: final cols == 512");
    EXPECT_EQ(r12.final_bgr.rows, 512, "normal: final rows == 512");
    EXPECT_EQ(r12.final_bgr.type(), CV_8UC3, "normal: final type == CV_8UC3");

    // 调试图应该也被填充
    EXPECT_TRUE(!r12.generated_96_sharp.empty(), "normal: generated_96_sharp non-empty");
    EXPECT_TRUE(!r12.restored_face_bgr.empty(), "normal: restored_face_bgr non-empty");
    EXPECT_TRUE(!r12.restored_mask_3c.empty(), "normal: restored_mask_3c non-empty");

    // 统计信息
    EXPECT_EQ(r12.info.output_width, 512, "normal: info width == 512");
    EXPECT_EQ(r12.info.output_height, 512, "normal: info height == 512");
    EXPECT_TRUE(r12.info.time_ms >= 0.0, "normal: time_ms >= 0");

    // ------------------------------------------------------------------
    // 测试 13: BlendMouthToOriginal — 空 base_bgr → kEmptyBaseImage
    // ------------------------------------------------------------------
    std::cout << "[13] BlendMouthToOriginal empty base ...\n";
    FaceBlendResult r13 = blender.BlendMouthToOriginal(cv::Mat(), gen_96, mask_96, M_unit);
    EXPECT_TRUE(!r13.success, "empty base: success == false");
    EXPECT_EQ(r13.status, FaceBlendStatus::kEmptyBaseImage, "empty base: kEmptyBaseImage");
    EXPECT_TRUE(!r13.error_message.empty(), "empty base: error_message non-empty");

    // ------------------------------------------------------------------
    // 测试 14: BlendMouthToOriginal — 空 generated_96 → kEmptyGeneratedImage
    // ------------------------------------------------------------------
    std::cout << "[14] BlendMouthToOriginal empty generated ...\n";
    FaceBlendResult r14 = blender.BlendMouthToOriginal(base_512, cv::Mat(), mask_96, M_unit);
    EXPECT_TRUE(!r14.success, "empty gen: success == false");
    EXPECT_EQ(r14.status, FaceBlendStatus::kEmptyGeneratedImage, "empty gen: kEmptyGeneratedImage");

    // ------------------------------------------------------------------
    // 测试 15: BlendMouthToOriginal — 错误矩阵尺寸（3×3 而非 2×3）
    // ------------------------------------------------------------------
    std::cout << "[15] BlendMouthToOriginal invalid M_inv ...\n";
    cv::Mat bad_M = cv::Mat::eye(3, 3, CV_64FC1);   // 3×3，不是 2×3
    FaceBlendResult r15 = blender.BlendMouthToOriginal(base_512, gen_96, mask_96, bad_M);
    EXPECT_TRUE(!r15.success, "bad M: success == false");
    EXPECT_EQ(r15.status, FaceBlendStatus::kInvalidInverseTransform, "bad M: kInvalidInverseTransform");

    // ------------------------------------------------------------------
    // 测试 16: BlendWithDetail — 尺寸不一致 → kSizeMismatch
    // ------------------------------------------------------------------
    std::cout << "[16] BlendWithDetail size mismatch ...\n";
    cv::Mat base_256 = MakeColorBgr(256, 256, cv::Scalar(100, 100, 100));
    cv::Mat gen_512_bad = MakeColorBgr(512, 512, cv::Scalar(200, 200, 200));
    cv::Mat mask_256(256, 256, CV_32FC1, cv::Scalar(0.5f));
    FaceBlendResult r16 = blender.BlendWithDetail(base_256, gen_512_bad, mask_256, opt9);
    EXPECT_TRUE(!r16.success, "mismatch: success == false");
    EXPECT_EQ(r16.status, FaceBlendStatus::kSizeMismatch, "mismatch: kSizeMismatch");

    // ------------------------------------------------------------------
    // 测试 17a: BlendWithDetail — enable_color_match 产生与禁用时不同的输出
    // ------------------------------------------------------------------
    std::cout << "[17a] BlendWithDetail color_match enabled vs disabled ...\n";
    {
        cv::Mat mask_half(512, 512, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
        FaceBlendOptions opt_on;
        opt_on.enable_color_match = true;
        FaceBlendResult r_on = blender.BlendWithDetail(base512, gen512, mask_half, opt_on);
        EXPECT_TRUE(r_on.success, "cm-on: success");

        FaceBlendOptions opt_off;
        opt_off.enable_color_match = false;
        FaceBlendResult r_off = blender.BlendWithDetail(base512, gen512, mask_half, opt_off);
        EXPECT_TRUE(r_off.success, "cm-off: success");

        // 开启与关闭颜色匹配的结果不应完全相同（mask 半透明时有效）
        EXPECT_TRUE(!MatApproxEqual(r_on.final_bgr, r_off.final_bgr, 0),
                    "color match on/off should differ");
    }

    // ------------------------------------------------------------------
    // 测试 17b: BlendWithDetail — 零方差生成图不会崩溃（epsilon 防护）
    // ------------------------------------------------------------------
    std::cout << "[17b] BlendWithDetail zero-variance gen (epsilon safety) ...\n";
    {
        // 全 0.5 的生成图 → 各通道 std=0
        cv::Mat gen_flat(512, 512, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::Mat mask_half(512, 512, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
        FaceBlendOptions opt;
        opt.enable_color_match = true;
        FaceBlendResult r = blender.BlendWithDetail(base512, gen_flat, mask_half, opt);
        EXPECT_TRUE(r.success, "flat-gen: success");
        EXPECT_EQ(r.status, FaceBlendStatus::kOk, "flat-gen: kOk");
        // 不应产生 NaN 或崩溃，输出应仍在合法范围
        EXPECT_TRUE(!r.final_bgr.empty(), "flat-gen: non-empty output");
    }

    // ------------------------------------------------------------------
    // 测试 17b2: clamped color matching does not worsen masked mean difference
    //            clamp regression case: base=[5×0,5×255], gen=[9×113,1×255], mask all valid.
    //            clamp 可能使校正后均值比校正前更偏离 base，本测试验证非恶化保护生效。
    // ------------------------------------------------------------------
    std::cout << "[17b2] color_match_clamp_regression: clamp does not worsen mean diff ...\n";
    {
        // 构造 clamp regression 输入：10 像素 × 1 行，每通道值相同
        cv::Mat base_10(1, 10, CV_8UC3);
        cv::Mat gen_10(1, 10, CV_8UC3);
        for (int i = 0; i < 10; ++i) {
            unsigned char bv = (i < 5) ? 0 : 255;
            unsigned char gv = (i < 9) ? 113 : 255;
            base_10.at<cv::Vec3b>(0, i) = cv::Vec3b(bv, bv, bv);
            gen_10.at<cv::Vec3b>(0, i)  = cv::Vec3b(gv, gv, gv);
        }
        cv::Mat mask_full(1, 10, CV_32FC3, cv::Scalar(1.0f, 1.0f, 1.0f));

        FaceBlendOptions opt;
        opt.enable_color_match = true;
        opt.enable_detail_restore = false;

        FaceBlendResult r = blender.BlendWithDetail(base_10, gen_10, mask_full, opt);
        EXPECT_TRUE(r.success, "codex: success");

        // 在 mask 区域内计算校正前后的每通道均值误差
        cv::Mat base_f, gen_f;
        base_10.convertTo(base_f, CV_32FC3, 1.0 / 255.0);
        gen_10.convertTo(gen_f, CV_32FC3, 1.0 / 255.0);
        cv::Mat result_f;
        r.final_bgr.convertTo(result_f, CV_32FC3, 1.0 / 255.0);

        cv::Mat mask_active = mask_full > 0.01f;
        // 取第一通道作为统计 mask
        std::vector<cv::Mat> mch;
        cv::split(mask_active, mch);

        for (int ch = 0; ch < 3; ++ch) {
            std::vector<cv::Mat> bch, gch, rch;
            cv::split(base_f, bch);
            cv::split(gen_f, gch);
            cv::split(result_f, rch);

            cv::Scalar bm, gmm, rm;
            cv::meanStdDev(bch[ch], bm, cv::noArray(), mch[0]);
            cv::meanStdDev(gch[ch], gmm, cv::noArray(), mch[0]);
            cv::meanStdDev(rch[ch], rm, cv::noArray(), mch[0]);

            float before_err = std::abs(static_cast<float>(gmm[0] - bm[0]));
            float after_err  = std::abs(static_cast<float>(rm[0] - bm[0]));

            // 校正后误差不得大于校正前（允许 1/255 量化容差）
            EXPECT_TRUE(after_err <= before_err + 1.5f / 255.0f,
                        ("ch=" + std::to_string(ch) + " after_err should not exceed before_err").c_str());
        }

        // 基本契约：无 NaN，尺寸/类型/通道不变
        EXPECT_TRUE(!r.final_bgr.empty(), "codex: non-empty");
        EXPECT_EQ(r.final_bgr.cols, 10, "codex: cols unchanged");
        EXPECT_EQ(r.final_bgr.rows, 1, "codex: rows unchanged");
        EXPECT_EQ(r.final_bgr.type(), CV_8UC3, "codex: type unchanged");
        EXPECT_EQ(r.final_bgr.channels(), 3, "codex: channels unchanged");
    }

    // ------------------------------------------------------------------
    // 测试 17c: BlendWithDetail — 空 mask 安全回退
    // ------------------------------------------------------------------
    std::cout << "[17c] BlendWithDetail nearly-empty mask ...\n";
    {
        cv::Mat mask_tiny(512, 512, CV_32FC3, cv::Scalar(0.001f, 0.001f, 0.001f));
        FaceBlendOptions opt;
        opt.enable_color_match = true;
        FaceBlendResult r = blender.BlendWithDetail(base512, gen512, mask_tiny, opt);
        EXPECT_TRUE(r.success, "tiny-mask: success");
        EXPECT_EQ(r.status, FaceBlendStatus::kOk, "tiny-mask: kOk");
    }

    // ------------------------------------------------------------------
    // 测试 17d: BlendWithDetail — mask 外区域不被颜色匹配修改
    // ------------------------------------------------------------------
    std::cout << "[17d] BlendWithDetail external pixels unchanged ...\n";
    {
        // mask 只在右下角 1/4 区域有效
        cv::Mat mask_quarter = cv::Mat::zeros(512, 512, CV_32FC3);
        mask_quarter(cv::Rect(256, 256, 256, 256)) = cv::Scalar(1.0f, 1.0f, 1.0f);
        FaceBlendOptions opt;
        opt.enable_color_match = true;
        FaceBlendResult r = blender.BlendWithDetail(base512, gen512, mask_quarter, opt);
        EXPECT_TRUE(r.success, "quarter-mask: success");

        // mask 外区域（左上角）应等于 base（无 mask=0 区域不融合 gen）
        cv::Rect corner(0, 0, 100, 100);
        cv::Mat corner_before = base512(corner).clone();
        cv::Mat corner_after = r.final_bgr(corner).clone();
        double diff = cv::norm(corner_before, corner_after, cv::NORM_INF);
        EXPECT_TRUE(diff < 1.0, "external pixels should be unchanged");
    }

    // ------------------------------------------------------------------
    // 测试 17: RestoreMaskToOriginal — 打开羽化后 mask 边缘应被柔化
    // ------------------------------------------------------------------
    std::cout << "[17] RestoreMaskToOriginal with blur ...\n";
    FaceBlendOptions opt17;
    opt17.enable_mask_blur = true;
    opt17.restored_mask_blur_kernel = 7;
    cv::Mat mask17_blur = blender.RestoreMaskToOriginal(mask96_f32, identity, out_sz, opt17);
    EXPECT_TRUE(!mask17_blur.empty(), "blur: non-empty");

    // 有羽化和无羽化的 mask 不应完全一样（默认 options 也开了 blur，这里只验证不崩溃）
    FaceBlendOptions opt17b;
    opt17b.enable_mask_blur = false;
    cv::Mat mask17_noblur = blender.RestoreMaskToOriginal(mask96_f32, identity, out_sz, opt17b);
    EXPECT_TRUE(!mask17_noblur.empty(), "no-blur: non-empty");
    // 两张 mask 应该不同（羽化会改变像素值）
    EXPECT_TRUE(!MatApproxEqual(mask17_blur, mask17_noblur, 0),
                "blur vs no-blur masks should differ");

    // ------------------------------------------------------------------
    // 测试 18: BlendWithDetail — 细节恢复不崩溃（验证 enable_detail_restore 路径）
    // ------------------------------------------------------------------
    std::cout << "[18] BlendWithDetail detail restore ...\n";
    FaceBlendOptions opt18;
    opt18.enable_detail_restore = true;
    opt18.detail_strength = 0.12;
    cv::Mat mask_half(512, 512, CV_32FC1, cv::Scalar(0.5f));
    FaceBlendResult r18 = blender.BlendWithDetail(base512, gen512, mask_half, opt18);
    EXPECT_TRUE(r18.success, "detail: success == true");
    EXPECT_EQ(r18.status, FaceBlendStatus::kOk, "detail: kOk");

    // ------------------------------------------------------------------
    // 结果
    // ------------------------------------------------------------------
    std::cout << "----------------------------------------\n";
    if (g_fail == 0) {
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << g_fail << " test(s) failed.\n";
    return EXIT_FAILURE;
}
