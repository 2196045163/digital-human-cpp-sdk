#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/face_mask_generator.h"

using namespace digital_human::core;

// 让枚举可打印
inline std::ostream& operator<<(std::ostream& os, FaceMaskStatus s) {
    return os << FaceMaskGenerator::StatusToString(s);
}

// ==========================================================================
// 简易断言
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
// 辅助：构造 68 个合成关键点（嘴部在 (200, 240) 附近，便于测试）
// ==========================================================================
static std::vector<cv::Point> MakeFakeLandmarks() {
    std::vector<cv::Point> pts(68);
    // 非嘴部点随便填
    for (int i = 0; i < 48; ++i) pts[i] = cv::Point(100 + i * 5, 100 + i * 3);
    // 嘴部 48~59（外嘴唇）——围成椭圆状
    pts[48] = cv::Point(180, 240);  pts[49] = cv::Point(195, 230);
    pts[50] = cv::Point(220, 225);  pts[51] = cv::Point(240, 230);
    pts[52] = cv::Point(250, 240);  pts[53] = cv::Point(240, 250);
    pts[54] = cv::Point(220, 255);  pts[55] = cv::Point(195, 250);
    pts[56] = cv::Point(190, 245);  pts[57] = cv::Point(200, 242);
    pts[58] = cv::Point(220, 240);  pts[59] = cv::Point(240, 243);
    // 嘴部 60~67（内嘴唇）
    pts[60] = cv::Point(200, 240);  pts[61] = cv::Point(210, 237);
    pts[62] = cv::Point(220, 237);  pts[63] = cv::Point(230, 237);
    pts[64] = cv::Point(230, 243);  pts[65] = cv::Point(220, 245);
    pts[66] = cv::Point(210, 243);  pts[67] = cv::Point(200, 243);
    return pts;
}

// 辅助：构造 68 个合成对齐关键点（嘴部在 96×96 图内的浮点坐标）
static std::vector<cv::Point2f> MakeFakeAlignedLandmarks() {
    std::vector<cv::Point2f> pts(68);
    for (int i = 0; i < 48; ++i) pts[i] = cv::Point2f(20.f + i * 1.0f, 20.f + i * 0.5f);
    // 嘴部 48~59（外嘴唇）——落在 96 图的下半部
    pts[48] = cv::Point2f(30.f, 60.f);  pts[49] = cv::Point2f(38.f, 55.f);
    pts[50] = cv::Point2f(48.f, 52.f);  pts[51] = cv::Point2f(58.f, 55.f);
    pts[52] = cv::Point2f(65.f, 60.f);  pts[53] = cv::Point2f(58.f, 68.f);
    pts[54] = cv::Point2f(48.f, 72.f);  pts[55] = cv::Point2f(38.f, 68.f);
    pts[56] = cv::Point2f(33.f, 64.f);  pts[57] = cv::Point2f(42.f, 62.f);
    pts[58] = cv::Point2f(52.f, 62.f);  pts[59] = cv::Point2f(60.f, 63.f);
    // 嘴部 60~67（内嘴唇）
    pts[60] = cv::Point2f(40.f, 62.f);  pts[61] = cv::Point2f(45.f, 60.f);
    pts[62] = cv::Point2f(50.f, 60.f);  pts[63] = cv::Point2f(55.f, 60.f);
    pts[64] = cv::Point2f(55.f, 64.f);  pts[65] = cv::Point2f(50.f, 65.f);
    pts[66] = cv::Point2f(45.f, 64.f);  pts[67] = cv::Point2f(40.f, 64.f);
    return pts;
}

// ==========================================================================
int main() {
    std::cout << "=== FaceMaskGenerator Unit Tests ===\n";

    FaceMaskGenerator generator;
    auto fake_lm = MakeFakeLandmarks();
    auto fake_aligned_lm = MakeFakeAlignedLandmarks();

    // ------------------------------------------------------------------
    // 测试1: StatusToString
    // ------------------------------------------------------------------
    std::cout << "[1] StatusToString ...\n";

    EXPECT_EQ(FaceMaskGenerator::StatusToString(FaceMaskStatus::kOk),
              std::string("Ok"), "StatusToString(kOk)");
    EXPECT_EQ(FaceMaskGenerator::StatusToString(FaceMaskStatus::kInvalidImageSize),
              std::string("InvalidImageSize"), "StatusToString(kInvalidImageSize)");
    EXPECT_EQ(FaceMaskGenerator::StatusToString(FaceMaskStatus::kInvalidLandmarkCount),
              std::string("InvalidLandmarkCount"), "StatusToString(kInvalidLandmarkCount)");
    EXPECT_EQ(FaceMaskGenerator::StatusToString(FaceMaskStatus::kMaskEmpty),
              std::string("MaskEmpty"), "StatusToString(kMaskEmpty)");
    EXPECT_EQ(FaceMaskGenerator::StatusToString(FaceMaskStatus::kInvalidMaskParameters),
              std::string("InvalidMaskParameters"), "StatusToString(kInvalidMaskParameters)");

    // ------------------------------------------------------------------
    // 测试2: GenerateMouthMask — 正常原图 mask
    // ------------------------------------------------------------------
    std::cout << "[2] GenerateMouthMask normal ...\n";

    cv::Size img_sz(512, 512);
    FaceMaskResult r2 = generator.GenerateMouthMask(img_sz, fake_lm);
    EXPECT_TRUE(r2.success, "normal: success == true");
    EXPECT_EQ(r2.status, FaceMaskStatus::kOk, "normal: kOk");
    EXPECT_EQ(r2.alpha_mask.cols, 512, "normal: cols == 512");
    EXPECT_EQ(r2.alpha_mask.rows, 512, "normal: rows == 512");
    EXPECT_EQ(r2.alpha_mask.type(), CV_32FC1, "normal: type == CV_32FC1");
    EXPECT_TRUE(r2.info.non_zero_count > 0, "normal: non_zero_count > 0");
    EXPECT_NEAR(r2.info.min_value, 0.0, 0.01, "normal: min >= 0");
    EXPECT_NEAR(r2.info.max_value, 1.0, 0.01, "normal: max <= 1");
    EXPECT_TRUE(r2.time_ms > 0.0, "normal: time_ms > 0");

    // ------------------------------------------------------------------
    // 测试3: GenerateMouthMask — 空尺寸
    // ------------------------------------------------------------------
    std::cout << "[3] Invalid image size ...\n";

    FaceMaskResult r3 = generator.GenerateMouthMask(cv::Size(0, 0), fake_lm);
    EXPECT_TRUE(!r3.success, "empty size: success == false");
    EXPECT_EQ(r3.status, FaceMaskStatus::kInvalidImageSize, "empty size: kInvalidImageSize");
    EXPECT_TRUE(!r3.error_message.empty(), "empty size: error_message 非空");

    FaceMaskResult r3b = generator.GenerateMouthMask(cv::Size(-10, 512), fake_lm);
    EXPECT_TRUE(!r3b.success, "negative width: success == false");

    // ------------------------------------------------------------------
    // 测试4: GenerateMouthMask — 关键点不足
    // ------------------------------------------------------------------
    std::cout << "[4] Invalid landmark count ...\n";

    std::vector<cv::Point> short_lm(10);
    FaceMaskResult r4 = generator.GenerateMouthMask(img_sz, short_lm);
    EXPECT_TRUE(!r4.success, "short lm: success == false");
    EXPECT_EQ(r4.status, FaceMaskStatus::kInvalidLandmarkCount, "short lm: kInvalidLandmarkCount");

    std::vector<cv::Point> long_lm(100);
    FaceMaskResult r4b = generator.GenerateMouthMask(img_sz, long_lm);
    EXPECT_TRUE(!r4b.success, "long lm: success == false");

    // ------------------------------------------------------------------
    // 测试5: GenerateMouthMask — 外嘴唇模式 (48~59)
    // ------------------------------------------------------------------
    std::cout << "[5] GenerateMouthMask kOuterLip ...\n";

    FaceMaskOptions opt5;
    opt5.mouth_region = MouthRegionMode::kOuterLip;
    opt5.use_convex_hull = false;
    opt5.dilate_radius = 0;
    opt5.blur_kernel_size = 1;
    FaceMaskResult r5 = generator.GenerateMouthMask(img_sz, fake_lm, opt5);
    EXPECT_TRUE(r5.success, "outer lip: success == true");
    EXPECT_TRUE(r5.info.non_zero_count > 0, "outer lip: non_zero > 0");

    // ------------------------------------------------------------------
    // 测试6: GenerateMouthMask — 不开凸包直接用 fillPoly
    // ------------------------------------------------------------------
    std::cout << "[6] GenerateMouthMask no convexHull ...\n";

    FaceMaskOptions opt6;
    opt6.use_convex_hull = false;
    FaceMaskResult r6 = generator.GenerateMouthMask(img_sz, fake_lm, opt6);
    EXPECT_TRUE(r6.success, "no hull: success == true");
    EXPECT_TRUE(r6.info.non_zero_count > 0, "no hull: non_zero > 0");

    // ------------------------------------------------------------------
    // 测试7: GenerateMouthMask — 膨胀半径负数
    // ------------------------------------------------------------------
    std::cout << "[7] Invalid dilate_radius ...\n";

    FaceMaskOptions opt7;
    opt7.dilate_radius = -1;
    FaceMaskResult r7 = generator.GenerateMouthMask(img_sz, fake_lm, opt7);
    EXPECT_TRUE(!r7.success, "negative dilate: success == false");
    EXPECT_EQ(r7.status, FaceMaskStatus::kInvalidMaskParameters, "negative dilate: kInvalidMaskParameters");

    // ------------------------------------------------------------------
    // 测试8: GenerateMouthMask — 偶数 blur_kernel 被修正
    // ------------------------------------------------------------------
    std::cout << "[8] Even blur kernel auto-fix ...\n";

    FaceMaskOptions opt8;
    opt8.blur_kernel_size = 12;  // 偶数，应被 NormalizeKernelSize 修正为 13
    FaceMaskResult r8 = generator.GenerateMouthMask(img_sz, fake_lm, opt8);
    EXPECT_TRUE(r8.success, "even blur: success == true");
    EXPECT_EQ(r8.options_used.blur_kernel_size, 13, "even blur: corrected to 13");

    // ------------------------------------------------------------------
    // 测试9: GenerateAlignedMouthMask — 正常 96 mask
    // ------------------------------------------------------------------
    std::cout << "[9] GenerateAlignedMouthMask normal ...\n";

    cv::Size aligned_sz(96, 96);
    FaceMaskResult r9 = generator.GenerateAlignedMouthMask(aligned_sz, fake_aligned_lm);
    EXPECT_TRUE(r9.success, "aligned: success == true");
    EXPECT_EQ(r9.status, FaceMaskStatus::kOk, "aligned: kOk");
    EXPECT_EQ(r9.alpha_mask.cols, 96, "aligned: cols == 96");
    EXPECT_EQ(r9.alpha_mask.rows, 96, "aligned: rows == 96");
    EXPECT_EQ(r9.alpha_mask.type(), CV_32FC1, "aligned: type == CV_32FC1");
    EXPECT_TRUE(r9.info.non_zero_count > 0, "aligned: non_zero > 0");
    // mask 不应占满整张图
    EXPECT_TRUE(r9.info.non_zero_count < 96 * 96 / 4,
                "aligned: non_zero < 1/4 of image");

    // ------------------------------------------------------------------
    // 测试10: GenerateAlignedMouthMask — 边界清零验证
    // ------------------------------------------------------------------
    std::cout << "[10] Aligned mask border clear ...\n";

    FaceMaskOptions opt10;
    opt10.border_clear = 2;
    FaceMaskResult r10 = generator.GenerateAlignedMouthMask(aligned_sz, fake_aligned_lm, opt10);
    EXPECT_TRUE(r10.success, "border: success == true");
    // 边界 2 像素应全为 0
    cv::Mat alpha = r10.alpha_mask;
    double b_min, b_max;
    cv::minMaxLoc(alpha(cv::Rect(0, 0, 96, 2)), &b_min, &b_max); // 前两行
    EXPECT_NEAR(b_max, 0.0, 0.01, "border: top rows ≈ 0");
    cv::minMaxLoc(alpha(cv::Rect(0, 94, 96, 2)), &b_min, &b_max); // 后两行
    EXPECT_NEAR(b_max, 0.0, 0.01, "border: bottom rows ≈ 0");

    // ------------------------------------------------------------------
    // 测试11: GenerateAlignedMouthMask — 空尺寸
    // ------------------------------------------------------------------
    std::cout << "[11] Aligned mask invalid size ...\n";

    FaceMaskResult r11 = generator.GenerateAlignedMouthMask(cv::Size(0, 96), fake_aligned_lm);
    EXPECT_TRUE(!r11.success, "aligned empty size: success == false");
    EXPECT_EQ(r11.status, FaceMaskStatus::kInvalidImageSize, "aligned empty size: kInvalidImageSize");

    // ------------------------------------------------------------------
    // 测试12: GenerateAlignedMouthMask — 关键点不足
    // ------------------------------------------------------------------
    std::cout << "[12] Aligned mask short landmarks ...\n";

    std::vector<cv::Point2f> short_aligned(10);
    FaceMaskResult r12 = generator.GenerateAlignedMouthMask(aligned_sz, short_aligned);
    EXPECT_TRUE(!r12.success, "aligned short lm: success == false");

    // ------------------------------------------------------------------
    // 测试13: To3ChannelMask — CV_32FC1 输入
    // ------------------------------------------------------------------
    std::cout << "[13] To3ChannelMask CV_32FC1 ...\n";

    cv::Mat mask_f32 = r9.alpha_mask.clone();
    cv::Mat mask_3c = generator.To3ChannelMask(mask_f32);
    EXPECT_EQ(mask_3c.type(), CV_32FC3, "to3ch: type == CV_32FC3");
    EXPECT_EQ(mask_3c.cols, 96, "to3ch: cols unchanged");
    EXPECT_EQ(mask_3c.rows, 96, "to3ch: rows unchanged");

    // ------------------------------------------------------------------
    // 测试14: To3ChannelMask — CV_8UC1 输入
    // ------------------------------------------------------------------
    std::cout << "[14] To3ChannelMask CV_8UC1 ...\n";

    cv::Mat mask_u8(96, 96, CV_8UC1, cv::Scalar(128));
    cv::Mat mask_3c_u8 = generator.To3ChannelMask(mask_u8);
    EXPECT_EQ(mask_3c_u8.type(), CV_32FC3, "to3ch u8: type == CV_32FC3");

    // ------------------------------------------------------------------
    // 测试15: To3ChannelMask — 空输入
    // ------------------------------------------------------------------
    std::cout << "[15] To3ChannelMask empty ...\n";

    cv::Mat empty3c = generator.To3ChannelMask(cv::Mat());
    EXPECT_TRUE(empty3c.empty(), "to3ch empty: result empty");

    // ------------------------------------------------------------------
    // 测试16: dilate_radius = 0 仍可生成基础区域
    // ------------------------------------------------------------------
    std::cout << "[16] dilate_radius = 0 ...\n";

    FaceMaskOptions opt16;
    opt16.dilate_radius = 0;
    FaceMaskResult r16 = generator.GenerateMouthMask(img_sz, fake_lm, opt16);
    EXPECT_TRUE(r16.success, "dilate=0: success == true");
    EXPECT_TRUE(r16.info.non_zero_count > 0, "dilate=0: non_zero > 0");

    // ------------------------------------------------------------------
    // 测试17: 默认 mask 参数正确（bbox 和 blur 已优化值）
    // ------------------------------------------------------------------
    std::cout << "[17] Default mask options verify ...\n";
    {
        FaceMaskOptions def;
        EXPECT_EQ(def.blur_kernel_size, 7, "default blur_kernel_size == 7");
        EXPECT_TRUE(def.bbox_expand_x < 0.40f, "default bbox_expand_x reduced");
        EXPECT_TRUE(def.bbox_expand_top < 0.60f, "default bbox_expand_top reduced");
        EXPECT_TRUE(def.bbox_expand_bottom < 0.70f, "default bbox_expand_bottom reduced");
        EXPECT_TRUE(def.use_convex_hull, "default use_convex_hull == true");
        EXPECT_TRUE(def.limit_to_mouth_bbox, "default limit_to_mouth_bbox == true");
        EXPECT_EQ(def.dilate_radius, 3, "default dilate_radius unchanged");
        EXPECT_EQ(def.border_clear, 2, "default border_clear unchanged");
    }

    // ------------------------------------------------------------------
    // 测试17b: 默认 mask 非空且尺寸类型正确（aligned 路径）
    // ------------------------------------------------------------------
    std::cout << "[17b] Default mask non-empty, correct size/type ...\n";
    {
        FaceMaskOptions def;
        FaceMaskResult r = generator.GenerateAlignedMouthMask(
            cv::Size(96, 96), fake_aligned_lm, def);
        EXPECT_TRUE(r.success, "aligned default: success");
        EXPECT_TRUE(!r.alpha_mask.empty(), "aligned default: non-empty");
        EXPECT_EQ(r.alpha_mask.cols, 96, "aligned default: width == 96");
        EXPECT_EQ(r.alpha_mask.rows, 96, "aligned default: height == 96");
        EXPECT_TRUE(r.alpha_mask.type() == CV_32FC1, "aligned default: type CV_32FC1");
        EXPECT_TRUE(r.info.non_zero_count > 0, "aligned default: non-zero pixels");
    }

    // ------------------------------------------------------------------
    // 测试17c: 数值范围始终保持 0~1
    // ------------------------------------------------------------------
    std::cout << "[17c] Value range always 0~1 ...\n";

    EXPECT_NEAR(r2.info.min_value, 0.0, 0.01, "range: normal mask min >= 0");
    EXPECT_NEAR(r2.info.max_value, 1.0, 0.01, "range: normal mask max <= 1");
    EXPECT_NEAR(r9.info.min_value, 0.0, 0.01, "range: aligned mask min >= 0");
    EXPECT_NEAR(r9.info.max_value, 1.0, 0.01, "range: aligned mask max <= 1");

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
