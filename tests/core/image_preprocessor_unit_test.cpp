#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/image_preprocessor.h"
#include "core/face_aligner.h"  // FaceAligner::StatusToString

using namespace digital_human::core;

// 让枚举可打印
inline std::ostream& operator<<(std::ostream& os, ImagePreprocessStatus s) {
    return os << FaceAligner::StatusToString(s);
}
inline std::ostream& operator<<(std::ostream& os, ColorOrder c) {
    return os << (c == ColorOrder::kBgr ? "BGR" : "RGB");
}
inline std::ostream& operator<<(std::ostream& os, NormalizeMode n) {
    return os << (n == NormalizeMode::kNone ? "None" :
                  n == NormalizeMode::kZeroToOne ? "[0,1]" : "[-1,1]");
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
int main() {
    std::cout << "=== ImagePreprocessor Unit Tests ===\n";

    ImagePreprocessor preprocessor;

    // ------------------------------------------------------------------
    // 测试1: 空图
    // ------------------------------------------------------------------
    std::cout << "[1] Empty image ...\n";

    cv::Mat empty_img;
    auto r1 = preprocessor.Process(empty_img);
    EXPECT_TRUE(!r1.success, "empty: success == false");
    EXPECT_EQ(r1.status, ImagePreprocessStatus::kEmptyImage, "empty: kEmptyImage");
    EXPECT_TRUE(!r1.error_message.empty(), "empty: error_message 非空");
    EXPECT_TRUE(r1.image.empty(), "empty: image 为空");

    // ------------------------------------------------------------------
    // 测试2: 非法 target_size
    // ------------------------------------------------------------------
    std::cout << "[2] Invalid target_size ...\n";

    cv::Mat valid_img(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    ImagePreprocessOptions bad_size_opt;
    bad_size_opt.target_size = cv::Size(0, 96);

    auto r2a = preprocessor.Process(valid_img, bad_size_opt);
    EXPECT_TRUE(!r2a.success, "width=0: success == false");
    EXPECT_EQ(r2a.status, ImagePreprocessStatus::kInvalidTargetSize, "width=0: kInvalidTargetSize");

    bad_size_opt.target_size = cv::Size(96, -1);
    auto r2b = preprocessor.Process(valid_img, bad_size_opt);
    EXPECT_TRUE(!r2b.success, "height=-1: success == false");

    // ------------------------------------------------------------------
    // 测试3: resize 到目标尺寸
    // ------------------------------------------------------------------
    std::cout << "[3] Resize ...\n";

    cv::Mat large_img(200, 300, CV_8UC3, cv::Scalar(50, 100, 150));
    ImagePreprocessOptions opt3;
    opt3.target_size = cv::Size(96, 96);
    opt3.normalize = NormalizeMode::kNone;

    auto r3 = preprocessor.Process(large_img, opt3);
    EXPECT_TRUE(r3.success, "resize: success == true");
    EXPECT_EQ(r3.image.cols, 96, "resize: cols == 96");
    EXPECT_EQ(r3.image.rows, 96, "resize: rows == 96");
    EXPECT_EQ(r3.image.type(), CV_8UC3, "resize: still CV_8UC3");

    // ------------------------------------------------------------------
    // 测试4: 通道转换 — 灰度 (1→3)
    // ------------------------------------------------------------------
    std::cout << "[4] Channel convert: 1→3 ...\n";

    cv::Mat gray_img(100, 100, CV_8UC1, cv::Scalar(128));
    ImagePreprocessOptions opt4;
    opt4.normalize = NormalizeMode::kNone;

    auto r4 = preprocessor.Process(gray_img, opt4);
    EXPECT_TRUE(r4.success, "gray->BGR: success == true");
    EXPECT_EQ(r4.image.channels(), 3, "gray->BGR: channels == 3");
    EXPECT_EQ(r4.image.type(), CV_8UC3, "gray->BGR: CV_8UC3");

    // ------------------------------------------------------------------
    // 测试5: 通道转换 — BGRA (4→3)
    // ------------------------------------------------------------------
    std::cout << "[5] Channel convert: 4→3 ...\n";

    cv::Mat bgra_img(100, 100, CV_8UC4, cv::Scalar(10, 20, 30, 255));
    auto r5 = preprocessor.Process(bgra_img, opt4);
    EXPECT_TRUE(r5.success, "BGRA->BGR: success == true");
    EXPECT_EQ(r5.image.channels(), 3, "BGRA->BGR: channels == 3");

    // ------------------------------------------------------------------
    // 测试6: 通道异常 (2通道)
    // ------------------------------------------------------------------
    std::cout << "[6] Invalid channels ...\n";

    cv::Mat ch2_img(100, 100, CV_8UC2, cv::Scalar(128, 128));
    auto r6 = preprocessor.Process(ch2_img, opt4);
    EXPECT_TRUE(!r6.success, "2ch: success == false");
    EXPECT_EQ(r6.status, ImagePreprocessStatus::kInvalidChannels, "2ch: kInvalidChannels");

    // ------------------------------------------------------------------
    // 测试7: BGR → RGB 颜色转换
    // ------------------------------------------------------------------
    std::cout << "[7] BGR → RGB ...\n";

    cv::Mat bgr_img(10, 10, CV_8UC3, cv::Scalar(50, 100, 200)); // B=50, G=100, R=200
    ImagePreprocessOptions opt7;
    opt7.output_color = ColorOrder::kRgb;
    opt7.normalize = NormalizeMode::kNone;

    auto r7 = preprocessor.Process(bgr_img, opt7);
    EXPECT_TRUE(r7.success, "BGR→RGB: success == true");
    EXPECT_EQ(r7.color_order, ColorOrder::kRgb, "BGR→RGB: color_order == kRgb");
    // 验证通道交换：原 BGR(50,100,200) → RGB 后 B=200, G=100, R=50
    cv::Vec3b px = r7.image.at<cv::Vec3b>(5, 5);
    EXPECT_EQ(static_cast<int>(px[0]), 200, "BGR→RGB: new B == old R (200)");
    EXPECT_EQ(static_cast<int>(px[1]), 100, "BGR→RGB: new G == old G (100)");
    EXPECT_EQ(static_cast<int>(px[2]), 50,  "BGR→RGB: new R == old B (50)");

    // ------------------------------------------------------------------
    // 测试8: 归一化 — kNone（保持 uint8）
    // ------------------------------------------------------------------
    std::cout << "[8] Normalize: kNone ...\n";

    cv::Mat img8(50, 50, CV_8UC3, cv::Scalar(10, 128, 240));
    ImagePreprocessOptions opt8;
    opt8.normalize = NormalizeMode::kNone;

    auto r8 = preprocessor.Process(img8, opt8);
    EXPECT_TRUE(r8.success, "kNone: success == true");
    EXPECT_EQ(r8.image.type(), CV_8UC3, "kNone: type == CV_8UC3");
    EXPECT_EQ(r8.normalize_mode, NormalizeMode::kNone, "kNone: normalize_mode");
    EXPECT_NEAR(r8.observed_min, 10.0, 1.0, "kNone: min≈10");
    EXPECT_NEAR(r8.observed_max, 240.0, 1.0, "kNone: max≈240");

    // ------------------------------------------------------------------
    // 测试9: 归一化 — kZeroToOne → [0,1]
    // ------------------------------------------------------------------
    std::cout << "[9] Normalize: kZeroToOne ...\n";

    cv::Mat img9(50, 50, CV_8UC3, cv::Scalar(0, 128, 255));
    ImagePreprocessOptions opt9;
    opt9.normalize = NormalizeMode::kZeroToOne;

    auto r9 = preprocessor.Process(img9, opt9);
    EXPECT_TRUE(r9.success, "[0,1]: success == true");
    EXPECT_EQ(r9.image.type(), CV_32FC3, "[0,1]: type == CV_32FC3");
    EXPECT_EQ(r9.normalize_mode, NormalizeMode::kZeroToOne, "[0,1]: normalize_mode");
    EXPECT_NEAR(r9.observed_min, 0.0f, 0.01f, "[0,1]: min≈0");
    EXPECT_NEAR(r9.observed_max, 1.0f, 0.01f, "[0,1]: max≈1");

    // 验证中间像素值：128/255 ≈ 0.502
    cv::Vec3f px9 = r9.image.at<cv::Vec3f>(25, 25);
    EXPECT_NEAR(px9[0], 0.0f, 0.01f, "[0,1]: B=0");
    EXPECT_NEAR(px9[1], 128.0f/255.0f, 0.01f, "[0,1]: G=128/255");
    EXPECT_NEAR(px9[2], 1.0f, 0.01f, "[0,1]: R=1");

    // ------------------------------------------------------------------
    // 测试10: 归一化 — kMinusOneToOne → [-1,1]
    // ------------------------------------------------------------------
    std::cout << "[10] Normalize: kMinusOneToOne ...\n";

    cv::Mat img10(50, 50, CV_8UC3, cv::Scalar(0, 128, 255));
    ImagePreprocessOptions opt10;
    opt10.normalize = NormalizeMode::kMinusOneToOne;

    auto r10 = preprocessor.Process(img10, opt10);
    EXPECT_TRUE(r10.success, "[-1,1]: success == true");
    EXPECT_EQ(r10.image.type(), CV_32FC3, "[-1,1]: type == CV_32FC3");
    EXPECT_EQ(r10.normalize_mode, NormalizeMode::kMinusOneToOne, "[-1,1]: normalize_mode");
    EXPECT_NEAR(r10.observed_min, -1.0f, 0.02f, "[-1,1]: min≈-1");
    EXPECT_NEAR(r10.observed_max, 1.0f, 0.02f, "[-1,1]: max≈1");

    // 验证关键值：0→-1, 128→0, 255→1
    cv::Vec3f px10 = r10.image.at<cv::Vec3f>(25, 25);
    EXPECT_NEAR(px10[0], -1.0f, 0.02f, "[-1,1]: B=0→-1");
    EXPECT_NEAR(px10[1], 0.0f, 0.02f, "[-1,1]: G=128→0");
    EXPECT_NEAR(px10[2], 1.0f, 0.02f, "[-1,1]: R=255→1");

    // ------------------------------------------------------------------
    // 测试11: ProcessBatch — 混合成功/失败，保持顺序
    // ------------------------------------------------------------------
    std::cout << "[11] ProcessBatch ...\n";

    std::vector<cv::Mat> batch_imgs = { valid_img, empty_img, valid_img };
    ImagePreprocessOptions opt11;
    opt11.target_size = cv::Size(100, 100);  // 和 valid_img 尺寸一致，不 resize
    opt11.normalize = NormalizeMode::kNone;

    auto batch = preprocessor.ProcessBatch(batch_imgs, opt11);
    EXPECT_EQ(batch.size(), size_t(3), "Batch: size == 3");
    EXPECT_TRUE(batch[0].success, "Batch[0]: success");
    EXPECT_TRUE(!batch[1].success, "Batch[1]: fail (empty)");
    EXPECT_EQ(batch[1].status, ImagePreprocessStatus::kEmptyImage, "Batch[1]: kEmptyImage");
    EXPECT_TRUE(batch[2].success, "Batch[2]: success");
    // 成功项的尺寸正确
    EXPECT_EQ(batch[0].image.cols, 100, "Batch[0]: cols unchanged");
    EXPECT_EQ(batch[0].image.rows, 100, "Batch[0]: rows unchanged");

    // ------------------------------------------------------------------
    // 测试12: 耗时记录
    // ------------------------------------------------------------------
    std::cout << "[12] Timing ...\n";

    auto r12 = preprocessor.Process(valid_img);
    EXPECT_TRUE(r12.time_ms > 0.0, "time_ms > 0");

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
