#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/face_aligner.h"

using namespace digital_human::core;

// 让枚举可打印
inline std::ostream& operator<<(std::ostream& os, ImagePreprocessStatus s) {
    return os << FaceAligner::StatusToString(s);
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
// 辅助：生成 68 个模拟关键点（双眼水平，眼距 100px，位于 (150,120) 和 (250,120)）
static std::vector<cv::Point> MakeFakeLandmarks() {
    std::vector<cv::Point> pts(68);
    // 左眼 36-41
    pts[36]=cv::Point(140,118); pts[37]=cv::Point(145,115); pts[38]=cv::Point(150,116);
    pts[39]=cv::Point(155,115); pts[40]=cv::Point(160,118); pts[41]=cv::Point(150,120);
    // 右眼 42-47
    pts[42]=cv::Point(240,118); pts[43]=cv::Point(245,115); pts[44]=cv::Point(250,116);
    pts[45]=cv::Point(255,115); pts[46]=cv::Point(260,118); pts[47]=cv::Point(250,120);
    // 其余点随便填
    for (int i = 0; i < 36; ++i) pts[i] = cv::Point(i * 5, 100 + i * 3);
    for (int i = 48; i < 68; ++i) pts[i] = cv::Point(150 + i, 200 + i);
    return pts;
}

int main() {
    std::cout << "=== FaceAligner Unit Tests ===\n";

    // ------------------------------------------------------------------
    // 测试1: StatusToString
    // ------------------------------------------------------------------
    std::cout << "[1] StatusToString ...\n";

    EXPECT_EQ(FaceAligner::StatusToString(ImagePreprocessStatus::kOk),
              std::string("Ok"), "StatusToString(kOk)");
    EXPECT_EQ(FaceAligner::StatusToString(ImagePreprocessStatus::kEmptyImage),
              std::string("EmptyImage"), "StatusToString(kEmptyImage)");
    EXPECT_EQ(FaceAligner::StatusToString(ImagePreprocessStatus::kInvalidLandmarkCount),
              std::string("InvalidLandmarkCount"), "StatusToString(kInvalidLandmarkCount)");
    EXPECT_EQ(FaceAligner::StatusToString(ImagePreprocessStatus::kDegenerateEyeGeometry),
              std::string("DegenerateEyeGeometry"), "StatusToString(kDegenerateEyeGeometry)");
    EXPECT_EQ(FaceAligner::StatusToString(ImagePreprocessStatus::kWarpFailed),
              std::string("WarpFailed"), "StatusToString(kWarpFailed)");

    // ------------------------------------------------------------------
    // 测试2: Align — 空图
    // ------------------------------------------------------------------
    std::cout << "[2] Align empty image ...\n";

    FaceAligner aligner;
    cv::Mat empty_img;
    auto r2 = aligner.Align(empty_img, cv::Rect(10,10,100,100), MakeFakeLandmarks());
    EXPECT_TRUE(!r2.success, "empty image: success == false");
    EXPECT_EQ(r2.status, ImagePreprocessStatus::kEmptyImage, "empty image: kEmptyImage");
    EXPECT_TRUE(!r2.error_message.empty(), "empty image: error_message 非空");

    // ------------------------------------------------------------------
    // 测试3: Align — 非法图像类型
    // ------------------------------------------------------------------
    std::cout << "[3] Align invalid type ...\n";

    cv::Mat gray_img(400, 400, CV_8UC1, cv::Scalar(128));
    auto r3 = aligner.Align(gray_img, cv::Rect(50,50,200,200), MakeFakeLandmarks());
    EXPECT_TRUE(!r3.success, "gray image: success == false");
    EXPECT_EQ(r3.status, ImagePreprocessStatus::kInvalidImageType, "gray image: kInvalidImageType");

    // ------------------------------------------------------------------
    // 测试4: Align — 人脸框非法（越界）
    // ------------------------------------------------------------------
    std::cout << "[4] Align invalid face rect ...\n";

    cv::Mat valid_img(400, 400, CV_8UC3, cv::Scalar(100, 100, 100));
    // 框完全在图像外
    auto r4a = aligner.Align(valid_img, cv::Rect(500, 500, 50, 50), MakeFakeLandmarks());
    EXPECT_TRUE(!r4a.success, "rect out of bounds: success == false");
    EXPECT_EQ(r4a.status, ImagePreprocessStatus::kInvalidFaceRect, "rect out of bounds: kInvalidFaceRect");

    // 框宽高为 0
    auto r4b = aligner.Align(valid_img, cv::Rect(10, 10, 0, 0), MakeFakeLandmarks());
    EXPECT_TRUE(!r4b.success, "zero size rect: success == false");

    // ------------------------------------------------------------------
    // 测试5: Align — 关键点数量错误
    // ------------------------------------------------------------------
    std::cout << "[5] Align invalid landmark count ...\n";

    std::vector<cv::Point> short_lm(67);
    auto r5a = aligner.Align(valid_img, cv::Rect(50,50,200,200), short_lm);
    EXPECT_TRUE(!r5a.success, "67 landmarks: success == false");
    EXPECT_EQ(r5a.status, ImagePreprocessStatus::kInvalidLandmarkCount, "67 landmarks: kInvalidLandmarkCount");

    std::vector<cv::Point> long_lm(69);
    auto r5b = aligner.Align(valid_img, cv::Rect(50,50,200,200), long_lm);
    EXPECT_TRUE(!r5b.success, "69 landmarks: success == false");

    // ------------------------------------------------------------------
    // 测试6: Align — 双眼退化（重合）
    // ------------------------------------------------------------------
    std::cout << "[6] Align degenerate eyes ...\n";

    std::vector<cv::Point> bad_lm = MakeFakeLandmarks();
    // 先算左眼中心，再把右眼所有点设成和左眼中心一样——这样左右眼中心重合
    cv::Point2f left_center(0, 0);
    for (int i = 36; i <= 41; ++i) { left_center.x += bad_lm[i].x; left_center.y += bad_lm[i].y; }
    left_center.x /= 6; left_center.y /= 6;
    for (int i = 42; i <= 47; ++i) {
        bad_lm[i].x = static_cast<int>(left_center.x);
        bad_lm[i].y = static_cast<int>(left_center.y);
    }
    auto r6 = aligner.Align(valid_img, cv::Rect(50,50,200,200), bad_lm);
    EXPECT_TRUE(!r6.success, "same eyes: success == false");
    EXPECT_EQ(r6.status, ImagePreprocessStatus::kDegenerateEyeGeometry, "same eyes: kDegenerateEyeGeometry");

    // ------------------------------------------------------------------
    // 测试7: Align — 正常对齐，验证输出规格
    // ------------------------------------------------------------------
    std::cout << "[7] Align normal output specs ...\n";

    cv::Mat face_img(400, 400, CV_8UC3, cv::Scalar(128, 128, 128));
    // 在中部画一个人脸假区域，让人脸框架住
    auto r7 = aligner.Align(face_img, cv::Rect(80, 60, 240, 280), MakeFakeLandmarks());
    EXPECT_TRUE(r7.success, "normal align: success == true");
    EXPECT_EQ(r7.status, ImagePreprocessStatus::kOk, "normal align: kOk");

    // 输出尺寸
    EXPECT_EQ(r7.aligned_face.cols, 96, "output: cols == 96");
    EXPECT_EQ(r7.aligned_face.rows, 96, "output: rows == 96");
    EXPECT_EQ(r7.aligned_face.type(), CV_8UC3, "output: CV_8UC3");

    // 矩阵维度
    EXPECT_EQ(r7.transform.rows,   2, "M: rows == 2");
    EXPECT_EQ(r7.transform.cols,   3, "M: cols == 3");
    EXPECT_EQ(r7.inverse_transform.rows, 2, "M_inv: rows == 2");
    EXPECT_EQ(r7.inverse_transform.cols, 3, "M_inv: cols == 3");

    // 对齐后关键点数量
    EXPECT_EQ(r7.aligned_landmarks.size(), size_t(68), "aligned landmarks: 68");

    // source_face_rect 保留
    EXPECT_EQ(r7.source_face_rect.width,  240, "source rect width");
    EXPECT_EQ(r7.source_face_rect.height, 280, "source rect height");

    // 耗时 > 0
    EXPECT_TRUE(r7.time_ms > 0.0, "time_ms > 0");

    // ------------------------------------------------------------------
    // 测试8: 正逆变换一致性（p → M → M_inv → p，误差 < 1e-3）
    // ------------------------------------------------------------------
    std::cout << "[8] M / M_inv consistency ...\n";

    // 取嘴部第一个点（索引48）做正逆变换验证
    std::vector<cv::Point> landmarks8 = MakeFakeLandmarks();
    cv::Point2f original_pt8(landmarks8[48].x, landmarks8[48].y);

    auto r8 = aligner.Align(face_img, cv::Rect(80, 60, 240, 280), landmarks8);
    if (r8.success) {
        cv::Mat M8 = r8.transform;
        cv::Mat M_inv8 = r8.inverse_transform;

        double t8x = M8.at<double>(0,0) * original_pt8.x + M8.at<double>(0,1) * original_pt8.y + M8.at<double>(0,2);
        double t8y = M8.at<double>(1,0) * original_pt8.x + M8.at<double>(1,1) * original_pt8.y + M8.at<double>(1,2);

        double r8x = M_inv8.at<double>(0,0) * t8x + M_inv8.at<double>(0,1) * t8y + M_inv8.at<double>(0,2);
        double r8y = M_inv8.at<double>(1,0) * t8x + M_inv8.at<double>(1,1) * t8y + M_inv8.at<double>(1,2);

        EXPECT_NEAR(r8x, original_pt8.x, 1e-3, "M_inv restore x");
        EXPECT_NEAR(r8y, original_pt8.y, 1e-3, "M_inv restore y");
    } else {
        std::cout << "  (Align failed unexpectedly, skipping)\n";
    }

    // ------------------------------------------------------------------
    // 测试9: TransformLandmarks — 人工矩阵 + 已知点
    // ------------------------------------------------------------------
    std::cout << "[9] TransformLandmarks ...\n";

    // 构造一个简单矩阵：缩小 0.5 倍，平移到 (10, 20)
    cv::Mat TM = (cv::Mat_<double>(2, 3) << 0.5, 0.0, 10.0, 0.0, 0.5, 20.0);
    std::vector<cv::Point> input_pts = { cv::Point(100, 200), cv::Point(300, 400) };
    auto transformed = FaceAligner::TransformLandmarks(input_pts, TM);

    EXPECT_EQ(transformed.size(), size_t(2), "Transform: count == 2");
    EXPECT_NEAR(transformed[0].x, 60.0,  1e-6, "Transform: 100*0.5+10=60");
    EXPECT_NEAR(transformed[0].y, 120.0, 1e-6, "Transform: 200*0.5+20=120");
    EXPECT_NEAR(transformed[1].x, 160.0, 1e-6, "Transform: 300*0.5+10=160");
    EXPECT_NEAR(transformed[1].y, 220.0, 1e-6, "Transform: 400*0.5+20=220");

    // ------------------------------------------------------------------
    // 测试10: AlignBatch — 混合成功/失败，保持顺序
    // ------------------------------------------------------------------
    std::cout << "[10] AlignBatch ...\n";

    std::vector<cv::Mat> images = { valid_img, empty_img, valid_img };
    std::vector<cv::Rect> rects  = { cv::Rect(50,50,200,200), cv::Rect(10,10,100,100), cv::Rect(80,60,240,280) };
    std::vector<std::vector<cv::Point>> lm_batch = { landmarks8, landmarks8, landmarks8 };

    auto batch = aligner.AlignBatch(images, rects, lm_batch);
    EXPECT_EQ(batch.size(), size_t(3), "Batch: size == 3");
    EXPECT_TRUE(batch[0].success, "Batch[0]: success");
    EXPECT_TRUE(!batch[1].success, "Batch[1]: fail (empty image)");
    EXPECT_EQ(batch[1].status, ImagePreprocessStatus::kEmptyImage, "Batch[1]: kEmptyImage");
    EXPECT_TRUE(batch[2].success, "Batch[2]: success");

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
