#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/face_detector.h"

using namespace digital_human::core;

// 让枚举可打印
inline std::ostream& operator<<(std::ostream& os, FaceDetectStatus s) {
    return os << FaceDetector::StatusToString(s);
}

// ==========================================================================
// 简易断言
// ==========================================================================
static int g_fail = 0;

#define EXPECT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; g_fail++; } \
    } while(0)

#define EXPECT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "FAIL: " << msg \
                      << " (expected=" << (b) << ", actual=" << (a) << ")\n"; \
            g_fail++; \
        } \
    } while(0)

// ==========================================================================
int main() {
    std::cout << "=== FaceDetector Unit Tests ===\n";

    // ------------------------------------------------------------------
    // 测试1: StatusToString
    // ------------------------------------------------------------------
    std::cout << "[1] StatusToString ...\n";

    EXPECT_EQ(FaceDetector::StatusToString(FaceDetectStatus::kOk),
              std::string("Ok"), "StatusToString(kOk)");
    EXPECT_EQ(FaceDetector::StatusToString(FaceDetectStatus::kEmptyImage),
              std::string("EmptyImage"), "StatusToString(kEmptyImage)");
    EXPECT_EQ(FaceDetector::StatusToString(FaceDetectStatus::kNoFace),
              std::string("NoFace"), "StatusToString(kNoFace)");
    EXPECT_EQ(FaceDetector::StatusToString(FaceDetectStatus::kDlibError),
              std::string("DlibError"), "StatusToString(kDlibError)");
    EXPECT_EQ(FaceDetector::StatusToString(FaceDetectStatus::kModelLoadFailed),
              std::string("ModelLoadFailed"), "StatusToString(kModelLoadFailed)");
    EXPECT_EQ(FaceDetector::StatusToString(FaceDetectStatus::kFaceTooSmall),
              std::string("FaceTooSmall"), "StatusToString(kFaceTooSmall)");

    // ------------------------------------------------------------------
    // 测试2: ClampRect
    // ------------------------------------------------------------------
    std::cout << "[2] ClampRect ...\n";

    cv::Size img_sz(100, 200);

    // 正常矩形
    cv::Rect r1 = FaceDetector::ClampRect(cv::Rect(10, 20, 30, 40), img_sz);
    EXPECT_EQ(r1.x, 10, "ClampRect: normal x");
    EXPECT_EQ(r1.y, 20, "ClampRect: normal y");
    EXPECT_EQ(r1.width,  30, "ClampRect: normal w");
    EXPECT_EQ(r1.height, 40, "ClampRect: normal h");

    // 越界矩形
    cv::Rect r2 = FaceDetector::ClampRect(cv::Rect(90, 180, 50, 50), img_sz);
    EXPECT_EQ(r2.x, 90, "ClampRect: clamp x");
    EXPECT_EQ(r2.y, 180, "ClampRect: clamp y");
    EXPECT_EQ(r2.width,  10, "ClampRect: clamp w (100-90=10)");
    EXPECT_EQ(r2.height, 20, "ClampRect: clamp h (200-180=20)");

    // 负数坐标
    cv::Rect r3 = FaceDetector::ClampRect(cv::Rect(-5, -10, 30, 40), img_sz);
    EXPECT_EQ(r3.x, 0, "ClampRect: negative x → 0");
    EXPECT_EQ(r3.y, 0, "ClampRect: negative y → 0");

    // ------------------------------------------------------------------
    // 测试3: IsValidFaceRect
    // ------------------------------------------------------------------
    std::cout << "[3] IsValidFaceRect ...\n";

    EXPECT_TRUE(FaceDetector::IsValidFaceRect(cv::Rect(10, 10, 50, 50), img_sz),
                "ValidRect: normal rect → true");
    EXPECT_TRUE(!FaceDetector::IsValidFaceRect(cv::Rect(-1, 10, 50, 50), img_sz),
                "ValidRect: negative x → false");
    EXPECT_TRUE(!FaceDetector::IsValidFaceRect(cv::Rect(10, 10, 0, 50), img_sz),
                "ValidRect: zero width → false");
    EXPECT_TRUE(!FaceDetector::IsValidFaceRect(cv::Rect(10, 10, 50, 0), img_sz),
                "ValidRect: zero height → false");
    EXPECT_TRUE(!FaceDetector::IsValidFaceRect(cv::Rect(90, 10, 50, 50), img_sz),
                "ValidRect: exceeds image width → false");

    // ------------------------------------------------------------------
    // 测试4: ExtractMouthLandmarks
    // ------------------------------------------------------------------
    std::cout << "[4] ExtractMouthLandmarks ...\n";

    // 伪造 68 个点
    std::vector<cv::Point> fake_landmarks(68);
    for (int i = 0; i < 68; ++i) {
        fake_landmarks[i] = cv::Point(i * 2, i * 2 + 1);
    }
    auto mouth = FaceDetector::ExtractMouthLandmarks(fake_landmarks);
    EXPECT_EQ(mouth.size(), size_t(20), "Mouth: 20 points");

    // 少于 68 个点
    std::vector<cv::Point> short_landmarks(10);
    auto mouth2 = FaceDetector::ExtractMouthLandmarks(short_landmarks);
    EXPECT_TRUE(mouth2.empty(), "Mouth: short input → empty");

    // ------------------------------------------------------------------
    // 测试5: Detect 空图
    // ------------------------------------------------------------------
    std::cout << "[5] Detect empty image ...\n";

    FaceDetector detector;
    cv::Mat empty_mat;
    FaceDetectionResult r5 = detector.Detect(empty_mat);
    EXPECT_TRUE(!r5.success, "Detect empty: success == false");
    EXPECT_EQ(r5.status, FaceDetectStatus::kEmptyImage, "Detect empty: kEmptyImage");
    EXPECT_TRUE(!r5.error_message.empty(), "Detect empty: error_message 非空");
    EXPECT_TRUE(r5.faces.empty(), "Detect empty: faces 为空");

    // ------------------------------------------------------------------
    // 测试6: LoadLandmarkModel
    // ------------------------------------------------------------------
    std::cout << "[6] LoadLandmarkModel ...\n";

    // 6a. 空路径
    ModelLoadResult m6a = detector.LoadLandmarkModel("");
    EXPECT_TRUE(!m6a.success, "Model: empty path → false");
    EXPECT_EQ(m6a.status, FaceDetectStatus::kModelPathEmpty, "Model: kModelPathEmpty");

    // 6b. 不存在的路径
    ModelLoadResult m6b = detector.LoadLandmarkModel("not_exist_model_12345.dat");
    EXPECT_TRUE(!m6b.success, "Model: not exist → false");
    EXPECT_EQ(m6b.status, FaceDetectStatus::kModelFileNotFound, "Model: kModelFileNotFound");

    // 6c. 正常加载模型
    ModelLoadResult m6c = detector.LoadLandmarkModel(
        "models/shape_predictor_68_face_landmarks.dat");
    EXPECT_TRUE(m6c.success, "Model: valid path → true");
    EXPECT_EQ(m6c.status, FaceDetectStatus::kOk, "Model: kOk");
    EXPECT_TRUE(detector.IsLandmarkModelLoaded(), "Model: IsLandmarkModelLoaded → true");

    // ------------------------------------------------------------------
    // 测试7: Detect 无人脸图（纯色图）
    // ------------------------------------------------------------------
    std::cout << "[7] Detect no face ...\n";

    cv::Mat solid_img(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    FaceDetectionResult r7 = detector.Detect(solid_img);
    EXPECT_TRUE(r7.success, "Detect noface: success == true");
    EXPECT_EQ(r7.status, FaceDetectStatus::kNoFace, "Detect noface: kNoFace");
    EXPECT_TRUE(r7.faces.empty(), "Detect noface: faces 为空");
    EXPECT_TRUE(r7.time_ms > 0.0, "Detect noface: time_ms > 0");

    // ------------------------------------------------------------------
    // 测试8: Detect 正常人脸
    // ------------------------------------------------------------------
    std::cout << "[8] Detect face.jpg ...\n";

    if (std::filesystem::exists("face.jpg")) {
        FaceDetectionResult r8 = detector.Detect(cv::imread("face.jpg"));
        EXPECT_TRUE(r8.success, "Detect face.jpg: success == true");
        EXPECT_EQ(r8.status, FaceDetectStatus::kOk, "Detect face.jpg: kOk");
        EXPECT_TRUE(!r8.faces.empty(), "Detect face.jpg: at least 1 face");
        EXPECT_TRUE(r8.time_ms > 0.0, "Detect face.jpg: time_ms > 0");

        if (!r8.faces.empty()) {
            EXPECT_TRUE(r8.faces[0].rect.width > 0, "Detect: face width > 0");
            EXPECT_TRUE(r8.faces[0].rect.height > 0, "Detect: face height > 0");
        }
    } else {
        std::cout << "  (face.jpg not found, skipping)\n";
    }

    // ------------------------------------------------------------------
    // 测试9: GetLandmarks — 模型未加载
    // ------------------------------------------------------------------
    std::cout << "[9] GetLandmarks model not loaded ...\n";

    FaceDetector detector2;
    cv::Mat fake_img(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Rect fake_rect(50, 50, 100, 100);
    LandmarkResult r9 = detector2.GetLandmarks(fake_img, fake_rect);
    EXPECT_TRUE(!r9.success, "GetLandmarks: no model → false");
    EXPECT_EQ(r9.status, FaceDetectStatus::kLandmarkModelNotLoaded, "GetLandmarks: kLandmarkModelNotLoaded");

    // ------------------------------------------------------------------
    // 测试10: GetLandmarks — 脸框太小
    // ------------------------------------------------------------------
    std::cout << "[10] GetLandmarks face too small ...\n";

    cv::Rect small_rect(10, 10, 30, 30);
    LandmarkResult r10 = detector.GetLandmarks(fake_img, small_rect);
    EXPECT_TRUE(!r10.success, "GetLandmarks: small face → false");
    EXPECT_EQ(r10.status, FaceDetectStatus::kFaceTooSmall, "GetLandmarks: kFaceTooSmall");

    // ------------------------------------------------------------------
    // 测试11: GetLandmarks — 正常关键点
    // ------------------------------------------------------------------
    std::cout << "[11] GetLandmarks normal ...\n";

    if (std::filesystem::exists("face.jpg") && detector.IsLandmarkModelLoaded()) {
        cv::Mat face_img = cv::imread("face.jpg");
        FaceDetectionResult detect_r = detector.Detect(face_img);
        if (!detect_r.faces.empty()) {
            LandmarkResult r11 = detector.GetLandmarks(face_img, detect_r.faces[0].rect);
            EXPECT_TRUE(r11.success, "GetLandmarks: success == true");
            EXPECT_EQ(r11.status, FaceDetectStatus::kOk, "GetLandmarks: kOk");
            EXPECT_EQ(r11.landmarks.size(), size_t(68), "GetLandmarks: 68 landmarks");
            EXPECT_EQ(r11.mouth_landmarks.size(), size_t(20), "GetLandmarks: 20 mouth points");
            EXPECT_TRUE(r11.quality_score >= 0.0, "GetLandmarks: quality_score >= 0");
            EXPECT_EQ(r11.face_rect.width, detect_r.faces[0].rect.width, "GetLandmarks: face_rect preserved");
        } else {
            std::cout << "  (no face detected in face.jpg, skipping)\n";
        }
    } else {
        std::cout << "  (face.jpg or model not available, skipping)\n";
    }

    // ------------------------------------------------------------------
    // 测试12: DetectAndLandmark — 无人脸图
    // ------------------------------------------------------------------
    std::cout << "[12] DetectAndLandmark no face ...\n";

    cv::Mat solid_img2(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    FaceAnalyzeResult r12 = detector.DetectAndLandmark(solid_img2);
    EXPECT_TRUE(r12.success, "D&L noface: success == true");
    EXPECT_EQ(r12.status, FaceDetectStatus::kNoFace, "D&L noface: kNoFace");
    EXPECT_TRUE(r12.landmarks.empty(), "D&L noface: landmarks empty");

    // ------------------------------------------------------------------
    // 测试13: DetectAndLandmark — 正常人脸
    // ------------------------------------------------------------------
    std::cout << "[13] DetectAndLandmark normal ...\n";

    if (std::filesystem::exists("face.jpg") && detector.IsLandmarkModelLoaded()) {
        cv::Mat face_img2 = cv::imread("face.jpg");
        FaceAnalyzeResult r13 = detector.DetectAndLandmark(face_img2);
        EXPECT_TRUE(r13.success, "D&L: success == true");
        EXPECT_EQ(r13.status, FaceDetectStatus::kOk, "D&L: kOk");
        EXPECT_TRUE(!r13.detection.faces.empty(), "D&L: faces not empty");
        EXPECT_EQ(r13.landmarks.size(), r13.detection.faces.size(),
                  "D&L: landmarks count == faces count");
        EXPECT_TRUE(r13.time_ms > 0.0, "D&L: time_ms > 0");
    } else {
        std::cout << "  (face.jpg or model not available, skipping)\n";
    }

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
