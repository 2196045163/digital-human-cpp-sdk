#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/face_detector.h"
#include "core/image_loader.h"

using namespace digital_human::core;

/// @brief 打印 LoadResult
static void PrintLoadResult(const std::string& label, const LoadResult& r) {
    std::cout << "[" << label << "]\n";
    std::cout << "  success: " << (r.success ? "YES" : "NO") << "\n";
    std::cout << "  status:  " << ImageLoader::StatusToString(r.status) << "\n";
    if (!r.success) {
        std::cout << "  error:   " << r.error_message << "\n";
    } else {
        std::cout << "  size:    " << r.info.width << "x" << r.info.height
                  << "  channels=" << r.info.channels
                  << "  format=" << r.info.format << "\n";
        std::cout << "  path:    " << r.info.source_path << "\n";
    }
    std::cout << std::endl;
}

/// @brief 打印 ModelLoadResult
static void PrintModelResult(const std::string& label, const ModelLoadResult& r) {
    std::cout << "[" << label << "]\n";
    std::cout << "  success: " << (r.success ? "YES" : "NO") << "\n";
    std::cout << "  status:  " << FaceDetector::StatusToString(r.status) << "\n";
    if (!r.success) {
        std::cout << "  error:   " << r.error_message << "\n";
    } else {
        std::cout << "  path:    " << r.model_path << "\n";
    }
    std::cout << std::endl;
}


int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: FaceDetector Test ===\n\n";

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << " <landmark_model_path>"
                  << " [output_path]";
        return -1;
    }
    std::string output_path = (argc >= 4) ? argv[3] : "face_detector_result.jpg";

    std::string image_path = argv[1];
    ImageLoader loader;

    // ---- 1. 从文件加载 ----
    std::cout << "--- [1] LoadFromFile ---\n";
    LoadResult r1 = loader.LoadFromFile(image_path);
    PrintLoadResult("LoadFromFile", r1);

    std::cout << "--- [2] LoadLadnmarkModel ---\n";
    FaceDetector face_detect;
    // 加载 landmark模型
    std::string landmark_model_path = argv[2];
    ModelLoadResult mlr = face_detect.LoadLandmarkModel(landmark_model_path);
    PrintModelResult("ModelLoadResult", mlr);
    if (mlr.success == false) {
        std::cerr << "Model load failed, cannot continue.\n";
        return -1;
    }

    // 检测并标记关键点
    if (!r1.success) {
        std::cerr << "Image load failed, cannot continue.\n";
        return -1;
    }
    FaceAnalyzeResult far = face_detect.DetectAndLandmark(r1.image);

    // 打印检测结果（人脸数、关键点数、耗时）
    std::cout << "--- [3] DetectAndLandmark ---\n";
    std::cout << "  faces:  " << far.detection.faces.size() << "\n";
    std::cout << "  landmarks: " << far.landmarks.size() << "\n";
    std::cout << "  time:   " << far.time_ms << " ms\n";

    for (size_t i = 0; i < far.landmarks.size(); ++i) {
        std::cout << "  Face " << i << ": "
                << far.landmarks[i].landmarks.size() << " pts, "
                << "mouth=" << far.landmarks[i].mouth_landmarks.size()
                << " score=" << far.landmarks[i].quality_score << "\n";
    }

    // 在图上画框、画点、画嘴部线
    cv::Mat draw_img = r1.image.clone();

    // 画蓝色人脸框
    for (const auto& face : far.detection.faces) {
        cv::rectangle(draw_img, face.rect, cv::Scalar(255, 0, 0), 2);
    }

    // 画绿色关键点 + 红色嘴部连线
    for (const auto& lm : far.landmarks) {
        if (!lm.success) continue;

        // 绿色关键点
        for (const auto& pt : lm.landmarks) {
            cv::circle(draw_img, pt, 1, cv::Scalar(0, 255, 0), -1);
        }

        // 红色嘴部连线 (mouth[0-11] 外轮廓, mouth[12-19] 内轮廓)
        const auto& mouth = lm.mouth_landmarks;
        if (mouth.size() == 20) {
            for (int i = 0; i < 12; ++i) {
                cv::line(draw_img, mouth[i], mouth[(i + 1) % 12], cv::Scalar(0, 0, 255), 1);
            }
            for (int i = 12; i < 20; ++i) {
                cv::line(draw_img, mouth[i], mouth[(i + 1 == 20) ? 12 : i + 1], cv::Scalar(0, 0, 255), 1);
            }
        }
    }

    // cv::imwrite 保存
    cv::imwrite(output_path, draw_img);
    std::cout << "[Output] saved to " << output_path << "\n";

    return 0;
}