#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <cmath>

#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"

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


int main(int argc, char* argv[]) {
    std::cout << "=== Digital Human SDK: FaceAligner Test ===\n\n";

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << " <landmark_model_path>"
                  << " [output_path]";
        return -1;
    }

    // output_path 默认 "aligned_face.jpg"
    std::string output_path = argc >= 4 ? argv[3] : "aligned_face.jpg";

    std::string image_path = argv[1];
    ImageLoader loader;

    // ---- 1. 从文件加载 ----
    std::cout << "--- [1] LoadFromFile ---\n";
    LoadResult r1 = loader.LoadFromFile(image_path);
    PrintLoadResult("LoadFromFile", r1); //原图尺寸、类型

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
    // 检查是否检测到人脸
    if (far.detection.faces.empty()) {
        std::cerr << "No face detected.\n";
        return -1;
    }

    // FaceAligner::Align(image, face_rect, landmarks) → 得 FaceAlignmentResult
    FaceAligner aligner;
    FaceAlignmentResult fa_res = aligner.Align(r1.image, // 原图
        far.detection.faces[0].rect,                    // 第一张脸的人脸框
        far.landmarks[0].landmarks);                    // 第一张脸的68个关键点

    // 6. 打印对齐结果
    std::cout << "--- [3] FaceAligner ---\n";
    // 对齐状态、对齐图尺寸（96×96）
    std::cout << "  status:   " << FaceAligner::StatusToString(fa_res.status) << "\n";
    std::cout << "  aligned:  " << fa_res.aligned_face.cols << "x"
            << fa_res.aligned_face.rows << " type=" << fa_res.aligned_face.type() << "\n";
    // M 矩阵（2×3）、M_inv 矩阵（2×3）
    std::cout << "  M (2x3):\n"
            << "    [" << fa_res.transform.at<double>(0,0) << " "
            << fa_res.transform.at<double>(0,1) << " "
            << fa_res.transform.at<double>(0,2) << "]\n"
            << "    [" << fa_res.transform.at<double>(1,0) << " "
            << fa_res.transform.at<double>(1,1) << " "
            << fa_res.transform.at<double>(1,2) << "]\n";
    std::cout << "  M_inv (2x3):\n"
            << "    [" << fa_res.inverse_transform.at<double>(0,0) << " "
            << fa_res.inverse_transform.at<double>(0,1) << " "
            << fa_res.inverse_transform.at<double>(0,2) << "]\n"
            << "    [" << fa_res.inverse_transform.at<double>(1,0) << " "
            << fa_res.inverse_transform.at<double>(1,1) << " "
            << fa_res.inverse_transform.at<double>(1,2) << "]\n";
    // 对齐后关键点数量（68）
    std::cout << "  aligned_landmarks: " << fa_res.aligned_landmarks.size() << "\n";
    // 耗时（ms）
    std::cout << "  time_ms:  " << fa_res.time_ms << "\n";

    // 7. 正逆变换验证：取 landmarks[48]，M 变换后 M_inv 回来
    cv::Point2f orig(far.landmarks[0].landmarks[48].x, far.landmarks[0].landmarks[48].y);
    cv::Mat M = fa_res.transform;
    cv::Mat M_inv = fa_res.inverse_transform;
    double tx = M.at<double>(0,0)*orig.x + M.at<double>(0,1)*orig.y + M.at<double>(0,2);
    double ty = M.at<double>(1,0)*orig.x + M.at<double>(1,1)*orig.y + M.at<double>(1,2);
    double rx = M_inv.at<double>(0,0)*tx + M_inv.at<double>(0,1)*ty + M_inv.at<double>(0,2);
    double ry = M_inv.at<double>(1,0)*tx + M_inv.at<double>(1,1)*ty + M_inv.at<double>(1,2);
    std::cout << "  verify:   orig=(" << orig.x << "," << orig.y
            << ") restored=(" << rx << "," << ry
            << ") error=(" << std::abs(rx-orig.x) << "," << std::abs(ry-orig.y) << ")\n";

    // 8. 保存对齐人脸
    cv::imwrite(output_path, fa_res.aligned_face);
    std::cout << "[Output] saved to " << output_path << "\n";

    
    

    return 0;
}