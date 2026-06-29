#include <iostream>
#include <cmath>
#include <filesystem>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>  // cv::applyColorMap cv::circle cv::COLORMAP_JET cv::cvtColor cv::COLOR_GRAY2BGR cv::addWeighted

#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"
#include "core/face_blender.h"

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
    std::cout << "=== Digital Human SDK: FaceBlender Test ===\n\n";

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << " <landmark_model_path>"
                  << " [output_path]";
        return -1;
    }

    std::string output_path = (argc >= 4) ? argv[3] : "./blend_output";
    std::filesystem::create_directories(output_path);   // 不存在就创建目录

    std::string image_path = argv[1];
    ImageLoader loader;


    // ---- 1. 从文件加载图片 ----
    std::cout << "--- [1] LoadFromFile ---\n";
    LoadResult r1 = loader.LoadFromFile(image_path);
    PrintLoadResult("LoadFromFile", r1);// 打印原图尺寸、类型

    std::cout << "--- [2] LoadLandnmarkModel ---\n";
    FaceDetector face_detect;

    // ---- 2. 从文件加载模型 ----
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
    // 关键点与人脸框的检测结果
    FaceAnalyzeResult far = face_detect.DetectAndLandmark(r1.image);
    // 检查是否检测到人脸
    if (far.detection.faces.empty()) {
        std::cerr << "No face detected.\n";
        return -1;
    }


    // ---- 3. FaceAligner 对齐人脸 ----
    std::cout << "--- [3] FaceAligner ---\n";

    FaceAligner aligner;
    FaceAlignmentResult fa_res = aligner.Align(
        r1.image,                               // 原图
        far.detection.faces[0].rect,            // 第一张脸的人脸框
        far.landmarks[0].landmarks              // 第一张脸的 68 个原图关键点
    );

    if (!fa_res.success) {
        std::cerr << "FaceAligner failed: " << fa_res.error_message << "\n";
        return -1;
    }

    std::cout << " aligned: " << fa_res.aligned_face.cols << "x"
              << fa_res.aligned_face.rows
              << " type=" << fa_res.aligned_face.type() << "\n";
    std::cout << " aligned_landmarks: " << fa_res.aligned_landmarks.size() << "\n";

    // 这里先保存一下 01_base.jpg 和 02_aligned_face.jpg 作为参照的可肉眼观测的输入，
    // 让我们能用肉眼对比之后的融合结果和原图
    // 保存原图
    cv::imwrite(output_path + "/01_base.jpg", r1.image);
    std::cout << "[Output] 01_base.jpg\n";

    // 保存对齐人脸
    cv::imwrite(output_path + "/02_aligned_face.jpg", fa_res.aligned_face);
    std::cout << "[Output] 02_aligned_face.jpg\n";


    // ---- 4. 生成 96×96 嘴部 mask ----
    std::cout << "--- [4] GenerateAlignedMouthMask ---\n";

    FaceMaskGenerator mask_gen;
    FaceMaskResult fm_aligned = mask_gen.GenerateAlignedMouthMask(
        cv::Size(fa_res.aligned_face.cols, fa_res.aligned_face.rows),
        fa_res.aligned_landmarks
    );

    if (!fm_aligned.success) {
        std::cerr << "GenerateAlignedMouthMask failed: " << fm_aligned.error_message << "\n";
        return -1;
    }

    std::cout << "  mask: " << fm_aligned.info.width << "x" << fm_aligned.info.height
          << " type=" << fm_aligned.info.type
          << " min=" << fm_aligned.info.min_value
          << " max=" << fm_aligned.info.max_value
          << " nonzero=" << fm_aligned.info.non_zero_count << "\n";

    // 保存 96×96 mask（float→uint8）图，即融合比权重图
    cv::Mat mask96_u8;
    fm_aligned.alpha_mask.convertTo(mask96_u8, CV_8UC1, 255.0);
    cv::imwrite(output_path + "/05_mouth_mask_96.png", mask96_u8);
    std::cout << "[Output] 05_mouth_mask_96.png\n";


    // ---- 5. 构造模拟的 generated_96 ----
    // 因为目前还没真正使用模型，得不到正常的模型输出generated_96
    // 因此这里只是模拟一个模型输出
    // 5.1 把 mask 转成三通道
    cv::Mat mask_32fc3 = mask_gen.To3ChannelMask(fm_aligned.alpha_mask);

    // 5.2 搞一张全蓝色的96*96的“画布”
    cv::Mat mask_blue(96, 96, CV_8UC3, cv::Scalar(255, 0, 0));

    // 5.3 用 alpha 融合公式混合对齐人脸和全蓝画布的像素值
    // a.把两张 uint8 图转成 float(0~1)，与 mask（已是 CV_32FC3）统一精度，避免乘法截断
    // 两张图 mask_blue 和 fm_aligned.aligned_face
    cv::Mat aligned_face_f32, mask_blue_f32;
    fa_res.aligned_face.convertTo(aligned_face_f32, CV_32FC3, 1.0/255.0);
    mask_blue.convertTo(mask_blue_f32, CV_32FC3, 1.0/255.0);

    // b.用 alpha 融合公式融合两张图
    cv::Mat res_f32 = mask_blue_f32.mul(mask_32fc3) + aligned_face_f32.mul(cv::Scalar(1, 1, 1) - mask_32fc3);
    cv::Mat res_u8; // CV_32FC3 -> CV_8UC3 作为目前模拟的模型输出图像 generated_96
    res_f32.convertTo(res_u8, CV_8UC3, 255.0);

    // c.保存这张模拟的模型输出图
    cv::imwrite(output_path + "/03_fake_generated_96.jpg", res_u8);
    std::cout << "[Output] 03_fake_generated_96.jpg\n";

    
    // 前面的原图base_bgr、刚造的假图generated_96、96×96 融合权重图mask_96、M_inv 都已获得，
    // 下面做人脸的融合
    // ---- 6. 嘴部融合 ----
    FaceBlender fb;
    FaceBlendResult fb_res;
    fb_res = fb.BlendMouthToOriginal(r1.image, res_u8, 
        fm_aligned.alpha_mask, fa_res.inverse_transform);

    if (!fb_res.success) {
        std::cerr << "Blending mouth has failed!\n";
        return -1;
    }


    // ---- 7. 保存调试图 + 打印统计信息 ----
    cv::imwrite(output_path + "/04_generated_96_sharp.jpg", fb_res.generated_96_sharp);
    std::cout << "[Output] 04_generated_96_sharp.jpg\n";

    cv::imwrite(output_path + "/06_restored_face.jpg", fb_res.restored_face_bgr);
    std::cout << "[Output] 06_restored_face.jpg\n";

    // restored_mask_3c 是 CV_32FC3（float 0~1），必须先 ×255 转 CV_8UC3 再 imwrite，否则保存出来是乱码
    cv::Mat restored_mask_u8c3;
    fb_res.restored_mask_3c.convertTo(restored_mask_u8c3, CV_8UC3, 255.0);
    cv::imwrite(output_path + "/07_restored_mask.png", restored_mask_u8c3);
    std::cout << "[Output] 07_restored_mask.png\n";

    cv::imwrite(output_path + "/08_final_blend.jpg", fb_res.final_bgr);
    std::cout << "[Output] 08_final_blend.jpg\n";

    // 打印融合结果统计信息
    std::cout << "--- [6] Blend Result ---\n"
              << "  status:     " << FaceBlender::StatusToString(fb_res.status) << "\n"
              << "  final_bgr:  " << fb_res.final_bgr.cols << "x" << fb_res.final_bgr.rows
              << " type=" << fb_res.final_bgr.type()
              << " channels=" << fb_res.final_bgr.channels() << "\n"
              << "  mask_min:   " << fb_res.info.mask_min << "\n"
              << "  mask_max:   " << fb_res.info.mask_max << "\n"
              << "  mask_mean:  " << fb_res.info.mask_mean << "\n"
              << "  mask_nonzero: " << fb_res.info.mask_non_zero_count << "\n"
              << "  time_ms:    " << fb_res.info.time_ms << "\n";

    std::cout << "\n[Output] blend_output/\n";

    return 0;
}