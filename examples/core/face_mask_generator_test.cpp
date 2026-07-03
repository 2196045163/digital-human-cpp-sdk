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

using namespace digital_human::core;

// ==========================================================================
// 测试流程（数据流向）
//
// argv[1]: image_path  ──→ ImageLoader::LoadFromFile
// argv[2]: model_path  ──→ FaceDetector::LoadLandmarkModel + DetectAndLandmark
//                              ↓
//              原图关键点可视化 ──→ 01_landmarks_68.jpg
//              原图嘴部 mask    ──→ 02_mouth_mask.png + 03_overlay + 04_blend
//                              ↓
//              FaceAligner::Align ──→ 05_aligned_face.jpg
//                              ↓
//              GenerateAlignedMouthMask ──→ 06_precise_mouth_mask_96.png
//                              ↓
//              aligned mask 可视化 ──→ 07_precise_mouth_mask_96_vis.jpg
// ==========================================================================

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
    std::cout << "=== Digital Human SDK: FacemaskGenerator Test ===\n\n";

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << " <landmark_model_path>"
                  << " [output_path]";
        return -1;
    }

    std::string output_path = (argc >= 4) ? argv[3] : "./mask_output";
    std::filesystem::create_directories(output_path);   // 不存在就创建目录

    std::string image_path = argv[1];
    ImageLoader loader;

    // ---- 1. 从文件加载图片 ----
    std::cout << "--- [1] LoadFromFile ---\n";
    LoadResult r1 = loader.LoadFromFile(image_path);
    PrintLoadResult("LoadFromFile", r1);// 打印原图尺寸、类型

    std::cout << "--- [2] LoadLandnmarkModel ---\n";
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
    // 关键点与人脸框的检测结果
    FaceAnalyzeResult far = face_detect.DetectAndLandmark(r1.image);
    // 检查是否检测到人脸
    if (far.detection.faces.empty()) {
        std::cerr << "No face detected.\n";
        return -1;
    }

    // 3. 画原图关键点 —— 在原图上每个点画一个绿色小圆
    cv::Mat vis_landmarks = r1.image.clone();
    for (const auto& pt : far.landmarks[0].landmarks) {
        // cv::circle(画布, 圆心, 半径=2px, BGR=(0,255,0)=纯绿, 线宽=-1=填充实心圆)
        cv::circle(vis_landmarks, pt, 2, cv::Scalar(0, 255, 0), -1);
    }
    // 第一张输出图 原图的关键点标注
    cv::imwrite(output_path + "/01_landmarks_68.jpg", vis_landmarks);
    std::cout << "[Output] 01_landmarks_68.jpg\n";

    // 4. GenerateMouthMask 生成原图尺寸下的 mask，保存 02_mouth_mask.png
    FaceMaskGenerator mask_gen;
    FaceMaskResult fm_res = mask_gen.GenerateMouthMask(r1.image.size(), far.landmarks[0].landmarks);
    // 检查 success，打印状态和 info（尺寸/类型/min/max/non_zero_count）
    if (!fm_res.success) {
        std::cerr << "GenerateAlignedMouthMask failed: " << fm_res.error_message << "\n";
        return -1;
    }

    std::cout << "  status:    " << FaceMaskGenerator::StatusToString(fm_res.status) << "\n";
    std::cout << "  size:      " << fm_res.info.width << "x" << fm_res.info.height
              << " type=" << fm_res.info.type << " channels=" << fm_res.info.channels << "\n"; // 这里 type() 和 channels() 是 fm_res.info 的成员，直接用就好
    std::cout << "  min/max:   " << fm_res.info.min_value << " / " << fm_res.info.max_value << "\n";
    std::cout << "  non_zero:  " << fm_res.info.non_zero_count << "\n";

    // 将 float mask 转回 uint8 可视图 → convertTo(..., CV_8UC1, 255.0)
    // 因为要输出 jpg/png 图片所以转一下到0-255
    cv::Mat mask_u8;
    fm_res.alpha_mask.convertTo(mask_u8, CV_8UC1, 255.0);
    cv::imwrite(output_path + "/02_mouth_mask.png", mask_u8);

    // 5. 热力图叠加
    // 创建与原图同尺寸的彩色（初始化为黑色）热力图
    cv::Mat heatmap(r1.image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // 把单通道 mask mask_u8 转成伪彩色（applyColorMap 返回 void，通过 dst 参数输出）
    cv::applyColorMap(mask_u8, heatmap, cv::COLORMAP_JET);

    // 将热力图叠加到原图
    cv::Mat overlay;
    cv::addWeighted(vis_landmarks, 0.6, heatmap, 0.4, 0, overlay);

    cv::imwrite(output_path + "/03_mouth_mask_overlay.jpg", overlay);

    std::cout << "[Output] 03_mouth_mask_overlay.jpg\n";

    // 6. 蓝色模拟融合 —— 验证 mask 位置和大小是否合理
    // 公式: out = blue × mask + original × (1 - mask)
    // 效果: 嘴部区域显示为蓝色，其余区域保留原图。
    // 蓝色只覆盖嘴附近 → mask 正确；蓝色蔓延到下巴/脸颊 → mask 过大

    // 创建全蓝色图（BGR = 255, 0, 0），同原图尺寸
    cv::Mat bluemap(r1.image.size(), CV_8UC3, cv::Scalar(255, 0, 0));

    // 把单通道的 fm_res.alpha_mask mask 转成三通道
    cv::Mat mask_3c = mask_gen.To3ChannelMask(fm_res.alpha_mask);

    // 转换为浮点做乘法（mask 是 CV_32FC3，图像也要 CV_32FC3）
    cv::Mat src_f32, blue_f32;
    r1.image.convertTo(src_f32, CV_32FC3, 1.0 / 255.0); // 原图
    bluemap.convertTo(blue_f32, CV_32FC3, 1.0 / 255.0); // 同尺寸全蓝图

    // mul 是逐像素乘法，Scalar(1,1,1) - mask_3c 对三通道每个通道都做 1-mask。
    cv::Mat blended_f32 = blue_f32.mul(mask_3c) + src_f32.mul(cv::Scalar(1, 1, 1) - mask_3c);

    // 转回 uint8 保存
    cv::Mat blended_u8;
    blended_f32.convertTo(blended_u8, CV_8UC3, 255.0);
    cv::imwrite(output_path + "/04_mask_blend_blue_test.jpg", blended_u8);
    std::cout << "[Output] 04_mask_blend_blue_test.jpg\n";

    // =========================================================================
    // 7. FaceAligner 对齐人脸，保存 05_aligned_face.jpg
    // =========================================================================
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

    std::cout << "  status:    " << FaceAligner::StatusToString(fa_res.status) << "\n";
    std::cout << "  aligned:   " << fa_res.aligned_face.cols << "x"
              << fa_res.aligned_face.rows
              << " type=" << fa_res.aligned_face.type() << "\n";
    std::cout << "  aligned_landmarks: " << fa_res.aligned_landmarks.size() << "\n";

    cv::imwrite(output_path + "/05_aligned_face.jpg", fa_res.aligned_face);
    std::cout << "[Output] 05_aligned_face.jpg\n";

    // =========================================================================
    // 8. GenerateAlignedMouthMask 生成 96x96 精细 mask，保存 06
    //
    // 这一步和前面第 4 步 GenerateMouthMask 的区别：
    //   - 第 4 步：输入原图关键点（int 坐标），输出原图尺寸 mask（用于可视化）
    //   - 第 8 步：输入对齐关键点（float 坐标，已用 M 变换到 96x96 坐标系），
    //     输出 96x96 精细 mask（用于后续模型融合）
    //
    // fa_res.aligned_landmarks 是 FaceAligner 把原图 68 点用 M 变换后的结果，
    // 必须传这个而不能传原图 far.landmarks——否则坐标越界
    // =========================================================================
    std::cout << "--- [4] GenerateAlignedMouthMask ---\n";

    FaceMaskResult fm_aligned = mask_gen.GenerateAlignedMouthMask(
        cv::Size(fa_res.aligned_face.cols, fa_res.aligned_face.rows), // 96x96
        fa_res.aligned_landmarks                                       // 对齐后的 68 个 Point2f
    );

    if (!fm_aligned.success) {
        std::cerr << "GenerateAlignedMouthMask failed: " << fm_aligned.error_message << "\n";
        return -1;
    }

    // 打印 mask 统计信息——用于验证类型、范围和嘴部区域是否合理
    std::cout << "  size:      " << fm_aligned.info.width << "x" << fm_aligned.info.height
              << " type=" << fm_aligned.info.type << " channels=" << fm_aligned.info.channels << "\n";
    std::cout << "  min/max:   " << fm_aligned.info.min_value << " / " << fm_aligned.info.max_value << "\n";
    std::cout << "  non_zero:  " << fm_aligned.info.non_zero_count << "\n";

    // float mask [0,1] 转 uint8 [0,255] 用于保存为 png 图片肉眼查看
    cv::Mat mask96_u8;
    fm_aligned.alpha_mask.convertTo(mask96_u8, CV_8UC1, 255.0);
    cv::imwrite(output_path + "/06_precise_mouth_mask_96.png", mask96_u8);
    std::cout << "[Output] 06_precise_mouth_mask_96.png\n";

    // =========================================================================
    // 9. 96x96 mask + 对齐关键点可视化，保存 07
    //
    // 把灰度 mask 叠加到对齐人脸上，再画上绿色关键点。
    // 肉眼检查：绿色点应该在 mask 白色/灰色区域附近 → mask 位置正确；
    // 绿色点在 mask 外面 → mask 有问题
    // =========================================================================
    std::cout << "--- [5] 96 mask visualization ---\n";

    // 第一步：单通道灰度图(0~255) 转 三通道彩色图(0~255)
    // COLOR_GRAY2BGR：把同一个灰度值复制到 B、G、R 三个通道
    cv::Mat face96_color;
    cv::cvtColor(mask96_u8, face96_color, cv::COLOR_GRAY2BGR);

    // 第二步：对齐人脸(50%) + mask彩色版(50%) 混合叠加
    // addWeighted(图A, 权重A, 图B, 权重B, 亮度补偿=0, 输出图)
    // 效果：白色 mask 区域显示为半透明白色覆盖，黑色区域完全透明（原图可见）
    cv::Mat aligned_vis;
    cv::addWeighted(fa_res.aligned_face, 0.5, face96_color, 0.5, 0, aligned_vis);

    // 第三步：在叠加图上画绿色小圆点——每个点是一个对齐后的关键点
    // 绿色点在 mask 白色区附近 → mask 围住了嘴；绿点跑到了 mask 外面 → mask 太小或偏了
    for (const auto& pt : fa_res.aligned_landmarks) {
        // cv::circle(画布, 圆心, 半径=1px, BGR=(0,255,0)=纯绿, 线宽=-1=实心圆)
        cv::circle(aligned_vis, pt, 1, cv::Scalar(0, 255, 0), -1);
    }

    cv::imwrite(output_path + "/07_precise_mouth_mask_96_vis.jpg", aligned_vis);
    std::cout << "[Output] 07_precise_mouth_mask_96_vis.jpg\n";

    return 0;
}
