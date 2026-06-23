# FaceAligner + ImagePreprocessor 图像预处理模块

## 1. 这个模块干什么

图像预处理模块拆成两个类，各自一个职责：

- **FaceAligner（几何对齐）**：用双眼关键点（索引 36-41、42-47）计算旋转角度、缩放比例和平移量，通过 2×3 仿射矩阵 M 把原图中歪斜的人脸变换到标准坐标系（默认 96×96），输出对齐人脸、正逆矩阵和对齐后的关键点。
- **ImagePreprocessor（像素预处理）**：把对齐人脸按模型要求调整尺寸、颜色顺序（BGR/RGB）、数据类型（uint8/float32）和数值范围（[0,1]/[-1,1]），输出可直接喂给模型的标准化图像。

> FaceAligner 回答"人脸在哪、怎么摆正"；ImagePreprocessor 回答"像素长什么样"。

## 2. 怎么用

```cpp
#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/image_preprocessor.h"

using namespace digital_human::core;

// 1. 加载图片
ImageLoader loader;
LoadResult img = loader.LoadFromFile("face.jpg");
if (!img.success) { return -1; }

// 2. 检测人脸 + 关键点
FaceDetector detector;
detector.LoadLandmarkModel("models/shape_predictor_68_face_landmarks.dat");
FaceAnalyzeResult detect_result = detector.DetectAndLandmark(img.image);
if (detect_result.detection.faces.empty()) { return -1; }

// 3. 人脸对齐
FaceAligner aligner;
FaceAlignmentResult align_result = aligner.Align(
    img.image,
    detect_result.detection.faces[0].rect,
    detect_result.landmarks[0].landmarks
);
// align_result.aligned_face      → 96×96 CV_8UC3 BGR
// align_result.transform         → M（原图→对齐图）
// align_result.inverse_transform → M_inv（对齐图→原图）
// align_result.aligned_landmarks → 对齐后的 68 个关键点

// 4. 像素预处理
ImagePreprocessor preprocessor;
ImagePreprocessOptions opt;
opt.output_color = ColorOrder::kBgr;
opt.normalize = NormalizeMode::kZeroToOne;
ImagePreprocessResult prep_result = preprocessor.Process(align_result.aligned_face, opt);
// prep_result.image → 96×96 CV_32FC3 BGR [0,1]，可直接喂给 Wav2Lip
```

## 3. 输入/输出

| 类 | 输入 | 来源 | 输出 |
|------|------|------|------|
| FaceAligner | cv::Mat（原图）+ cv::Rect（人脸框）+ 68个cv::Point（关键点） | ImageLoader + FaceDetector | FaceAlignmentResult：96×96 对齐脸 + M + M_inv + 对齐后关键点 |
| ImagePreprocessor | cv::Mat（对齐脸） | FaceAligner | ImagePreprocessResult：指定尺寸/颜色/类型/范围的标准化图像 + observed_min/max |

## 4. 错误码

```cpp
enum class ImagePreprocessStatus {
    kOk,                    // 成功
    kEmptyImage,            // 输入图像为空
    kInvalidImageType,      // 图像类型不是 CV_8UC3
    kInvalidChannels,       // 不支持的通道数（不是 1/3/4）
    kInvalidFaceRect,       // 人脸框越界或尺寸非法
    kInvalidLandmarkCount,  // 关键点数量 != 68
    kDegenerateEyeGeometry, // 双眼距离过小，无法计算缩放比
    kInvalidTargetSize,     // 目标尺寸 <= 0
    kInvalidAffineMatrix,   // 仿射矩阵非法
    kWarpFailed,            // cv::warpAffine 执行失败
    kOpenCvError            // OpenCV 内部异常
};
```

## 5. 怎么编译和测试

```bash
# 构建
cmake -S . -B build && cmake --build build

# 示例程序（分阶段验证）
./build/bin/face_aligner_test ./face.jpg ./models/shape_predictor_68_face_landmarks.dat
./build/bin/image_preprocessor_test ./face.jpg ./models/shape_predictor_68_face_landmarks.dat

# 全量 CTest（4 个测试程序，40+ 组测试用例）
ctest --test-dir build --output-on-failure
```
