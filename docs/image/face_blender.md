1. 这个模块干什么
此模块负责把模型输出的 96×96 嘴部图像和 96×96 alpha mask，借助 FaceAligner 的逆仿射矩阵 M_inv 映射回原图尺寸，并通过 alpha 融合公式在原图上只替换嘴部区域，使输出帧既有生成的口型又尽量保留原图的人脸细节。

模块本身不做人脸检测、对齐、mask 生成或模型推理——这四个分别由 FaceDetector、FaceAligner、FaceMaskGenerator、OutputProcessor 负责。FaceBlender 只做最后一步：回贴 + 融合。

核心链路：Sharpen96（可选锐化）→ RestoreToOriginal（warpAffine 回贴嘴部图）→ RestoreMaskToOriginal（warpAffine 回贴 mask + 可选羽化 + clamp）→ BlendWithDetail（alpha 融合 + 可选细节恢复）。一站式接口 BlendMouthToOriginal 串起以上全部。

alpha 融合公式：`out = gen × mask + base × (1 − mask)`。mask=1.0 全用生成图，mask=0.0 全保留原图，mask=0.5 各占一半。

2. 怎么用

```cpp
#include "core/face_blender.h"

using namespace digital_human::core;

FaceBlender blender;

// 方式一：分步调试（单独验证每一步的中间结果）
cv::Mat sharp = blender.Sharpen96(generated_96, options);                 // 1.锐化
cv::Mat restored_face = blender.RestoreToOriginal(sharp, M_inv,           // 2.回贴嘴部图
                                                   base_bgr.size(), options);
cv::Mat restored_mask = blender.RestoreMaskToOriginal(mask_96, M_inv,     // 3.回贴 mask
                                                       base_bgr.size(), options);
FaceBlendResult r = blender.BlendWithDetail(base_bgr, restored_face,      // 4.融合
                                             restored_mask, options);

// 方式二：一站式（Pipeline 推荐）
FaceBlendResult r = blender.BlendMouthToOriginal(
    base_bgr,                // 原图 CV_8UC3，任意尺寸
    generated_96,            // 模型输出嘴部图 CV_8UC3，96×96
    mask_96,                 // alpha mask CV_32FC1(0~1)，96×96，来自 FaceMaskGenerator
    inverse_transform,       // M_inv 2×3 CV_64F，来自 FaceAligner::inverse_transform
    FaceBlendOptions()       // 可选参数（锐化/羽化/细节恢复均可配置）
);

if (r.success) {
    cv::imwrite("final.jpg", r.final_bgr);
    // 调试图：r.generated_96_sharp / r.restored_face_bgr / r.restored_mask_3c
}
```

3. 输入/输出

- 输入：
  - `base_bgr`：原图（`CV_8UC3`，任意尺寸，原图坐标系），来自 ImageLoader
  - `generated_96`：模型输出的 96×96 嘴部图（`CV_8UC3`，对齐坐标系），来自 OutputProcessor
  - `mask_96`：96×96 alpha mask（`CV_32FC1` 或 `CV_8UC1` 或 `CV_32FC3`，0~1，对齐坐标系），来自 FaceMaskGenerator
  - `inverse_transform`：逆仿射矩阵 M_inv（2×3 `CV_64F` 或 `CV_32F`，对齐→原图），来自 FaceAligner
  - `options`：FaceBlendOptions（锐化开关/强度/sigma、mask 羽化核大小、细节恢复开关/强度/sigma、clamp 开关、插值方式）
- 输出：
  - `FaceBlendResult` 结构体：success/status/error_message + final_bgr（CV_8UC3，原图尺寸）+ 三张调试图（generated_96_sharp、restored_face_bgr、restored_mask_3c）+ info（输出图与 mask 统计 + 耗时）+ options_used

4. 错误码

```cpp
enum class FaceBlendStatus {
    kOk,                      // 成功
    kEmptyBaseImage,          // 原图为空
    kEmptyGeneratedImage,     // generated_96 为空
    kEmptyMask,               // alpha mask 为空
    kInvalidGeneratedSize,    // generated_96 尺寸不是预期的 96×96
    kInvalidBaseImageType,    // base_bgr 类型不是 CV_8UC3
    kInvalidGeneratedType,    // generated_96 类型不是 CV_8UC3
    kInvalidMaskType,         // mask 类型不支持（须 CV_32FC1/CV_8UC1/CV_32FC3）
    kInvalidInverseTransform, // M_inv 为空/不是 2×3/类型非 CV_64F 或 CV_32F
    kInvalidOutputSize,       // 输出尺寸不合法（宽或高 ≤ 0）
    kSizeMismatch,            // 融合时三张图尺寸不一致
    kWarpFailed,              // warpAffine 执行失败
    kBlendFailed,             // 融合过程失败
    kOpenCvError              // OpenCV 内部异常（被 try/catch 捕获）
};
```

5. 怎么编译和测试

```bash
# 构建
cmake -S . -B build && cmake --build build

# 可视化 example（输出 8 张调试图到 blend_output/）
./build/bin/face_blender_test ./face.jpg ./models/shape_predictor_68_face_landmarks.dat

# 单元测试（18 组合成图，不依赖 dlib 模型）
ctest --test-dir build -R face_blender_unit_test --output-on-failure
```

调试图说明：

| 图片 | 看什么 |
|------|--------|
| 01_base.jpg | 原图是否正常 |
| 02_aligned_face.jpg | 人脸是否正确对齐到 96×96 |
| 03_fake_generated_96.jpg | 假生成嘴部（蓝色）是否只在对齐图嘴部 |
| 04_generated_96_sharp.jpg | 锐化后有无噪声或过曝 |
| 05_mouth_mask_96.png | 96×96 mask 是否只覆盖嘴部 |
| 06_restored_face.jpg | 生成图是否通过 M_inv 回到原图嘴部 |
| 07_restored_mask.png | 回贴 mask 边缘是否渐变（无硬边） |
| 08_final_blend.jpg | 最终图只改变嘴部、无整块方形边界、边缘过渡自然 |
