1. 这个模块干什么
此模块根据人脸关键点生成 alpha mask，控制模型输出与原始人脸之间的融合区域：白色（1.0）区域给模型覆盖，黑色（0.0）区域保留原图，灰色（0~1）边缘过渡让融合自然。

当前 Wav2Lip 只生成嘴部，所以默认使用嘴部关键点 48~67 生成 mask。如果替换为能生成全脸表情的模型（如 MuseTalk），只需通过 FaceMaskOptions 扩大关键点范围——核心流程（凸包→填白→膨胀→羽化→归一化）不变。

模块通过两个接口区分两类 mask：原图尺寸 mask 用于可视化对比，以确认模型只融合了掩码指示的目标区域；96x96 对齐 mask 是后续模型融合时的正式输入。

2. 怎么用

```cpp
#include "core/face_mask_generator.h"

using namespace digital_human::core;

FaceMaskGenerator generator;

// 原图尺寸 mask（可视化）
FaceMaskResult r1 = generator.GenerateMouthMask(
    cv::Size(512, 512), original_landmarks  // 原图尺寸 + 原图 68 点
);

// 96x96 对齐 mask（模型融合用）
FaceMaskResult r2 = generator.GenerateAlignedMouthMask(
    cv::Size(96, 96), aligned_landmarks     // 对齐尺寸 + 对齐后的 68 点(Point2f)
);

// 单通道转三通道，供逐通道融合使用
cv::Mat mask_3c = generator.To3ChannelMask(r2.alpha_mask);
// mask_3c 类型为 CV_32FC3，用于公式: out = gen × mask + orig × (1 - mask)
```

3. 输入/输出

- 输入：人脸关键点（来自 FaceDetector 的 68 点或 FaceAligner 对齐后的 68 点）、图像尺寸、生成选项（FaceMaskOptions：嘴部范围、凸包开关、膨胀半径、模糊核大小等）
- 输出：
  - 一张 alpha mask 图（`cv::Mat`，`CV_32FC1`，0~1），即一张灰度图，每个像素的值决定了那一点上模型生成结果该占多少比例。1.0 就是完全用模型生成的，0.0 就是完全保留原图，0.5 就是各占一半
  - FaceMaskResult 结构体：包含状态码（成功/失败/原因）、上述图的统计信息（宽高、类型、最小最大值、非零像素数）、实际使用的选项参数

4. 错误码

```cpp
enum class FaceMaskStatus {
    kOk,                      // 成功
    kInvalidImageSize,        // 图像尺寸无效
    kInvalidLandmarkCount,    // 关键点数量不足 68
    kInvalidLandmarkGeometry, // 嘴部几何异常（点重合、凸包不足3点）
    kInvalidMaskParameters,   // 参数非法（膨胀半径负数等）
    kMaskEmpty,               // mask 非零区域为空
    kOpenCvError              // OpenCV 内部异常
};
```

5. 怎么编译和测试

```bash
# 构建
cmake -S . -B build && cmake --build build

# 可视化 example（输出 7 张调试图）
./build/bin/face_mask_generator_test ./face.jpg ./models/shape_predictor_68_face_landmarks.dat

# 单元测试 + 全量回归
ctest --test-dir build --output-on-failure
```
