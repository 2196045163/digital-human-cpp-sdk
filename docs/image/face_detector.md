1. 这个模块干什么
这个模块只对输入的图像数据 cv::Mat 做人脸框检测和五官的关键点预测

2. 怎么用（代码示例）

```cpp
#include "core/image_loader.h"
#include "core/face_detector.h"

using namespace digital_human::core;

ImageLoader loader;
LoadResult img = loader.LoadFromFile("face.jpg");
if (!img.success) { return -1; }

FaceDetector detector;
ModelLoadResult m = detector.LoadLandmarkModel(
    "models/shape_predictor_68_face_landmarks.dat");
if (!m.success) { return -1; }

FaceAnalyzeResult r = detector.DetectAndLandmark(img.image);
// r.detection.faces              → 人脸框
// r.landmarks[0].landmarks       → 68 关键点
// r.landmarks[0].mouth_landmarks → 嘴部 20 点
```

3. 支持什么输入/输出（cv::Mat → 人脸框 + 68 点 + 嘴部点）
输入：来自 ImageLoader 的cv::Mat 图像数据
输出：FaceAnalyzeResult，主要的信息是 人脸框坐标（opencv） + 68 关键点 + 嘴部 20 点 + 状态码 + 耗时

4. 错误码有哪些（FaceDetectStatus 列表）
``` cpp
enum class FaceDetectStatus {
        kOk,
        kEmptyImage,
        kInvalidImageChannels,
        kNoFace,
        kFaceTooSmall,
        kInvalidFaceRect,
        kModelPathEmpty,
        kModelFileNotFound,
        kModelLoadFailed,
        kLandmarkModelNotLoaded,
        kLandmarkFailed,
        kDlibError
    };
```

5. 怎么编译和测试（跟 image_loader 一样的命令格式）
这个模块的单元测试在'D:\VM_Shared\Project\digital_human\digital_human-sdk\src\core\face_detector.cpp'
模块测试在 'D:\VM_Shared\Project\digital_human\digital_human-sdk\examples\face_detector_test.cpp'
主要就是测试模块是否能对输入的 cv::Mat 图像数据做人脸框检测、坐标转换以及人脸五官的关键点预测
编译就是在项目根目录下 使用 cmake --build build 然后运行build的bin下的测试程序
```bash
    # 构建
    cmake -S . -B build && cmake --build build

    # 手动测试
    ./build/bin/face_detector_test ./face.jpg ./models/shape_predictor_68_face_landmarks.dat

    # 单元测试
    ctest --test-dir build --output-on-failure
```