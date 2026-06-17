# 学习复盘卡


### 模块：image_loader

| 问题 | 你的回答 |
|------|---------|
| 一句话作用 | ？ |
| 输入是什么 | ？ |
| 输出是什么 | ？ |
| 核心数据结构 | ？ |
| 核心流程 | ？ |
| 最容易失败在哪里 | ？ |
| 如何验证 | ？ |
| 后续哪个模块会调用它 | ？ |
| 面试时 2 分钟怎么讲 | ？ |

1. 这个模块的作用：
这个模块的输入是图片内容，输出是图片内容的解码结果，为上层提供图片的 info 信息

2. 输入输出分别是 std::string 文件路径 或 std::vector<unsigned char> 内存 buffer
和LoadResult（含 cv::Mat 解码图像 + ImageInfo 宽高通道 + 状态码 + 错误信息）

3. 核心数据结构主要是 图片解码结果 LoadResult 和 ImageInfo 

4. 核心流程：空路径检查 → 文件存在检查 → 普通文件检查 → 扩展名检查 → ifstream + imdecode 主解码 → cv::imread fallback → image.empty() 检查 → 填充 ImageInfo → 返回 LoadResult

5. 最容易失败的地方，这个图片加载模块从文件加载图片内容时最容易失败，因为imread不知道为什么一直不成功（你也没说清楚是什么原因！）
cv::imread 在当前环境下存在偶发性崩溃，根因未完全定位（独立调用正常，特定上下文触发），所以决定在 LoadFromFile 中改为 ifstream + imdecode 做主要读取方式。

6. 验证这个模块主要是验证它能正常从文件/内存中加载图片内容，批量加载图片内容，其中还包括图片路径的空路径、中文路径、目录等非目标文件的检查。

7. 后续应该是人脸检测和图像预处理模块之类的调用这个图像加载模块

8. 面试中的表述：
我封装了统一图像加载模块，用 LoadResult 代替异常做错误返回，批量加载保留失败项保证下标对应，头文件只暴露 opencv2/core.hpp 避免编译膨胀，用 Pimpl 隐藏实现。后续人脸检测、对齐、融合都通过这个模块获取 cv::Mat。

### 模块：face_detect 

1.这个模块叫什么？代码类名叫什么？
模块名是 人脸检测模块，代码的类名是 FaceDetector

2.它解决什么问题？（检测 + 关键点，不做身份识别）
该模块的主要功能是从加载进来的图片中检测人脸框，然后使用 `dlib` 标记人脸的68个五官关键点

3.输入是什么？输出是什么？
输入是 来自 ImageLoader 的cv::Mat 图像，也就是加载进来的图片
输出是 FaceAnalyzeResult，主要的信息是 人脸框坐标（opencv） + 68 关键点 + 嘴部 20 点 + 状态码 + 耗时

4.核心数据结构有哪些？（FaceDetectionResult、LandmarkResult 等）
ModelResult、FaceDetectionResult、LandmarkResult、FaceAnalyzeResult

5.核心流程是什么？（LoadImage → Detect → GetLandmarks → 嘴部点）
检测流程是 从文件/内存中加载图片（LoadImage） -> Detect 检测人脸框 -> GetLandmarks 对人脸框做关键点预测 -> 提取主要嘴部48-67关键点（参考项目是这样的，但是是不是还可以考虑捕捉更多的关键点？）

6.最容易失败在哪？（坐标转换、小脸、模型路径错误、通道不对）
容易失败的地方有 opencv 的人脸框坐标与dlib人脸框坐标的转换、图片中的人脸太小，dlib检测不到等

7.怎么验证？（13 组 CTest + example 可视化）
主要就是通过测试代码测试边界情况来验证功能实现

8.后续哪个模块会调用它？（人脸对齐）
人脸对齐模块 FaceAligner，它需要原图 cv::Mat + 68 点关键点，这就是本模块直接输出的这两个

9.面试 2 分钟怎么讲？
这个模块我使用 dlib 封装的 HOG+SVM 做人脸检测和 68 点 人脸五官预测，
实现细节主要是使用几个结构体 Result 代替异常、12 个状态码区分错误、Pimpl 隐藏 dlib 依赖、复用 ImageLoader 做图片输入，最后通过 13 组 CTest 自动化测试和可视化 example 做验收。