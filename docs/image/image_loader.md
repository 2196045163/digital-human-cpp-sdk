

1. 这个模块是干什么的
这个模块只用来从文件或者内存中解码图片获得 cv::Mat 为之后的人脸检测等模块提供图片式的人脸素材
2. 怎么用（代码示例）
如从文件获取图片内容：
```cpp
    #include "core/image_loader.h"

    digital_human::core::ImageLoader loader;
    digital_human::core::LoadResult res = loader.LoadFromFile("face.jpg");

    if (res.success) {
        // 用 res.image (cv::Mat) 和 res.info
    } else {
        // res.status 和 res.error_message
    }
```
3. 支持什么格式
bmp jpg jpeg png
4. 错误怎么处理
解码错误一般都是以错误码和错误信息的方式返回，保存在 LoadResult 结构体中，大致是
```text
    空路径 → kEmptyPath、不存在 → kFileNotFound、目录 → kNotRegularFile、
    不支持格式 → kUnsupportedFormat、损坏 → kDecodeFailed
```
5. 怎么编译和测试
这个模块的单元测试在'D:\VM_Shared\Project\digital_human\digital_human-sdk\tests\image_loader_unit_test.cpp'
模块测试在 'D:\VM_Shared\Project\digital_human\digital_human-sdk\examples\image_loader_test.cpp'
主要就是测试模块是否能做 空路径检测、文件存在性检查、是否为普通文件和扩展名检查，然后是 能否从文件或者内存解码正确的图片，能否做批量图片的加载
编译就是在项目根目录下 使用 cmake --build build 然后运行build的bin下的测试程序
```bash
    # 构建
    cmake -S . -B build && cmake --build build

    # 手动测试
    ./build/bin/image_loader_test ./face.jpg

    # 单元测试
    ctest --test-dir build --output-on-failure
```