#pragma once

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <opencv2/core.hpp>


namespace digital_human {
namespace core {


    enum class ImageLoadStatus {
        kOk,
        kEmptyPath,
        kFileNotFound,
        kNotRegularFile,
        kUnsupportedFormat,
        kFileOpenFailed,
        kEmptyBuffer,
        kDecodeFailed,
        kEmptyImage,
        kOpenCvError
    };

    // Color     -> cv::IMREAD_COLOR BGR 三字节彩色
    // Grayscale -> cv::IMREAD_GRAYSCALE 灰色 1字节
    // Unchanged -> cv::IMREAD_UNCHANGED 不修改
    enum class ImageReadMode {
        kColor,
        kGrayscale,
        kUnchanged
    };


    struct ImageLoadOptions {
        ImageReadMode read_mode = ImageReadMode::kColor; // 控制读取模式
        bool check_extension = true;                     // 默认检查格式，避免明显不是图片的文件进入 OpenCV
        bool use_imdecode_fallback = true;               // 用于 Windows 中文路径或 `cv::imread` 失败时的二次解码尝试
    };

    struct ImageInfo {
        int width = 0;              // cv Mat 的宽度 col
        int height = 0;             // cv Mat 的高度 row
        int channels = 0;           // 通道数 灰度0 彩色3  BGRA 4
        int type = 0;               // 类型编码 cv::Mat::type()
        int depth = 0;              // cv::Mat::depth() 单通道数据类型: CV_8U=0, CV_32F=5
        size_t elem_size = 0;       // cv::Mat::elemSize() 每个像素占多少字节 (BGR图 = 3)
        bool is_continuous = false; // 内存是否连续 (没有填充/ROI裁剪) 对应 `cv::Mat::isContinuous()`
        std::string format;         // 来自扩展名，例如 `jpg`
        std::string source_path;    // 来源路径
    };

    // - 成功：`success == true`，`status == Ok`，`image` 非空，`error_message` 为空。
    // - 失败：`success == false`，`status != Ok`，`image` 为空，`error_message` 非空。
    struct LoadResult {
        bool success = false;
        ImageLoadStatus status = ImageLoadStatus::kDecodeFailed;
        std::string error_message = "";
        cv::Mat image;
        ImageInfo info;
    };


    // - 采用 Pimpl，和参考源码风格接近，也能隐藏 `filesystem` 和 OpenCV 解码细节。
    // - 不用异常作为主要错误返回，便于调试和后续模块判断。
    // - 保留 `static` 工具函数，方便 tests 单独验证。
    class ImageLoader {
    public:
        ImageLoader();
        ~ImageLoader();

        ImageLoader(const ImageLoader&) = delete;
        ImageLoader& operator=(const ImageLoader&) = delete;

        ImageLoader(ImageLoader&&) noexcept;
        ImageLoader& operator=(ImageLoader&&) noexcept;

        LoadResult LoadFromFile(
            const std::string& file_path,
            const ImageLoadOptions& options = ImageLoadOptions()
        );

        LoadResult LoadFromMemory(
            const std::vector<unsigned char>& buffer,
            const ImageLoadOptions& options = ImageLoadOptions()
        );

        std::vector<LoadResult> LoadBatch(
            const std::vector<std::string>& file_paths,
            const ImageLoadOptions& options = ImageLoadOptions()
        );

        static bool IsSupportedFormat(const std::string& file_path);
        static std::vector<std::string> GetSupportedExtensions();
        static ImageInfo GetImageInfo(
            const cv::Mat& image,
            const std::string& source_path = "",
            const std::string& format = ""
        );
        static std::string StatusToString(ImageLoadStatus status);

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl_;
    };



} // namespace core
} // namespace digital_human
