
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "core/image_loader.h"

namespace digital_human {
namespace core {

namespace {
    static int ToOpenCvFlag(ImageReadMode mode) {
        switch (mode) {
            case ImageReadMode::kColor:     return cv::IMREAD_COLOR;
            case ImageReadMode::kGrayscale: return cv::IMREAD_GRAYSCALE;
            case ImageReadMode::kUnchanged: return cv::IMREAD_UNCHANGED;
            default:                        return cv::IMREAD_COLOR;
        }
    }
} // 匿名命名空间放工具函数

    struct ImageLoader::Impl {
        LoadResult LoadFromFile(const std::string &file_path, 
                                const ImageLoadOptions &options) {
            // TODO
            LoadResult load_res;
            if (file_path.empty()) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kEmptyPath;
                load_res.error_message = "The filepath is empty!";
                return load_res;
            }

            std::filesystem::path fs_path;
            try {
                fs_path = std::filesystem::path(file_path);
            } catch (const std::filesystem::filesystem_error& e) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kFileNotFound;
                load_res.error_message = std::string("Invalide file path: ") + e.what();
                return load_res;
            }

            // exists 检查 — 用 error_code 不抛异常
            std::error_code ec;
            if (!std::filesystem::exists(fs_path, ec)) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kFileNotFound;
                load_res.error_message = "File does not exist: " + file_path;
                return load_res;
            }
            // ec 非零也要处理（权限不够等）
            if (ec) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kFileNotFound;
                load_res.error_message = "Failed to check file: " + ec.message();
                return load_res;
            }

            // is_regular_file 检查 → kNotRegularFile
            if (!std::filesystem::is_regular_file(file_path, ec)) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kNotRegularFile;
                load_res.error_message = "Does not regular file: " + file_path;
                return load_res;
            }
            if (ec) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kFileNotFound;
                load_res.error_message = "Failed to check file: " + ec.message();
                return load_res;
            }

            // 扩展名检查（若 check_extension==true）→ kUnsupportedFormat
            std::string format = "";
            if (options.check_extension) {
                std::string extension = fs_path.extension().string(); // 如 ".jpg"
                // ... 转小写、去点号 ...
                std::transform(extension.begin(), extension.end(), extension.begin(),
                                [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });

                if (!extension.empty() && extension[0] == '.') {
                    extension.erase(0, 1);
                }
                
                if (!IsSupportedFormat(file_path)) {
                    load_res.success = false;
                    load_res.status = ImageLoadStatus::kUnsupportedFormat;
                    load_res.error_message = "The file extension is not support: " + file_path;
                    return load_res;
                }

                // 拿到文件的完整后缀
                format = extension;
            }

            // 主路径：二进制读取 + cv::imdecode（兼容性更好）
            // fallback：cv::imread（主路径失败时尝试）
            int read_mode = ToOpenCvFlag(options.read_mode);
            cv::Mat img_mat;

            // === 主路径：ifstream + imdecode ===
            {
                std::ifstream file(file_path, std::ios::binary | std::ios::ate);
                if (file.is_open()) {
                    std::streamsize size = file.tellg();
                    file.seekg(0, std::ios::beg);
                    std::vector<unsigned char> buffer(static_cast<size_t>(size));
                    file.read(reinterpret_cast<char*>(buffer.data()), size);

                    try {
                        img_mat = cv::imdecode(buffer, read_mode);
                    } catch (const cv::Exception& e) {
                        std::cerr << "imdecode error: " << e.what() << "\n";
                    }
                }
            }

            // === fallback：cv::imread ===
            if (img_mat.empty() && options.use_imdecode_fallback) {
                try {
                    img_mat = cv::imread(file_path, read_mode);
                } catch (const cv::Exception& e) {
                    std::cerr << "imread fallback failed: " << e.what() << "\n";
                }
            }

            // === 成功 ===
            if (!img_mat.empty()) {
                load_res.success = true;
                load_res.status = ImageLoadStatus::kOk;
                load_res.error_message = "";
                load_res.image = img_mat;
                load_res.info = ImageLoader::GetImageInfo(img_mat, file_path, format);
                return load_res;
            }

            // === 两种方式都失败 ===
            load_res.success = false;
            load_res.status = ImageLoadStatus::kDecodeFailed;
            load_res.error_message = "Failed to decode image: " + file_path;
            return load_res;
        }

        LoadResult LoadFromMemory(const std::vector<unsigned char> &buffer,
                                  const ImageLoadOptions &options) {
            
            LoadResult load_res;
            if (buffer.empty()) {
                load_res.success = false;
                load_res.status = ImageLoadStatus::kEmptyBuffer;
                load_res.error_message = "The buffer is empty in the memory!";
                return load_res;
            }

            int read_mode = ToOpenCvFlag(options.read_mode);
            cv::Mat img_mat;
            try {
                img_mat = cv::imdecode(buffer, read_mode);
            } catch (const cv::Exception& e) {
                std::cerr << "cv::imdecode error: " << e.what();
                load_res.success = false;
                load_res.status = ImageLoadStatus::kOpenCvError;
                load_res.error_message = "Decoding image is failed in the memory!";
                return load_res;
            }

            if (img_mat.empty()) {
                std::cout << "The img_mat is empty, openCv decode failed!\n";
                load_res.success = false;
                load_res.status = ImageLoadStatus::kDecodeFailed;
                load_res.error_message = "Image is empty in the memory!";
                return load_res;
            }

            // struct LoadResult {
            //     bool success = false;
            //     ImageLoadStatus status = ImageLoadStatus::kDecodeFailed;
            //     std::string error_message = "";
            //     cv::Mat image;
            //     ImageInfo info;
            // };
            load_res.success = true;
            load_res.status = ImageLoadStatus::kOk;
            std::string error_message = "";
            load_res.image = img_mat;
            // 内存加载没有来源路径 内存加载格式未知
            load_res.info = GetImageInfo(load_res.image, "", "");

            return load_res;
        }

        std::vector<LoadResult> LoadBatch(
            const std::vector<std::string> &file_paths,
            const ImageLoadOptions &options) {
            // TODO
            std::vector<LoadResult> results;
            results.reserve(file_paths.size());
            for (const auto& path : file_paths) {
                results.push_back(LoadFromFile(path, options));
            }

            return results;
        }
    };

    ImageLoader::ImageLoader() : pImpl_(std::make_unique<Impl>()) {}
    ImageLoader::~ImageLoader() = default;

    ImageLoader::ImageLoader(ImageLoader &&) noexcept = default;
    ImageLoader& ImageLoader::operator=(ImageLoader &&) noexcept = default;

    LoadResult ImageLoader::LoadFromFile(
        const std::string &file_path,
        const ImageLoadOptions &options) {
        return pImpl_->LoadFromFile(file_path, options);
    }

    LoadResult ImageLoader::LoadFromMemory(
        const std::vector<unsigned char> &buffer,
        const ImageLoadOptions &options) {
        return pImpl_->LoadFromMemory(buffer, options);
    }

    std::vector<LoadResult> ImageLoader::LoadBatch(
        const std::vector<std::string> &file_paths,
        const ImageLoadOptions &options) {
        return pImpl_->LoadBatch(file_paths, options);
    }

    bool ImageLoader::IsSupportedFormat(const std::string &file_path) {
        // TODO
        std::string extension = std::filesystem::path(file_path).extension().string();
        if (extension.empty()) {
            return false;
        }

        // 扩展名带 . 如 .PNG
        // 转换成小写 —— unsigned char c 是为了安全地传给 std::tolower，
        // 因为 char 可能为负数，tolower 识别int容易出现未定义行为
        std::transform(extension.begin(), extension.end(), extension.begin(), 
            [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        // 去掉 .
        if (!extension.empty() && extension[0] == '.') {
            extension.erase(0,1);
        }
        if (extension == "bmp" || extension == "jpg"
            || extension == "jpeg" || extension == "png") {
            return true;
        }

        return false;
    }
    std::vector<std::string> ImageLoader::GetSupportedExtensions() {
        return {"bmp", "jpeg", "jpg", "png"};
    }

    ImageInfo ImageLoader::GetImageInfo(
        const cv::Mat &image,
        const std::string &source_path,
        const std::string &format) {
        // struct ImageInfo {
        //     int width = 0;              // cv Mat 的宽度 col        
        //     int height = 0;             // cv Mat 的高度 row
        //     int channels = 0;           // 通道数 灰度0 彩色3  BGRA 4
        //     int type = 0;               // 类型编码 cv::Mat::type()
        //     int depth = 0;              // cv::Mat::depth() 单通道数据类型: CV_8U=0, CV_32F=5
        //     size_t elem_size = 0;       // cv::Mat::elemSize() 每个像素占多少字节 (BGR图 = 3)
        //     bool is_continuous = false; // 内存是否连续 (没有填充/ROI裁剪) 对应 `cv::Mat::isContinuous()`
        //     std::string format;         // 来自扩展名，例如 `jpg`
        //     std::string source_path;    // 来源路径
        // };
        ImageInfo im_info;
        if (image.empty()) {
            return im_info;
        }

        im_info.width = image.cols;
        im_info.height = image.rows;
        im_info.channels = image.channels();
        im_info.type = image.type();
        im_info.depth = image.depth();
        im_info.elem_size = image.elemSize();
        im_info.is_continuous = image.isContinuous();

        im_info.format = format;
        im_info.source_path = source_path;

        return im_info;
    }
    std::string ImageLoader::StatusToString(ImageLoadStatus status) {
        switch (status) {
            case ImageLoadStatus::kOk:                  return "Ok";
            case ImageLoadStatus::kEmptyPath:           return "EmptyPath";
            case ImageLoadStatus::kFileNotFound:        return "FileNotFound";
            case ImageLoadStatus::kNotRegularFile:      return "NotRegularFile";
            case ImageLoadStatus::kUnsupportedFormat:   return "UnsupportedFormat";
            case ImageLoadStatus::kFileOpenFailed:      return "FileOpenFailed";
            case ImageLoadStatus::kEmptyBuffer:         return "EmptyBuffer";
            case ImageLoadStatus::kDecodeFailed:        return "DecodeFailed";
            case ImageLoadStatus::kEmptyImage:          return "EmptyImage";
            case ImageLoadStatus::kOpenCvError:         return "OpenCvError";
            default: return "Unknown";
        }
    }

} // namespace core
} // namespace digital_human
