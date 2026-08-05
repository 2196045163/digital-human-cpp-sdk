#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include <fstream>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/image_loader.h"

using namespace digital_human::core;

// 让 enum class 可以被 std::cerr 打印
inline std::ostream& operator<<(std::ostream& os, ImageLoadStatus s) {
    return os << ImageLoader::StatusToString(s);
}

// ==========================================================================
// 简易断言
// ==========================================================================
static int g_fail = 0;

#define EXPECT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << "\n"; \
            g_fail++; \
        } \
    } while(0)

#define EXPECT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "FAIL: " << msg \
                      << " (expected=" << (b) << ", actual=" << (a) << ")\n"; \
            g_fail++; \
        } \
    } while(0)

// ==========================================================================
int main() {
    std::cout << "=== ImageLoader Unit Tests ===\n";

    // ------------------------------------------------------------------
    // 测试1: StatusToString — 每个状态码对应正确字符串
    // ------------------------------------------------------------------
    std::cout << "[1] StatusToString ...\n";

    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kOk),
              std::string("Ok"), "StatusToString(kOk)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kEmptyPath),
              std::string("EmptyPath"), "StatusToString(kEmptyPath)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kFileNotFound),
              std::string("FileNotFound"), "StatusToString(kFileNotFound)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kNotRegularFile),
              std::string("NotRegularFile"), "StatusToString(kNotRegularFile)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kUnsupportedFormat),
              std::string("UnsupportedFormat"), "StatusToString(kUnsupportedFormat)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kFileOpenFailed),
              std::string("FileOpenFailed"), "StatusToString(kFileOpenFailed)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kEmptyBuffer),
              std::string("EmptyBuffer"), "StatusToString(kEmptyBuffer)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kDecodeFailed),
              std::string("DecodeFailed"), "StatusToString(kDecodeFailed)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kEmptyImage),
              std::string("EmptyImage"), "StatusToString(kEmptyImage)");
    EXPECT_EQ(ImageLoader::StatusToString(ImageLoadStatus::kOpenCvError),
              std::string("OpenCvError"), "StatusToString(kOpenCvError)");

    // ------------------------------------------------------------------
    // 测试2: GetSupportedExtensions — 包含全部4种格式
    // ------------------------------------------------------------------
    std::cout << "[2] GetSupportedExtensions ...\n";

    auto exts = ImageLoader::GetSupportedExtensions();
    EXPECT_EQ(exts.size(), size_t(4), "GetSupportedExtensions: size == 4");

    // 排序后比较，避免顺序依赖
    std::sort(exts.begin(), exts.end());
    EXPECT_EQ(exts[0], std::string("bmp"),  "GetSupportedExtensions: bmp");
    EXPECT_EQ(exts[1], std::string("jpeg"), "GetSupportedExtensions: jpeg");
    EXPECT_EQ(exts[2], std::string("jpg"),  "GetSupportedExtensions: jpg");
    EXPECT_EQ(exts[3], std::string("png"),  "GetSupportedExtensions: png");

    // ------------------------------------------------------------------
    // 测试3: IsSupportedFormat — 正确格式返回true，其他返回false
    // ------------------------------------------------------------------
    std::cout << "[3] IsSupportedFormat ...\n";

    EXPECT_TRUE(ImageLoader::IsSupportedFormat("photo.jpg"),       "jpg → true");
    EXPECT_TRUE(ImageLoader::IsSupportedFormat("photo.jpeg"),      "jpeg → true");
    EXPECT_TRUE(ImageLoader::IsSupportedFormat("photo.png"),       "png → true");
    EXPECT_TRUE(ImageLoader::IsSupportedFormat("photo.bmp"),       "bmp → true");
    EXPECT_TRUE(ImageLoader::IsSupportedFormat("photo.JPG"),       "JPG → true");
    EXPECT_TRUE(ImageLoader::IsSupportedFormat("photo.PNG"),       "PNG → true");
    EXPECT_TRUE(ImageLoader::IsSupportedFormat("a.b.c.png"),       "多点的 png → true");

    EXPECT_TRUE(!ImageLoader::IsSupportedFormat("photo.txt"),      "txt → false");
    EXPECT_TRUE(!ImageLoader::IsSupportedFormat("photo"),           "无扩展名 → false");
    EXPECT_TRUE(!ImageLoader::IsSupportedFormat(""),                "空字符串 → false");

    // ------------------------------------------------------------------
    // 测试4: GetImageInfo — 从 cv::Mat 提取信息
    // ------------------------------------------------------------------
    std::cout << "[4] GetImageInfo ...\n";

    // 空图 → 全零
    cv::Mat empty_mat;
    ImageInfo empty_info = ImageLoader::GetImageInfo(empty_mat);
    EXPECT_EQ(empty_info.width,  0, "empty: width == 0");
    EXPECT_EQ(empty_info.height, 0, "empty: height == 0");

    // 20x30 BGR 彩色图
    cv::Mat img(20, 30, CV_8UC3);
    ImageInfo info = ImageLoader::GetImageInfo(img, "/root/face.jpg", "jpg");

    EXPECT_EQ(info.width,         30,   "20x30: width == 30");
    EXPECT_EQ(info.height,        20,   "20x30: height == 20");
    EXPECT_EQ(info.channels,      3,    "20x30: channels == 3");
    EXPECT_EQ(info.depth,         CV_8U,"20x30: depth == CV_8U");
    EXPECT_EQ(info.elem_size,     size_t(3), "20x30: elemSize == 3");
    EXPECT_TRUE(info.is_continuous,      "20x30: isContinuous == true");
    EXPECT_EQ(info.source_path,   std::string("/root/face.jpg"), "source_path");
    EXPECT_EQ(info.format,        std::string("jpg"),            "format");

    // ------------------------------------------------------------------
    // 测试5: LoadFromMemory
    // ------------------------------------------------------------------
    std::cout << "[5] LoadFromMemory ...\n";

    ImageLoader loader;

    // 5a. 生成一张测试图片 → 编码为 jpg bytes
    cv::Mat test_img(40, 50, CV_8UC3, cv::Scalar(255, 0, 0)); // 纯蓝 50x40
    std::vector<unsigned char> jpg_bytes;
    cv::imencode(".jpg", test_img, jpg_bytes);
    EXPECT_TRUE(!jpg_bytes.empty(), "5a: imencode 成功");

    // 5b. 正常内存加载
    LoadResult ok = loader.LoadFromMemory(jpg_bytes);
    EXPECT_TRUE(ok.success,                        "5b: success == true");
    EXPECT_EQ(ok.status, ImageLoadStatus::kOk,     "5b: status == kOk");
    EXPECT_TRUE(!ok.image.empty(),                 "5b: image 非空");
    EXPECT_EQ(ok.info.width,  50,                  "5b: width == 50");
    EXPECT_EQ(ok.info.height, 40,                  "5b: height == 40");
    EXPECT_EQ(ok.info.channels, 3,                 "5b: channels == 3");
    EXPECT_TRUE(ok.error_message.empty(),           "5b: error_message 为空");

    // 5c. 空 buffer → EmptyBuffer
    LoadResult eb = loader.LoadFromMemory({});
    EXPECT_TRUE(!eb.success,                             "5c: success == false");
    EXPECT_EQ(eb.status, ImageLoadStatus::kEmptyBuffer,  "5c: status == kEmptyBuffer");
    EXPECT_TRUE(!eb.error_message.empty(),               "5c: error_message 非空");
    EXPECT_TRUE(eb.image.empty(),                         "5c: image 为空");

    // 5d. 随机 bytes → 解码失败
    std::vector<unsigned char> random_bytes = {
        0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0xFC,
        0xAA, 0xBB, 0xCC, 0xDD
    };
    LoadResult rb = loader.LoadFromMemory(random_bytes);
    EXPECT_TRUE(!rb.success,                         "5d: success == false");
    EXPECT_TRUE(rb.status == ImageLoadStatus::kDecodeFailed
               || rb.status == ImageLoadStatus::kEmptyImage
               || rb.status == ImageLoadStatus::kOpenCvError,
               "5d: 解码失败状态合理");
    EXPECT_TRUE(!rb.error_message.empty(),           "5d: error_message 非空");
    EXPECT_TRUE(rb.image.empty(),                     "5d: image 为空");

    // 5e. 灰度模式加载
    ImageLoadOptions gray_opt;
    gray_opt.read_mode = ImageReadMode::kGrayscale;
    LoadResult gray = loader.LoadFromMemory(jpg_bytes, gray_opt);
    EXPECT_TRUE(gray.success,                   "5e: 灰度加载 success == true");
    EXPECT_EQ(gray.info.channels, 1,            "5e: channels == 1");

    // ------------------------------------------------------------------
    // 测试6: LoadFromFile
    // ------------------------------------------------------------------
    std::cout << "[6] LoadFromFile ...\n";

    // 准备工作：在 build 目录生成测试文件
    const std::string tmp_jpg  = "test_loadfromfile_tmp.jpg";
    const std::string tmp_png  = "test_loadfromfile_tmp.png";
    const std::string tmp_bmp  = "test_loadfromfile_tmp.bmp";
    const std::string tmp_broken = "test_loadfromfile_broken.jpg";
    const std::string tmp_txt  = "test_loadfromfile_tmp.txt";
    const std::string tmp_dir  = "test_loadfromfile_tmp_dir";

    // 生成正常的 jpg / png / bmp
    cv::imwrite(tmp_jpg, test_img);
    cv::imwrite(tmp_png, test_img);
    cv::imwrite(tmp_bmp, test_img);

    // vmhgfs 可能有延迟，确认文件写入成功再继续
    EXPECT_TRUE(std::filesystem::exists(tmp_jpg), "tmp_jpg created");
    EXPECT_TRUE(std::filesystem::file_size(tmp_jpg) > 0, "tmp_jpg non-empty");
    EXPECT_TRUE(std::filesystem::file_size(tmp_png) > 0, "tmp_png non-empty");
    EXPECT_TRUE(std::filesystem::file_size(tmp_bmp) > 0, "tmp_bmp non-empty");

    // 生成损坏的 "jpg"（其实是文本）
    {
        std::ofstream f(tmp_broken);
        f << "this is not a real jpeg file";
    }

    // 生成 .txt 文件
    {
        std::ofstream f(tmp_txt);
        f << "hello";
    }

    // 生成一个目录
    std::filesystem::create_directory(tmp_dir);

    // 6a. 正常加载 jpg
    LoadResult lf_jpg = loader.LoadFromFile(tmp_jpg);
    EXPECT_TRUE(lf_jpg.success,                     "6a: jpg success == true");
    EXPECT_EQ(lf_jpg.status, ImageLoadStatus::kOk,  "6a: jpg status == kOk");
    EXPECT_TRUE(!lf_jpg.image.empty(),               "6a: jpg image 非空");
    EXPECT_EQ(lf_jpg.info.width,  50,                "6a: jpg width == 50");
    EXPECT_EQ(lf_jpg.info.height, 40,                "6a: jpg height == 40");
    EXPECT_EQ(lf_jpg.info.format, std::string("jpg"),"6a: jpg format == jpg");
    EXPECT_TRUE(lf_jpg.info.source_path.find(tmp_jpg) != std::string::npos,
               "6a: source_path 包含文件名");
    EXPECT_TRUE(lf_jpg.error_message.empty(),         "6a: error_message 为空");

    // 6b. 正常加载 png
    LoadResult lf_png = loader.LoadFromFile(tmp_png);
    EXPECT_TRUE(lf_png.success,                      "6b: png success == true");
    EXPECT_EQ(lf_png.info.format, std::string("png"), "6b: png format == png");

    // 6c. 正常加载 bmp
    LoadResult lf_bmp = loader.LoadFromFile(tmp_bmp);
    EXPECT_TRUE(lf_bmp.success,                      "6c: bmp success == true");
    EXPECT_EQ(lf_bmp.info.format, std::string("bmp"), "6c: bmp format == bmp");

    // 6d. 空路径 → kEmptyPath
    LoadResult lf_empty = loader.LoadFromFile("");
    EXPECT_TRUE(!lf_empty.success,                       "6d: success == false");
    EXPECT_EQ(lf_empty.status, ImageLoadStatus::kEmptyPath, "6d: status == kEmptyPath");
    EXPECT_TRUE(!lf_empty.error_message.empty(),         "6d: error_message 非空");
    EXPECT_TRUE(lf_empty.image.empty(),                   "6d: image 为空");

    // 6e. 不存在路径 → kFileNotFound
    LoadResult lf_no = loader.LoadFromFile("not_exist_file_12345.jpg");
    EXPECT_TRUE(!lf_no.success,                               "6e: success == false");
    EXPECT_EQ(lf_no.status, ImageLoadStatus::kFileNotFound,   "6e: status == kFileNotFound");
    EXPECT_TRUE(!lf_no.error_message.empty(),                 "6e: error_message 非空");

    // 6f. 目录路径 → kNotRegularFile
    LoadResult lf_dir = loader.LoadFromFile(tmp_dir);
    EXPECT_TRUE(!lf_dir.success,                                "6f: success == false");
    EXPECT_EQ(lf_dir.status, ImageLoadStatus::kNotRegularFile,  "6f: status == kNotRegularFile");
    EXPECT_TRUE(!lf_dir.error_message.empty(),                   "6f: error_message 非空");

    // 6g. .txt 文件 → kUnsupportedFormat
    LoadResult lf_txt = loader.LoadFromFile(tmp_txt);
    EXPECT_TRUE(!lf_txt.success,                                    "6g: success == false");
    EXPECT_EQ(lf_txt.status, ImageLoadStatus::kUnsupportedFormat,   "6g: status == kUnsupportedFormat");
    EXPECT_TRUE(!lf_txt.error_message.empty(),                       "6g: error_message 非空");

    // 6h. 损坏的 jpg → 解码失败
    LoadResult lf_broken = loader.LoadFromFile(tmp_broken);
    EXPECT_TRUE(!lf_broken.success,                          "6h: success == false");
    EXPECT_TRUE(lf_broken.status == ImageLoadStatus::kDecodeFailed
               || lf_broken.status == ImageLoadStatus::kEmptyImage
               || lf_broken.status == ImageLoadStatus::kOpenCvError,
               "6h: 解码失败状态合理");
    EXPECT_TRUE(!lf_broken.error_message.empty(),             "6h: error_message 非空");
    EXPECT_TRUE(lf_broken.image.empty(),                      "6h: image 为空");

    // 6i. 关闭扩展名检查后加载 .txt — fallback 走 imdecode，解码失败
    ImageLoadOptions no_check;
    no_check.check_extension = false;
    LoadResult lf_txt2 = loader.LoadFromFile(tmp_txt, no_check);
    EXPECT_TRUE(!lf_txt2.success,                              "6i: .txt 不开扩展检查 → decode 失败");

    // ------------------------------------------------------------------
    // 测试7: LoadBatch
    // ------------------------------------------------------------------
    std::cout << "[7] LoadBatch ...\n";

    // 构造混合列表：成功 + 不存在 + 空路径 + 成功
    std::vector<std::string> batch_paths = {
        tmp_jpg,                    // [0] 成功
        "not_exist_file_99999.jpg", // [1] 失败
        "",                         // [2] 失败
        tmp_png                     // [3] 成功
    };

    std::vector<LoadResult> batch_results = loader.LoadBatch(batch_paths);

    // 7a. 数量必须等于输入
    EXPECT_EQ(batch_results.size(), size_t(4), "7a: results.size() == 4");

    // 7b. [0] 成功加载 jpg
    EXPECT_TRUE(batch_results[0].success,                    "7b: [0] success");
    EXPECT_EQ(batch_results[0].status, ImageLoadStatus::kOk, "7b: [0] status == kOk");
    EXPECT_EQ(batch_results[0].info.width, 50,               "7b: [0] width == 50");
    EXPECT_EQ(batch_results[0].info.height, 40,              "7b: [0] height == 40");

    // 7c. [1] 文件不存在
    EXPECT_TRUE(!batch_results[1].success,                        "7c: [1] success == false");
    EXPECT_EQ(batch_results[1].status, ImageLoadStatus::kFileNotFound, "7c: [1] status == kFileNotFound");
    EXPECT_TRUE(!batch_results[1].error_message.empty(),          "7c: [1] error_message 非空");
    EXPECT_TRUE(batch_results[1].image.empty(),                    "7c: [1] image 为空");

    // 7d. [2] 空路径
    EXPECT_TRUE(!batch_results[2].success,                        "7d: [2] success == false");
    EXPECT_EQ(batch_results[2].status, ImageLoadStatus::kEmptyPath, "7d: [2] status == kEmptyPath");
    EXPECT_TRUE(!batch_results[2].error_message.empty(),          "7d: [2] error_message 非空");

    // 7e. [3] 成功加载 png
    EXPECT_TRUE(batch_results[3].success,                    "7e: [3] success");
    EXPECT_EQ(batch_results[3].status, ImageLoadStatus::kOk, "7e: [3] status == kOk");
    EXPECT_EQ(batch_results[3].info.format, std::string("png"), "7e: [3] format == png");

    // 清理临时文件
    std::filesystem::remove(tmp_jpg);
    std::filesystem::remove(tmp_png);
    std::filesystem::remove(tmp_bmp);
    std::filesystem::remove(tmp_broken);
    std::filesystem::remove(tmp_txt);
    std::filesystem::remove(tmp_dir);

    // ------------------------------------------------------------------
    // 测试8: 带空格路径
    // ------------------------------------------------------------------
    std::cout << "[8] Space path ...\n";

    const std::string space_dir  = "test space dir";
    const std::string space_file = space_dir + "/space image.jpg";

    std::filesystem::create_directory(space_dir);
    cv::imwrite(space_file, test_img);
    EXPECT_TRUE(std::filesystem::exists(space_file), "8a: space file created");

    LoadResult lf_space = loader.LoadFromFile(space_file);
    EXPECT_TRUE(lf_space.success,                    "8b: space path success");
    EXPECT_EQ(lf_space.info.width,  50,              "8c: space path width == 50");
    EXPECT_EQ(lf_space.info.height, 40,              "8d: space path height == 40");

    std::filesystem::remove(space_file);
    std::filesystem::remove(space_dir);

    // ------------------------------------------------------------------
    // 测试9: 中文路径
    // ------------------------------------------------------------------
    std::cout << "[9] Chinese path ...\n";

    const std::string cn_dir  = "测试中文目录";
    const std::string cn_file = cn_dir + "/测试图片.jpg";

    std::filesystem::create_directory(cn_dir);
    cv::imwrite(cn_file, test_img);
    EXPECT_TRUE(std::filesystem::exists(cn_file), "9a: chinese file created");

    LoadResult lf_cn = loader.LoadFromFile(cn_file);
    EXPECT_TRUE(lf_cn.success,                    "9b: chinese path success");
    EXPECT_EQ(lf_cn.info.width,  50,              "9c: chinese path width == 50");
    EXPECT_EQ(lf_cn.info.height, 40,              "9d: chinese path height == 40");

    std::filesystem::remove(cn_file);
    std::filesystem::remove(cn_dir);

    // ------------------------------------------------------------------
    // 结果
    // ------------------------------------------------------------------
    std::cout << "----------------------------------------\n";
    if (g_fail == 0) {
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << g_fail << " test(s) failed.\n";
    return EXIT_FAILURE;
}
