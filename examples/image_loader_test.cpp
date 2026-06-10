#include <iostream>
#include <fstream>
#include <vector>

#include "core/image_loader.h"

using namespace digital_human::core;

/// @brief 读取文件为二进制 buffer
static std::vector<unsigned char> ReadFileToBuffer(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

/// @brief 打印 LoadResult
static void PrintResult(const std::string& label, const LoadResult& r) {
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

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: ImageLoader Test ===\n\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    std::string test_path = argv[1];
    ImageLoader loader;

    // ---- 1. 从文件加载 ----
    std::cout << "--- [1] LoadFromFile ---\n";
    LoadResult r1 = loader.LoadFromFile(test_path);
    PrintResult("LoadFromFile", r1);

    // ---- 2. 从内存加载 ----
    std::cout << "--- [2] LoadFromMemory ---\n";
    std::vector<unsigned char> buffer = ReadFileToBuffer(test_path);
    if (buffer.empty()) {
        std::cout << "[LoadFromMemory] Cannot read file to buffer.\n\n";
    } else {
        std::cout << "  Read " << buffer.size() << " bytes from file.\n";
        LoadResult r2 = loader.LoadFromMemory(buffer);
        PrintResult("LoadFromMemory", r2);
    }

    // ---- 3. 批量加载 ----
    std::cout << "--- [3] LoadBatch ---\n";
    std::vector<std::string> batch_paths = {
        test_path,             // 正常
        "not_exist_12345.jpg", // 不存在
        test_path              // 正常
    };
    auto batch = loader.LoadBatch(batch_paths);
    for (size_t i = 0; i < batch.size(); ++i) {
        PrintResult("Batch[" + std::to_string(i) + "]", batch[i]);
    }

    return 0;
}
