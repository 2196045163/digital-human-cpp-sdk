/// @file cli_failure_injection_test.cpp
/// @brief M03 CLI 失败注入测试：Fake MediaWriter 注入 RunCli，验证四条件成功校验。
///
/// 背景：CLI 成功判定是"四条件联合"（apps/digital_human_app.cpp RunCli
/// 第 11-12 步）：
///   1) pipeline.Wait() 返回 success
///   2) writer->IsFinalized() 为 true
///   3) writer->GetLastError() == kOk
///   4) 输出文件存在且非空
///
/// 真实 FinalMediaWriter 的编码/mux/flush/trailer 失败必须改 FFmpeg 才能
/// 注入，因此本套件通过 MediaWriter 抽象接口注入 FakeFinalMediaWriter：
/// 在真实 Pipeline（Golden 素材）跑完并到达 Succeeded 终态的前提下，逐项
/// 破坏四条件之一，验证 CLI 仍然：
/// - 返回非零退出码
/// - stdout 输出恰好一个可解析的 JSON 对象
/// - JSON status != "success"
/// - error.message 定位到具体失败阶段（编码/mux/flush/trailer/未 Finalize/
///   文件缺失/空文件）
/// - 不残留 MP4 伪产物（或文件被删除）

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "digital_human_cli.h"
#include "output/final_media_writer.h"
#include "pipeline/pipeline_types.h"

namespace fs = std::filesystem;
using namespace digital_human;

// ============================================================================
// 极简 JSON 解析器（仅测试用：验证"恰好一个可解析 JSON 对象"并读取字段）
// ============================================================================

namespace mini_json {

struct Value;
using Object = std::map<std::string, std::shared_ptr<Value>>;
using Array = std::vector<std::shared_ptr<Value>>;

struct Value {
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
    Type type = Type::kNull;
    bool b = false;
    double num = 0.0;
    std::string str;
    Array arr;
    Object obj;

    /// @brief 直接子键查找（不递归）
    const Value* Find(const std::string& key) const {
        if (type != Type::kObject) return nullptr;
        auto it = obj.find(key);
        return (it == obj.end()) ? nullptr : it->second.get();
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    /// @brief 解析整个文本；要求恰好一个顶层 JSON 值且其后仅有空白。
    /// @return 顶层值；解析失败（含尾随内容）返回 nullptr
    std::shared_ptr<Value> ParseDocument() {
        SkipWhitespace();
        if (pos_ >= s_.size()) return nullptr;
        auto v = ParseValue();
        if (!v) return nullptr;
        SkipWhitespace();
        if (pos_ != s_.size()) return nullptr;  // 尾随内容 → 不是"恰好一个对象"
        return v;
    }

private:
    std::shared_ptr<Value> ParseValue() {
        SkipWhitespace();
        if (pos_ >= s_.size()) return nullptr;
        char c = s_[pos_];
        if (c == '{') return ParseObject();
        if (c == '[') return ParseArray();
        if (c == '"') return ParseStringValue();
        if (c == 't' || c == 'f') return ParseBool();
        if (c == 'n') return ParseNull();
        if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();
        return nullptr;
    }

    std::shared_ptr<Value> ParseObject() {
        auto v = std::make_shared<Value>();
        v->type = Value::Type::kObject;
        ++pos_;  // '{'
        SkipWhitespace();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return v; }
        for (;;) {
            SkipWhitespace();
            if (pos_ >= s_.size() || s_[pos_] != '"') return nullptr;
            std::string key;
            if (!ParseString(&key)) return nullptr;
            SkipWhitespace();
            if (pos_ >= s_.size() || s_[pos_] != ':') return nullptr;
            ++pos_;
            auto val = ParseValue();
            if (!val) return nullptr;
            v->obj[key] = val;
            SkipWhitespace();
            if (pos_ >= s_.size()) return nullptr;
            char c = s_[pos_];
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return v; }
            return nullptr;
        }
    }

    std::shared_ptr<Value> ParseArray() {
        auto v = std::make_shared<Value>();
        v->type = Value::Type::kArray;
        ++pos_;  // '['
        SkipWhitespace();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return v; }
        for (;;) {
            auto val = ParseValue();
            if (!val) return nullptr;
            v->arr.push_back(val);
            SkipWhitespace();
            if (pos_ >= s_.size()) return nullptr;
            char c = s_[pos_];
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return v; }
            return nullptr;
        }
    }

    bool ParseString(std::string* out) {
        if (pos_ >= s_.size() || s_[pos_] != '"') return false;
        ++pos_;
        std::string result;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') { *out = result; return true; }
            if (c == '\\') {
                if (pos_ >= s_.size()) return false;
                char e = s_[pos_++];
                switch (e) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/';  break;
                    case 'b':  result += '\b'; break;
                    case 'f':  result += '\f'; break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    case 'u': {
                        // 仅处理 BMP 字符（CLI 输出为 ASCII 转义，足够测试用）
                        if (pos_ + 4 > s_.size()) return false;
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = s_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9')       code |= h - '0';
                            else if (h >= 'a' && h <= 'f')  code |= h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F')  code |= h - 'A' + 10;
                            else return false;
                        }
                        if (code < 0x80) {
                            result += static_cast<char>(code);
                        } else if (code < 0x800) {
                            result += static_cast<char>(0xC0 | (code >> 6));
                            result += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            result += static_cast<char>(0xE0 | (code >> 12));
                            result += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: return false;
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                return false;  // 未转义控制字符非法
            } else {
                result += c;
            }
        }
        return false;
    }

    std::shared_ptr<Value> ParseStringValue() {
        auto v = std::make_shared<Value>();
        v->type = Value::Type::kString;
        if (!ParseString(&v->str)) return nullptr;
        return v;
    }

    std::shared_ptr<Value> ParseBool() {
        if (s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            auto v = std::make_shared<Value>();
            v->type = Value::Type::kBool;
            v->b = true;
            return v;
        }
        if (s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            auto v = std::make_shared<Value>();
            v->type = Value::Type::kBool;
            v->b = false;
            return v;
        }
        return nullptr;
    }

    std::shared_ptr<Value> ParseNull() {
        if (s_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return std::make_shared<Value>();
        }
        return nullptr;
    }

    std::shared_ptr<Value> ParseNumber() {
        size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
        bool has_digits = false;
        while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') { ++pos_; has_digits = true; }
        if (!has_digits) return nullptr;
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        auto v = std::make_shared<Value>();
        v->type = Value::Type::kNumber;
        v->num = std::strtod(s_.substr(start, pos_ - start).c_str(), nullptr);
        return v;
    }

    void SkipWhitespace() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                                    s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    const std::string& s_;
    size_t pos_ = 0;
};

}  // namespace mini_json

// ============================================================================
// 测试辅助：stdout 捕获 / JSON 字段读取 / 唯一路径
// ============================================================================

/// @brief 将进程 stdout 重定向到临时文件，析构时恢复（本套件每个用例独立进程）
class StdoutCapture {
public:
    explicit StdoutCapture(const std::string& file_path) : path_(file_path) {
        std::fflush(stdout);
        saved_fd_ = ::dup(STDOUT_FILENO);
        out_file_ = std::fopen(path_.c_str(), "w");
        if (out_file_) {
            ::dup2(::fileno(out_file_), STDOUT_FILENO);
        }
    }

    ~StdoutCapture() {
        std::fflush(stdout);
        if (out_file_) {
            ::dup2(saved_fd_, STDOUT_FILENO);
            std::fclose(out_file_);
        }
        ::close(saved_fd_);
    }

    std::string Str() const {
        std::ifstream ifs(path_);
        std::ostringstream oss;
        oss << ifs.rdbuf();
        return oss.str();
    }

private:
    std::string path_;
    int saved_fd_ = -1;
    std::FILE* out_file_ = nullptr;
};

/// @brief 读取 JSON 的嵌套字符串字段，路径形如 "error.message"
static std::string JsonString(const mini_json::Value* root, const std::string& path) {
    const mini_json::Value* v = root;
    size_t start = 0;
    while (v != nullptr) {
        size_t dot = path.find('.', start);
        std::string key = path.substr(
            start, dot == std::string::npos ? std::string::npos : dot - start);
        v = v->Find(key);
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return (v != nullptr && v->type == mini_json::Value::Type::kString) ? v->str : "";
}

static bool JsonBool(const mini_json::Value* root, const std::string& path,
                     bool def = false) {
    const mini_json::Value* v = root;
    size_t start = 0;
    while (v != nullptr) {
        size_t dot = path.find('.', start);
        std::string key = path.substr(
            start, dot == std::string::npos ? std::string::npos : dot - start);
        v = v->Find(key);
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return (v != nullptr && v->type == mini_json::Value::Type::kBool) ? v->b : def;
}

static double JsonNumber(const mini_json::Value* root, const std::string& path,
                         double def = 0.0) {
    const mini_json::Value* v = root;
    size_t start = 0;
    while (v != nullptr) {
        size_t dot = path.find('.', start);
        std::string key = path.substr(
            start, dot == std::string::npos ? std::string::npos : dot - start);
        v = v->Find(key);
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return (v != nullptr && v->type == mini_json::Value::Type::kNumber) ? v->num : def;
}

/// @brief 校验 stdout 恰好一个可解析 JSON 对象，返回顶层对象（失败返回 nullptr）
static std::shared_ptr<mini_json::Value> ParseExactlyOneJson(const std::string& text) {
    mini_json::Parser parser(text);
    auto root = parser.ParseDocument();
    EXPECT_NE(root, nullptr) << "stdout 不是恰好一个可解析 JSON 对象";
    if (root) {
        EXPECT_EQ(root->type, mini_json::Value::Type::kObject);
    }
    return root;
}

/// @brief 每个用例唯一的临时输出路径
static std::string UniqueOutputPath(const std::string& tag) {
    static std::atomic<int> counter{0};
    return "/tmp/cli_failure_inject_" + tag + "_" +
           std::to_string(static_cast<long>(::getpid())) + "_" +
           std::to_string(counter.fetch_add(1)) + ".mp4";
}

/// @brief 析构时删除文件（幂等）
class ScopedRemove {
public:
    explicit ScopedRemove(const fs::path& p) : path_(p) {}
    ~ScopedRemove() {
        std::error_code ec;
        fs::remove(path_, ec);
    }

private:
    fs::path path_;
};

/// @brief Golden 素材参数（与 cli.contract / e2e 套件一致）
static std::vector<std::string> GoldenArgs(const std::string& output_path) {
    return {
        "--image", "testdata/golden/face.jpg",
        "--audio", "testdata/golden/audio.wav",
        "--model-param", "models/wav2lip/wav2lip.param",
        "--model-bin", "models/wav2lip/wav2lip.bin",
        "--landmark", "models/shape_predictor_68_face_landmarks.dat",
        "--output", output_path,
        "--fps", "25",
    };
}

// ============================================================================
// FakeFinalMediaWriter — 可配置失败阶段的测试替身
// ============================================================================

/// @brief Fake writer 的文件行为配置
struct FakeFileBehavior {
    bool write_partial_file = false;  ///< 首个 OnFrame 时写几字节半成品
    bool write_empty_file = false;    ///< 构造时写 0 字节文件
    bool remove_on_failure = true;    ///< 失败时移除输出文件（模拟真实 writer 清理）
};

/// @brief 实现 output::MediaWriter 接口的 Fake writer。
///
/// 在真实 Pipeline 跑完（OnTerminal 收到 Succeeded）后按配置注入故障：
/// - FailStage::kVideoEncode / kAudioMux / kFlush / kTrailer：
///   设置对应 WriterError + 错误消息，IsFinalized() 返回 false，
///   并按 FileBehavior 清理输出文件（模拟真实 writer 失败时不留半成品）。
/// - FailStage::kNotFinalized：无错误码但 IsFinalized() == false。
/// - FailStage::kNone：模拟成功 writer（错误 kOk 且已 Finalize），
///   是否落盘由 FileBehavior 控制（不落盘 → 文件缺失；0 字节 → 空文件）。
class FakeFinalMediaWriter : public output::MediaWriter {
public:
    /// @brief 注入的失败阶段
    enum class FailStage {
        kNone,          ///< 模拟成功路径（不注入失败）
        kVideoEncode,   ///< 视频编码失败
        kAudioMux,      ///< 音频 mux（interleaved write）失败
        kFlush,         ///< 编码器 flush 失败
        kTrailer,       ///< trailer 写失败
        kNotFinalized,  ///< 不 Finalize（无错误码）
    };

    FakeFinalMediaWriter(const output::WriterConfig& config, FailStage stage,
                         FakeFileBehavior file_behavior = FakeFileBehavior{})
        : output_path_(config.output_path),
          stage_(stage),
          file_behavior_(file_behavior) {
        if (file_behavior_.write_empty_file) {
            std::ofstream ofs(output_path_, std::ios::binary | std::ios::trunc);
            ofs.close();
        }
    }

    // ---- PipelineOutputSink 实现 ----
    void OnFrame(const pipeline::PipelineFrame& /*frame*/) override {
        if (!partial_written_ && file_behavior_.write_partial_file) {
            std::ofstream ofs(output_path_, std::ios::binary | std::ios::trunc);
            ofs << "partial-mp4-bytes";
            ofs.close();
            partial_written_ = true;
        }
    }

    void OnTerminal(const pipeline::PipelineResult& result) override {
        terminal_called_ = true;
        terminal_success_ = result.success;
        ApplyFailure();
    }

    // ---- MediaWriter 查询接口 ----
    bool IsFinalized() const override { return finalized_; }
    output::WriterError GetLastError() const override { return last_error_; }
    std::string GetLastErrorMessage() const override { return last_error_message_; }
    int64_t GetWrittenFrameCount() const override { return frame_count_; }
    const std::string& GetOutputPath() const override { return output_path_; }

    // ---- 测试断言辅助 ----
    bool terminal_called() const { return terminal_called_; }
    bool terminal_success() const { return terminal_success_; }

private:
    void ApplyFailure() {
        frame_count_ = 1;  // 模拟至少写入了一帧/一个包
        switch (stage_) {
            case FailStage::kVideoEncode:
                last_error_ = output::WriterError::kVideoEncodeFailed;
                last_error_message_ = "Video encode failed at stage: " +
                                      output::WriterErrorToString(last_error_);
                finalized_ = false;
                break;
            case FailStage::kAudioMux:
                last_error_ = output::WriterError::kInterleavedWriteFailed;
                last_error_message_ = "Audio mux failed at stage: " +
                                      output::WriterErrorToString(last_error_);
                finalized_ = false;
                break;
            case FailStage::kFlush:
                last_error_ = output::WriterError::kFlushFailed;
                last_error_message_ = "Flush failed at stage: " +
                                      output::WriterErrorToString(last_error_);
                finalized_ = false;
                break;
            case FailStage::kTrailer:
                last_error_ = output::WriterError::kTrailerWriteFailed;
                last_error_message_ = "Trailer write failed at stage: " +
                                      output::WriterErrorToString(last_error_);
                finalized_ = false;
                break;
            case FailStage::kNotFinalized:
                last_error_ = output::WriterError::kOk;
                last_error_message_ = "";
                finalized_ = false;
                break;
            case FailStage::kNone:
                last_error_ = output::WriterError::kOk;
                last_error_message_ = "";
                finalized_ = true;
                break;
        }
        if (last_error_ != output::WriterError::kOk && file_behavior_.remove_on_failure) {
            std::error_code ec;
            fs::remove(output_path_, ec);
        }
    }

    std::string output_path_;
    FailStage stage_;
    FakeFileBehavior file_behavior_;
    bool partial_written_ = false;
    bool finalized_ = false;
    bool terminal_called_ = false;
    bool terminal_success_ = false;
    int64_t frame_count_ = 0;
    output::WriterError last_error_ = output::WriterError::kOk;
    std::string last_error_message_;
};

// ============================================================================
// RunCli 注入辅助
// ============================================================================

struct FakeRunResult {
    int exit_code = -1;
    std::string stdout_text;
    std::shared_ptr<FakeFinalMediaWriter> fake;
};

/// @brief 以 Fake writer 工厂调用 RunCli，捕获 stdout 与退出码
static FakeRunResult RunCliWithFake(
    const std::vector<std::string>& args,
    FakeFinalMediaWriter::FailStage stage,
    FakeFileBehavior file_behavior = FakeFileBehavior{}) {
    std::vector<std::string> argv_strs = {"digital_human_app"};
    argv_strs.insert(argv_strs.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : argv_strs) argv.push_back(s.data());
    argv.push_back(nullptr);

    std::shared_ptr<FakeFinalMediaWriter> fake;
    cli::MediaWriterFactory factory =
        [&fake, stage, file_behavior](const output::WriterConfig& cfg) {
            fake = std::make_shared<FakeFinalMediaWriter>(cfg, stage, file_behavior);
            return std::static_pointer_cast<output::MediaWriter>(fake);
        };

    static std::atomic<int> capture_counter{0};
    const std::string stdout_path =
        "/tmp/cli_failure_inject_stdout_" +
        std::to_string(static_cast<long>(::getpid())) + "_" +
        std::to_string(capture_counter.fetch_add(1)) + ".log";

    FakeRunResult r;
    {
        StdoutCapture capture(stdout_path);
        r.exit_code = cli::RunCli(static_cast<int>(argv.size() - 1), argv.data(),
                                  factory);
        r.stdout_text = capture.Str();
    }
    r.fake = fake;
    std::remove(stdout_path.c_str());
    return r;
}

// ============================================================================
// 失败注入用例：四条件成功校验的逐项破坏
// ============================================================================

// 公共断言：退出码非 0 + 恰好一个可解析 JSON + status=error + writer 阶段定位
static void ExpectStageFailure(const FakeRunResult& r, const std::string& stage_token) {
    // 1) 非零退出码
    EXPECT_NE(r.exit_code, 0) << "CLI 应因 writer 失败返回非零退出码";
    // 2) stdout 恰好一个可解析 JSON 对象
    auto root = ParseExactlyOneJson(r.stdout_text);
    ASSERT_NE(root, nullptr);
    // 3) JSON status != "success"
    EXPECT_NE(root->Find("status"), nullptr);
    EXPECT_NE(JsonString(root.get(), "status"), "success");
    EXPECT_EQ(JsonString(root.get(), "status"), "error");
    // 4) error 字段定位失败阶段
    EXPECT_EQ(JsonString(root.get(), "error.code"), "WriterOutputInvalid");
    EXPECT_NE(JsonString(root.get(), "error.message").find("FinalMediaWriter failed"),
              std::string::npos)
        << "error.message 应以 'FinalMediaWriter failed:' 开头";
    EXPECT_NE(JsonString(root.get(), "error.message").find(stage_token),
              std::string::npos)
        << "error.message 未定位到阶段: " << stage_token
        << " message=" << JsonString(root.get(), "error.message");
    EXPECT_NE(JsonString(root.get(), "error.writer_error").find(stage_token),
              std::string::npos)
        << "error.writer_error 未包含阶段错误码: " << stage_token;
    EXPECT_FALSE(JsonBool(root.get(), "error.writer_finalized"))
        << "失败时 writer_finalized 应为 false";
    // 5) 管线确实成功到达终态（否则走的是另一条失败分支）
    ASSERT_NE(r.fake, nullptr);
    EXPECT_TRUE(r.fake->terminal_called());
    EXPECT_TRUE(r.fake->terminal_success())
        << "前置条件失败：Pipeline 未成功到达终态，本用例测的不是 writer 分支";
}

// ----------------------------------------------------------------------------
// 1. 视频编码失败：writer 报 kVideoEncodeFailed
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, VideoEncodeFailure) {
    const std::string output = UniqueOutputPath("video_encode_failure");
    ScopedRemove cleanup(output);

    FakeFileBehavior file_behavior;
    file_behavior.write_partial_file = true;  // 半成品文件，失败时清理

    auto r = RunCliWithFake(GoldenArgs(output),
                            FakeFinalMediaWriter::FailStage::kVideoEncode,
                            file_behavior);

    ExpectStageFailure(r, "VideoEncodeFailed");

    // 无 MP4 伪产物残留（半成品已被清理）
    EXPECT_FALSE(fs::exists(output)) << "编码失败后不应残留输出文件";
    auto root = ParseExactlyOneJson(r.stdout_text);
    ASSERT_NE(root, nullptr);
    EXPECT_FALSE(JsonBool(root.get(), "error.file_exists"))
        << "失败时 file_exists 应为 false（半成品已清理）";
}

// ----------------------------------------------------------------------------
// 2. 音频 mux 失败：writer 报 kInterleavedWriteFailed
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, AudioMuxFailure) {
    const std::string output = UniqueOutputPath("audio_mux_failure");
    ScopedRemove cleanup(output);

    FakeFileBehavior file_behavior;
    file_behavior.write_partial_file = true;

    auto r = RunCliWithFake(GoldenArgs(output),
                            FakeFinalMediaWriter::FailStage::kAudioMux,
                            file_behavior);

    ExpectStageFailure(r, "InterleavedWriteFailed");
    EXPECT_FALSE(fs::exists(output)) << "mux 失败后不应残留输出文件";
}

// ----------------------------------------------------------------------------
// 3. flush 失败：writer 报 kFlushFailed
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, FlushFailure) {
    const std::string output = UniqueOutputPath("flush_failure");
    ScopedRemove cleanup(output);

    FakeFileBehavior file_behavior;
    file_behavior.write_partial_file = true;

    auto r = RunCliWithFake(GoldenArgs(output),
                            FakeFinalMediaWriter::FailStage::kFlush,
                            file_behavior);

    ExpectStageFailure(r, "FlushFailed");
    EXPECT_FALSE(fs::exists(output)) << "flush 失败后不应残留输出文件";
}

// ----------------------------------------------------------------------------
// 4. trailer 失败：writer 报 kTrailerWriteFailed
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, TrailerFailure) {
    const std::string output = UniqueOutputPath("trailer_failure");
    ScopedRemove cleanup(output);

    FakeFileBehavior file_behavior;
    file_behavior.write_partial_file = true;

    auto r = RunCliWithFake(GoldenArgs(output),
                            FakeFinalMediaWriter::FailStage::kTrailer,
                            file_behavior);

    ExpectStageFailure(r, "TrailerWriteFailed");
    EXPECT_FALSE(fs::exists(output)) << "trailer 失败后不应残留输出文件";
}

// ----------------------------------------------------------------------------
// 5. writer 未 Finalize：IsFinalized() == false（无错误码）
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, WriterNotFinalized) {
    const std::string output = UniqueOutputPath("writer_not_finalized");
    ScopedRemove cleanup(output);

    auto r = RunCliWithFake(GoldenArgs(output),
                            FakeFinalMediaWriter::FailStage::kNotFinalized);

    // 1) 非零退出码
    EXPECT_NE(r.exit_code, 0) << "CLI 应因 writer 未 Finalize 返回非零退出码";
    // 2) stdout 恰好一个可解析 JSON 对象
    auto root = ParseExactlyOneJson(r.stdout_text);
    ASSERT_NE(root, nullptr);
    // 3) status != "success"
    EXPECT_EQ(JsonString(root.get(), "status"), "error");
    // 4) 错误消息定位"未 Finalize"阶段
    EXPECT_EQ(JsonString(root.get(), "error.code"), "WriterOutputInvalid");
    EXPECT_NE(JsonString(root.get(), "error.message").find("not finalized"),
              std::string::npos)
        << "error.message 未定位到 not finalized: "
        << JsonString(root.get(), "error.message");
    EXPECT_FALSE(JsonBool(root.get(), "error.writer_finalized"));
    EXPECT_TRUE(JsonString(root.get(), "error.writer_error").empty());
    // 5) 无 MP4 伪产物残留
    EXPECT_FALSE(fs::exists(output));
    // 前置条件：管线成功到达终态
    ASSERT_NE(r.fake, nullptr);
    EXPECT_TRUE(r.fake->terminal_called());
    EXPECT_TRUE(r.fake->terminal_success());
}

// ----------------------------------------------------------------------------
// 6. 输出文件缺失：路径合法但 writer 未落盘任何文件
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, OutputFileMissing) {
    const std::string output = UniqueOutputPath("output_file_missing");
    ScopedRemove cleanup(output);

    // writer 无错误且已 Finalize，但从未创建文件
    auto r = RunCliWithFake(GoldenArgs(output), FakeFinalMediaWriter::FailStage::kNone);

    // 1) 非零退出码
    EXPECT_NE(r.exit_code, 0) << "CLI 应因输出文件缺失返回非零退出码";
    // 2) stdout 恰好一个可解析 JSON 对象
    auto root = ParseExactlyOneJson(r.stdout_text);
    ASSERT_NE(root, nullptr);
    // 3) status != "success"
    EXPECT_EQ(JsonString(root.get(), "status"), "error");
    // 4) 错误消息定位"文件缺失"阶段
    EXPECT_EQ(JsonString(root.get(), "error.code"), "WriterOutputInvalid");
    EXPECT_NE(JsonString(root.get(), "error.message").find("Output file missing"),
              std::string::npos)
        << "error.message 未定位到 Output file missing: "
        << JsonString(root.get(), "error.message");
    EXPECT_TRUE(JsonBool(root.get(), "error.writer_finalized"))
        << "writer 本身成功，writer_finalized 应为 true";
    EXPECT_FALSE(JsonBool(root.get(), "error.file_exists"));
    // 5) 无 MP4 伪产物残留
    EXPECT_FALSE(fs::exists(output));
    // 前置条件：管线成功到达终态
    ASSERT_NE(r.fake, nullptr);
    EXPECT_TRUE(r.fake->terminal_called());
    EXPECT_TRUE(r.fake->terminal_success());
}

// ----------------------------------------------------------------------------
// 7. 输出文件为空：0 字节文件 → CLI 失败，文件随后被删除
// ----------------------------------------------------------------------------

TEST(CliFailureInjectionTest, OutputFileEmpty) {
    const std::string output = UniqueOutputPath("output_file_empty");
    ScopedRemove cleanup(output);

    // writer 无错误且已 Finalize，但只留下 0 字节文件
    FakeFileBehavior file_behavior;
    file_behavior.write_empty_file = true;

    auto r = RunCliWithFake(GoldenArgs(output), FakeFinalMediaWriter::FailStage::kNone,
                            file_behavior);

    // 1) 非零退出码
    EXPECT_NE(r.exit_code, 0) << "CLI 应因输出文件为空返回非零退出码";
    // 2) stdout 恰好一个可解析 JSON 对象
    auto root = ParseExactlyOneJson(r.stdout_text);
    ASSERT_NE(root, nullptr);
    // 3) status != "success"
    EXPECT_EQ(JsonString(root.get(), "status"), "error");
    // 4) 错误消息定位"空文件"阶段
    EXPECT_EQ(JsonString(root.get(), "error.code"), "WriterOutputInvalid");
    EXPECT_NE(JsonString(root.get(), "error.message").find("Output file is empty"),
              std::string::npos)
        << "error.message 未定位到 Output file is empty: "
        << JsonString(root.get(), "error.message");
    EXPECT_TRUE(JsonBool(root.get(), "error.writer_finalized"));
    EXPECT_TRUE(JsonBool(root.get(), "error.file_exists"));
    EXPECT_EQ(JsonNumber(root.get(), "error.file_size", -1.0), 0.0);
    // 5) 0 字节伪产物：确认存在后删除（"无 MP4 残留或文件被删除"）
    EXPECT_TRUE(fs::exists(output));
    EXPECT_EQ(fs::file_size(output), 0);
    std::error_code ec;
    fs::remove(output, ec);
    EXPECT_FALSE(fs::exists(output)) << "空文件伪产物应被删除";
    // 前置条件：管线成功到达终态
    ASSERT_NE(r.fake, nullptr);
    EXPECT_TRUE(r.fake->terminal_called());
    EXPECT_TRUE(r.fake->terminal_success());
}
