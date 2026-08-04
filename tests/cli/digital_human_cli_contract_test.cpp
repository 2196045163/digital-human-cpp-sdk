/// @file digital_human_cli_contract_test.cpp
/// @brief M03 CLI 契约测试：help、参数错误、实时拒绝、集成 smoke
///
/// 测试矩阵覆盖：
/// - Contract: --help / 未知参数 → 稳定文本和退出码
/// - Error: 缺文件 / 坏路径 / realtime → JSON 与退出码一致
/// - Integration: Golden face+audio+model → MP4 + 成功 JSON
/// - Regression: 已有模块回归（通过 CTest 并行运行）
///
/// 所有测试通过 popen 调起 digital_human_app 二进制并捕获 stdout + 退出码。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// CMake 编译时注入；若未定义则回退
#ifndef CLI_BINARY_DIR
#define CLI_BINARY_DIR "."
#endif
#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

// ============================================================================
// 测试辅助
// ============================================================================

/// @brief 运行 digital_human_app 并捕获 stdout + 退出码
struct CliOutput {
    std::string stdout_text;
    int exit_code = -1;
    bool timed_out = false;
};

/// @brief 返回 digital_human_app 二进制绝对路径
static std::string CliBinaryPath() {
    // CLI_BINARY_DIR 和 PROJECT_SOURCE_DIR 由 CMake 在编译时注入
    // 默认回退到相对路径
    std::string dir = CLI_BINARY_DIR;
    if (dir.empty()) dir = ".";
    return dir + "/digital_human_app";
}

static CliOutput RunCli(const std::vector<std::string>& extra_args,
                        int timeout_seconds = 120) {
    std::string cmd = CliBinaryPath();
    for (const auto& a : extra_args) {
        cmd += " '" + a + "'";
    }
    // 将 stderr 重定向到 stdout 一并捕获
    cmd += " 2>&1";

    std::string timeout_cmd =
        "timeout " + std::to_string(timeout_seconds) + " " + cmd;

    FILE* pipe = popen(timeout_cmd.c_str(), "r");
    if (!pipe) {
        return {{}, -1, false};
    }

    CliOutput out;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        out.stdout_text += buf;
    }

    int status = pclose(pipe);
    if (WIFEXITED(status)) {
        int raw = WEXITSTATUS(status);
        // timeout 命令在超时时返回 124
        if (raw == 124) {
            out.timed_out = true;
            out.exit_code = raw;
        } else {
            out.exit_code = raw;
        }
    } else {
        out.exit_code = -1;
    }

    return out;
}

/// @brief 在 JSON 文本中查找简单 key:"value" 或 key:value
static bool JsonHasKeyValue(const std::string& json,
                            const std::string& key,
                            const std::string& expected_value) {
    std::string pattern = "\"" + key + "\": \"" + expected_value + "\"";
    return json.find(pattern) != std::string::npos;
}

/// @brief 在 JSON 文本中查找 key: null
static bool JsonKeyIsNull(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\": null";
    return json.find(pattern) != std::string::npos;
}

/// @brief 在 JSON 文本中查找 key 存在（任意值）
static bool JsonHasKey(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    return json.find(pattern) != std::string::npos;
}

// ============================================================================
// Contract 层: --help / 未知参数
// ============================================================================

TEST(CliContractTest, HelpFlagShowsUsage) {
    auto out = RunCli({"--help"});
    EXPECT_EQ(out.exit_code, 0);
    EXPECT_NE(out.stdout_text.find("Usage:"), std::string::npos);
    EXPECT_NE(out.stdout_text.find("--image"), std::string::npos);
    EXPECT_NE(out.stdout_text.find("--audio"), std::string::npos);
    EXPECT_NE(out.stdout_text.find("--model-param"), std::string::npos);
    EXPECT_NE(out.stdout_text.find("--landmark"), std::string::npos);
    EXPECT_NE(out.stdout_text.find("--output"), std::string::npos);
    EXPECT_NE(out.stdout_text.find("Exit codes:"), std::string::npos);
}

TEST(CliContractTest, HelpFlagWithDashH) {
    auto out = RunCli({"-h"});
    EXPECT_EQ(out.exit_code, 0);
    EXPECT_NE(out.stdout_text.find("Usage:"), std::string::npos);
}

TEST(CliContractTest, UnknownArgumentReturnsExit2) {
    auto out = RunCli({"--unknown-flag"});
    EXPECT_EQ(out.exit_code, 2);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));
}

TEST(CliContractTest, NoArgumentsReturnsExit2) {
    auto out = RunCli({});
    EXPECT_EQ(out.exit_code, 2);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));
}

// ============================================================================
// Error 层: 缺文件 / 坏路径 / realtime
// ============================================================================

TEST(CliErrorTest, MissingRequiredArgs) {
    // 只给部分参数
    auto out = RunCli({"--image", "/nonexistent/face.jpg"});
    EXPECT_EQ(out.exit_code, 2);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));
}

TEST(CliErrorTest, FileNotFoundReturnsExit1) {
    auto out = RunCli({
        "--image", "/nonexistent/face.jpg",
        "--audio", "/nonexistent/audio.wav",
        "--model-param", "/nonexistent/model.param",
        "--landmark", "/nonexistent/landmark.dat",
        "--output", "/tmp/test_output.mp4",
    });
    EXPECT_EQ(out.exit_code, 1);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));
}

TEST(CliErrorTest, InvalidFpsReturnsExit2) {
    auto out = RunCli({
        "--image", "testdata/golden/face.jpg",
        "--audio", "testdata/golden/audio.wav",
        "--model-param", "models/wav2lip/wav2lip.param",
        "--landmark", "models/shape_predictor_68_face_landmarks.dat",
        "--output", "/tmp/test_output.mp4",
        "--fps", "0",
    });
    EXPECT_EQ(out.exit_code, 2);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));
}

TEST(CliErrorTest, RealtimeModeReturnsExit3) {
    auto out = RunCli({
        "--image", "testdata/golden/face.jpg",
        "--audio", "testdata/golden/audio.wav",
        "--model-param", "models/wav2lip/wav2lip.param",
        "--landmark", "models/shape_predictor_68_face_landmarks.dat",
        "--output", "/tmp/test_output.mp4",
        "--mode", "realtime",
    });
    EXPECT_EQ(out.exit_code, 3);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "realtime_not_implemented"));
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "code", "RealtimeNotImplemented"));
}

TEST(CliErrorTest, OutputJsonHasSchemaVersion) {
    auto out = RunCli({"--unknown-flag"});
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "schema_version"));
    EXPECT_NE(out.stdout_text.find("\"schema_version\": 1"), std::string::npos);
}

// ============================================================================
// Integration 层: Golden face+audio+model
// ============================================================================

TEST(CliIntegrationTest, GoldenSmokeProducesMp4AndSuccessJson) {
    // 使用临时输出路径避免污染仓库
    std::string output_path = "/tmp/m03_cli_golden_smoke_test.mp4";

    // 清理可能残留的输出文件
    std::remove(output_path.c_str());

    auto out = RunCli({
        "--image", "testdata/golden/face.jpg",
        "--audio", "testdata/golden/audio.wav",
        "--model-param", "models/wav2lip/wav2lip.param",
        "--model-bin", "models/wav2lip/wav2lip.bin",
        "--landmark", "models/shape_predictor_68_face_landmarks.dat",
        "--output", output_path,
        "--fps", "25",
    }, 120);

    EXPECT_EQ(out.exit_code, 0);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "success"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "frame_count"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "duration_ms"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "stats"));
    EXPECT_TRUE(JsonKeyIsNull(out.stdout_text, "error"));

    // 验证 MP4 文件已生成且非空
    EXPECT_TRUE(fs::exists(output_path));
    auto file_size = fs::file_size(output_path);
    EXPECT_GT(file_size, 0);

    // 验证输出路径在 JSON 中
    EXPECT_NE(out.stdout_text.find(output_path), std::string::npos);

    // 清理
    std::remove(output_path.c_str());
}

TEST(CliIntegrationTest, GoldenSmokeAutoModelBinDerivation) {
    // 不传 --model-bin，测试自动推导（.param → .bin）
    std::string output_path = "/tmp/m03_cli_auto_bin_test.mp4";
    std::remove(output_path.c_str());

    auto out = RunCli({
        "--image", "testdata/golden/face.jpg",
        "--audio", "testdata/golden/audio.wav",
        "--model-param", "models/wav2lip/wav2lip.param",
        "--landmark", "models/shape_predictor_68_face_landmarks.dat",
        "--output", output_path,
        "--fps", "25",
    }, 120);

    EXPECT_EQ(out.exit_code, 0);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "success"));
    EXPECT_TRUE(JsonKeyIsNull(out.stdout_text, "error"));

    EXPECT_TRUE(fs::exists(output_path));
    EXPECT_GT(fs::file_size(output_path), 0);

    std::remove(output_path.c_str());
}

// ============================================================================
// 完整 JSON 结构验证
// ============================================================================

TEST(CliContractTest, SuccessJsonContainsAllRequiredFields) {
    std::string output_path = "/tmp/m03_cli_schema_test.mp4";
    std::remove(output_path.c_str());

    auto out = RunCli({
        "--image", "testdata/golden/face.jpg",
        "--audio", "testdata/golden/audio.wav",
        "--model-param", "models/wav2lip/wav2lip.param",
        "--model-bin", "models/wav2lip/wav2lip.bin",
        "--landmark", "models/shape_predictor_68_face_landmarks.dat",
        "--output", output_path,
        "--fps", "25",
    }, 120);

    EXPECT_EQ(out.exit_code, 0);

    // 必需顶层字段
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "schema_version"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "status"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "output_path"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "frame_count"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "duration_ms"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "stats"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));

    // stats 子字段
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "generated_task_count"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "rendered_unique_frame_count"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "total_wall_time_ms"));

    std::remove(output_path.c_str());
}

// ============================================================================
// 错误 JSON 结构验证
// ============================================================================

TEST(CliContractTest, ErrorJsonContainsRequiredFields) {
    auto out = RunCli({"--unknown-flag"});

    EXPECT_TRUE(JsonHasKey(out.stdout_text, "schema_version"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "status"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "output_path"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "frame_count"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "duration_ms"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "stats"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));

    // error 子字段
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "code"));
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "message"));
}
