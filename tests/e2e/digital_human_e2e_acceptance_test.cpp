/// @file digital_human_e2e_acceptance_test.cpp
/// @brief M04 最终 MP4 验收测试：完整用户路径端到端验证
///
/// 从 CLI 入口使用 Golden face.jpg + 3s audio.wav + 正式模型，
/// 在 Linux 本地磁盘生成 MP4 并严格验证。
/// 复用现有 FullOfflinePipeline75Frames 的 Golden 输入和测试工具，
/// 不复制或重写 Pipeline Golden。
///
/// 测试矩阵：
/// - Final MP4 acceptance: CLI → exact 75 frames, dual stream, NormalEos
/// - Artifact consistency: MP4 + manifest + ffprobe 互相一致
/// - Failure injection: 输出不可写/模型错误 → 明确失败，无伪产物
/// - Stability: 同配置 3 轮全部通过
/// - ASAN/TSAN: 通过构建配置覆盖（文档化于 docs/testing/）

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "audio/audio_loader.h"
#include "output/final_media_writer.h"
#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"
#include "core/timestamp_manager.h"

// CMake 编译时注入；若未定义则回退
#ifndef CLI_BINARY_DIR
#define CLI_BINARY_DIR "."
#endif
#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;
using namespace digital_human;

// ============================================================================
// 测试常量
// ============================================================================

namespace {

const fs::path kGoldenImagePath = "testdata/golden/face.jpg";
const fs::path kGoldenAudioPath = "testdata/golden/audio.wav";
const fs::path kModelParamPath = "models/wav2lip/wav2lip.param";
const fs::path kLandmarkModelPath = "models/shape_predictor_68_face_landmarks.dat";
const std::string kDedicatedOutputBase = "/tmp/m04_e2e_acceptance";
// Golden 音频实测恰好 3.000s：3s @ 25fps = 精确 75 帧（75 = 3*25，整数）。
// 断言必须"精确 75"而非"74~76 容差"：离线链路是确定性的，每帧都应被编码
// 写入，容差会掩盖丢帧（74）/重复帧（76）之类的回归，见测试 2 的说明。
const int kExpectedFrames = 75;            // 3s audio @ 25fps
const double kExpectedDuration = 3.0;      // seconds
const double kDurationTolerance = 0.5;     // ±0.5s
const int kFpsNum = 25;
const int kFpsDen = 1;

}  // namespace

// ============================================================================
// CLI 调用辅助（与 digital_human_cli_contract_test.cpp 模式一致）
// ============================================================================

/// @brief 运行 digital_human_app 并捕获 stdout + 退出码
struct CliOutput {
    std::string stdout_text;
    int exit_code = -1;
    bool timed_out = false;
};

/// @brief 返回 digital_human_app 二进制绝对路径
static std::string CliBinaryPath() {
    std::string dir = CLI_BINARY_DIR;
    if (dir.empty()) dir = ".";
    return dir + "/digital_human_app";
}

static CliOutput RunCli(const std::vector<std::string>& extra_args,
                        int timeout_seconds = 180) {
    std::string cmd = CliBinaryPath();
    for (const auto& a : extra_args) {
        cmd += " '" + a + "'";
    }
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

/// @brief 在 JSON 文本中查找简单 key:"value"
static bool JsonHasKeyValue(const std::string& json,
                            const std::string& key,
                            const std::string& expected_value) {
    std::string pattern = "\"" + key + "\": \"" + expected_value + "\"";
    return json.find(pattern) != std::string::npos;
}

/// @brief 在 JSON 文本中查找 key 存在（任意值）
static bool JsonHasKey(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    return json.find(pattern) != std::string::npos;
}

/// @brief 在 JSON 文本中查找 key: null
static bool JsonKeyIsNull(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\": null";
    return json.find(pattern) != std::string::npos;
}

/// @brief 在 JSON 文本中查找 key 的整数值
static int64_t JsonGetIntValue(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\": ";
    auto pos = json.find(pattern);
    if (pos == std::string::npos) return -1;
    pos += pattern.size();
    auto end = json.find_first_of(",\n}", pos);
    if (end == std::string::npos) return -1;
    std::string val_str = json.substr(pos, end - pos);
    // 去除空白
    val_str.erase(std::remove(val_str.begin(), val_str.end(), ' '), val_str.end());
    return std::atoll(val_str.c_str());
}

// ============================================================================
// ffprobe 辅助
// ============================================================================

/// @brief 运行 ffprobe 并返回输出字符串
static std::string RunFfprobe(const std::string& path, const std::string& args) {
    std::string cmd = "ffprobe -v error " + args + " " + path + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[512];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        result += buf;
    }
    pclose(pipe);
    return result;
}

/// @brief 获取指定类型的流数量
static int GetStreamCount(const std::string& path, const std::string& stream_type) {
    std::string sel = stream_type == "video" ? "v" : "a";
    std::string output = RunFfprobe(path,
        "-select_streams " + sel + " -show_entries stream=index -of csv=p=0");
    if (output.empty()) return 0;
    int count = 0;
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty()) count++;
    }
    return count;
}

/// @brief 是否存在到 PTS 单调不递减
static bool FfprobePtsMonotonic(const std::string& path) {
    std::string output = RunFfprobe(path,
        "-select_streams v:0 -show_entries frame=pts_time -of csv=p=0");
    if (output.empty()) return false;

    std::istringstream iss(output);
    std::string line;
    double prev = -1.0;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        double pts = std::atof(line.c_str());
        if (pts < prev - 0.001) {  // 允许微小浮点误差
            return false;
        }
        prev = pts;
    }
    return true;
}

/// @brief MP4 正常 trailer 校验：文件尾部存在 moov atom（非截断/非伪产物）
static bool HasNormalMoovTrailer(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) return false;
    std::streampos size = f.tellg();
    if (size < 8) return false;
    const long kTailBytes = 65536;  // 非 faststart MP4 的 moov 落在文件末尾
    long read_len = std::min<long>(kTailBytes, static_cast<long>(size));
    f.seekg(-read_len, std::ios::end);
    std::string tail(read_len, '\0');
    f.read(&tail[0], read_len);
    if (!f.good()) return false;
    return tail.find("moov") != std::string::npos;
}

/// @brief ffprobe -v error 全量解析无错误（moov/尾部完整 → 正常 trailer）
/// @note 不带 show_entries：健康文件 stdout 为空，任何截断/损坏都会输出错误
static bool FfprobeParsesClean(const std::string& path) {
    return RunFfprobe(path, "").empty();
}

/// @brief 文件是否存在且非空
static bool FileExistsNonEmpty(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

// ============================================================================
// 日志/证据保存
// ============================================================================

/// @brief 将文本保存到指定路径
static void SaveEvidence(const std::string& path, const std::string& content) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
}

// ============================================================================
// 同步 Pipeline Golden Sink（复用自 FullOfflinePipeline75Frames 模式）
// ============================================================================

/// @brief 轻量 Sink：记录帧计数、PTS、task ID、OnTerminal 结果
/// @note 复用现有 GoldenSink 的统计模式，不写入帧文件（由 CLI/MP4 覆盖）
class AcceptanceSink : public pipeline::PipelineOutputSink {
public:
    void OnFrame(const pipeline::PipelineFrame& frame) override {
        std::lock_guard<std::mutex> lock(mutex_);
        frames_.push_back(frame);
        frame_indices_.insert(frame.video_frame.frame_index);
        pts_values_.push_back(frame.video_frame.pts.microseconds);
        task_ids_.insert(frame.source_task_id);
    }

    void OnTerminal(const pipeline::PipelineResult& result) override {
        std::lock_guard<std::mutex> lock(mutex_);
        terminal_result_ = result;
        terminal_called_ = true;
        terminal_cv_.notify_all();
    }

    void WaitForTerminal(std::chrono::milliseconds timeout = std::chrono::milliseconds(180000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        terminal_cv_.wait_for(lock, timeout, [this]() { return terminal_called_; });
    }

    int frame_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(frames_.size());
    }
    bool terminal_called() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return terminal_called_;
    }
    pipeline::PipelineResult terminal_result() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return terminal_result_;
    }
    const std::vector<pipeline::PipelineFrame>& frames() const { return frames_; }

    bool FrameIndicesContinuous(int expected_count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(frame_indices_.size()) != expected_count) {
            return false;
        }
        for (int i = 0; i < expected_count; ++i) {
            if (frame_indices_.count(i) != 1) {
                return false;
            }
        }
        return true;
    }

    bool PtsStrictlyMonotonic() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 1; i < pts_values_.size(); ++i) {
            if (pts_values_[i] <= pts_values_[i - 1]) {
                return false;
            }
        }
        return true;
    }

    bool EachTaskIdOnce() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return task_ids_.size() == frames_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable terminal_cv_;
    std::vector<pipeline::PipelineFrame> frames_;
    std::set<std::int64_t> frame_indices_;
    std::vector<std::int64_t> pts_values_;
    std::set<std::int64_t> task_ids_;
    bool terminal_called_ = false;
    pipeline::PipelineResult terminal_result_;
};

// ============================================================================
// E2E 验收测试夹具
// ============================================================================

class E2EAcceptanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        output_dir_ = kDedicatedOutputBase + "_" +
                      std::to_string(
                          std::chrono::steady_clock::now().time_since_epoch().count());
        output_path_ = output_dir_ + "/output.mp4";
        manifest_path_ = output_dir_ + "/manifest.json";
        ffprobe_report_path_ = output_dir_ + "/ffprobe_report.txt";
        log_path_ = output_dir_ + "/test.log";

        // 专用输出目录在测试前清理且路径受控
        fs::remove_all(output_dir_);
        fs::create_directories(output_dir_);

        // 记录测试日志
        WriteLog("=== M04 E2E Acceptance Test Started ===");
        WriteLog("Output dir: " + output_dir_);
        WriteLog("Output MP4: " + output_path_);
    }

    void TearDown() override {
        // 保留产出物供人工查验；仅在测试通过时可选清理
        WriteLog("=== M04 E2E Acceptance Test Ended ===");
    }

    void WriteLog(const std::string& msg) {
        std::ofstream ofs(log_path_, std::ios::app);
        ofs << msg << "\n";
        ofs.close();
    }

    /// @brief 运行标准 Golden CLI（face + 3s audio + model → MP4）
    CliOutput RunGoldenCli(int timeout_s = 180) {
        return RunCli({
            "--image", kGoldenImagePath.string(),
            "--audio", kGoldenAudioPath.string(),
            "--model-param", kModelParamPath.string(),
            "--landmark", kLandmarkModelPath.string(),
            "--output", output_path_,
            "--fps", std::to_string(kFpsNum),
        }, timeout_s);
    }

    /// @brief 保存 CLI JSON 输出为 manifest
    void SaveManifest(const std::string& json) {
        SaveEvidence(manifest_path_, json);
        WriteLog("Manifest saved: " + manifest_path_);
    }

    /// @brief 保存 ffprobe 报告
    void SaveFfprobeReport() {
        std::string report;
        report += "=== ffprobe Video Stream ===\n";
        report += RunFfprobe(output_path_,
            "-select_streams v:0 -show_entries stream -of json");
        report += "\n=== ffprobe Audio Stream ===\n";
        report += RunFfprobe(output_path_,
            "-select_streams a:0 -show_entries stream -of json");
        report += "\n=== ffprobe Frame Count ===\n";
        report += RunFfprobe(output_path_,
            "-select_streams v:0 -count_frames -show_entries stream=nb_read_frames,r_frame_rate,avg_frame_rate,duration,time_base -of default");
        SaveEvidence(ffprobe_report_path_, report);
        WriteLog("ffprobe report saved: " + ffprobe_report_path_);
    }

    std::string output_dir_;
    std::string output_path_;
    std::string manifest_path_;
    std::string ffprobe_report_path_;
    std::string log_path_;
};

// ============================================================================
// 测试 1: Pipeline Golden Regression（现有 FullOfflinePipeline75Frames 回归）
// ============================================================================

/// @brief 使用现有 Pipeline 直接运行 Golden 输入，验证 75 帧 + NormalEos
/// 此测试复用 FullOfflinePipeline75Frames 的输入和测试工具模式，
/// 但不复制其完整逻辑（仅做精要回归断言）。
TEST_F(E2EAcceptanceTest, PipelineGoldenRegression) {
    ASSERT_TRUE(fs::exists(kGoldenImagePath))
        << "Golden image not found: " << kGoldenImagePath;
    ASSERT_TRUE(fs::exists(kGoldenAudioPath))
        << "Golden audio not found: " << kGoldenAudioPath;
    ASSERT_TRUE(fs::exists(kModelParamPath))
        << "Model param not found: " << kModelParamPath;
    ASSERT_TRUE(fs::exists(kLandmarkModelPath))
        << "Landmark model not found: " << kLandmarkModelPath;

    WriteLog("PipelineGoldenRegression: Starting pipeline...");

    auto sink = std::make_shared<AcceptanceSink>();
    pipeline::DigitalHumanPipeline pipeline;

    pipeline::PipelineConfig config = pipeline::PipelineConfig::OfflineDefault();
    config.image_path = kGoldenImagePath;
    config.audio_path = kGoldenAudioPath;
    config.model_param_path = kModelParamPath;
    config.landmark_model_path = kLandmarkModelPath;
    config.fps_num = kFpsNum;
    config.fps_den = kFpsDen;
    config.q1_capacity = 4;
    config.q2_capacity = 2;
    config.scheduler_worker_count = 1;
    config.ncnn_threads = 1;

    auto t_start = std::chrono::steady_clock::now();

    auto start_result = pipeline.Start(config, sink);
    ASSERT_TRUE(start_result.success)
        << "Pipeline Start failed: " << start_result.error_message;

    // 不调用 RequestStop —— 管线必须经 EOS（音频输入耗尽）自然完成，
    // 终态 kSucceeded 仅在全部帧处理完毕、队列排空后交付。
    // 正常完成应在最后一帧后立即到达；180s 为失败上限。
    sink->WaitForTerminal(std::chrono::milliseconds(180000));

    auto t_end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    int frame_count = sink->frame_count();
    auto stats = pipeline.GetStats();

    // ===== 核心回归断言 =====

    // 1. 精确 75 帧
    EXPECT_EQ(frame_count, kExpectedFrames)
        << "Expected exactly " << kExpectedFrames
        << " frames for 3s audio at 25fps, got " << frame_count;

    // 2. OnTerminal 必须交付，且终态必须为 kSucceeded（正常 EOS 完成）。
    //    拒绝 kDraining / kCancelled —— 任何非正常完成均视为失败。
    EXPECT_TRUE(sink->terminal_called())
        << "OnTerminal was not called within timeout";
    if (sink->terminal_called()) {
        auto term = sink->terminal_result();
        EXPECT_TRUE(term.terminal_state == pipeline::PipelineState::kSucceeded)
            << "Terminal state must be kSucceeded (NormalEos), got: "
            << pipeline::PipelineStateToString(term.terminal_state);
        WriteLog("Terminal state: " +
                 pipeline::PipelineStateToString(term.terminal_state));
    }

    // 3. 零丢帧（离线模式）
    EXPECT_EQ(stats.dropped_late_count, 0);
    EXPECT_EQ(stats.dropped_overflow_count, 0);

    // 4. Frame index 连续
    if (frame_count == kExpectedFrames) {
        EXPECT_TRUE(sink->FrameIndicesContinuous(frame_count));
    }

    // 5. PTS 严格单调
    EXPECT_TRUE(sink->PtsStrictlyMonotonic());

    // 6. 每个 task ID 恰好一次
    EXPECT_TRUE(sink->EachTaskIdOnce());

    // 清理
    pipeline.Stop();

    WriteLog("PipelineGoldenRegression: total_ms=" + std::to_string(total_ms) +
             " frame_count=" + std::to_string(frame_count));

    std::cout << "\n=== M04 Pipeline Golden Regression ===" << std::endl;
    std::cout << "Frames: " << frame_count << std::endl;
    std::cout << "Wall time: " << total_ms << " ms" << std::endl;
    std::cout << "OnTerminal called: " << sink->terminal_called() << std::endl;
    std::cout << "=======================================" << std::endl;
}

// ============================================================================
// 测试 2: Final MP4 Acceptance（CLI → MP4 精确 75 帧、双流、NormalEos）
// ============================================================================
//
// Golden 断言设计原因：
// - 必须要求正常 EOS（kSucceeded），而非 kDraining/kCancelled：只有音频输入
//   耗尽后自然完成的终态才代表"全部帧已被生成并写出"；kDraining/kCancelled
//   意味着调度中途停止，帧序列不完整，属于提前终止回归。
// - 精确 75 帧而非容差：3s @ 25fps 恰好整除（75），离线链路无丢帧/重复的
//   正当理由，任何 ±1 都直接对应一次丢帧或重复帧回归，必须精确暴露。
// - ffprobe 双流验证：确认 mux 后容器结构为恰好 1 video + 1 audio，防止
//   "有画面无声音"、多余流或封装错位等回归。
// - PTS 单调性与 start_pts=0：容器时间戳错乱会破坏播放器 seek/音画同步，
//   是 writer PTS 换算（微秒 → 编码器 time_base）出错的直接证据。
// - trailer 检查（文件尾部 moov atom + ffprobe 全量解析无错误）：未写
//   trailer 的 MP4 无法正常解析，这是 writer 未 Finalize 或中途失败时
//   留下的"坏文件"最直接的检测手段。
// - 同时校验 JSON（status/frame_count/error）与 ffprobe 两路独立证据，
//   互相印证，任何一路不一致都判失败。
//
/// @brief CLI Golden Smoke: 生成 MP4 且 JSON 报告成功
TEST_F(E2EAcceptanceTest, FinalMp4NormalEos) {
    WriteLog("FinalMp4NormalEos: Starting CLI...");

    auto out = RunGoldenCli(180);

    // 保存 manifest 和 ffprobe 证据
    SaveManifest(out.stdout_text);

    EXPECT_EQ(out.exit_code, 0) << "CLI exit code";
    EXPECT_FALSE(out.timed_out) << "CLI timed out";

    // JSON status = "success"
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "success"))
        << "CLI JSON does not report success";
    EXPECT_TRUE(JsonKeyIsNull(out.stdout_text, "error"))
        << "CLI JSON error should be null";

    // MP4 文件存在且非空
    EXPECT_TRUE(FileExistsNonEmpty(output_path_))
        << "MP4 output file does not exist or is empty: " << output_path_;

    // frame_count 必须为精确 75 帧
    int64_t json_frame_count = JsonGetIntValue(out.stdout_text, "frame_count");
    EXPECT_EQ(json_frame_count, kExpectedFrames)
        << "frame_count must be exactly " << kExpectedFrames
        << ", got " << json_frame_count;
    WriteLog("FinalMp4NormalEos: json_frame_count=" + std::to_string(json_frame_count));

    // 保存 ffprobe 证据
    if (FileExistsNonEmpty(output_path_)) {
        SaveFfprobeReport();
    }

    WriteLog("FinalMp4NormalEos: PASS");
}

/// @brief 精确 75 帧 + 双流断言（ffprobe）
TEST_F(E2EAcceptanceTest, Exact75FramesDualStream) {
    WriteLog("Exact75FramesDualStream: Starting CLI...");

    auto out = RunGoldenCli(180);
    ASSERT_EQ(out.exit_code, 0) << "CLI failed, cannot verify MP4";
    ASSERT_TRUE(FileExistsNonEmpty(output_path_)) << "MP4 not produced";

    SaveManifest(out.stdout_text);
    SaveFfprobeReport();

    // 1. 双流：1 video + 1 audio
    int video_streams = GetStreamCount(output_path_, "video");
    int audio_streams = GetStreamCount(output_path_, "audio");
    EXPECT_EQ(video_streams, 1) << "Expected exactly 1 video stream";
    EXPECT_EQ(audio_streams, 1) << "Expected exactly 1 audio stream";

    // 2. 精确帧数：必须正好 75 帧（不接受编码器 ±1 容差）
    std::string nb_frames = RunFfprobe(output_path_,
        "-select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0");
    int probed_frames = 0;
    if (!nb_frames.empty()) {
        nb_frames.erase(std::remove(nb_frames.begin(), nb_frames.end(), '\n'),
                        nb_frames.end());
        probed_frames = std::atoi(nb_frames.c_str());
    }
    EXPECT_EQ(probed_frames, kExpectedFrames)
        << "ffprobe frame count must be exactly " << kExpectedFrames
        << ", got " << probed_frames;

    // 3. 帧率
    std::string r_frame_rate = RunFfprobe(output_path_,
        "-select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0");
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
        s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    };
    trim(r_frame_rate);
    EXPECT_EQ(r_frame_rate, "25/1") << "r_frame_rate mismatch";

    // 4. 时长在批准容差内
    std::string duration_str = RunFfprobe(output_path_,
        "-select_streams v:0 -show_entries stream=duration -of csv=p=0");
    trim(duration_str);
    double duration = 0.0;
    if (!duration_str.empty()) {
        duration = std::atof(duration_str.c_str());
    }
    EXPECT_GE(duration, kExpectedDuration - kDurationTolerance)
        << "Duration too short: " << duration << "s";
    EXPECT_LE(duration, kExpectedDuration + kDurationTolerance)
        << "Duration too long: " << duration << "s";

    // 5. 正常 trailer：moov atom 存在且 ffprobe 全量解析无错误
    EXPECT_TRUE(HasNormalMoovTrailer(output_path_))
        << "MP4 missing moov trailer atom (possibly truncated output)";
    EXPECT_TRUE(FfprobeParsesClean(output_path_))
        << "ffprobe reported errors parsing MP4 (corrupt or truncated trailer)";

    WriteLog("Exact75FramesDualStream: probed_frames=" +
             std::to_string(probed_frames) + " duration=" +
             std::to_string(duration) + "s");
}

/// @brief PTS 单调且起始为 0
TEST_F(E2EAcceptanceTest, PtsMonotonicAndStartZero) {
    WriteLog("PtsMonotonicAndStartZero: Starting CLI...");

    auto out = RunGoldenCli(180);
    ASSERT_EQ(out.exit_code, 0);
    ASSERT_TRUE(FileExistsNonEmpty(output_path_));

    // PTS 单调不递减
    EXPECT_TRUE(FfprobePtsMonotonic(output_path_))
        << "Video PTS is not monotonic";

    // 起始 PTS 为 0
    std::string start_pts = RunFfprobe(output_path_,
        "-select_streams v:0 -show_entries stream=start_pts -of csv=p=0");
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    };
    trim(start_pts);
    EXPECT_EQ(start_pts, "0") << "start_pts should be 0, got: " << start_pts;

    // 视频编码应为 h264
    std::string video_codec = RunFfprobe(output_path_,
        "-select_streams v:0 -show_entries stream=codec_name -of csv=p=0");
    trim(video_codec);
    EXPECT_EQ(video_codec, "h264") << "Video codec: " << video_codec;

    // 音频编码应为 aac
    std::string audio_codec = RunFfprobe(output_path_,
        "-select_streams a:0 -show_entries stream=codec_name -of csv=p=0");
    trim(audio_codec);
    EXPECT_EQ(audio_codec, "aac") << "Audio codec: " << audio_codec;

    WriteLog("PtsMonotonicAndStartZero: PASS");
}

// ============================================================================
// 测试 3: Artifact Consistency（manifest + ffprobe 互相一致）
// ============================================================================

/// @brief CLI JSON 中的 frame_count 与 ffprobe 一致
TEST_F(E2EAcceptanceTest, ManifestConsistentWithFfprobe) {
    WriteLog("ManifestConsistentWithFfprobe: Starting CLI...");

    auto out = RunGoldenCli(180);
    ASSERT_EQ(out.exit_code, 0);
    ASSERT_TRUE(FileExistsNonEmpty(output_path_));

    SaveManifest(out.stdout_text);
    SaveFfprobeReport();

    // CLI JSON frame_count
    int64_t json_frame_count = JsonGetIntValue(out.stdout_text, "frame_count");
    ASSERT_GE(json_frame_count, 0) << "frame_count not found in CLI JSON";

    // ffprobe frame count
    std::string nb_frames = RunFfprobe(output_path_,
        "-select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0");
    int probed_frames = 0;
    if (!nb_frames.empty()) {
        nb_frames.erase(std::remove(nb_frames.begin(), nb_frames.end(), '\n'),
                        nb_frames.end());
        probed_frames = std::atoi(nb_frames.c_str());
    }

    // JSON frame_count 必须与 ffprobe 完全一致，且均为精确 75 帧
    EXPECT_EQ(json_frame_count, probed_frames)
        << "JSON frame_count (" << json_frame_count
        << ") != ffprobe nb_read_frames (" << probed_frames << ")";
    EXPECT_EQ(json_frame_count, kExpectedFrames)
        << "JSON frame_count must be exactly " << kExpectedFrames
        << ", got " << json_frame_count;
    EXPECT_EQ(probed_frames, kExpectedFrames)
        << "ffprobe frame count must be exactly " << kExpectedFrames
        << ", got " << probed_frames;

    // output_path 在 JSON 中存在
    EXPECT_NE(out.stdout_text.find(output_path_), std::string::npos)
        << "output_path not found in CLI JSON";

    // stats 非空
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "stats"))
        << "stats missing in CLI JSON";
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "generated_task_count"))
        << "generated_task_count missing in stats";

    WriteLog("ManifestConsistentWithFfprobe: json_frames=" +
             std::to_string(json_frame_count) + " ffprobe_frames=" +
             std::to_string(probed_frames));
}

/// @brief manifest 非空，所有必需字段存在
TEST_F(E2EAcceptanceTest, ManifestContainsAllRequiredFields) {
    WriteLog("ManifestContainsAllRequiredFields: Starting CLI...");

    auto out = RunGoldenCli(180);
    ASSERT_EQ(out.exit_code, 0);

    SaveManifest(out.stdout_text);

    // 顶层必需字段
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

    WriteLog("ManifestContainsAllRequiredFields: PASS");
}

// ============================================================================
// 测试 4: Failure Injection（输出不可写/模型错误 → 明确失败，无伪产物）
// ============================================================================

/// @brief 输出到不可写路径 → 明确错误，不生成伪产物
TEST_F(E2EAcceptanceTest, OutputNotWritable) {
    WriteLog("OutputNotWritable: Starting CLI with unwritable path...");

    std::string bad_path = "/dev/null/subdir/output.mp4";
    auto out = RunCli({
        "--image", kGoldenImagePath.string(),
        "--audio", kGoldenAudioPath.string(),
        "--model-param", kModelParamPath.string(),
        "--landmark", kLandmarkModelPath.string(),
        "--output", bad_path,
        "--fps", "25",
    }, 60);

    // 应该失败（退出码非 0）
    EXPECT_NE(out.exit_code, 0) << "CLI should have failed with unwritable output";

    // JSON status = "error"
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"))
        << "CLI JSON should report error status";

    // error 字段有内容
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));
    EXPECT_FALSE(JsonKeyIsNull(out.stdout_text, "error"))
        << "error field should not be null on failure";

    // 伪产物不应存在
    EXPECT_FALSE(FileExistsNonEmpty(bad_path))
        << "Pseudo-artifact found at unwritable path: " << bad_path;

    SaveManifest(out.stdout_text);
    WriteLog("OutputNotWritable: exit_code=" + std::to_string(out.exit_code));
}

/// @brief 不存在的模型文件 → 明确失败
TEST_F(E2EAcceptanceTest, MissingModelFile) {
    WriteLog("MissingModelFile: Starting CLI with bad model path...");

    std::string bad_model = "/nonexistent/model.param";
    auto out = RunCli({
        "--image", kGoldenImagePath.string(),
        "--audio", kGoldenAudioPath.string(),
        "--model-param", bad_model,
        "--landmark", kLandmarkModelPath.string(),
        "--output", output_path_,
        "--fps", "25",
    }, 60);

    // 应该失败
    EXPECT_NE(out.exit_code, 0) << "CLI should have failed with missing model";

    // JSON status = "error"
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));

    // error 字段有错误信息
    EXPECT_TRUE(JsonHasKey(out.stdout_text, "error"));
    EXPECT_FALSE(JsonKeyIsNull(out.stdout_text, "error"));

    // 不应生成有效的 MP4
    EXPECT_FALSE(FileExistsNonEmpty(output_path_))
        << "MP4 should not be produced with missing model";

    SaveManifest(out.stdout_text);
    WriteLog("MissingModelFile: exit_code=" + std::to_string(out.exit_code));
}

/// @brief 无效 FPS → 明确失败
TEST_F(E2EAcceptanceTest, InvalidFps) {
    WriteLog("InvalidFps: Starting CLI with fps=0...");

    auto out = RunCli({
        "--image", kGoldenImagePath.string(),
        "--audio", kGoldenAudioPath.string(),
        "--model-param", kModelParamPath.string(),
        "--landmark", kLandmarkModelPath.string(),
        "--output", output_path_,
        "--fps", "0",
    }, 60);

    EXPECT_NE(out.exit_code, 0);
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "error"));
    EXPECT_FALSE(FileExistsNonEmpty(output_path_));

    WriteLog("InvalidFps: exit_code=" + std::to_string(out.exit_code));
}

/// @brief 实时模式拒绝
TEST_F(E2EAcceptanceTest, RealtimeModeRejected) {
    WriteLog("RealtimeModeRejected: Starting CLI with --mode realtime...");

    auto out = RunCli({
        "--image", kGoldenImagePath.string(),
        "--audio", kGoldenAudioPath.string(),
        "--model-param", kModelParamPath.string(),
        "--landmark", kLandmarkModelPath.string(),
        "--output", output_path_,
        "--mode", "realtime",
    }, 60);

    EXPECT_EQ(out.exit_code, 3) << "Realtime mode should return exit code 3";
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "realtime_not_implemented"));
    EXPECT_FALSE(FileExistsNonEmpty(output_path_));

    WriteLog("RealtimeModeRejected: exit_code=" + std::to_string(out.exit_code));
}

// ============================================================================
// 测试 5: Stability（同配置 3 轮全部通过并记录差异）
// ============================================================================
//
// 三轮稳定性判定的必要条件：
// - 单轮通过不足以证明稳定性（ncnn/编码/文件系统的偶发问题只有重复执行
//   才能暴露），故要求同配置连续 3 轮全部通过。
// - 每轮都执行与测试 2 相同的全量断言（退出码、JSON status/error、JSON 与
//   ffprobe 帧数精确 75、PTS 单调、双流、时长容差），任何一轮任一条件不
//   满足即判负——"3 轮全部通过"必须是逐项严格通过，而非只记 3 个退出码。
// - 每轮前清理该轮输出目录：上一轮残留的 MP4 若仍在磁盘上，即使本轮 CLI
//   失败，"文件存在且非空"也会误通过（伪产物误判）；从空目录开始保证
//   产物字节可归因于本轮，这是验收测试不产生假阳性的前提。
// - 另记录耗时 spread 与变异系数（CV）作为证据，但不作为阈值断言——
//   耗时随机器负载波动，不能当作正确性标准。
//
/// @brief 同配置连续运行 3 轮，记录通过情况和差异
TEST_F(E2EAcceptanceTest, StabilityThreeRounds) {
    WriteLog("StabilityThreeRounds: Starting 3 rounds...");

    struct RoundResult {
        int round;
        int exit_code;
        int64_t frame_count;
        double duration_ms;
        int ffprobe_frames = -1;
        double ffprobe_duration = -1.0;
        bool pts_monotonic = false;
        bool dual_stream_ok = false;
        bool passed;
        std::string output_path;
    };

    std::vector<RoundResult> rounds;

    for (int round = 0; round < 3; ++round) {
        std::string round_output_dir = output_dir_ + "/round_" + std::to_string(round);
        // 每轮开始前清理该轮目录：若本轮 CLI 失败，旧轮残留的 MP4 会让
        // "文件存在且非空"误通过（伪产物假阳性），且产物无法归因到本轮
        fs::remove_all(round_output_dir);
        fs::create_directories(round_output_dir);
        std::string round_mp4 = round_output_dir + "/output.mp4";

        WriteLog("StabilityThreeRounds: Round " + std::to_string(round) + " starting...");

        auto t_start = std::chrono::steady_clock::now();

        auto out = RunCli({
            "--image", kGoldenImagePath.string(),
            "--audio", kGoldenAudioPath.string(),
            "--model-param", kModelParamPath.string(),
            "--landmark", kLandmarkModelPath.string(),
            "--output", round_mp4,
            "--fps", "25",
        }, 180);

        auto t_end = std::chrono::steady_clock::now();

        RoundResult rr;
        rr.round = round;
        rr.exit_code = out.exit_code;
        rr.output_path = round_mp4;
        rr.duration_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();

        rr.frame_count = JsonGetIntValue(out.stdout_text, "frame_count");

        // ffprobe + 产物校验
        if (FileExistsNonEmpty(round_mp4)) {
            std::string nb = RunFfprobe(round_mp4,
                "-select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0");
            if (!nb.empty()) {
                nb.erase(std::remove(nb.begin(), nb.end(), '\n'), nb.end());
                rr.ffprobe_frames = std::atoi(nb.c_str());
            }

            std::string dur = RunFfprobe(round_mp4,
                "-select_streams v:0 -show_entries stream=duration -of csv=p=0");
            if (!dur.empty()) {
                dur.erase(std::remove(dur.begin(), dur.end(), '\n'), dur.end());
                rr.ffprobe_duration = std::atof(dur.c_str());
            }

            rr.pts_monotonic = FfprobePtsMonotonic(round_mp4);
            rr.dual_stream_ok =
                GetStreamCount(round_mp4, "video") == 1
                && GetStreamCount(round_mp4, "audio") == 1;

            // 保存本轮 evidence
            SaveEvidence(round_output_dir + "/manifest.json", out.stdout_text);
        }

        // 本轮通过要求全部条件：退出码、JSON 成功、精确 75 帧、
        // ffprobe 精确 75 帧、PTS 单调、时长容差、双流齐全
        rr.passed = (rr.exit_code == 0)
                    && !out.timed_out
                    && JsonHasKeyValue(out.stdout_text, "status", "success")
                    && JsonKeyIsNull(out.stdout_text, "error")
                    && (rr.frame_count == kExpectedFrames)
                    && (rr.ffprobe_frames == kExpectedFrames)
                    && rr.pts_monotonic
                    && rr.dual_stream_ok
                    && (rr.ffprobe_duration >= kExpectedDuration - kDurationTolerance)
                    && (rr.ffprobe_duration <= kExpectedDuration + kDurationTolerance);

        rounds.push_back(rr);

        // 本轮断言（逐项，全部必须通过）
        EXPECT_EQ(rr.exit_code, 0)
            << "Round " << round << " CLI failed with exit code " << rr.exit_code;
        EXPECT_FALSE(out.timed_out)
            << "Round " << round << " CLI timed out";
        EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "success"))
            << "Round " << round << " JSON status is not success";
        EXPECT_TRUE(JsonKeyIsNull(out.stdout_text, "error"))
            << "Round " << round << " JSON error should be null";
        EXPECT_EQ(rr.frame_count, kExpectedFrames)
            << "Round " << round << " JSON frame count must be exactly "
            << kExpectedFrames << ", got " << rr.frame_count;
        EXPECT_EQ(rr.ffprobe_frames, kExpectedFrames)
            << "Round " << round << " ffprobe frame count must be exactly "
            << kExpectedFrames << ", got " << rr.ffprobe_frames;
        EXPECT_TRUE(rr.pts_monotonic)
            << "Round " << round << " video PTS is not monotonic";
        EXPECT_TRUE(rr.dual_stream_ok)
            << "Round " << round << " must have exactly 1 video + 1 audio stream";
        EXPECT_GE(rr.ffprobe_duration, kExpectedDuration - kDurationTolerance)
            << "Round " << round << " duration too short: " << rr.ffprobe_duration << "s";
        EXPECT_LE(rr.ffprobe_duration, kExpectedDuration + kDurationTolerance)
            << "Round " << round << " duration too long: " << rr.ffprobe_duration << "s";

        WriteLog("StabilityThreeRounds: Round " + std::to_string(round) +
                 " exit=" + std::to_string(rr.exit_code) +
                 " frames=" + std::to_string(rr.frame_count) +
                 " ffprobe_frames=" + std::to_string(rr.ffprobe_frames) +
                 " dur=" + std::to_string(rr.duration_ms) + "ms" +
                 " passed=" + (rr.passed ? "YES" : "NO"));
    }

    // 汇总
    int passed_count = 0;
    for (const auto& r : rounds) {
        if (r.passed) passed_count++;
    }

    EXPECT_EQ(passed_count, 3)
        << "Only " << passed_count << " / 3 rounds passed";

    // 计算差异并记录
    if (rounds.size() == 3) {
        double min_dur = std::min({rounds[0].duration_ms,
                                   rounds[1].duration_ms,
                                   rounds[2].duration_ms});
        double max_dur = std::max({rounds[0].duration_ms,
                                   rounds[1].duration_ms,
                                   rounds[2].duration_ms});
        double spread = max_dur - min_dur;
        double mean = (rounds[0].duration_ms + rounds[1].duration_ms +
                       rounds[2].duration_ms) / 3.0;
        double cv = (mean > 0) ? (spread / mean * 100.0) : 0.0;

        WriteLog("StabilityThreeRounds: duration spread=" + std::to_string(spread) +
                 "ms cv=" + std::to_string(cv) + "%");

        std::cout << "\n=== M04 Stability 3 Rounds ===" << std::endl;
        std::cout << "Round 1: " << rounds[0].frame_count << " frames (ffprobe "
                  << rounds[0].ffprobe_frames << "), "
                  << rounds[0].duration_ms << " ms, exit="
                  << rounds[0].exit_code << std::endl;
        std::cout << "Round 2: " << rounds[1].frame_count << " frames (ffprobe "
                  << rounds[1].ffprobe_frames << "), "
                  << rounds[1].duration_ms << " ms, exit="
                  << rounds[1].exit_code << std::endl;
        std::cout << "Round 3: " << rounds[2].frame_count << " frames (ffprobe "
                  << rounds[2].ffprobe_frames << "), "
                  << rounds[2].duration_ms << " ms, exit="
                  << rounds[2].exit_code << std::endl;
        std::cout << "Duration spread: " << spread << " ms" << std::endl;
        std::cout << "CV: " << cv << " %" << std::endl;
        std::cout << "All passed: " << (passed_count == 3 ? "YES" : "NO") << std::endl;
        std::cout << "==============================" << std::endl;
    }
}

// ============================================================================
// 测试 6: Model Bin 自动推导
// ============================================================================

/// @brief 不传 --model-bin，验证自动从 .param 推导 .bin
TEST_F(E2EAcceptanceTest, AutoModelBinDerivation) {
    WriteLog("AutoModelBinDerivation: Starting CLI without --model-bin...");

    auto out = RunCli({
        "--image", kGoldenImagePath.string(),
        "--audio", kGoldenAudioPath.string(),
        "--model-param", kModelParamPath.string(),
        "--landmark", kLandmarkModelPath.string(),
        "--output", output_path_,
        "--fps", "25",
    }, 180);

    EXPECT_EQ(out.exit_code, 0) << "CLI should succeed with auto-derived model-bin";
    EXPECT_TRUE(JsonHasKeyValue(out.stdout_text, "status", "success"));
    EXPECT_TRUE(JsonKeyIsNull(out.stdout_text, "error"));

    if (out.exit_code == 0) {
        EXPECT_TRUE(FileExistsNonEmpty(output_path_))
            << "MP4 should be produced with auto-derived model-bin";
    }

    SaveManifest(out.stdout_text);
    WriteLog("AutoModelBinDerivation: exit_code=" + std::to_string(out.exit_code));
}

// ============================================================================
// 测试 7: CLI 超时保护
// ============================================================================

/// @brief 验证 CLI 在合理时间内完成（不作为取消/fallback 伪装完成）
TEST_F(E2EAcceptanceTest, CompletesInReasonableTime) {
    WriteLog("CompletesInReasonableTime: Starting CLI...");

    auto t_start = std::chrono::steady_clock::now();

    auto out = RunGoldenCli(300);

    auto t_end = std::chrono::steady_clock::now();
    auto elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    EXPECT_EQ(out.exit_code, 0);
    EXPECT_FALSE(out.timed_out) << "CLI timed out after " << elapsed_s << "s";

    // 180 秒内必须完成（ncnn 推理时间在 CI 机器上预期 < 120s）
    EXPECT_LT(elapsed_s, 180.0)
        << "CLI took " << elapsed_s << "s, expected < 180s";

    WriteLog("CompletesInReasonableTime: elapsed=" + std::to_string(elapsed_s) + "s");
}

// ============================================================================
// 测试 8: Schema Version 一致性
// ============================================================================

/// @brief 验证 JSON schema_version 始终为 1
TEST_F(E2EAcceptanceTest, SchemaVersionIsOne) {
    auto out_success = RunGoldenCli(180);
    if (out_success.exit_code == 0) {
        EXPECT_TRUE(JsonHasKey(out_success.stdout_text, "schema_version"));
        EXPECT_NE(out_success.stdout_text.find("\"schema_version\": 1"),
                  std::string::npos);
    }

    auto out_error = RunCli({"--unknown-flag"}, 10);
    EXPECT_NE(out_error.stdout_text.find("\"schema_version\": 1"),
              std::string::npos);
}

// ============================================================================
// 测试 9: 音频流验证
// ============================================================================

/// @brief 验证音频流采样率和时长
TEST_F(E2EAcceptanceTest, AudioStreamValid) {
    auto out = RunGoldenCli(180);
    ASSERT_EQ(out.exit_code, 0);
    ASSERT_TRUE(FileExistsNonEmpty(output_path_));

    // 音频流存在
    EXPECT_EQ(GetStreamCount(output_path_, "audio"), 1);

    // 采样率
    std::string sample_rate = RunFfprobe(output_path_,
        "-select_streams a:0 -show_entries stream=sample_rate -of csv=p=0");
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    };
    trim(sample_rate);
    EXPECT_EQ(sample_rate, "16000") << "Audio sample rate: " << sample_rate;

    // 声道数
    std::string channels = RunFfprobe(output_path_,
        "-select_streams a:0 -show_entries stream=channels -of csv=p=0");
    trim(channels);
    EXPECT_EQ(channels, "1") << "Audio channels: " << channels;

    WriteLog("AudioStreamValid: sample_rate=" + sample_rate +
             " channels=" + channels);
}
