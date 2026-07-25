/// @file timestamp_manager_test.cpp
/// @brief 演示音频样本索引和视频帧索引如何进入同一微秒时间轴。
///
/// 数据流：
/// frame_index / audio_sample_index
///   -> TimestampManager
///   -> MediaTimestamp(us)
///   -> DifferenceMicroseconds(video, audio)
///   -> ToMilliseconds
///   -> 控制台表格 + golden_output 文本证据

#include "core/timestamp_manager.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace {

constexpr int kDefaultSampleRate = 16000;
constexpr int kDefaultFpsNumerator = 25;
constexpr int kDefaultFpsDenominator = 1;
constexpr int64_t kDefaultFrameCount = 5;
constexpr int64_t kMaximumFrameCount = 100000;
constexpr int kFrameIndexColumnWidth = 12;
constexpr int kAudioSampleIndexColumnWidth = 20;
constexpr int kTimestampColumnWidth = 16;
constexpr int kDifferenceColumnWidth = 12;
constexpr char kGoldenOutputPath[] =
    "golden_output/timestamp_manager_example.txt";

/// @brief 解析必须大于零的 int 参数；失败时不修改输出值。
bool ParsePositiveInt(const char* text, int* value) {
    const std::string_view input(text);
    int parsed_value = 0;
    const std::from_chars_result parse_result = std::from_chars(
        input.data(),
        input.data() + input.size(),
        parsed_value);
    if (parse_result.ec != std::errc() ||
        parse_result.ptr != input.data() + input.size() ||
        parsed_value <= 0) {
        return false;
    }

    *value = parsed_value;
    return true;
}

/// @brief 解析受上限保护的帧数，避免 example 意外生成超大文件。
bool ParseFrameCount(const char* text, int64_t* frame_count) {
    const std::string_view input(text);
    int64_t parsed_value = 0;
    const std::from_chars_result parse_result = std::from_chars(
        input.data(),
        input.data() + input.size(),
        parsed_value);
    if (parse_result.ec != std::errc() ||
        parse_result.ptr != input.data() + input.size() ||
        parsed_value <= 0 ||
        parsed_value > kMaximumFrameCount) {
        return false;
    }

    *frame_count = parsed_value;
    return true;
}

/// @brief 为指定视频帧选择最接近同一媒体时刻的音频样本索引。
/// @note 这只是 example 的输入构造，不属于 TimestampManager 生产接口。
bool ComputeAlignedAudioSampleIndex(
    int64_t frame_index,
    int sample_rate,
    const digital_human::core::FrameRate& frame_rate,
    int64_t* sample_index) {
    const long double exact_sample_index =
        static_cast<long double>(frame_index) *
        static_cast<long double>(sample_rate) *
        static_cast<long double>(frame_rate.denominator) /
        static_cast<long double>(frame_rate.numerator);
    if (exact_sample_index >
        static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }

    *sample_index = static_cast<int64_t>(
        std::llround(exact_sample_index));
    return true;
}

/// @brief 将表格写入 golden_output；目录或文件创建失败会显式返回 false。
bool WriteGoldenOutput(
    const std::filesystem::path& output_path,
    const std::string& content) {
    std::error_code error;
    std::filesystem::create_directories(
        output_path.parent_path(),
        error);
    if (error) {
        std::cerr << "ERROR: Cannot create golden output directory: "
                  << error.message() << '\n';
        return false;
    }

    std::ofstream output(output_path);
    if (!output.is_open()) {
        std::cerr << "ERROR: Cannot write " << output_path << '\n';
        return false;
    }

    output << content;
    if (!output.good()) {
        std::cerr << "ERROR: Failed while writing " << output_path << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    int sample_rate = kDefaultSampleRate;
    int fps_numerator = kDefaultFpsNumerator;
    int fps_denominator = kDefaultFpsDenominator;
    int64_t frame_count = kDefaultFrameCount;

    if (argc > 5 ||
        (argc >= 2 && !ParsePositiveInt(argv[1], &sample_rate)) ||
        (argc >= 3 && !ParsePositiveInt(argv[2], &fps_numerator)) ||
        (argc >= 4 && !ParsePositiveInt(argv[3], &fps_denominator)) ||
        (argc >= 5 && !ParseFrameCount(argv[4], &frame_count))) {
        std::cerr
            << "Usage: timestamp_manager_test "
            << "[sample_rate] [fps_num] [fps_den] [frame_count]\n"
            << "All values must be positive; frame_count must be <= "
            << kMaximumFrameCount << ".\n";
        return 1;
    }

    const digital_human::core::FrameRate frame_rate{
        fps_numerator,
        fps_denominator
    };
    std::ostringstream report;
    report << "sample_rate=" << sample_rate
           << " fps=" << fps_numerator << '/' << fps_denominator
           << " frame_count=" << frame_count << '\n';
    report << std::left
           << std::setw(kFrameIndexColumnWidth) << "frame_index"
           << std::setw(kAudioSampleIndexColumnWidth) << "audio_sample_index"
           << std::setw(kTimestampColumnWidth) << "audio_pts_us"
           << std::setw(kTimestampColumnWidth) << "video_pts_us"
           << std::setw(kDifferenceColumnWidth) << "diff_us"
           << "video_pts_ms\n";

    for (int64_t frame_index = 0;
         frame_index < frame_count;
         ++frame_index) {
        int64_t audio_sample_index = 0;
        if (!ComputeAlignedAudioSampleIndex(
                frame_index,
                sample_rate,
                frame_rate,
                &audio_sample_index)) {
            std::cerr << "ERROR: Audio sample index overflow at frame "
                      << frame_index << '\n';
            return 1;
        }

        const digital_human::core::TimestampResult audio_timestamp =
            digital_human::core::TimestampManager::FromAudioSampleIndex(
                audio_sample_index,
                sample_rate);
        const digital_human::core::TimestampResult video_timestamp =
            digital_human::core::TimestampManager::FromVideoFrameIndex(
                frame_index,
                frame_rate);
        if (!audio_timestamp.success || !video_timestamp.success) {
            std::cerr << "ERROR: Timestamp conversion failed at frame "
                      << frame_index << ": audio="
                      << audio_timestamp.error_message << ", video="
                      << video_timestamp.error_message << '\n';
            return 1;
        }

        const int64_t difference_microseconds =
            digital_human::core::TimestampManager::DifferenceMicroseconds(
                video_timestamp.value,
                audio_timestamp.value);
        const int64_t video_timestamp_milliseconds =
            digital_human::core::TimestampManager::ToMilliseconds(
                video_timestamp.value);

        report << std::left
               << std::setw(kFrameIndexColumnWidth) << frame_index
               << std::setw(kAudioSampleIndexColumnWidth)
               << audio_sample_index
               << std::setw(kTimestampColumnWidth)
               << audio_timestamp.value.microseconds
               << std::setw(kTimestampColumnWidth)
               << video_timestamp.value.microseconds
               << std::setw(kDifferenceColumnWidth)
               << difference_microseconds
               << video_timestamp_milliseconds << '\n';
    }

    // 偏移样例只展示差值方向；动作策略属于后续 FrameScheduler。
    const digital_human::core::MediaTimestamp forty_milliseconds{40000};
    const digital_human::core::MediaTimestamp fifty_milliseconds{50000};
    report << "offset_examples:\n"
           << "video_50ms_minus_audio_40ms="
           << digital_human::core::TimestampManager::DifferenceMicroseconds(
                  fifty_milliseconds,
                  forty_milliseconds)
           << "us\n"
           << "video_40ms_minus_audio_50ms="
           << digital_human::core::TimestampManager::DifferenceMicroseconds(
                  forty_milliseconds,
                  fifty_milliseconds)
           << "us\n";

    const std::string report_text = report.str();
    std::cout << report_text;
    if (!WriteGoldenOutput(kGoldenOutputPath, report_text)) {
        return 1;
    }

    std::cout << "Golden output: " << kGoldenOutputPath << '\n';
    return 0;
}
