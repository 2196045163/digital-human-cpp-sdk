#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace digital_human {
namespace core {

/// @brief 统一媒体时间戳，单位固定为微秒。
struct MediaTimestamp {
    int64_t microseconds = 0;
};

/// @brief 有理帧率：每秒 numerator / denominator 帧。
struct FrameRate {
    int numerator = 25;
    int denominator = 1;
};

/// @brief 时间戳换算和校验状态。
enum class TimestampStatus {
    kOk,
    kNegativeIndex,
    kInvalidSampleRate,
    kInvalidFrameRate,
    kNegativeTimestamp,
    kOriginAfterTimestamp,
    kOverflow,
    kNonMonotonic,
    kUnknownError
};

/// @brief 时间戳换算结果。
/// @note 成功时 success=true、status=kOk、error_message 为空且 value 有效；
///       失败时 success=false、status 精确、error_message 非空且 value 不可使用。
struct TimestampResult {
    bool success = false;
    TimestampStatus status = TimestampStatus::kUnknownError;
    std::string error_message;
    MediaTimestamp value;
};

/// @brief 将音频/视频索引转换到统一微秒时间轴，并提供无状态时间运算。
class TimestampManager {
public:
    /// @brief 将流中的音频采样索引换算为微秒 PTS。
    static TimestampResult FromAudioSampleIndex(
        int64_t sample_index,
        int sample_rate);

    /// @brief 将视频帧索引和有理帧率换算为微秒 PTS。
    /// @note 此函数由用户在检查点 2 亲手实现。
    static TimestampResult FromVideoFrameIndex(
        int64_t frame_index,
        const FrameRate& frame_rate);

    /// @brief 将非负整数毫秒安全转换为微秒时间戳。
    static TimestampResult FromMilliseconds(int64_t pts_ms);

    /// @brief 将时间戳按 origin 归一化。
    static TimestampResult Normalize(
        const MediaTimestamp& timestamp,
        const MediaTimestamp& origin);

    /// @brief 将微秒转换为最近的整数毫秒，中点远离零。
    static int64_t ToMilliseconds(const MediaTimestamp& timestamp);

    /// @brief 返回 lhs - rhs；同步调用约定为 video_pts - audio_pts。
    /// @pre lhs 和 rhs 必须来自成功结果，因此均为非负时间戳。
    static int64_t DifferenceMicroseconds(
        const MediaTimestamp& lhs,
        const MediaTimestamp& rhs);

    /// @brief 校验时间是否单调不减。
    static TimestampStatus ValidateMonotonic(
        const MediaTimestamp& previous,
        const MediaTimestamp& current);

    /// @brief 将所有状态转换为非空可读字符串。
    static std::string StatusToString(TimestampStatus status);
};

/// @brief 支持测试框架输出 TimestampStatus。
std::ostream& operator<<(std::ostream& stream, TimestampStatus status);

}  // namespace core
}  // namespace digital_human
