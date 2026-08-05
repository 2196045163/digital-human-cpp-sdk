#include "core/timestamp_manager.h"

#include <limits>
#include <string>

extern "C" {
#include <libavutil/mathematics.h>
#include <libavutil/rational.h>
}

// 本文件实现 TimestampManager 的无状态时间换算：
// 调用方传入索引或时间值，接口返回统一的微秒时间戳或明确错误。
namespace digital_human {
namespace core {
namespace {

// 集中定义单位换算常量，避免业务代码中散落 1000、1000000 等魔法数。
constexpr int64_t kMicrosecondsPerSecond = 1000000;
constexpr int64_t kMicrosecondsPerMillisecond = 1000;
constexpr int64_t kHalfMillisecondMicroseconds =
    kMicrosecondsPerMillisecond / 2;

// 统一构造失败结果：失败状态必须精确，错误信息必须非空。
TimestampResult MakeErrorResult(
    TimestampStatus status,
    const std::string& error_message) {
    TimestampResult result;
    result.success = false;
    result.status = status;
    result.error_message = error_message;
    return result;
}

// 统一构造成功结果：成功状态为 kOk，value 的单位固定为微秒。
TimestampResult MakeSuccessResult(int64_t microseconds) {
    TimestampResult result;
    result.success = true;
    result.status = TimestampStatus::kOk;
    result.error_message.clear();
    result.value.microseconds = microseconds;
    return result;
}

}  // namespace

TimestampResult TimestampManager::FromAudioSampleIndex(
    int64_t sample_index,
    int sample_rate) {
    TimestampResult result;

    // sample_index 表示样本在整条音频流中的绝对位置，不能为负数。
    if (sample_index < 0) {
        result = MakeErrorResult(
            TimestampStatus::kNegativeIndex,
            "Audio sample index must be non-negative");
        return result;
    }

    // 采样率是每秒样本数，必须大于零才能组成有效 time base。
    if (sample_rate <= 0) {
        result = MakeErrorResult(
            TimestampStatus::kInvalidSampleRate,
            "Audio sample rate must be greater than zero");
        return result;
    }

    // 将 sample_index × (1/sample_rate) 秒一次性换算为微秒。
    // 使用 FFmpeg 整数 rescale，避免手写乘除造成中间溢出或舍入不一致。
    const AVRational sample_time_base = {1, sample_rate};
    const AVRational microsecond_time_base = {
        1,
        static_cast<int>(kMicrosecondsPerSecond)
    };
    const int64_t microseconds = av_rescale_q(
        sample_index,
        sample_time_base,
        microsecond_time_base);

    // 对非负输入，FFmpeg rescale 的负结果只能表示无法交付的溢出结果。
    if (microseconds < 0) {
        result = MakeErrorResult(
            TimestampStatus::kOverflow,
            "Audio timestamp conversion overflowed int64 microseconds");
        return result;
    }

    result = MakeSuccessResult(microseconds);
    return result;
}

TimestampResult TimestampManager::FromVideoFrameIndex(
    int64_t frame_index,
    const FrameRate& frame_rate) {
    TimestampResult result;

    // frame_index 表示帧在整条视频流中的绝对位置，不能为负数。
    if (frame_index < 0) {
        result = MakeErrorResult(
            TimestampStatus::kNegativeIndex,
            "Video frame index must be non-negative");
        return result;
    }

    // 帧率 numerator/denominator 必须都为正数，才能形成有效时间基准。
    if (frame_rate.numerator <= 0 || frame_rate.denominator <= 0) {
        result = MakeErrorResult(
            TimestampStatus::kInvalidFrameRate,
            "Video frame rate numerator and denominator must be greater than zero");
        return result;
    }

    // fps 为 numerator/denominator，因此每帧持续 denominator/numerator 秒。
    const AVRational video_time_base = {
        frame_rate.denominator,
        frame_rate.numerator
    };
    // 1 秒等于 1,000,000 微秒，因此一个微秒等于 1/1,000,000 秒。
    const AVRational microsecond_time_base = {
        1,
        static_cast<int>(kMicrosecondsPerSecond)
    };
    const int64_t microseconds = av_rescale_q(
        frame_index,
        video_time_base,
        microsecond_time_base);

    // 对非负输入，FFmpeg rescale 的负结果只能表示无法交付的溢出结果。
    if (microseconds < 0) {
        result = MakeErrorResult(
            TimestampStatus::kOverflow,
            "Video timestamp conversion overflowed int64 microseconds");
        return result;
    }

    result = MakeSuccessResult(microseconds);
    return result;
}

TimestampResult TimestampManager::FromMilliseconds(int64_t pts_ms) {
    TimestampResult result;

    // 当前媒体时间轴不接受负时间戳。
    if (pts_ms < 0) {
        result = MakeErrorResult(
            TimestampStatus::kNegativeTimestamp,
            "Timestamp milliseconds must be non-negative");
        return result;
    }

    // 在执行 pts_ms × 1000 前先检查上界，防止有符号整数溢出。
    const int64_t maximum_milliseconds =
        std::numeric_limits<int64_t>::max() /
        kMicrosecondsPerMillisecond;
    if (pts_ms > maximum_milliseconds) {
        result = MakeErrorResult(
            TimestampStatus::kOverflow,
            "Millisecond timestamp conversion overflowed int64 microseconds");
        return result;
    }

    result = MakeSuccessResult(
        pts_ms * kMicrosecondsPerMillisecond);
    return result;
}

TimestampResult TimestampManager::Normalize(
    const MediaTimestamp& timestamp,
    const MediaTimestamp& origin) {
    TimestampResult result;

    // Normalize 只接受成功接口产生的非负媒体时间。
    if (timestamp.microseconds < 0 || origin.microseconds < 0) {
        result = MakeErrorResult(
            TimestampStatus::kNegativeTimestamp,
            "Timestamp and origin must both be non-negative");
        return result;
    }

    // 起点晚于当前时间会得到负结果，本模块用明确状态拒绝。
    if (origin.microseconds > timestamp.microseconds) {
        result = MakeErrorResult(
            TimestampStatus::kOriginAfterTimestamp,
            "Origin must not be later than timestamp");
        return result;
    }

    // 归一化结果表示 timestamp 相对 origin 已经过了多少微秒。
    result = MakeSuccessResult(
        timestamp.microseconds - origin.microseconds);
    return result;
}

int64_t TimestampManager::ToMilliseconds(
    const MediaTimestamp& timestamp) {
    // 先求商和余数，再做最近整数舍入，避免先加 500 导致上界溢出。
    const int64_t whole_milliseconds =
        timestamp.microseconds / kMicrosecondsPerMillisecond;
    const int64_t remaining_microseconds =
        timestamp.microseconds % kMicrosecondsPerMillisecond;

    int64_t rounded_milliseconds = whole_milliseconds;
    if (remaining_microseconds >= kHalfMillisecondMicroseconds) {
        ++rounded_milliseconds;
    } else if (remaining_microseconds <= -kHalfMillisecondMicroseconds) {
        --rounded_milliseconds;
    }

    return rounded_milliseconds;
}

int64_t TimestampManager::DifferenceMicroseconds(
    const MediaTimestamp& lhs,
    const MediaTimestamp& rhs) {
    // 同步场景按 (video, audio) 调用：正值表示视频时间在前，负值表示视频落后。
    return lhs.microseconds - rhs.microseconds;
}

TimestampStatus TimestampManager::ValidateMonotonic(
    const MediaTimestamp& previous,
    const MediaTimestamp& current) {
    // 相同 PTS 合法；只有 current 小于 previous 才属于时间倒退。
    TimestampStatus status = TimestampStatus::kOk;
    if (current.microseconds < previous.microseconds) {
        status = TimestampStatus::kNonMonotonic;
    }
    return status;
}

std::string TimestampManager::StatusToString(TimestampStatus status) {
    // 为日志上层和测试失败信息提供稳定、非空的状态文本；核心库自身不打印。
    std::string status_text;
    switch (status) {
        case TimestampStatus::kOk:
            status_text = "ok";
            break;
        case TimestampStatus::kNegativeIndex:
            status_text = "negative index";
            break;
        case TimestampStatus::kInvalidSampleRate:
            status_text = "invalid sample rate";
            break;
        case TimestampStatus::kInvalidFrameRate:
            status_text = "invalid frame rate";
            break;
        case TimestampStatus::kNegativeTimestamp:
            status_text = "negative timestamp";
            break;
        case TimestampStatus::kOriginAfterTimestamp:
            status_text = "origin after timestamp";
            break;
        case TimestampStatus::kOverflow:
            status_text = "timestamp overflow";
            break;
        case TimestampStatus::kNonMonotonic:
            status_text = "non-monotonic timestamp";
            break;
        case TimestampStatus::kUnknownError:
            status_text = "unknown timestamp error";
            break;
        default:
            status_text = "unrecognized timestamp status";
            break;
    }
    return status_text;
}

std::ostream& operator<<(std::ostream& stream, TimestampStatus status) {
    // 供 GTest 在断言失败时打印枚举含义，不代表核心库主动输出日志。
    stream << TimestampManager::StatusToString(status);
    return stream;
}

}  // namespace core
}  // namespace digital_human
