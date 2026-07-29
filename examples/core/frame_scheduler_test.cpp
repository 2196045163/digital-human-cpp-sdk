/// @example core/frame_scheduler_test.cpp
/// @brief 帧调度模块合成示例，输出确定性 golden trace。
///
/// 使用具名配置和合成 CV_8UC3 帧演示 PushFrame / Schedule 的基本数据流。
/// 不依赖 GUI、真实模型或音频设备。
///
/// 输出同时写入 stdout 和 golden_output/frame_scheduler_example.txt，
/// 后者用于回归对比，防止自欺欺人。
///
/// 核心库不打印——所有输出由本示例负责。

#include "core/frame_scheduler.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace digital_human::core;
using namespace digital_human::video;

namespace {

/// @brief 示例配置（非通用值，仅用于演示）。
const FrameSchedulerConfig kDemoConfig{
    /*max_queue_size=*/6,
    /*min_buffered_frames=*/2,
    /*sync_tolerance_us=*/20000,  // 20ms 演示容忍窗口
    /*overflow_policy=*/OverflowPolicy::kDropOldest
};

/// @brief 创建最小的合法 CV_8UC3 帧用于演示。
VideoFrame MakeDemoFrame(std::int64_t pts_us, std::int64_t index) {
    VideoFrame frame;
    frame.pts.microseconds = pts_us;
    frame.frame_index = index;
    // 64x64 小图，只用于证明链路接通
    frame.frame_bgr = cv::Mat(64, 64, CV_8UC3, cv::Scalar(50, 100, 150));
    return frame;
}

/// @brief 将 ScheduleAction 转为可读字符串。
const char* ActionToString(ScheduleAction action) {
    switch (action) {
    case ScheduleAction::kDeliver:                      return "Deliver";
    case ScheduleAction::kDropAndDeliver:               return "DropAndDeliver";
    case ScheduleAction::kRepeatLast:                   return "RepeatLast";
    case ScheduleAction::kBufferingNoFrame:             return "BufferingNoFrame";
    case ScheduleAction::kRejectedInvalidReferenceTime: return "RejectedInvalidRef";
    }
    return "?";
}

/// @brief 将 PushResult 转为可读字符串。
const char* PushResultToString(PushResult result) {
    switch (result) {
    case PushResult::kAccepted:             return "Accepted";
    case PushResult::kDroppedOldest:        return "DroppedOldest";
    case PushResult::kRejectedQueueFull:    return "RejectedQueueFull";
    case PushResult::kRejectedDuplicatePts: return "RejectedDuplicatePts";
    case PushResult::kRejectedStalePts:     return "RejectedStalePts";
    case PushResult::kRejectedInvalidFrame: return "RejectedInvalidFrame";
    }
    return "?";
}

}  // namespace

int main() {
    // 确保 golden_output 目录存在
    std::filesystem::create_directories("golden_output");

    // 同时输出到 stdout 和 golden 文件
    std::ofstream golden("golden_output/frame_scheduler_example.txt");
    if (!golden) {
        std::cerr << "Failed to create golden_output/frame_scheduler_example.txt\n";
        return 1;
    }

    // 辅助 lambda：同时写入 stdout 和 golden 文件
    auto emit = [&golden](const std::string& line) {
        std::cout << line;
        golden << line;
    };

    FrameScheduler scheduler(kDemoConfig);

    // 1. 推入一组帧（PTS 故意不按顺序，检查点 4 实现有序插入后会自动排序）
    emit("=== PushFrames ===\n");
    struct {
        std::int64_t pts_us;
        std::int64_t index;
    } frames[] = {
        {300000, 3},
        {100000, 1},
        {200000, 2},
        {  0000, 0},
    };
    for (auto& f : frames) {
        auto frame = MakeDemoFrame(f.pts_us, f.index);
        auto result = scheduler.PushFrame(frame);
        std::ostringstream oss;
        oss << "Push PTS=" << f.pts_us
            << " index=" << f.index
            << " -> " << PushResultToString(result) << "\n";
        emit(oss.str());
    }

    // 2. 按参考时钟逐步调度
    emit("\n=== Schedule ===\n");
    std::int64_t refs[] = {0, 50000, 110000, 210000, 310000, 410000};
    for (auto ref : refs) {
        MediaTimestamp ref_ts{ref};
        auto result = scheduler.Schedule(ref_ts);

        std::ostringstream oss;
        oss << "ref=" << ref
            << " action=" << ActionToString(result.action)
            << " index=";
        if (result.selected_frame.has_value()) {
            oss << result.selected_frame->frame_index;
        } else {
            oss << "none";
        }
        oss << " diff=";
        if (result.timing_diff_us.has_value()) {
            oss << result.timing_diff_us.value();
        } else {
            oss << "none";
        }
        oss << " dropped=" << result.dropped_this_call
            << " qsize=" << result.queue_size_after
            << " state=" << (result.state_after == SchedulerState::kRunning
                             ? "Running" : "Buffering")
            << "\n";
        emit(oss.str());
    }

    // 3. Reset 后重新开始
    emit("\n=== After Reset ===\n");
    scheduler.Reset();
    auto frame = MakeDemoFrame(100000, 10);
    std::ostringstream oss;
    oss << "Push after reset: "
        << PushResultToString(scheduler.PushFrame(frame)) << "\n";
    emit(oss.str());

    golden.close();
    std::cout << "\nGolden output written to golden_output/frame_scheduler_example.txt\n";
    return 0;
}
