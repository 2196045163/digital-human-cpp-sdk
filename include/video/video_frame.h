#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

#include "core/timestamp_manager.h"

namespace digital_human {
namespace video {

/// @brief 已完成融合、可直接显示的视频帧。
///
/// 只携带三个字段：
/// - frame_bgr: 完整 CV_8UC3 显示图像（通常来自 FaceBlender::final_bgr）
/// - pts: 排序和同步的唯一时间依据（微秒）
/// - frame_index: 日志、测试和丢帧追踪用的帧序号
struct VideoFrame {
    cv::Mat frame_bgr;
    core::MediaTimestamp pts;
    std::int64_t frame_index = 0;
};

}  // namespace video
}  // namespace digital_human
