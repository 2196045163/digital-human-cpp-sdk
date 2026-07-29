/// @brief 帧调度窄集成测试。
///
/// 验证链路: TimestampManager → FaceBlender → FrameScheduler
/// 不加载真实模型，使用合成图像。
/// 核心库不打印。

#include "core/face_blender.h"
#include "core/frame_scheduler.h"
#include "core/timestamp_manager.h"

#include <cstdint>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace digital_human {
namespace core {
namespace {

// ===================================================================
// 测试工具
// ===================================================================

/// @brief 创建合成原图（CV_8UC3, 任意尺寸）。
cv::Mat MakeBaseImage(int width, int height) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
}

/// @brief 创建合成 96x96 生成嘴部图。
cv::Mat MakeGeneratedFace96() {
    return cv::Mat(96, 96, CV_8UC3, cv::Scalar(50, 100, 150));
}

/// @brief 创建合成 96x96 alpha mask（CV_32FC1, 全 0.5）。
cv::Mat MakeMask96() {
    return cv::Mat(96, 96, CV_32FC1, cv::Scalar(0.5f));
}

/// @brief 创建简单的 2x3 逆仿射矩阵（CV_64F）。
/// 映射: 96x96 左上角 → 原图 (0,0) 附近。
cv::Mat MakeInverseTransform() {
    // clang-format off
    double data[2 * 3] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };
    // clang-format on
    cv::Mat m(2, 3, CV_64F);
    std::memcpy(m.data, data, sizeof(data));
    return m;
}

/// @brief 帧调度器窄集成配置。
FrameSchedulerConfig MakeIntegrationConfig() {
    FrameSchedulerConfig cfg;
    cfg.max_queue_size = 8;
    cfg.min_buffered_frames = 1;
    cfg.sync_tolerance_us = 20000;
    cfg.overflow_policy = OverflowPolicy::kDropOldest;
    return cfg;
}

}  // namespace

// ===================================================================
// 窄集成测试
// ===================================================================

TEST(FrameSchedulerIntegrationTest, TimestampFaceBlenderToScheduler) {
    // 1. 用 TimestampManager 生成 PTS（不手填毫秒值）
    FrameRate fps{25, 1};  // 25fps
    auto pts_result = TimestampManager::FromVideoFrameIndex(10, fps);
    ASSERT_TRUE(pts_result.success);
    MediaTimestamp pts = pts_result.value;

    // 2. 合成 FaceBlender 输入并融合
    FaceBlender blender;
    cv::Mat base = MakeBaseImage(192, 192);
    cv::Mat gen = MakeGeneratedFace96();
    cv::Mat mask = MakeMask96();
    cv::Mat inv = MakeInverseTransform();

    FaceBlendResult blend = blender.BlendMouthToOriginal(base, gen, mask, inv);
    ASSERT_TRUE(blend.success);
    ASSERT_FALSE(blend.final_bgr.empty());

    // 3. 验证融合图是原图尺寸 CV_8UC3
    EXPECT_EQ(blend.final_bgr.cols, 192);
    EXPECT_EQ(blend.final_bgr.rows, 192);
    EXPECT_EQ(blend.final_bgr.type(), CV_8UC3);

    // 4. 构建 VideoFrame 并推入调度器
    video::VideoFrame vf;
    vf.frame_bgr = blend.final_bgr;
    vf.pts = pts;
    vf.frame_index = 10;

    FrameScheduler scheduler(MakeIntegrationConfig());
    EXPECT_EQ(scheduler.PushFrame(vf), PushResult::kAccepted);

    // 5. 调度交付
    auto result = scheduler.Schedule(pts);  // ref 匹配 PTS
    EXPECT_EQ(result.action, ScheduleAction::kDeliver);
    ASSERT_TRUE(result.selected_frame.has_value());

    // 6. 验收: 交付帧是原图尺寸 CV_8UC3, PTS 来自 TimestampManager
    const auto& delivered = result.selected_frame.value();
    EXPECT_EQ(delivered.frame_bgr.cols, 192);
    EXPECT_EQ(delivered.frame_bgr.rows, 192);
    EXPECT_EQ(delivered.frame_bgr.type(), CV_8UC3);
    EXPECT_EQ(delivered.pts.microseconds, pts.microseconds);
    EXPECT_EQ(delivered.frame_index, 10);
    // timing_diff_us 应存在（交付时）
    EXPECT_TRUE(result.timing_diff_us.has_value());
}

TEST(FrameSchedulerIntegrationTest, MultipleFramesThroughPipeline) {
    // 完整链路: 3 帧合成 → FaceBlender → FrameScheduler → 验证 PTS 递增
    FrameScheduler scheduler(MakeIntegrationConfig());
    FrameRate fps{25, 1};

    for (int i = 0; i < 3; i++) {
        auto pts_result = TimestampManager::FromVideoFrameIndex(i, fps);
        ASSERT_TRUE(pts_result.success);

        FaceBlender blender;
        auto blend = blender.BlendMouthToOriginal(
            MakeBaseImage(128, 128),
            MakeGeneratedFace96(),
            MakeMask96(),
            MakeInverseTransform());
        ASSERT_TRUE(blend.success);

        video::VideoFrame vf;
        vf.frame_bgr = blend.final_bgr;
        vf.pts = pts_result.value;
        vf.frame_index = i;

        EXPECT_EQ(scheduler.PushFrame(vf), PushResult::kAccepted);
    }

    // 按 PTS 顺序调度，验证交付
    for (int i = 0; i < 3; i++) {
        auto pts_result = TimestampManager::FromVideoFrameIndex(i, fps);
        auto result = scheduler.Schedule(pts_result.value);

        EXPECT_EQ(result.action, ScheduleAction::kDeliver);
        ASSERT_TRUE(result.selected_frame.has_value());
        EXPECT_EQ(result.selected_frame->frame_index, i);
        EXPECT_EQ(result.selected_frame->pts.microseconds,
                  pts_result.value.microseconds);
        // 交付图是原图尺寸
        EXPECT_EQ(result.selected_frame->frame_bgr.cols, 128);
        EXPECT_EQ(result.selected_frame->frame_bgr.rows, 128);
    }
}

}  // namespace core
}  // namespace digital_human
