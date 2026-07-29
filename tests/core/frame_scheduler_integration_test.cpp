/// @brief 帧调度窄集成测试。
///
/// 验证链路: TimestampManager → FaceBlender → FrameScheduler。
/// 合成测试证明契约正确；真实数据测试证明非合成图像能正常走通。
/// 核心库不打印。

#include "core/face_blender.h"
#include "core/frame_scheduler.h"
#include "core/timestamp_manager.h"

#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

// ===================================================================
// 真实数据测试——防止自欺欺人
// 素材: testdata/golden/face.jpg (不可控——真实图片，不是 64x64 色块)
// 证明: 真实图像经 PushFrame/Schedule 进出后像素、尺寸、PTS 全部保持
// ===================================================================

TEST(FrameSchedulerRealDataTest, RealImageRoundTrip) {
    // 1. 加载真实人脸图
    cv::Mat real_face = cv::imread("testdata/golden/face.jpg");
    ASSERT_FALSE(real_face.empty()) << "FATAL: testdata/golden/face.jpg 无法加载";
    ASSERT_EQ(real_face.type(), CV_8UC3) << "FATAL: 素材格式不是 CV_8UC3";

    std::cout << "[REAL-DATA] 素材: testdata/golden/face.jpg "
              << real_face.cols << "x" << real_face.rows
              << " CV_8UC3\n";

    // 2. 用 TimestampManager 生成 PTS（不是手填的常量）
    FrameRate fps{25, 1};  // 25fps，frame 0
    auto pts_result = TimestampManager::FromVideoFrameIndex(0, fps);
    ASSERT_TRUE(pts_result.success);
    std::cout << "[REAL-DATA] PTS: " << pts_result.value.microseconds
              << "us (TimestampManager, 25fps frame 0)\n";

    // 3. 构建 VideoFrame
    video::VideoFrame vf;
    vf.frame_bgr = real_face;
    vf.pts = pts_result.value;
    vf.frame_index = 0;

    // 4. PushFrame
    FrameScheduler scheduler(MakeIntegrationConfig());
    PushResult push_r = scheduler.PushFrame(vf);
    EXPECT_EQ(push_r, PushResult::kAccepted);
    std::cout << "[REAL-DATA] PushFrame → " << (push_r == PushResult::kAccepted ? "Accepted" : "REJECTED") << "\n";

    // 5. Schedule
    auto result = scheduler.Schedule(pts_result.value);
    ASSERT_EQ(result.action, ScheduleAction::kDeliver)
        << "ref 匹配 PTS 时必须交付";
    ASSERT_TRUE(result.selected_frame.has_value());

    std::cout << "[REAL-DATA] Schedule → action=Deliver"
              << " diff=" << result.timing_diff_us.value()
              << "us dropped=" << result.dropped_this_call
              << " qsize=" << result.queue_size_after
              << " state=" << (result.state_after == SchedulerState::kRunning ? "Running" : "Buffering")
              << "\n";

    // 6. 验收交付帧
    const auto& out = result.selected_frame.value();
    EXPECT_EQ(out.frame_bgr.cols, real_face.cols);
    EXPECT_EQ(out.frame_bgr.rows, real_face.rows);
    EXPECT_EQ(out.frame_bgr.type(), CV_8UC3);
    EXPECT_EQ(out.pts.microseconds, pts_result.value.microseconds);
    EXPECT_EQ(out.frame_index, 0);
    EXPECT_TRUE(result.timing_diff_us.has_value());

    // 逐像素对比——最关键的防自欺断言
    cv::Mat diff;
    cv::absdiff(real_face, out.frame_bgr, diff);
    int non_zero = cv::countNonZero(cv::Mat(diff.reshape(1, 0)));

    auto p0_in = real_face.at<cv::Vec3b>(0, 0);
    auto p0_out = out.frame_bgr.at<cv::Vec3b>(0, 0);

    std::cout << "[REAL-DATA] 输入像素(0,0): B=" << (int)p0_in[0]
              << " G=" << (int)p0_in[1] << " R=" << (int)p0_in[2] << "\n";
    std::cout << "[REAL-DATA] 输出像素(0,0): B=" << (int)p0_out[0]
              << " G=" << (int)p0_out[1] << " R=" << (int)p0_out[2] << "\n";
    std::cout << "[REAL-DATA] 全图差异像素数: " << non_zero
              << " / " << (real_face.cols * real_face.rows * 3) << "\n";

    EXPECT_EQ(non_zero, 0) << "真实图像经过调度器后像素完全不能变";
}

}  // namespace core
}  // namespace digital_human
