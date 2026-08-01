#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "core/frame_scheduler.h"
#include "sync/audio_clock.h"
#include "sync/audio_video_synchronous.h"
#include "video/video_frame.h"

namespace digital_human {
namespace sync {
namespace {

class FakeAudioClock final : public AudioClock {
public:
    explicit FakeAudioClock(std::vector<AudioClockResult> results)
        : results_(std::move(results)) {}

    AudioClockResult CurrentTime() const override {
        if (results_.empty()) {
            AudioClockResult result;
            result.success = false;
            result.status = AudioClockStatus::kClockUnavailable;
            result.error_message = "fake clock has no observations";
            return result;
        }
        const std::size_t selected = std::min(index_, results_.size() - 1);
        ++index_;
        return results_[selected];
    }

private:
    std::vector<AudioClockResult> results_;
    mutable std::size_t index_ = 0;
};

AudioClockResult ClockAt(std::int64_t microseconds) {
    AudioClockResult result;
    result.success = true;
    result.status = AudioClockStatus::kOk;
    result.snapshot.timestamp.microseconds = microseconds;
    result.snapshot.state = AudioClockState::kRunning;
    result.snapshot.reference_frame_index = static_cast<std::uint64_t>(
        microseconds >= 0 ? microseconds : 0);
    return result;
}

AudioClockResult ClockFailure() {
    AudioClockResult result;
    result.success = false;
    result.status = AudioClockStatus::kClockUnavailable;
    result.error_message = "clock anchor unavailable";
    result.snapshot.state = AudioClockState::kRunning;
    return result;
}

core::FrameSchedulerConfig SchedulerConfig() {
    core::FrameSchedulerConfig config;
    config.max_queue_size = 8;
    config.min_buffered_frames = 1;
    config.sync_tolerance_us = 10000;
    config.overflow_policy = core::OverflowPolicy::kDropOldest;
    return config;
}

video::VideoFrame Frame(std::int64_t pts_us, std::int64_t index) {
    video::VideoFrame frame;
    frame.frame_bgr = cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3)).clone();
    frame.pts.microseconds = pts_us;
    frame.frame_index = index;
    return frame;
}

TEST(AudioVideoSynchronousTest, RejectsNullClock) {
    EXPECT_THROW(
        AudioVideoSynchronous(nullptr, SchedulerConfig()),
        std::invalid_argument);
}

TEST(AudioVideoSynchronousTest, DelegatesPushAndDeliveryToFrameScheduler) {
    auto clock = std::make_shared<FakeAudioClock>(
        std::vector<AudioClockResult>{ClockAt(100000)});
    AudioVideoSynchronous synchronizer(clock, SchedulerConfig());

    EXPECT_EQ(synchronizer.PushVideoFrame(Frame(100000, 7)),
              core::PushResult::kAccepted);
    const AudioVideoSyncResult result = synchronizer.ScheduleNext();
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.status, AudioVideoSyncStatus::kOk);
    EXPECT_EQ(result.clock.timestamp.microseconds, 100000);
    EXPECT_EQ(result.schedule.action, core::ScheduleAction::kDeliver);
    ASSERT_TRUE(result.schedule.selected_frame.has_value());
    EXPECT_EQ(result.schedule.selected_frame->frame_index, 7);
}

TEST(AudioVideoSynchronousTest, ClockFailureDoesNotConsumeQueuedFrame) {
    auto clock = std::make_shared<FakeAudioClock>(
        std::vector<AudioClockResult>{ClockFailure(), ClockAt(100000)});
    AudioVideoSynchronous synchronizer(clock, SchedulerConfig());
    ASSERT_EQ(synchronizer.PushVideoFrame(Frame(100000, 3)),
              core::PushResult::kAccepted);

    const AudioVideoSyncResult failed = synchronizer.ScheduleNext();
    EXPECT_FALSE(failed.success);
    EXPECT_EQ(failed.status, AudioVideoSyncStatus::kClockUnavailable);
    EXPECT_FALSE(failed.error_message.empty());

    const AudioVideoSyncResult delivered = synchronizer.ScheduleNext();
    ASSERT_TRUE(delivered.success);
    ASSERT_TRUE(delivered.schedule.selected_frame.has_value());
    EXPECT_EQ(delivered.schedule.selected_frame->frame_index, 3);
}

TEST(AudioVideoSynchronousTest, SurfacesRegressiveReferenceRejection) {
    auto clock = std::make_shared<FakeAudioClock>(
        std::vector<AudioClockResult>{ClockAt(200000), ClockAt(100000)});
    AudioVideoSynchronous synchronizer(clock, SchedulerConfig());

    const AudioVideoSyncResult first = synchronizer.ScheduleNext();
    EXPECT_TRUE(first.success);
    EXPECT_EQ(first.schedule.action,
              core::ScheduleAction::kBufferingNoFrame);

    const AudioVideoSyncResult second = synchronizer.ScheduleNext();
    EXPECT_FALSE(second.success);
    EXPECT_EQ(second.status,
              AudioVideoSyncStatus::kInvalidReferenceTime);
    EXPECT_EQ(second.schedule.action,
              core::ScheduleAction::kRejectedInvalidReferenceTime);
}

TEST(AudioVideoSynchronousTest, ResetClearsSchedulerReferenceHistory) {
    auto clock = std::make_shared<FakeAudioClock>(
        std::vector<AudioClockResult>{ClockAt(200000), ClockAt(100000)});
    AudioVideoSynchronous synchronizer(clock, SchedulerConfig());
    EXPECT_TRUE(synchronizer.ScheduleNext().success);
    synchronizer.ResetScheduler();
    EXPECT_TRUE(synchronizer.ScheduleNext().success);
}

TEST(AudioVideoSynchronousTest, InvalidVideoFrameUsesExistingPushContract) {
    auto clock = std::make_shared<FakeAudioClock>(
        std::vector<AudioClockResult>{ClockAt(0)});
    AudioVideoSynchronous synchronizer(clock, SchedulerConfig());
    video::VideoFrame invalid;
    EXPECT_EQ(synchronizer.PushVideoFrame(invalid),
              core::PushResult::kRejectedInvalidFrame);
}

TEST(AudioVideoSynchronousStatusTest, EveryKnownStatusHasText) {
    const std::vector<AudioVideoSyncStatus> statuses = {
        AudioVideoSyncStatus::kOk,
        AudioVideoSyncStatus::kClockUnavailable,
        AudioVideoSyncStatus::kInvalidReferenceTime,
        AudioVideoSyncStatus::kInternalError};
    for (AudioVideoSyncStatus status : statuses) {
        EXPECT_FALSE(AudioVideoSynchronous::StatusToString(status).empty());
    }

    const std::vector<AudioClockStatus> clock_statuses = {
        AudioClockStatus::kOk,
        AudioClockStatus::kNotReady,
        AudioClockStatus::kClockUnavailable,
        AudioClockStatus::kBackendError,
        AudioClockStatus::kNonMonotonic,
        AudioClockStatus::kInternalError};
    for (AudioClockStatus status : clock_statuses) {
        EXPECT_FALSE(AudioClockStatusToString(status).empty());
    }
}

}  // namespace
}  // namespace sync
}  // namespace digital_human
