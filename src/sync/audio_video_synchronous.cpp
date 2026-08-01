#include "sync/audio_video_synchronous.h"

#include <stdexcept>
#include <utility>

namespace digital_human {
namespace sync {

struct AudioVideoSynchronous::Impl {
    Impl(
        std::shared_ptr<AudioClock> clock,
        const core::FrameSchedulerConfig& scheduler_config)
        : audio_clock(std::move(clock)), scheduler(scheduler_config) {}

    std::shared_ptr<AudioClock> audio_clock;
    core::FrameScheduler scheduler;
};

AudioVideoSynchronous::AudioVideoSynchronous(
    std::shared_ptr<AudioClock> audio_clock,
    const core::FrameSchedulerConfig& scheduler_config) {
    if (!audio_clock) {
        throw std::invalid_argument("Audio clock must not be null");
    }
    impl_ = std::make_unique<Impl>(
        std::move(audio_clock), scheduler_config);
}

AudioVideoSynchronous::~AudioVideoSynchronous() = default;

core::PushResult AudioVideoSynchronous::PushVideoFrame(
    const video::VideoFrame& frame) {
    return impl_->scheduler.PushFrame(frame);
}

AudioVideoSyncResult AudioVideoSynchronous::ScheduleNext() {
    AudioVideoSyncResult result;
    const AudioClockResult clock_result = impl_->audio_clock->CurrentTime();
    result.clock = clock_result.snapshot;
    if (!clock_result.success) {
        result.success = false;
        result.status = AudioVideoSyncStatus::kClockUnavailable;
        result.error_message = clock_result.error_message.empty()
            ? AudioClockStatusToString(clock_result.status)
            : clock_result.error_message;
        return result;
    }

    result.schedule = impl_->scheduler.Schedule(
        clock_result.snapshot.timestamp);
    if (result.schedule.action
        == core::ScheduleAction::kRejectedInvalidReferenceTime) {
        result.success = false;
        result.status = AudioVideoSyncStatus::kInvalidReferenceTime;
        result.error_message =
            "FrameScheduler rejected the audio reference timestamp";
        return result;
    }

    result.success = true;
    result.status = AudioVideoSyncStatus::kOk;
    result.error_message.clear();
    return result;
}

void AudioVideoSynchronous::ResetScheduler() {
    impl_->scheduler.Reset();
}

std::string AudioVideoSynchronous::StatusToString(
    AudioVideoSyncStatus status) {
    switch (status) {
        case AudioVideoSyncStatus::kOk:
            return "ok";
        case AudioVideoSyncStatus::kClockUnavailable:
            return "audio clock unavailable";
        case AudioVideoSyncStatus::kInvalidReferenceTime:
            return "invalid audio reference time";
        case AudioVideoSyncStatus::kInternalError:
            return "internal audio-video synchronization error";
        default:
            return "unrecognized audio-video synchronization status";
    }
}

}  // namespace sync
}  // namespace digital_human
