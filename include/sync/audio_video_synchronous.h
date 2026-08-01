#pragma once

#include <memory>
#include <string>

#include "core/frame_scheduler.h"
#include "sync/audio_clock.h"
#include "video/video_frame.h"

namespace digital_human {
namespace sync {

/// @brief Coordinator status; frame actions remain FrameScheduler's contract.
enum class AudioVideoSyncStatus {
    kOk,
    kClockUnavailable,
    kInvalidReferenceTime,
    kInternalError
};

/// @brief One audio-clock-driven scheduling decision.
struct AudioVideoSyncResult {
    bool success = false;
    AudioVideoSyncStatus status = AudioVideoSyncStatus::kInternalError;
    std::string error_message;
    AudioClockSnapshot clock;
    core::ScheduleResult schedule;
};

/// @brief Thin coordinator between an injectable audio clock and FrameScheduler.
class AudioVideoSynchronous {
public:
    AudioVideoSynchronous(
        std::shared_ptr<AudioClock> audio_clock,
        const core::FrameSchedulerConfig& scheduler_config);
    ~AudioVideoSynchronous();

    AudioVideoSynchronous(const AudioVideoSynchronous&) = delete;
    AudioVideoSynchronous& operator=(const AudioVideoSynchronous&) = delete;
    AudioVideoSynchronous(AudioVideoSynchronous&&) = delete;
    AudioVideoSynchronous& operator=(AudioVideoSynchronous&&) = delete;

    /// @brief Multi-producer frame insertion; semantics come from FrameScheduler.
    core::PushResult PushVideoFrame(const video::VideoFrame& frame);

    /// @brief Read the audio clock and make exactly one scheduler decision.
    AudioVideoSyncResult ScheduleNext();

    /// @brief Reset only video scheduling state; playback lifecycle is separate.
    void ResetScheduler();

    static std::string StatusToString(AudioVideoSyncStatus status);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sync
}  // namespace digital_human
