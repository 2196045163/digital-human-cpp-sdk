#pragma once

#include <cstdint>
#include <string>

#include "core/timestamp_manager.h"

namespace digital_human {
namespace sync {

/// @brief Generic lifecycle states for an audio-backed reference clock.
enum class AudioClockState {
    kIdle,
    kPrepared,
    kRunning,
    kCompleted,
    kStopped,
    kError
};

/// @brief Result status for reading the current audio reference time.
enum class AudioClockStatus {
    kOk,
    kNotReady,
    kClockUnavailable,
    kBackendError,
    kNonMonotonic,
    kInternalError
};

/// @brief One immutable observation of the audio clock.
struct AudioClockSnapshot {
    core::MediaTimestamp timestamp;
    AudioClockState state = AudioClockState::kIdle;
    std::uint64_t reference_frame_index = 0;
};

/// @brief Explicit success/error contract for an audio clock observation.
struct AudioClockResult {
    bool success = false;
    AudioClockStatus status = AudioClockStatus::kInternalError;
    std::string error_message;
    AudioClockSnapshot snapshot;
};

/// @brief Injectable audio-master clock used by the synchronization coordinator.
class AudioClock {
public:
    virtual ~AudioClock() = default;

    /// @brief Return the current monotonic media time without sleeping or printing.
    virtual AudioClockResult CurrentTime() const = 0;
};

/// @brief Stable text for diagnostics and tests; core code does not print it.
std::string AudioClockStatusToString(AudioClockStatus status);

/// @brief Stable text for diagnostics and tests; core code does not print it.
std::string AudioClockStateToString(AudioClockState state);

}  // namespace sync
}  // namespace digital_human
