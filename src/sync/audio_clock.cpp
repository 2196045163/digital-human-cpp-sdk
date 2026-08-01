#include "sync/audio_clock.h"

namespace digital_human {
namespace sync {

std::string AudioClockStatusToString(AudioClockStatus status) {
    switch (status) {
        case AudioClockStatus::kOk:
            return "ok";
        case AudioClockStatus::kNotReady:
            return "audio clock not ready";
        case AudioClockStatus::kClockUnavailable:
            return "audio clock unavailable";
        case AudioClockStatus::kBackendError:
            return "audio clock backend error";
        case AudioClockStatus::kNonMonotonic:
            return "non-monotonic audio clock";
        case AudioClockStatus::kInternalError:
            return "internal audio clock error";
        default:
            return "unrecognized audio clock status";
    }
}

std::string AudioClockStateToString(AudioClockState state) {
    switch (state) {
        case AudioClockState::kIdle:
            return "idle";
        case AudioClockState::kPrepared:
            return "prepared";
        case AudioClockState::kRunning:
            return "running";
        case AudioClockState::kCompleted:
            return "completed";
        case AudioClockState::kStopped:
            return "stopped";
        case AudioClockState::kError:
            return "error";
        default:
            return "unrecognized audio clock state";
    }
}

}  // namespace sync
}  // namespace digital_human
