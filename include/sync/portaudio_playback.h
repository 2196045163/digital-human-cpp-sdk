#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "audio/audio_loader.h"
#include "sync/audio_clock.h"

namespace digital_human {
namespace sync {

/// @brief PortAudio lifecycle and validation status.
enum class AudioPlaybackStatus {
    kOk,
    kInvalidConfiguration,
    kEmptyAudio,
    kInvalidSampleRate,
    kUnsupportedChannels,
    kUnsupportedSampleFormat,
    kInvalidPcmData,
    kInvalidState,
    kInitializeFailed,
    kNoOutputDevice,
    kInvalidOutputDevice,
    kUnsupportedOutputFormat,
    kOpenStreamFailed,
    kFinishedCallbackFailed,
    kStartStreamFailed,
    kStopStreamFailed,
    kAbortStreamFailed,
    kCloseStreamFailed,
    kTerminateFailed,
    kInternalError
};

/// @brief Fixed playback configuration.
struct PortAudioPlaybackConfig {
    /// -1 selects PortAudio's default output device.
    int output_device_index = -1;

    /// 0 maps to paFramesPerBufferUnspecified.
    std::size_t frames_per_buffer = 0;
};

/// @brief Reviewable callback and stream diagnostics.
struct AudioPlaybackStats {
    AudioClockState state = AudioClockState::kIdle;
    std::uint64_t total_audio_frames = 0;
    std::uint64_t submitted_audio_frames = 0;
    std::uint64_t estimated_played_frames = 0;
    std::uint64_t callback_count = 0;
    std::uint64_t output_underflow_count = 0;
    int sample_rate = 0;
    int channels = 0;
    int output_device_index = -1;
    double output_latency_seconds = 0.0;
};

/// @brief Explicit result for prepare/start/stop/abort/reset operations.
struct AudioPlaybackResult {
    bool success = false;
    AudioPlaybackStatus status = AudioPlaybackStatus::kInternalError;
    std::string error_message;
    AudioClockState state = AudioClockState::kIdle;
    AudioPlaybackStats stats;
};

/// @brief Preloaded float-PCM playback and PortAudio-backed media clock.
///
/// PortAudio types are hidden behind PImpl. Lifecycle calls belong to one control
/// thread. CurrentTime()/GetStats() may be polled by the single video consumer.
class PortAudioPlayback final : public AudioClock {
public:
    explicit PortAudioPlayback(
        const PortAudioPlaybackConfig& config = PortAudioPlaybackConfig());
    ~PortAudioPlayback() override;

    PortAudioPlayback(const PortAudioPlayback&) = delete;
    PortAudioPlayback& operator=(const PortAudioPlayback&) = delete;
    PortAudioPlayback(PortAudioPlayback&&) = delete;
    PortAudioPlayback& operator=(PortAudioPlayback&&) = delete;

    /// @brief Validate/copy immutable PCM and open a stopped PortAudio stream.
    AudioPlaybackResult Prepare(const audio::AudioData& audio);

    /// @brief Start asynchronous playback from frame zero.
    AudioPlaybackResult Start();

    /// @brief Gracefully stop, allowing PortAudio to drain pending output.
    AudioPlaybackResult Stop();

    /// @brief Abort playback as soon as possible.
    AudioPlaybackResult Abort();

    /// @brief Close/terminate all PortAudio resources and return to Idle.
    AudioPlaybackResult Reset();

    AudioClockState GetState() const;
    AudioPlaybackStats GetStats() const;
    AudioClockResult CurrentTime() const override;

    static std::string StatusToString(AudioPlaybackStatus status);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sync
}  // namespace digital_human
