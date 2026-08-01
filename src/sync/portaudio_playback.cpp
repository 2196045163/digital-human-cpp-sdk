#include "sync/portaudio_playback.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <portaudio.h>

#include "core/timestamp_manager.h"
#include "portaudio_callback_state.h"

namespace digital_human {
namespace sync {
namespace {

constexpr std::int64_t kMicrosecondsPerSecond = 1000000;

std::string PortAudioError(const std::string& operation, PaError error) {
    return operation + " failed: " + Pa_GetErrorText(error)
        + " (" + std::to_string(error) + ")";
}

AudioClockResult MakeClockError(
    AudioClockStatus status,
    AudioClockState state,
    const std::string& message) {
    AudioClockResult result;
    result.success = false;
    result.status = status;
    result.error_message = message;
    result.snapshot.state = state;
    return result;
}

}  // namespace

struct PortAudioPlayback::Impl {
    explicit Impl(const PortAudioPlaybackConfig& playback_config)
        : config(playback_config) {}

    PortAudioPlaybackConfig config;
    std::vector<float> pcm;
    int sample_rate = 0;
    int channels = 0;
    int output_device_index = -1;
    double output_latency_seconds = 0.0;
    std::uint64_t total_frames = 0;

    bool portaudio_initialized = false;
    PaStream* stream = nullptr;
    internal::PortAudioCallbackState callback_state;
    mutable std::atomic<AudioClockState> state{AudioClockState::kIdle};
    mutable std::atomic<std::uint64_t> last_reported_frame{0};

    AudioClockState RefreshState() const {
        AudioClockState current = state.load(std::memory_order_acquire);
        if (callback_state.callback_error.load(std::memory_order_acquire)) {
            state.store(AudioClockState::kError, std::memory_order_release);
            return AudioClockState::kError;
        }
        if (current == AudioClockState::kRunning
            && callback_state.stream_finished.load(std::memory_order_acquire)) {
            state.store(AudioClockState::kCompleted, std::memory_order_release);
            return AudioClockState::kCompleted;
        }
        if (current == AudioClockState::kRunning && stream != nullptr
            && callback_state.source_exhausted.load(std::memory_order_acquire)) {
            const PaError active = Pa_IsStreamActive(stream);
            if (active == 0) {
                state.store(AudioClockState::kCompleted, std::memory_order_release);
                return AudioClockState::kCompleted;
            }
            if (active < 0) {
                state.store(AudioClockState::kError, std::memory_order_release);
                return AudioClockState::kError;
            }
        }
        return current;
    }

    AudioPlaybackStats BuildStats() const {
        AudioPlaybackStats stats;
        stats.state = RefreshState();
        stats.total_audio_frames = total_frames;
        stats.submitted_audio_frames =
            callback_state.next_frame.load(std::memory_order_acquire);
        stats.estimated_played_frames =
            last_reported_frame.load(std::memory_order_acquire);
        stats.callback_count =
            callback_state.callback_count.load(std::memory_order_relaxed);
        stats.output_underflow_count =
            callback_state.output_underflow_count.load(
                std::memory_order_relaxed);
        stats.sample_rate = sample_rate;
        stats.channels = channels;
        stats.output_device_index = output_device_index;
        stats.output_latency_seconds = output_latency_seconds;
        return stats;
    }

    AudioPlaybackResult Result(
        bool success,
        AudioPlaybackStatus status_value,
        const std::string& message) const {
        AudioPlaybackResult result;
        result.success = success;
        result.status = status_value;
        result.error_message = message;
        result.state = RefreshState();
        result.stats = BuildStats();
        return result;
    }

    /// Best-effort cleanup used by Reset and failure paths. Returns first error.
    std::pair<PaError, std::string> Cleanup() {
        PaError first_error = paNoError;
        std::string first_operation;

        if (stream != nullptr) {
            const PaError active = Pa_IsStreamActive(stream);
            if (active == 1) {
                const PaError abort_error = Pa_AbortStream(stream);
                if (abort_error != paNoError && first_error == paNoError) {
                    first_error = abort_error;
                    first_operation = "Pa_AbortStream";
                }
            } else if (active < 0 && first_error == paNoError) {
                first_error = active;
                first_operation = "Pa_IsStreamActive";
            }

            const PaError close_error = Pa_CloseStream(stream);
            if (close_error != paNoError && first_error == paNoError) {
                first_error = close_error;
                first_operation = "Pa_CloseStream";
            }
            stream = nullptr;
        }

        if (portaudio_initialized) {
            const PaError terminate_error = Pa_Terminate();
            if (terminate_error != paNoError && first_error == paNoError) {
                first_error = terminate_error;
                first_operation = "Pa_Terminate";
            }
            portaudio_initialized = false;
        }
        return {first_error, first_operation};
    }

    void ClearAudioState() {
        pcm.clear();
        sample_rate = 0;
        channels = 0;
        output_device_index = -1;
        output_latency_seconds = 0.0;
        total_frames = 0;
        callback_state.Reset(nullptr, 0, 0);
        last_reported_frame.store(0, std::memory_order_relaxed);
    }
};

PortAudioPlayback::PortAudioPlayback(
    const PortAudioPlaybackConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

PortAudioPlayback::~PortAudioPlayback() {
    impl_->Cleanup();
}

AudioPlaybackResult PortAudioPlayback::Prepare(
    const audio::AudioData& audio) {
    const AudioClockState state = impl_->RefreshState();
    if (state != AudioClockState::kIdle) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidState,
            "Prepare requires Idle state; call Reset before reuse");
    }
    if (impl_->config.output_device_index < -1
        || impl_->config.frames_per_buffer
            > static_cast<std::size_t>(
                std::numeric_limits<unsigned long>::max())) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidConfiguration,
            "Output device must be -1 or non-negative and frames_per_buffer "
            "must fit unsigned long");
    }
    if (audio.pcm.empty()) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kEmptyAudio,
            "Audio PCM must not be empty");
    }
    if (audio.sample_rate <= 0) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidSampleRate,
            "Audio sample rate must be greater than zero");
    }
    if (audio.channels != 1) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kUnsupportedChannels,
            "PortAudio playback currently supports mono AudioData only");
    }
    if (audio.format != audio::AudioSampleFormat::kFloat32) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kUnsupportedSampleFormat,
            "PortAudio playback requires float32 AudioData");
    }
    if (audio.pcm.size() % static_cast<std::size_t>(audio.channels) != 0) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidPcmData,
            "PCM sample count must be divisible by the channel count");
    }
    for (float sample : audio.pcm) {
        if (!std::isfinite(sample)) {
            return impl_->Result(
                false,
                AudioPlaybackStatus::kInvalidPcmData,
                "PCM must contain only finite float samples");
        }
    }

    impl_->pcm = audio.pcm;
    impl_->sample_rate = audio.sample_rate;
    impl_->channels = audio.channels;
    impl_->total_frames = static_cast<std::uint64_t>(
        impl_->pcm.size() / static_cast<std::size_t>(impl_->channels));
    impl_->callback_state.Reset(
        impl_->pcm.data(), impl_->total_frames, impl_->channels);
    impl_->last_reported_frame.store(0, std::memory_order_relaxed);

    PaError error = Pa_Initialize();
    if (error != paNoError) {
        impl_->ClearAudioState();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInitializeFailed,
            PortAudioError("Pa_Initialize", error));
    }
    impl_->portaudio_initialized = true;

    const PaDeviceIndex device_count = Pa_GetDeviceCount();
    if (device_count < 0) {
        const std::string message = PortAudioError(
            "Pa_GetDeviceCount", static_cast<PaError>(device_count));
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false, AudioPlaybackStatus::kInvalidOutputDevice, message);
    }

    PaDeviceIndex device = impl_->config.output_device_index;
    if (device < 0) {
        device = Pa_GetDefaultOutputDevice();
    }
    if (device == paNoDevice) {
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kNoOutputDevice,
            "PortAudio did not report a default output device");
    }
    if (device < 0 || device >= device_count) {
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidOutputDevice,
            "Configured PortAudio output device index is out of range");
    }

    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(device);
    if (device_info == nullptr
        || device_info->maxOutputChannels < impl_->channels) {
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidOutputDevice,
            "PortAudio output device does not support the requested channels");
    }

    PaStreamParameters output_parameters{};
    output_parameters.device = device;
    output_parameters.channelCount = impl_->channels;
    output_parameters.sampleFormat = paFloat32;
    output_parameters.suggestedLatency = device_info->defaultLowOutputLatency;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    error = Pa_IsFormatSupported(
        nullptr, &output_parameters, static_cast<double>(impl_->sample_rate));
    if (error != paFormatIsSupported) {
        const std::string message = PortAudioError(
            "Pa_IsFormatSupported", error);
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kUnsupportedOutputFormat,
            message);
    }

    const unsigned long frames_per_buffer =
        impl_->config.frames_per_buffer == 0
            ? paFramesPerBufferUnspecified
            : static_cast<unsigned long>(impl_->config.frames_per_buffer);
    error = Pa_OpenStream(
        &impl_->stream,
        nullptr,
        &output_parameters,
        static_cast<double>(impl_->sample_rate),
        frames_per_buffer,
        paClipOff,
        &internal::PortAudioCallbackState::Callback,
        &impl_->callback_state);
    if (error != paNoError) {
        const std::string message = PortAudioError("Pa_OpenStream", error);
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false, AudioPlaybackStatus::kOpenStreamFailed, message);
    }

    error = Pa_SetStreamFinishedCallback(
        impl_->stream, &internal::PortAudioCallbackState::Finished);
    if (error != paNoError) {
        const std::string message = PortAudioError(
            "Pa_SetStreamFinishedCallback", error);
        impl_->Cleanup();
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false, AudioPlaybackStatus::kFinishedCallbackFailed, message);
    }

    const PaStreamInfo* stream_info = Pa_GetStreamInfo(impl_->stream);
    impl_->output_device_index = static_cast<int>(device);
    impl_->output_latency_seconds = stream_info == nullptr
        ? output_parameters.suggestedLatency
        : stream_info->outputLatency;
    impl_->state.store(AudioClockState::kPrepared, std::memory_order_release);
    return impl_->Result(true, AudioPlaybackStatus::kOk, "");
}

AudioPlaybackResult PortAudioPlayback::Start() {
    if (impl_->RefreshState() != AudioClockState::kPrepared
        || impl_->stream == nullptr) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidState,
            "Start requires a successfully prepared stream");
    }

    impl_->callback_state.Reset(
        impl_->pcm.data(), impl_->total_frames, impl_->channels);
    impl_->last_reported_frame.store(0, std::memory_order_relaxed);
    const PaError error = Pa_StartStream(impl_->stream);
    if (error != paNoError) {
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kStartStreamFailed,
            PortAudioError("Pa_StartStream", error));
    }
    impl_->state.store(AudioClockState::kRunning, std::memory_order_release);
    return impl_->Result(true, AudioPlaybackStatus::kOk, "");
}

AudioPlaybackResult PortAudioPlayback::Stop() {
    const AudioClockState state = impl_->RefreshState();
    if (state == AudioClockState::kCompleted) {
        impl_->last_reported_frame.store(
            impl_->total_frames, std::memory_order_release);
        return impl_->Result(true, AudioPlaybackStatus::kOk, "");
    }
    if (state != AudioClockState::kRunning || impl_->stream == nullptr) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidState,
            "Stop requires a running or completed stream");
    }

    (void)CurrentTime();
    const PaError error = Pa_StopStream(impl_->stream);
    if (error != paNoError) {
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kStopStreamFailed,
            PortAudioError("Pa_StopStream", error));
    }
    const bool completed = impl_->callback_state.source_exhausted.load(
        std::memory_order_acquire);
    impl_->state.store(
        completed ? AudioClockState::kCompleted : AudioClockState::kStopped,
        std::memory_order_release);
    if (completed) {
        impl_->last_reported_frame.store(
            impl_->total_frames, std::memory_order_release);
    }
    return impl_->Result(true, AudioPlaybackStatus::kOk, "");
}

AudioPlaybackResult PortAudioPlayback::Abort() {
    const AudioClockState state = impl_->RefreshState();
    if (state == AudioClockState::kCompleted) {
        impl_->last_reported_frame.store(
            impl_->total_frames, std::memory_order_release);
        return impl_->Result(true, AudioPlaybackStatus::kOk, "");
    }
    if (state != AudioClockState::kRunning || impl_->stream == nullptr) {
        return impl_->Result(
            false,
            AudioPlaybackStatus::kInvalidState,
            "Abort requires a running or completed stream");
    }

    (void)CurrentTime();
    const PaError error = Pa_AbortStream(impl_->stream);
    if (error != paNoError) {
        impl_->state.store(AudioClockState::kError, std::memory_order_release);
        return impl_->Result(
            false,
            AudioPlaybackStatus::kAbortStreamFailed,
            PortAudioError("Pa_AbortStream", error));
    }
    impl_->state.store(AudioClockState::kStopped, std::memory_order_release);
    return impl_->Result(true, AudioPlaybackStatus::kOk, "");
}

AudioPlaybackResult PortAudioPlayback::Reset() {
    const auto cleanup = impl_->Cleanup();
    impl_->ClearAudioState();
    impl_->state.store(AudioClockState::kIdle, std::memory_order_release);
    if (cleanup.first != paNoError) {
        AudioPlaybackStatus status = AudioPlaybackStatus::kInternalError;
        if (cleanup.second == "Pa_CloseStream") {
            status = AudioPlaybackStatus::kCloseStreamFailed;
        } else if (cleanup.second == "Pa_Terminate") {
            status = AudioPlaybackStatus::kTerminateFailed;
        } else if (cleanup.second == "Pa_AbortStream") {
            status = AudioPlaybackStatus::kAbortStreamFailed;
        }
        return impl_->Result(
            false,
            status,
            PortAudioError(cleanup.second, cleanup.first));
    }
    return impl_->Result(true, AudioPlaybackStatus::kOk, "");
}

AudioClockState PortAudioPlayback::GetState() const {
    return impl_->RefreshState();
}

AudioPlaybackStats PortAudioPlayback::GetStats() const {
    return impl_->BuildStats();
}

AudioClockResult PortAudioPlayback::CurrentTime() const {
    const AudioClockState state = impl_->RefreshState();
    if (state == AudioClockState::kIdle
        || state == AudioClockState::kPrepared) {
        return MakeClockError(
            AudioClockStatus::kNotReady,
            state,
            "Audio stream has not started");
    }
    if (state == AudioClockState::kError) {
        return MakeClockError(
            AudioClockStatus::kBackendError,
            state,
            "PortAudio stream is in an error state");
    }

    std::uint64_t estimated_frame =
        impl_->last_reported_frame.load(std::memory_order_acquire);
    if (state == AudioClockState::kCompleted) {
        estimated_frame = impl_->total_frames;
    } else if (state == AudioClockState::kRunning) {
        if (impl_->stream == nullptr) {
            return MakeClockError(
                AudioClockStatus::kBackendError,
                state,
                "PortAudio stream handle is unavailable");
        }

        std::uint64_t anchor_frame = 0;
        std::int64_t anchor_dac_time_us = -1;
        if (!impl_->callback_state.ReadAnchor(
                &anchor_frame, &anchor_dac_time_us)) {
            return MakeClockError(
                AudioClockStatus::kClockUnavailable,
                state,
                "PortAudio has not published the first DAC-time anchor");
        }

        const PaTime stream_time = Pa_GetStreamTime(impl_->stream);
        if (!(stream_time >= 0.0)) {
            return MakeClockError(
                AudioClockStatus::kBackendError,
                state,
                "Pa_GetStreamTime returned an invalid value");
        }
        const double stream_time_us_double =
            stream_time * static_cast<double>(kMicrosecondsPerSecond);
        if (stream_time_us_double
            > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            return MakeClockError(
                AudioClockStatus::kBackendError,
                state,
                "PortAudio stream time overflowed int64 microseconds");
        }
        const std::int64_t stream_time_us = static_cast<std::int64_t>(
            stream_time_us_double + 0.5);

        const std::uint64_t submitted =
            impl_->callback_state.next_frame.load(std::memory_order_acquire);
        estimated_frame =
            internal::PortAudioCallbackState::EstimatePlayedFrame(
                anchor_frame,
                anchor_dac_time_us,
                stream_time_us,
                impl_->sample_rate,
                submitted,
                impl_->total_frames,
                estimated_frame);
    }

    std::uint64_t previous =
        impl_->last_reported_frame.load(std::memory_order_acquire);
    while (estimated_frame > previous
           && !impl_->last_reported_frame.compare_exchange_weak(
               previous,
               estimated_frame,
               std::memory_order_acq_rel,
               std::memory_order_acquire)) {
    }
    estimated_frame = std::max(estimated_frame, previous);

    if (estimated_frame
        > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return MakeClockError(
            AudioClockStatus::kInternalError,
            state,
            "Audio frame index overflowed int64");
    }
    const core::TimestampResult timestamp =
        core::TimestampManager::FromAudioSampleIndex(
            static_cast<std::int64_t>(estimated_frame), impl_->sample_rate);
    if (!timestamp.success) {
        return MakeClockError(
            AudioClockStatus::kInternalError,
            state,
            timestamp.error_message);
    }

    AudioClockResult result;
    result.success = true;
    result.status = AudioClockStatus::kOk;
    result.error_message.clear();
    result.snapshot.timestamp = timestamp.value;
    result.snapshot.state = state;
    result.snapshot.reference_frame_index = estimated_frame;
    return result;
}

std::string PortAudioPlayback::StatusToString(
    AudioPlaybackStatus status) {
    switch (status) {
        case AudioPlaybackStatus::kOk:
            return "ok";
        case AudioPlaybackStatus::kInvalidConfiguration:
            return "invalid playback configuration";
        case AudioPlaybackStatus::kEmptyAudio:
            return "empty audio";
        case AudioPlaybackStatus::kInvalidSampleRate:
            return "invalid sample rate";
        case AudioPlaybackStatus::kUnsupportedChannels:
            return "unsupported channel count";
        case AudioPlaybackStatus::kUnsupportedSampleFormat:
            return "unsupported sample format";
        case AudioPlaybackStatus::kInvalidPcmData:
            return "invalid PCM data";
        case AudioPlaybackStatus::kInvalidState:
            return "invalid playback state";
        case AudioPlaybackStatus::kInitializeFailed:
            return "PortAudio initialization failed";
        case AudioPlaybackStatus::kNoOutputDevice:
            return "no PortAudio output device";
        case AudioPlaybackStatus::kInvalidOutputDevice:
            return "invalid PortAudio output device";
        case AudioPlaybackStatus::kUnsupportedOutputFormat:
            return "unsupported PortAudio output format";
        case AudioPlaybackStatus::kOpenStreamFailed:
            return "PortAudio stream open failed";
        case AudioPlaybackStatus::kFinishedCallbackFailed:
            return "PortAudio finished callback registration failed";
        case AudioPlaybackStatus::kStartStreamFailed:
            return "PortAudio stream start failed";
        case AudioPlaybackStatus::kStopStreamFailed:
            return "PortAudio stream stop failed";
        case AudioPlaybackStatus::kAbortStreamFailed:
            return "PortAudio stream abort failed";
        case AudioPlaybackStatus::kCloseStreamFailed:
            return "PortAudio stream close failed";
        case AudioPlaybackStatus::kTerminateFailed:
            return "PortAudio termination failed";
        case AudioPlaybackStatus::kInternalError:
            return "internal playback error";
        default:
            return "unrecognized playback status";
    }
}

}  // namespace sync
}  // namespace digital_human
