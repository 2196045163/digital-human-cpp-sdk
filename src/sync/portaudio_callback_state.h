#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <portaudio.h>

namespace digital_human {
namespace sync {
namespace internal {

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "PortAudio callback counters must be lock-free");
static_assert(std::atomic<std::int64_t>::is_always_lock_free,
              "PortAudio callback clock anchors must be lock-free");
static_assert(std::atomic<bool>::is_always_lock_free,
              "PortAudio callback flags must be lock-free");

/// @brief Fixed-size state shared with the PortAudio real-time callback.
///
/// The callback performs no allocation, locking, I/O, sleep, or PortAudio API
/// call. PCM storage is owned by PortAudioPlayback and remains immutable while
/// the stream is open.
struct PortAudioCallbackState {
    const float* pcm = nullptr;
    std::uint64_t total_frames = 0;
    int channels = 0;

    std::atomic<std::uint64_t> next_frame{0};
    std::atomic<std::uint64_t> callback_count{0};
    std::atomic<std::uint64_t> output_underflow_count{0};
    std::atomic<bool> source_exhausted{false};
    std::atomic<bool> stream_finished{false};
    std::atomic<bool> callback_error{false};

    // Sequence lock for a consistent (frame, DAC time) clock anchor snapshot.
    std::atomic<std::uint64_t> anchor_generation{0};
    std::atomic<std::uint64_t> anchor_frame{0};
    std::atomic<std::int64_t> anchor_dac_time_us{-1};

    /// @brief Pure clock math shared by production polling and unit tests.
    static std::uint64_t EstimatePlayedFrame(
        std::uint64_t anchor_frame_value,
        std::int64_t anchor_dac_time_us_value,
        std::int64_t stream_time_us,
        int sample_rate,
        std::uint64_t submitted_frames,
        std::uint64_t total_frames_value,
        std::uint64_t last_reported_frame) {
        const std::uint64_t upper_bound = std::min(
            submitted_frames, total_frames_value);
        std::uint64_t estimate = std::min(
            last_reported_frame, upper_bound);
        if (sample_rate <= 0 || anchor_dac_time_us_value < 0
            || stream_time_us < anchor_dac_time_us_value) {
            return estimate;
        }

        const std::uint64_t elapsed_us = static_cast<std::uint64_t>(
            stream_time_us - anchor_dac_time_us_value);
        const std::uint64_t rate = static_cast<std::uint64_t>(sample_rate);
        const std::uint64_t elapsed_frames =
            elapsed_us > std::numeric_limits<std::uint64_t>::max() / rate
                ? std::numeric_limits<std::uint64_t>::max()
                : elapsed_us * rate / 1000000U;
        const std::uint64_t anchored_estimate =
            anchor_frame_value
                    > std::numeric_limits<std::uint64_t>::max()
                        - elapsed_frames
                ? std::numeric_limits<std::uint64_t>::max()
                : anchor_frame_value + elapsed_frames;
        estimate = std::max(estimate, anchored_estimate);
        return std::min(estimate, upper_bound);
    }

    void Reset(const float* pcm_data, std::uint64_t frames, int channel_count) {
        pcm = pcm_data;
        total_frames = frames;
        channels = channel_count;
        next_frame.store(0, std::memory_order_relaxed);
        callback_count.store(0, std::memory_order_relaxed);
        output_underflow_count.store(0, std::memory_order_relaxed);
        source_exhausted.store(false, std::memory_order_relaxed);
        stream_finished.store(false, std::memory_order_relaxed);
        callback_error.store(false, std::memory_order_relaxed);
        anchor_generation.store(0, std::memory_order_relaxed);
        anchor_frame.store(0, std::memory_order_relaxed);
        anchor_dac_time_us.store(-1, std::memory_order_relaxed);
    }

    void PublishAnchor(std::uint64_t frame, PaTime dac_time_seconds) {
        if (!(dac_time_seconds >= 0.0)) {
            return;
        }
        constexpr double kMicrosecondsPerSecond = 1000000.0;
        const double dac_time_us = dac_time_seconds * kMicrosecondsPerSecond;
        if (dac_time_us > static_cast<double>(
                std::numeric_limits<std::int64_t>::max())) {
            return;
        }

        anchor_generation.fetch_add(1, std::memory_order_acq_rel);
        anchor_frame.store(frame, std::memory_order_relaxed);
        anchor_dac_time_us.store(
            static_cast<std::int64_t>(dac_time_us + 0.5),
            std::memory_order_relaxed);
        anchor_generation.fetch_add(1, std::memory_order_release);
    }

    bool ReadAnchor(std::uint64_t* frame, std::int64_t* dac_time_us) const {
        if (frame == nullptr || dac_time_us == nullptr) {
            return false;
        }
        for (int attempt = 0; attempt < 8; ++attempt) {
            const std::uint64_t before =
                anchor_generation.load(std::memory_order_acquire);
            if ((before & 1U) != 0U) {
                continue;
            }
            const std::uint64_t candidate_frame =
                anchor_frame.load(std::memory_order_relaxed);
            const std::int64_t candidate_time =
                anchor_dac_time_us.load(std::memory_order_relaxed);
            const std::uint64_t after =
                anchor_generation.load(std::memory_order_acquire);
            if (before == after && (after & 1U) == 0U
                && candidate_time >= 0) {
                *frame = candidate_frame;
                *dac_time_us = candidate_time;
                return true;
            }
        }
        return false;
    }

    static int Callback(
        const void* input,
        void* output,
        unsigned long frame_count,
        const PaStreamCallbackTimeInfo* time_info,
        PaStreamCallbackFlags status_flags,
        void* user_data) {
        (void)input;
        auto* state = static_cast<PortAudioCallbackState*>(user_data);
        auto* output_samples = static_cast<float*>(output);
        if (state == nullptr || output_samples == nullptr
            || state->pcm == nullptr || state->channels <= 0) {
            if (state != nullptr) {
                state->callback_error.store(true, std::memory_order_release);
            }
            return paAbort;
        }

        state->callback_count.fetch_add(1, std::memory_order_relaxed);
        if ((status_flags & paOutputUnderflow) != 0U) {
            state->output_underflow_count.fetch_add(
                1, std::memory_order_relaxed);
        }

        const std::uint64_t first_frame =
            state->next_frame.load(std::memory_order_relaxed);
        if (time_info != nullptr) {
            state->PublishAnchor(first_frame, time_info->outputBufferDacTime);
        }

        const std::uint64_t remaining =
            first_frame < state->total_frames
                ? state->total_frames - first_frame
                : 0;
        const std::uint64_t frames_to_copy = std::min<std::uint64_t>(
            remaining, static_cast<std::uint64_t>(frame_count));
        const std::uint64_t channel_count =
            static_cast<std::uint64_t>(state->channels);

        for (std::uint64_t frame = 0; frame < frames_to_copy; ++frame) {
            const std::uint64_t source_offset =
                (first_frame + frame) * channel_count;
            const std::uint64_t output_offset = frame * channel_count;
            for (std::uint64_t channel = 0; channel < channel_count; ++channel) {
                output_samples[output_offset + channel] =
                    state->pcm[source_offset + channel];
            }
        }

        for (std::uint64_t frame = frames_to_copy;
             frame < static_cast<std::uint64_t>(frame_count);
             ++frame) {
            const std::uint64_t output_offset = frame * channel_count;
            for (std::uint64_t channel = 0; channel < channel_count; ++channel) {
                output_samples[output_offset + channel] = 0.0F;
            }
        }

        const std::uint64_t next = first_frame + frames_to_copy;
        state->next_frame.store(next, std::memory_order_release);
        if (next >= state->total_frames) {
            state->source_exhausted.store(true, std::memory_order_release);
            return paComplete;
        }
        return paContinue;
    }

    static void Finished(void* user_data) {
        auto* state = static_cast<PortAudioCallbackState*>(user_data);
        if (state != nullptr) {
            state->stream_finished.store(true, std::memory_order_release);
        }
    }
};

}  // namespace internal
}  // namespace sync
}  // namespace digital_human
