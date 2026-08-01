#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <portaudio.h>

#include "audio/audio_loader.h"
#include "sync/portaudio_callback_state.h"
#include "sync/portaudio_playback.h"

namespace digital_human {
namespace sync {
namespace {

audio::AudioData ValidAudio() {
    audio::AudioData audio;
    audio.pcm = {0.0F, 0.25F, -0.25F, 0.5F};
    audio.sample_rate = 16000;
    audio.channels = 1;
    audio.format = audio::AudioSampleFormat::kFloat32;
    return audio;
}

TEST(PortAudioCallbackStateTest, CopiesBuffersAndPublishesAnchor) {
    const std::vector<float> pcm = {0.1F, 0.2F, 0.3F, 0.4F};
    internal::PortAudioCallbackState state;
    state.Reset(pcm.data(), 4, 1);

    PaStreamCallbackTimeInfo time_info{};
    time_info.outputBufferDacTime = 1.25;
    std::vector<float> first_output(2, -1.0F);
    EXPECT_EQ(internal::PortAudioCallbackState::Callback(
                  nullptr,
                  first_output.data(),
                  2,
                  &time_info,
                  0,
                  &state),
              paContinue);
    EXPECT_EQ(first_output, (std::vector<float>{0.1F, 0.2F}));
    EXPECT_EQ(state.next_frame.load(), 2U);

    std::uint64_t anchor_frame = 99;
    std::int64_t anchor_time_us = -1;
    ASSERT_TRUE(state.ReadAnchor(&anchor_frame, &anchor_time_us));
    EXPECT_EQ(anchor_frame, 0U);
    EXPECT_EQ(anchor_time_us, 1250000);

    time_info.outputBufferDacTime = 1.5;
    std::vector<float> second_output(2, -1.0F);
    EXPECT_EQ(internal::PortAudioCallbackState::Callback(
                  nullptr,
                  second_output.data(),
                  2,
                  &time_info,
                  0,
                  &state),
              paComplete);
    EXPECT_EQ(second_output, (std::vector<float>{0.3F, 0.4F}));
    EXPECT_TRUE(state.source_exhausted.load());
    EXPECT_EQ(state.callback_count.load(), 2U);
}

TEST(PortAudioCallbackStateTest, ZeroFillsPartialFinalBuffer) {
    const std::vector<float> pcm = {0.1F, 0.2F, 0.3F};
    internal::PortAudioCallbackState state;
    state.Reset(pcm.data(), 3, 1);
    PaStreamCallbackTimeInfo time_info{};
    time_info.outputBufferDacTime = 0.0;
    std::vector<float> output(5, -1.0F);

    EXPECT_EQ(internal::PortAudioCallbackState::Callback(
                  nullptr,
                  output.data(),
                  5,
                  &time_info,
                  paOutputUnderflow,
                  &state),
              paComplete);
    EXPECT_EQ(output, (std::vector<float>{0.1F, 0.2F, 0.3F, 0.0F, 0.0F}));
    EXPECT_EQ(state.next_frame.load(), 3U);
    EXPECT_EQ(state.output_underflow_count.load(), 1U);
}

TEST(PortAudioCallbackStateTest, HandlesInterleavedChannelsWithoutOverflow) {
    const std::vector<float> pcm = {0.1F, -0.1F, 0.2F, -0.2F};
    internal::PortAudioCallbackState state;
    state.Reset(pcm.data(), 2, 2);
    PaStreamCallbackTimeInfo time_info{};
    std::vector<float> output(6, -1.0F);

    EXPECT_EQ(internal::PortAudioCallbackState::Callback(
                  nullptr,
                  output.data(),
                  3,
                  &time_info,
                  0,
                  &state),
              paComplete);
    EXPECT_EQ(output, (std::vector<float>{
        0.1F, -0.1F, 0.2F, -0.2F, 0.0F, 0.0F}));
}

TEST(PortAudioCallbackStateTest, AbortsOnInvalidCallbackPointers) {
    internal::PortAudioCallbackState state;
    const float pcm[] = {0.0F};
    state.Reset(pcm, 1, 1);
    EXPECT_EQ(internal::PortAudioCallbackState::Callback(
                  nullptr, nullptr, 1, nullptr, 0, &state),
              paAbort);
    EXPECT_TRUE(state.callback_error.load());
}

TEST(PortAudioClockMathTest, FutureDacAnchorDoesNotAdvanceClockEarly) {
    EXPECT_EQ(internal::PortAudioCallbackState::EstimatePlayedFrame(
                  400,
                  2000000,
                  1900000,
                  16000,
                  800,
                  1000,
                  320),
              320U);
}

TEST(PortAudioClockMathTest, AdvancesFromDacAnchorAtSampleRate) {
    // 25 ms at 16 kHz = 400 frames, added to anchor frame 800.
    EXPECT_EQ(internal::PortAudioCallbackState::EstimatePlayedFrame(
                  800,
                  1000000,
                  1025000,
                  16000,
                  1400,
                  2000,
                  700),
              1200U);
}

TEST(PortAudioClockMathTest, ClampsToSubmittedAndTotalFrames) {
    EXPECT_EQ(internal::PortAudioCallbackState::EstimatePlayedFrame(
                  900,
                  0,
                  1000000,
                  16000,
                  1000,
                  1200,
                  950),
              1000U);
    EXPECT_EQ(internal::PortAudioCallbackState::EstimatePlayedFrame(
                  std::numeric_limits<std::uint64_t>::max() - 2,
                  0,
                  std::numeric_limits<std::int64_t>::max(),
                  16000,
                  1200,
                  1000,
                  900),
              1000U);
}

TEST(PortAudioPlaybackValidationTest, RejectsInvalidConfigurationBeforeBackend) {
    PortAudioPlaybackConfig config;
    config.output_device_index = -2;
    PortAudioPlayback playback(config);
    const AudioPlaybackResult result = playback.Prepare(ValidAudio());
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.status, AudioPlaybackStatus::kInvalidConfiguration);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(PortAudioPlaybackValidationTest, RejectsInvalidAudioBeforeBackend) {
    {
        PortAudioPlayback playback;
        audio::AudioData audio = ValidAudio();
        audio.pcm.clear();
        EXPECT_EQ(playback.Prepare(audio).status,
                  AudioPlaybackStatus::kEmptyAudio);
    }
    {
        PortAudioPlayback playback;
        audio::AudioData audio = ValidAudio();
        audio.sample_rate = 0;
        EXPECT_EQ(playback.Prepare(audio).status,
                  AudioPlaybackStatus::kInvalidSampleRate);
    }
    {
        PortAudioPlayback playback;
        audio::AudioData audio = ValidAudio();
        audio.channels = 2;
        EXPECT_EQ(playback.Prepare(audio).status,
                  AudioPlaybackStatus::kUnsupportedChannels);
    }
    {
        PortAudioPlayback playback;
        audio::AudioData audio = ValidAudio();
        audio.format = audio::AudioSampleFormat::kInt16;
        EXPECT_EQ(playback.Prepare(audio).status,
                  AudioPlaybackStatus::kUnsupportedSampleFormat);
    }
    {
        PortAudioPlayback playback;
        audio::AudioData audio = ValidAudio();
        audio.pcm[1] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_EQ(playback.Prepare(audio).status,
                  AudioPlaybackStatus::kInvalidPcmData);
    }
}

TEST(PortAudioPlaybackStateTest, IdleClockIsExplicitlyUnavailable) {
    PortAudioPlayback playback;
    const AudioClockResult clock = playback.CurrentTime();
    EXPECT_FALSE(clock.success);
    EXPECT_EQ(clock.status, AudioClockStatus::kNotReady);
    EXPECT_EQ(clock.snapshot.state, AudioClockState::kIdle);
    EXPECT_FALSE(clock.error_message.empty());

    const AudioPlaybackResult reset = playback.Reset();
    EXPECT_TRUE(reset.success);
    EXPECT_EQ(reset.state, AudioClockState::kIdle);
}

TEST(PortAudioPlaybackStatusTest, EveryKnownStatusHasText) {
    const std::vector<AudioPlaybackStatus> statuses = {
        AudioPlaybackStatus::kOk,
        AudioPlaybackStatus::kInvalidConfiguration,
        AudioPlaybackStatus::kEmptyAudio,
        AudioPlaybackStatus::kInvalidSampleRate,
        AudioPlaybackStatus::kUnsupportedChannels,
        AudioPlaybackStatus::kUnsupportedSampleFormat,
        AudioPlaybackStatus::kInvalidPcmData,
        AudioPlaybackStatus::kInvalidState,
        AudioPlaybackStatus::kInitializeFailed,
        AudioPlaybackStatus::kNoOutputDevice,
        AudioPlaybackStatus::kInvalidOutputDevice,
        AudioPlaybackStatus::kUnsupportedOutputFormat,
        AudioPlaybackStatus::kOpenStreamFailed,
        AudioPlaybackStatus::kFinishedCallbackFailed,
        AudioPlaybackStatus::kStartStreamFailed,
        AudioPlaybackStatus::kStopStreamFailed,
        AudioPlaybackStatus::kAbortStreamFailed,
        AudioPlaybackStatus::kCloseStreamFailed,
        AudioPlaybackStatus::kTerminateFailed,
        AudioPlaybackStatus::kInternalError};
    for (AudioPlaybackStatus status : statuses) {
        EXPECT_FALSE(PortAudioPlayback::StatusToString(status).empty());
    }

    const std::vector<AudioClockState> states = {
        AudioClockState::kIdle,
        AudioClockState::kPrepared,
        AudioClockState::kRunning,
        AudioClockState::kCompleted,
        AudioClockState::kStopped,
        AudioClockState::kError};
    for (AudioClockState state : states) {
        EXPECT_FALSE(AudioClockStateToString(state).empty());
    }
}

}  // namespace
}  // namespace sync
}  // namespace digital_human
