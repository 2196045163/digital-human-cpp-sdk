#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "audio/audio_loader.h"
#include "core/frame_scheduler.h"
#include "sync/audio_clock.h"
#include "sync/audio_video_synchronous.h"
#include "sync/portaudio_playback.h"
#include "video/video_frame.h"

namespace {

using digital_human::audio::AudioLoader;
using digital_human::core::FrameSchedulerConfig;
using digital_human::core::OverflowPolicy;
using digital_human::core::ScheduleAction;
using digital_human::sync::AudioClock;
using digital_human::sync::AudioClockResult;
using digital_human::sync::AudioClockState;
using digital_human::sync::AudioClockStatus;
using digital_human::sync::AudioPlaybackResult;
using digital_human::sync::AudioVideoSynchronous;
using digital_human::sync::PortAudioPlayback;
using digital_human::video::VideoFrame;

class SequenceClock final : public AudioClock {
public:
    explicit SequenceClock(std::vector<std::int64_t> timestamps_us)
        : timestamps_us_(std::move(timestamps_us)) {}

    AudioClockResult CurrentTime() const override {
        AudioClockResult result;
        result.success = true;
        result.status = AudioClockStatus::kOk;
        result.snapshot.state = AudioClockState::kRunning;
        const std::size_t selected = index_ < timestamps_us_.size()
            ? index_
            : timestamps_us_.size() - 1;
        result.snapshot.timestamp.microseconds = timestamps_us_[selected];
        result.snapshot.reference_frame_index = selected;
        ++index_;
        return result;
    }

private:
    std::vector<std::int64_t> timestamps_us_;
    mutable std::size_t index_ = 0;
};

std::string ActionText(ScheduleAction action) {
    switch (action) {
        case ScheduleAction::kDeliver:
            return "deliver";
        case ScheduleAction::kDropAndDeliver:
            return "drop_and_deliver";
        case ScheduleAction::kRepeatLast:
            return "repeat_last";
        case ScheduleAction::kBufferingNoFrame:
            return "buffering_no_frame";
        case ScheduleAction::kRejectedInvalidReferenceTime:
            return "rejected_invalid_reference_time";
        default:
            return "unknown";
    }
}

VideoFrame MakeFrame(std::int64_t pts_us, std::int64_t index) {
    VideoFrame frame;
    frame.frame_bgr = cv::Mat(
        4, 4, CV_8UC3, cv::Scalar(index, index + 1, index + 2)).clone();
    frame.pts.microseconds = pts_us;
    frame.frame_index = index;
    return frame;
}

int RunDeterministicTrace() {
    auto clock = std::make_shared<SequenceClock>(
        std::vector<std::int64_t>{0, 50000, 100000, 120000});
    FrameSchedulerConfig config;
    config.max_queue_size = 8;
    config.min_buffered_frames = 1;
    config.sync_tolerance_us = 10000;
    config.overflow_policy = OverflowPolicy::kDropOldest;
    AudioVideoSynchronous synchronizer(clock, config);

    for (std::int64_t index = 0; index < 4; ++index) {
        const std::int64_t pts_us = index * 40000;
        synchronizer.PushVideoFrame(MakeFrame(pts_us, index));
    }

    std::filesystem::create_directories("golden_output");
    std::ofstream output(
        "golden_output/audio_video_synchronous_trace.json",
        std::ios::trunc);
    if (!output) {
        std::cerr << "Failed to open deterministic trace output\n";
        return 1;
    }

    output << "{\n  \"source\": \"deterministic_fake_audio_clock\",\n"
           << "  \"sync_tolerance_us\": 10000,\n"
           << "  \"decisions\": [\n";
    for (int index = 0; index < 4; ++index) {
        const auto result = synchronizer.ScheduleNext();
        if (!result.success) {
            std::cerr << "Schedule failed: " << result.error_message << "\n";
            return 2;
        }
        output << "    {\"audio_pts_us\": "
               << result.clock.timestamp.microseconds
               << ", \"action\": \""
               << ActionText(result.schedule.action)
               << "\", \"selected_frame\": ";
        if (result.schedule.selected_frame.has_value()) {
            output << result.schedule.selected_frame->frame_index;
        } else {
            output << "null";
        }
        output << ", \"dropped\": "
               << result.schedule.dropped_this_call
               << ", \"queue_after\": "
               << result.schedule.queue_size_after << "}";
        output << (index == 3 ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"proves\": \"clock-to-scheduler contract\",\n"
           << "  \"does_not_prove\": \"hardware playback timing\"\n"
           << "}\n";
    std::cout << "Wrote golden_output/audio_video_synchronous_trace.json\n";
    return 0;
}

int RunDeviceSmoke(const std::string& audio_path) {
    AudioLoader loader(16000);
    const auto loaded = loader.LoadFromFile(audio_path);
    if (!loaded.success) {
        std::cerr << "Audio load failed: " << loaded.error_message << "\n";
        return 3;
    }

    PortAudioPlayback playback;
    const auto prepared = playback.Prepare(loaded.audio);
    if (!prepared.success) {
        std::cerr << "Prepare failed: " << prepared.error_message << "\n";
        return 4;
    }
    const auto started = playback.Start();
    if (!started.success) {
        std::cerr << "Start failed: " << started.error_message << "\n";
        playback.Reset();
        return 5;
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const auto deadline = wall_start + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        const AudioClockState state = playback.GetState();
        if (state == AudioClockState::kCompleted
            || state == AudioClockState::kError) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    const auto clock = playback.CurrentTime();
    const auto stats = playback.GetStats();
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();
    std::cout << "state="
              << digital_human::sync::AudioClockStateToString(stats.state)
              << " clock_success=" << (clock.success ? "true" : "false")
              << " audio_pts_us="
              << (clock.success ? clock.snapshot.timestamp.microseconds : -1)
              << " total_frames=" << stats.total_audio_frames
              << " submitted_frames=" << stats.submitted_audio_frames
              << " estimated_played_frames=" << stats.estimated_played_frames
              << " callbacks=" << stats.callback_count
              << " underflows=" << stats.output_underflow_count
              << " latency_seconds=" << stats.output_latency_seconds
              << " wall_seconds=" << wall_seconds << "\n";

    AudioPlaybackResult finalized;
    bool finalized_stream = false;
    if (stats.state == AudioClockState::kRunning) {
        finalized = playback.Abort();
        finalized_stream = true;
    } else if (stats.state == AudioClockState::kCompleted) {
        finalized = playback.Stop();
        finalized_stream = true;
    }
    if (finalized_stream && !finalized.success) {
        std::cerr << "Playback finalization failed: "
                  << finalized.error_message << "\n";
        playback.Reset();
        return 8;
    }
    const auto reset = playback.Reset();
    if (!reset.success) {
        std::cerr << "Reset failed: " << reset.error_message << "\n";
        return 6;
    }
    return stats.state == AudioClockState::kError ? 7 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--device-smoke") {
        const std::string audio_path = argc >= 3
            ? argv[2]
            : "testdata/golden/audio.wav";
        return RunDeviceSmoke(audio_path);
    }
    return RunDeterministicTrace();
}
