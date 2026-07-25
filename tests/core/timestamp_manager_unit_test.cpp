#include "audio/audio_stream_buffer.h"
#include "core/timestamp_manager.h"
#include "model/input_processor.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

// 本文件测试 TimestampManager 的公开静态接口。
// 每个测试都用独立常量作为期望值，避免“用生产函数验证生产函数”。
namespace digital_human {
namespace core {
namespace {

// 成功结果必须同时满足四项契约：成功标志、kOk、空错误信息和精确微秒值。
void ExpectSuccessfulTimestamp(
    const TimestampResult& result,
    int64_t expected_microseconds) {
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.status, TimestampStatus::kOk);
    EXPECT_TRUE(result.error_message.empty());
    EXPECT_EQ(result.value.microseconds, expected_microseconds);
}

// 失败结果必须同时给出 false、精确状态和非空错误信息。
void ExpectTimestampFailure(
    const TimestampResult& result,
    TimestampStatus expected_status) {
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.status, expected_status);
    EXPECT_FALSE(result.error_message.empty());
}

// 验证接口：FromAudioSampleIndex(sample_index, sample_rate)。
// 它把样本在整条流中的位置换算为微秒 PTS，并统一执行最近整数舍入。
TEST(TimestampManagerAudioTest, ConvertsExactSamplePositions) {
    struct TestCase {
        int64_t sample_index;
        int sample_rate;
        int64_t expected_microseconds;
    };

    const std::vector<TestCase> test_cases = {
        {0, 16000, 0},
        {1, 16000, 63},
        {8000, 16000, 500000},
        {16000, 16000, 1000000}
    };

    for (const TestCase& test_case : test_cases) {
        const TimestampResult result =
            TimestampManager::FromAudioSampleIndex(
                test_case.sample_index,
                test_case.sample_rate);
        ExpectSuccessfulTimestamp(
            result,
            test_case.expected_microseconds);
    }
}

// 负索引没有合法媒体位置，必须返回 kNegativeIndex，不能伪装成 0us。
TEST(TimestampManagerAudioTest, RejectsNegativeSampleIndex) {
    const TimestampResult result =
        TimestampManager::FromAudioSampleIndex(-1, 16000);

    ExpectTimestampFailure(
        result,
        TimestampStatus::kNegativeIndex);
}

// 采样率为零或负数时 time base 无效，必须返回 kInvalidSampleRate。
TEST(TimestampManagerAudioTest, RejectsNonPositiveSampleRate) {
    const TimestampResult zero_rate_result =
        TimestampManager::FromAudioSampleIndex(1, 0);
    const TimestampResult negative_rate_result =
        TimestampManager::FromAudioSampleIndex(1, -16000);

    ExpectTimestampFailure(
        zero_rate_result,
        TimestampStatus::kInvalidSampleRate);
    ExpectTimestampFailure(
        negative_rate_result,
        TimestampStatus::kInvalidSampleRate);
}

// 极大索引换算后超出 int64 微秒范围时，接口必须报告溢出。
TEST(TimestampManagerAudioTest, RejectsMicrosecondOverflow) {
    const TimestampResult result =
        TimestampManager::FromAudioSampleIndex(
            std::numeric_limits<int64_t>::max(),
            1);

    ExpectTimestampFailure(result, TimestampStatus::kOverflow);
}

// 验证接口：FromMilliseconds(pts_ms)，用于把旧整数毫秒安全接入微秒时间轴。
TEST(TimestampManagerMillisecondsTest, ConvertsExactValues) {
    const TimestampResult zero_result =
        TimestampManager::FromMilliseconds(0);
    const TimestampResult one_second_result =
        TimestampManager::FromMilliseconds(1000);

    ExpectSuccessfulTimestamp(zero_result, 0);
    ExpectSuccessfulTimestamp(one_second_result, 1000000);
}

// 当前契约不支持负媒体时间。
TEST(TimestampManagerMillisecondsTest, RejectsNegativeTimestamp) {
    const TimestampResult result =
        TimestampManager::FromMilliseconds(-1);

    ExpectTimestampFailure(
        result,
        TimestampStatus::kNegativeTimestamp);
}

// 验证乘以 1000 前执行上界检查，不发生有符号整数溢出。
TEST(TimestampManagerMillisecondsTest, RejectsOverflow) {
    constexpr int64_t kMicrosecondsPerMillisecond = 1000;
    const int64_t first_overflowing_milliseconds =
        std::numeric_limits<int64_t>::max() /
        kMicrosecondsPerMillisecond + 1;

    const TimestampResult result =
        TimestampManager::FromMilliseconds(
            first_overflowing_milliseconds);

    ExpectTimestampFailure(result, TimestampStatus::kOverflow);
}

// 验证接口：ToMilliseconds(timestamp)，采用最近整数且中点远离零的舍入。
TEST(TimestampManagerMillisecondsTest, RoundsToNearestMillisecond) {
    EXPECT_EQ(
        TimestampManager::ToMilliseconds(MediaTimestamp{33367}),
        33);
    EXPECT_EQ(
        TimestampManager::ToMilliseconds(MediaTimestamp{33499}),
        33);
    EXPECT_EQ(
        TimestampManager::ToMilliseconds(MediaTimestamp{33500}),
        34);
    EXPECT_EQ(
        TimestampManager::ToMilliseconds(MediaTimestamp{-33499}),
        -33);
    EXPECT_EQ(
        TimestampManager::ToMilliseconds(MediaTimestamp{-33500}),
        -34);
}

// 验证接口：Normalize(timestamp, origin)，结果应为 timestamp - origin。
TEST(TimestampManagerNormalizeTest, SubtractsOrigin) {
    const TimestampResult result = TimestampManager::Normalize(
        MediaTimestamp{5000000},
        MediaTimestamp{2000000});

    ExpectSuccessfulTimestamp(result, 3000000);
}

// 起点等于当前时间时，合法归一化结果为 0us。
TEST(TimestampManagerNormalizeTest, AllowsEqualOrigin) {
    const TimestampResult result = TimestampManager::Normalize(
        MediaTimestamp{2000000},
        MediaTimestamp{2000000});

    ExpectSuccessfulTimestamp(result, 0);
}

// 起点晚于当前时间会产生负相对时间，当前契约必须明确拒绝。
TEST(TimestampManagerNormalizeTest, RejectsOriginAfterTimestamp) {
    const TimestampResult result = TimestampManager::Normalize(
        MediaTimestamp{1000000},
        MediaTimestamp{2000000});

    ExpectTimestampFailure(
        result,
        TimestampStatus::kOriginAfterTimestamp);
}

// timestamp 和 origin 任一为负都不属于当前支持的时间轴。
TEST(TimestampManagerNormalizeTest, RejectsNegativeInputs) {
    const TimestampResult negative_timestamp_result =
        TimestampManager::Normalize(
            MediaTimestamp{-1},
            MediaTimestamp{0});
    const TimestampResult negative_origin_result =
        TimestampManager::Normalize(
            MediaTimestamp{0},
            MediaTimestamp{-1});

    ExpectTimestampFailure(
        negative_timestamp_result,
        TimestampStatus::kNegativeTimestamp);
    ExpectTimestampFailure(
        negative_origin_result,
        TimestampStatus::kNegativeTimestamp);
}

// 验证接口：DifferenceMicroseconds(lhs, rhs) 保留 lhs - rhs 的正负方向。
TEST(TimestampManagerDifferenceTest, PreservesSignedDirection) {
    constexpr int64_t kTenMillisecondsInMicroseconds = 10000;
    const MediaTimestamp video_ahead{50000};
    const MediaTimestamp audio_behind{40000};
    const MediaTimestamp video_behind{40000};
    const MediaTimestamp audio_ahead{50000};

    EXPECT_EQ(
        TimestampManager::DifferenceMicroseconds(
            video_ahead,
            audio_behind),
        kTenMillisecondsInMicroseconds);
    EXPECT_EQ(
        TimestampManager::DifferenceMicroseconds(
            video_behind,
            audio_ahead),
        -kTenMillisecondsInMicroseconds);
    EXPECT_EQ(
        TimestampManager::DifferenceMicroseconds(
            video_ahead,
            video_ahead),
        0);
}

// 验证接口：ValidateMonotonic(previous, current)，相等和前进都合法。
TEST(TimestampManagerMonotonicTest, AllowsEqualAndIncreasingTimestamps) {
    EXPECT_EQ(
        TimestampManager::ValidateMonotonic(
            MediaTimestamp{1000},
            MediaTimestamp{1000}),
        TimestampStatus::kOk);
    EXPECT_EQ(
        TimestampManager::ValidateMonotonic(
            MediaTimestamp{1000},
            MediaTimestamp{1001}),
        TimestampStatus::kOk);
}

// 当前时间小于上一时间时，必须返回 kNonMonotonic。
TEST(TimestampManagerMonotonicTest, RejectsTimestampRegression) {
    EXPECT_EQ(
        TimestampManager::ValidateMonotonic(
            MediaTimestamp{1001},
            MediaTimestamp{1000}),
        TimestampStatus::kNonMonotonic);
}

// StatusToString 必须覆盖每个公开状态，避免上层收到空错误名称。
TEST(TimestampManagerStatusTest, ReturnsTextForEveryStatus) {
    const std::vector<TimestampStatus> statuses = {
        TimestampStatus::kOk,
        TimestampStatus::kNegativeIndex,
        TimestampStatus::kInvalidSampleRate,
        TimestampStatus::kInvalidFrameRate,
        TimestampStatus::kNegativeTimestamp,
        TimestampStatus::kOriginAfterTimestamp,
        TimestampStatus::kOverflow,
        TimestampStatus::kNonMonotonic,
        TimestampStatus::kUnknownError
    };

    for (TimestampStatus status : statuses) {
        EXPECT_FALSE(
            TimestampManager::StatusToString(status).empty());
    }
}

// operator<< 让 GTest 在枚举断言失败时输出可读状态，而不是编译失败。
TEST(TimestampManagerStatusTest, SupportsTestFailureFormatting) {
    std::ostringstream stream;
    stream << TimestampStatus::kOverflow;

    EXPECT_EQ(stream.str(), "timestamp overflow");
}

// 验证 25/1 和 30000/1001 两种帧率的已知精确位置。
// 每个期望值都由公式独立手算，不依赖生产函数生成。
TEST(TimestampManagerVideoTest, ConvertsExactVideoFramePositions) {
    struct TestCase {
        FrameRate frame_rate;
        int64_t frame_index;
        int64_t expected_microseconds;
    };

    const std::vector<TestCase> test_cases = {
        {FrameRate{25, 1}, 0, 0},
        {FrameRate{25, 1}, 1, 40000},
        {FrameRate{25, 1}, 25, 1000000},
        {FrameRate{30000, 1001}, 30000, 1001000000}
    };

    for (const TestCase& test_case : test_cases) {
        const TimestampResult result =
            TimestampManager::FromVideoFrameIndex(
                test_case.frame_index,
                test_case.frame_rate);
        ExpectSuccessfulTimestamp(
            result,
            test_case.expected_microseconds);
    }
}

// 30000/1001 fps 的第一帧位于 33366.666...us，
// 最近整数舍入应得到 33367us，不能直接向下截断为 33366us。
TEST(TimestampManagerVideoTest, RoundsFractionalMicrosecondsToNearestInteger) {
    const TimestampResult result =
        TimestampManager::FromVideoFrameIndex(
            1,
            FrameRate{30000, 1001});

    ExpectSuccessfulTimestamp(result, 33367);
}

// 负帧索引没有合法媒体位置，必须返回 kNegativeIndex。
TEST(TimestampManagerVideoTest, RejectsNegativeFrameIndex) {
    const TimestampResult result =
        TimestampManager::FromVideoFrameIndex(
            -1,
            FrameRate{25, 1});

    ExpectTimestampFailure(
        result,
        TimestampStatus::kNegativeIndex);
}

// 帧率分子或分母为零、负数时，时间基准无效。
TEST(TimestampManagerVideoTest, RejectsInvalidFrameRates) {
    const std::vector<FrameRate> invalid_frame_rates = {
        FrameRate{0, 1},
        FrameRate{-1, 1},
        FrameRate{25, 0},
        FrameRate{25, -1}
    };

    for (const FrameRate& frame_rate : invalid_frame_rates) {
        const TimestampResult result =
            TimestampManager::FromVideoFrameIndex(1, frame_rate);
        ExpectTimestampFailure(
            result,
            TimestampStatus::kInvalidFrameRate);
    }
}

// 极大帧索引换算后超出 int64 微秒范围时，必须报告溢出。
TEST(TimestampManagerVideoTest, RejectsMicrosecondOverflow) {
    const TimestampResult result =
        TimestampManager::FromVideoFrameIndex(
            std::numeric_limits<int64_t>::max(),
            FrameRate{1, 1});

    ExpectTimestampFailure(result, TimestampStatus::kOverflow);
}

// 用户完成的长期时间轴测试：
// 在 30000/1001 fps 下，第 107892 帧位于约一小时处。
// 期望值 3599996400us 来自独立手算，不由生产函数生成。
// 该断言用于捕获“先舍入单帧时长、再逐帧累加”造成的长期漂移。
TEST(TimestampManagerVideoTest, ConvertsOneHourRationalFrameRateWithoutDrift) {
    const FrameRate frame_rate{30000, 1001};
    constexpr int64_t kOneHourFrameIndex = 107892;
    constexpr int64_t kExpectedMicroseconds = 3599996400LL;

    const TimestampResult result =
        TimestampManager::FromVideoFrameIndex(
            kOneHourFrameIndex,
            frame_rate);

    ExpectSuccessfulTimestamp(result, kExpectedMicroseconds);
}

// 真实窄集成：
// 1. AudioStreamBuffer 实际拉取第二个非零起点音频块；
// 2. TimestampManager 把样本索引和帧索引换算到同一时间点；
// 3. Wav2LipInputBuilder 实际接收并原样透传 pts_ms/frame_index。
// 该测试只证明时间字段跨模块契约接通，不证明真实播放同步或视觉效果。
TEST(TimestampManagerIntegrationTest, PropagatesNonZeroTimeMetadata) {
    constexpr int kSampleRate = 16000;
    constexpr int kVideoFps = 25;
    constexpr int64_t kVideoFrameIndex = 1;
    constexpr size_t kSamplesPerVideoFrame =
        static_cast<size_t>(kSampleRate / kVideoFps);
    constexpr size_t kBufferedSampleCount =
        kSamplesPerVideoFrame * 2;
    constexpr int64_t kExpectedTimestampMicroseconds = 40000;
    constexpr int64_t kExpectedTimestampMilliseconds = 40;
    constexpr int kFaceSize = 96;
    constexpr size_t kMelValueCount = 1280;

    digital_human::audio::AudioStreamBufferOptions buffer_options;
    buffer_options.capacity_samples = kBufferedSampleCount;
    buffer_options.sample_rate = kSampleRate;
    digital_human::audio::AudioStreamBuffer buffer(buffer_options);

    const std::vector<float> pcm(kBufferedSampleCount, 0.25f);
    const digital_human::audio::AudioStreamPushResult push_result =
        buffer.PushSamples(
            pcm,
            digital_human::audio::AudioBufferOverflowStrategy::kBlock,
            -1);
    ASSERT_TRUE(push_result.success) << push_result.error_message;

    const digital_human::audio::AudioStreamPullResult first_chunk =
        buffer.PullChunk(kSamplesPerVideoFrame, -1);
    ASSERT_TRUE(first_chunk.success) << first_chunk.error_message;
    EXPECT_EQ(first_chunk.chunk.start_sample_index, 0);

    const digital_human::audio::AudioStreamPullResult second_chunk =
        buffer.PullChunk(kSamplesPerVideoFrame, -1);
    ASSERT_TRUE(second_chunk.success) << second_chunk.error_message;
    ASSERT_EQ(
        second_chunk.chunk.start_sample_index,
        static_cast<int64_t>(kSamplesPerVideoFrame));
    // 640 samples / 16000 Hz = 40ms；验证旧毫秒字段与样本索引来自同一起点。
    EXPECT_DOUBLE_EQ(
        second_chunk.chunk.start_pts_ms,
        static_cast<double>(kExpectedTimestampMilliseconds));

    const TimestampResult audio_timestamp =
        TimestampManager::FromAudioSampleIndex(
            second_chunk.chunk.start_sample_index,
            second_chunk.chunk.sample_rate);
    ExpectSuccessfulTimestamp(
        audio_timestamp,
        kExpectedTimestampMicroseconds);

    const TimestampResult video_timestamp =
        TimestampManager::FromVideoFrameIndex(
            kVideoFrameIndex,
            FrameRate{kVideoFps, 1});
    ExpectSuccessfulTimestamp(
        video_timestamp,
        kExpectedTimestampMicroseconds);

    digital_human::model::ModelInputMetadata metadata;
    metadata.pts_ms =
        TimestampManager::ToMilliseconds(video_timestamp.value);
    metadata.frame_index = kVideoFrameIndex;

    const cv::Mat face(
        kFaceSize,
        kFaceSize,
        CV_8UC3,
        cv::Scalar(10, 20, 30));
    const std::vector<float> mel(kMelValueCount, 0.0f);
    const digital_human::model::Wav2LipInputBuilder builder;
    const digital_human::model::Wav2LipInputResult input_result =
        builder.Build(face, mel, metadata);

    ASSERT_TRUE(input_result.success) << input_result.error_message;
    ASSERT_TRUE(input_result.data.metadata.pts_ms.has_value());
    ASSERT_TRUE(input_result.data.metadata.frame_index.has_value());
    EXPECT_EQ(
        input_result.data.metadata.pts_ms.value(),
        kExpectedTimestampMilliseconds);
    EXPECT_EQ(
        input_result.data.metadata.frame_index.value(),
        kVideoFrameIndex);
}

// 并发消费者可以先后拿到不同音频块，但每块的累计样本起点必须唯一且连续。
// 起点与 PCM 在同一次加锁读取中产生，供 TimestampManager 后续安全换算 PTS。
TEST(TimestampManagerIntegrationTest, PreservesChunkStartIndexAcrossConsumers) {
    constexpr int kSampleRate = 1000;
    constexpr size_t kConsumerCount = 8;
    constexpr size_t kSamplesPerChunk = 10;
    constexpr size_t kTotalSampleCount =
        kConsumerCount * kSamplesPerChunk;

    digital_human::audio::AudioStreamBufferOptions buffer_options;
    buffer_options.capacity_samples = kTotalSampleCount;
    buffer_options.sample_rate = kSampleRate;
    digital_human::audio::AudioStreamBuffer buffer(buffer_options);

    const std::vector<float> pcm(kTotalSampleCount, 0.25f);
    const digital_human::audio::AudioStreamPushResult push_result =
        buffer.PushSamples(
            pcm,
            digital_human::audio::AudioBufferOverflowStrategy::kBlock,
            -1);
    ASSERT_TRUE(push_result.success) << push_result.error_message;

    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    size_t ready_consumer_count = 0;
    bool consumers_can_start = false;
    std::vector<digital_human::audio::AudioStreamPullResult> results(
        kConsumerCount);
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerCount);

    for (size_t consumer_index = 0;
         consumer_index < kConsumerCount;
         ++consumer_index) {
        consumers.emplace_back([&, consumer_index] {
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                ++ready_consumer_count;
                gate_cv.notify_all();
                gate_cv.wait(lock, [&] {
                    return consumers_can_start;
                });
            }
            results[consumer_index] =
                buffer.PullChunk(kSamplesPerChunk, -1);
        });
    }

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&] {
            return ready_consumer_count == kConsumerCount;
        });
        consumers_can_start = true;
    }
    gate_cv.notify_all();

    for (std::thread& consumer : consumers) {
        consumer.join();
    }

    std::vector<int64_t> start_sample_indices;
    start_sample_indices.reserve(kConsumerCount);
    for (const auto& result : results) {
        ASSERT_TRUE(result.success) << result.error_message;
        start_sample_indices.push_back(
            result.chunk.start_sample_index);
    }
    std::sort(
        start_sample_indices.begin(),
        start_sample_indices.end());

    for (size_t chunk_index = 0;
         chunk_index < kConsumerCount;
         ++chunk_index) {
        EXPECT_EQ(
            start_sample_indices[chunk_index],
            static_cast<int64_t>(
                chunk_index * kSamplesPerChunk));
    }
}


}  // namespace
}  // namespace core
}  // namespace digital_human
