#include <algorithm>          // std::min, std::max, std::copy
#include <chrono>             // std::chrono::milliseconds
#include <cmath>              // std::isnan, std::isinf
#include <condition_variable> // std::condition_variable
#include <memory>
#include <mutex>              // std::mutex, std::unique_lock, std::lock_guard
#include <assert.h>

#include "audio/audio_stream_buffer.h"

namespace digital_human {
namespace audio {

namespace {
    /// @brief 构造失败 Push 结果
    static AudioStreamPushResult MakePushError(AudioStreamBufferStatus status) {
        AudioStreamPushResult res;
        res.success = false;
        res.status = status;
        res.error_message = AudioStreamBuffer::StatusToString(status);
        return res;
    }

    /// @brief 构造失败 Pull 结果
    static AudioStreamPullResult MakePullError(AudioStreamBufferStatus status) {
        AudioStreamPullResult res;
        res.success = false;
        res.status = status;
        res.error_message = AudioStreamBuffer::StatusToString(status);
        return res;
    }
} // namespace

// ====== Impl ======
struct AudioStreamBuffer::Impl {
    AudioStreamBufferOptions options;
    std::vector<float> buffer;      // 环形缓冲区物理存储
    size_t capacity = 0;            // 总容量（采样点数）
    size_t read_pos = 0;            // 读指针
    size_t write_pos = 0;           // 写指针
    size_t current_size = 0;        // 当前有效数据量
    bool closed = false;            // 是否已关闭

    // 溢出统计（PushSamples / PullSamples 需要）
    int64_t total_dropped = 0;
    int64_t total_overwritten = 0;
    int64_t total_pulled = 0;       // 累计读取样本数，供 PullChunk 算 PTS
    int64_t write_wrap = 0;
    int64_t read_wrap = 0;

    std::mutex mtx;
    std::condition_variable not_empty_cv;  // 消费者在这等数据
    std::condition_variable not_full_cv;   // 生产者在这等空间

    WarningCallback warning_cb;

    explicit Impl(const AudioStreamBufferOptions& opts)
        : options(opts), capacity(opts.capacity_samples) {
        buffer.resize(capacity, 0.0f);
    }

    void WriteInternal(const float* data, size_t len) {
        assert(len <= capacity - current_size);

        size_t first = std::min(len, capacity - write_pos);
        std::copy(data, data + first, buffer.begin() + write_pos);

        size_t second = len - first;
        if (second > 0) {
            write_wrap++;
            std::copy(data + first, data + len, buffer.begin());
        }

        write_pos = (write_pos + len) % capacity;
        current_size += len;
    }

    void ReadInternal(std::vector<float>& out, size_t len) {
        assert(len <= current_size);
        out.resize(len);

        size_t first = std::min(len, capacity - read_pos);
        std::copy(buffer.begin() + read_pos,
                  buffer.begin() + read_pos + first, out.begin());

        size_t second = len - first;
        if (second > 0) {
            read_wrap++;
            std::copy(buffer.begin(), buffer.begin() + second,
                      out.begin() + first);
        }

        read_pos = (read_pos + len) % capacity;
        current_size -= len;
        total_pulled += len;
    }
};

// ====== 构造/析构/移动 ======
AudioStreamBuffer::AudioStreamBuffer(const AudioStreamBufferOptions& options)
    : pImpl_(std::make_unique<Impl>(options)) {}

AudioStreamBuffer::AudioStreamBuffer(size_t capacity_samples)
    : pImpl_(std::make_unique<Impl>(
          AudioStreamBufferOptions{capacity_samples})) {}

AudioStreamBuffer::~AudioStreamBuffer() = default;
AudioStreamBuffer::AudioStreamBuffer(AudioStreamBuffer&&) noexcept = default;
AudioStreamBuffer& AudioStreamBuffer::operator=(AudioStreamBuffer&&) noexcept = default;

// ====== 静态工具 ======

AudioStreamBufferOptions AudioStreamBuffer::FromDuration(int sample_rate, double duration_ms) {
    AudioStreamBufferOptions opts;
    opts.sample_rate = sample_rate;
    opts.capacity_samples = static_cast<size_t>(
        sample_rate * duration_ms / 1000.0);
    return opts;
}

std::string AudioStreamBuffer::StatusToString(AudioStreamBufferStatus status) {
    switch (status) {
        case AudioStreamBufferStatus::kOk:                      return "成功";
        case AudioStreamBufferStatus::kInvalidCapacity:         return "无效的容量";
        case AudioStreamBufferStatus::kInvalidSampleRate:       return "无效的采样率";
        case AudioStreamBufferStatus::kInvalidDuration:         return "无效的时长";
        case AudioStreamBufferStatus::kEmptyInput:              return "输入数据为空";
        case AudioStreamBufferStatus::kInvalidPcmData:          return "PCM 包含 NaN 或 Inf";
        case AudioStreamBufferStatus::kInvalidChunk:            return "无效的音频块";
        case AudioStreamBufferStatus::kInvalidReadSize:         return "无效的读取大小";
        case AudioStreamBufferStatus::kInsufficientData:        return "数据不足";
        case AudioStreamBufferStatus::kInsufficientSpace:       return "空间不足";
        case AudioStreamBufferStatus::kTimeout:                 return "操作超时";
        case AudioStreamBufferStatus::kWouldBlock:              return "非阻塞模式需等待";
        case AudioStreamBufferStatus::kClosed:                  return "缓冲区已关闭";
        case AudioStreamBufferStatus::kUnsupportedOperation:    return "不支持的操作";
        case AudioStreamBufferStatus::kInternalError:           return "内部错误";
        default:                                                return "未定义错误";
    }
}

// ====== 状态查询 ======

size_t AudioStreamBuffer::Size() const {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    return pImpl_->current_size;
}

size_t AudioStreamBuffer::Capacity() const {
    return pImpl_->capacity;  // 构造后不变，不需加锁
}

size_t AudioStreamBuffer::FreeSpace() const {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    return pImpl_->capacity - pImpl_->current_size;
}

double AudioStreamBuffer::Occupancy() const {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    if (pImpl_->capacity == 0) { return 0.0; }
    return static_cast<double>(pImpl_->current_size) / pImpl_->capacity;
}

AudioStreamBufferStats AudioStreamBuffer::GetStats() const {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    AudioStreamBufferStats s;
    s.capacity_samples          = pImpl_->capacity;
    s.size_samples              = pImpl_->current_size;
    s.free_samples              = pImpl_->capacity - pImpl_->current_size;
    s.occupancy                 = pImpl_->capacity == 0 ? 0.0
        : static_cast<double>(pImpl_->current_size) / pImpl_->capacity;
    s.closed                    = pImpl_->closed;
    s.total_dropped_samples     = pImpl_->total_dropped;
    s.total_overwritten_samples = pImpl_->total_overwritten;
    s.write_wrap_count          = pImpl_->write_wrap;
    s.read_wrap_count           = pImpl_->read_wrap;
    return s;
}

bool AudioStreamBuffer::IsClosed() const {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    return pImpl_->closed;
}

void AudioStreamBuffer::SetWarningCallback(WarningCallback callback) {
    pImpl_->warning_cb = std::move(callback);
}

// ====== 核心接口 stub ======

AudioStreamPushResult AudioStreamBuffer::PushSamples(const std::vector<float>& samples,
    AudioBufferOverflowStrategy strategy, int timeout_ms) {

    AudioStreamPushResult asp_res;

    // ====== 锁外：轻量输入校验（不改共享数据，不用拿锁） ======
    if (samples.empty()) {
        asp_res = MakePushError(AudioStreamBufferStatus::kEmptyInput);
        return asp_res;
    }
    if (pImpl_->options.validate_finite) {
        for (auto v : samples) {
            if (std::isnan(v) || std::isinf(v)) {
                asp_res = MakePushError(AudioStreamBufferStatus::kInvalidPcmData);
                return asp_res;
            }
        }
    }

    // ====== 拿锁：以下所有操作涉及 read_pos/write_pos/current_size，需要互斥 ======
    std::unique_lock<std::mutex> lock(pImpl_->mtx);

    // 拿锁后重新确认 closed（锁外读到 false，进来时可能已被关）
    if (pImpl_->closed) {
        asp_res = MakePushError(AudioStreamBufferStatus::kClosed);
        return asp_res;
    }

    asp_res.requested_samples = samples.size();

    // 空闲空间足够 → 直接写入
    if (pImpl_->capacity - pImpl_->current_size >= samples.size()) {
        pImpl_->WriteInternal(samples.data(), samples.size());
        asp_res.pushed_samples = samples.size();
    } else {
        // 空闲不够 → 根据溢出策略处理
        if (strategy == AudioBufferOverflowStrategy::kDropNewest) {
            // 丢弃新数据：一个不写，记录丢弃量 + 告警
            asp_res.pushed_samples = 0;
            asp_res.dropped_samples = samples.size();
            pImpl_->total_dropped += samples.size();
            asp_res.status = AudioStreamBufferStatus::kInsufficientSpace;
            if (pImpl_->warning_cb) {
                pImpl_->warning_cb(static_cast<double>(pImpl_->current_size) / pImpl_->capacity, samples.size());
            }
            return asp_res;
        }

        else if (strategy == AudioBufferOverflowStrategy::kOverwriteOldest) {
            // 覆盖旧数据：前移读指针腾空间，记录覆盖量 + 告警
            size_t needed = samples.size() - (pImpl_->capacity - pImpl_->current_size);
            pImpl_->read_pos = (pImpl_->read_pos + needed) % pImpl_->capacity;
            pImpl_->current_size -= needed;
            pImpl_->total_overwritten += needed;
            asp_res.overwritten_samples = needed;
            if (pImpl_->warning_cb) {
                pImpl_->warning_cb(static_cast<double>(pImpl_->current_size) / pImpl_->capacity, needed);
            }
            pImpl_->WriteInternal(samples.data(), samples.size());
            asp_res.pushed_samples = samples.size();
        } else if (strategy == AudioBufferOverflowStrategy::kBlock) {
            // 阻塞等待：睡着，等消费者 pull 后叫醒，或 Close 后唤醒
            bool space_ok = false;
            if (timeout_ms > 0) {
                // 带超时的等待
                space_ok = pImpl_->not_full_cv.wait_for(
                    lock, std::chrono::milliseconds(timeout_ms),
                    [&] { return (pImpl_->capacity - pImpl_->current_size >= samples.size()) || pImpl_->closed; });
                if (!space_ok) {
                    asp_res = MakePushError(AudioStreamBufferStatus::kTimeout);
                    return asp_res;
                }
            } else if (timeout_ms == 0) {
                // 永久阻塞：一直等到有空间或被关闭
                pImpl_->not_full_cv.wait(
                    lock, [&] { return (pImpl_->capacity - pImpl_->current_size >= samples.size()) || pImpl_->closed; });
            } else {
                // timeout_ms < 0：非阻塞，立即返回
                asp_res = MakePushError(AudioStreamBufferStatus::kWouldBlock);
                return asp_res;
            }
            // 醒来后检查是不是因为 Close 被唤醒
            if (pImpl_->closed) {
                asp_res = MakePushError(AudioStreamBufferStatus::kClosed);
                return asp_res;
            }
            // 空间够了，写入
            pImpl_->WriteInternal(samples.data(), samples.size());
            asp_res.pushed_samples = samples.size();
        } else {
            asp_res = MakePushError(AudioStreamBufferStatus::kUnsupportedOperation);
            return asp_res;
        }
    }

    // 写入完成后，叫醒一个正在等数据的消费者（PullSamples 可能在 wait）
    pImpl_->not_empty_cv.notify_one();

    // 填充 stats（直接访问 pImpl_，不再调 GetStats 导致死锁）
    auto& s = asp_res.stats;
    s.capacity_samples = pImpl_->capacity;
    s.size_samples = pImpl_->current_size;
    s.free_samples = pImpl_->capacity - pImpl_->current_size;
    s.occupancy = pImpl_->capacity == 0 ? 0.0
        : static_cast<double>(pImpl_->current_size) / pImpl_->capacity;
    s.closed = pImpl_->closed;
    s.total_dropped_samples = pImpl_->total_dropped;
    s.total_overwritten_samples = pImpl_->total_overwritten;
    s.write_wrap_count = pImpl_->write_wrap;
    s.read_wrap_count = pImpl_->read_wrap;

    asp_res.success = true;
    asp_res.status = AudioStreamBufferStatus::kOk;
    return asp_res;
}

AudioStreamPullResult AudioStreamBuffer::PullSamples(size_t sample_count, int timeout_ms) {

    AudioStreamPullResult asp_res;

    // ====== 锁外：轻量校验 ======
    if (sample_count == 0) {
        asp_res = MakePullError(AudioStreamBufferStatus::kInvalidReadSize);
        return asp_res;
    }

    // ====== 拿锁 ======
    std::unique_lock<std::mutex> lock(pImpl_->mtx);

    asp_res.requested_samples = sample_count;

    // 数据不够 → 需要等生产者 push（或等 Close）
    if (sample_count > pImpl_->current_size) {
        if (pImpl_->closed) {
            asp_res = MakePullError(AudioStreamBufferStatus::kClosed);
            return asp_res;
        }

        bool data_ok = false;
        if (timeout_ms > 0) {
            data_ok = pImpl_->not_empty_cv.wait_for(
                lock, std::chrono::milliseconds(timeout_ms),
                [&] { return pImpl_->current_size >= sample_count || pImpl_->closed; });
            if (!data_ok) {
                asp_res = MakePullError(AudioStreamBufferStatus::kTimeout);
                return asp_res;
            }
        } else if (timeout_ms == 0) {
            pImpl_->not_empty_cv.wait(
                lock, [&] { return pImpl_->current_size >= sample_count || pImpl_->closed; });
        } else {
            asp_res = MakePullError(AudioStreamBufferStatus::kInsufficientData);
            return asp_res;
        }

        // 醒来后检查是否是 Close 唤醒
        if (pImpl_->closed && pImpl_->current_size < sample_count) {
            asp_res = MakePullError(AudioStreamBufferStatus::kClosed);
            return asp_res;
        }
    }

    // 起始索引必须在同一把锁内读取，避免并发消费者推进累计计数。
    const int64_t start_sample_index = pImpl_->total_pulled;

    // 数据够了，读取
    pImpl_->ReadInternal(asp_res.pcm, sample_count);
    asp_res.pulled_samples = sample_count;
    // PullChunk 通过同一个结果取得与 PCM 对应的时间 metadata。
    asp_res.chunk.start_sample_index = start_sample_index;
    asp_res.chunk.start_pts_ms =
        static_cast<double>(start_sample_index) /
        pImpl_->options.sample_rate * 1000.0;
    asp_res.chunk.sample_rate = pImpl_->options.sample_rate;
    asp_res.chunk.channels = 1;

    // 叫醒一个正在等空间的生产者（PushSamples 可能在 Block 等待）
    pImpl_->not_full_cv.notify_one();

    auto& sl = asp_res.stats;
    sl.capacity_samples = pImpl_->capacity;
    sl.size_samples = pImpl_->current_size;
    sl.free_samples = pImpl_->capacity - pImpl_->current_size;
    sl.occupancy = pImpl_->capacity == 0 ? 0.0
        : static_cast<double>(pImpl_->current_size) / pImpl_->capacity;
    sl.closed = pImpl_->closed;
    sl.total_dropped_samples = pImpl_->total_dropped;
    sl.total_overwritten_samples = pImpl_->total_overwritten;
    sl.write_wrap_count = pImpl_->write_wrap;
    sl.read_wrap_count = pImpl_->read_wrap;

    asp_res.success = true;
    asp_res.status = AudioStreamBufferStatus::kOk;
    return asp_res;
}

AudioStreamPushResult AudioStreamBuffer::PushChunk(const AudioChunk& chunk,
    AudioBufferOverflowStrategy strategy, int timeout_ms) {
    // Chunk 输入校验后转发给 PushSamples
    if (chunk.pcm.empty()) {
        return MakePushError(AudioStreamBufferStatus::kInvalidChunk);
    }
    if (chunk.sample_rate <= 0) {
        return MakePushError(AudioStreamBufferStatus::kInvalidSampleRate);
    }
    return PushSamples(chunk.pcm, strategy, timeout_ms);
}

AudioStreamPullResult AudioStreamBuffer::PullChunk(size_t sample_count, int timeout_ms) {
    // 从 PullSamples 读取后包装为 AudioChunk，填上 PTS
    auto result = PullSamples(sample_count, timeout_ms);
    if (!result.success) {
        // 错误路径：直接把 PullSamples 的结果转成 PullResult 返回
        AudioStreamPullResult r;
        r.success = result.success;
        r.status = result.status;
        r.error_message = result.error_message;
        r.requested_samples = result.requested_samples;
        r.stats = result.stats;
        return r;
    }

    AudioStreamPullResult r;
    r.success = true;
    r.status = AudioStreamBufferStatus::kOk;
    r.requested_samples = sample_count;
    r.pulled_samples = result.pulled_samples;
    r.stats = result.stats;

    // metadata 已在读取 PCM 的同一锁内捕获；这里仅组装公开 AudioChunk。
    r.chunk = std::move(result.chunk);
    r.chunk.pcm = std::move(result.pcm);

    return r;
}

void AudioStreamBuffer::Close() {
    {
        std::lock_guard<std::mutex> lock(pImpl_->mtx);
        pImpl_->closed = true;
    }
    // 唤醒所有正在 wait 的生产者和消费者，让它们看到 closed 后退出
    pImpl_->not_empty_cv.notify_all();
    pImpl_->not_full_cv.notify_all();
}

void AudioStreamBuffer::Reset() {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    pImpl_->read_pos = 0;
    pImpl_->write_pos = 0;
    pImpl_->current_size = 0;
    pImpl_->total_dropped = 0;
    pImpl_->total_overwritten = 0;
    pImpl_->total_pulled = 0;
    pImpl_->write_wrap = 0;
    pImpl_->read_wrap = 0;
}

void AudioStreamBuffer::Clear() {
    std::lock_guard<std::mutex> lock(pImpl_->mtx);
    pImpl_->read_pos = 0;
    pImpl_->write_pos = 0;
    pImpl_->current_size = 0;
}

}   // namespace audio
}   // namespace digital_human
