/// @file ncnn_input_adapter.cpp
/// @brief NcnnInputAdapter 实现：float buffer → ncnn::Mat 的逐通道逐行复制
///
/// 这是三兄弟中代码最少、职责最窄的模块——只做内存映射，不做推理、不创建 Extractor。
///
/// 核心操作：
///   人脸：face_chw[c*9216 + y*96 + x] → face_mat.channel(c)[y*96 + x]
///       6 个通道 × 96×96，逐通道 memcpy（通道间不连续，通道内连续）
///   Mel：mel_freq_time[freq*16 + time] → mel_mat.row(freq)[time]
///       80 行 × 16 列，逐行 memcpy
///
/// 为什么不能一个 memcpy 覆盖整个 Mat？
///   ncnn::Mat 的 channel(i) 和 channel(i+1) 之间不保证内存连续。
///   各通道可能是独立分配的，通道间有间隙。
///   但单通道内部（9216 个 float）和单行内部（16 个 float）是连续的，
///   所以逐通道/逐行使用 memcpy 是安全的，且比手写 for 循环快。
///
/// 存在意义：隔离 ncnn 依赖。
///   Builder 产出纯 float buffer，不依赖 ncnn。
///   换推理框架（如 ONNX）→ 只写新 Adapter，Builder 不动。

#include "model/ncnn_input_adapter.h"
#include "detail/wav2lip_model_spec.h"

#include <chrono>
#include <cmath>
#include <cstring>   // std::memcpy

namespace digital_human {
namespace model {
namespace {

using namespace detail;
using namespace std::chrono;

/// 统一构造失败 Result
NcnnInputResult MakeAdapterError(ModelInputStatus status,
                                  std::string msg,
                                  double time_ms) {
    NcnnInputResult result;
    result.success = false;
    result.status = status;
    result.error_message = std::move(msg);
    result.time_ms = time_ms;
    return result;
}

/// 检查后端无关的 float buffer 是否全部为有限值。
/// 边界：Adapter 可能被测试或其他调用方直接使用，不能假设输入一定来自 Builder。
bool AllFinite(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

} // anonymous namespace

using namespace std::chrono;

// ============================================================================
// Adapt — float buffer → ncnn::Mat
// ============================================================================

NcnnInputResult NcnnInputAdapter::Adapt(const Wav2LipInputData& data) const {
    using Spec = Wav2LipModelSpec;
    auto t_start = steady_clock::now();

    auto elapsed_ms = [&t_start]() -> double {
        auto t_now = steady_clock::now();
        return static_cast<double>(
            duration_cast<microseconds>(t_now - t_start).count()) / 1000.0;
    };

    const int HW = Spec::kFaceHeight * Spec::kFaceWidth;  // 9216

    // ============ 校验输入 buffer 尺寸 ============
    // Adapter 不能假设调用方一定是 Builder（测试代码可能构造假的 Wav2LipInputData），
    // 所以仍然要做基本校验。

    if (data.face_chw.size() != static_cast<size_t>(Spec::kFaceChannels * HW)) {
        return MakeAdapterError(
            ModelInputStatus::kInvalidFaceSize,
            "face_chw size mismatch: expected " +
                std::to_string(Spec::kFaceChannels * HW) +
                ", got " + std::to_string(data.face_chw.size()),
            elapsed_ms());
    }

    if (data.mel_freq_time.size() != static_cast<size_t>(Spec::kMelChunkSize)) {
        return MakeAdapterError(
            ModelInputStatus::kInvalidMelChunkSize,
            "mel size mismatch: expected " +
                std::to_string(Spec::kMelChunkSize) +
                ", got " + std::to_string(data.mel_freq_time.size()),
            elapsed_ms());
    }

    // ============ 校验数值有限性 ============
    // face/mel 只要含一个 NaN 或 Inf，memcpy 虽然能成功，模型输出却会被污染。
    // 因此必须在分配 ncnn::Mat 前拒绝，失败结果中也不会留下半成品 tensor。
    if (!AllFinite(data.face_chw)) {
        return MakeAdapterError(
            ModelInputStatus::kNonFiniteFaceValue,
            "face_chw contains NaN or Inf",
            elapsed_ms());
    }

    if (!AllFinite(data.mel_freq_time)) {
        return MakeAdapterError(
            ModelInputStatus::kNonFiniteMelValue,
            "mel_freq_time contains NaN or Inf",
            elapsed_ms());
    }

    // ============ 分配 ncnn::Mat ============
    // ncnn::Mat(width, height, channels, elem_size)
    // elem_size = 4u 表示每个元素 4 字节 = sizeof(float)

    // 人脸：(w=96, h=96, c=6)，每通道是 96×96 的二维 float 平面
    ncnn::Mat face_mat(Spec::kFaceWidth, Spec::kFaceHeight, Spec::kFaceChannels,
                       static_cast<size_t>(4u));
    if (face_mat.empty()) {
        return MakeAdapterError(ModelInputStatus::kAllocationFailed,
                                "failed to allocate face ncnn::Mat(96,96,6)",
                                elapsed_ms());
    }

    // Mel：(w=16, h=80, c=1)，80 行 × 16 列，单通道
    ncnn::Mat mel_mat(Spec::kMelFrames, Spec::kMelBins, 1,
                      static_cast<size_t>(4u));
    if (mel_mat.empty()) {
        return MakeAdapterError(ModelInputStatus::kAllocationFailed,
                                "failed to allocate mel ncnn::Mat(16,80,1)",
                                elapsed_ms());
    }

    // ============ 逐通道复制人脸 ============
    // face_chw 是连续的 vector<float>：[ch0 的 9216 个][ch1 的 9216 个]...
    // ncnn::Mat.channel(c) 返回第 c 个通道的首指针。
    // channel(c) 和 channel(c+1) 之间不保证连续 → 必须逐通道 memcpy。
    // 同一通道内的 9216 个 float 是连续的（ncnn 保证）→ memcpy 安全。
    for (int c = 0; c < Spec::kFaceChannels; ++c) {
        const float* src = data.face_chw.data() + c * HW;
        float* dst = face_mat.channel(c);
        std::memcpy(dst, src, HW * sizeof(float));
    }

    // ============ 逐行复制 Mel ============
    // mel_freq_time 是 freq-major：[f0_t0..f0_t15][f1_t0..f1_t15]...
    // ncnn::Mat.row(freq) 返回第 freq 行的首指针（16 个连续 float）。
    // row(freq) 和 row(freq+1) 之间不保证连续 → 必须逐行 memcpy。
    // 同一行内的 16 个 float 是连续的 → memcpy 安全。
    //
    // 验证手段：测试中构造 mel[f*16+t]=1000*f+t，然后检查
    //   mel_mat.row(7)[3] == 7003 → 布局正确（freq-major）
    //   mel_mat.row(7)[3] == 3007 → 布局反了（time-major 被当 freq-major）
    for (int freq = 0; freq < Spec::kMelBins; ++freq) {
        const float* src = data.mel_freq_time.data() + freq * Spec::kMelFrames;
        float* dst = mel_mat.row(freq);
        std::memcpy(dst, src, Spec::kMelFrames * sizeof(float));
    }

    // ============ 组装结果 ============

    NcnnInputResult result;
    result.success = true;
    result.status = ModelInputStatus::kOk;
    result.input.face = face_mat;
    result.input.mel  = mel_mat;
    // metadata 原样透传，不在此模块计算或修改
    result.input.metadata = data.metadata;
    result.time_ms = elapsed_ms();

    return result;
}

} // namespace model
} // namespace digital_human
