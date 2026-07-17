/// @file input_processor.cpp
/// @brief Wav2LipInputBuilder 实现：人脸六通道构建 + Mel 校验透传
///
/// 图像转换四步（cv::Mat → float buffer）：
///   1. 校验：empty? 尺寸不对? 类型不对? → 不合法直接拒绝（Fail Fast）
///   2. 构造 masked 人脸：clone 原图，下半部分 (y≥48) BGR 置 (0,0,0)
///   3.交织→平面：cv::Mat 的 BGRBGR... 存储 → 6 个独立的通道平面
///      - channel 0-2: masked B/G/R（下半脸是零，告诉模型"这里需要生成"）
///      - channel 3-5: original B/G/R（完整人脸，告诉模型肤色/光照/身份）
///   4. 归一化：所有值 × (1/255)，uint8 [0,255] → float [0,1]
///
/// Mel 处理（只校验不修改）：
///   - 检查 size == 1280（80 bin × 16 帧）
///   - 检查全部 finite（无 NaN/Inf）
///   - 原样复制到 Wav2LipInputData.mel_freq_time
///
/// 设计理由：
///   - 不做 resize/颜色转换：上游出错应立即暴露，不在本模块掩盖（Fail Fast）
///   - 不依赖 ncnn：产出纯 float buffer，换推理框架只需换 Adapter
///   - 下半脸一刀切 (y≥48)：Builder 不知道嘴的具体位置（那是 FaceAligner 的活）

#include "model/input_processor.h"
#include "detail/wav2lip_model_spec.h"

#include <opencv2/core.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <exception>
#include <new>

namespace digital_human {
namespace model {
namespace {

using namespace detail;
using namespace std::chrono;

// ============================================================================
// 内部 helper
// ============================================================================

/// 统一构造失败 Result，避免每个错误分支重复填充字段
Wav2LipInputResult MakeInputError(ModelInputStatus status,
                                   std::string msg,
                                   double time_ms) {
    Wav2LipInputResult result;
    result.success = false;
    result.status = status;
    result.error_message = std::move(msg);
    result.time_ms = time_ms;
    return result;
}

/// 检查 vector 中是否所有值都是有限值（非 NaN、非 Inf）
/// Mel 谱如果出现 NaN/Inf，后续推理会直接产出 NaN → 必须提前拦截
bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) { return false; }
    }
    return true;
}

/// 构造 Wav2LipInputInfo 诊断信息
/// 把 12 行字段填充抽成独立函数，保持 Build() 主流程干净
Wav2LipInputInfo MakeInputInfo(const std::vector<float>& face_chw,
                                const std::vector<float>& mel_chunk) {
    using Spec = Wav2LipModelSpec;
    Wav2LipInputInfo info;
    // 静态 shape 信息（编译期常量，来自 Wav2LipModelSpec）
    info.face_channels  = Spec::kFaceChannels;
    info.face_height    = Spec::kFaceHeight;
    info.face_width     = Spec::kFaceWidth;
    info.mel_bins       = Spec::kMelBins;
    info.mel_frames     = Spec::kMelFrames;
    info.mask_start_row = Spec::kFaceMaskStartRow;

    // 动态范围信息（一次遍历同时拿 min 和 max）
    auto [face_min, face_max] = std::minmax_element(face_chw.begin(), face_chw.end());
    auto [mel_min,  mel_max]  = std::minmax_element(mel_chunk.begin(), mel_chunk.end());
    info.face_min_value = *face_min;
    info.face_max_value = *face_max;
    info.mel_min_value  = *mel_min;
    info.mel_max_value  = *mel_max;
    info.has_nan_or_inf = false;  // Builder 已经校验过 finite，这里始终 false
    return info;
}

} // anonymous namespace

using namespace std::chrono;

// ============================================================================
// Build — 主构建方法
// ============================================================================

Wav2LipInputResult Wav2LipInputBuilder::Build(
    const cv::Mat& aligned_face,
    const std::vector<float>& freq_major_mel_chunk,
    const ModelInputMetadata& metadata) const {

    using Spec = Wav2LipModelSpec;
    auto t_start = steady_clock::now();

    // 耗时 lambda：从函数入口到调用时的毫秒数
    auto elapsed_ms = [&t_start]() -> double {
        auto t_now = steady_clock::now();
        return static_cast<double>(
            duration_cast<microseconds>(t_now - t_start).count()) / 1000.0;
    };
    try {
    // Build 对外返回 Result，因此分配和 OpenCV 实现异常也要在本接口边界内转成状态码。
    // 正常输入错误仍在最前面 Fail Fast，不进入昂贵的 clone 和 buffer 分配。

    // ============ ① 校验人脸图像（Fail Fast：不合法就拒绝，不修正）============

    if (aligned_face.empty()) {
        return MakeInputError(ModelInputStatus::kEmptyAlignedFace,
                              "aligned_face is empty", elapsed_ms());
    }

    // 尺寸必须严格 96×96 — 绝不 resize。
    // 如果上游 FaceAligner 给错了尺寸，本模块拒绝比悄悄修正更有助于定位 bug。
    if (aligned_face.rows != Spec::kFaceHeight || aligned_face.cols != Spec::kFaceWidth) {
        return MakeInputError(
            ModelInputStatus::kInvalidFaceSize,
            "expected " + std::to_string(Spec::kFaceWidth) + "x" +
                std::to_string(Spec::kFaceHeight) + ", got " +
                std::to_string(aligned_face.cols) + "x" +
                std::to_string(aligned_face.rows) + " — will not resize",
            elapsed_ms());
    }

    // 类型必须 CV_8UC3 — 绝不 convert。
    // 灰度 (CV_8UC1)、BGRA (CV_8UC4)、float (CV_32FC3) 都拒绝。
    if (aligned_face.type() != CV_8UC3) {
        return MakeInputError(
            ModelInputStatus::kInvalidFaceType,
            "expected CV_8UC3, got type " + std::to_string(aligned_face.type()) +
                " — will not convert",
            elapsed_ms());
    }

    // ============ ② 校验 Mel chunk ============

    // Wav2Lip 严格约定：1280 = 80 频率 bin × 16 时间帧，freq-major 排列
    if (freq_major_mel_chunk.size() != static_cast<size_t>(Spec::kMelChunkSize)) {
        return MakeInputError(
            ModelInputStatus::kInvalidMelChunkSize,
            "expected " + std::to_string(Spec::kMelChunkSize) + " mel values, got " +
                std::to_string(freq_major_mel_chunk.size()),
            elapsed_ms());
    }

    // Mel 含 NaN/Inf → 后续推理结果全 NaN → 必须提前拦截
    if (!AllFinite(freq_major_mel_chunk)) {
        return MakeInputError(ModelInputStatus::kNonFiniteMelValue,
                              "mel chunk contains NaN or Inf", elapsed_ms());
    }

    // ============ ③ 构建六通道人脸 ============

    const int HW = Spec::kFaceHeight * Spec::kFaceWidth;    // 96×96 = 9216
    const int total = Spec::kFaceChannels * HW;              // 6×9216 = 55296
    std::vector<float> face_chw(total, 0.0f);

    // ③a. 构造 masked 人脸 — 下半脸 BGR 全部置零
    //
    // OpenCV ROI 操作：masked(cv::Rect(x,y,w,h)) 返回子矩阵引用，
    // setTo(Scalar(0,0,0)) 直接修改原矩阵的对应区域，高效无额外拷贝。
    cv::Mat masked = aligned_face.clone();
    const int mask_start = Spec::kFaceMaskStartRow;          // 48 = height/2
    cv::Rect lower_half(0, mask_start, Spec::kFaceWidth,
                        Spec::kFaceHeight - mask_start);     // 宽 96, 高 48
    masked(lower_half).setTo(cv::Scalar(0, 0, 0));

    const float scale = Spec::kImageNormalizeScale;          // 1.0f / 255.0f

    // ③b. 逐像素：交织 BGR → 平面 CHW
    //
    // OpenCV 存储格式（像素交织）：[B0,G0,R0, B1,G1,R1, B2,G2,R2, ...]
    //   - ptr<uchar>(y) 返回第 y 行首地址
    //   - 像素 (x,y) 的 B 在 row_ptr[x*3+0], G 在 [x*3+1], R 在 [x*3+2]
    //
    // 目标格式（通道平面 CHW）：channel(c)[y * width + x]
    //   - channel 0-2: masked B/G/R
    //   - channel 3-5: original B/G/R
    //
    // 用 ptr<> 而非 at<> 避免每次访问做边界检查，性能差异在 96×96×6 级别不明显
    // 但这是良好的 C++ 习惯。
    for (int y = 0; y < Spec::kFaceHeight; ++y) {
        const uchar* row_orig   = aligned_face.ptr<uchar>(y);
        const uchar* row_masked = masked.ptr<uchar>(y);
        for (int x = 0; x < Spec::kFaceWidth; ++x) {
            int px = x * 3;  // 像素 (x,y) 在行内存中的起始偏移（B 分量）

            // masked BGR → channel 0,1,2
            face_chw[0 * HW + y * Spec::kFaceWidth + x] =
                static_cast<float>(row_masked[px + 0]) * scale;  // B
            face_chw[1 * HW + y * Spec::kFaceWidth + x] =
                static_cast<float>(row_masked[px + 1]) * scale;  // G
            face_chw[2 * HW + y * Spec::kFaceWidth + x] =
                static_cast<float>(row_masked[px + 2]) * scale;  // R

            // original BGR → channel 3,4,5
            face_chw[3 * HW + y * Spec::kFaceWidth + x] =
                static_cast<float>(row_orig[px + 0]) * scale;    // B
            face_chw[4 * HW + y * Spec::kFaceWidth + x] =
                static_cast<float>(row_orig[px + 1]) * scale;    // G
            face_chw[5 * HW + y * Spec::kFaceWidth + x] =
                static_cast<float>(row_orig[px + 2]) * scale;    // R
        }
    }

    // ============ ④ 组装成功结果 ============

    Wav2LipInputResult result;
    result.success = true;
    result.status = ModelInputStatus::kOk;
    // std::move — face_chw 是 55296 个 float（约 220KB），移动比拷贝高效
    result.data.face_chw = std::move(face_chw);
    // Mel 和 metadata 只做拷贝透传，不做数值变换
    result.data.mel_freq_time = freq_major_mel_chunk;
    result.data.metadata = metadata;
    // info 由 helper 统一填充
    result.info = MakeInputInfo(result.data.face_chw, result.data.mel_freq_time);
    result.time_ms = elapsed_ms();

    return result;
    } catch (const std::bad_alloc& ex) {
        // face_chw、Mel 拷贝等标准容器分配失败，映射为明确的内存状态。
        return MakeInputError(ModelInputStatus::kAllocationFailed,
                              ex.what(), elapsed_ms());
    } catch (const cv::Exception& ex) {
        // 输入已完成类型/尺寸校验后，clone/ROI 的 OpenCV 异常属于构建阶段失败。
        return MakeInputError(ModelInputStatus::kAllocationFailed,
                              ex.what(), elapsed_ms());
    } catch (const std::exception& ex) {
        // 其余标准异常不伪装成某个输入契约错误，统一走未知错误兜底。
        return MakeInputError(ModelInputStatus::kUnknownError,
                              ex.what(), elapsed_ms());
    } catch (...) {
        return MakeInputError(ModelInputStatus::kUnknownError,
                              "unknown exception while building model input",
                              elapsed_ms());
    }
}

// ============================================================================
// StatusToString
// ============================================================================

std::string Wav2LipInputBuilder::StatusToString(ModelInputStatus status) {
    switch (status) {
    case ModelInputStatus::kOk:                return "成功";
    case ModelInputStatus::kEmptyAlignedFace:  return "对齐人脸为空";
    case ModelInputStatus::kInvalidFaceSize:   return "人脸尺寸不是 96×96";
    case ModelInputStatus::kInvalidFaceType:   return "人脸类型不是 CV_8UC3";
    case ModelInputStatus::kNonFiniteFaceValue:return "人脸输入包含 NaN 或 Inf";
    case ModelInputStatus::kInvalidMelChunkSize: return "Mel chunk 长度不是 1280";
    case ModelInputStatus::kInvalidMelLayout:  return "Mel 布局异常";
    case ModelInputStatus::kNonFiniteMelValue: return "Mel 包含 NaN 或 Inf";
    case ModelInputStatus::kInvalidMetadata:   return "元数据非法";
    case ModelInputStatus::kAllocationFailed:  return "内存分配失败";
    case ModelInputStatus::kUnknownError:
    default:                                   return "未知错误";
    }
}

} // namespace model
} // namespace digital_human
