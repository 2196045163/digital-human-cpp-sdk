#pragma once

#include <memory>
#include <string>
#include <mat.h>

#include "model/input_processor.h"   // Wav2LipInputData, ModelInputMetadata, ModelInputStatus

namespace digital_human {
namespace model {

/// @brief Adapter 产出的 ncnn 格式输入，可直接喂给 ncnn::Extractor
/// @note  mel 布局 (w=16,h=80,c=1)，face 布局 (w=96,h=96,c=6)
struct NcnnWav2LipInput {
    ncnn::Mat mel;                   ///< 音频 Mel 输入，row(freq)[time]
    ncnn::Mat face;                  ///< 六通道人脸输入，channel(c)[y*w+x]
    ModelInputMetadata metadata;     ///< 透传的时间元数据
};

/// @brief Adapter 构建结果
struct NcnnInputResult {
    bool success = false;                                        ///< 是否成功（默认失败）
    ModelInputStatus status = ModelInputStatus::kUnknownError;   ///< 具体状态码
    std::string error_message;                                   ///< 人类可读的错误描述
    NcnnWav2LipInput input;                                      ///< ncnn 格式输入
    double time_ms = 0.0;                                        ///< 适配耗时（毫秒）
};

/// @brief ncnn 输入适配器（Ncnn Input Adapter）
///
/// 把后端无关的 Wav2LipInputData（float buffer）按 ncnn 的 (w,h,c) 维度复制到
/// ncnn::Mat 中。是三兄弟中最窄的模块——只做内存映射，不做任何语义转换。
///
/// 核心设计：
/// - 窄接口、单一职责：输入 float buffer → 输出 ncnn::Mat。不做推理、不创建
///   Extractor、不调用 input()/extract()（那是 ModelInference 的事）。
/// - 隔离 ncnn 依赖：Builder 不依赖 ncnn，换框架只需换 Adapter。
/// - 独立守门：不能假设调用方一定经过 Builder，因此再次校验两个 buffer 的长度和
///   NaN/Inf；失败时返回空 Mat，不留下可被误用的半成品。
/// - 逐通道逐行复制：ncnn::Mat 各通道之间不保证连续，不能 memcpy 一把梭。
///
/// 验证手段：
/// - Mel：构造 mel[f*16+t]=1000*f+t，检查 row(7)[3]==7003 锁定 freq-major 布局
/// - 人脸：构造不对称 BGR 像素，检查 channel 首/中/尾位置的精确值
class NcnnInputAdapter {
public:
    /// @brief 将语义数据适配为 ncnn 格式
    /// @param data 来自 Builder 的 Wav2LipInputData（或测试用合成数据）
    /// @return NcnnInputResult，成功时 mel(16,80,1)、face(96,96,6)
    /// @note  只校验并复制，不修正范围、不改变布局、不执行模型推理
    NcnnInputResult Adapt(const Wav2LipInputData& data) const;
};

} // namespace model
} // namespace digital_human
