#pragma once

#include <cstddef>
#include <cstdint>

// ============================================================================
// Wav2LipModelSpec — Wav2Lip 模型常量规格（内部文件，不对外暴露）
// ============================================================================
//
// 是什么：
//   本文件的唯一职责是集中定义 Wav2Lip 模型的所有硬编码常量（blob 名、shape、
//   归一化因子等）。src/model/ 下的所有 .cpp 从这里取常量，不自己写裸数字。
//
// 作用：
//   1. 单一真相来源 — 96、80、16、6、"mel"等只在这里出现一次，修改一处全局生效。
//   2. 防止"常量散落" — 任何 .cpp 中出现裸数字都能在 CR 时被揪出来。
//   3. 换模型时改口小 — 从 Wav2Lip 换成其他模型，只需改这一个文件。
//
// 边界：
//   - 本文件放在 src/model/detail/ 而非 include/model/，表示它属于内部实现细节，
//     不是 SDK 的公开 API。外部使用者不应 #include 本文件。
//   - 本文件只定义编译期常量（constexpr），不包含任何运行时逻辑、函数或类方法。
//   - 如果未来支持多模型，新增的模型常量应另建对应的 *_model_spec.h，不在此文件中
//     混入多套配置。
// ============================================================================
namespace digital_human::model::detail {

struct Wav2LipModelSpec {
    // ---- 模型 blob 名（来自 wav2lip.param 的真实 Input/输出名） ----
    // 不能从截图或旧文档抄，必须以当前 .param 为准。
    static constexpr const char* kInputMel   = "mel";   // 音频 Mel 谱输入
    static constexpr const char* kInputFace  = "face";  // 六通道人脸输入
    static constexpr const char* kOutputPred = "pred";  // 模型预测输出（96×96 BGR 图像）

    // ---- 人脸输入规格 ----
    // 上游 FaceAligner 交付：CV_8UC3 BGR [0,255] 96×96 → Builder 转为 6 通道 float [0,1]。
    static constexpr int kFaceWidth       = 96;   // 对齐人脸宽度
    static constexpr int kFaceHeight      = 96;   // 对齐人脸高度
    static constexpr int kFaceChannels    = 6;    // 前 3 通道 = masked BGR，后 3 通道 = original BGR
    static constexpr int kFaceMaskStartRow = 48;  // 下半脸起始行（height/2），Wav2Lip 训练约定
    static constexpr float kImageNormalizeScale = 1.0f / 255.0f;  // 归一化因子：[0,255] → [0,1]

    // ---- 音频 Mel 输入规格 ----
    // 上游 MelFeatureExtractor 交付：1280 float freq-major → 最终 ncnn::Mat(16,80,1)。
    static constexpr int kMelBins      = 80;    // Mel 频率 bin 数
    static constexpr int kMelFrames    = 16;    // 时间帧数
    static constexpr int kMelChunkSize = kMelBins * kMelFrames;  // = 1280

    // ---- 模型输出规格 ----
    // Extractor::extract("pred") 应产出 ncnn::Mat(96,96,3)。
    static constexpr int kPredChannels = 3;
    static constexpr int kPredWidth    = 96;
    static constexpr int kPredHeight   = 96;
    static constexpr int kTensorDims = 3;
    static constexpr int kUnpackedElementPack = 1;
    static constexpr std::size_t kUnpackedFp32ElementSize = sizeof(float);
    static constexpr float kPredValueMin = 0.0f;
    static constexpr float kPredValueMax = 1.0f;
    static constexpr float kPredRangeTolerance = 1e-6f;
    static constexpr float kImageQuantizationScale = 255.0f;
    static constexpr int kBluePlaneIndex = 0;
    static constexpr int kGreenPlaneIndex = 1;
    static constexpr int kRedPlaneIndex = 2;
};

}   // namespace digital_human::model::detail
