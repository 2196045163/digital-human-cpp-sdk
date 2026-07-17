#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <ncnn/net.h>

namespace digital_human {
namespace model {

/// @brief 推理后端类型
enum class ModelBackend {
    kCpu,       ///< CPU 推理（必须通过）
    kVulkan     ///< Vulkan GPU 推理（本轮仅做能力检测，不发布 GPU 模型）
};

/// @brief 模型加载状态码，每个失败点对应一个明确的枚举值
/// @note  不使用 bool 返回值，让调用方能区分"路径错误/文件缺失/加载失败/预热失败"等
enum class ModelLoadStatus {
    // ---- 成功 ----
    kOk,                        ///< 加载成功

    // ---- 路径/文件校验 ----
    kEmptyParamPath,            ///< 传入的 param 路径为空
    kParamFileNotFound,         ///< .param 文件不存在
    kParamPathNotRegularFile,   ///< .param 是目录而非文件
    kBinFileNotFound,           ///< 推导出的同名 .bin 不存在
    kBinPathNotRegularFile,     ///< 推导出的 .bin 是目录而非文件

    // ---- 参数校验 ----
    kInvalidThreadCount,        ///< num_threads 小于等于 0

    // ---- 后端 ----
    kVulkanUnavailable,         ///< 预留：当前版本不执行 Vulkan 能力探测，因此不会返回
    kVulkanNotVerified,         ///< Vulkan 能力存在但本轮未验证通过（不发布 GPU 模型）

    // ---- ncnn 加载 ----
    kLoadParamFailed,           ///< ncnn::Net::load_param 返回非 0
    kLoadBinFailed,             ///< ncnn::Net::load_model 返回非 0

    // ---- warmup ----
    kWarmupInputFailed,         ///< 预热时 input("mel"/"face") 失败（blob 名不匹配）
    kWarmupExtractFailed,       ///< 预热时 extract("pred") 失败（输出 blob 名或模型内部错误）

    // ---- 运行时 ----
    kModelNotReady,             ///< 预留：当前 AcquireModel 未就绪时直接返回空 shared_ptr
    kFileSystemError,           ///< filesystem 操作抛出异常
    kUnknownError               ///< 未知错误（兜底）
};

/// @brief 加载选项，控制 ModelLoader 的加载行为
/// @note  所有字段都有默认值，调用方可只改关心的部分
struct ModelLoadOptions {
    ModelBackend backend = ModelBackend::kCpu;   ///< 推理后端，默认 CPU
    int num_threads = 1;                         ///< 线程数（必须 > 0；此版本 ncnn 不支持 num_threads=0）
    bool enable_warmup = true;                   ///< 发布前是否执行零输入预热
};

/// @brief 加载过程的诊断信息，记录本次加载实际发生了什么
struct ModelLoadInfo {
    std::filesystem::path param_path;                       ///< 请求的 .param 路径
    std::filesystem::path bin_path;                         ///< 实际使用的 .bin 路径（推导得出）
    ModelBackend requested_backend = ModelBackend::kCpu;    ///< 请求的后端
    int effective_num_threads = 0;                           ///< 实际生效的线程数
    bool warmup_performed = false;                           ///< 预热是否执行
    bool model_replaced = false;                             ///< 本次加载是否替换了旧模型
};

/// @brief 加载结果，包含成功/失败标志、状态码、诊断信息和耗时
/// @note  无论是否成功都返回此结构体，失败时 success=false、error_message 有值
struct ModelLoadResult {
    bool success = false;                                     ///< 是否加载成功（默认失败）
    ModelLoadStatus status = ModelLoadStatus::kUnknownError;  ///< 具体状态码
    std::string error_message;                                ///< 人类可读的错误描述
    ModelLoadInfo info;                                       ///< 诊断信息
    double time_ms = 0.0;                                     ///< 加载耗时（毫秒）
};

/// @brief 模型加载模块（Model Loader）
///
/// 本模块是 ncnn 推理链路中模型生命周期的唯一管理者。负责把一对 Wav2Lip
/// .param/.bin 文件安全地加载并发布为可共享的只读 CPU ncnn::Net 实例。
///
/// 核心设计：
/// - 候选发布模式：先在局部创建候选 ncnn::Net，全部验证通过后再原子替换。
///   失败不破坏旧模型，避免"先清空再加载失败"导致服务裸奔。
/// - shared_ptr<const ncnn::Net>：外部持有只读共享指针，热替换时旧模型
///   不会提前释放，防止悬空指针。
/// - 零输入 warmup：发布前用全零数据走一遍完整推理流程，提前发现 blob 名
///   或输入 shape 不匹配。
///
/// 职责边界：
/// - 只做同步加载（不做异步/热更新/文件监控/UI 回调）
/// - 只要求 CPU 后端通过（Vulkan 仅做能力检测，本轮不发布 GPU 模型）
/// - 不做推理调度、Extractor 管理、输入构建、输出处理
///
/// 使用 PImpl 模式隐藏 ncnn 实现细节，禁止拷贝。
class ModelLoader {
public:
    /// @brief 唯一加载入口
    /// @param param_path .param 文件路径（.bin 自动推导同名文件）
    /// @param options 加载选项（后端、线程数、是否预热）
    /// @return ModelLoadResult，成功时 IsReady()=true、AcquireModel() 非空
    /// @note  新模型全部验证通过后才替换旧模型；失败时旧模型保持可用
    ModelLoadResult Load(
        const std::filesystem::path& param_path,
        const ModelLoadOptions& options = ModelLoadOptions());

    /// @brief 模型是否已加载成功并可获取
    bool IsReady() const;

    /// @brief 获取只读模型快照
    /// @return 已加载模型的 shared_ptr，未加载时返回空指针
    /// @note  const 限定 + shared_ptr 保证推理方安全持有：热替换不悬空、不能修改权重
    std::shared_ptr<const ncnn::Net> AcquireModel() const;

    /// @brief 状态码 → 人类可读字符串，用于日志和 example 打印
    /// @param status 状态码
    /// @return 非空字符串（覆盖所有状态码）
    static std::string StatusToString(ModelLoadStatus status);

    // PImpl 惯用法
    ModelLoader();
    ~ModelLoader();
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

private:
    struct Impl;                      ///< 前向声明，实现在 .cpp 中隐藏 ncnn 实现细节
    std::unique_ptr<Impl> pImpl_;     ///< PImpl 惯用法
};

} // namespace model
} // namespace digital_human
