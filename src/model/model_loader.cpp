/// @file model_loader.cpp
/// @brief ModelLoader 实现：候选发布模式 + 零输入预热 + 共享指针生命周期管理
///
/// 核心流程（Load 方法）：
///   1. 校验 param_path（空？存在？是文件？）
///   2. 推导同名 .bin 路径（replace_extension(".bin")）
///   3. 校验 bin_path
///   4. 参数校验（num_threads > 0？不是 Vulkan？）
///   5. 在栈上创建候选 ncnn::Net → load_param → load_model
///   6. 可选 warmup（用 Wav2LipModelSpec 的 blob 名和零 tensor 走一遍推理）
///   7. 全部成功后，mutex 短临界区内原子替换 shared_ptr
///
/// 设计理由：
///   - "候选发布"保证 Load 失败不破坏旧模型（不先 clear 再 load）
///   - shared_ptr<const ncnn::Net> 让外部推理方安全持有旧模型快照，热替换不悬空
///   - mutex 只在交换指针时持有（毫秒级），不在加载/推理时持有（秒级）
///   - warmup 是"加载成功"和"推理可用"之间的最后一关：load_model 成功 ≠ 能推理

#include <string>
#include <filesystem>
#include <memory>
#include <mutex>
#include <chrono>
#include <exception>
#include <cstdio>


#include "model/model_loader.h"
#include "detail/wav2lip_model_spec.h" // 模型参数



namespace digital_human {
namespace model {
namespace {

    // 统一构造失败 Result
    ModelLoadResult MakeErrorResult(ModelLoadStatus status, std::string msg,
        const std::filesystem::path& param,      // 用户传的 param 路径
        const std::filesystem::path& bin,        // 推导出的 bin 路径
        const ModelLoadOptions& opts, double time_ms) {

        ModelLoadResult result;
        result.success = false;
        result.status = status;
        result.error_message = std::move(msg);
        result.info.param_path = param;
        result.info.bin_path = bin;
        result.info.requested_backend = opts.backend;
        result.info.effective_num_threads = opts.num_threads;
        result.info.warmup_performed = false;
        result.time_ms = time_ms;
        return result;
    }

    // 从 .param 推导 .bin 路径 —— 因为将 它两放一起了
    std::filesystem::path DeriveBinPath(const std::filesystem::path& param_path) {
        auto bin = param_path;
        bin.replace_extension(".bin");
        return bin;
    }

} // 匿名 namespace

    using namespace detail;

    // 统一构造成功 Result，与 MakeErrorResult 对称，避免填 info 字段的重复代码
    ModelLoadResult MakeSuccessResult(
        const std::filesystem::path& param,
        const std::filesystem::path& bin,
        const ModelLoadOptions& opts,
        bool warmup_performed,
        double time_ms) {
        ModelLoadResult result;
        result.success   = true;
        result.status    = ModelLoadStatus::kOk;
        result.info.param_path          = param;
        result.info.bin_path            = bin;
        result.info.requested_backend   = opts.backend;
        result.info.effective_num_threads = opts.num_threads;
        result.info.warmup_performed    = warmup_performed;
        // 是否替换了旧模型只能在发布临界区内判断；这里先保留默认 false。
        result.info.model_replaced      = false;
        result.time_ms = time_ms;
        return result;
    }

    // 前向声明：定义在 Load() 之后
    ModelLoadResult WarmupCandidate(ncnn::Net& candidate,
                                    const ModelLoadOptions& opts,
                                    const std::filesystem::path& param,
                                    const std::filesystem::path& bin,
                                    double load_time_ms);

    struct ModelLoader::Impl {
        ModelRuntimeSnapshot current_snapshot_;  // 已发布的运行时快照：Net + generation + 后端 + 线程数

        mutable std::mutex mtx_;    // 保护 current_snapshot_ 的读写；mutable 允许 const 方法加锁
    };

    ModelLoader::ModelLoader() : pImpl_(std::make_unique<Impl>()) {}
    ModelLoader::~ModelLoader() = default;

    bool ModelLoader::IsReady() const {
        std::lock_guard<std::mutex> lock(pImpl_->mtx_);
        return pImpl_->current_snapshot_.IsValid();
    }

    ModelRuntimeSnapshot ModelLoader::AcquireSnapshot() const {
        std::lock_guard<std::mutex> lock(pImpl_->mtx_);
        return pImpl_->current_snapshot_;
    }

    std::shared_ptr<const ncnn::Net> ModelLoader::AcquireModel() const {
        // 不能持锁调 AcquireSnapshot（它自己也加锁）→ 直接读快照的 model 字段
        std::lock_guard<std::mutex> lock(pImpl_->mtx_);
        return pImpl_->current_snapshot_.model;
    }


    std::string ModelLoader::StatusToString(ModelLoadStatus status) {
        switch (status) {
        case ModelLoadStatus::kOk:                      return "成功";
        case ModelLoadStatus::kEmptyParamPath:          return "param 路径为空";
        case ModelLoadStatus::kParamFileNotFound:       return "param 文件不存在";
        case ModelLoadStatus::kParamPathNotRegularFile: return "param 路径不是常规文件";
        case ModelLoadStatus::kBinFileNotFound:         return "bin 文件不存在（由 param 路径推导）";
        case ModelLoadStatus::kBinPathNotRegularFile:   return "bin 路径不是常规文件";
        case ModelLoadStatus::kInvalidThreadCount:      return "线程数非法（必须 > 0）";
        case ModelLoadStatus::kVulkanUnavailable:       return "Vulkan 不可用";
        case ModelLoadStatus::kVulkanNotVerified:       return "Vulkan 本轮未验证（不发布 GPU 模型）";
        case ModelLoadStatus::kLoadParamFailed:         return "ncnn load_param 失败";
        case ModelLoadStatus::kLoadBinFailed:           return "ncnn load_model 失败";
        case ModelLoadStatus::kWarmupInputFailed:       return "预热 input 失败（blob 名不匹配？）";
        case ModelLoadStatus::kWarmupExtractFailed:     return "预热 extract 失败（输出 blob 名或模型内部错误）";
        case ModelLoadStatus::kModelNotReady:           return "模型未就绪（请先调用 Load）";
        case ModelLoadStatus::kFileSystemError:         return "文件系统错误";
        case ModelLoadStatus::kUnknownError:
        default:                                        return "未知错误";
        }
    }


    /// @brief 模型唯一加载入口
    /// @note  新模型全部验证通过后才替换旧模型；失败时旧模型保持可用
    ModelLoadResult ModelLoader::Load(const std::filesystem::path& param_path,
        const ModelLoadOptions& options) {

        auto t_start = std::chrono::steady_clock::now();
        auto elapsed_ms = [&t_start]() -> double {
            auto t_now = std::chrono::steady_clock::now();
            return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                   t_now - t_start).count()) / 1000.0;
        };
        ModelLoadResult ml_res;
        std::filesystem::path bin_path;
        try {
        // Load 的公开边界是不让实现细节异常泄漏给调用方。
        // 正常的校验失败走精确状态码；意外的标准异常统一在函数末尾映射为 Result。

        // 1. 校验 param_path：空 → kEmptyParamPath；不存在
        // → kParamFileNotFound；是目录 → kParamPathNotRegularFile
        if (param_path.empty()) {
            ml_res = MakeErrorResult(ModelLoadStatus::kEmptyParamPath,
                                   "param path is empty", param_path, {}, options,
                                   elapsed_ms());
            return ml_res;
        }
        std::error_code ec;
        // exists 与 is_regular_file 都使用 error_code 重载。
        // 边界：权限、损坏挂载等文件系统错误必须映射为 Result，不能让异常穿出 SDK。
        const bool param_exists = std::filesystem::exists(param_path, ec);
        if (ec) {
            ml_res = MakeErrorResult(ModelLoadStatus::kFileSystemError,
                                     "检查 param 文件失败：" + ec.message(),
                                     param_path, {}, options, elapsed_ms());
            return ml_res;
        }
        if (!param_exists) {
            ml_res = MakeErrorResult(ModelLoadStatus::kParamFileNotFound,
                                     "param file is not found", param_path, {}, options,
                                     elapsed_ms());
            return ml_res;
        }

        ec.clear();
        const bool param_is_regular = std::filesystem::is_regular_file(param_path, ec);
        if (ec) {
            ml_res = MakeErrorResult(ModelLoadStatus::kFileSystemError,
                                     "检查 param 文件类型失败：" + ec.message(),
                                     param_path, {}, options, elapsed_ms());
            return ml_res;
        }
        if (!param_is_regular) {
            // 目录、设备等都不是可交给 ncnn 的常规模型文件。
            ml_res = MakeErrorResult(ModelLoadStatus::kParamPathNotRegularFile,
                                     "param path is a dir", param_path, {}, options,
                                     elapsed_ms());

            return ml_res;
        }

        // 2.推导 bin_path = param_path.replace_extension(".bin")；同上校验
        bin_path = DeriveBinPath(param_path);
        if (bin_path.empty()) {
            ml_res = MakeErrorResult(ModelLoadStatus::kEmptyParamPath,
                                   "bin path is empty", param_path, bin_path, options,
                                   elapsed_ms());
            return ml_res;
        }
        ec.clear();
        const bool bin_exists = std::filesystem::exists(bin_path, ec);
        if (ec) {
            ml_res = MakeErrorResult(ModelLoadStatus::kFileSystemError,
                                     "检查 bin 文件失败：" + ec.message(),
                                     param_path, bin_path, options, elapsed_ms());
            return ml_res;
        }
        if (!bin_exists) {
            ml_res = MakeErrorResult(ModelLoadStatus::kBinFileNotFound,
                                     "bin file is not found", param_path, bin_path, options,
                                     elapsed_ms());
            return ml_res;
        }

        ec.clear();
        const bool bin_is_regular = std::filesystem::is_regular_file(bin_path, ec);
        if (ec) {
            ml_res = MakeErrorResult(ModelLoadStatus::kFileSystemError,
                                     "检查 bin 文件类型失败：" + ec.message(),
                                     param_path, bin_path, options, elapsed_ms());
            return ml_res;
        }
        if (!bin_is_regular) {
            // 目录、设备等不是 ncnn 可读取的权重文件。
            ml_res = MakeErrorResult(ModelLoadStatus::kBinPathNotRegularFile,
                                     "bin path is a dir", param_path, bin_path, options,
                                     elapsed_ms());

            return ml_res;
        }

        // 3. 校验 options：num_threads <= 0 → kInvalidThreadCount；backend==Vulkan → kVulkanNotVerified
        // 当前项目使用的 ncnn 构建中 0 会触发已知崩溃，因此这里明确要求正整数。
        if (options.num_threads <= 0) {
            ml_res = MakeErrorResult(ModelLoadStatus::kInvalidThreadCount,
                                     "num_threads must be greater than zero",
                                     param_path, bin_path, options,
                                     elapsed_ms());

            return ml_res;
        }
        if (options.backend == ModelBackend::kVulkan) {
            ml_res = MakeErrorResult(ModelLoadStatus::kVulkanNotVerified,
                                     "Vulkan 后端本轮未验证，请使用 CPU 后端", param_path, bin_path, options,
                                     elapsed_ms());

            return ml_res;
        }

        // 4.创建候选：make_shared<ncnn::Net>()，设置 opt（use_vulkan_compute=false, num_threads）
        // 创建候选模型实例（局部变量，不碰 current_snapshot_）
        // shared_ptr 确保异常时也能正确析构；线程参数在加载前设置，使加载和 warmup 使用同一配置。
        auto candidate = std::make_shared<ncnn::Net>();
        candidate->opt.use_vulkan_compute = false;
        candidate->opt.num_threads = options.num_threads;

        // 5. load_param（路径需转 string 再 c_str）
        // 加载网络结构（.param）：路径需转换为 std::string 再取 C 字符串
        // 因为 ncnn 是 C 风格的 API，load_param 接收 const char*，做了两次转换：path → string → C 字符串指针。
        // 临时 string 在整个表达式结束前有效，指针不会悬空
        if (candidate->load_param(param_path.string().c_str()) != 0) {
            ml_res = MakeErrorResult(ModelLoadStatus::kLoadParamFailed,
                             "ncnn load_param 失败：" + param_path.string(),
                             param_path, bin_path, options, elapsed_ms());
            return ml_res;
        }

        // 6. 加载权重数据（.bin）：统一用 fopen / fclose + FILE* 版本 兼容 NCNN_STRING 开启或关闭的 ncnn 编译配置
        // 用 unique_ptr 管理 FILE*：正常返回、提前返回或异常都统一 fclose，
        // 避免权重加载中途失败时泄漏文件描述符。
        std::unique_ptr<FILE, decltype(&fclose)> fp(
            fopen(bin_path.string().c_str(), "rb"), &fclose);
        if (!fp) {
            // 文件已通过 exists 校验后仍打不开，可能是权限、编码或检查后的竞态，
            // 不能再误报成“文件不存在”。
            ml_res = MakeErrorResult(ModelLoadStatus::kFileSystemError,
                             "无法打开 bin 文件：" + bin_path.string(),
                             param_path, bin_path, options, elapsed_ms());
            return ml_res;
        }
        int ret = candidate->load_model(fp.get());
        if (ret != 0) {
            ml_res = MakeErrorResult(ModelLoadStatus::kLoadBinFailed,
                                    "ncnn load_model 失败：" + bin_path.string(),
                                    param_path, bin_path, options, elapsed_ms());
            return ml_res;
        }

        double load_time_ms = elapsed_ms();

        // 7. 可选预热验证：用全零 tensor 走一遍完整推理，提前暴露 blob 名不匹配等问题
        //    预热失败视为加载失败，不能只打印警告
        if (options.enable_warmup) {
            ml_res = WarmupCandidate(*candidate, options, param_path, bin_path,
                                     load_time_ms);
            if (!ml_res.success) {
                return ml_res;  // WarmupCandidate 内部已填充了精确的失败原因
            }
        } else {
            ml_res = MakeSuccessResult(param_path, bin_path, options,
                                         false, elapsed_ms());
        }

        // 8. 短临界区原子替换：加锁→交换 shared_ptr→解锁
        //    lock_guard 构造时加锁，离开作用域自动解锁（RAII），异常安全
        {
            std::lock_guard<std::mutex> lock(pImpl_->mtx_);
            // 无论是否启用 warmup，都在同一个锁内读取旧状态并发布候选，
            // 保证 model_replaced 的诊断值与真正发生的交换一致。
            ml_res.info.model_replaced = (pImpl_->current_snapshot_.model != nullptr);
            ModelRuntimeSnapshot new_snapshot;
            new_snapshot.model = candidate;
            new_snapshot.generation = pImpl_->current_snapshot_.generation + 1;
            new_snapshot.backend = options.backend;
            new_snapshot.effective_num_threads = options.num_threads;
            pImpl_->current_snapshot_ = new_snapshot;
        }

        return ml_res;
        } catch (const std::filesystem::filesystem_error& ex) {
            // 理论上 error_code 重载已覆盖常见文件系统失败；这里兜住竞态或库内部异常。
            return MakeErrorResult(ModelLoadStatus::kFileSystemError,
                                   ex.what(), param_path, bin_path, options, elapsed_ms());
        } catch (const std::exception& ex) {
            // 分配失败等非预期标准异常不能伪装成某个具体模型错误。
            return MakeErrorResult(ModelLoadStatus::kUnknownError,
                                   ex.what(), param_path, bin_path, options, elapsed_ms());
        } catch (...) {
            // 最终兜底：保持旧模型不变，并向调用方返回可诊断失败。
            return MakeErrorResult(ModelLoadStatus::kUnknownError,
                                   "unknown exception while loading model",
                                   param_path, bin_path, options, elapsed_ms());
        }
    }

    // ============================================================
    // WarmupCandidate — 零输入预热
    // ============================================================
    // 用全零 tensor 验证：blob 名存在、extract 成功、输出 shape 正确。
    // pred 的具体值不重要：带偏置的网络即使零输入也未必产生零输出。
    // warmup 只验证固定输入能走通 blob 与 shape，不承担视觉正确性验收。
    ModelLoadResult WarmupCandidate(
        ncnn::Net& candidate,                   // 候选模型（已 load_param + load_model）
        const ModelLoadOptions& opts,           // 加载选项（用于填充 Result.info）
        const std::filesystem::path& param,     // .param 路径
        const std::filesystem::path& bin,       // .bin 路径
        double load_time_ms                     // 前面 load_param+load_model 的耗时
    ) {
        using Spec = Wav2LipModelSpec;
        auto warmup_start = std::chrono::steady_clock::now();

        // 1.创建 ncnn::Mat zero_mel(16, 80, 1) 和 ncnn::Mat zero_face(96, 96, 6)——尺寸来自 Wav2LipModelSpec
        ncnn::Mat zero_mel(Spec::kMelFrames, Spec::kMelBins, 1);    // (16, 80, 1)
        ncnn::Mat zero_face(Spec::kFaceWidth, Spec::kFaceHeight,
                            Spec::kFaceChannels);                     // (96, 96, 6)
        if (zero_mel.empty() || zero_face.empty()) {
            // ncnn::Mat 分配失败时不能继续把空指针交给 Extractor。
            return MakeErrorResult(ModelLoadStatus::kWarmupInputFailed,
                                   "预热输入 tensor 分配失败",
                                   param, bin, opts, load_time_ms);
        }

        // ncnn::Mat 的构造函数只分配内存，不保证清零；必须显式 fill，
        // 才能让“零输入预热”可重复、可解释，并避免读取未初始化内存。
        zero_mel.fill(0.0f);
        zero_face.fill(0.0f);


        // 2.创建 Extractor （相当于一个计算任务），每次推理新建一个
        ncnn::Extractor ex = candidate.create_extractor();
        ex.set_light_mode(true);  // 轻量模式，不保留中间结果

        // 3.验证 mel blob 名 — 如果 .param 里不叫 "mel"，input() 返回非 0
        ModelLoadResult ml_res;
        if (ex.input(Spec::kInputMel, zero_mel) != 0) {
            // 失败 → kWarmupInputFailed
            ml_res = MakeErrorResult(ModelLoadStatus::kWarmupInputFailed,
                                    "Warmup input(" + std::string(Spec::kInputMel) + ") 失败",
                                    param, bin, opts, load_time_ms);
            return ml_res;
        }

        // 4.同上验证 face blob 名
        if (ex.input(Spec::kInputFace, zero_face) != 0) {
            // 失败 → kWarmupInputFailed
            ml_res = MakeErrorResult(ModelLoadStatus::kWarmupInputFailed,
                                    "Warmup input(" + std::string(Spec::kInputFace) + ") 失败",
                                    param, bin, opts, load_time_ms);
            return ml_res;
        }

        // 5.验证输出 — 能拿到 pred 才算模型真正可用
        ncnn::Mat pred;
        if (ex.extract(Spec::kOutputPred, pred) != 0 || pred.empty()) {
            // 失败 → kWarmupExtractFailed
            ml_res = MakeErrorResult(ModelLoadStatus::kWarmupExtractFailed,
                                    "预热 extract 失败",
                                    param, bin, opts, load_time_ms);
            return ml_res;
        }


        // 6.验证输出 shape
        if (pred.w != Spec::kPredWidth || pred.h != Spec::kPredHeight ||
            pred.c != Spec::kPredChannels) {
            // 失败 → kWarmupExtractFailed（shape 不匹配）
            ml_res = MakeErrorResult(ModelLoadStatus::kWarmupExtractFailed,
                                    "预热 extract 失败：shape 不匹配",
                                    param, bin, opts, load_time_ms);
            return ml_res;
        }


        // 7.全部通过
        auto warmup_end = std::chrono::steady_clock::now();
        double warmup_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                warmup_end - warmup_start).count()) / 1000.0;

        ml_res = MakeSuccessResult(param, bin, opts, true, load_time_ms + warmup_ms);
        return ml_res;
    }


} // namespace model
} // namespace digital_human
