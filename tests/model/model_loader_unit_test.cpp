/// @file model_loader_unit_test.cpp
/// @brief 验证 ModelLoader：路径校验、参数校验、warmup、候选发布旧模型保留
///
/// 测试范围：
///   - 首次未加载时 IsReady=false / AcquireModel=nullptr
///   - 路径校验（空路径、不存在、是目录）
///   - 参数校验（负数线程、Vulkan 拒绝）
///   - CPU 加载 + warmup 成功
///   - 关闭 warmup 也能加载
///   - 坏加载不破坏旧模型（候选发布核心回归测试，防"先 clear 再 load"）
///   - StatusToString 全覆盖
///
/// 不测试：
///   - 异步加载（本轮不做）
///   - Vulkan 真实加载（本轮只做能力检测）
///   - 多线程并发 AcquireModel（需要线程 sanitizer）

#include "model/model_loader.h"

#include <gtest/gtest.h>
#include <ncnn/net.h>
#include <array>

using namespace digital_human::model;

// ============================================================================
// 首次不加载时 IsReady/AcquireModel 为 false/null
// ============================================================================
TEST(ModelLoaderTest, IsReadyBeforeLoad) {
    ModelLoader loader;
    EXPECT_FALSE(loader.IsReady());
    EXPECT_EQ(loader.AcquireModel(), nullptr);
}

// ============================================================================
// 路径校验
// ============================================================================
TEST(ModelLoaderTest, EmptyPathReturnsError) {
    ModelLoader loader;
    auto r = loader.Load({});
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kEmptyParamPath);
}

TEST(ModelLoaderTest, NonexistentPathReturnsError) {
    ModelLoader loader;
    auto r = loader.Load("/nonexistent/path/model.param");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kParamFileNotFound);
}

TEST(ModelLoaderTest, DirectoryPathReturnsError) {
    ModelLoader loader;
    auto r = loader.Load(".");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kParamPathNotRegularFile);
}

TEST(ModelLoaderTest, MissingDerivedBinReturnsError) {
    ModelLoader loader;

    // 公开头文件是现成的常规文件，但同名 .bin 不存在。
    // 这样只验证“param → bin 推导与缺失检查”，无需创建会污染工作区的临时文件。
    auto r = loader.Load("include/model/model_loader.h");

    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kBinFileNotFound);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_EQ(r.info.bin_path, "include/model/model_loader.bin");
}

// ============================================================================
// 参数校验
// ============================================================================
TEST(ModelLoaderTest, NegativeThreadCountReturnsError) {
    ModelLoader loader;
    ModelLoadOptions opts;
    opts.num_threads = -1;
    auto r = loader.Load("models/wav2lip/wav2lip.param", opts);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kInvalidThreadCount);
}

TEST(ModelLoaderTest, ZeroThreadCountReturnsError) {
    ModelLoader loader;
    ModelLoadOptions opts;
    opts.num_threads = 0;

    // 当前 ncnn 构建对 0 存在已知崩溃风险，Loader 必须在调用 ncnn 前拒绝。
    auto r = loader.Load("models/wav2lip/wav2lip.param", opts);

    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kInvalidThreadCount);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_FALSE(loader.IsReady());
}

TEST(ModelLoaderTest, VulkanBackendReturnsNotVerified) {
    ModelLoader loader;
    ModelLoadOptions opts;
    opts.backend = ModelBackend::kVulkan;
    auto r = loader.Load("models/wav2lip/wav2lip.param", opts);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.status, ModelLoadStatus::kVulkanNotVerified);
}

// ============================================================================
// 真实模型加载（CPU + warmup）
// ============================================================================
TEST(ModelLoaderTest, RealModelLoadCpuWithWarmup) {
    ModelLoader loader;
    ModelLoadOptions opts;
    opts.backend = ModelBackend::kCpu;
    opts.enable_warmup = true;

    auto r = loader.Load("models/wav2lip/wav2lip.param", opts);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_TRUE(r.info.warmup_performed);
    EXPECT_FALSE(r.info.model_replaced);
    EXPECT_TRUE(loader.IsReady());
    EXPECT_NE(loader.AcquireModel(), nullptr);
    EXPECT_GT(r.time_ms, 0.0);
}

// ============================================================================
// 无 warmup 加载
// ============================================================================
TEST(ModelLoaderTest, LoadWithoutWarmup) {
    ModelLoader loader;
    ModelLoadOptions opts;
    opts.enable_warmup = false;

    auto first = loader.Load("models/wav2lip/wav2lip.param", opts);

    ASSERT_TRUE(first.success) << first.error_message;
    EXPECT_FALSE(first.info.warmup_performed);
    EXPECT_FALSE(first.info.model_replaced);
    EXPECT_TRUE(loader.IsReady());

    auto old_snapshot = loader.AcquireModel();
    ASSERT_NE(old_snapshot, nullptr);

    // 第二次成功加载才算“替换旧模型”；该字段不能受 enable_warmup 分支影响。
    auto second = loader.Load("models/wav2lip/wav2lip.param", opts);
    ASSERT_TRUE(second.success) << second.error_message;
    EXPECT_TRUE(second.info.model_replaced);

    auto new_snapshot = loader.AcquireModel();
    ASSERT_NE(new_snapshot, nullptr);
    EXPECT_NE(old_snapshot, new_snapshot);

    // shared_ptr 快照保证替换后旧模型仍存活，调用方不会拿到悬空指针。
    ncnn::Extractor old_extractor = old_snapshot->create_extractor();
    ncnn::Mat zero_mel(16, 80, 1);
    ncnn::Mat zero_face(96, 96, 6);
    zero_mel.fill(0.0f);
    zero_face.fill(0.0f);
    EXPECT_EQ(old_extractor.input("mel", zero_mel), 0);
    EXPECT_EQ(old_extractor.input("face", zero_face), 0);
}

// ============================================================================
// generation 递增与失败不变
// ============================================================================

// 第一次成功加载 generation == 1
TEST(ModelLoaderTest, GenerationIsOneAfterFirstLoad) {
    ModelLoader loader;
    auto r = loader.Load("models/wav2lip/wav2lip.param");
    ASSERT_TRUE(r.success) << r.error_message;

    auto snapshot = loader.AcquireSnapshot();
    ASSERT_TRUE(snapshot.IsValid());
    EXPECT_EQ(snapshot.generation, 1u);
}

// 第二次成功加载 generation == 2
TEST(ModelLoaderTest, GenerationIncrementsAfterSecondLoad) {
    ModelLoader loader;

    auto r1 = loader.Load("models/wav2lip/wav2lip.param");
    ASSERT_TRUE(r1.success) << r1.error_message;
    EXPECT_EQ(loader.AcquireSnapshot().generation, 1u);

    auto r2 = loader.Load("models/wav2lip/wav2lip.param");
    ASSERT_TRUE(r2.success) << r2.error_message;
    EXPECT_EQ(loader.AcquireSnapshot().generation, 2u);
}

// 加载失败 generation 保持不变
TEST(ModelLoaderTest, GenerationUnchangedAfterFailedLoad) {
    ModelLoader loader;

    auto r1 = loader.Load("models/wav2lip/wav2lip.param");
    ASSERT_TRUE(r1.success) << r1.error_message;
    auto gen_before = loader.AcquireSnapshot().generation;

    auto r2 = loader.Load("/nonexistent/bad.param");
    EXPECT_FALSE(r2.success);

    auto gen_after = loader.AcquireSnapshot().generation;
    EXPECT_EQ(gen_after, gen_before);
}

// ============================================================================
// 坏加载不破坏旧模型（候选发布核心回归测试）
// ============================================================================
TEST(ModelLoaderTest, BadLoadPreservesOldModel) {
    ModelLoader loader;

    // 先成功加载
    auto r1 = loader.Load("models/wav2lip/wav2lip.param");
    ASSERT_TRUE(r1.success) << r1.error_message;

    auto model1 = loader.AcquireModel();
    ASSERT_NE(model1, nullptr);

    // 尝试加载不存在的模型 — 应失败
    auto r2 = loader.Load("/nonexistent/bad.param");
    EXPECT_FALSE(r2.success);

    // 旧模型仍在，且是同一个实例
    auto model2 = loader.AcquireModel();
    ASSERT_NE(model2, nullptr);
    EXPECT_EQ(model1, model2);

    // 旧模型仍可推理
    ncnn::Extractor ex = model2->create_extractor();
    ncnn::Mat zero_mel(16, 80, 1);
    ncnn::Mat zero_face(96, 96, 6);
    zero_mel.fill(0.0f);
    zero_face.fill(0.0f);
    ASSERT_EQ(ex.input("mel", zero_mel), 0);
    ASSERT_EQ(ex.input("face", zero_face), 0);

    // 只调用 input() 不能证明旧模型仍可用；必须真正 extract 并校验输出边界。
    ncnn::Mat pred;
    ASSERT_EQ(ex.extract("pred", pred), 0);
    EXPECT_FALSE(pred.empty());
    EXPECT_EQ(pred.w, 96);
    EXPECT_EQ(pred.h, 96);
    EXPECT_EQ(pred.c, 3);
}

// ============================================================================
// StatusToString 全覆盖
// ============================================================================
TEST(ModelLoaderTest, StatusToStringNotEmpty) {
    // 显式列出全部公开枚举；新增状态后必须更新本表，避免抽查少数值却声称全覆盖。
    constexpr std::array<ModelLoadStatus, 16> kAllStatuses = {
        ModelLoadStatus::kOk,
        ModelLoadStatus::kEmptyParamPath,
        ModelLoadStatus::kParamFileNotFound,
        ModelLoadStatus::kParamPathNotRegularFile,
        ModelLoadStatus::kBinFileNotFound,
        ModelLoadStatus::kBinPathNotRegularFile,
        ModelLoadStatus::kInvalidThreadCount,
        ModelLoadStatus::kVulkanUnavailable,
        ModelLoadStatus::kVulkanNotVerified,
        ModelLoadStatus::kLoadParamFailed,
        ModelLoadStatus::kLoadBinFailed,
        ModelLoadStatus::kWarmupInputFailed,
        ModelLoadStatus::kWarmupExtractFailed,
        ModelLoadStatus::kModelNotReady,
        ModelLoadStatus::kFileSystemError,
        ModelLoadStatus::kUnknownError,
    };

    for (ModelLoadStatus status : kAllStatuses) {
        EXPECT_FALSE(ModelLoader::StatusToString(status).empty());
    }
}
