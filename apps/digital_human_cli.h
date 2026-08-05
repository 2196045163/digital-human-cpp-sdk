#pragma once

/// @file digital_human_cli.h
/// @brief CLI 编排入口声明（供测试注入 Fake MediaWriter）。
///
/// 生产入口 digital_human_app::main 委托给 RunCli + DefaultMediaWriterFactory；
/// 测试通过注入自定义工厂，在不修改 FFmpeg 的前提下验证 CLI 的"四条件成功校验"。

#include <functional>
#include <memory>

#include "output/final_media_writer.h"

namespace digital_human {
namespace cli {

/// @brief 最终媒体写出器工厂；测试注入 Fake 实现以注入写失败。
using MediaWriterFactory =
    std::function<std::shared_ptr<output::MediaWriter>(const output::WriterConfig&)>;

/// @brief 默认工厂：创建生产 FinalMediaWriter。
MediaWriterFactory DefaultMediaWriterFactory();

/// @brief CLI 编排主流程：参数解析 → Pipeline + writer → 四条件成功校验 → JSON。
///
/// 成功判定（四条件联合，全部满足才 exit 0 / status success）：
///   1) pipeline.Wait() 返回 success
///   2) writer->IsFinalized() 为 true
///   3) writer->GetLastError() == kOk
///   4) 输出文件存在且非空
///
/// @param argc / argv CLI 参数（与 main 相同，argv[0] 为程序名）
/// @param writer_factory writer 工厂（生产用 DefaultMediaWriterFactory，
///                        测试注入 Fake 工厂）
/// @return 退出码：0 成功；1 执行/产物错误；2 参数错误；3 realtime 拒绝
int RunCli(int argc, char* argv[], const MediaWriterFactory& writer_factory);

}  // namespace cli
}  // namespace digital_human
