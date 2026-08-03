/// @file digital_human_pipeline_demo.cpp
/// @brief Pipeline 离线模式示例程序：完整数据流演示和帧输出
///
/// 用法：
///   ./digital_human_pipeline_demo <image> <audio> <model.param> <landmark.dat> [output_dir]
///
/// 示例：
///   ./digital_human_pipeline_demo testdata/golden/face.jpg testdata/golden/audio.wav \
///       models/wav2lip/wav2lip.param models/shape_predictor_68_face_landmarks.dat \
///       golden_output/pipeline_e2e

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "pipeline/digital_human_pipeline.h"
#include "pipeline/pipeline_output_sink.h"
#include "pipeline/pipeline_types.h"

namespace fs = std::filesystem;
using namespace digital_human::pipeline;

// ============================================================================
// 示例 Sink 实现：写入帧文件并打印进度
// ============================================================================

class DemoSink : public PipelineOutputSink {
public:
    explicit DemoSink(const fs::path& output_dir)
        : output_dir_(output_dir) {
        fs::create_directories(output_dir_);
    }

    void OnFrame(const PipelineFrame& frame) override {
        frame_count_++;

        // 保存帧
        char filename[256];
        std::snprintf(filename, sizeof(filename), "frame_%06d.jpg",
                     static_cast<int>(frame.video_frame.frame_index));
        fs::path filepath = output_dir_ / filename;
        cv::imwrite(filepath.string(), frame.video_frame.frame_bgr);

        // 进度打印
        if (frame_count_ % 10 == 0 || frame_count_ == 1) {
            std::cout << "  Frame " << std::setw(4) << frame_count_
                      << ": index=" << frame.video_frame.frame_index
                      << " pts=" << frame.video_frame.pts.microseconds << " us"
                      << " task_id=" << frame.source_task_id
                      << std::endl;
        }
    }

    void OnTerminal(const PipelineResult& result) override {
        std::cout << "\n=== Pipeline 完成 ===" << std::endl;
        std::cout << "成功: " << (result.success ? "是" : "否") << std::endl;
        std::cout << "终态: " << PipelineStateToString(result.terminal_state) << std::endl;
        std::cout << "总帧数: " << frame_count_ << std::endl;

        if (!result.success) {
            std::cout << "错误码: " << PipelineErrorCodeToString(result.error_code) << std::endl;
            std::cout << "错误信息: " << result.error_message << std::endl;
        }

        std::cout << "\n--- 统计 ---" << std::endl;
        std::cout << "生成任务数: " << result.stats.generated_task_count << std::endl;
        std::cout << "Scheduler 接受: " << result.stats.scheduler_accepted_count << std::endl;
        std::cout << "Scheduler 失败: " << result.stats.scheduler_failed_count << std::endl;
        std::cout << "渲染唯一帧: " << result.stats.rendered_unique_frame_count << std::endl;
        std::cout << "Sink 回调数: " << result.stats.sink_callback_count << std::endl;
        std::cout << "Q1 峰值深度: " << result.stats.q1_high_watermark << std::endl;
        std::cout << "Q2 峰值深度: " << result.stats.q2_high_watermark << std::endl;
        std::cout << "准备耗时: " << result.stats.prepare_time_ms << " ms" << std::endl;
        std::cout << "总耗时: " << result.stats.total_wall_time_ms << " ms" << std::endl;
        std::cout << "丢弃(过期): " << result.stats.dropped_late_count << std::endl;
        std::cout << "丢弃(溢满): " << result.stats.dropped_overflow_count << std::endl;
        std::cout << "取消丢弃: " << result.stats.cancelled_discarded_count << std::endl;
        std::cout << "输出目录: " << output_dir_ << std::endl;
    }

    int frame_count() const { return frame_count_; }

private:
    fs::path output_dir_;
    int frame_count_ = 0;
};

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "用法: " << argv[0]
                  << " <image> <audio> <model.param> <landmark.dat> [output_dir]"
                  << std::endl;
        return 1;
    }

    fs::path image_path = argv[1];
    fs::path audio_path = argv[2];
    fs::path model_param = argv[3];
    fs::path landmark_model = argv[4];
    fs::path output_dir = (argc >= 6) ? argv[5] : "golden_output/pipeline_e2e";

    // 检查输入文件
    for (const auto& p : {image_path, audio_path, model_param, landmark_model}) {
        if (!fs::exists(p)) {
            std::cerr << "错误: 文件不存在: " << p << std::endl;
            return 1;
        }
    }

    std::cout << "=== 数字人 Pipeline 离线示例 ===" << std::endl;
    std::cout << "图片: " << image_path << std::endl;
    std::cout << "音频: " << audio_path << std::endl;
    std::cout << "模型: " << model_param << std::endl;
    std::cout << "关键点: " << landmark_model << std::endl;
    std::cout << "输出: " << output_dir << std::endl;
    std::cout << std::endl;

    auto sink = std::make_shared<DemoSink>(output_dir);
    DigitalHumanPipeline pipeline;

    PipelineConfig config = PipelineConfig::OfflineDefault();
    config.image_path = image_path;
    config.audio_path = audio_path;
    config.model_param_path = model_param;
    config.landmark_model_path = landmark_model;
    config.fps_num = 25;
    config.fps_den = 1;
    config.output_dir = output_dir;

    std::cout << "启动 Pipeline..." << std::endl;
    auto start_result = pipeline.Start(config, sink);
    if (!start_result.success) {
        std::cerr << "启动失败: " << start_result.error_message << std::endl;
        return 1;
    }

    std::cout << "Pipeline 已启动，等待完成..." << std::endl;
    auto wait_result = pipeline.Wait();

    std::cout << "完成。总帧数: " << sink->frame_count() << std::endl;
    return wait_result.success ? 0 : 1;
}
