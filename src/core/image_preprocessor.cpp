

#include <opencv2/imgproc.hpp>
#include <chrono>


#include "core/image_preprocessor.h"


namespace digital_human {
namespace core {

    ImagePreprocessResult ImagePreprocessor::Process(const cv::Mat& image,
        const ImagePreprocessOptions& options) {
        ImagePreprocessResult iprep_res;
        auto start = std::chrono::high_resolution_clock::now();
        // 1. 检查输入非空
        if (image.empty()) {
            iprep_res.success = false;
            iprep_res.status = ImagePreprocessStatus::kEmptyImage;
            iprep_res.error_message = "Input image is empty";

            return iprep_res;
        }

        // 2. 检查 options.target_size > 0
        if (options.target_size.width <= 0 || options.target_size.height <= 0) {
            iprep_res.success = false;
            iprep_res.status = ImagePreprocessStatus::kInvalidTargetSize;
            iprep_res.error_message = "target_size must be > 0";

            return iprep_res;
        }

        // 3. 如果尺寸不一致，按选项 resize
        cv::Mat working_img;
        if (image.cols != options.target_size.width || image.rows != options.target_size.height) {
            cv::resize(image, working_img, options.target_size);
        } else {
            working_img = image;
        }

        // 4. 明确输入通道约定，转换成 3 通道 —— 如果输入不是 3 通道，转成 3 通道"
        cv::Mat img_3ch;
        if (working_img.channels() == 3) {
            img_3ch = working_img;
        } else if (working_img.channels() == 1) {
            cv::cvtColor(working_img, img_3ch, cv::COLOR_GRAY2BGR); // 单通道灰度值复制到 B、G、R 三个通道
        } else if (working_img.channels() == 4) {
            cv::cvtColor(working_img, img_3ch, cv::COLOR_BGRA2BGR); // BGR（3 通道），扔掉透明度通道 A
        } else {
            iprep_res.success = false;
            iprep_res.status = ImagePreprocessStatus::kInvalidChannels;
            iprep_res.error_message = "Unsupported channels: " + std::to_string(working_img.channels());

            return iprep_res;
        }

        // 5. 根据 options.output_color 决定保持 BGR 或转 RGB
        cv::Mat color_img;
        if (options.output_color == ColorOrder::kRgb) {
            cv::cvtColor(img_3ch, color_img, cv::COLOR_BGR2RGB);
        } else {
            color_img = img_3ch;
        }

        // 6. 根据 options.normalize 处理：类型转换
        // kNone            -> 保持 uint8 0~255
        // kZeroToOne       -> float32, x/255
        // kMinusOneToOne   -> float32, x/127.5-1
        cv::Mat result_img;
        switch (options.normalize) {
            case NormalizeMode::kNone:
                result_img = color_img;  // 保持 uint8 0~255
                break;
            case NormalizeMode::kZeroToOne:
                color_img.convertTo(result_img, CV_32FC3, 1.0 / 255.0); // float32, x/255
                break;
            case NormalizeMode::kMinusOneToOne:
                color_img.convertTo(result_img, CV_32FC3, 1.0 / 127.5, -1.0); // float32, x/127.5-1
                break;
        }

        // 7. 用 cv::minMaxLoc（分通道或 reshape）记录实际范围
        // 记录预处理后的实际像素最小值和最大值，用于验证处理结果。
        // 比如转到 [0,1] 后，范围应该接近 min≈0, max≈1
        cv::Mat flat = result_img.reshape(1); // 把多通道拍平成一维
        double obs_min = 0, obs_max = 0;
        cv::minMaxLoc(flat, &obs_min, &obs_max);
        iprep_res.observed_min = static_cast<float>(obs_min);
        iprep_res.observed_max = static_cast<float>(obs_max);

        // 8. 返回结果和耗时
        // struct ImagePreprocessResult {
        //     bool success = false;
        //     ImagePreprocessStatus status = ImagePreprocessStatus::kOpenCvError;
        //     std::string error_message;
        //     cv::Mat image;
        //     ColorOrder color_order = ColorOrder::kBgr;
        //     NormalizeMode normalize_mode = NormalizeMode::kNone;
        //     float observed_min = 0.0f;
        //     float observed_max = 0.0f;
        //     double time_ms = 0.0;
        // };

        iprep_res.success = true;
        iprep_res.status = ImagePreprocessStatus::kOk;
        iprep_res.error_message = "";
        iprep_res.image = result_img;
        iprep_res.color_order = options.output_color;
        iprep_res.normalize_mode = options.normalize;

        auto end = std::chrono::high_resolution_clock::now();
        iprep_res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return iprep_res;
    }

    std::vector<ImagePreprocessResult> ImagePreprocessor::ProcessBatch(
            const std::vector<cv::Mat>& images,
            const ImagePreprocessOptions& options) {
        std::vector<ImagePreprocessResult> results;
        results.reserve(images.size());
        for (size_t i = 0; i < images.size(); i++) {
            results.push_back(Process(images[i], options));
        }

        return results;
    }


} // namespace core
} // namespace digital_human