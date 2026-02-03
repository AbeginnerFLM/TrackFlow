#include "processors/image_decoder.hpp"
#include "core/processor_factory.hpp"
#include "utils/base64.hpp"
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace yolo_edge {

bool ImageDecoder::process(ProcessingContext &ctx) {
  try {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    std::vector<uint8_t> decoded;
    cv::Mat data_wrapper;

    // 1. 优先检查二进制数据 (使用shared_ptr)
    if (ctx.has("image_binary")) {
      // ... (existing code, implied) ...
      // We will copy the existing logic here in the next turn if I don't use
      // multi_replace Wait, replace_file_content replaces the WHOLE range. I
      // need to be careful not to delete logic. I should use multi-step or just
      // wrap the outer part. Actually, I can just wrap the whole function.

      // Re-reading file content via view_file showed me the content.
      // I'll replicate the logic carefully or use the specific inner block?
      // No, wrapping the whole function is safer.

      spdlog::debug("ImageDecoder: Using binary image data (shared_ptr)");
      auto ptr = ctx.get<std::shared_ptr<std::vector<uint8_t>>>("image_binary");
      data_wrapper = cv::Mat(1, ptr->size(), CV_8U, ptr->data());
    }
    // 2. 回退到Base64
    else if (ctx.has("image_base64")) {
      std::string base64_data = ctx.get<std::string>("image_base64");
      std::string pure_base64 = base64::strip_data_url(base64_data);
      decoded = base64::decode(pure_base64);
      if (decoded.empty())
        return false;
      data_wrapper = cv::Mat(1, decoded.size(), CV_8U, decoded.data());
    } else {
      spdlog::error("ImageDecoder: Missing image data");
      return false;
    }

    if (data_wrapper.empty()) {
      spdlog::error("ImageDecoder: Image data wrapper is empty");
      return false;
    }

    // Decode
    ctx.frame = cv::imdecode(data_wrapper, cv::IMREAD_COLOR);
    if (ctx.frame.empty()) {
      spdlog::error("ImageDecoder: Failed to decode image");
      return false;
    }

    auto end = Clock::now();
    ctx.decode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    spdlog::debug("ImageDecoder: Decoded {}x{} image in {:.2f}ms",
                  ctx.frame.cols, ctx.frame.rows, ctx.decode_time_ms);

    ctx.remove("image_base64");
    return true;
  } catch (const std::exception &e) {
    spdlog::error("ImageDecoder: Exception in process: {}", e.what());
    return false;
  }
}

// 注册处理器
REGISTER_PROCESSOR("decoder", ImageDecoder);

} // namespace yolo_edge
