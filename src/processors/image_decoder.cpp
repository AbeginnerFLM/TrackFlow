#include "processors/image_decoder.hpp"
#include "core/processor_factory.hpp"
#include "utils/base64.hpp"
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace yolo_edge {

bool ImageDecoder::process(ProcessingContext &ctx) {
  using Clock = std::chrono::high_resolution_clock;
  auto start = Clock::now();

  std::vector<uint8_t> decoded;

  // 1. 优先检查二进制数据
  if (ctx.has("image_binary")) {
    spdlog::debug("ImageDecoder: Using binary image data");
    decoded = ctx.get<std::vector<uint8_t>>("image_binary");
  }
  // 2. 回退到Base64
  else if (ctx.has("image_base64")) {
    spdlog::debug("ImageDecoder: Getting base64 data from context...");
    std::string base64_data = ctx.get<std::string>("image_base64");
    spdlog::debug("ImageDecoder: Base64 data size: {} bytes",
                  base64_data.size());

    // 移除data URL前缀
    std::string pure_base64 = base64::strip_data_url(base64_data);

    // Base64解码
    spdlog::debug("ImageDecoder: Starting base64 decode...");
    try {
      decoded = base64::decode(pure_base64);
      spdlog::debug("ImageDecoder: Decoded size: {} bytes", decoded.size());
    } catch (const std::exception &e) {
      spdlog::error("ImageDecoder: Base64 decode failed: {}", e.what());
      return false;
    }
  } else {
    spdlog::error(
        "ImageDecoder: Missing image data (binary or base64) in context");
    return false;
  }

  if (decoded.empty()) {
    spdlog::error("ImageDecoder: Image data is empty");
    return false;
  }

  // 解码为OpenCV图像
  spdlog::debug("ImageDecoder: Running cv::imdecode...");
  ctx.frame = cv::imdecode(decoded, cv::IMREAD_COLOR);

  if (ctx.frame.empty()) {
    spdlog::error("ImageDecoder: Failed to decode image");
    return false;
  }

  auto end = Clock::now();
  ctx.decode_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  spdlog::debug("ImageDecoder: Decoded {}x{} image in {:.2f}ms", ctx.frame.cols,
                ctx.frame.rows, ctx.decode_time_ms);

  // 解码后可以清除原始数据以节省内存
  ctx.remove("image_base64");

  return true;
}

// 注册处理器
REGISTER_PROCESSOR("decoder", ImageDecoder);

} // namespace yolo_edge
