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

  cv::Mat data_wrapper;

  // 1. 优先检查二进制数据 (使用shared_ptr)
  if (ctx.has("image_binary")) {
    spdlog::debug("ImageDecoder: Using binary image data (shared_ptr)");
    spdlog::default_logger()->flush();
    auto ptr = ctx.get<std::shared_ptr<std::vector<uint8_t>>>("image_binary");

    // Check size
    spdlog::info("ImageDecoder: Binary data size: {} bytes at ptr={}",
                 ptr->size(), fmt::ptr(ptr.get()));
    spdlog::default_logger()->flush();

    // Create wrapper directly from pointer data (no copy)
    data_wrapper = cv::Mat(1, ptr->size(), CV_8U, ptr->data());
    spdlog::info(
        "ImageDecoder: cv::Mat wrapper created. Rows={}, Cols={}, Type={}",
        data_wrapper.rows, data_wrapper.cols, data_wrapper.type());
    spdlog::default_logger()->flush();
  }
  // 2. 回退到Base64
  else if (ctx.has("image_base64")) {
    spdlog::info("ImageDecoder: Using Base64 image data");
    spdlog::default_logger()->flush();
    std::string base64_data = ctx.get<std::string>("image_base64");
    std::string pure_base64 = base64::strip_data_url(base64_data);

    try {
      decoded = base64::decode(pure_base64);
      spdlog::info("ImageDecoder: Base64 decoded size: {}", decoded.size());
      spdlog::default_logger()->flush();
    } catch (const std::exception &e) {
      spdlog::error("ImageDecoder: Base64 decode failed: {}", e.what());
      spdlog::default_logger()->flush();
      return false;
    }

    if (decoded.empty()) {
      spdlog::error("ImageDecoder: Image data is empty");
      spdlog::default_logger()->flush();
      return false;
    }

    // Create wrapper from local vector
    data_wrapper = cv::Mat(1, decoded.size(), CV_8U, decoded.data());
  } else {
    spdlog::error("ImageDecoder: Missing image data");
    spdlog::default_logger()->flush();
    return false;
  }

  // data_wrapper should be valid at this point (either from binary or base64)
  if (data_wrapper.empty()) {
    spdlog::error("ImageDecoder: Image data wrapper is empty");
    spdlog::default_logger()->flush();
    return false;
  }

  // Decode
  try {
    spdlog::info("ImageDecoder: Starting cv::imdecode...");
    spdlog::default_logger()->flush();
    ctx.frame = cv::imdecode(data_wrapper, cv::IMREAD_COLOR);
    spdlog::info("ImageDecoder: cv::imdecode complete. Frame: {}x{}",
                 ctx.frame.cols, ctx.frame.rows);
    spdlog::default_logger()->flush();
  } catch (const std::exception &e) {
    spdlog::error("ImageDecoder: imdecode failed: {}", e.what());
    spdlog::default_logger()->flush();
    return false;
  }

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
