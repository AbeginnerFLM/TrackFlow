#include "processors/image_decoder.hpp"
#include "core/processor_factory.hpp"
#include "utils/base64.hpp"
#include <chrono>
#include <cstdio>
#include <opencv2/imgcodecs.hpp>

namespace yolo_edge {

bool ImageDecoder::process(ProcessingContext &ctx) {
  try {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    std::vector<uint8_t> decoded;
    cv::Mat data_wrapper;

    // 1. Check binary (shared_ptr) - Priority
    if (ctx.has("image_binary")) {
      // fprintf(stderr, "DEBUG: ImageDecoder: Using binary image data\n");
      auto ptr = ctx.get<std::shared_ptr<std::vector<uint8_t>>>("image_binary");
      data_wrapper = cv::Mat(1, ptr->size(), CV_8U, ptr->data());
    }
    // 2. Fallback to Base64
    else if (ctx.has("image_base64")) {
      std::string base64_data = ctx.get<std::string>("image_base64");
      std::string pure_base64 = base64::strip_data_url(base64_data);
      decoded = base64::decode(pure_base64);
      if (decoded.empty())
        return false;
      data_wrapper = cv::Mat(1, decoded.size(), CV_8U, decoded.data());
    } else {
      fprintf(stderr, "ImageDecoder: Missing image data\n");
      return false;
    }

    if (data_wrapper.empty()) {
      fprintf(stderr, "ImageDecoder: Image data wrapper is empty\n");
      return false;
    }

    // Decode
    ctx.frame = cv::imdecode(data_wrapper, cv::IMREAD_COLOR);
    if (ctx.frame.empty()) {
      fprintf(stderr, "ImageDecoder: Failed to decode image\n");
      return false;
    }

    auto end = Clock::now();
    ctx.decode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // fprintf(stderr, "ImageDecoder: Decoded %dx%d image in %.2fms\n",
    //       ctx.frame.cols, ctx.frame.rows, ctx.decode_time_ms);

    ctx.remove("image_base64");
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "ImageDecoder: Exception in process: %s\n", e.what());
    return false;
  }
}

// Register
REGISTER_PROCESSOR("decoder", ImageDecoder);

} // namespace yolo_edge
