#include "processors/image_decoder.hpp"
#include "core/processor_factory.hpp"
#include <chrono>
#include <cstdio>
#include <opencv2/imgcodecs.hpp>

namespace yolo_edge {

bool ImageDecoder::process(ProcessingContext &ctx) {
  try {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    cv::Mat data_wrapper;

    if (ctx.has("image_binary")) {
      auto ptr = ctx.get<std::shared_ptr<std::vector<uint8_t>>>("image_binary");
      // 关键点：这里只包一层 Mat 视图，不做额外拷贝，底层数据由 shared_ptr 保活。
      data_wrapper = cv::Mat(1, ptr->size(), CV_8U, ptr->data());
    } else {
      fprintf(stderr, "ImageDecoder: Missing binary image data\n");
      return false;
    }

    if (data_wrapper.empty()) {
      fprintf(stderr, "ImageDecoder: Image data wrapper is empty\n");
      return false;
    }

    ctx.frame = cv::imdecode(data_wrapper, cv::IMREAD_COLOR);
    if (ctx.frame.empty()) {
      fprintf(stderr, "ImageDecoder: Failed to decode image\n");
      return false;
    }

    auto end = Clock::now();
    ctx.decode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    ctx.remove("image_binary");
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "ImageDecoder: Exception in process: %s\n", e.what());
    return false;
  }
}

REGISTER_PROCESSOR("decoder", ImageDecoder);

} // namespace yolo_edge
