#pragma once

#include "core/image_processor.hpp"

namespace yolo_edge {

/**
 * 图像解码器
 * 将Base64编码的图像解码为OpenCV Mat
 */
class ImageDecoder : public ImageProcessor {
public:
  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "ImageDecoder"; }
};

} // namespace yolo_edge
