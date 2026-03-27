#pragma once

#include "core/image_processor.hpp"

namespace yolo_edge {

class ImageDecoder : public ImageProcessor {
public:
  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "ImageDecoder"; }
};

} // namespace yolo_edge
