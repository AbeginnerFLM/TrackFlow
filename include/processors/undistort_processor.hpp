#pragma once

#include "core/image_processor.hpp"
#include <opencv2/core.hpp>

namespace yolo_edge {

/**
 * 畸变校正处理器
 * 只校正检测目标的中心点坐标（用于后续经纬度计算），不处理整帧图像
 */
class UndistortProcessor : public ImageProcessor {
public:
  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "UndistortProcessor"; }
  void configure(const json &config) override;

private:
  cv::Mat K_;    // 相机内参
  cv::Mat dist_; // 畸变系数

  bool initialized_ = false;
};

} // namespace yolo_edge
