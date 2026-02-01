#pragma once

#include "core/image_processor.hpp"
#include <opencv2/core.hpp>

namespace yolo_edge {

/**
 * 畸变校正处理器
 * 在检测前对整帧图像进行畸变校正
 */
class UndistortProcessor : public ImageProcessor {
public:
  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "UndistortProcessor"; }
  void configure(const json &config) override;

private:
  cv::Mat K_;           // 相机内参
  cv::Mat dist_;        // 畸变系数
  cv::Mat map1_, map2_; // 预计算的映射表
  cv::Mat new_K_;       // 新的相机矩阵

  bool initialized_ = false;
};

} // namespace yolo_edge
