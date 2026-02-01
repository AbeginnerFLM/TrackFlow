#pragma once

#include "core/image_processor.hpp"
#include <opencv2/core.hpp>

namespace yolo_edge {

/**
 * 地理坐标转换器
 * 将像素坐标转换为经纬度
 */
class GeoTransformer : public ImageProcessor {
public:
  GeoTransformer();
  ~GeoTransformer();

  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "GeoTransformer"; }
  void configure(const json &config) override;

private:
  // 初始化PROJ投影
  void init_proj();

  // UTM坐标转经纬度
  std::pair<double, double> utm_to_lonlat(double easting, double northing);

  // 单应矩阵 (像素→地面米)
  cv::Mat H_;

  // 相机参数 (可选,用于畸变校正)
  cv::Mat K_;    // 内参矩阵
  cv::Mat dist_; // 畸变系数
  bool has_camera_params_ = false;

  // 原点坐标
  double origin_lon_ = 0.0;
  double origin_lat_ = 0.0;
  double origin_utm_x_ = 0.0;
  double origin_utm_y_ = 0.0;

  // PROJ对象 (使用void*避免头文件冲突)
  void *proj_ctx_ = nullptr;
  void *proj_ = nullptr;

  bool initialized_ = false;
};

} // namespace yolo_edge
