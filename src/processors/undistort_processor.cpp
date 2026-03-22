#include "processors/undistort_processor.hpp"
#include "core/processor_factory.hpp"
#include <cstdio>
#include <opencv2/calib3d.hpp>

namespace yolo_edge {

void UndistortProcessor::configure(const json &config) {
  if (!config.contains("camera_matrix")) {
    fprintf(stderr, "[WARN] UndistortProcessor: No camera_matrix provided, "
                    "will skip undistortion\n");
    return;
  }

  auto k_data = config["camera_matrix"].get<std::vector<double>>();
  if (k_data.size() != 9) {
    fprintf(stderr,
            "[ERROR] UndistortProcessor: camera_matrix must have 9 elements\n");
    return;
  }

  K_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) {
    K_.at<double>(i / 3, i % 3) = k_data[i];
  }

  if (!config.contains("dist_coeffs")) {
    fprintf(stderr, "[ERROR] UndistortProcessor: No dist_coeffs provided\n");
    return;
  }

  auto d_data = config["dist_coeffs"].get<std::vector<double>>();
  dist_ = cv::Mat(1, d_data.size(), CV_64F);
  for (size_t i = 0; i < d_data.size(); ++i) {
    dist_.at<double>(0, i) = d_data[i];
  }

  initialized_ = true;
  fprintf(stderr, "[INFO] UndistortProcessor: Initialized (point-only mode)\n");
}

bool UndistortProcessor::process(ProcessingContext &ctx) {
  if (!initialized_ || ctx.detections.empty()) {
    return true;
  }

  // 收集所有检测中心点
  std::vector<cv::Point2f> pts_in;
  pts_in.reserve(ctx.detections.size());
  for (const auto &det : ctx.detections) {
    pts_in.push_back(det.obb.center);
  }

  // 批量校正畸变 (只校正点坐标, 不处理整帧图像)
  std::vector<cv::Point2f> pts_out;
  cv::undistortPoints(pts_in, pts_out, K_, dist_, cv::noArray(), K_);

  // 写回校正后的中心点
  for (size_t i = 0; i < ctx.detections.size(); ++i) {
    ctx.detections[i].obb.center = pts_out[i];
  }

  return true;
}

REGISTER_PROCESSOR("undistort", UndistortProcessor);

} // namespace yolo_edge
