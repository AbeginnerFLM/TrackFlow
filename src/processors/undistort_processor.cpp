#include "processors/undistort_processor.hpp"
#include "core/processor_factory.hpp"
#include <cstdio>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_edge {

void UndistortProcessor::configure(const json &config) {
  // 相机内参矩阵 (必须)
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

  // 畸变系数 (必须)
  if (!config.contains("dist_coeffs")) {
    fprintf(stderr, "[ERROR] UndistortProcessor: No dist_coeffs provided\n");
    return;
  }

  auto d_data = config["dist_coeffs"].get<std::vector<double>>();
  dist_ = cv::Mat(1, d_data.size(), CV_64F);
  for (size_t i = 0; i < d_data.size(); ++i) {
    dist_.at<double>(0, i) = d_data[i];
  }

  // 图像尺寸 (必须)
  int width = config.value("width", 1920);
  int height = config.value("height", 1080);

  // 计算最优新相机矩阵 (可选参数alpha: 0=裁剪所有黑边, 1=保留所有像素)
  double alpha = config.value("alpha", 0.0);
  new_K_ = cv::getOptimalNewCameraMatrix(K_, dist_, cv::Size(width, height),
                                         alpha, cv::Size(width, height));

  // 预计算映射表 (显著加速后续处理)
  cv::initUndistortRectifyMap(K_, dist_, cv::Mat(), new_K_,
                              cv::Size(width, height), CV_32FC1, map1_, map2_);

  initialized_ = true;

  fprintf(stderr, "[INFO] UndistortProcessor: Initialized for %dx%d images\n",
          width, height);
}

bool UndistortProcessor::process(ProcessingContext &ctx) {
  // 如果未初始化或帧为空，跳过
  if (!initialized_ || ctx.frame.empty()) {
    return true;
  }

  // 使用预计算的映射表进行快速畸变校正
  cv::Mat undistorted;
  cv::remap(ctx.frame, undistorted, map1_, map2_, cv::INTER_LINEAR);
  ctx.frame = undistorted;

  // fprintf(stderr, "[DEBUG] UndistortProcessor: Frame undistorted\n");

  return true;
}

// 注册处理器
REGISTER_PROCESSOR("undistort", UndistortProcessor);

} // namespace yolo_edge
