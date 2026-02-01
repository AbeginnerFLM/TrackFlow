#include "processors/geo_transformer.hpp"
#include "core/processor_factory.hpp"
#include <chrono>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

// 包含PROJ头文件 (只在cpp中)
#include <proj.h>

namespace yolo_edge {

GeoTransformer::GeoTransformer() = default;

GeoTransformer::~GeoTransformer() {
  if (proj_) {
    proj_destroy(static_cast<PJ *>(proj_));
  }
  if (proj_ctx_) {
    proj_context_destroy(static_cast<PJ_CONTEXT *>(proj_ctx_));
  }
}

void GeoTransformer::configure(const json &config) {
  // 单应矩阵 (必须)
  if (!config.contains("homography")) {
    spdlog::warn("GeoTransformer: No homography matrix provided, will skip "
                 "transformation");
    return;
  }

  auto h_data = config["homography"].get<std::vector<double>>();
  if (h_data.size() != 9) {
    spdlog::error("GeoTransformer: Homography matrix must have 9 elements");
    return;
  }

  H_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) {
    H_.at<double>(i / 3, i % 3) = h_data[i];
  }

  // 原点经纬度 (必须)
  if (!config.contains("origin_lon") || !config.contains("origin_lat")) {
    spdlog::error("GeoTransformer: Missing origin_lon or origin_lat");
    return;
  }

  origin_lon_ = config["origin_lon"].get<double>();
  origin_lat_ = config["origin_lat"].get<double>();

  // 相机参数 (可选)
  if (config.contains("camera_matrix") && config.contains("dist_coeffs")) {
    auto k_data = config["camera_matrix"].get<std::vector<double>>();
    if (k_data.size() == 9) {
      K_ = cv::Mat(3, 3, CV_64F);
      for (int i = 0; i < 9; ++i) {
        K_.at<double>(i / 3, i % 3) = k_data[i];
      }

      auto d_data = config["dist_coeffs"].get<std::vector<double>>();
      dist_ = cv::Mat(1, d_data.size(), CV_64F);
      for (size_t i = 0; i < d_data.size(); ++i) {
        dist_.at<double>(0, i) = d_data[i];
      }

      has_camera_params_ = true;
      spdlog::debug("GeoTransformer: Camera parameters loaded");
    }
  }

  init_proj();
  initialized_ = true;

  spdlog::info("GeoTransformer: Configured with origin ({:.6f}, {:.6f})",
               origin_lon_, origin_lat_);
}

void GeoTransformer::init_proj() {
  // 创建PROJ上下文
  proj_ctx_ = proj_context_create();

  // 计算UTM区号
  int zone = static_cast<int>((origin_lon_ + 180.0) / 6.0) + 1;
  bool is_north = origin_lat_ >= 0;
  int epsg = is_north ? (32600 + zone) : (32700 + zone);

  // 创建坐标转换器: UTM -> WGS84
  std::string utm_def = "EPSG:" + std::to_string(epsg);
  proj_ = proj_create_crs_to_crs(static_cast<PJ_CONTEXT *>(proj_ctx_),
                                 utm_def.c_str(), "EPSG:4326", nullptr);

  if (!proj_) {
    spdlog::error("GeoTransformer: Failed to create PROJ transformation");
    return;
  }

  // 计算原点的UTM坐标
  PJ_COORD origin_wgs84 = proj_coord(origin_lon_, origin_lat_, 0, 0);
  PJ_COORD origin_utm =
      proj_trans(static_cast<PJ *>(proj_), PJ_INV, origin_wgs84);
  origin_utm_x_ = origin_utm.xy.x;
  origin_utm_y_ = origin_utm.xy.y;

  spdlog::debug("GeoTransformer: Origin UTM ({:.2f}, {:.2f}) Zone {}{}",
                origin_utm_x_, origin_utm_y_, zone, is_north ? "N" : "S");
}

std::pair<double, double> GeoTransformer::utm_to_lonlat(double easting,
                                                        double northing) {
  // 相对坐标转换为绝对UTM坐标
  double world_x = easting + origin_utm_x_;
  double world_y = northing + origin_utm_y_;

  // UTM -> WGS84
  PJ_COORD utm = proj_coord(world_x, world_y, 0, 0);
  PJ_COORD lonlat = proj_trans(static_cast<PJ *>(proj_), PJ_FWD, utm);

  return {lonlat.lp.lam, lonlat.lp.phi};
}

bool GeoTransformer::process(ProcessingContext &ctx) {
  // 如果未初始化或没有检测结果，直接返回成功
  if (!initialized_ || ctx.detections.empty()) {
    return true;
  }

  using Clock = std::chrono::high_resolution_clock;
  auto start = Clock::now();

  for (auto &det : ctx.detections) {
    // 1. 获取bbox中心点
    cv::Point2f center = det.obb.center;

    // 2. 畸变校正 (如果有相机参数)
    if (has_camera_params_) {
      std::vector<cv::Point2f> pts_in = {center};
      std::vector<cv::Point2f> pts_out;
      cv::undistortPoints(pts_in, pts_out, K_, dist_, cv::noArray(), K_);
      center = pts_out[0];
    }

    // 3. 透视变换: 像素坐标 -> 地面坐标 (相对原点的米)
    std::vector<cv::Point2f> src = {center};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H_);

    det.ground_x = dst[0].x;
    det.ground_y = dst[0].y;

    // 4. UTM -> 经纬度
    auto [lon, lat] = utm_to_lonlat(dst[0].x, dst[0].y);
    det.lon = lon;
    det.lat = lat;
  }

  auto end = Clock::now();
  ctx.geo_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  spdlog::debug("GeoTransformer: Transformed {} detections in {:.2f}ms",
                ctx.detections.size(), ctx.geo_time_ms);

  return true;
}

// 注册处理器
REGISTER_PROCESSOR("geo_transform", GeoTransformer);

} // namespace yolo_edge
