#pragma once

#include <opencv2/core.hpp>
#include <any>
#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace yolo_edge {

struct Detection {
  int track_id = -1;
  int class_id = 0;
  std::string class_name;
  float confidence = 0.0f;
  cv::RotatedRect obb;
  std::array<float, 8> obb_points{};
  std::array<int, 4> bbox{};
  bool geometry_ready = false;

  std::optional<double> lon;
  std::optional<double> lat;
  std::optional<double> ground_x;
  std::optional<double> ground_y;

  void refresh_geometry() {
    cv::Point2f pts[4];
    obb.points(pts);
    obb_points = {pts[0].x, pts[0].y, pts[1].x, pts[1].y,
                  pts[2].x, pts[2].y, pts[3].x, pts[3].y};

    cv::Rect rect = obb.boundingRect();
    bbox = {rect.x, rect.y, rect.width, rect.height};
    geometry_ready = true;
  }
};

class ProcessingContext {
public:
  cv::Mat frame;
  int frame_id = 0;
  std::string session_id;
  std::string request_id;
  std::vector<Detection> detections;

  double decode_time_ms = 0.0;
  double infer_time_ms = 0.0;
  double track_time_ms = 0.0;
  double geo_time_ms = 0.0;
  double total_time_ms = 0.0;

  template <typename T> void set(const std::string &key, T &&value) {
    extras_[key] = std::forward<T>(value);
  }

  template <typename T> T get(const std::string &key) const {
    return std::any_cast<T>(extras_.at(key));
  }

  template <typename T>
  T get_or(const std::string &key, T default_val) const {
    auto it = extras_.find(key);
    if (it == extras_.end()) {
      return default_val;
    }
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &) {
      return default_val;
    }
  }

  bool has(const std::string &key) const {
    return extras_.find(key) != extras_.end();
  }

  void remove(const std::string &key) { extras_.erase(key); }

  void clear_extras() { extras_.clear(); }

private:
  std::unordered_map<std::string, std::any> extras_;
};

} // namespace yolo_edge
