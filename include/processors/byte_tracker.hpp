#pragma once

#include "core/image_processor.hpp"
#include <array>
#include <opencv2/video/tracking.hpp>
#include <vector>

namespace yolo_edge {

enum class TrackState {
  New = 0,
  Tracked = 1,
  Lost = 2,
  Removed = 3
};

struct STrack {
  int track_id = 0;
  TrackState state = TrackState::New;
  cv::RotatedRect bbox;
  int frame_id = 0;
  int start_frame = 0;
  int hits = 0;
  int time_since_update = 0;
  float score = 0.0f;
  int class_id = 0;

  cv::KalmanFilter kf;
  bool kf_initialized = false;

  static constexpr int MAX_TRAJECTORY = 100;
  std::array<cv::Point2f, MAX_TRAJECTORY> trajectory_buf{};
  int traj_head = 0;
  int traj_size = 0;

  void add_trajectory_point(cv::Point2f pt) {
    trajectory_buf[traj_head] = pt;
    traj_head = (traj_head + 1) % MAX_TRAJECTORY;
    if (traj_size < MAX_TRAJECTORY)
      traj_size++;
  }

  void init_kalman(const cv::RotatedRect &det);
  void predict();
  void update(const cv::RotatedRect &det, int frame, float conf);
  cv::RotatedRect get_predicted_bbox() const;
};

class ByteTracker : public ImageProcessor {
public:
  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "ByteTracker"; }
  void configure(const json &config) override;
  bool is_stateful() const override { return true; }

  void reset();

private:
  void update(std::vector<Detection> &detections, int frame_id);

  std::vector<std::vector<float>>
  iou_distance(const std::vector<STrack> &tracks,
               const std::vector<std::pair<Detection, int>> &dets);

  // 关键点：这里做全局最优匹配，避免贪心分配导致 ID 抖动。
  std::vector<std::pair<int, int>>
  linear_assignment(const std::vector<std::vector<float>> &cost_matrix,
                    float thresh);

  float track_thresh_ = 0.5f;
  float high_thresh_ = 0.6f;
  float match_thresh_ = 0.8f;
  int max_time_lost_ = 30;
  int min_hits_ = 3;

  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  int frame_id_ = 0;
  int next_id_ = 1;
};

} // namespace yolo_edge
