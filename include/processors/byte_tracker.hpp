#pragma once

#include "core/image_processor.hpp"
#include <map>
#include <opencv2/video/tracking.hpp>
#include <vector>

namespace yolo_edge {

/**
 * 跟踪对象状态
 */
enum class TrackState {
  New = 0,     // 新建
  Tracked = 1, // 跟踪中
  Lost = 2,    // 丢失
  Removed = 3  // 已移除
};

/**
 * 单个跟踪对象
 */
struct STrack {
  int track_id = 0;
  TrackState state = TrackState::New;
  cv::RotatedRect bbox;      // 当前边界框
  int frame_id = 0;          // 最后更新帧
  int start_frame = 0;       // 开始帧
  int hits = 0;              // 连续命中次数
  int time_since_update = 0; // 距上次更新的帧数
  float score = 0.0f;        // 置信度
  int class_id = 0;          // 类别ID

  // 卡尔曼滤波器状态
  cv::KalmanFilter kf;
  bool kf_initialized = false;

  // 历史轨迹
  std::vector<cv::Point2f> trajectory;

  // 初始化卡尔曼滤波器
  void init_kalman(const cv::RotatedRect &det);

  // 预测下一状态
  void predict();

  // 用检测结果更新
  void update(const cv::RotatedRect &det, int frame, float conf);

  // 获取预测位置
  cv::RotatedRect get_predicted_bbox() const;
};

/**
 * ByteTrack跟踪器
 * 实现多目标跟踪算法
 */
class ByteTracker : public ImageProcessor {
public:
  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "ByteTracker"; }
  void configure(const json &config) override;
  bool is_stateful() const override { return true; } // 有状态!

  /**
   * 重置跟踪器状态
   */
  void reset();

private:
  // 核心跟踪更新
  void update(std::vector<Detection> &detections, int frame_id);

  // 计算IOU距离矩阵
  std::vector<std::vector<float>>
  iou_distance(const std::vector<STrack> &tracks,
               const std::vector<std::pair<Detection, int>> &dets);

  // 匈牙利算法匹配
  std::vector<std::pair<int, int>>
  linear_assignment(const std::vector<std::vector<float>> &cost_matrix,
                    float thresh);

  // 配置参数
  float track_thresh_ = 0.5f; // 高置信度阈值
  float high_thresh_ = 0.6f;  // 新轨迹阈值
  float match_thresh_ = 0.8f; // IOU匹配阈值
  int max_time_lost_ = 30;    // 最大丢失帧数
  int min_hits_ = 3;          // 激活所需最小命中数

  // 跟踪状态
  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  int frame_id_ = 0;
  int next_id_ = 1;
};

} // namespace yolo_edge
