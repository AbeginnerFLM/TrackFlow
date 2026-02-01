#include "processors/byte_tracker.hpp"
#include "core/processor_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>

namespace yolo_edge {

// ============================================================================
// STrack 实现
// ============================================================================

void STrack::init_kalman(const cv::RotatedRect &det) {
  // 状态向量: [cx, cy, w, h, vx, vy, vw, vh]
  // 测量向量: [cx, cy, w, h]
  kf = cv::KalmanFilter(8, 4, 0);

  // 状态转移矩阵 (匀速模型)
  kf.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
  kf.transitionMatrix.at<float>(0, 4) = 1; // cx += vx
  kf.transitionMatrix.at<float>(1, 5) = 1; // cy += vy
  kf.transitionMatrix.at<float>(2, 6) = 1; // w += vw
  kf.transitionMatrix.at<float>(3, 7) = 1; // h += vh

  // 测量矩阵
  kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
  kf.measurementMatrix.at<float>(0, 0) = 1;
  kf.measurementMatrix.at<float>(1, 1) = 1;
  kf.measurementMatrix.at<float>(2, 2) = 1;
  kf.measurementMatrix.at<float>(3, 3) = 1;

  // 过程噪声
  cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
  kf.processNoiseCov.at<float>(4, 4) = 1e-1;
  kf.processNoiseCov.at<float>(5, 5) = 1e-1;
  kf.processNoiseCov.at<float>(6, 6) = 1e-2;
  kf.processNoiseCov.at<float>(7, 7) = 1e-2;

  // 测量噪声
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

  // 后验误差协方差
  cv::setIdentity(kf.errorCovPost, cv::Scalar(1));
  kf.errorCovPost.at<float>(4, 4) = 1e4;
  kf.errorCovPost.at<float>(5, 5) = 1e4;
  kf.errorCovPost.at<float>(6, 6) = 1e4;
  kf.errorCovPost.at<float>(7, 7) = 1e4;

  // 初始状态
  kf.statePost.at<float>(0) = det.center.x;
  kf.statePost.at<float>(1) = det.center.y;
  kf.statePost.at<float>(2) = det.size.width;
  kf.statePost.at<float>(3) = det.size.height;

  bbox = det;
  kf_initialized = true;
}

void STrack::predict() {
  if (!kf_initialized)
    return;

  cv::Mat pred = kf.predict();
  bbox.center.x = pred.at<float>(0);
  bbox.center.y = pred.at<float>(1);
  bbox.size.width = std::max(pred.at<float>(2), 1.0f);
  bbox.size.height = std::max(pred.at<float>(3), 1.0f);

  time_since_update++;
}

void STrack::update(const cv::RotatedRect &det, int frame, float conf) {
  bbox = det;
  score = conf;
  frame_id = frame;
  hits++;
  time_since_update = 0;

  // 更新轨迹
  trajectory.push_back(det.center);
  if (trajectory.size() > 100) {
    trajectory.erase(trajectory.begin());
  }

  // 卡尔曼更新
  if (kf_initialized) {
    cv::Mat measurement = (cv::Mat_<float>(4, 1) << det.center.x, det.center.y,
                           det.size.width, det.size.height);
    kf.correct(measurement);
  }
}

cv::RotatedRect STrack::get_predicted_bbox() const { return bbox; }

// ============================================================================
// ByteTracker 实现
// ============================================================================

void ByteTracker::configure(const json &config) {
  track_thresh_ = config.value("track_thresh", 0.5f);
  high_thresh_ = config.value("high_thresh", 0.6f);
  match_thresh_ = config.value("match_thresh", 0.8f);
  max_time_lost_ = config.value("max_time_lost", 30);
  min_hits_ = config.value("min_hits", 3);

  spdlog::debug("ByteTracker configured: track_thresh={}, high_thresh={}, "
                "match_thresh={}",
                track_thresh_, high_thresh_, match_thresh_);
}

void ByteTracker::reset() {
  tracked_stracks_.clear();
  lost_stracks_.clear();
  frame_id_ = 0;
  next_id_ = 1;
}

bool ByteTracker::process(ProcessingContext &ctx) {
  using Clock = std::chrono::high_resolution_clock;
  auto start = Clock::now();

  update(ctx.detections, ctx.frame_id);

  auto end = Clock::now();
  ctx.track_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  spdlog::debug("ByteTracker: Tracking {} objects in {:.2f}ms",
                ctx.detections.size(), ctx.track_time_ms);

  return true;
}

void ByteTracker::update(std::vector<Detection> &detections, int frame_id) {
  frame_id_ = frame_id;

  // ========================================
  // Step 1: 将检测分为高分和低分
  // ========================================
  std::vector<std::pair<Detection, int>>
      dets_high; // {detection, original_index}
  std::vector<std::pair<Detection, int>> dets_low;

  for (size_t i = 0; i < detections.size(); ++i) {
    if (detections[i].confidence >= track_thresh_) {
      dets_high.push_back({detections[i], static_cast<int>(i)});
    } else {
      dets_low.push_back({detections[i], static_cast<int>(i)});
    }
  }

  // ========================================
  // Step 2: 预测所有跟踪对象的新位置
  // ========================================
  for (auto &track : tracked_stracks_) {
    track.predict();
  }
  for (auto &track : lost_stracks_) {
    track.predict();
  }

  // ========================================
  // Step 3: 高分检测与已跟踪对象匹配
  // ========================================
  std::vector<std::pair<int, int>> matches_high; // {track_idx, det_idx}
  std::vector<int> unmatched_tracks_high;
  std::vector<int> unmatched_dets_high;

  if (!tracked_stracks_.empty() && !dets_high.empty()) {
    auto cost = iou_distance(tracked_stracks_, dets_high);
    matches_high = linear_assignment(cost, match_thresh_);

    // 找出未匹配的
    std::vector<bool> track_matched(tracked_stracks_.size(), false);
    std::vector<bool> det_matched(dets_high.size(), false);

    for (auto &[ti, di] : matches_high) {
      track_matched[ti] = true;
      det_matched[di] = true;
    }

    for (size_t i = 0; i < tracked_stracks_.size(); ++i) {
      if (!track_matched[i])
        unmatched_tracks_high.push_back(i);
    }
    for (size_t i = 0; i < dets_high.size(); ++i) {
      if (!det_matched[i])
        unmatched_dets_high.push_back(i);
    }
  } else {
    for (size_t i = 0; i < tracked_stracks_.size(); ++i) {
      unmatched_tracks_high.push_back(i);
    }
    for (size_t i = 0; i < dets_high.size(); ++i) {
      unmatched_dets_high.push_back(i);
    }
  }

  // 更新已匹配的跟踪对象
  for (auto &[ti, di] : matches_high) {
    tracked_stracks_[ti].update(dets_high[di].first.obb, frame_id_,
                                dets_high[di].first.confidence);
    tracked_stracks_[ti].state = TrackState::Tracked;
    tracked_stracks_[ti].class_id = dets_high[di].first.class_id;
  }

  // ========================================
  // Step 4: 低分检测与未匹配的跟踪对象匹配
  // ========================================
  std::vector<STrack> remain_tracks;
  for (int idx : unmatched_tracks_high) {
    remain_tracks.push_back(tracked_stracks_[idx]);
  }

  std::vector<std::pair<int, int>> matches_low;
  std::vector<int> unmatched_tracks_low;

  if (!remain_tracks.empty() && !dets_low.empty()) {
    auto cost = iou_distance(remain_tracks, dets_low);
    matches_low = linear_assignment(cost, 0.5f); // 低分匹配阈值更宽松

    std::vector<bool> track_matched(remain_tracks.size(), false);
    for (auto &[ti, di] : matches_low) {
      track_matched[ti] = true;
      // 更新tracked_stracks_中的对应项
      int orig_idx = unmatched_tracks_high[ti];
      tracked_stracks_[orig_idx].update(dets_low[di].first.obb, frame_id_,
                                        dets_low[di].first.confidence);
      tracked_stracks_[orig_idx].state = TrackState::Tracked;
    }

    for (size_t i = 0; i < remain_tracks.size(); ++i) {
      if (!track_matched[i])
        unmatched_tracks_low.push_back(unmatched_tracks_high[i]);
    }
  } else {
    unmatched_tracks_low = unmatched_tracks_high;
  }

  // ========================================
  // Step 5: 处理未匹配的跟踪对象 (标记为Lost)
  // ========================================
  for (int idx : unmatched_tracks_low) {
    if (tracked_stracks_[idx].state != TrackState::Lost) {
      tracked_stracks_[idx].state = TrackState::Lost;
      lost_stracks_.push_back(tracked_stracks_[idx]);
    }
  }

  // ========================================
  // Step 6: 未匹配的高分检测与Lost对象匹配
  // ========================================
  std::vector<std::pair<Detection, int>> remain_dets_high;
  for (int idx : unmatched_dets_high) {
    remain_dets_high.push_back(dets_high[idx]);
  }

  std::vector<int> reactivated_lost;

  if (!lost_stracks_.empty() && !remain_dets_high.empty()) {
    auto cost = iou_distance(lost_stracks_, remain_dets_high);
    auto matches = linear_assignment(cost, match_thresh_);

    std::vector<bool> det_matched(remain_dets_high.size(), false);

    for (auto &[ti, di] : matches) {
      lost_stracks_[ti].update(remain_dets_high[di].first.obb, frame_id_,
                               remain_dets_high[di].first.confidence);
      lost_stracks_[ti].state = TrackState::Tracked;
      lost_stracks_[ti].class_id = remain_dets_high[di].first.class_id;
      reactivated_lost.push_back(ti);
      det_matched[di] = true;
    }

    // 更新未匹配检测列表
    unmatched_dets_high.clear();
    for (size_t i = 0; i < remain_dets_high.size(); ++i) {
      if (!det_matched[i]) {
        // 找到原始索引
        for (size_t j = 0; j < dets_high.size(); ++j) {
          if (dets_high[j].second == remain_dets_high[i].second) {
            unmatched_dets_high.push_back(j);
            break;
          }
        }
      }
    }
  }

  // ========================================
  // Step 7: 创建新轨迹
  // ========================================
  for (int idx : unmatched_dets_high) {
    const auto &det = dets_high[idx].first;
    if (det.confidence >= high_thresh_) {
      STrack new_track;
      new_track.track_id = next_id_++;
      new_track.state = TrackState::New;
      new_track.frame_id = frame_id_;
      new_track.start_frame = frame_id_;
      new_track.hits = 1;
      new_track.score = det.confidence;
      new_track.class_id = det.class_id;
      new_track.init_kalman(det.obb);
      tracked_stracks_.push_back(new_track);
    }
  }

  // ========================================
  // Step 8: 整理跟踪列表
  // ========================================

  // 重新激活的Lost对象加回tracked
  std::sort(reactivated_lost.rbegin(), reactivated_lost.rend());
  for (int idx : reactivated_lost) {
    tracked_stracks_.push_back(lost_stracks_[idx]);
    lost_stracks_.erase(lost_stracks_.begin() + idx);
  }

  // 移除Lost太久的对象
  lost_stracks_.erase(std::remove_if(lost_stracks_.begin(), lost_stracks_.end(),
                                     [this](const STrack &t) {
                                       return t.time_since_update >
                                              max_time_lost_;
                                     }),
                      lost_stracks_.end());

  // 从tracked中移除Lost状态的
  tracked_stracks_.erase(std::remove_if(tracked_stracks_.begin(),
                                        tracked_stracks_.end(),
                                        [](const STrack &t) {
                                          return t.state == TrackState::Lost ||
                                                 t.state == TrackState::Removed;
                                        }),
                         tracked_stracks_.end());

  // ========================================
  // Step 9: 更新检测结果的track_id
  // ========================================
  for (auto &det : detections) {
    det.track_id = -1; // 默认无跟踪
  }

  // 高分匹配
  for (auto &[ti, di] : matches_high) {
    int orig_idx = dets_high[di].second;
    if (tracked_stracks_[ti].hits >= min_hits_) {
      detections[orig_idx].track_id = tracked_stracks_[ti].track_id;
    }
  }

  // 低分匹配
  for (auto &[ti, di] : matches_low) {
    int orig_ti = unmatched_tracks_high[ti];
    int orig_idx = dets_low[di].second;
    if (tracked_stracks_[orig_ti].hits >= min_hits_) {
      detections[orig_idx].track_id = tracked_stracks_[orig_ti].track_id;
    }
  }

  // 新创建的轨迹
  for (auto &track : tracked_stracks_) {
    if (track.start_frame == frame_id_ && track.hits >= min_hits_) {
      // 找到对应的检测
      for (auto &det : detections) {
        if (det.track_id == -1 &&
            std::abs(det.obb.center.x - track.bbox.center.x) < 1.0f &&
            std::abs(det.obb.center.y - track.bbox.center.y) < 1.0f) {
          det.track_id = track.track_id;
          break;
        }
      }
    }
  }
}

// ============================================================================
// IOU距离计算
// ============================================================================
namespace {

float rotated_iou(const cv::RotatedRect &a, const cv::RotatedRect &b) {
  std::vector<cv::Point2f> inter_pts;
  int ret = cv::rotatedRectangleIntersection(a, b, inter_pts);

  if (ret == cv::INTERSECT_NONE || inter_pts.size() < 3) {
    return 0.0f;
  }

  cv::convexHull(inter_pts, inter_pts);
  float inter_area = static_cast<float>(cv::contourArea(inter_pts));

  float area_a = a.size.width * a.size.height;
  float area_b = b.size.width * b.size.height;
  float union_area = area_a + area_b - inter_area;

  if (union_area < 1e-6f)
    return 0.0f;

  return inter_area / union_area;
}

} // anonymous namespace

std::vector<std::vector<float>>
ByteTracker::iou_distance(const std::vector<STrack> &tracks,
                          const std::vector<std::pair<Detection, int>> &dets) {

  std::vector<std::vector<float>> cost(tracks.size(),
                                       std::vector<float>(dets.size(), 1.0f));

  for (size_t i = 0; i < tracks.size(); ++i) {
    for (size_t j = 0; j < dets.size(); ++j) {
      float iou = rotated_iou(tracks[i].bbox, dets[j].first.obb);
      cost[i][j] = 1.0f - iou; // 转换为代价
    }
  }

  return cost;
}

// ============================================================================
// 匈牙利算法 (简化贪心版本)
// ============================================================================
std::vector<std::pair<int, int>> ByteTracker::linear_assignment(
    const std::vector<std::vector<float>> &cost_matrix, float thresh) {

  if (cost_matrix.empty() || cost_matrix[0].empty()) {
    return {};
  }

  size_t num_tracks = cost_matrix.size();
  size_t num_dets = cost_matrix[0].size();

  std::vector<std::pair<int, int>> matches;
  std::vector<bool> track_matched(num_tracks, false);
  std::vector<bool> det_matched(num_dets, false);

  // 贪心匹配: 每次选择代价最小的配对
  while (true) {
    float min_cost = thresh;
    int best_track = -1;
    int best_det = -1;

    for (size_t i = 0; i < num_tracks; ++i) {
      if (track_matched[i])
        continue;
      for (size_t j = 0; j < num_dets; ++j) {
        if (det_matched[j])
          continue;
        if (cost_matrix[i][j] < min_cost) {
          min_cost = cost_matrix[i][j];
          best_track = i;
          best_det = j;
        }
      }
    }

    if (best_track < 0)
      break;

    matches.push_back({best_track, best_det});
    track_matched[best_track] = true;
    det_matched[best_det] = true;
  }

  return matches;
}

// 注册处理器
REGISTER_PROCESSOR("tracker", ByteTracker);

} // namespace yolo_edge
