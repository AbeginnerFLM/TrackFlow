#include "processors/byte_tracker.hpp"
#include "core/processor_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace yolo_edge {

// ============================================================================
// STrack 实现
// ============================================================================

void STrack::init_kalman(const cv::RotatedRect &det) {
  kf = cv::KalmanFilter(8, 4, 0);

  kf.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
  kf.transitionMatrix.at<float>(0, 4) = 1;
  kf.transitionMatrix.at<float>(1, 5) = 1;
  kf.transitionMatrix.at<float>(2, 6) = 1;
  kf.transitionMatrix.at<float>(3, 7) = 1;

  kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
  kf.measurementMatrix.at<float>(0, 0) = 1;
  kf.measurementMatrix.at<float>(1, 1) = 1;
  kf.measurementMatrix.at<float>(2, 2) = 1;
  kf.measurementMatrix.at<float>(3, 3) = 1;

  cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
  kf.processNoiseCov.at<float>(4, 4) = 1e-1;
  kf.processNoiseCov.at<float>(5, 5) = 1e-1;
  kf.processNoiseCov.at<float>(6, 6) = 1e-2;
  kf.processNoiseCov.at<float>(7, 7) = 1e-2;

  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

  cv::setIdentity(kf.errorCovPost, cv::Scalar(1));
  kf.errorCovPost.at<float>(4, 4) = 1e4;
  kf.errorCovPost.at<float>(5, 5) = 1e4;
  kf.errorCovPost.at<float>(6, 6) = 1e4;
  kf.errorCovPost.at<float>(7, 7) = 1e4;

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

  // 使用环形缓冲区 (O(1) 替代 vector erase O(N))
  add_trajectory_point(det.center);

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
}

void ByteTracker::reset() {
  tracked_stracks_.clear();
  lost_stracks_.clear();
  frame_id_ = 0;
  next_id_ = 1;
}

bool ByteTracker::process(ProcessingContext &ctx) {
  try {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    update(ctx.detections, ctx.frame_id);

    auto end = Clock::now();
    ctx.track_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] ByteTracker: %s\n", e.what());
    return false;
  }
}

void ByteTracker::update(std::vector<Detection> &detections, int frame_id) {
  frame_id_ = frame_id;

  // Step 1: 分离高分/低分检测
  std::vector<std::pair<Detection, int>> dets_high, dets_low;

  for (size_t i = 0; i < detections.size(); ++i) {
    if (detections[i].confidence >= track_thresh_)
      dets_high.push_back({detections[i], static_cast<int>(i)});
    else
      dets_low.push_back({detections[i], static_cast<int>(i)});
  }

  // Step 2: 预测
  for (auto &track : tracked_stracks_)
    track.predict();
  for (auto &track : lost_stracks_)
    track.predict();

  // Step 3: 高分检测匹配
  std::vector<std::pair<int, int>> matches_high;
  std::vector<int> unmatched_tracks_high;
  std::vector<int> unmatched_dets_high;

  if (!tracked_stracks_.empty() && !dets_high.empty()) {
    auto cost = iou_distance(tracked_stracks_, dets_high);
    matches_high = linear_assignment(cost, match_thresh_);

    std::vector<bool> track_matched(tracked_stracks_.size(), false);
    std::vector<bool> det_matched(dets_high.size(), false);

    for (auto &[ti, di] : matches_high) {
      track_matched[ti] = true;
      det_matched[di] = true;
    }

    for (size_t i = 0; i < tracked_stracks_.size(); ++i)
      if (!track_matched[i])
        unmatched_tracks_high.push_back(i);
    for (size_t i = 0; i < dets_high.size(); ++i)
      if (!det_matched[i])
        unmatched_dets_high.push_back(i);
  } else {
    for (size_t i = 0; i < tracked_stracks_.size(); ++i)
      unmatched_tracks_high.push_back(i);
    for (size_t i = 0; i < dets_high.size(); ++i)
      unmatched_dets_high.push_back(i);
  }

  // 更新匹配
  for (auto &[ti, di] : matches_high) {
    tracked_stracks_[ti].update(dets_high[di].first.obb, frame_id_,
                                dets_high[di].first.confidence);
    tracked_stracks_[ti].state = TrackState::Tracked;
    tracked_stracks_[ti].class_id = dets_high[di].first.class_id;
  }

  // Step 4: 低分检测匹配
  std::vector<STrack> remain_tracks;
  for (int idx : unmatched_tracks_high)
    remain_tracks.push_back(tracked_stracks_[idx]);

  std::vector<std::pair<int, int>> matches_low;
  std::vector<int> unmatched_tracks_low;

  if (!remain_tracks.empty() && !dets_low.empty()) {
    auto cost = iou_distance(remain_tracks, dets_low);
    matches_low = linear_assignment(cost, 0.5f);

    std::vector<bool> track_matched(remain_tracks.size(), false);
    for (auto &[ti, di] : matches_low) {
      track_matched[ti] = true;
      int orig_idx = unmatched_tracks_high[ti];
      tracked_stracks_[orig_idx].update(dets_low[di].first.obb, frame_id_,
                                        dets_low[di].first.confidence);
      tracked_stracks_[orig_idx].state = TrackState::Tracked;
    }

    for (size_t i = 0; i < remain_tracks.size(); ++i)
      if (!track_matched[i])
        unmatched_tracks_low.push_back(unmatched_tracks_high[i]);
  } else {
    unmatched_tracks_low = unmatched_tracks_high;
  }

  // Step 5: 未匹配 → Lost
  for (int idx : unmatched_tracks_low) {
    if (tracked_stracks_[idx].state != TrackState::Lost) {
      tracked_stracks_[idx].state = TrackState::Lost;
      lost_stracks_.push_back(tracked_stracks_[idx]);
    }
  }

  // Step 6: 与 Lost 匹配
  std::vector<std::pair<Detection, int>> remain_dets_high;
  for (int idx : unmatched_dets_high)
    remain_dets_high.push_back(dets_high[idx]);

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

    unmatched_dets_high.clear();
    for (size_t i = 0; i < remain_dets_high.size(); ++i) {
      if (!det_matched[i]) {
        for (size_t j = 0; j < dets_high.size(); ++j) {
          if (dets_high[j].second == remain_dets_high[i].second) {
            unmatched_dets_high.push_back(j);
            break;
          }
        }
      }
    }
  }

  // Step 7: 新轨迹
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

  // Step 8: 整理
  std::sort(reactivated_lost.rbegin(), reactivated_lost.rend());
  for (int idx : reactivated_lost) {
    tracked_stracks_.push_back(lost_stracks_[idx]);
    lost_stracks_.erase(lost_stracks_.begin() + idx);
  }

  lost_stracks_.erase(
      std::remove_if(lost_stracks_.begin(), lost_stracks_.end(),
                     [this](const STrack &t) {
                       return t.time_since_update > max_time_lost_;
                     }),
      lost_stracks_.end());

  tracked_stracks_.erase(
      std::remove_if(tracked_stracks_.begin(), tracked_stracks_.end(),
                     [](const STrack &t) {
                       return t.state == TrackState::Lost ||
                              t.state == TrackState::Removed;
                     }),
      tracked_stracks_.end());

  // Step 9: 更新 track_id
  for (auto &det : detections)
    det.track_id = -1;

  for (auto &[ti, di] : matches_high) {
    int orig_idx = dets_high[di].second;
    if (tracked_stracks_[ti].hits >= min_hits_)
      detections[orig_idx].track_id = tracked_stracks_[ti].track_id;
  }

  for (auto &[ti, di] : matches_low) {
    int orig_ti = unmatched_tracks_high[ti];
    int orig_idx = dets_low[di].second;
    if (tracked_stracks_[orig_ti].hits >= min_hits_)
      detections[orig_idx].track_id = tracked_stracks_[orig_ti].track_id;
  }

  for (auto &track : tracked_stracks_) {
    if (track.start_frame == frame_id_ && track.hits >= min_hits_) {
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
// IOU 距离
// ============================================================================
namespace {

float rotated_iou(const cv::RotatedRect &a, const cv::RotatedRect &b) {
  std::vector<cv::Point2f> inter_pts, hull_pts;
  int ret = cv::rotatedRectangleIntersection(a, b, inter_pts);

  if (ret == cv::INTERSECT_NONE || inter_pts.size() < 3)
    return 0.0f;

  cv::convexHull(inter_pts, hull_pts);
  float inter_area = static_cast<float>(cv::contourArea(hull_pts));

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
      cost[i][j] = 1.0f - iou;
    }
  }

  return cost;
}

// ============================================================================
// Hungarian/Munkres 算法 (Kuhn-Munkres, O(N^3))
// ============================================================================
std::vector<std::pair<int, int>> ByteTracker::linear_assignment(
    const std::vector<std::vector<float>> &cost_matrix, float thresh) {

  if (cost_matrix.empty() || cost_matrix[0].empty())
    return {};

  int n = cost_matrix.size();    // rows (tracks)
  int m = cost_matrix[0].size(); // cols (dets)
  int sz = std::max(n, m);

  // Pad to square matrix with threshold values
  std::vector<std::vector<float>> C(sz, std::vector<float>(sz, thresh));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      C[i][j] = cost_matrix[i][j];

  // Hungarian algorithm (Kuhn-Munkres)
  const float INF = 1e9f;
  std::vector<float> u(sz + 1, 0), v(sz + 1, 0);
  std::vector<int> p(sz + 1, 0), way(sz + 1, 0);

  for (int i = 1; i <= sz; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(sz + 1, INF);
    std::vector<bool> used(sz + 1, false);

    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      float delta = INF;

      for (int j = 1; j <= sz; ++j) {
        if (!used[j]) {
          float cur = C[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }

      for (int j = 0; j <= sz; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }

      j0 = j1;
    } while (p[j0] != 0);

    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  // Extract matches
  std::vector<std::pair<int, int>> matches;
  for (int j = 1; j <= sz; ++j) {
    int i = p[j] - 1;
    int jj = j - 1;
    if (i < n && jj < m && cost_matrix[i][jj] < thresh)
      matches.push_back({i, jj});
  }

  return matches;
}

REGISTER_PROCESSOR("tracker", ByteTracker);

} // namespace yolo_edge
