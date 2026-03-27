#pragma once

#include "processor_factory.hpp"
#include "processing_pipeline.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace yolo_edge {

class Session {
public:
  std::string session_id;
  ProcessingPipeline pipeline;
  std::chrono::steady_clock::time_point last_active;
  json config;

  Session() = default;

  Session(std::string id, ProcessingPipeline pipe, json cfg = {})
      : session_id(std::move(id)), pipeline(std::move(pipe)),
        last_active(std::chrono::steady_clock::now()), config(std::move(cfg)) {}

  void touch() { last_active = std::chrono::steady_clock::now(); }

  bool is_expired(std::chrono::seconds timeout) const {
    auto now = std::chrono::steady_clock::now();
    return (now - last_active) > timeout;
  }

  mutable std::mutex pipeline_mutex;
  mutable std::mutex tracker_mutex;
  std::condition_variable tracker_cv;
  int next_tracker_frame = 1;

  void wait_for_turn(int frame_id) {
    std::unique_lock<std::mutex> lock(tracker_mutex);
    // 关键点：客户端重启后 frame_id 可能从 1 重新开始，这里只在“明确重启”时回卷窗口，
    // 避免乱序旧帧把 next_tracker_frame 拉回去导致追踪顺序被破坏。
    if (frame_id == 1 && next_tracker_frame > 1) {
      next_tracker_frame = frame_id;
      tracker_cv.notify_all();
    }
    tracker_cv.wait(lock, [&] { return frame_id == next_tracker_frame; });
  }

  void advance_turn() {
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      next_tracker_frame++;
    }
    tracker_cv.notify_all();
  }
};

class SessionManager {
public:
  Session &get_or_create(const std::string &session_id, const json &config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (++access_count_ % 100 == 0) {
      cleanup_expired_unlocked(default_timeout_);
    }

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
      it->second->touch();
      return *it->second;
    }

    auto pipeline = ProcessorFactory::instance().create_pipeline(config);
    auto session =
        std::make_shared<Session>(session_id, std::move(pipeline), config);
    sessions_.emplace(session_id, session);

    fprintf(stderr, "[INFO] Created new session: %s\n", session_id.c_str());
    return *session;
  }

  Session *get(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
      it->second->touch();
      return it->second.get();
    }
    return nullptr;
  }

  void remove(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.erase(session_id);
    fprintf(stderr, "[INFO] Removed session: %s\n", session_id.c_str());
  }

  size_t cleanup_expired(std::chrono::seconds timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    return cleanup_expired_unlocked(timeout);
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
  }

private:
  size_t cleanup_expired_unlocked(std::chrono::seconds timeout) {
    size_t removed = 0;
    for (auto it = sessions_.begin(); it != sessions_.end();) {
      if (it->second->is_expired(timeout)) {
        fprintf(stderr, "[INFO] Session expired: %s\n", it->first.c_str());
        it = sessions_.erase(it);
        ++removed;
      } else {
        ++it;
      }
    }
    return removed;
  }

  mutable std::mutex mutex_;
  int access_count_ = 0;
  static constexpr std::chrono::seconds default_timeout_{300};
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

} // namespace yolo_edge
