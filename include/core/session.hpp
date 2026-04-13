#pragma once

#include "processor_factory.hpp"
#include "processing_pipeline.hpp"
#include <atomic>
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

  bool try_acquire_request_slot(size_t max_outstanding) {
    if (max_outstanding == 0) {
      active_requests_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    size_t current = active_requests_.load(std::memory_order_relaxed);
    while (true) {
      if (current >= max_outstanding) {
        return false;
      }
      if (active_requests_.compare_exchange_weak(
              current, current + 1, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  void release_request_slot() {
    active_requests_.fetch_sub(1, std::memory_order_relaxed);
  }

  size_t active_requests() const {
    return active_requests_.load(std::memory_order_relaxed);
  }

  void retire() { retired_.store(true, std::memory_order_relaxed); }
  bool is_retired() const {
    return retired_.load(std::memory_order_relaxed);
  }

  mutable std::mutex tracker_mutex;
  std::condition_variable tracker_cv;
  int next_tracker_frame = 1;

  bool wait_for_turn(int frame_id) {
    std::unique_lock<std::mutex> lock(tracker_mutex);
    if (frame_id <= 0) {
      return false;
    }
    if (frame_id == 1 && next_tracker_frame > 1) {
      next_tracker_frame = frame_id;
      tracker_cv.notify_all();
    }
    while (frame_id != next_tracker_frame) {
      if (frame_id < next_tracker_frame) {
        return false;
      }
      if (tracker_cv.wait_for(lock, std::chrono::milliseconds(1200)) ==
          std::cv_status::timeout) {
        if (frame_id > next_tracker_frame) {
          fprintf(stderr,
                  "[WARN] Session %s tracker gap: expected frame %d, got %d. "
                  "Skip missing frame(s).\n",
                  session_id.c_str(), next_tracker_frame, frame_id);
          next_tracker_frame = frame_id;
          break;
        }
      }
    }
    return frame_id == next_tracker_frame;
  }

  void advance_turn() {
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      next_tracker_frame++;
    }
    tracker_cv.notify_all();
  }

private:
  std::atomic<size_t> active_requests_{0};
  std::atomic<bool> retired_{false};
};

class SessionManager {
public:
  std::shared_ptr<Session> get_or_create(const std::string &session_id,
                                         const json &config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (++access_count_ % 100 == 0) {
      cleanup_expired_unlocked(default_timeout_);
    }

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
      if (it->second->config != config) {
        it->second->retire();
        sessions_.erase(it);
      } else {
        it->second->touch();
        return it->second;
      }
    }

    auto pipeline = ProcessorFactory::instance().create_pipeline(config);
    auto session =
        std::make_shared<Session>(session_id, std::move(pipeline), config);
    sessions_.emplace(session_id, session);

    fprintf(stderr, "[INFO] Created new session: %s\n", session_id.c_str());
    return session;
  }

  std::shared_ptr<Session> get(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
      it->second->touch();
      return it->second;
    }
    return nullptr;
  }

  void retire(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
      return;
    }

    it->second->retire();
    sessions_.erase(it);
    fprintf(stderr, "[INFO] Retired session: %s\n", session_id.c_str());
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
      const auto &session = it->second;
      if (session->active_requests() == 0 && session->is_expired(timeout)) {
        session->retire();
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
