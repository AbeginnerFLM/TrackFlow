#pragma once

#include "processing_pipeline.hpp"
#include <chrono>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace yolo_edge {

/**
 * 会话信息 (预留多客户端支持)
 * 每个客户端连接对应一个Session
 */
class Session {
public:
  std::string session_id;
  ProcessingPipeline pipeline;
  std::chrono::steady_clock::time_point last_active;

  // 会话配置
  json config;

  Session() = default;

  Session(std::string id, ProcessingPipeline pipe, json cfg = {})
      : session_id(std::move(id)), pipeline(std::move(pipe)),
        last_active(std::chrono::steady_clock::now()), config(std::move(cfg)) {}

  /**
   * 更新活跃时间
   */
  void touch() { last_active = std::chrono::steady_clock::now(); }

  /**
   * 检查是否过期
   */
  bool is_expired(std::chrono::seconds timeout) const {
    auto now = std::chrono::steady_clock::now();
    return (now - last_active) > timeout;
  }

  // Mutex for serializing pipeline execution (required for stateful trackers)
  mutable std::mutex pipeline_mutex;
};

/**
 * 会话管理器
 * 管理多个客户端的会话状态
 */
class SessionManager {
public:
  /**
   * 获取或创建会话
   */
  /**
   * 获取或创建会话
   */
  Session &get_or_create(const std::string &session_id, const json &config) {
    std::lock_guard<std::mutex> lock(mutex_);

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

  /**
   * 获取已存在的会话
   */
  Session *get(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
      it->second->touch();
      return it->second.get();
    }
    return nullptr;
  }

  /**
   * 移除会话
   */
  void remove(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.erase(session_id);
    fprintf(stderr, "[INFO] Removed session: %s\n", session_id.c_str());
  }

  /**
   * 清理过期会话
   */
  size_t cleanup_expired(std::chrono::seconds timeout) {
    std::lock_guard<std::mutex> lock(mutex_);

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

  /**
   * 获取当前会话数量
   */
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

} // namespace yolo_edge
