#pragma once

#include "core/session.hpp"
#include "utils/thread_pool.hpp"
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace yolo_edge {

using json = nlohmann::json;

class WebSocketServer {
public:
  WebSocketServer(int port, ThreadPool &pool);
  ~WebSocketServer();

  void run();
  void stop();
  void set_session_timeout(std::chrono::seconds timeout);

private:
  void handle_message(void *ws, std::string_view message, bool is_binary);
  json build_response(const ProcessingContext &ctx, bool success);
  json build_error(const std::string &message,
                   const std::string &request_id = "");

  int port_;
  ThreadPool &pool_;
  SessionManager sessions_;
  std::chrono::seconds session_timeout_{300};

  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool running_ = false;
};

} // namespace yolo_edge
