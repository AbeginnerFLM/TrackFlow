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
  void set_server_config(const json &config) { server_config_ = config; }

private:
  void handle_message(void *ws, std::string_view message, bool is_binary);
  json build_response(const ProcessingContext &ctx, bool success);
  json build_error(const std::string &message,
                   const std::string &request_id = "");
  json build_error(const std::string &code, const std::string &message,
                   const std::string &request_id, const json &extra) const;
  json sanitize_client_overrides(const json &request) const;
  json build_pipeline_config(const json &request) const;
  std::vector<std::string> build_pipeline(const json &features,
                                          const json &config) const;
  static void merge_defaults(json &target, const json &defaults);
  static void clamp_number(json &obj, const char *key, double min_value,
                           double max_value);

  int port_;
  ThreadPool &pool_;
  SessionManager sessions_;
  std::chrono::seconds session_timeout_{300};
  json server_config_;

  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool running_ = false;
};

} // namespace yolo_edge
