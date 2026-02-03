#pragma once

#include "core/processor_factory.hpp"
#include "core/session.hpp"
#include "utils/thread_pool.hpp"
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace yolo_edge {

using json = nlohmann::json;

/**
 * WebSocket服务端
 * 处理客户端连接和消息
 */
class WebSocketServer {
public:
  /**
   * 构造函数
   * @param port 监听端口
   * @param pool 线程池引用
   */
  WebSocketServer(int port, ThreadPool &pool);
  ~WebSocketServer();

  /**
   * 启动服务 (阻塞)
   */
  void run();

  /**
   * 停止服务
   */
  void stop();

  /**
   * 设置会话超时时间
   */
  void set_session_timeout(std::chrono::seconds timeout);

private:
  // 处理收到的消息
  void handle_message(void *ws, std::string_view message, bool is_binary);

  // 构建响应JSON
  json build_response(const ProcessingContext &ctx, const json &request,
                      bool success);

  // 构建错误响应
  json build_error(const std::string &message,
                   const std::string &request_id = "");

  int port_;
  ThreadPool &pool_;
  SessionManager sessions_;
  std::chrono::seconds session_timeout_{300}; // 5分钟

  // pimpl隐藏uWebSockets依赖
  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool running_ = false;
};

} // namespace yolo_edge
