#include "network/ws_server.hpp"
#include <App.h>
#include <chrono>
#include <spdlog/spdlog.h>

namespace yolo_edge {

// ============================================================================
// 实现细节 (pimpl)
// ============================================================================
struct WebSocketServer::Impl {
  uWS::Loop *loop = nullptr;
  us_listen_socket_t *listen_socket = nullptr;
};

// Socket附加数据
struct PerSocketData {
  std::string session_id;
};

// ============================================================================
// 构造/析构
// ============================================================================
WebSocketServer::WebSocketServer(int port, ThreadPool &pool)
    : port_(port), pool_(pool), impl_(std::make_unique<Impl>()) {}

WebSocketServer::~WebSocketServer() { stop(); }

void WebSocketServer::set_session_timeout(std::chrono::seconds timeout) {
  session_timeout_ = timeout;
}

// ============================================================================
// 启动服务
// ============================================================================
void WebSocketServer::run() {
  running_ = true;
  impl_->loop = uWS::Loop::get();

  uWS::App()
      .ws<PerSocketData>(
          "/*",
          {// 连接设置
           .compression = uWS::SHARED_COMPRESSOR,
           .maxPayloadLength = 100 * 1024 * 1024, // 100MB (支持大图)
           .idleTimeout = 120,
           .maxBackpressure = 16 * 1024 * 1024,

           // 新连接
           .open =
               [](auto *ws) {
                 auto *data = ws->getUserData();
                 data->session_id =
                     "session_" +
                     std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count());
                 spdlog::info("Client connected: {}", data->session_id);
               },

           // 收到消息
           .message =
               [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                 if (opCode != uWS::OpCode::TEXT) {
                   spdlog::warn("Received non-text message, ignoring");
                   return;
                 }

                 // 复制消息内容 (因为要传入线程池)
                 std::string msg_copy(message);
                 handle_message(ws, msg_copy);
               },

           // 连接关闭
           .close =
               [this](auto *ws, int code, std::string_view message) {
                 auto *data = ws->getUserData();
                 spdlog::info("Client disconnected: {} (code: {})",
                              data->session_id, code);
                 sessions_.remove(data->session_id);
               }})
      .listen(port_,
              [this](auto *listen_socket) {
                if (listen_socket) {
                  impl_->listen_socket = listen_socket;
                  spdlog::info("WebSocket server listening on port {}", port_);
                } else {
                  spdlog::error("Failed to listen on port {}", port_);
                  running_ = false;
                }
              })
      .run();

  running_ = false;
}

void WebSocketServer::stop() {
  if (running_ && impl_->listen_socket) {
    us_listen_socket_close(0, impl_->listen_socket);
    impl_->listen_socket = nullptr;
    running_ = false;
  }
}

// ============================================================================
// 消息处理
// ============================================================================
void WebSocketServer::handle_message(void *ws_ptr, const std::string &message) {
  auto *ws = static_cast<uWS::WebSocket<false, true, PerSocketData> *>(ws_ptr);
  auto *socket_data = ws->getUserData();
  auto *loop = impl_->loop;

  // 投入线程池处理 (不阻塞IO线程)
  pool_.enqueue([this, ws, socket_data, loop, message]() {
    json response;

    try {
      // 解析请求
      json request = json::parse(message);

      std::string request_id = request.value("request_id", "");
      std::string request_type = request.value("type", "infer");

      if (request_type == "ping") {
        // 心跳
        response = {{"type", "pong"}, {"request_id", request_id}};
      } else if (request_type == "infer") {
        // 推理请求

        // 获取或创建Session (每个session有独立的有状态处理器)
        std::string session_id =
            request.value("session_id", socket_data->session_id);

        // 创建Pipeline配置
        json pipeline_config;
        if (request.contains("pipeline")) {
          pipeline_config = request;
        } else {
          // 默认Pipeline
          pipeline_config = {
              {"pipeline", {"decoder", "yolo", "tracker"}},
              {"config", request.value("config", json::object())}};
        }

        // 获取或创建Session
        auto &session = sessions_.get_or_create(session_id, pipeline_config);

        // 创建处理上下文
        ProcessingContext ctx;
        ctx.frame_id = request.value("frame_id", 0);
        ctx.session_id = session_id;
        ctx.request_id = request_id;

        // 设置图像数据
        if (request.contains("image")) {
          ctx.set("image_base64", request["image"].get<std::string>());
        } else {
          throw std::runtime_error("Missing 'image' field in request");
        }

        // 执行Pipeline
        bool success = session.pipeline.execute(ctx);

        // 构建响应
        response = build_response(ctx, request, success);
      } else if (request_type == "reset") {
        // 重置会话
        std::string session_id =
            request.value("session_id", socket_data->session_id);
        sessions_.remove(session_id);
        response = {{"type", "reset_ack"},
                    {"request_id", request_id},
                    {"session_id", session_id}};
      } else {
        response =
            build_error("Unknown request type: " + request_type, request_id);
      }

    } catch (const json::parse_error &e) {
      spdlog::error("JSON parse error: {}", e.what());
      response = build_error("Invalid JSON: " + std::string(e.what()));
    } catch (const std::exception &e) {
      spdlog::error("Processing error: {}", e.what());
      response = build_error(e.what());
    }

    // 回到IO线程发送响应
    loop->defer(
        [ws, response]() { ws->send(response.dump(), uWS::OpCode::TEXT); });
  });
}

// ============================================================================
// 构建响应
// ============================================================================
json WebSocketServer::build_response(const ProcessingContext &ctx,
                                     const json &request, bool success) {
  json response = {{"type", success ? "result" : "error"},
                   {"request_id", ctx.request_id},
                   {"session_id", ctx.session_id},
                   {"frame_id", ctx.frame_id}};

  if (!success) {
    response["error"] = "Processing failed";
    return response;
  }

  // 检测结果
  json detections = json::array();
  for (const auto &det : ctx.detections) {
    json det_json = {{"track_id", det.track_id},
                     {"class_id", det.class_id},
                     {"class_name", det.class_name},
                     {"confidence", det.confidence},
                     {"bbox",
                      {{"center", {det.obb.center.x, det.obb.center.y}},
                       {"size", {det.obb.size.width, det.obb.size.height}},
                       {"angle", det.obb.angle}}}};

    // 可选地理坐标
    if (det.ground_x.has_value()) {
      det_json["ground_x"] = det.ground_x.value();
    }
    if (det.ground_y.has_value()) {
      det_json["ground_y"] = det.ground_y.value();
    }
    if (det.lon.has_value()) {
      det_json["lon"] = det.lon.value();
    }
    if (det.lat.has_value()) {
      det_json["lat"] = det.lat.value();
    }

    detections.push_back(det_json);
  }
  response["detections"] = detections;

  // 计时信息
  response["timing"] = {{"decode_ms", ctx.decode_time_ms},
                        {"infer_ms", ctx.infer_time_ms},
                        {"track_ms", ctx.track_time_ms},
                        {"geo_ms", ctx.geo_time_ms},
                        {"total_ms", ctx.total_time_ms}};

  return response;
}

json WebSocketServer::build_error(const std::string &message,
                                  const std::string &request_id) {
  return {{"type", "error"}, {"request_id", request_id}, {"error", message}};
}

} // namespace yolo_edge
