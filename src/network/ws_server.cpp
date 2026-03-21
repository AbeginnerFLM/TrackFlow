#include "network/ws_server.hpp"
#include <App.h>
#include <chrono>
#include <cstdio>

namespace yolo_edge {

// ============================================================================
// 实现细节 (pimpl)
// ============================================================================
struct WebSocketServer::Impl {
  uWS::Loop *loop = nullptr;
  us_listen_socket_t *listen_socket = nullptr;
};

struct PerSocketData {
  std::string session_id;
  bool waiting_for_image = false;
  json pending_header;
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
          {.compression = uWS::SHARED_COMPRESSOR,
           .maxPayloadLength = 100 * 1024 * 1024,
           .idleTimeout = 120,
           .maxBackpressure = 16 * 1024 * 1024,

           .open =
               [](auto *ws) {
                 auto *data = ws->getUserData();
                 data->session_id =
                     "session_" +
                     std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count());
                 fprintf(stderr, "[INFO] Client connected: %s\n",
                         data->session_id.c_str());
               },

           .message =
               [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                 handle_message(ws, message, opCode == uWS::OpCode::BINARY);
               },

           .close =
               [this](auto *ws, int code, std::string_view message) {
                 auto *data = ws->getUserData();
                 fprintf(stderr, "[INFO] Client disconnected: %s\n",
                         data->session_id.c_str());
                 sessions_.remove(data->session_id);
               }})
      .listen(port_,
              [this](auto *listen_socket) {
                if (listen_socket) {
                  impl_->listen_socket = listen_socket;
                  fprintf(stderr, "[INFO] WebSocket server listening on port %d\n",
                          port_);
                } else {
                  fprintf(stderr, "[ERROR] Failed to listen on port %d\n",
                          port_);
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
void WebSocketServer::handle_message(void *ws_ptr, std::string_view message,
                                     bool is_binary) {
  auto *ws = static_cast<uWS::WebSocket<false, true, PerSocketData> *>(ws_ptr);
  auto *socket_data = ws->getUserData();
  auto *loop = impl_->loop;

  // 1. 二进制消息 (图片数据)
  if (is_binary) {
    if (!socket_data->waiting_for_image) return;

    socket_data->waiting_for_image = false;
    json request = socket_data->pending_header;

    if (message.size() > 100 * 1024 * 1024) return;

    auto image_data = std::make_shared<std::vector<uint8_t>>(
        message.begin(), message.end());

    pool_.enqueue([this, ws, socket_data, loop, request,
                   image_data]() {
      json response;
      try {
        ProcessingContext ctx;
        std::string request_id = request.value("request_id", "");
        ctx.request_id = request_id;
        ctx.session_id = request.value("session_id", socket_data->session_id);
        ctx.frame_id = request.value("frame_id", 0);
        ctx.set("image_binary", image_data);

        json pipeline_config;
        pipeline_config = {{"pipeline", {"decoder", "yolo", "tracker"}},
                           {"config", request.value("config", json::object())}};

        auto &session = sessions_.get_or_create(ctx.session_id, pipeline_config);

        bool success;
        {
          std::lock_guard<std::mutex> lock(session.pipeline_mutex);
          success = session.pipeline.execute(ctx);
        }

        response = build_response(ctx, request, success);

      } catch (const std::exception &e) {
        response = build_error(e.what());
      }

      loop->defer(
          [ws, response]() { ws->send(response.dump(), uWS::OpCode::TEXT); });
    });
    return;
  }

  // 2. 文本消息 (JSON)
  try {
    json request = json::parse(message);
    std::string request_type = request.value("type", "infer");

    if (request_type == "infer_header") {
      socket_data->pending_header = request;
      socket_data->waiting_for_image = true;
      return;
    }

    std::string msg_copy(message);

    pool_.enqueue([this, ws, socket_data, loop, msg_copy]() {
      json response;
      try {
        json req = json::parse(msg_copy);
        std::string type = req.value("type", "infer");
        std::string rid = req.value("request_id", "");

        if (type == "ping") {
          response = {{"type", "pong"}, {"request_id", rid}};
        } else if (type == "reset") {
          std::string sid = req.value("session_id", socket_data->session_id);
          sessions_.remove(sid);
          response = {
              {"type", "reset_ack"}, {"request_id", rid}, {"session_id", sid}};
        } else if (type == "infer") {
          ProcessingContext ctx;
          ctx.request_id = rid;
          ctx.session_id = req.value("session_id", socket_data->session_id);
          ctx.frame_id = req.value("frame_id", 0);

          if (req.contains("image"))
            ctx.set("image_base64", req["image"].get<std::string>());

          json pipeline_config;
          if (req.contains("pipeline"))
            pipeline_config = req;
          else
            pipeline_config = {{"pipeline", {"decoder", "yolo", "tracker"}},
                               {"config", req.value("config", json::object())}};

          auto &session =
              sessions_.get_or_create(ctx.session_id, pipeline_config);
          bool success;
          {
            std::lock_guard<std::mutex> lock(session.pipeline_mutex);
            success = session.pipeline.execute(ctx);
          }
          response = build_response(ctx, req, success);
        } else {
          response = build_error("Unknown type: " + type, rid);
        }
      } catch (const std::exception &e) {
        response = build_error(e.what());
      }

      loop->defer(
          [ws, response]() { ws->send(response.dump(), uWS::OpCode::TEXT); });
    });

  } catch (const std::exception &e) {
    json err = build_error("Invalid JSON: " + std::string(e.what()));
    ws->send(err.dump(), uWS::OpCode::TEXT);
  }
}

// ============================================================================
// 构建响应 (优化: 减少冗余计算)
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

  json detections = json::array();
  detections.get_ref<json::array_t &>().reserve(ctx.detections.size());

  for (const auto &det : ctx.detections) {
    json det_json;

    // 基本信息
    det_json["track_id"] = det.track_id;
    det_json["class_id"] = det.class_id;
    det_json["class_name"] = det.class_name;
    det_json["confidence"] = det.confidence;
    det_json["angle"] = det.obb.angle;

    // OBB 顶点
    cv::Point2f pts[4];
    det.obb.points(pts);
    det_json["obb"] = {pts[0].x, pts[0].y, pts[1].x, pts[1].y,
                       pts[2].x, pts[2].y, pts[3].x, pts[3].y};

    // HBB
    cv::Rect rect = det.obb.boundingRect();
    det_json["bbox"] = {rect.x, rect.y, rect.width, rect.height};

    // 可选地理坐标
    if (det.ground_x.has_value())
      det_json["ground_x"] = det.ground_x.value();
    if (det.ground_y.has_value())
      det_json["ground_y"] = det.ground_y.value();
    if (det.lon.has_value())
      det_json["lon"] = det.lon.value();
    if (det.lat.has_value())
      det_json["lat"] = det.lat.value();

    detections.push_back(std::move(det_json));
  }
  response["detections"] = std::move(detections);

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
