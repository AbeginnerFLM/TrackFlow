#include "network/ws_server.hpp"
#include <App.h>
#include <chrono>
#include <cstdio>
#include <iostream>

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
                 fprintf(stderr, "Client connected: %s\n",
                         data->session_id.c_str());

                 // Send welcome message v5.1
                 // Note: We cannot easily use send_json here as it is not
                 // member of ws But we can send raw text. Actually, let's skip
                 // it to avoid crashes.
               },

           // 收到消息
           .message =
               [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                 handle_message(ws, message, opCode == uWS::OpCode::BINARY);
               },

           // 连接关闭
           .close =
               [this](auto *ws, int code, std::string_view message) {
                 auto *data = ws->getUserData();
                 fprintf(stderr, "Client disconnected: %s (code: %d)\n",
                         data->session_id.c_str(), code);
                 sessions_.remove(data->session_id);
               }})
      .listen(port_,
              [this](auto *listen_socket) {
                if (listen_socket) {
                  impl_->listen_socket = listen_socket;
                  fprintf(stderr, "WebSocket server listening on port %d\n",
                          port_);
                } else {
                  fprintf(stderr, "Failed to listen on port %d\n", port_);
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

  // ==========================================================================
  // 1. 处理二进制消息 (图片数据)
  // ==========================================================================
  if (is_binary) {
    // std::cerr << "DEBUG: Binary message received. Size=" << message.size() <<
    // std::endl;
    if (!socket_data->waiting_for_image) {
      fprintf(stderr, "Received unexpected binary message\n");
      return;
    }

    // 重置状态
    socket_data->waiting_for_image = false;
    json request = socket_data->pending_header;

    // Log
    // Log
    // fprintf(stderr, "Binary message received: ptr=%p, size=%zu\n",
    //        (void *)message.data(), message.size());

    if (message.size() > 100 * 1024 * 1024) {
      fprintf(stderr, "Binary message too large: %zu\n", message.size());
      return;
    }

    // 复制二进制数据 (使用shared_ptr避免拷贝)
    // fprintf(stderr, "Allocating shared_ptr vector...\n");
    auto image_data = std::make_shared<std::vector<uint8_t>>();

    // Check max_size
    size_t vec_max = image_data->max_size();
    if (message.size() > vec_max) {
      fprintf(stderr, "Requested size %zu > vector max_size %zu\n",
              message.size(), vec_max);
      return;
    }

    try {
      // fprintf(stderr, "Assigning data to vector (size=%zu)...\n",
      // message.size());
      image_data->assign(message.begin(), message.end());
      // fprintf(stderr, "Vector assignment complete. Vector size: %zu\n",
      //              image_data->size());
    } catch (const std::exception &e) {
      fprintf(stderr, "Vector allocation failed: size=%zu, error=%s\n",
              message.size(), e.what());
      return;
    }

    // 投入线程池执行推理
    // fprintf(stderr, "Enqueuing task to thread pool...\n");
    pool_.enqueue([this, ws, socket_data, loop, request, image_data]() {
      ProcessingContext ctx;
      json response;

      try {
        // std::cerr << "DEBUG_TRACE: Task started" << std::endl;

        std::string request_id = request.value("request_id", "");

        // 准备上下文
        ctx.request_id = request_id;
        ctx.session_id = request.value("session_id", socket_data->session_id);
        ctx.frame_id = request.value("frame_id", 0);

        // std::cerr << "DEBUG_TRACE: Context prepared. ID=" << ctx.session_id
        // << std::endl;

        // **关键修改**: 直接放入二进制数据，而不是base64字符串
        // 我们利用 std::any 存储 vector<uint8_t>
        ctx.set("image_binary", image_data);

        // std::cerr << "DEBUG_TRACE: Binary data set to context" << std::endl;

        // 获取Session
        json pipeline_config;
        pipeline_config = {{"pipeline", {"decoder", "yolo"}},
                           {"config", request.value("config", json::object())}};

        // std::cerr << "DEBUG_TRACE: Getting session..." << std::endl;

        auto &session =
            sessions_.get_or_create(ctx.session_id, pipeline_config);

        // std::cerr << "DEBUG_TRACE: Session retrieved. Starting pipeline
        // execute..." << std::endl;

        // 执行Pipeline
        bool success;
        {
          std::lock_guard<std::mutex> lock(session.pipeline_mutex);
          success = session.pipeline.execute(ctx);
        }

        // std::cerr << "DEBUG_TRACE: Pipeline finished. Success=" << success <<
        // std::endl;

        if (!success) {
          fprintf(stderr,
                  "DEBUG_TRACE: Pipeline execution FAILED for session %s\n",
                  ctx.session_id.c_str());
        }

        response = build_response(ctx, request, success);

      } catch (const std::exception &e) {
        fprintf(stderr, "Binary processing error: %s\n", e.what());
        response = build_error(e.what());
      }

      // 发送响应
      loop->defer(
          [ws, response]() { ws->send(response.dump(), uWS::OpCode::TEXT); });
    });
    return;
  }

  // ==========================================================================
  // 2. 处理文本消息 (JSON)
  // ==========================================================================
  try {
    json request = json::parse(message);
    std::string request_type = request.value("type", "infer");

    // std::cerr << "DEBUG_TRACE: Text handler. Type='" << request_type << "'"
    // << std::endl;

    // 2.1 推理头信息 (准备接收二进制图片)
    if (request_type == "infer_header") {
      // std::cerr << "DEBUG_TRACE: Sync handling infer_header" << std::endl;
      socket_data->pending_header = request;
      socket_data->waiting_for_image = true;
      return; // 等待下一帧二进制数据
    }

    // 2.2 其他请求 (ping, reset, 或旧版base64 infer)
    std::string msg_copy(message); // 复制用于异步处理

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
          // 旧版带Base64的请求
          ProcessingContext ctx;
          ctx.request_id = rid;
          ctx.session_id = req.value("session_id", socket_data->session_id);
          ctx.frame_id = req.value("frame_id", 0);

          if (req.contains("image")) {
            ctx.set("image_base64", req["image"].get<std::string>());
          }

          // 获取Session (同上)
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
    // JSON解析失败等
    json err = build_error("Invalid JSON: " + std::string(e.what()));
    ws->send(err.dump(), uWS::OpCode::TEXT);
  }
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
    json det_json = {
        {"track_id", det.track_id},
        {"class_id", det.class_id},
        {"class_name", det.class_name},
        {"confidence", det.confidence},
    };

    // OBB points (8 floats)
    cv::Point2f pts[4];
    det.obb.points(pts);
    det_json["obb"] = {pts[0].x, pts[0].y, pts[1].x, pts[1].y,
                       pts[2].x, pts[2].y, pts[3].x, pts[3].y};

    // HBB (x, y, w, h)
    cv::Rect rect = det.obb.boundingRect();
    det_json["bbox"] = {rect.x, rect.y, rect.width, rect.height};

    // Angle
    det_json["angle"] = det.obb.angle;

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
