#include "network/ws_server.hpp"
#include <App.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>

namespace yolo_edge {

struct WebSocketServer::Impl {
  uWS::Loop *loop = nullptr;
  us_listen_socket_t *listen_socket = nullptr;
};

struct ConnectionState {
  void *ws = nullptr;
  bool closed = false;
};

struct PerSocketData {
  std::string session_id;
  bool waiting_for_image = false;
  json pending_header;
  std::shared_ptr<ConnectionState> connection;
};

using WsHandle = uWS::WebSocket<false, true, PerSocketData>;

namespace {

void defer_send_text(uWS::Loop *loop, const std::shared_ptr<ConnectionState> &connection,
                     json response) {
  if (!loop || !connection) {
    return;
  }

  loop->defer([connection, response = std::move(response)]() mutable {
    if (connection->closed || connection->ws == nullptr) {
      return;
    }

    auto *ws = static_cast<WsHandle *>(connection->ws);
    ws->send(response.dump(), uWS::OpCode::TEXT);
  });
}

bool execute_default_pipeline(Session &session, ProcessingContext &ctx) {
  using Clock = std::chrono::high_resolution_clock;
  const auto start = Clock::now();

  int tracker_idx = session.pipeline.find_index("ByteTracker");
  const size_t split = (tracker_idx >= 0)
      ? static_cast<size_t>(tracker_idx)
      : session.pipeline.size();
  bool success = session.pipeline.execute_range(ctx, 0, split);
  if (session.pipeline.size() <= split) {
    const auto end = Clock::now();
    ctx.total_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return success;
  }

  // 难点：即使前半段失败，也要消耗当前 frame_id 的 tracker 时序位，
  // 否则后续帧会卡在 wait_for_turn，导致整个 session 堵塞。
  session.wait_for_turn(ctx.frame_id);
  try {
    if (success) {
      success = session.pipeline.execute_range(ctx, split, session.pipeline.size());
    }
  } catch (...) {
    session.advance_turn();
    throw;
  }
  session.advance_turn();

  const auto end = Clock::now();
  ctx.total_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return success;
}

} // namespace

WebSocketServer::WebSocketServer(int port, ThreadPool &pool)
    : port_(port), pool_(pool), impl_(std::make_unique<Impl>()) {}

WebSocketServer::~WebSocketServer() { stop(); }

void WebSocketServer::set_session_timeout(std::chrono::seconds timeout) {
  session_timeout_ = timeout;
}

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
                 data->connection = std::make_shared<ConnectionState>();
                 data->connection->ws = ws;
                 data->connection->closed = false;
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
                 if (data->connection) {
                   data->connection->closed = true;
                   data->connection->ws = nullptr;
                 }
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

void WebSocketServer::handle_message(void *ws_ptr, std::string_view message,
                                     bool is_binary) {
  auto *ws = static_cast<WsHandle *>(ws_ptr);
  auto *socket_data = ws->getUserData();
  auto *loop = impl_->loop;

  static std::atomic<uint64_t> cleanup_tick{0};
  if ((cleanup_tick.fetch_add(1, std::memory_order_relaxed) & 0xFF) == 0) {
    sessions_.cleanup_expired(session_timeout_);
  }

  if (is_binary) {
    if (!socket_data->waiting_for_image)
      return;

    socket_data->waiting_for_image = false;
    json request = socket_data->pending_header;

    if (message.size() > 100 * 1024 * 1024)
      return;

    auto image_data = std::make_shared<std::vector<uint8_t>>(
        message.begin(), message.end());

    auto connection = socket_data->connection;
    auto default_session_id = socket_data->session_id;
    pool_.enqueue([this, loop, request, image_data, connection,
                   default_session_id]() {
      json response;
      try {
        ProcessingContext ctx;
        std::string request_id = request.value("request_id", "");
        ctx.request_id = request_id;
        ctx.session_id = request.value("session_id", default_session_id);
        ctx.frame_id = request.value("frame_id", 0);
        ctx.set("image_binary", image_data);

        json pipeline_config;
        if (!request.contains("pipeline")) {
          pipeline_config = {{"pipeline", {"decoder", "yolo", "tracker"}},
                             {"config", request.value("config", json::object())}};
        } else {
          pipeline_config = request;
        }

        auto &session = sessions_.get_or_create(ctx.session_id, pipeline_config);
        bool success = execute_default_pipeline(session, ctx);
        response = build_response(ctx, success);

      } catch (const std::exception &e) {
        response = build_error(e.what());
      }

      defer_send_text(loop, connection, std::move(response));
    });
    return;
  }

  try {
    json request = json::parse(message);
    std::string request_type = request.value("type", "");

    if (request_type == "infer_header") {
      socket_data->session_id =
          request.value("session_id", socket_data->session_id);
      socket_data->pending_header = request;
      socket_data->waiting_for_image = true;
      return;
    }

    if (request_type == "ping") {
      json pong = {{"type", "pong"}, {"request_id", request.value("request_id", "")}};
      ws->send(pong.dump(), uWS::OpCode::TEXT);
      return;
    }

    if (request_type == "reset") {
      std::string rid = request.value("request_id", "");
      std::string sid = request.value("session_id", socket_data->session_id);
      socket_data->session_id = sid;
      sessions_.remove(sid);
      json response = {{"type", "reset_ack"}, {"request_id", rid}, {"session_id", sid}};
      ws->send(response.dump(), uWS::OpCode::TEXT);
      return;
    }

    if (request_type == "infer") {
      ws->send(build_error(
                   "Legacy JSON infer is no longer supported. "
                   "Use infer_header followed by a binary image frame.",
                   request.value("request_id", ""))
                   .dump(),
               uWS::OpCode::TEXT);
      return;
    }

    if (request_type.empty()) {
      ws->send(build_error("Missing type in request",
                           request.value("request_id", ""))
                   .dump(),
               uWS::OpCode::TEXT);
      return;
    }

    ws->send(build_error("Unknown type: " + request_type,
                         request.value("request_id", ""))
                 .dump(),
             uWS::OpCode::TEXT);
    return;

  } catch (const std::exception &e) {
    json err = build_error("Invalid JSON: " + std::string(e.what()));
    ws->send(err.dump(), uWS::OpCode::TEXT);
  }
}

json WebSocketServer::build_response(const ProcessingContext &ctx, bool success) {
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

    det_json["track_id"] = det.track_id;
    det_json["class_id"] = det.class_id;
    det_json["class_name"] = det.class_name;
    det_json["confidence"] = det.confidence;
    det_json["angle"] = det.obb.angle;

    cv::Point2f pts[4];
    det.obb.points(pts);
    det_json["obb"] = {pts[0].x, pts[0].y, pts[1].x, pts[1].y,
                       pts[2].x, pts[2].y, pts[3].x, pts[3].y};

    cv::Rect rect = det.obb.boundingRect();
    det_json["bbox"] = {rect.x, rect.y, rect.width, rect.height};

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
