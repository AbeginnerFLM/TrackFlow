# TrackFlow 视频分析系统 C++ 架构设计方案

> **项目目标**: 将Python视频分析系统重构为现代化C++项目，实现分布式YOLO推理、目标跟踪和地理坐标转换
> **文档版本**: 1.0 | **日期**: 2026-02-01

---

## 1. 系统概述

### 1.1 业务场景
基于无人机/固定摄像头视频的智能交通分析系统，实现车辆检测、跟踪、轨迹提取和真实地理坐标计算。

### 1.2 技术栈

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| 构建系统 | CMake 3.20+ | 跨平台构建 |
| C++标准 | C++20 | 现代特性支持 |
| 推理后端 | ONNX Runtime / TensorRT | 可切换 |
| 视频处理 | OpenCV 4.x | 解码/图像处理 |
| 网络通信 | uWebSockets | 高性能WebSocket |
| JSON解析 | nlohmann/json | Header-only |
| 日志系统 | spdlog | 高性能日志 |
| 坐标转换 | PROJ 9.x | 经纬度↔UTM |
| 跟踪算法 | ByteTrack (自实现) | 多目标跟踪 |
| 线程池 | 自定义实现 (C++20) | 任务调度 |

### 1.3 部署架构

```
┌─────────────┐     WebSocket      ┌─────────────┐    WebSocket     ┌─────────────────┐
│   Browser   │ ◄────────────────► │     VPS     │ ◄──────────────► │  4090 Server    │
│  (Canvas)   │    帧数据/结果      │  (中继服务)  │   帧数据/结果     │  (WSL + YOLO)   │
└─────────────┘                    └─────────────┘                  └─────────────────┘
```

---

## 2. 项目目录结构

```
yolo-edge-server/
├── CMakeLists.txt                    # 主构建文件
├── conanfile.txt                     # 依赖管理(可选)
├── config/
│   └── config.yaml                   # 运行时配置
├── include/
│   ├── core/
│   │   ├── image_processor.hpp       # 处理器基类
│   │   ├── processing_context.hpp    # 处理上下文
│   │   ├── processing_pipeline.hpp   # 处理管道
│   │   ├── processor_factory.hpp     # 工厂类
│   │   └── session.hpp               # 会话管理(预留)
│   ├── processors/
│   │   ├── image_decoder.hpp         # Base64解码
│   │   ├── yolo_detector.hpp         # YOLO检测
│   │   ├── byte_tracker.hpp          # ByteTrack跟踪
│   │   ├── undistort_processor.hpp   # 畸变校正(预留)
│   │   └── geo_transformer.hpp       # 坐标转换
│   ├── network/
│   │   ├── ws_server.hpp             # WebSocket服务端
│   │   └── message_handler.hpp       # 消息处理
│   └── utils/
│       ├── thread_pool.hpp           # 线程池
│       ├── base64.hpp                # Base64编解码
│       └── config.hpp                # 配置读取
├── src/
│   ├── core/                         # 核心实现
│   ├── processors/                   # 处理器实现
│   ├── network/                      # 网络实现
│   └── main.cpp                      # 入口
├── web/                              # 前端文件
│   ├── index.html
│   └── app.js
└── tests/                            # 单元测试
```

---

## 3. 核心设计模式

### 3.1 策略模式 + 工厂模式 + 责任链模式

```
ProcessorFactory (工厂)
    │
    ▼ 根据JSON配置创建
ProcessingPipeline (责任链)
    │
    ├── ImageDecoder
    ├── UndistortProcessor (可选)
    ├── YoloDetector
    ├── ByteTracker
    └── GeoTransformer (可选)
    │
    ▼
ProcessingContext (共享上下文)
```

---

## 4. 核心接口定义

### 4.1 ProcessingContext (处理上下文)

```cpp
// include/core/processing_context.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <any>
#include <unordered_map>
#include <optional>

namespace yolo_edge {

struct Detection {
    int track_id = -1;              // 跟踪ID (-1表示未跟踪)
    int class_id;                   // 类别ID
    std::string class_name;         // 类别名
    float confidence;               // 置信度
    cv::RotatedRect obb;            // OBB: center, size, angle
    
    // 地理坐标 (可选,由GeoTransformer填充)
    std::optional<double> lon;
    std::optional<double> lat;
    std::optional<double> ground_x; // 相对原点的米
    std::optional<double> ground_y;
};

class ProcessingContext {
public:
    // === 核心数据 ===
    cv::Mat frame;                      // 当前帧图像
    int frame_id = 0;                   // 帧序号
    std::string session_id;             // 会话ID (预留多客户端)
    std::vector<Detection> detections;  // 检测/跟踪结果
    
    // === 计时信息 ===
    double decode_time_ms = 0;
    double infer_time_ms = 0;
    double track_time_ms = 0;
    double total_time_ms = 0;
    
    // === 扩展数据存储 ===
    template<typename T>
    void set(const std::string& key, T value) {
        extras_[key] = std::move(value);
    }
    
    template<typename T>
    T get(const std::string& key) const {
        return std::any_cast<T>(extras_.at(key));
    }
    
    template<typename T>
    T get_or(const std::string& key, T default_val) const {
        auto it = extras_.find(key);
        if (it == extras_.end()) return default_val;
        return std::any_cast<T>(it->second);
    }
    
    bool has(const std::string& key) const {
        return extras_.contains(key);
    }

private:
    std::unordered_map<std::string, std::any> extras_;
};

} // namespace yolo_edge
```

### 4.2 ImageProcessor (处理器基类)

```cpp
// include/core/image_processor.hpp
#pragma once
#include "processing_context.hpp"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

namespace yolo_edge {

using json = nlohmann::json;

class ImageProcessor {
public:
    virtual ~ImageProcessor() = default;
    
    // 核心处理方法 (返回false表示处理失败,中断Pipeline)
    virtual bool process(ProcessingContext& ctx) = 0;
    
    // 处理器名称 (用于日志/调试)
    virtual std::string name() const = 0;
    
    // 从JSON配置初始化 (可选覆盖)
    virtual void configure(const json& config) {}
    
    // 是否有状态 (有状态处理器不能跨Session共享)
    virtual bool is_stateful() const { return false; }
};

using ProcessorPtr = std::unique_ptr<ImageProcessor>;

} // namespace yolo_edge
```

### 4.3 ProcessingPipeline (处理管道)

```cpp
// include/core/processing_pipeline.hpp
#pragma once
#include "image_processor.hpp"
#include <vector>
#include <spdlog/spdlog.h>
#include <chrono>

namespace yolo_edge {

class ProcessingPipeline {
public:
    void add(ProcessorPtr processor) {
        processors_.push_back(std::move(processor));
    }
    
    bool execute(ProcessingContext& ctx) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (auto& proc : processors_) {
            spdlog::debug("Executing: {}", proc->name());
            if (!proc->process(ctx)) {
                spdlog::error("Processor {} failed", proc->name());
                return false;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        ctx.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return true;
    }
    
    void clear() { processors_.clear(); }
    size_t size() const { return processors_.size(); }

private:
    std::vector<ProcessorPtr> processors_;
};

} // namespace yolo_edge
```

### 4.4 ProcessorFactory (工厂类)

```cpp
// include/core/processor_factory.hpp
#pragma once
#include "processing_pipeline.hpp"
#include <functional>
#include <unordered_map>
#include <stdexcept>

namespace yolo_edge {

class ProcessorFactory {
public:
    using Creator = std::function<ProcessorPtr()>;
    
    // 单例模式
    static ProcessorFactory& instance() {
        static ProcessorFactory factory;
        return factory;
    }
    
    // 注册处理器类型
    void register_processor(const std::string& type, Creator creator) {
        creators_[type] = std::move(creator);
    }
    
    // 创建单个处理器
    ProcessorPtr create(const std::string& type, const json& config = {}) {
        auto it = creators_.find(type);
        if (it == creators_.end()) {
            throw std::runtime_error("Unknown processor: " + type);
        }
        auto processor = it->second();
        processor->configure(config);
        return processor;
    }
    
    // 根据JSON配置创建Pipeline
    ProcessingPipeline create_pipeline(const json& config) {
        ProcessingPipeline pipeline;
        
        // JSON格式: {"pipeline": [{"type":"decoder"}, {"type":"yolo",...}]}
        if (!config.contains("pipeline")) {
            throw std::runtime_error("Config missing 'pipeline' array");
        }
        
        for (const auto& proc_config : config["pipeline"]) {
            std::string type = proc_config.at("type");
            pipeline.add(create(type, proc_config));
        }
        
        return pipeline;
    }

private:
    ProcessorFactory() = default;
    std::unordered_map<std::string, Creator> creators_;
};

// 辅助宏: 自动注册处理器
#define REGISTER_PROCESSOR(type_name, class_name) \
    namespace { \
        struct class_name##Registrar { \
            class_name##Registrar() { \
                ProcessorFactory::instance().register_processor( \
                    type_name, []{ return std::make_unique<class_name>(); }); \
            } \
        } class_name##_registrar_instance; \
    }

} // namespace yolo_edge
```

---

## 5. 处理器实现

### 5.1 ImageDecoder (Base64解码)

```cpp
// include/processors/image_decoder.hpp
#pragma once
#include "core/image_processor.hpp"
#include "utils/base64.hpp"
#include <opencv2/imgcodecs.hpp>

namespace yolo_edge {

class ImageDecoder : public ImageProcessor {
public:
    bool process(ProcessingContext& ctx) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!ctx.has("image_base64")) {
            spdlog::error("Missing image_base64 in context");
            return false;
        }
        
        std::string base64_data = ctx.get<std::string>("image_base64");
        
        // 移除 data:image/jpeg;base64, 前缀
        size_t comma_pos = base64_data.find(',');
        if (comma_pos != std::string::npos) {
            base64_data = base64_data.substr(comma_pos + 1);
        }
        
        // Base64解码
        std::vector<uchar> decoded = base64_decode(base64_data);
        
        // 解码为图像
        ctx.frame = cv::imdecode(decoded, cv::IMREAD_COLOR);
        
        auto end = std::chrono::high_resolution_clock::now();
        ctx.decode_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        return !ctx.frame.empty();
    }
    
    std::string name() const override { return "ImageDecoder"; }
};

REGISTER_PROCESSOR("decoder", ImageDecoder)

} // namespace yolo_edge
```

### 5.2 YoloDetector (YOLO检测)

```cpp
// include/processors/yolo_detector.hpp
#pragma once
#include "core/image_processor.hpp"
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>  // 或 TensorRT

namespace yolo_edge {

class YoloDetector : public ImageProcessor {
public:
    bool process(ProcessingContext& ctx) override {
        if (ctx.frame.empty()) return false;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 预处理
        cv::Mat blob = preprocess(ctx.frame);
        
        // 推理 (ONNX Runtime / TensorRT)
        auto outputs = infer(blob);
        
        // 后处理 (NMS + 解析)
        ctx.detections = postprocess(outputs, ctx.frame.size());
        
        auto end = std::chrono::high_resolution_clock::now();
        ctx.infer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        return true;
    }
    
    std::string name() const override { return "YoloDetector"; }
    
    void configure(const json& config) override {
        model_path_ = config.value("model_path", "yolo.onnx");
        conf_threshold_ = config.value("confidence", 0.5f);
        nms_threshold_ = config.value("nms_threshold", 0.45f);
        is_obb_ = config.value("is_obb", true);
        
        load_model();
    }

private:
    cv::Mat preprocess(const cv::Mat& frame);
    std::vector<cv::Mat> infer(const cv::Mat& blob);
    std::vector<Detection> postprocess(const std::vector<cv::Mat>& outputs, cv::Size img_size);
    void load_model();
    
    std::string model_path_;
    float conf_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;
    bool is_obb_ = true;
    
    // ONNX Runtime 会话
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YoloDetector"};
    std::unique_ptr<Ort::Session> session_;
};

REGISTER_PROCESSOR("yolo", YoloDetector)

} // namespace yolo_edge
```

### 5.3 ByteTracker (目标跟踪)

```cpp
// include/processors/byte_tracker.hpp
#pragma once
#include "core/image_processor.hpp"
#include <map>

namespace yolo_edge {

// 跟踪对象状态
struct STrack {
    int track_id;
    int state;  // 0=New, 1=Tracked, 2=Lost
    cv::RotatedRect bbox;
    int frame_id;
    int hits;
    int time_since_update;
    std::vector<cv::RotatedRect> history;
    
    // 卡尔曼滤波器状态 (可选)
    cv::KalmanFilter kf;
};

class ByteTracker : public ImageProcessor {
public:
    bool process(ProcessingContext& ctx) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        // ByteTrack核心逻辑
        update(ctx.detections, ctx.frame_id);
        
        auto end = std::chrono::high_resolution_clock::now();
        ctx.track_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        return true;
    }
    
    std::string name() const override { return "ByteTracker"; }
    bool is_stateful() const override { return true; }  // 有状态!
    
    void configure(const json& config) override {
        track_thresh_ = config.value("track_thresh", 0.5f);
        high_thresh_ = config.value("high_thresh", 0.6f);
        match_thresh_ = config.value("match_thresh", 0.8f);
        max_time_lost_ = config.value("max_time_lost", 30);
    }

private:
    void update(std::vector<Detection>& detections, int frame_id);
    float iou(const cv::RotatedRect& a, const cv::RotatedRect& b);
    std::vector<std::vector<float>> iou_distance(
        const std::vector<STrack>& tracks,
        const std::vector<Detection>& dets);
    
    std::vector<STrack> tracked_stracks_;
    std::vector<STrack> lost_stracks_;
    int frame_id_ = 0;
    int next_id_ = 1;
    
    float track_thresh_ = 0.5f;
    float high_thresh_ = 0.6f;
    float match_thresh_ = 0.8f;
    int max_time_lost_ = 30;
};

REGISTER_PROCESSOR("tracker", ByteTracker)

} // namespace yolo_edge
```

### 5.4 GeoTransformer (坐标转换)

```cpp
// include/processors/geo_transformer.hpp
#pragma once
#include "core/image_processor.hpp"
#include <opencv2/calib3d.hpp>
#include <proj.h>

namespace yolo_edge {

class GeoTransformer : public ImageProcessor {
public:
    bool process(ProcessingContext& ctx) override {
        if (!initialized_ || ctx.detections.empty()) return true;
        
        for (auto& det : ctx.detections) {
            // 1. 获取bbox中心点
            cv::Point2f center = det.obb.center;
            
            // 2. 畸变校正 (如果有相机参数)
            if (has_camera_params_) {
                std::vector<cv::Point2f> pts_in = {center};
                std::vector<cv::Point2f> pts_out;
                cv::undistortPoints(pts_in, pts_out, K_, dist_, cv::noArray(), K_);
                center = pts_out[0];
            }
            
            // 3. 透视变换 → 地面坐标 (相对原点的米)
            std::vector<cv::Point2f> src = {center};
            std::vector<cv::Point2f> dst;
            cv::perspectiveTransform(src, dst, H_);
            
            det.ground_x = dst[0].x;
            det.ground_y = dst[0].y;
            
            // 4. UTM → 经纬度
            auto [lon, lat] = utm_to_lonlat(dst[0].x, dst[0].y);
            det.lon = lon;
            det.lat = lat;
        }
        
        return true;
    }
    
    std::string name() const override { return "GeoTransformer"; }
    
    void configure(const json& config) override {
        // 单应矩阵 (必须)
        auto h_data = config.at("homography").get<std::vector<double>>();
        H_ = cv::Mat(3, 3, CV_64F);
        std::copy(h_data.begin(), h_data.end(), H_.ptr<double>());
        
        // 原点经纬度 (必须)
        origin_lon_ = config.at("origin_lon").get<double>();
        origin_lat_ = config.at("origin_lat").get<double>();
        
        // 相机参数 (可选)
        if (config.contains("camera_matrix") && config.contains("dist_coeffs")) {
            auto k_data = config["camera_matrix"].get<std::vector<double>>();
            K_ = cv::Mat(3, 3, CV_64F);
            std::copy(k_data.begin(), k_data.end(), K_.ptr<double>());
            
            auto d_data = config["dist_coeffs"].get<std::vector<double>>();
            dist_ = cv::Mat(1, d_data.size(), CV_64F);
            std::copy(d_data.begin(), d_data.end(), dist_.ptr<double>());
            
            has_camera_params_ = true;
        }
        
        init_proj();
        initialized_ = true;
    }

private:
    void init_proj() {
        // 计算UTM区号
        int zone = static_cast<int>((origin_lon_ + 180.0) / 6.0) + 1;
        bool is_north = origin_lat_ >= 0;
        int epsg = is_north ? (32600 + zone) : (32700 + zone);
        
        // 创建PROJ转换器
        ctx_ = proj_context_create();
        std::string utm_def = "EPSG:" + std::to_string(epsg);
        P_ = proj_create_crs_to_crs(ctx_, utm_def.c_str(), "EPSG:4326", nullptr);
        
        // 计算原点的UTM坐标
        PJ_COORD origin = proj_coord(origin_lon_, origin_lat_, 0, 0);
        PJ_COORD origin_utm = proj_trans(P_, PJ_INV, origin);
        origin_utm_x_ = origin_utm.xy.x;
        origin_utm_y_ = origin_utm.xy.y;
    }
    
    std::pair<double, double> utm_to_lonlat(double easting, double northing) {
        double world_x = easting + origin_utm_x_;
        double world_y = northing + origin_utm_y_;
        
        PJ_COORD utm = proj_coord(world_x, world_y, 0, 0);
        PJ_COORD lonlat = proj_trans(P_, PJ_FWD, utm);
        
        return {lonlat.lp.lam, lonlat.lp.phi};
    }
    
    cv::Mat H_, K_, dist_;
    double origin_lon_, origin_lat_;
    double origin_utm_x_, origin_utm_y_;
    bool has_camera_params_ = false;
    bool initialized_ = false;
    
    PJ_CONTEXT* ctx_ = nullptr;
    PJ* P_ = nullptr;
};

REGISTER_PROCESSOR("geo_transform", GeoTransformer)

} // namespace yolo_edge
```

### 5.5 UndistortProcessor (预留-整帧畸变校正)

```cpp
// include/processors/undistort_processor.hpp
#pragma once
#include "core/image_processor.hpp"

namespace yolo_edge {

// 预留接口: 如果需要在检测前校正整帧图像
class UndistortProcessor : public ImageProcessor {
public:
    bool process(ProcessingContext& ctx) override {
        if (!initialized_ || ctx.frame.empty()) return true;
        
        cv::Mat undistorted;
        cv::remap(ctx.frame, undistorted, map1_, map2_, cv::INTER_LINEAR);
        ctx.frame = undistorted;
        return true;
    }
    
    std::string name() const override { return "UndistortProcessor"; }
    
    void configure(const json& config) override {
        auto k_data = config.at("camera_matrix").get<std::vector<double>>();
        K_ = cv::Mat(3, 3, CV_64F);
        std::copy(k_data.begin(), k_data.end(), K_.ptr<double>());
        
        auto d_data = config.at("dist_coeffs").get<std::vector<double>>();
        dist_ = cv::Mat(1, d_data.size(), CV_64F);
        std::copy(d_data.begin(), d_data.end(), dist_.ptr<double>());
        
        int width = config.at("width").get<int>();
        int height = config.at("height").get<int>();
        
        // 预计算映射表 (只需一次!)
        cv::initUndistortRectifyMap(K_, dist_, cv::Mat(), K_, 
                                     cv::Size(width, height), CV_32FC1, 
                                     map1_, map2_);
        initialized_ = true;
    }

private:
    cv::Mat K_, dist_, map1_, map2_;
    bool initialized_ = false;
};

REGISTER_PROCESSOR("undistort", UndistortProcessor)

} // namespace yolo_edge
```

---

## 6. 网络通信设计

### 6.1 WebSocket消息协议

#### 请求 (Browser → Server)

```json
{
  "type": "infer",
  "request_id": "uuid-xxx",
  "session_id": "client-001",
  "frame_id": 42,
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  
  "pipeline": ["decoder", "yolo", "tracker", "geo_transform"],
  
  "config": {
    "yolo": {
      "model_path": "/models/yolo_obb.onnx",
      "confidence": 0.5,
      "is_obb": true
    },
    "tracker": {
      "track_thresh": 0.5,
      "max_time_lost": 30
    },
    "geo_transform": {
      "homography": [h11,h12,h13, h21,h22,h23, h31,h32,1.0],
      "origin_lon": 116.397,
      "origin_lat": 39.909,
      "camera_matrix": [fx,0,cx, 0,fy,cy, 0,0,1],
      "dist_coeffs": [k1, k2, p1, p2, k3]
    }
  }
}
```

#### 响应 (Server → Browser)

```json
{
  "type": "result",
  "request_id": "uuid-xxx",
  "session_id": "client-001",
  "frame_id": 42,
  "detections": [
    {
      "track_id": 1,
      "class_id": 0,
      "class_name": "car",
      "confidence": 0.92,
      "bbox": {
        "center": [500, 300],
        "size": [120, 60],
        "angle": 15.5
      },
      "ground_x": 12.5,
      "ground_y": 8.3,
      "lon": 116.3975,
      "lat": 39.9092
    }
  ],
  "timing": {
    "decode_ms": 2.1,
    "infer_ms": 12.5,
    "track_ms": 0.8,
    "total_ms": 16.2
  }
}
```

### 6.2 WebSocket服务端实现

```cpp
// include/network/ws_server.hpp
#pragma once
#include <uwebsockets/App.h>
#include "core/processor_factory.hpp"
#include "utils/thread_pool.hpp"
#include <nlohmann/json.hpp>

namespace yolo_edge {

struct PerSocketData {
    std::string session_id;
};

class WebSocketServer {
public:
    WebSocketServer(int port, ThreadPool& pool) 
        : port_(port), pool_(pool) {}
    
    void run() {
        auto loop = uWS::Loop::get();
        
        uWS::App()
            .ws<PerSocketData>("/*", {
                .open = [](auto* ws) {
                    spdlog::info("Client connected");
                },
                
                .message = [this, loop](auto* ws, std::string_view message, uWS::OpCode) {
                    handle_message(ws, message, loop);
                },
                
                .close = [](auto* ws, int, std::string_view) {
                    spdlog::info("Client disconnected");
                }
            })
            .listen(port_, [this](auto* listen_socket) {
                if (listen_socket) {
                    spdlog::info("Listening on port {}", port_);
                }
            })
            .run();
    }

private:
    void handle_message(auto* ws, std::string_view message, uWS::Loop* loop) {
        // 解析JSON
        json req = json::parse(message);
        
        // 投入线程池处理 (不阻塞IO线程)
        pool_.enqueue([this, ws, req, loop]() {
            try {
                // 创建Pipeline
                auto& factory = ProcessorFactory::instance();
                auto pipeline = factory.create_pipeline(req);
                
                // 创建上下文
                ProcessingContext ctx;
                ctx.frame_id = req.value("frame_id", 0);
                ctx.session_id = req.value("session_id", "default");
                ctx.set("image_base64", req.at("image").get<std::string>());
                
                // 执行Pipeline
                bool success = pipeline.execute(ctx);
                
                // 构建响应
                json response = build_response(ctx, req, success);
                
                // 回到IO线程发送 (关键!)
                loop->defer([ws, response]() {
                    ws->send(response.dump(), uWS::OpCode::TEXT);
                });
                
            } catch (const std::exception& e) {
                json error = {{"type", "error"}, {"message", e.what()}};
                loop->defer([ws, error]() {
                    ws->send(error.dump(), uWS::OpCode::TEXT);
                });
            }
        });
    }
    
    json build_response(const ProcessingContext& ctx, const json& req, bool success);
    
    int port_;
    ThreadPool& pool_;
};

} // namespace yolo_edge
```

---

## 7. 线程池实现 (用户提供)

```cpp
// include/utils/thread_pool.hpp
#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> 
    {
        using return_type = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                return std::invoke(std::move(f), std::move(args)...);
            }
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(queue_mutex); stop = true; }
        condition.notify_all();
        for (auto& w : workers) if (w.joinable()) w.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
```

---

## 8. 多客户端扩展设计 (预留)

### 8.1 Session管理器

```cpp
// include/core/session.hpp (预留)
#pragma once
#include "processing_pipeline.hpp"
#include <unordered_map>
#include <mutex>

namespace yolo_edge {

class Session {
public:
    std::string session_id;
    ProcessingPipeline pipeline;  // 每个Session独立的Pipeline
    std::chrono::steady_clock::time_point last_active;
    
    // 有状态处理器的状态存储
    // (Tracker等有状态处理器的实例属于Session)
};

class SessionManager {
public:
    Session& get_or_create(const std::string& session_id, const json& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            it->second.last_active = std::chrono::steady_clock::now();
            return it->second;
        }
        
        // 创建新Session
        Session session;
        session.session_id = session_id;
        session.pipeline = ProcessorFactory::instance().create_pipeline(config);
        session.last_active = std::chrono::steady_clock::now();
        
        auto [inserted, _] = sessions_.emplace(session_id, std::move(session));
        return inserted->second;
    }
    
    void cleanup_expired(std::chrono::seconds timeout) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        
        for (auto it = sessions_.begin(); it != sessions_.end(); ) {
            if (now - it->second.last_active > timeout) {
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    std::unordered_map<std::string, Session> sessions_;
    std::mutex mutex_;
};

} // namespace yolo_edge
```

---

## 9. 主程序入口

```cpp
// src/main.cpp
#include "network/ws_server.hpp"
#include "utils/thread_pool.hpp"
#include "processors/image_decoder.hpp"
#include "processors/yolo_detector.hpp"
#include "processors/byte_tracker.hpp"
#include "processors/geo_transformer.hpp"
#include <spdlog/spdlog.h>

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("YOLO-Edge Server starting...");
    
    // 处理器已通过REGISTER_PROCESSOR宏自动注册
    
    // 创建线程池 (推理线程数建议1-2)
    ThreadPool pool(2);
    
    // 启动WebSocket服务
    int port = 9001;
    yolo_edge::WebSocketServer server(port, pool);
    server.run();
    
    return 0;
}
```

---

## 10. CMake构建配置

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(yolo-edge-server VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 依赖查找
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PROJ REQUIRED)

# uWebSockets (header-only + uSockets)
# 可能需要手动指定路径

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 可执行文件
add_executable(yolo_edge_server ${SOURCES})

target_link_libraries(yolo_edge_server
    ${OpenCV_LIBS}
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    PROJ::proj
    # uSockets
    # ONNX Runtime
)
```

---

## 11. 开发路线图

| 阶段 | 任务 | 预估时间 |
|------|------|----------|
| **Phase 1** | 项目骨架 + CMake + 核心接口 | 2天 |
| **Phase 2** | ImageDecoder + 简单WebSocket | 2天 |
| **Phase 3** | YOLO推理 (ONNX Runtime) | 3天 |
| **Phase 4** | ByteTrack实现 | 3天 |
| **Phase 5** | 坐标转换 (PROJ) | 1天 |
| **Phase 6** | 前端Canvas集成 | 2天 |
| **Phase 7** | 测试 + 优化 | 2天 |

---

## 12. 附录：关键算法说明

### 12.1 坐标转换流程

```
像素点 (px, py)
    │
    ▼ cv::undistortPoints(K, dist) [可选]
校正后像素点
    │
    ▼ cv::perspectiveTransform(H)
地面坐标 (相对原点的米)
    │
    ▼ + origin_utm → PROJ → WGS84
经纬度 (lon, lat)
```

### 12.2 ByteTrack核心流程

```
输入: 当前帧检测结果
    │
    ▼ 分为高分框 / 低分框
    │
    ▼ 高分框与已跟踪对象IOU匹配 (匈牙利算法)
    │
    ▼ 未匹配的高分框与Lost对象匹配
    │
    ▼ 低分框与剩余已跟踪对象匹配
    │
    ▼ 创建新轨迹 / 更新Lost状态
    │
输出: 带track_id的检测结果
```

---

**文档结束 | 版本 1.0**
