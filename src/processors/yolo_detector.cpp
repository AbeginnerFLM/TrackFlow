#include "processors/yolo_detector.hpp"
#include "core/processor_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace yolo_edge {

// 前向声明
std::vector<int> rotated_nms(const std::vector<cv::RotatedRect> &boxes,
                             const std::vector<float> &scores,
                             float nms_threshold);

// ============================================================================
// ONNX Runtime Session封装
// ============================================================================
struct YoloDetector::OrtSession {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "YoloDetector"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<const char *> input_names;
  std::vector<const char *> output_names;
  std::vector<std::string> input_names_storage;
  std::vector<std::string> output_names_storage;

  std::vector<int64_t> input_shape;
};

// ============================================================================
// 构造/析构
// ============================================================================
YoloDetector::YoloDetector() : ort_(std::make_unique<OrtSession>()) {}

YoloDetector::~YoloDetector() = default;

// ============================================================================
// 配置
// ============================================================================
void YoloDetector::configure(const json &config) {
  model_path_ = config.value("model_path", "models/yolo_obb.onnx");
  conf_threshold_ = config.value("confidence", 0.5f);
  nms_threshold_ = config.value("nms_threshold", 0.45f);
  is_obb_ = config.value("is_obb", true);
  input_width_ = config.value("input_width", 640);
  input_height_ = config.value("input_height", 640);
  use_cuda_ = config.value("use_cuda", true);

  // 加载类别名称
  if (config.contains("class_names")) {
    class_names_ = config["class_names"].get<std::vector<std::string>>();
  } else {
    // 默认COCO类别 (简化版)
    class_names_ = {"car",     "truck",  "bus",           "motorcycle",
                    "bicycle", "person", "traffic_light", "stop_sign"};
  }

  load_model();
}

// ============================================================================
// 加载模型
// ============================================================================
void YoloDetector::load_model() {
  spdlog::info("YoloDetector: Start loading model from '{}'", model_path_);
  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (false && use_cuda_) {
      // CUDA disabled
    } else {
      spdlog::info("YoloDetector: Using CPU execution provider (Explicit)");
    }

    spdlog::info("YoloDetector: Creating Ort::Session...");
    ort_->session = std::make_unique<Ort::Session>(
        ort_->env, model_path_.c_str(), session_options);
    spdlog::info("YoloDetector: Ort::Session created successfully");

    Ort::AllocatorWithDefaultOptions allocator;

    // 获取输入信息
    size_t num_inputs = ort_->session->GetInputCount();
    spdlog::info("YoloDetector: Model has {} inputs", num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
      spdlog::info("YoloDetector: Processing input {}", i);
      spdlog::default_logger()->flush();

      auto name = ort_->session->GetInputNameAllocated(i, allocator);
      ort_->input_names_storage.push_back(name.get());
      ort_->input_names.push_back(ort_->input_names_storage.back().c_str());

      spdlog::info("YoloDetector: Input {} name acquired: {}", i, name.get());
      spdlog::default_logger()->flush();

      auto type_info = ort_->session->GetInputTypeInfo(i);
      auto shape_info = type_info.GetTensorTypeAndShapeInfo();
      spdlog::info("YoloDetector: Input {} shape info retrieved", i);
      spdlog::default_logger()->flush();

      ort_->input_shape = shape_info.GetShape();
      spdlog::info("YoloDetector: Input {} processed", i);
    }

    // 获取输出信息
    size_t num_outputs = ort_->session->GetOutputCount();
    spdlog::info("YoloDetector: Model has {} outputs", num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
      auto name = ort_->session->GetOutputNameAllocated(i, allocator);
      ort_->output_names_storage.push_back(name.get());
      ort_->output_names.push_back(ort_->output_names_storage.back().c_str());
      spdlog::info("YoloDetector: Output {} processed", i);
    }

    // ... (rest is fine)
    if (ort_->input_shape.size() >= 4) {
      if (ort_->input_shape[2] > 0)
        input_height_ = ort_->input_shape[2];
      if (ort_->input_shape[3] > 0)
        input_width_ = ort_->input_shape[3];
    }

    initialized_ = true;
    spdlog::info("YoloDetector: Loaded model '{}' (input: {}x{})", model_path_,
                 input_width_, input_height_);

  } catch (const Ort::Exception &e) {
    spdlog::error("YoloDetector: Failed to load model (Ort::Exception): {}",
                  e.what());
    initialized_ = false;
  } catch (const std::exception &e) {
    spdlog::error("YoloDetector: Failed to load model (std::exception): {}",
                  e.what());
    initialized_ = false;
  }
}

// ============================================================================
// 处理
// ============================================================================
bool YoloDetector::process(ProcessingContext &ctx) {
  try {
    if (!initialized_) {
      spdlog::error("YoloDetector: Model not initialized");
      return false;
    }

    if (ctx.frame.empty()) {
      spdlog::error("YoloDetector: Input frame is empty");
      return false;
    }

    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    spdlog::info("YoloDetector: Starting preprocess...");
    spdlog::default_logger()->flush();

    // 预处理
    cv::Mat blob = preprocess(ctx.frame);

    spdlog::info("YoloDetector: Preprocess complete. Starting inference...");
    spdlog::default_logger()->flush();

    // 推理
    auto outputs = infer(blob);

    spdlog::info("YoloDetector: Inference complete. Starting postprocess...");
    spdlog::default_logger()->flush();

    // 后处理
    ctx.detections = postprocess(outputs, ctx.frame.size());

    auto end = Clock::now();
    ctx.infer_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    spdlog::debug("YoloDetector: Found {} objects in {:.2f}ms",
                  ctx.detections.size(), ctx.infer_time_ms);
    spdlog::default_logger()->flush();

    return true;
  } catch (const std::exception &e) {
    spdlog::error("YoloDetector: Exception in process: {}", e.what());
    return false;
  }
}

// ============================================================================
// 预处理 (Letterbox + 归一化)
// ============================================================================
cv::Mat YoloDetector::preprocess(const cv::Mat &frame) {
  int img_w = frame.cols;
  int img_h = frame.rows;

  // 计算缩放比例 (保持宽高比)
  float scale_w = static_cast<float>(input_width_) / img_w;
  float scale_h = static_cast<float>(input_height_) / img_h;
  scale_ = std::min(scale_w, scale_h);

  int new_w = static_cast<int>(img_w * scale_);
  int new_h = static_cast<int>(img_h * scale_);

  // 计算padding
  pad_x_ = (input_width_ - new_w) / 2;
  pad_y_ = (input_height_ - new_h) / 2;

  // 缩放
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(new_w, new_h));

  // 创建letterbox图像 (填充灰色)
  cv::Mat letterbox(input_height_, input_width_, CV_8UC3,
                    cv::Scalar(114, 114, 114));
  resized.copyTo(letterbox(cv::Rect(pad_x_, pad_y_, new_w, new_h)));

  // BGR -> RGB
  cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

  // 归一化到 [0, 1]
  letterbox.convertTo(letterbox, CV_32F, 1.0 / 255.0);

  // HWC -> CHW
  cv::Mat blob;
  cv::dnn::blobFromImage(letterbox, blob);

  return blob;
}

// ============================================================================
// 推理
// ============================================================================
std::vector<cv::Mat> YoloDetector::infer(const cv::Mat &blob) {
  try {
    // 准备输入tensor
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    size_t input_size = 1 * 3 * input_height_ * input_width_;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_->memory_info, const_cast<float *>(blob.ptr<float>()), input_size,
        input_shape.data(), input_shape.size());

    // 运行推理
    auto output_tensors = ort_->session->Run(
        Ort::RunOptions{nullptr}, ort_->input_names.data(), &input_tensor, 1,
        ort_->output_names.data(), ort_->output_names.size());

    // 转换输出为cv::Mat
    std::vector<cv::Mat> outputs;
    for (auto &tensor : output_tensors) {
      auto *data = tensor.GetTensorMutableData<float>();
      auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();

      // 假设输出是 [1, num_features, num_boxes] 或 [1, num_boxes, num_features]
      if (shape.size() == 3) {
        int dim1 = shape[1];
        int dim2 = shape[2];
        cv::Mat mat(dim1, dim2, CV_32F, data);
        outputs.push_back(mat.clone());
      } else if (shape.size() == 2) {
        int dim1 = shape[0];
        int dim2 = shape[1];
        cv::Mat mat(dim1, dim2, CV_32F, data);
        outputs.push_back(mat.clone());
      }
    }

    return outputs;
  } catch (const std::exception &e) {
    spdlog::error("YoloDetector: Exception in infer: {}", e.what());
    throw; // Rethrow to let caller handle (but now we logged it)
  }
}

// ============================================================================
// 后处理 (解析 + NMS)
// ============================================================================
std::vector<Detection>
YoloDetector::postprocess(const std::vector<cv::Mat> &outputs,
                          const cv::Size &original_size) {
  try {
    if (outputs.empty()) {
      return {};
    }

    std::vector<Detection> detections;
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    cv::Mat output = outputs[0];

    // YOLOv8-OBB输出格式: [1, num_features, num_boxes]
    // num_features = 4 (x,y,w,h) + num_classes + 1 (angle)
    // 需要转置
    if (output.rows < output.cols) {
      output = output.t();
    }

    int num_boxes = output.rows;
    int num_features = output.cols;

    spdlog::info("YoloDetector: Postprocess input shape: {}x{}", num_boxes,
                 num_features);
    spdlog::default_logger()->flush();

    // DEBUG: Print output shape
    // std::cout << "DEBUG_SHAPE: " << num_boxes << "x" << num_features <<
    // std::endl;
    spdlog::info("DEBUG_SHAPE: {}x{}", num_boxes, num_features);

    if (num_boxes > 100000 || num_boxes < 0) {
      spdlog::error("YoloDetector: Invalid number of boxes: {}", num_boxes);
      spdlog::default_logger()->flush();
      return {};
    }

    // Log predicted memory usage for vectors
    try {
      spdlog::info("YoloDetector: Preparing to reserve vectors for {} boxes",
                   num_boxes);
      if (num_boxes > 0) {
        size_t required_bytes =
            num_boxes * (sizeof(cv::RotatedRect) + sizeof(float) + sizeof(int));
        spdlog::info("YoloDetector: Estimated temp vector memory: {} bytes",
                     required_bytes);
      }
    } catch (...) {
    }

    // 确定是OBB还是HBB
    int num_classes;
    bool has_angle = is_obb_;
    if (has_angle) {
      num_classes = num_features - 5; // x,y,w,h + classes + angle
    } else {
      num_classes = num_features - 4; // x,y,w,h + classes
    }

    if (num_classes <= 0) {
      spdlog::warn("YoloDetector: Invalid output shape");
      return {};
    }

    for (int i = 0; i < num_boxes; ++i) {
      const float *row = output.ptr<float>(i);

      // 解析坐标 (中心点格式)
      float cx = row[0];
      float cy = row[1];
      float w = row[2];
      float h = row[3];

      // 解析角度
      float angle = 0;
      float max_conf = 0;
      int max_class = 0;

      // Special handling for End-to-End export (NMS inside model)
      // Format usually: [x, y, w, h, score, class_id, angle]
      // We assume this format if num_features is small (e.g. 7) and matches our
      // observation
      if (num_features == 7) {
        // DEBUG: Print first detection raw values
        if (i == 0) {
          std::cout << "DEBUG_RAW_0: " << row[0] << "," << row[1] << ","
                    << row[2] << "," << row[3] << "," << row[4] << "," << row[5]
                    << "," << row[6] << std::endl;
        }

        max_conf = row[4];
        max_class = static_cast<int>(row[5]);
        angle = row[6];

        if (max_conf < conf_threshold_) {
          continue;
        }
        // Convert angle to degrees
        angle = angle * 180.0f / static_cast<float>(CV_PI);

      } else {
        // Standard YOLOv8 Raw Output: [x, y, w, h, class_scores..., angle]
        for (int c = 0; c < num_classes; ++c) {
          float conf = row[4 + c];
          if (conf > max_conf) {
            max_conf = conf;
            max_class = c;
          }
        }

        if (max_conf < conf_threshold_) {
          continue;
        }

        if (has_angle && num_features > 4 + num_classes) {
          angle = row[4 + num_classes];
          // 转换为度数
          angle = angle * 180.0f / static_cast<float>(CV_PI);
        }
      }

      // 还原到原图坐标
      cx = (cx - pad_x_) / scale_;
      cy = (cy - pad_y_) / scale_;
      w = w / scale_;
      h = h / scale_;

      // 边界检查
      cx = std::clamp(cx, 0.0f, static_cast<float>(original_size.width));
      cy = std::clamp(cy, 0.0f, static_cast<float>(original_size.height));

      if (std::isnan(cx) || std::isnan(cy) || std::isnan(w) || std::isnan(h) ||
          std::isnan(max_conf)) {
        spdlog::warn("YoloDetector: Skipping detection with NaN values");
        continue;
      }

      boxes.emplace_back(cv::Point2f(cx, cy), cv::Size2f(w, h), angle);
      confidences.push_back(max_conf);
      class_ids.push_back(max_class);
    }

    // NMS (对于OBB，使用旋转NMS)
    std::vector<int> indices;
    if (is_obb_) {
      // 自定义旋转矩形NMS
      indices = rotated_nms(boxes, confidences, nms_threshold_);
    } else {
      // 标准HBB NMS
      std::vector<cv::Rect> hbb_boxes;
      for (const auto &rr : boxes) {
        hbb_boxes.push_back(rr.boundingRect());
      }
      cv::dnn::NMSBoxes(hbb_boxes, confidences, conf_threshold_, nms_threshold_,
                        indices);
    }

    // 构建结果
    for (int idx : indices) {
      Detection det;
      det.class_id = class_ids[idx];
      det.confidence = confidences[idx];
      det.obb = boxes[idx];

      if (det.class_id < static_cast<int>(class_names_.size())) {
        det.class_name = class_names_[det.class_id];
      } else {
        det.class_name = "class_" + std::to_string(det.class_id);
      }

      detections.push_back(det);
    }

    return detections;
  } catch (const std::exception &e) {
    spdlog::error("YoloDetector: Exception in postprocess: {}", e.what());
    throw;
  }
}

// ============================================================================
// 辅助函数：旋转矩形IOU
// ============================================================================
namespace {

float rotated_iou(const cv::RotatedRect &a, const cv::RotatedRect &b) {
  std::vector<cv::Point2f> inter_pts;
  int ret = cv::rotatedRectangleIntersection(a, b, inter_pts);

  if (ret == cv::INTERSECT_NONE || inter_pts.size() < 3) {
    return 0.0f;
  }

  // 计算交集多边形面积
  std::vector<cv::Point2f> hull_pts;
  cv::convexHull(inter_pts, hull_pts);
  float inter_area = static_cast<float>(cv::contourArea(hull_pts));

  // 计算并集面积
  float area_a = a.size.width * a.size.height;
  float area_b = b.size.width * b.size.height;
  float union_area = area_a + area_b - inter_area;

  if (union_area < 1e-6f) {
    return 0.0f;
  }

  return inter_area / union_area;
}

} // anonymous namespace

std::vector<int> rotated_nms(const std::vector<cv::RotatedRect> &boxes,
                             const std::vector<float> &scores,
                             float nms_threshold) {
  // 按置信度排序
  std::vector<int> indices(boxes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

  std::vector<int> keep;
  std::vector<bool> suppressed(boxes.size(), false);

  for (int i : indices) {
    if (suppressed[i])
      continue;

    keep.push_back(i);

    for (int j : indices) {
      if (i == j || suppressed[j])
        continue;

      float iou = rotated_iou(boxes[i], boxes[j]);
      if (iou > nms_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return keep;
}

// 注册处理器
REGISTER_PROCESSOR("yolo", YoloDetector);

} // namespace yolo_edge
