#include "processors/yolo_detector.hpp"
#include "core/processor_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

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
  // Debug: print full config
  fprintf(stderr, "[DEBUG] YoloDetector::configure received config: %s\n",
          config.dump().c_str());

  model_path_ = config.value("model_path", "models/yolo_obb.onnx");
  conf_threshold_ = config.value("confidence", 0.5f);
  nms_threshold_ = config.value("nms_threshold", 0.45f);
  is_obb_ = config.value("is_obb", true);
  input_width_ = config.value("input_width", 640);
  input_height_ = config.value("input_height", 640);

  // Explicitly check use_cuda
  if (config.contains("use_cuda")) {
    use_cuda_ = config["use_cuda"].get<bool>();
    fprintf(stderr, "[DEBUG] Found use_cuda in config: %s\n",
            use_cuda_ ? "true" : "false");
  } else {
    use_cuda_ = true; // Default
    fprintf(stderr, "[DEBUG] use_cuda not in config, defaulting to true\n");
  }

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
  fprintf(stderr, "[DEBUG] YoloDetector::load_model called. use_cuda_=%s\n",
          use_cuda_ ? "true" : "false");
  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool cuda_enabled = false;
    if (use_cuda_) {
      // ... (rest of the code)
      try {
        // 配置 CUDA Execution Provider
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0; // 默认策略
        cuda_options.gpu_mem_limit = 0;         // 无内存限制
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(cuda_options);
        cuda_enabled = true;
        fprintf(stderr,
                "[INFO] YoloDetector: CUDA Execution Provider configured\n");
      } catch (const Ort::Exception &e) {
        fprintf(stderr, "[WARN] YoloDetector: Failed to enable CUDA: %s\n",
                e.what());
        fprintf(stderr, "[WARN] YoloDetector: Falling back to CPU inference\n");
      }
    }

    ort_->session = std::make_unique<Ort::Session>(
        ort_->env, model_path_.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // 获取输入信息
    size_t num_inputs = ort_->session->GetInputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
      auto name = ort_->session->GetInputNameAllocated(i, allocator);
      ort_->input_names_storage.push_back(name.get());
      ort_->input_names.push_back(ort_->input_names_storage.back().c_str());

      auto type_info = ort_->session->GetInputTypeInfo(i);
      auto shape_info = type_info.GetTensorTypeAndShapeInfo();
      ort_->input_shape = shape_info.GetShape();
    }

    // 获取输出信息
    size_t num_outputs = ort_->session->GetOutputCount();

    for (size_t i = 0; i < num_outputs; ++i) {
      auto name = ort_->session->GetOutputNameAllocated(i, allocator);
      ort_->output_names_storage.push_back(name.get());
      ort_->output_names.push_back(ort_->output_names_storage.back().c_str());
    }

    if (ort_->input_shape.size() >= 4) {
      if (ort_->input_shape[2] > 0)
        input_height_ = ort_->input_shape[2];
      if (ort_->input_shape[3] > 0)
        input_width_ = ort_->input_shape[3];
    }

    initialized_ = true;
    fprintf(stderr,
            "[INFO] YoloDetector: Loaded model '%s' (input: %dx%d, %s)\n",
            model_path_.c_str(), input_width_, input_height_,
            cuda_enabled ? "GPU" : "CPU");
    // Warmup
    if (cuda_enabled) {
      fprintf(stderr, "[INFO] YoloDetector: Warming up CUDA engine...\n");
      cv::Mat dummy(input_height_, input_width_, CV_8UC3, cv::Scalar(0, 0, 0));
      cv::Mat blob = preprocess(dummy);
      infer(blob);
      fprintf(stderr, "[INFO] YoloDetector: Warmup complete.\n");
    }

  } catch (const Ort::Exception &e) {
    fprintf(stderr,
            "[ERROR] YoloDetector: Failed to load model (Ort::Exception): %s\n",
            e.what());
    initialized_ = false;
  } catch (const std::exception &e) {
    fprintf(stderr,
            "[ERROR] YoloDetector: Failed to load model (std::exception): %s\n",
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
      fprintf(stderr, "[ERROR] YoloDetector: Model not initialized\n");
      return false;
    }

    if (ctx.frame.empty()) {
      fprintf(stderr, "[ERROR] YoloDetector: Input frame is empty\n");
      return false;
    }

    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    // fprintf(stderr, "[INFO] YoloDetector: Starting preprocess...\n");

    // 预处理
    auto t1 = Clock::now();
    cv::Mat blob = preprocess(ctx.frame);
    auto t2 = Clock::now();

    // fprintf(stderr, "[INFO] YoloDetector: Preprocess complete. Starting
    // inference...\n");

    // 推理
    auto outputs = infer(blob);
    auto t3 = Clock::now();

    // fprintf(stderr, "[INFO] YoloDetector: Inference complete. Starting
    // postprocess...\n");

    // 后处理
    ctx.detections = postprocess(outputs, ctx.frame.size());
    auto t4 = Clock::now();

    auto end = Clock::now();
    ctx.infer_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    double pre_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double infer_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    double post_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    fprintf(
        stderr,
        "[DEBUG] Timing: Pre=%.2fms, Infer=%.2fms, Post=%.2fms, Total=%.2fms\n",
        pre_ms, infer_ms, post_ms, ctx.infer_time_ms);

    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] YoloDetector: Exception in process: %s\n",
            e.what());
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
    fprintf(stderr, "[ERROR] YoloDetector: Exception in infer: %s\n", e.what());
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

    // fprintf(stderr, "[INFO] YoloDetector: Postprocess input shape: %dx%d\n",
    // num_boxes,
    //              num_features);

    // fprintf(stderr, "[INFO] DEBUG_SHAPE: %dx%d\n", num_boxes, num_features);

    if (num_boxes > 100000 || num_boxes < 0) {
      fprintf(stderr, "[ERROR] YoloDetector: Invalid number of boxes: %d\n",
              num_boxes);
      return {};
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
      fprintf(stderr, "[WARN] YoloDetector: Invalid output shape\n");
      // Force 7-feature logic if matched
      if (num_features != 7)
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
        // fprintf(stderr, "[WARN] YoloDetector: Skipping detection with NaN
        // values\n");
        continue;
      }

      boxes.emplace_back(cv::Point2f(cx, cy), cv::Size2f(w, h), angle);
      confidences.push_back(max_conf);
      class_ids.push_back(max_class);
    }

    // NMS (对于OBB，使用旋转NMS)
    std::vector<int> indices;
    if (num_features == 7) {
      // End-to-End model usually has NMS built-in or output is sparse.
      // We can assume valid output or run NMS anyway if needed.
      // The original code ran user-side NMS. We will keep it.
      // Wait, if NMS is 'inside model' (End-to-End), usually output is already
      // filtered. But if we parsed ALL rows, we might need simple thresholding.
      // The loop above already thresholded by conf.
      // Let's run NMS to be safe, assuming output might be raw candidates.
      // Actually user said "NMS inside model".
      // If so, output boxes should be few.
      // But let's keep logic simple.
    }

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
    fprintf(stderr, "[ERROR] YoloDetector: Exception in postprocess: %s\n",
            e.what());
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
