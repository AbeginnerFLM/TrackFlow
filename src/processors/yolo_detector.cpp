#include "processors/yolo_detector.hpp"
#include "core/processor_factory.hpp"
#include "processors/batch_inference_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
  model_path_ = config.value("model_path", "models/yolo_obb.onnx");
  conf_threshold_ = config.value("confidence", 0.5f);
  nms_threshold_ = config.value("nms_threshold", 0.45f);
  is_obb_ = config.value("is_obb", true);
  input_width_ = config.value("input_width", 640);
  input_height_ = config.value("input_height", 640);

  if (config.contains("use_cuda")) {
    use_cuda_ = config["use_cuda"].get<bool>();
  } else {
    use_cuda_ = true;
  }

  if (config.contains("class_names")) {
    class_names_ = config["class_names"].get<std::vector<std::string>>();
  } else {
    class_names_ = {"car",     "truck",  "bus",           "motorcycle",
                    "bicycle", "person", "traffic_light", "stop_sign"};
  }

  load_model();
}

// ============================================================================
// 加载模型
// ============================================================================
void YoloDetector::load_model() {
  try {
    // Step 1: 创建本地 session 读取模型元数据 (input shape)
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool cuda_enabled = false;
    if (use_cuda_) {
      try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        cuda_enabled = true;
      } catch (const Ort::Exception &e) {
        fprintf(stderr,
                "[WARN] Failed to enable CUDA: %s, falling back to CPU\n",
                e.what());
      }
    }

    ort_->session = std::make_unique<Ort::Session>(
        ort_->env, model_path_.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = ort_->session->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
      auto name = ort_->session->GetInputNameAllocated(i, allocator);
      ort_->input_names_storage.push_back(name.get());
      ort_->input_names.push_back(ort_->input_names_storage.back().c_str());

      auto type_info = ort_->session->GetInputTypeInfo(i);
      auto shape_info = type_info.GetTensorTypeAndShapeInfo();
      ort_->input_shape = shape_info.GetShape();
    }

    size_t num_outputs = ort_->session->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
      auto name = ort_->session->GetOutputNameAllocated(i, allocator);
      ort_->output_names_storage.push_back(name.get());
      ort_->output_names.push_back(ort_->output_names_storage.back().c_str());
    }

    // 从模型元数据更新 input dimensions
    if (ort_->input_shape.size() >= 4) {
      if (ort_->input_shape[2] > 0)
        input_height_ = ort_->input_shape[2];
      if (ort_->input_shape[3] > 0)
        input_width_ = ort_->input_shape[3];
    }

    fprintf(stderr, "[INFO] Loaded model '%s' (input: %dx%d, %s)\n",
            model_path_.c_str(), input_width_, input_height_,
            cuda_enabled ? "GPU" : "CPU");

    // Step 2: 预分配 letterbox buffer (必须在 warmup 之前)
    letterbox_buf_ = cv::Mat(input_height_, input_width_, CV_8UC3);

    // Step 3: 尝试 BatchInferenceEngine (使用正确的 dimensions)
    auto &batch_engine = BatchInferenceEngine::instance();
    if (!batch_engine.is_initialized()) {
      batch_engine.init(model_path_, input_height_, input_width_, use_cuda_,
                        /*max_batch_size=*/4, /*max_wait_ms=*/8);
    }

    if (batch_engine.is_initialized()) {
      use_batch_engine_ = true;
      // 释放本地 session (BatchEngine 有自己的)
      ort_->session.reset();
      fprintf(stderr,
              "[INFO] YoloDetector: Using shared BatchInferenceEngine\n");
    } else {
      use_batch_engine_ = false;
      // Warmup 本地 session
      if (cuda_enabled) {
        fprintf(stderr, "[INFO] Warming up CUDA engine...\n");
        cv::Mat dummy(input_height_, input_width_, CV_8UC3,
                      cv::Scalar(0, 0, 0));
        auto prep = preprocess(dummy);
        infer(prep.blob);
        fprintf(stderr, "[INFO] Warmup complete.\n");
      }
    }

    initialized_ = true;

  } catch (const Ort::Exception &e) {
    fprintf(stderr, "[ERROR] Failed to load model: %s\n", e.what());
    initialized_ = false;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] Failed to load model: %s\n", e.what());
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

    // 预处理 (使用局部 PreprocessResult, 线程安全)
    auto prep = preprocess(ctx.frame);
    auto t2 = Clock::now();

    // 推理 (零拷贝输出)
    auto outputs = infer(prep.blob);
    auto t3 = Clock::now();

    // 后处理 (使用 prep 的 scale/pad)
    ctx.detections = postprocess(outputs, ctx.frame.size(), prep);
    auto t4 = Clock::now();

    auto end = Clock::now();
    ctx.infer_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] YoloDetector: %s\n", e.what());
    return false;
  }
}

// ============================================================================
// 预处理 (Letterbox + 归一化) - Buffer 复用 + 参数局部化
// ============================================================================
PreprocessResult YoloDetector::preprocess(const cv::Mat &frame) {
  PreprocessResult result;

  // Thread-local buffer for concurrent preprocess (batch pipeline)
  thread_local cv::Mat letterbox_buf;
  if (letterbox_buf.rows != input_height_ || letterbox_buf.cols != input_width_ ||
      letterbox_buf.type() != CV_8UC3) {
    letterbox_buf.create(input_height_, input_width_, CV_8UC3);
  }

  int img_w = frame.cols;
  int img_h = frame.rows;

  float scale_w = static_cast<float>(input_width_) / img_w;
  float scale_h = static_cast<float>(input_height_) / img_h;
  result.scale = std::min(scale_w, scale_h);

  int new_w = static_cast<int>(img_w * result.scale);
  int new_h = static_cast<int>(img_h * result.scale);

  result.pad_x = (input_width_ - new_w) / 2;
  result.pad_y = (input_height_ - new_h) / 2;

  // 缩放到目标尺寸
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(new_w, new_h));

  // 填充灰色 + 粘贴缩放图
  letterbox_buf.setTo(cv::Scalar(114, 114, 114));
  resized.copyTo(
      letterbox_buf(cv::Rect(result.pad_x, result.pad_y, new_w, new_h)));

  // BGR -> RGB
  cv::Mat rgb;
  cv::cvtColor(letterbox_buf, rgb, cv::COLOR_BGR2RGB);

  // 归一化到 [0, 1]
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // HWC -> CHW
  cv::dnn::blobFromImage(rgb, result.blob);

  return result;
}

// ============================================================================
// 推理 - 支持 BatchInferenceEngine 共享推理
// ============================================================================
std::vector<YoloDetector::InferOutput>
YoloDetector::infer(const cv::Mat &blob) {
  try {
    // 优先使用 BatchInferenceEngine (共享 GPU, 动态批处理)
    if (use_batch_engine_) {
      auto &engine = BatchInferenceEngine::instance();
      if (engine.is_initialized()) {
        BatchInferenceEngine::InferRequest req;
        req.blob = blob; // shallow copy (共享数据)
        req.scale = 1.0f;
        req.pad_x = 0;
        req.pad_y = 0;
        req.orig_size = cv::Size(input_width_, input_height_);

        auto future = engine.submit(std::move(req));
        auto result = future.get();

        if (result.success) {
          InferOutput out;
          out.data = std::move(result.data);
          out.shape = std::move(result.shape);
          return {std::move(out)};
        }
        // 失败则回退到本地推理
      }
    }

    // 回退: 本地 ONNX session 推理
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    size_t input_size = 1 * 3 * input_height_ * input_width_;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_->memory_info, const_cast<float *>(blob.ptr<float>()), input_size,
        input_shape.data(), input_shape.size());

    auto output_tensors = ort_->session->Run(
        Ort::RunOptions{nullptr}, ort_->input_names.data(), &input_tensor, 1,
        ort_->output_names.data(), ort_->output_names.size());

    std::vector<InferOutput> outputs;
    outputs.reserve(output_tensors.size());

    for (auto &tensor : output_tensors) {
      auto *data = tensor.GetTensorMutableData<float>();
      auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();

      size_t total = 1;
      for (auto d : shape)
        total *= d;

      InferOutput out;
      out.shape = shape;
      out.data.assign(data, data + total);
      outputs.push_back(std::move(out));
    }

    return outputs;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] YoloDetector infer: %s\n", e.what());
    throw;
  }
}

// ============================================================================
// 后处理 (解析 + NMS, E2E 模型跳过 NMS)
// ============================================================================
std::vector<Detection>
YoloDetector::postprocess(const std::vector<InferOutput> &outputs,
                          const cv::Size &original_size,
                          const PreprocessResult &prep) {
  try {
    if (outputs.empty())
      return {};

    const auto &out = outputs[0];

    // 确定行列 (将 shape 映射到 [num_boxes, num_features])
    int dim1, dim2;
    if (out.shape.size() == 3) {
      dim1 = out.shape[1];
      dim2 = out.shape[2];
    } else if (out.shape.size() == 2) {
      dim1 = out.shape[0];
      dim2 = out.shape[1];
    } else {
      return {};
    }

    // 转置: 如果 features < boxes, 则需要转置
    bool transposed = (dim1 < dim2);
    int num_boxes = transposed ? dim2 : dim1;
    int num_features = transposed ? dim1 : dim2;

    if (num_boxes > 100000 || num_boxes < 0)
      return {};

    std::vector<Detection> detections;
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // End-to-End 检测 (7 列: x,y,w,h,score,class,angle)
    bool is_e2e = (num_features == 7);

    int num_classes;
    bool has_angle = is_obb_;
    if (is_e2e) {
      num_classes = 1; // E2E 直接给 class_id
    } else if (has_angle) {
      num_classes = num_features - 5;
    } else {
      num_classes = num_features - 4;
    }

    if (num_classes <= 0 && !is_e2e)
      return {};

    const float *raw = out.data.data();

    for (int i = 0; i < num_boxes; ++i) {
      const float *row;
      if (transposed) {
        // 需要跳着读: raw[feat * num_boxes + i]
        // 为简单起见, 构造临时行
        // 但这种格式通常 num_features < 20, 所以还好
        float tmp[32];
        for (int f = 0; f < num_features && f < 32; ++f)
          tmp[f] = raw[f * num_boxes + i];
        row = tmp;

        // 内联处理以避免 tmp 生命周期问题
        float cx = tmp[0], cy = tmp[1], w = tmp[2], h = tmp[3];
        float angle = 0, max_conf = 0;
        int max_class = 0;

        if (is_e2e) {
          max_conf = tmp[4];
          max_class = static_cast<int>(tmp[5]);
          angle = tmp[6] * 180.0f / static_cast<float>(CV_PI);
        } else {
          for (int c = 0; c < num_classes; ++c) {
            if (tmp[4 + c] > max_conf) {
              max_conf = tmp[4 + c];
              max_class = c;
            }
          }
          if (has_angle && num_features > 4 + num_classes)
            angle = tmp[4 + num_classes] * 180.0f / static_cast<float>(CV_PI);
        }

        if (max_conf < conf_threshold_)
          continue;

        cx = (cx - prep.pad_x) / prep.scale;
        cy = (cy - prep.pad_y) / prep.scale;
        w = w / prep.scale;
        h = h / prep.scale;

        cx = std::clamp(cx, 0.0f, static_cast<float>(original_size.width));
        cy = std::clamp(cy, 0.0f, static_cast<float>(original_size.height));

        if (std::isnan(cx) || std::isnan(cy) || std::isnan(w) ||
            std::isnan(h) || std::isnan(max_conf))
          continue;

        boxes.emplace_back(cv::Point2f(cx, cy), cv::Size2f(w, h), angle);
        confidences.push_back(max_conf);
        class_ids.push_back(max_class);
        continue; // 已处理, 跳过下面的 row 版本
      }

      row = raw + i * num_features;
      float cx = row[0], cy = row[1], w = row[2], h = row[3];
      float angle = 0, max_conf = 0;
      int max_class = 0;

      if (is_e2e) {
        max_conf = row[4];
        max_class = static_cast<int>(row[5]);
        angle = row[6] * 180.0f / static_cast<float>(CV_PI);
      } else {
        for (int c = 0; c < num_classes; ++c) {
          if (row[4 + c] > max_conf) {
            max_conf = row[4 + c];
            max_class = c;
          }
        }
        if (has_angle && num_features > 4 + num_classes)
          angle =
              row[4 + num_classes] * 180.0f / static_cast<float>(CV_PI);
      }

      if (max_conf < conf_threshold_)
        continue;

      cx = (cx - prep.pad_x) / prep.scale;
      cy = (cy - prep.pad_y) / prep.scale;
      w = w / prep.scale;
      h = h / prep.scale;

      cx = std::clamp(cx, 0.0f, static_cast<float>(original_size.width));
      cy = std::clamp(cy, 0.0f, static_cast<float>(original_size.height));

      if (std::isnan(cx) || std::isnan(cy) || std::isnan(w) ||
          std::isnan(h) || std::isnan(max_conf))
        continue;

      boxes.emplace_back(cv::Point2f(cx, cy), cv::Size2f(w, h), angle);
      confidences.push_back(max_conf);
      class_ids.push_back(max_class);
    }

    // NMS: E2E 模型跳过 (NMS 已在模型内完成)
    std::vector<int> indices;
    if (is_e2e) {
      // End-to-End 模型输出已经是 NMS 后的结果, 直接使用
      indices.resize(boxes.size());
      std::iota(indices.begin(), indices.end(), 0);
    } else if (is_obb_) {
      indices = rotated_nms(boxes, confidences, nms_threshold_);
    } else {
      std::vector<cv::Rect> hbb_boxes;
      hbb_boxes.reserve(boxes.size());
      for (const auto &rr : boxes)
        hbb_boxes.push_back(rr.boundingRect());
      cv::dnn::NMSBoxes(hbb_boxes, confidences, conf_threshold_,
                        nms_threshold_, indices);
    }

    // 构建结果
    detections.reserve(indices.size());
    for (int idx : indices) {
      Detection det;
      det.class_id = class_ids[idx];
      det.confidence = confidences[idx];
      det.obb = boxes[idx];

      if (det.class_id < static_cast<int>(class_names_.size()))
        det.class_name = class_names_[det.class_id];
      else
        det.class_name = "class_" + std::to_string(det.class_id);

      detections.push_back(det);
    }

    return detections;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] YoloDetector postprocess: %s\n", e.what());
    throw;
  }
}

// ============================================================================
// 旋转矩形 NMS (AABB 预过滤优化)
// ============================================================================
namespace {

float rotated_iou(const cv::RotatedRect &a, const cv::RotatedRect &b) {
  std::vector<cv::Point2f> inter_pts;
  int ret = cv::rotatedRectangleIntersection(a, b, inter_pts);

  if (ret == cv::INTERSECT_NONE || inter_pts.size() < 3)
    return 0.0f;

  std::vector<cv::Point2f> hull_pts;
  cv::convexHull(inter_pts, hull_pts);
  float inter_area = static_cast<float>(cv::contourArea(hull_pts));

  float area_a = a.size.width * a.size.height;
  float area_b = b.size.width * b.size.height;
  float union_area = area_a + area_b - inter_area;

  if (union_area < 1e-6f)
    return 0.0f;

  return inter_area / union_area;
}

} // anonymous namespace

std::vector<int> rotated_nms(const std::vector<cv::RotatedRect> &boxes,
                             const std::vector<float> &scores,
                             float nms_threshold) {
  std::vector<int> indices(boxes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

  // 预计算 AABB 用于快速排除
  std::vector<cv::Rect> aabbs(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i)
    aabbs[i] = boxes[i].boundingRect();

  std::vector<int> keep;
  std::vector<bool> suppressed(boxes.size(), false);

  for (int i : indices) {
    if (suppressed[i])
      continue;

    keep.push_back(i);

    for (int j : indices) {
      if (i == j || suppressed[j])
        continue;

      // AABB 预过滤: 如果 AABB 不相交, 旋转 IoU 必为 0
      if ((aabbs[i] & aabbs[j]).area() == 0)
        continue;

      float iou = rotated_iou(boxes[i], boxes[j]);
      if (iou > nms_threshold)
        suppressed[j] = true;
    }
  }

  return keep;
}

// 注册处理器
REGISTER_PROCESSOR("yolo", YoloDetector);

} // namespace yolo_edge
