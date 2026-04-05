#include "processors/yolo_detector.hpp"
#include "core/processor_factory.hpp"
#include "processors/batch_inference_engine.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_edge {

std::vector<int> rotated_nms(const std::vector<cv::RotatedRect> &boxes,
                             const std::vector<float> &scores,
                             float nms_threshold);

namespace {

std::vector<std::string> default_class_names() {
  return {"car",     "truck",  "bus",           "motorcycle",
          "bicycle", "person", "traffic_light", "stop_sign"};
}

std::vector<std::string> parse_class_names_from_string(const std::string &raw) {
  if (raw.empty()) {
    return {};
  }

  // 支持 YAML 简易解析器把 `["a","b"]` 读成字符串的情况。
  try {
    json parsed = json::parse(raw);
    if (parsed.is_array()) {
      return parsed.get<std::vector<std::string>>();
    }
  } catch (...) {
    // Ignore and fallback to CSV-like parsing below.
  }

  std::vector<std::string> result;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    auto trim = [](std::string &s) {
      while (!s.empty() &&
             std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
      }
      while (!s.empty() &&
             std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
      }
    };

    trim(token);
    if (token.size() >= 2 &&
        ((token.front() == '"' && token.back() == '"') ||
         (token.front() == '\'' && token.back() == '\''))) {
      token = token.substr(1, token.size() - 2);
    }
    trim(token);
    if (!token.empty()) {
      result.push_back(token);
    }
  }

  return result;
}

} // anonymous namespace

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

YoloDetector::YoloDetector() : ort_(std::make_unique<OrtSession>()) {}

YoloDetector::~YoloDetector() = default;

void YoloDetector::configure(const json &config) {
  model_path_ = config.value("model_path", "models/yolo_obb.onnx");
  conf_threshold_ = config.value("confidence", 0.5f);
  nms_threshold_ = config.value("nms_threshold", 0.45f);
  is_obb_ = config.value("is_obb", true);
  input_width_ = config.value("input_width", 640);
  input_height_ = config.value("input_height", 640);
  ort_threads_ = config.value("ort_threads", 4);
  batch_size_ = config.value("batch_size", 4);
  batch_wait_ms_ = config.value("batch_wait_ms", 8);
  batch_max_pending_ = config.value("batch_max_pending", 16);

  if (config.contains("use_cuda")) {
    use_cuda_ = config["use_cuda"].get<bool>();
  } else {
    use_cuda_ = true;
  }

  class_names_ = default_class_names();
  if (config.contains("class_names")) {
    try {
      const auto &raw_names = config["class_names"];
      if (raw_names.is_array()) {
        class_names_ = raw_names.get<std::vector<std::string>>();
      } else if (raw_names.is_string()) {
        auto parsed = parse_class_names_from_string(raw_names.get<std::string>());
        if (!parsed.empty()) {
          class_names_ = std::move(parsed);
        } else {
          fprintf(stderr,
                  "[WARN] class_names is string but cannot be parsed, using defaults\n");
        }
      } else {
        fprintf(stderr,
                "[WARN] class_names has invalid type, using defaults\n");
      }
    } catch (const std::exception &e) {
      fprintf(stderr,
              "[WARN] Failed to parse class_names: %s, using defaults\n",
              e.what());
      class_names_ = default_class_names();
    }
  }

  load_model();
}

void YoloDetector::load_model() {
  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(ort_threads_);
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

    auto &batch_engine = BatchInferenceEngine::instance();
    if (!batch_engine.is_initialized()) {
      batch_engine.init(model_path_, input_height_, input_width_, use_cuda_,
                        batch_size_, batch_wait_ms_, batch_max_pending_,
                        ort_threads_);
    }

    if (batch_engine.is_initialized()) {
      use_batch_engine_ = true;
      fprintf(stderr,
              "[INFO] YoloDetector: Using shared BatchInferenceEngine\n");
    } else {
      use_batch_engine_ = false;
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

    auto prep = preprocess(ctx.frame);

    auto outputs = infer(prep.blob);

    ctx.detections = postprocess(outputs, ctx.frame.size(), prep);

    auto end = Clock::now();
    ctx.infer_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] YoloDetector: %s\n", e.what());
    return false;
  }
}

PreprocessResult YoloDetector::preprocess(const cv::Mat &frame) {
  PreprocessResult result;

  // 难点：这个函数会被线程池并发调用，使用 thread_local 缓冲避免加锁和反复分配。
  thread_local cv::Mat letterbox_buf;
  thread_local cv::Mat resized_buf;
  thread_local cv::Mat rgb_buf;
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

  cv::resize(frame, resized_buf, cv::Size(new_w, new_h));

  letterbox_buf.setTo(cv::Scalar(114, 114, 114));
  resized_buf.copyTo(
      letterbox_buf(cv::Rect(result.pad_x, result.pad_y, new_w, new_h)));

  cv::cvtColor(letterbox_buf, rgb_buf, cv::COLOR_BGR2RGB);

  rgb_buf.convertTo(rgb_buf, CV_32F, 1.0 / 255.0);

  cv::dnn::blobFromImage(rgb_buf, result.blob);

  return result;
}

std::vector<YoloDetector::InferOutput>
YoloDetector::infer(const cv::Mat &blob) {
  try {
    if (use_batch_engine_) {
      auto &engine = BatchInferenceEngine::instance();
      if (engine.is_initialized()) {
        try {
          BatchInferenceEngine::InferRequest req;
          req.blob = blob;

          auto future = engine.submit(std::move(req));
          auto result = future.get();

          if (result.success) {
            InferOutput out;
            out.data = std::move(result.data);
            out.shape = std::move(result.shape);
            return {std::move(out)};
          }
          fprintf(stderr, "[WARN] YoloDetector: Batch inference failed, fallback to local session\n");
        } catch (const std::exception &e) {
          fprintf(stderr,
                  "[WARN] YoloDetector: Batch engine unavailable (%s), fallback to local session\n",
                  e.what());
        }
      }
    }

    if (!ort_->session) {
      throw std::runtime_error("Local ONNX session is unavailable");
    }

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    const size_t input_size = static_cast<size_t>(3) * input_height_ * input_width_;

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

std::vector<Detection>
YoloDetector::postprocess(const std::vector<InferOutput> &outputs,
                          const cv::Size &original_size,
                          const PreprocessResult &prep) {
  try {
    if (outputs.empty())
      return {};

    const auto &out = outputs[0];

    int dim1 = 0;
    int dim2 = 0;
    if (out.shape.size() == 3) {
      dim1 = static_cast<int>(out.shape[1]);
      dim2 = static_cast<int>(out.shape[2]);
    } else if (out.shape.size() == 2) {
      dim1 = static_cast<int>(out.shape[0]);
      dim2 = static_cast<int>(out.shape[1]);
    } else {
      return {};
    }

    bool transposed = (dim1 < dim2);
    int num_boxes = transposed ? dim2 : dim1;
    int num_features = transposed ? dim1 : dim2;

    if (num_boxes > 100000 || num_boxes < 0)
      return {};

    std::vector<Detection> detections;
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    bool is_e2e = (num_features == 7);

    int num_classes;
    bool has_angle = is_obb_;
    if (is_e2e) {
      num_classes = 1;
    } else if (has_angle) {
      num_classes = num_features - 5;
    } else {
      num_classes = num_features - 4;
    }

    if (num_classes <= 0 && !is_e2e)
      return {};

    const float *raw = out.data.data();
    std::vector<float> row_buf;
    row_buf.reserve(static_cast<size_t>(num_features));

    auto read_feature = [&](int box_idx, int feat_idx) -> float {
      if (transposed) {
        return raw[feat_idx * num_boxes + box_idx];
      }
      return raw[box_idx * num_features + feat_idx];
    };

    for (int i = 0; i < num_boxes; ++i) {
      if (transposed) {
        row_buf.resize(static_cast<size_t>(num_features));
        for (int f = 0; f < num_features; ++f) {
          row_buf[static_cast<size_t>(f)] = read_feature(i, f);
        }
      }

      auto feature = [&](int idx) -> float {
        if (transposed) {
          return row_buf[static_cast<size_t>(idx)];
        }
        return read_feature(i, idx);
      };

      float cx = feature(0);
      float cy = feature(1);
      float w = feature(2);
      float h = feature(3);
      float angle = 0, max_conf = 0;
      int max_class = 0;

      if (is_e2e) {
        max_conf = feature(4);
        max_class = static_cast<int>(feature(5));
        angle = feature(6) * 180.0f / static_cast<float>(CV_PI);
      } else {
        for (int c = 0; c < num_classes; ++c) {
          float score = feature(4 + c);
          if (score > max_conf) {
            max_conf = score;
            max_class = c;
          }
        }
        if (has_angle && num_features > 4 + num_classes)
          angle = feature(4 + num_classes) * 180.0f / static_cast<float>(CV_PI);
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

    std::vector<int> indices;
    if (is_e2e) {
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

      det.refresh_geometry();
      detections.push_back(det);
    }

    return detections;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] YoloDetector postprocess: %s\n", e.what());
    throw;
  }
}

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

  std::vector<cv::Rect> aabbs(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i)
    aabbs[i] = boxes[i].boundingRect();

  std::vector<int> keep;
  std::vector<bool> suppressed(boxes.size(), false);

  for (size_t p = 0; p < indices.size(); ++p) {
    int i = indices[p];
    if (suppressed[i])
      continue;

    keep.push_back(i);

    for (size_t q = p + 1; q < indices.size(); ++q) {
      int j = indices[q];
      if (suppressed[j])
        continue;

      if ((aabbs[i] & aabbs[j]).area() == 0)
        continue;

      float iou = rotated_iou(boxes[i], boxes[j]);
      if (iou > nms_threshold)
        suppressed[j] = true;
    }
  }

  return keep;
}

REGISTER_PROCESSOR("yolo", YoloDetector);

} // namespace yolo_edge
