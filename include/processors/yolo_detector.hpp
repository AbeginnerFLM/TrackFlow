#pragma once

#include "core/image_processor.hpp"
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// 前向声明ONNX Runtime类型
namespace Ort {
struct Env;
struct Session;
struct SessionOptions;
struct MemoryInfo;
} // namespace Ort

namespace yolo_edge {

/**
 * 预处理结果 (线程安全: 每帧独立的 scale/pad 信息)
 */
struct PreprocessResult {
  cv::Mat blob;
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
};

/**
 * YOLO目标检测器
 * 支持ONNX Runtime推理，兼容OBB和HBB检测
 */
class YoloDetector : public ImageProcessor {
public:
  YoloDetector();
  ~YoloDetector();

  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "YoloDetector"; }
  void configure(const json &config) override;

private:
  // 预处理：letterbox + 归一化 (返回独立的 scale/pad 信息)
  PreprocessResult preprocess(const cv::Mat &frame);

  // 运行推理 (零拷贝: 直接返回 float* + shape)
  struct InferOutput {
    std::vector<float> data; // 持有数据所有权
    std::vector<int64_t> shape;
  };
  std::vector<InferOutput> infer(const cv::Mat &blob);

  // 后处理：解析输出 + NMS
  std::vector<Detection>
  postprocess(const std::vector<InferOutput> &outputs,
              const cv::Size &original_size, const PreprocessResult &prep);

  // 加载ONNX模型
  void load_model();

  // 配置参数
  std::string model_path_;
  float conf_threshold_ = 0.5f;
  float nms_threshold_ = 0.45f;
  bool is_obb_ = true;
  int input_width_ = 640;
  int input_height_ = 640;
  bool use_cuda_ = true;
  bool is_end2end_ = false; // End-to-End 模型跳过 NMS
  bool use_batch_engine_ = false; // 使用共享 BatchInferenceEngine

  // 类别名称
  std::vector<std::string> class_names_;

  // ONNX Runtime对象 (pimpl)
  struct OrtSession;
  std::unique_ptr<OrtSession> ort_;

  // 预分配 buffer (避免每帧重新分配)
  cv::Mat letterbox_buf_;

  bool initialized_ = false;
};

} // namespace yolo_edge
