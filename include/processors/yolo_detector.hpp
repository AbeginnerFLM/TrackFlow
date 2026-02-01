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
  // 预处理：letterbox + 归一化
  cv::Mat preprocess(const cv::Mat &frame);

  // 运行推理
  std::vector<cv::Mat> infer(const cv::Mat &blob);

  // 后处理：解析输出 + NMS
  std::vector<Detection> postprocess(const std::vector<cv::Mat> &outputs,
                                     const cv::Size &original_size);

  // 加载ONNX模型
  void load_model();

  // 配置参数
  std::string model_path_;
  float conf_threshold_ = 0.5f;
  float nms_threshold_ = 0.45f;
  bool is_obb_ = true; // OBB或HBB模式
  int input_width_ = 640;
  int input_height_ = 640;
  bool use_cuda_ = true;

  // 类别名称
  std::vector<std::string> class_names_;

  // ONNX Runtime对象 (使用pimpl避免头文件依赖)
  struct OrtSession;
  std::unique_ptr<OrtSession> ort_;

  // letterbox缩放信息 (用于坐标还原)
  float scale_ = 1.0f;
  int pad_x_ = 0;
  int pad_y_ = 0;

  bool initialized_ = false;
};

} // namespace yolo_edge
