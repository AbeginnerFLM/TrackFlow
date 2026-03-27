#pragma once

#include "core/image_processor.hpp"
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace Ort {
struct Env;
struct Session;
struct SessionOptions;
struct MemoryInfo;
} // namespace Ort

namespace yolo_edge {

struct PreprocessResult {
  cv::Mat blob;
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
};

class YoloDetector : public ImageProcessor {
public:
  YoloDetector();
  ~YoloDetector();

  bool process(ProcessingContext &ctx) override;
  std::string name() const override { return "YoloDetector"; }
  void configure(const json &config) override;

private:
  PreprocessResult preprocess(const cv::Mat &frame);

  struct InferOutput {
    std::vector<float> data;
    std::vector<int64_t> shape;
  };
  std::vector<InferOutput> infer(const cv::Mat &blob);

  std::vector<Detection>
  postprocess(const std::vector<InferOutput> &outputs,
              const cv::Size &original_size, const PreprocessResult &prep);

  void load_model();

  std::string model_path_;
  float conf_threshold_ = 0.5f;
  float nms_threshold_ = 0.45f;
  bool is_obb_ = true;
  int input_width_ = 640;
  int input_height_ = 640;
  bool use_cuda_ = true;
  bool use_batch_engine_ = false;

  std::vector<std::string> class_names_;

  struct OrtSession;
  std::unique_ptr<OrtSession> ort_;

  bool initialized_ = false;
};

} // namespace yolo_edge
