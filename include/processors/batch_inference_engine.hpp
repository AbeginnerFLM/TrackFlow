#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace yolo_edge {

class BatchInferenceEngine {
public:
  struct InferRequest {
    cv::Mat blob;
  };

  struct InferResult {
    std::vector<float> data;
    std::vector<int64_t> shape;
    bool success = false;
  };

  static BatchInferenceEngine &instance();

  void init(const std::string &model_path, int input_h, int input_w,
            bool use_cuda, int max_batch_size = 4, int max_wait_ms = 8,
            int max_pending = 0, int ort_threads = 4);

  std::future<InferResult> submit(InferRequest req);
  std::optional<std::future<InferResult>> try_submit(InferRequest req);

  void shutdown();

  bool is_initialized() const { return initialized_; }
  size_t pending() const;

  ~BatchInferenceEngine();

private:
  BatchInferenceEngine() = default;
  BatchInferenceEngine(const BatchInferenceEngine &) = delete;
  BatchInferenceEngine &operator=(const BatchInferenceEngine &) = delete;

  void worker_loop();
  void run_batch(std::vector<InferRequest> &requests,
                 std::vector<std::promise<InferResult>> &promises);

  struct OrtData;
  std::unique_ptr<OrtData> ort_;

  int input_h_ = 640;
  int input_w_ = 640;
  int max_batch_size_ = 4;
  int max_wait_ms_ = 8;
  int max_pending_ = 0;
  int ort_threads_ = 4;

  struct PendingItem {
    InferRequest request;
    std::promise<InferResult> promise;
  };
  std::queue<PendingItem> queue_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  std::thread worker_;
  bool running_ = false;
  bool initialized_ = false;
};

} // namespace yolo_edge
