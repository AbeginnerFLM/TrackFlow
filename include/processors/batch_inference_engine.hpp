#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <queue>
#include <thread>
#include <vector>

namespace yolo_edge {

/**
 * Batch 推理引擎 (单例)
 *
 * 收集多个请求的预处理结果, 合并为一个 batch 送入 GPU 推理,
 * 然后拆分输出返回给各请求线程.
 *
 * 动态批处理策略:
 *   - 攒够 max_batch_size 立即触发
 *   - 或等待 max_wait_ms 后触发 (避免低负载时延迟过高)
 */
class BatchInferenceEngine {
public:
  struct InferRequest {
    cv::Mat blob;       // 预处理后的 {1, 3, H, W} 张量
    float scale;        // 缩放比例 (用于后处理)
    int pad_x, pad_y;   // padding (用于后处理)
    cv::Size orig_size; // 原始图像尺寸
  };

  struct InferResult {
    std::vector<float> data;
    std::vector<int64_t> shape;
    bool success = false;
  };

  /**
   * 获取全局单例
   */
  static BatchInferenceEngine &instance();

  /**
   * 初始化引擎 (首次调用时设置参数)
   */
  void init(const std::string &model_path, int input_h, int input_w,
            bool use_cuda, int max_batch_size = 4, int max_wait_ms = 8);

  /**
   * 提交推理请求 (线程安全, 阻塞等待结果)
   * 返回该帧在 batch 输出中对应的结果
   */
  std::future<InferResult> submit(InferRequest req);

  /**
   * 停止引擎
   */
  void shutdown();

  bool is_initialized() const { return initialized_; }

  ~BatchInferenceEngine();

private:
  BatchInferenceEngine() = default;
  BatchInferenceEngine(const BatchInferenceEngine &) = delete;
  BatchInferenceEngine &operator=(const BatchInferenceEngine &) = delete;

  void worker_loop();
  void run_batch(std::vector<InferRequest> &requests,
                 std::vector<std::promise<InferResult>> &promises);

  // ONNX Runtime (pimpl)
  struct OrtData;
  std::unique_ptr<OrtData> ort_;

  // 配置
  int input_h_ = 640;
  int input_w_ = 640;
  int max_batch_size_ = 4;
  int max_wait_ms_ = 8;

  // 队列
  struct PendingItem {
    InferRequest request;
    std::promise<InferResult> promise;
  };
  std::queue<PendingItem> queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  // Worker 线程
  std::thread worker_;
  bool running_ = false;
  bool initialized_ = false;
};

} // namespace yolo_edge
