#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace yolo_edge {

/**
 * 线程池
 * 简洁高效的C++20实现
 */
class ThreadPool {
public:
  /**
   * 构造函数
   * @param num_threads 线程数量
   * @param max_pending 最大排队任务数（0 表示无限制）
   */
  explicit ThreadPool(size_t num_threads, size_t max_pending = 0)
      : max_pending_(max_pending), stop_(false) {
    workers_.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  /**
   * 析构函数 - 等待所有任务完成
   */
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();

    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  // 禁止拷贝
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  /**
   * 提交任务到线程池
   * @param f 任务函数
   * @param args 参数
   * @return future对象，用于获取结果
   */
  template <typename F, typename... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        [func = std::forward<F>(f),
         ... captured_args = std::forward<Args>(args)]() mutable {
          return std::invoke(std::move(func), std::move(captured_args)...);
        });

    std::future<ReturnType> result = task->get_future();

    {
      std::unique_lock<std::mutex> lock(mutex_);

      if (stop_) {
        throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
      }

      if (max_pending_ > 0 && tasks_.size() >= max_pending_) {
        throw std::runtime_error("ThreadPool queue is full");
      }

      tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return result;
  }

  template <typename F, typename... Args>
  auto try_enqueue(F &&f, Args &&...args)
      -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        [func = std::forward<F>(f),
         ... captured_args = std::forward<Args>(args)]() mutable {
          return std::invoke(std::move(func), std::move(captured_args)...);
        });

    std::future<ReturnType> result = task->get_future();

    {
      std::unique_lock<std::mutex> lock(mutex_);

      if (stop_) {
        return std::nullopt;
      }

      if (max_pending_ > 0 && tasks_.size() >= max_pending_) {
        return std::nullopt;
      }

      tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return std::optional<std::future<ReturnType>>(std::move(result));
  }

  /**
   * 获取线程数量
   */
  size_t size() const { return workers_.size(); }

  /**
   * 获取待处理任务数量
   */
  size_t pending() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return tasks_.size();
  }

  size_t max_pending() const { return max_pending_; }

private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;

      {
        std::unique_lock<std::mutex> lock(mutex_);

        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

        if (stop_ && tasks_.empty()) {
          return;
        }

        task = std::move(tasks_.front());
        tasks_.pop();
      }

      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  size_t max_pending_ = 0;
  bool stop_;
};

} // namespace yolo_edge
