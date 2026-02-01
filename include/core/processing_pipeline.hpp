#pragma once

#include "image_processor.hpp"
#include <chrono>
#include <spdlog/spdlog.h>
#include <vector>

namespace yolo_edge {

/**
 * 处理管道 (责任链模式)
 * 按顺序执行多个处理器
 */
class ProcessingPipeline {
public:
  /**
   * 添加处理器到管道末尾
   */
  void add(ProcessorPtr processor) {
    processors_.push_back(std::move(processor));
  }

  /**
   * 执行整个管道
   * @param ctx 处理上下文
   * @return true=所有处理器成功, false=某个处理器失败
   */
  bool execute(ProcessingContext &ctx) {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    for (auto &proc : processors_) {
      spdlog::debug("Executing processor: {}", proc->name());

      if (!proc->process(ctx)) {
        spdlog::error("Processor '{}' failed", proc->name());
        return false;
      }
    }

    auto end = Clock::now();
    ctx.total_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    spdlog::debug("Pipeline completed in {:.2f}ms", ctx.total_time_ms);
    return true;
  }

  /**
   * 清空管道
   */
  void clear() { processors_.clear(); }

  /**
   * 获取处理器数量
   */
  size_t size() const { return processors_.size(); }

  /**
   * 检查管道是否为空
   */
  bool empty() const { return processors_.empty(); }

private:
  std::vector<ProcessorPtr> processors_;
};

} // namespace yolo_edge
