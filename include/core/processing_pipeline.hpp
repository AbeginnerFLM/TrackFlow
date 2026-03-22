#pragma once

#include "image_processor.hpp"
#include <chrono>
#include <cstdio>
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
    return execute_range(ctx, 0, processors_.size());
  }

  /**
   * 执行管道中的一段 [from, to)
   * 用于分阶段执行: decode+yolo 并行, tracker 串行
   */
  bool execute_range(ProcessingContext &ctx, size_t from, size_t to) {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    for (size_t i = from; i < to && i < processors_.size(); ++i) {
      if (!processors_[i]->process(ctx)) {
        fprintf(stderr, "[ERROR] Processor '%s' failed\n", processors_[i]->name().c_str());
        return false;
      }
    }

    if (from == 0 && to >= processors_.size()) {
      auto end = Clock::now();
      ctx.total_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
    }
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
