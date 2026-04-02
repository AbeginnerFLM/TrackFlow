#pragma once

#include "image_processor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace yolo_edge {

class ProcessingPipeline {
public:
  void add(ProcessorPtr processor) {
    processors_.push_back(std::move(processor));
  }

  bool execute(ProcessingContext &ctx) {
    return execute_range(ctx, 0, processors_.size());
  }

  bool execute_range(ProcessingContext &ctx, size_t from, size_t to) {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();
    const size_t begin = std::min(from, processors_.size());
    const size_t to_idx = std::min(to, processors_.size());

    for (size_t i = begin; i < to_idx; ++i) {
      if (!processors_[i]->process(ctx)) {
        fprintf(stderr, "[ERROR] Processor '%s' failed\n", processors_[i]->name().c_str());
        return false;
      }
    }

    if (begin == 0 && to_idx == processors_.size()) {
      auto end = Clock::now();
      ctx.total_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
    }
    return true;
  }

  int find_index(const std::string &proc_name) const {
    for (size_t i = 0; i < processors_.size(); ++i) {
      if (processors_[i]->name() == proc_name)
        return static_cast<int>(i);
    }
    return -1;
  }

  void clear() { processors_.clear(); }
  size_t size() const { return processors_.size(); }
  bool empty() const { return processors_.empty(); }

private:
  std::vector<ProcessorPtr> processors_;
};

} // namespace yolo_edge
