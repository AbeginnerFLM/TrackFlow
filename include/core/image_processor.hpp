#pragma once

#include "processing_context.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace yolo_edge {

using json = nlohmann::json;

/**
 * 图像处理器基类 (策略模式)
 * 所有处理器都继承自此类
 */
class ImageProcessor {
public:
  virtual ~ImageProcessor() = default;

  /**
   * 核心处理方法
   * @param ctx 处理上下文
   * @return true=成功, false=失败(中断Pipeline)
   */
  virtual bool process(ProcessingContext &ctx) = 0;

  /**
   * 处理器名称 (用于日志和调试)
   */
  virtual std::string name() const = 0;

  /**
   * 从JSON配置初始化
   * 子类可覆盖以接收特定配置
   */
  virtual void configure(const json &config) {
    (void)config; // 默认忽略配置
  }

  /**
   * 处理器是否有状态
   * 有状态处理器不能跨Session共享 (如Tracker)
   */
  virtual bool is_stateful() const { return false; }
};

// 处理器智能指针类型
using ProcessorPtr = std::unique_ptr<ImageProcessor>;

} // namespace yolo_edge
