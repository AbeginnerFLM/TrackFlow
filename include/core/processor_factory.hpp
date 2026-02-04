#pragma once

#include "processing_pipeline.hpp"
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace yolo_edge {

/**
 * 处理器工厂 (工厂模式 + 单例模式)
 * 负责注册和创建处理器
 */
class ProcessorFactory {
public:
  // 处理器创建函数类型
  using Creator = std::function<ProcessorPtr()>;

  /**
   * 获取工厂单例
   */
  static ProcessorFactory &instance() {
    static ProcessorFactory factory;
    return factory;
  }

  /**
   * 注册处理器类型
   * @param type 类型名称 (如 "decoder", "yolo")
   * @param creator 创建函数
   */
  void register_processor(const std::string &type, Creator creator) {
    // fprintf(stderr, "DEBUG: Registering processor: %s\n", type.c_str());
    creators_[type] = std::move(creator);
  }

  /**
   * 创建单个处理器
   * @param type 类型名称
   * @param config JSON配置
   * @return 处理器实例
   */
  ProcessorPtr create(const std::string &type, const json &config = {}) {
    auto it = creators_.find(type);
    if (it == creators_.end()) {
      throw std::runtime_error("Unknown processor type: " + type);
    }

    auto processor = it->second();
    processor->configure(config);
    return processor;
  }

  /**
   * 根据JSON配置创建完整Pipeline
   * JSON格式: {"pipeline": [{"type":"decoder"}, {"type":"yolo", ...}]}
   */
  ProcessingPipeline create_pipeline(const json &config) {
    ProcessingPipeline pipeline;

    // 从config中获取pipeline数组
    json pipeline_config;

    if (config.contains("pipeline")) {
      // 如果是完整配置对象
      pipeline_config = config["pipeline"];
    } else if (config.is_array()) {
      // 如果直接是数组
      pipeline_config = config;
    } else {
      throw std::runtime_error(
          "Config must contain 'pipeline' array or be an array");
    }

    // 获取各处理器的个性化配置
    json processor_configs = config.value("config", json::object());

    for (const auto &proc_cfg : pipeline_config) {
      std::string type;

      // 支持两种格式: 字符串或对象
      if (proc_cfg.is_string()) {
        type = proc_cfg.get<std::string>();
      } else if (proc_cfg.is_object()) {
        type = proc_cfg.at("type").get<std::string>();
      } else {
        throw std::runtime_error(
            "Pipeline item must be string or object with 'type'");
      }

      // 合并配置: proc_cfg中的配置优先于processor_configs
      json merged_config = processor_configs.value(type, json::object());
      if (proc_cfg.is_object()) {
        merged_config.merge_patch(proc_cfg);
      }

      pipeline.add(create(type, merged_config));
      // fprintf(stderr, "DEBUG: Added processor '%s' to pipeline\n",
      // type.c_str());
    }

    // fprintf(stderr, "DEBUG: Created pipeline with %zu processors\n",
    // pipeline.size());
    return pipeline;
  }

  /**
   * 检查处理器类型是否已注册
   */
  bool has(const std::string &type) const {
    return creators_.find(type) != creators_.end();
  }

  /**
   * 获取所有已注册的处理器类型
   */
  std::vector<std::string> registered_types() const {
    std::vector<std::string> types;
    types.reserve(creators_.size());
    for (const auto &[type, _] : creators_) {
      types.push_back(type);
    }
    return types;
  }

private:
  ProcessorFactory() = default;
  ProcessorFactory(const ProcessorFactory &) = delete;
  ProcessorFactory &operator=(const ProcessorFactory &) = delete;

  std::unordered_map<std::string, Creator> creators_;
};

/**
 * 处理器自动注册辅助类
 */
template <typename T> class ProcessorRegistrar {
public:
  explicit ProcessorRegistrar(const std::string &type) {
    ProcessorFactory::instance().register_processor(
        type, []() { return std::make_unique<T>(); });
  }
};

/**
 * 处理器注册宏
 * 用法: REGISTER_PROCESSOR("decoder", ImageDecoder)
 */
#define REGISTER_PROCESSOR(type_name, class_name)                              \
  static ::yolo_edge::ProcessorRegistrar<class_name> class_name##_registrar(   \
      type_name)

} // namespace yolo_edge
