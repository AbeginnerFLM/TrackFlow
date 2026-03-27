#pragma once

#include "processing_pipeline.hpp"
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolo_edge {

class ProcessorFactory {
public:
  using Creator = std::function<ProcessorPtr()>;

  static ProcessorFactory &instance() {
    static ProcessorFactory factory;
    return factory;
  }

  void register_processor(const std::string &type, Creator creator) {
    creators_[type] = std::move(creator);
  }

  ProcessorPtr create(const std::string &type, const json &config = {}) {
    auto it = creators_.find(type);
    if (it == creators_.end()) {
      throw std::runtime_error("Unknown processor type: " + type);
    }

    auto processor = it->second();
    processor->configure(config);
    return processor;
  }

  ProcessingPipeline create_pipeline(const json &config) {
    ProcessingPipeline pipeline;

    json pipeline_config;

    if (config.contains("pipeline")) {
      pipeline_config = config["pipeline"];
    } else if (config.is_array()) {
      pipeline_config = config;
    } else {
      throw std::runtime_error(
          "Config must contain 'pipeline' array or be an array");
    }

    json processor_configs = config.value("config", json::object());

    for (const auto &proc_cfg : pipeline_config) {
      std::string type;

      if (proc_cfg.is_string()) {
        type = proc_cfg.get<std::string>();
      } else if (proc_cfg.is_object()) {
        type = proc_cfg.at("type").get<std::string>();
      } else {
        throw std::runtime_error(
            "Pipeline item must be string or object with 'type'");
      }

      // 允许全局配置+节点级配置并存，节点级字段覆盖全局默认值。
      json merged_config = processor_configs.value(type, json::object());
      if (proc_cfg.is_object()) {
        merged_config.merge_patch(proc_cfg);
      }

      pipeline.add(create(type, merged_config));
    }
    return pipeline;
  }

  bool has(const std::string &type) const {
    return creators_.find(type) != creators_.end();
  }

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

template <typename T> class ProcessorRegistrar {
public:
  explicit ProcessorRegistrar(const std::string &type) {
    ProcessorFactory::instance().register_processor(
        type, []() { return std::make_unique<T>(); });
  }
};

#define REGISTER_PROCESSOR(type_name, class_name)                              \
  static ::yolo_edge::ProcessorRegistrar<class_name> class_name##_registrar(   \
      type_name)

} // namespace yolo_edge
