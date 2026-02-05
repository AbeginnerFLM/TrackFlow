#pragma once

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace yolo_edge {

using json = nlohmann::json;

/**
 * 配置管理器
 * 支持YAML/JSON格式配置文件
 */
class Config {
public:
  /**
   * 从文件加载配置
   */
  static Config load(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open config file: " + path);
    }

    Config config;

    // 简单支持YAML (基础键值对)
    // 对于复杂配置建议使用JSON
    if (path.ends_with(".yaml") || path.ends_with(".yml")) {
      config.data_ = parse_simple_yaml(file);
    } else {
      file >> config.data_;
    }

    fprintf(stderr, "[INFO] Loaded config from: %s\n", path.c_str());
    return config;
  }

  /**
   * 从JSON对象创建
   */
  static Config from_json(const json &data) {
    Config config;
    config.data_ = data;
    return config;
  }

  /**
   * 获取值
   */
  template <typename T> T get(const std::string &key) const {
    return data_.at(key).get<T>();
  }

  /**
   * 获取值 (带默认值)
   */
  template <typename T>
  T get(const std::string &key, const T &default_val) const {
    if (data_.contains(key)) {
      return data_[key].get<T>();
    }
    return default_val;
  }

  /**
   * 获取嵌套值 (使用路径如 "server.port")
   */
  template <typename T>
  T get_nested(const std::string &path, const T &default_val) const {
    const json *current = &data_;

    size_t start = 0;
    size_t dot_pos;

    while ((dot_pos = path.find('.', start)) != std::string::npos) {
      std::string key = path.substr(start, dot_pos - start);
      if (!current->contains(key)) {
        return default_val;
      }
      current = &(*current)[key];
      start = dot_pos + 1;
    }

    std::string last_key = path.substr(start);
    if (!current->contains(last_key)) {
      return default_val;
    }

    return (*current)[last_key].get<T>();
  }

  /**
   * 获取子配置
   */
  json get_section(const std::string &key) const {
    if (data_.contains(key)) {
      return data_[key];
    }
    return json::object();
  }

  /**
   * 检查是否包含某个键
   */
  bool contains(const std::string &key) const { return data_.contains(key); }

  /**
   * 获取原始JSON数据
   */
  const json &data() const { return data_; }

private:
  json data_;

  /**
   * 简单YAML解析 (仅支持基础格式)
   */
  static json parse_simple_yaml(std::ifstream &file) {
    json result = json::object();
    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
      // 跳过空行和注释
      if (line.empty() || line[0] == '#') {
        continue;
      }

      // 去除行尾空格
      while (!line.empty() && (line.back() == ' ' || line.back() == '\r')) {
        line.pop_back();
      }

      // 检查缩进
      size_t indent = 0;
      while (indent < line.size() && line[indent] == ' ') {
        ++indent;
      }

      if (indent == line.size())
        continue;

      std::string content = line.substr(indent);

      // 解析键值对
      size_t colon_pos = content.find(':');
      if (colon_pos == std::string::npos)
        continue;

      std::string key = content.substr(0, colon_pos);
      std::string value = "";

      if (colon_pos + 1 < content.size()) {
        value = content.substr(colon_pos + 1);
        // 去除前导空格
        size_t val_start = 0;
        while (val_start < value.size() && value[val_start] == ' ') {
          ++val_start;
        }
        value = value.substr(val_start);
      }

      // 处理值类型
      json parsed_value;
      if (value.empty()) {
        // 这是一个section
        current_section = key;
        result[key] = json::object();
        continue;
      } else if (value == "true") {
        parsed_value = true;
      } else if (value == "false") {
        parsed_value = false;
      } else if (value.front() == '"' && value.back() == '"') {
        parsed_value = value.substr(1, value.size() - 2);
      } else {
        // 尝试解析为数字
        try {
          if (value.find('.') != std::string::npos) {
            parsed_value = std::stod(value);
          } else {
            parsed_value = std::stoi(value);
          }
        } catch (...) {
          parsed_value = value;
        }
      }

      // 根据缩进决定放在哪里
      if (indent > 0 && !current_section.empty()) {
        result[current_section][key] = parsed_value;
      } else {
        result[key] = parsed_value;
        current_section = "";
      }
    }

    return result;
  }
};

} // namespace yolo_edge
