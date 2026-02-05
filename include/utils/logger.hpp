#pragma once

/**
 * 简易日志库 - 替代 spdlog
 * 特点：无模板元编程，避免 WSL 下 GCC 编译崩溃
 */

#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace yolo_edge {

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
public:
  static Logger &instance() {
    static Logger inst;
    return inst;
  }

  void set_level(LogLevel level) { level_ = level; }
  void set_level(const std::string &level_str) {
    if (level_str == "debug")
      level_ = LogLevel::DEBUG;
    else if (level_str == "info")
      level_ = LogLevel::INFO;
    else if (level_str == "warn")
      level_ = LogLevel::WARN;
    else if (level_str == "error")
      level_ = LogLevel::ERROR;
  }

  LogLevel level() const { return level_; }

  void log(LogLevel level, const char *file, int line, const char *fmt, ...) {
    if (level < level_)
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    // 时间戳
    std::time_t now = std::time(nullptr);
    char time_buf[20];
    std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&now));

    // 级别字符串
    const char *level_str = "";
    switch (level) {
    case LogLevel::DEBUG:
      level_str = "DEBUG";
      break;
    case LogLevel::INFO:
      level_str = "INFO";
      break;
    case LogLevel::WARN:
      level_str = "WARN";
      break;
    case LogLevel::ERROR:
      level_str = "ERROR";
      break;
    }

    // 提取文件名（去掉路径）
    const char *filename = file;
    for (const char *p = file; *p; ++p) {
      if (*p == '/' || *p == '\\')
        filename = p + 1;
    }

    // 输出格式: [HH:MM:SS] [LEVEL] [file:line] message
    fprintf(stderr, "[%s] [%s] [%s:%d] ", time_buf, level_str, filename, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
  }

private:
  Logger() = default;
  LogLevel level_ = LogLevel::INFO;
  std::mutex mutex_;
};

} // namespace yolo_edge

// 便捷宏
#define LOG_DEBUG(fmt, ...)                                                    \
  yolo_edge::Logger::instance().log(yolo_edge::LogLevel::DEBUG, __FILE__,      \
                                    __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)                                                     \
  yolo_edge::Logger::instance().log(yolo_edge::LogLevel::INFO, __FILE__,       \
                                    __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)                                                     \
  yolo_edge::Logger::instance().log(yolo_edge::LogLevel::WARN, __FILE__,       \
                                    __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...)                                                    \
  yolo_edge::Logger::instance().log(yolo_edge::LogLevel::ERROR, __FILE__,      \
                                    __LINE__, fmt, ##__VA_ARGS__)
