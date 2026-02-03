/**
 * TrackFlow YOLO Edge Server
 *
 * 分布式视频分析系统 - 支持YOLO推理、目标跟踪和地理坐标转换
 *
 * 用法:
 *   ./yolo_edge_server [选项]
 *
 * 选项:
 *   -p, --port PORT      监听端口 (默认: 9001)
 *   -t, --threads NUM    线程池大小 (默认: 2)
 *   -c, --config PATH    配置文件路径
 *   -v, --verbose        详细日志
 *   -h, --help           显示帮助
 */

#include "core/processor_factory.hpp"
#include "network/ws_server.hpp"
#include "utils/config.hpp"
#include "utils/thread_pool.hpp"

// 包含所有处理器以确保注册
#include "processors/byte_tracker.hpp"
#include "processors/geo_transformer.hpp"
#include "processors/image_decoder.hpp"
#include "processors/undistort_processor.hpp"
#include "processors/yolo_detector.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    spdlog::info("Received signal {}, shutting down...", signal);
    g_running = false;
  }
}

void print_usage(const char *program) {
  std::cout << "Usage: " << program << " [OPTIONS]\n\n"
            << "Options:\n"
            << "  -p, --port PORT      Listen port (default: 9001)\n"
            << "  -t, --threads NUM    Thread pool size (default: 2)\n"
            << "  -c, --config PATH    Config file path\n"
            << "  -v, --verbose        Enable debug logging\n"
            << "  -h, --help           Show this help\n"
            << std::endl;
}

struct Args {
  int port = 9001;
  int threads = 2;
  std::string config_path;
  bool verbose = false;
};

Args parse_args(int argc, char *argv[]) {
  Args args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "-v" || arg == "--verbose") {
      args.verbose = true;
    } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
      args.port = std::stoi(argv[++i]);
    } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
      args.threads = std::stoi(argv[++i]);
    } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
      args.config_path = argv[++i];
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      print_usage(argv[0]);
      std::exit(1);
    }
  }

  return args;
}

#include <spdlog/sinks/basic_file_sink.h>

void setup_logging(bool verbose) {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "server_debug_fixed.log", true);

  std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
  auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(),
                                                 sinks.end());

  spdlog::set_default_logger(logger);

  if (verbose) {
    spdlog::set_level(spdlog::level::debug);
    spdlog::default_logger()->flush_on(spdlog::level::debug);
  } else {
    spdlog::set_level(spdlog::level::info);
    spdlog::default_logger()->flush_on(spdlog::level::info);
  }

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  // 解析命令行参数
  Args args = parse_args(argc, argv);

  // 设置日志
  setup_logging(args.verbose);

  spdlog::info("=================================");
  spdlog::info("  TrackFlow YOLO Edge Server");
  spdlog::info("=================================");

  // 加载配置文件
  if (!args.config_path.empty()) {
    try {
      auto config = yolo_edge::Config::load(args.config_path);
      args.port = config.get_nested("server.port", args.port);
      args.threads = config.get_nested("server.threads", args.threads);

      std::string log_level =
          config.get_nested("logging.level", std::string("info"));
      if (log_level == "debug") {
        spdlog::set_level(spdlog::level::debug);
      } else if (log_level == "warn") {
        spdlog::set_level(spdlog::level::warn);
      } else if (log_level == "error") {
        spdlog::set_level(spdlog::level::err);
      }
    } catch (const std::exception &e) {
      spdlog::warn("Failed to load config: {}", e.what());
    }
  }

  // 显示已注册的处理器
  auto &factory = yolo_edge::ProcessorFactory::instance();
  auto types = factory.registered_types();
  spdlog::info("Registered processors: {}", types.size());
  for (const auto &type : types) {
    spdlog::debug("  - {}", type);
  }

  // 设置信号处理
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // 创建线程池
  spdlog::info("Creating thread pool with {} threads", args.threads);
  yolo_edge::ThreadPool pool(args.threads);

  // 创建并启动WebSocket服务器
  spdlog::info("Starting WebSocket server on port {}", args.port);
  yolo_edge::WebSocketServer server(args.port, pool);

  try {
    server.run();
  } catch (const std::exception &e) {
    spdlog::error("Server error: {}", e.what());
    return 1;
  }

  spdlog::info("Server stopped");
  return 0;
}
