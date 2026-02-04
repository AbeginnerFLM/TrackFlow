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
#include <cstdio>
#include <iostream>

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    fprintf(stderr, "[INFO] Received signal %d, shutting down...\n", signal);
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

void setup_logging(bool verbose) {
  // No-op for now using fprintf
  if (verbose) {
    fprintf(stderr, "[INFO] Verbose logging enabled (approximate)\n");
  }
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  // 解析命令行参数
  Args args = parse_args(argc, argv);

  // 设置日志
  setup_logging(args.verbose);

  fprintf(stderr, "=================================\n");
  fprintf(stderr, "  TrackFlow YOLO Edge Server\n");
  fprintf(stderr, "=================================\n");

  // 加载配置文件
  if (!args.config_path.empty()) {
    try {
      auto config = yolo_edge::Config::load(args.config_path);
      args.port = config.get_nested("server.port", args.port);
      args.threads = config.get_nested("server.threads", args.threads);

      // Logging level config ignored
    } catch (const std::exception &e) {
      fprintf(stderr, "[WARN] Failed to load config: %s\n", e.what());
    }
  }

  // 显示已注册的处理器
  auto &factory = yolo_edge::ProcessorFactory::instance();
  auto types = factory.registered_types();
  fprintf(stderr, "[INFO] Registered processors: %zu\n", types.size());
  for (const auto &type : types) {
    fprintf(stderr, "  - %s\n", type.c_str());
  }

  // 设置信号处理
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // 创建线程池
  fprintf(stderr, "[INFO] Creating thread pool with %d threads\n",
          args.threads);
  yolo_edge::ThreadPool pool(args.threads);

  // 创建并启动WebSocket服务器
  fprintf(stderr, "[INFO] Starting WebSocket server on port %d\n", args.port);
  yolo_edge::WebSocketServer server(args.port, pool);

  try {
    server.run();
  } catch (const std::exception &e) {
    fprintf(stderr, "[ERROR] Server error: %s\n", e.what());
    return 1;
  }

  fprintf(stderr, "[INFO] Server stopped\n");
  return 0;
}
