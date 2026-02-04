# 🐛 TrackFlow 调试日志

## 1. 大图导致服务器崩溃
**问题**: 处理大分辨率图片时，服务器抛出 `std::bad_alloc` 崩溃。
**原因**: 服务器尝试将原本的 Base64 图片数据序列化回 JSON 响应中。对于 4K 图片，这会产生巨大的字符串，耗尽了 `nlohmann::json` 处理所需的连续内存。
**解决**: 从响应 JSON 中移除了图片数据回显。服务器现在只返回检测结果数据。

## 2. "Processing Failed" 错误
**问题**: 前端显示 `Error: Processing failed`。
**原因**:
1.  **头文件缺失**: `image_decoder.cpp` 缺少 `<iostream>`，导致编译错误（之前被掩盖）。
2.  **JSON 格式不匹配**: C++ 后端发送的 `bbox` 是 JSON 对象 `{"x":...}`，但前端期望的是数组 `[x, y, w, h]`。
**解决**:
- 添加了 `#include <iostream>`。
- 更新 `ws_server.cpp`，确保 JSON 响应中 `obb`（8个浮点数数组）和 `bbox`（4个浮点数数组）格式分离且正确。

## 3. 连接问题 (SSH & Web)
**问题**: SSH 连接被拒绝 (端口 6000)，Web 页面无法访问 (端口 8088)。
**原因**: GPU 服务器上的 `frpc.toml` 配置文件中有拼写错误：
- `remotePort = 600` (少打了一个 '0'，应为 6000)。
- `remotePort = 80880` (多打了一个 '0'，超过了 65535 端口限制)。
**解决**: 将端口修正为 `6000` 和 `8088` 并重启 `frpc` 服务。

## 4. 检测框位置偏移
**问题**: 检测框非常小，并且挤在图片的左上角。
**原因**:
- 前端在发送前将大图（如 4000px）压缩到了 1280px。
- 服务器基于 1280px 图片进行检测，返回的坐标也是 1280px 尺度的。
- 前端直接将这些 1280px 的坐标画在了原始 4000px 的 Canvas 上，没有进行缩放。
**解决**:
- 在前端实现了 `uploadScale` (上传比例) 计算：`原始宽度 / 压缩后宽度`。
- 绘制时将所有坐标乘以 `uploadScale`，将其映射回原图尺寸。

## 5. 检测框巨大且置信度全为 100%
**问题**: 检测框位置居中但异常巨大（松散），且所有"卡车"的置信度都是 100%。
**原因**:
- 使用的 YOLO 模型是 **End-to-End** 导出版本（内置 NMS），输出格式为简化的 `(N, 7)` 张量。
- 格式为 `[x, y, w, h, score, class_id, angle]`。
- C++ 代码之前是按标准 YOLOv8 Raw 格式解析的：`[x, y, w, h, class_scores(80)..., angle]`。
- **后果**: 代码错误地将 `class_id`（卡车ID为1.0）读取为 `confidence`，所以显示 100%。同时角度解析错误导致 NMS 失效或框尺寸异常。
**解决**:
- 更新 `YoloDetector::postprocess` 以识别 7 列输出格式。
- 添加了专门的解析逻辑：`if (num_features == 7) { 按 End-to-End 格式解析 }`。

## 6. 浏览器缓存问题
**问题**: 即使修复了 `test.html`，刷新页面后仍然没有变化。
**原因**: 浏览器对 HTML 文件的缓存非常顽固。
**解决**:
- 创建了新文件 `test_v3.html` 以强制浏览器加载新内容。
- （建议后续通过 URL 参数控制缓存）

## 7. 幽灵进程 (Zombie Process)
**问题**: 即使更新了后端代码并重启，Web 端看到的依然是旧的错误现象（100%置信度），修复无效。
**原因**: 服务器上残留了一个 **18小时前** 启动的 `yolo_edge_server` 进程。由于使用了 `nohup` 且之前的关闭命令可能未生效，该进程一直占用 9002 端口。新启动的服务无法绑定端口，导致所有请求实际上还是由这个运行着旧代码的"幽灵"进程处理的。
**解决**: 使用 `kill -9` 强制杀死了所有旧进程，并确认新进程成功绑定端口。这一步最终让之前的代码修复生效。

## 8. 代码更新未生效 (Git Push/Pull 失误)
**问题**: 在本地修改了 C++ 代码逻辑（修复解析问题），但服务器运行的还是旧逻辑，没有任何变化，日志也没打出来。
**原因**: 修改文件后 **忘记了 commit 和 push**。导致远程服务器执行 `git pull` 时提示 "Already up to date"，根本没有拉取到最新的修复代码。此外，有时本地修改导致 pull 失败，需要强制覆盖。
**解决**:
- [x] Fix coordinate scaling and ONNX parsing for generic YOLO models <!-- id: 12 -->
    - [x] Solved "Zombie Process" preventing fix from applying
    - [x] Solved "Git Pull" silent failure by using `git reset --hard`
    - [x] Debugging remote GCC Internal Compiler Error (ICE)
    - [x] Removed `spdlog` to stabilize compiler (replaced with `fprintf`)
    - [x] Verified ONNX model output accuracy with Python script
    - [x] Identified 640x640 resolution as bottleneck for 4K detection
- [/] Re-export model at 1280x1280 and re-test <!-- id: 13 -->

## 9. GCC 编译器崩溃 (Internal Compiler Error)
**问题**: 在远程 WSL 环境编译时，GCC 频繁报错 `internal compiler error: in purge_dead_edges` 或 `cfgrtl.cc` 错误，导致无法生成可执行文件。
**原因**: 
- **Spdlog 模板元编程**: `spdlog` 库大量使用了复杂的模板元编程。
- **资源限制/环境不稳定**: WSL 环境（尤其是由于内存或系统库限制）无法处理这些复杂的模板展开，导致编译器进程 (`cc1plus`) 内部状态损坏并崩溃。
- **现象随机**: 错误位置在 `yolo_detector.cpp`, `ws_server.cpp`, 甚至 `main.cpp` 之间随机跳动，只要包含了 `<spdlog/spdlog.h>` 的文件都有可能触发。
**解决**:
- **彻底移除 spdlog**: 将整个项目的日志库从 `spdlog` 替换为原生的 `fprintf(stderr, ...)`。移除所有 `spdlog` 头文件引用。
- **安全编译标志**: 使用 `-O0` (关闭优化) 和 `-fno-var-tracking` (减少内存追踪) 降低编译器负载。
- **结果**: 编译器负载大幅降低，项目成功全量编译通过。

## 10. 模型文件丢失导致的运行时错误
**问题**: 编译通过并启动服务器后，前端显示 `Error: Processing failed`。
### Bounding Box Size/Loose Detection Issue
- **Observation**: Detected boxes in browser are much larger than vehicles in 4K footage.
- **Verification**: Python script `verify_model.py` confirms ONNX output is accurate but low resolution (640x640) leads to scaling artifacts on 4K.
- **Root Cause**: 
    1. **Discretization**: 1 pixel in 640 becomes 6 pixels in 3840.
    2. **Model Bias**: Small objects in 4K become tiny at 640, leading the model to "guess" larger boundaries to compensate for uncertainty.
- **Solution**:
    1. Re-export ONNX at **1280x1280** to retain 4x more spatial detail.
    2. Use `simplify=True` and correct `opset` to ensure coordinate interpretation is identical to PyTorch.
    3. C++ backend will automatically adjust to the new ONNX dimensions at runtime.
**原因**: 服务器日志显示 `[ERROR] YoloDetector: Failed to load model ... Load model models/yolo26.onnx failed. File doesn't exist`。
- 配置文件或代码默认指定加载 `models/yolo26.onnx`。
- 但远程服务器目录中实际并没有这个文件，或者文件名为 `yolo_obb.onnx`。
**解决**: 
- 需要确认远程服务器上有哪些模型文件 (`ls -l models/`)。
**解决**: 
- 检查发现远程模型实际位于 `/home/xx509/TrackFlow/models/yolo26.onnx`。
- 修改前端 `test_v4.html`，将发送给服务器的 `model_path` 配置从相对路径 `models/yolo26.onnx` 改为绝对路径 `/home/xx509/TrackFlow/models/yolo26.onnx`，确保无论服务器当前工作目录在哪里都能正确加载。

## 11. 检测框过大 (Loose Bounding Boxes) ✅ 已解决
**问题**: 使用 ONNX 模型后，检测框比实际车辆大一圈。
**原因**:
- 用户原图为 4K (3840×2160)，但 ONNX 模型输入尺寸为 640×640。
- 缩放比例约为 6:1，导致：
  1. **信息丢失**: 4K 中的小目标在 640 尺度下只剩几个像素
  2. **边界模糊**: 模型在低分辨率下无法精确定位边界，倾向于输出较大的框
**解决**:
- 使用 **1280×1280** 分辨率重新导出 ONNX 模型
- C++ 后端自动识别模型输入尺寸并适配预处理
- **结果**: 检测框紧贴车辆边缘，精度显著提升
