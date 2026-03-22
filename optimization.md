# TrackFlow 优化方案

> 基于全项目代码审查，按优先级排列。预估收益基于典型场景（1080p图片，~50个检测目标）。

---

## 第一部分：后端优化

### 1. ~~[高优先] Batch 推理 —— 多帧合并为一个 batch 送入 GPU~~ [已完成]

> **实现文件**: `src/processors/batch_inference_engine.cpp`, `include/processors/batch_inference_engine.hpp`
> **集成点**: `src/processors/yolo_detector.cpp` 中 `infer()` 方法自动提交到共享 BatchInferenceEngine

**现状问题：**
- 当前每帧独立推理，input shape 硬编码为 `{1, 3, H, W}`（`yolo_detector.cpp:265`）
- 图片到达速度远大于推理速度，GPU 大部分时间在等待单帧推理完成
- 线程池有 8 个线程，但每个线程独立调用 `ort_->session->Run()`，无法利用 GPU 的并行能力

**优化方案：**

```
架构变化：
当前：  Client → WebSocket → ThreadPool → [Decode → Infer → Track] per frame
优化后：Client → WebSocket → DecodeQueue → BatchCollector → GPU Infer → Dispatch → Track per session
```

**具体实现：**

1. **新增 BatchInferenceEngine 类**：
   - 维护一个收集队列（bounded queue, 容量如16）
   - 独立线程以固定间隔或帧数触发 batch（如攒够 N 帧或等待 T ms）
   - 将 N 帧的预处理结果拼接为 `{N, 3, H, W}` 的 tensor
   - 一次调用 `session->Run()` 推理整个 batch
   - 将输出拆分回各帧，通过 promise/future 返回给各请求线程

2. **动态 batch 策略**：
   ```
   max_batch_size = 4~8（取决于 GPU 显存）
   max_wait_ms = 5~10ms（避免低负载时延迟过高）
   策略：min(积攒到 max_batch_size, 等待 max_wait_ms) 触发推理
   ```

3. **预处理并行化**：
   - Decode + Letterbox + Normalize 仍在各线程池线程中并行执行
   - 只有 GPU 推理合并为 batch

4. **ONNX 模型适配**：
   - 当前模型 input shape 的 batch 维度为 1，需要检查是否支持动态 batch
   - 如果模型是静态 batch=1，需用 `onnxruntime` 的动态 axes 重新导出
   - 或在导出 ONNX 时设置 `dynamic_axes={'images': {0: 'batch'}}`

**预估收益：** 吞吐量提升 2~4x（batch=4 时），单帧延迟略增 5~10ms（等待凑批），总吞吐延迟比大幅改善。

**需要注意：**
- Tracker 是有状态的（per-session），batch 推理后仍需按 session 分发回各自的 tracker
- `scale_`、`pad_x_`、`pad_y_` 是 YoloDetector 的成员变量，batch 化后每帧有不同值，需改为 per-frame 参数传递（当前如果所有帧尺寸相同则无此问题）

---

### 2. ~~[高优先] 预处理 buffer 复用 —— 消除每帧内存分配~~ [已完成]

> **实现文件**: `src/processors/yolo_detector.cpp` 中 `preprocess()` 复用 `letterbox_buf_`

**现状问题：**
- `preprocess()` 每次调用创建 `resized`、`letterbox`、`blob` 三个 cv::Mat（`yolo_detector.cpp:238-254`）
- 每帧约分配 640×640×3×4 = 4.7MB（float32 blob），高频场景下 GC 压力大

**优化方案：**
```cpp
// 在 YoloDetector 中添加预分配的 buffer
cv::Mat resized_buf_;
cv::Mat letterbox_buf_;
cv::Mat blob_buf_;

// preprocess 中复用：
cv::resize(frame, resized_buf_, cv::Size(new_w, new_h));
letterbox_buf_.setTo(cv::Scalar(114, 114, 114));
resized_buf_.copyTo(letterbox_buf_(cv::Rect(pad_x_, pad_y_, new_w, new_h)));
```

**预估收益：** 每帧减少 ~5MB 内存分配，预处理耗时降低 10~20%。

---

### 3. ~~[高优先] ONNX 推理输出零拷贝~~ [已完成]

> **实现文件**: `src/processors/yolo_detector.cpp` 中 `infer()` 返回 `InferOutput`（`vector<float>` + shape），不经过 cv::Mat 中转

**现状问题：**
- `infer()` 中对每个输出 tensor 调用了 `mat.clone()`（`yolo_detector.cpp:288,293`）
- 这是一次完整的内存拷贝，对于大输出（如 8400×7 的 float32 = 235KB）不必要

**优化方案：**
- 持有 `output_tensors` 的生命周期直到 postprocess 完成，避免 clone
- 或者直接在 postprocess 中操作 tensor data 指针，不经过 cv::Mat 中转

```cpp
// 返回 output_tensors 本身而非 cv::Mat
std::vector<Ort::Value> infer(const cv::Mat& blob);
// postprocess 直接操作 float* data
```

**预估收益：** 推理后处理阶段减少 ~0.5ms。

---

### 4. ~~[中优先] 旋转 NMS 优化 —— 从 O(N²) 降低复杂度~~ [已完成: 方案 A + B]

> **实现文件**: `src/processors/yolo_detector.cpp` 中 `rotated_nms()` 使用 AABB 预过滤；E2E 模型 (7列输出) 完全跳过 NMS

**现状问题：**
- `rotated_nms()` 是 O(N²) 的嵌套循环（`yolo_detector.cpp:528-543`）
- 每对 box 都调用 `cv::rotatedRectangleIntersection()` + `cv::convexHull()` + `cv::contourArea()`
- 50 个 box → 2500 次 IoU 计算；100 个 box → 10000 次

**优化方案：**

**方案 A：空间预过滤**
```cpp
// 先用 AABB (axis-aligned bounding box) 快速排除不可能重叠的对
// rotatedRect.boundingRect() 的 IoU 为 0 → 旋转 IoU 必为 0
cv::Rect aabb_i = boxes[i].boundingRect();
cv::Rect aabb_j = boxes[j].boundingRect();
if ((aabb_i & aabb_j).area() == 0) continue; // 快速跳过
float iou = rotated_iou(boxes[i], boxes[j]); // 仅对可能重叠的计算
```

**方案 B：利用 End-to-End 模型内置 NMS**
- 当前模型 (`yolo26.onnx`) 输出 7 列，属于 End-to-End 格式（NMS 已内置）
- `yolo_detector.cpp:434-445` 的注释也表明了这一点
- **如果模型已做 NMS，可以完全跳过用户侧 NMS**，仅保留置信度过滤

**预估收益：** 方案 A 减少 50~80% 的 IoU 计算；方案 B 直接省掉整个 NMS 步骤（~2-5ms）。

---

### 5. ~~[中优先] 贪心匹配 → 真正的 Hungarian 算法~~ [已完成]

> **实现文件**: `src/processors/byte_tracker.cpp` 中 `linear_assignment()` 使用 Kuhn-Munkres 算法

**现状问题：**
- `linear_assignment()` 是 O(M×N×min(M,N)) 的贪心匹配（`byte_tracker.cpp:435-478`）
- 贪心匹配不是全局最优，可能导致跟踪漂移（A 匹配了 B 的最佳目标，B 被迫匹配次优）
- 每次迭代都扫描整个代价矩阵

**优化方案：**
- 实现 Kuhn-Munkres（Hungarian）算法，O(N³) 但保证最优匹配
- 或使用 `scipy` 风格的 `lapjv`（Jonker-Volgenant）算法，实际运行更快
- 对于典型场景（M, N < 50），差异在 1ms 以内，但匹配质量显著提升

**预估收益：** 跟踪准确率提升（减少 ID switch），耗时差异可忽略。

---

### 6. ~~[中优先] STrack 轨迹存储优化~~ [已完成]

> **实现文件**: `include/processors/byte_tracker.hpp` 中 `STrack` 使用 `std::array<cv::Point2f, 100>` 环形缓冲区

**现状问题：**
- `trajectory` 使用 `std::vector`，达到 100 时 `erase(begin())` 是 O(N) 操作（`byte_tracker.cpp:84-86`）
- 每帧每个 track 都可能触发

**优化方案：**
```cpp
// 改用 circular buffer（环形缓冲区）
std::array<cv::Point2f, 100> trajectory;
int traj_head = 0;
int traj_size = 0;

void add_point(cv::Point2f pt) {
    trajectory[traj_head] = pt;
    traj_head = (traj_head + 1) % 100;
    if (traj_size < 100) traj_size++;
}
```

**预估收益：** 消除 O(N) 的 erase，每 track 每帧节省 ~0.01ms（多 track 时累积可观）。

---

### 7. [中优先] Pipeline 阶段并行化 [未实现]

**现状问题：**
- Pipeline 严格串行执行：Decode → YOLO → Tracker → GeoTransform（`processing_pipeline.hpp`）
- Decode 和 GeoTransform 是 CPU 密集的，YOLO 是 GPU 密集的
- GPU 推理期间 CPU 空闲，反之亦然

**优化方案：**
- **流水线并行（Pipeline Parallelism）**：当 frame N 在 GPU 推理时，frame N+1 可以在 CPU 上 decode
- 与 batch 推理方案互补：batch 提升单步吞吐，pipeline 并行掩盖各步延迟

```
时间线（串行）：
Frame1: [Decode][Infer      ][Track][Geo]
Frame2:                                   [Decode][Infer      ][Track][Geo]

时间线（流水线）：
Frame1: [Decode][Infer      ][Track][Geo]
Frame2:         [Decode][Infer      ][Track][Geo]
Frame3:                 [Decode][Infer      ][Track][Geo]
```

**预估收益：** 稳态吞吐量提升约 30%（取决于各阶段比例）。

---

### 8. ~~[中优先] WebSocket 响应构建优化~~ [已完成: 部分]

> **实现文件**: `src/network/ws_server.cpp` 中 `build_response()` 使用 `reserve()` + `std::move()`，清理了冗余日志

**现状问题：**
- `build_response()` 为每个 detection 构建完整 JSON 对象（`ws_server.cpp:330-367`）
- `det.obb.points()` 计算 4 个顶点，`det.obb.boundingRect()` 计算 AABB —— 两者都是每帧重新计算
- `response.dump()` 序列化整个 JSON 对象为 string

**优化方案：**

1. **预计算 OBB 顶点**：在 postprocess 阶段计算一次，存入 Detection struct
2. **使用 JSON 流式写入**：避免构建完整 JSON 树再序列化
   ```cpp
   // 直接拼字符串（对于固定格式的高频响应更快）
   std::string build_response_fast(const ProcessingContext& ctx) {
       // 使用 fmt 或 snprintf 直接构建 JSON 字符串
   }
   ```
3. **可选 MessagePack 二进制协议**：替代 JSON 文本，减少序列化/反序列化开销

**预估收益：** 响应构建时间减少 30~50%（50 个检测目标时 ~0.5ms）。

---

### 9. ~~[低优先] 清理残留 Debug 日志~~ [已完成]

> **实现文件**: `src/processors/yolo_detector.cpp`, `src/processors/byte_tracker.cpp`, `src/network/ws_server.cpp` 中删除了所有 debug fprintf

**现状问题：**
- 大量被注释掉的 `fprintf`/`std::cerr` 语句散布在关键路径上（`ws_server.cpp`、`yolo_detector.cpp`）
- 部分未注释的 `[DEBUG]` 级别日志仍在运行（`yolo_detector.cpp:49,62,85`、`byte_tracker.cpp:109`）
- `fprintf(stderr, ...)` 即使没有被重定向，调用本身也有开销

**优化方案：**
- 删除所有被注释的日志行（减少代码噪音）
- 将 `[DEBUG]` 日志改为条件编译或运行时级别控制
```cpp
#ifdef NDEBUG
#define LOG_DEBUG(...) ((void)0)
#else
#define LOG_DEBUG(...) fprintf(stderr, __VA_ARGS__)
#endif
```

**预估收益：** 代码可读性提升，Release 模式下省掉每帧 ~10 次 fprintf 调用。

---

### 10. ~~[低优先] SessionManager 过期清理无定时触发~~ [已完成]

> **实现文件**: `include/core/session.hpp` 中 `get_or_create()` 每 100 次访问自动触发过期清理

**现状问题：**
- `cleanup_expired()` 方法已实现（`session.hpp:103-117`），但从未被任何地方调用
- 长时间运行后，断开连接但未触发 close 的 session 会泄漏内存

**优化方案：**
- 在 main 中添加定时器，或在 `get_or_create()` 中以一定频率触发清理
```cpp
Session& get_or_create(const std::string& session_id, const json& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 每 100 次调用清理一次
    if (++access_count_ % 100 == 0) {
        cleanup_expired_unlocked(std::chrono::seconds(300));
    }
    // ... 原逻辑
}
```

**预估收益：** 防止内存泄漏，长期运行稳定性提升。

---

### 11. ~~[低优先] ONNX Session 线程安全~~ [已完成]

> **实现文件**: `include/processors/yolo_detector.hpp` 中 `PreprocessResult` 结构体将 scale/pad 局部化

**现状问题：**
- `Ort::Session` 本身是线程安全的（ONNX Runtime 文档确认），但 `YoloDetector` 的成员变量 `scale_`、`pad_x_`、`pad_y_` 不是
- 如果同一个 YoloDetector 实例被多线程并发调用（虽然当前 pipeline_mutex 阻止了这种情况），会产生数据竞争

**优化方案：**
- 将 `scale_`、`pad_x_`、`pad_y_` 改为 `preprocess()` 的返回值或局部变量
```cpp
struct PreprocessResult {
    cv::Mat blob;
    float scale;
    int pad_x, pad_y;
};
PreprocessResult preprocess(const cv::Mat& frame);
```
- 这也是 batch 推理的前置需求（每帧有不同的 scale/pad）

**预估收益：** 为 batch 推理和更细粒度的并发打基础。

---

### 12. [低优先] 二进制消息的零拷贝传递 [未实现]

**现状问题：**
- WebSocket 收到二进制数据后，`image_data->assign(message.begin(), message.end())`（`ws_server.cpp:157`）执行了一次完整拷贝
- `message` 是 `std::string_view`，底层 buffer 在回调返回后失效，拷贝是必要的

**优化方案：**
- uWebSockets 支持通过 `.cork()` 和自定义分配器减少拷贝
- 但由于回调生命周期限制，当前的拷贝策略是正确的，优化空间有限
- 可考虑使用 memory pool 预分配 image buffer，避免每次 `new vector`

**预估收益：** 微小（~0.1ms），主要减少内存碎片。

---

## 第二部分：前端优化

### 13. ~~[高优先] 添加连续推理模式（视频流）~~ [已完成]

> **实现文件**: `test_v4.html` — 视频上传 + 逐帧抽取 + 背压控制发送

**现状问题：**
- 当前前端只支持单帧手动推理（点击按钮发送一次）
- 无法测试连续多帧场景下的 tracker 效果
- 无法测试系统在持续负载下的性能

**优化方案：**
```javascript
// 添加视频上传和逐帧发送
let isStreaming = false;
let videoElement = document.createElement('video');

async function startVideoStream(file) {
    videoElement.src = URL.createObjectURL(file);
    isStreaming = true;

    while (isStreaming) {
        // 从 video 抓取当前帧
        const frame = captureFrame(videoElement);
        await sendFrame(frame);
        // 等待响应后再发下一帧（背压控制）
        await waitForResponse();
    }
}
```

**关键设计：**
- 背压控制：收到上一帧响应后再发下一帧，防止服务端队列堆积
- 可调帧率：允许用户设置发送间隔（如 30fps → 每 33ms 发一帧）
- 帧跳过：如果推理速度跟不上，自动跳帧

---

### 14. [高优先] Canvas 渲染性能优化 [未实现]

**现状问题：**
- `drawDetections()` 每次都重绘整张图片（`test_v4.html:607-613`）
- `ctx.drawImage(img, 0, 0)` 对大图（如 4K）耗时 5~10ms
- `ctx.measureText()` 每个 detection 调用两次（`test_v4.html:638,654`）

**优化方案：**

1. **OffscreenCanvas + 双缓冲**：
```javascript
// 图片只绘制一次到 offscreen canvas
const offscreen = new OffscreenCanvas(width, height);
const offCtx = offscreen.getContext('2d');
offCtx.drawImage(img, 0, 0); // 仅在图片变化时执行

// 每帧只需：
ctx.drawImage(offscreen, 0, 0); // 从 offscreen 拷贝（更快）
// 然后绘制检测框
```

2. **文本测量缓存**：
```javascript
const textWidthCache = {};
function measureText(text) {
    if (!textWidthCache[text]) {
        textWidthCache[text] = ctx.measureText(text).width;
    }
    return textWidthCache[text];
}
```

3. **requestAnimationFrame 节流**：
```javascript
let pendingDetections = null;
function drawDetections(detections) {
    pendingDetections = detections;
    if (!rafId) rafId = requestAnimationFrame(doRender);
}
```

**预估收益：** 连续推理模式下渲染帧率提升 2~3x。

---

### 15. ~~[中优先] 图片压缩优化~~ [已完成]

> **实现文件**: `test_v4.html` 中 `sendNext()` 直接使用 `canvas.toBlob()` 发送二进制数据

**现状问题：**
- `compressImage()` 使用 Canvas `toDataURL('image/jpeg', 0.85)` 压缩（`test_v4.html:462`）
- 然后 `sendInfer()` 中又将 Base64 转回 Blob（`test_v4.html:533-535`）
- 经历了 原始 → Canvas绘制 → toDataURL(Base64) → fetch → Blob → ArrayBuffer 的多次转换

**优化方案：**
```javascript
async function sendInfer() {
    // 直接用 canvas.toBlob() 避免 Base64 中间步骤
    const blob = await new Promise(resolve => {
        tempCanvas.toBlob(resolve, 'image/jpeg', 0.85);
    });
    const arrayBuffer = await blob.arrayBuffer();

    // 发送
    ws.send(JSON.stringify(header));
    ws.send(arrayBuffer);
}
```

- 省掉了 `toDataURL()` → `fetch()` → `blob()` → `arrayBuffer()` 的链路
- `toBlob()` 直接返回二进制数据，性能更优

**预估收益：** 每帧客户端准备时间减少 50~70%。

---

### 16. [中优先] 响应时间计算修正 [未实现]

**现状问题：**
- Latency 使用 `performance.now() - inferStartTime` 计算（`test_v4.html:584`）
- 但 `inferStartTime` 在 `sendInfer()` 最开始设置（`test_v4.html:529`），包含了客户端的 Base64→Blob 转换时间
- 计算的是"客户端端到端延迟"而非"服务端推理延迟"

**优化方案：**
```javascript
// 1. 客户端端到端延迟
const e2eLatency = performance.now() - inferStartTime;

// 2. 服务端各阶段延迟（已在 response.timing 中返回）
const serverLatency = response.timing.total_ms;
const networkLatency = e2eLatency - serverLatency;

// 3. 分别显示
document.getElementById('statLatency').textContent = serverLatency.toFixed(0);
document.getElementById('statNetwork').textContent = networkLatency.toFixed(0);
```

**预估收益：** 更准确的性能分析，帮助定位瓶颈。

---

### 17. [中优先] Detection 信息展示增强 [未实现]

**现状问题：**
- 检测信息列表每次使用 `innerHTML` 重建整个 DOM（`test_v4.html:718`）
- 无法高亮/选择单个检测目标
- 没有显示地理坐标信息

**优化方案：**
- 使用 DocumentFragment 或虚拟 DOM 差量更新
- 添加鼠标悬停高亮：hover 某个检测条目时，在 canvas 上高亮对应的 box
- 显示完整信息：track_id、geo 坐标、角度

---

### 18. [低优先] WebSocket 重连机制 [未实现]

**现状问题：**
- 断线后需要手动点击 Connect 重连
- 没有心跳检测（虽然有 ping/pong 按钮，但不是自动的）

**优化方案：**
```javascript
let reconnectAttempts = 0;
const MAX_RECONNECT = 5;

ws.onclose = () => {
    if (reconnectAttempts < MAX_RECONNECT) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
        setTimeout(() => {
            reconnectAttempts++;
            connect();
        }, delay);
    }
};

// 心跳
setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);
```

---

## 第三部分：优化实施路线图

### Phase 1：快速见效（1~2天）
| # | 优化项 | 预估收益 | 状态 |
|---|--------|----------|------|
| 2 | 预处理 buffer 复用 | 每帧 -1ms | **已完成** |
| 3 | ONNX 输出零拷贝 | 每帧 -0.5ms | **已完成** |
| 4B | 跳过 End-to-End 模型的 NMS | 每帧 -2~5ms | **已完成** |
| 9 | 清理 Debug 日志 | 代码整洁 | **已完成** |
| 15 | 前端 toBlob 优化 | 客户端 -50% 准备时间 | **已完成** |

### Phase 2：核心架构（3~5天）
| # | 优化项 | 预估收益 | 状态 |
|---|--------|----------|------|
| 1 | Batch 推理 | 吞吐量 2~4x | **已完成** |
| 11 | preprocess 参数局部化 | batch 前置需求 | **已完成** |
| 7 | Pipeline 流水线并行 | 吞吐量 +30% | 未实现 |
| 13 | 前端视频流模式 | 功能完善 | **已完成** |

### Phase 3：精细打磨（按需）
| # | 优化项 | 预估收益 | 状态 |
|---|--------|----------|------|
| 5 | Hungarian 算法 | 跟踪质量提升 | **已完成** |
| 6 | STrack 环形缓冲 | 微小性能提升 | **已完成** |
| 8 | JSON 响应优化 | 每帧 -0.5ms | **部分完成** |
| 10 | Session 过期清理 | 稳定性 | **已完成** |
| 14 | Canvas 双缓冲 | 前端帧率提升 | 未实现 |

**总计: 18 项中 13 项已完成, 1 项部分完成, 4 项未实现**

---

## 附录：Batch 推理架构设计草图

```
                    ┌─────────────────────────────────┐
                    │        WebSocket Server          │
                    └─────────┬───────────────────────┘
                              │ 收到图片
                              ▼
                    ┌─────────────────────┐
                    │    ThreadPool       │
                    │  (Decode并行)       │
                    │  frame1: decode()   │
                    │  frame2: decode()   │
                    │  frame3: decode()   │
                    └─────────┬───────────┘
                              │ preprocessed blob
                              ▼
                    ┌─────────────────────┐
                    │  BatchCollector     │
                    │  queue: [f1,f2,f3]  │
                    │  trigger: N帧/Tms  │
                    └─────────┬───────────┘
                              │ batched tensor {N,3,640,640}
                              ▼
                    ┌─────────────────────┐
                    │  GPU Inference      │
                    │  单次 session.Run() │
                    │  output: {N,...}    │
                    └─────────┬───────────┘
                              │ split outputs
                              ▼
                    ┌─────────────────────┐
                    │  Dispatch + Track   │
                    │  frame1 → session1  │
                    │  frame2 → session2  │
                    │  frame3 → session1  │
                    └─────────────────────┘
```
