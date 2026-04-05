# TrackFlow 优化方案

> 基于全项目代码审查，按优先级整理，并随着代码推进持续更新状态。

---

## 第一部分：后端优化

### 1. ~~[高优先] Batch 推理 —— 多帧合并为一个 batch 送入 GPU~~ [已完成]
- 已实现共享 `BatchInferenceEngine`
- 已接入 YOLO 推理路径
- 现已进一步接入配置：`ort_threads`、`batch_size`、`batch_wait_ms`、`batch_max_pending`

### 2. ~~[高优先] 预处理 buffer 复用 —— 消除每帧内存分配~~ [已完成]
- `yolo_detector.cpp` 已使用 `thread_local` buffer 复用预处理内存

### 3. ~~[高优先] ONNX 推理输出零拷贝~~ [部分完成]
- 已去掉早期 `cv::Mat clone()` 中转
- 当前仍会把 ORT 输出复制到 `std::vector<float>`，所以不是严格意义上的零拷贝

### 4. ~~[中优先] 旋转 NMS 优化 —— 从 O(N²) 降低复杂度~~ [已完成]
- 已加入 AABB 预过滤
- End-to-End 7 列输出模型已跳过额外 NMS

### 5. ~~[中优先] 贪心匹配 → 真正的 Hungarian 算法~~ [已完成]
- ByteTrack 已切换到 Kuhn-Munkres / Hungarian 风格匹配实现

### 6. ~~[中优先] STrack 轨迹存储优化~~ [已完成]
- 已改为环形缓冲风格，避免频繁 `erase(begin())`

### 7. [中优先] Pipeline 阶段并行化 [未实现]
- 当前仍是 “tracker 前并行、tracker 及之后按帧序串行” 的折中方案
- 这已经保证了跟踪稳定性，但不是完整的流水线并行

### 8. ~~[中优先] WebSocket 响应构建优化~~ [进一步完成]
- 早期已做 `reserve()` / `move()` 优化
- 本轮进一步把 detection 的 `obb_points` / `bbox` 预计算并缓存到 `Detection`，减少响应时重复计算
- JSON 协议仍保留，二进制响应协议尚未引入

### 9. ~~[低优先] 清理残留 Debug 日志~~ [已完成]
- 已完成主要日志收敛

### 10. ~~[低优先] SessionManager 过期清理无定时触发~~ [已完成]
- 已在 `get_or_create()` 和服务消息路径中周期触发清理

### 11. ~~[低优先] ONNX Session 线程安全~~ [已完成]
- `scale / pad` 已从共享成员状态转为局部化路径

### 12. [低优先] 二进制消息的零拷贝传递 [未实现]
- 由于 uWebSockets 回调生命周期限制，当前仍保留必要拷贝

### 13. ~~[高优先] 服务端配置接入与配置权威模型~~ [已完成]
- 已将 `config/config.yaml` 中的默认值、features、limits 接入服务端启动配置
- 已实现“后端默认值 + 前端允许覆盖项”的合并模式
- pipeline 由服务端依据 feature flags 生成，不再信任任意前端顺序

### 14. ~~[高优先] WebSocket 协议健壮性~~ [已完成]
- `infer_header` 与 binary 现在有显式配对状态
- duplicate header、unexpected binary、payload 超限、invalid request 都会返回结构化错误
- 不再静默丢帧

### 15. ~~[高优先] 后端背压与限流~~ [已完成]
- `ThreadPool` 已支持有界排队
- `BatchInferenceEngine` 已支持有界排队
- 每个 session 也限制了同时 outstanding 请求数

### 16. ~~[高优先] Session 生命周期与 reset 健壮性~~ [已完成]
- `SessionManager` 现在返回 `shared_ptr<Session>`
- worker 全程持有强引用
- reset/过期清理改为 retire 语义，避免直接破坏仍在执行中的 worker 所引用对象
- 当同一 `session_id` 的配置变化时，会自动重建 session pipeline

---

## 第二部分：前端优化

### 17. ~~[高优先] 添加连续推理模式（视频流）~~ [已完成]
- 连续视频推理已在前端推理页中具备

### 18. ~~[高优先] 前端结构重组~~ [已完成]
- 已新增 `frontend/` 目录
- 已拆分为静态多文件结构（HTML + shared JS/CSS + 页面主模块）
- 已删除 `test_v4.html`
- 根目录 `test_v5.html` / `calibration.html` 仅保留兼容跳转

### 19. [高优先] Canvas 渲染性能优化 [部分完成]
- 已增加文本宽度缓存，减少 `measureText()` 重复计算
- 已减少 lane UI 的全量重渲染倾向
- 仍未引入 OffscreenCanvas / requestAnimationFrame 双缓冲完整方案

### 20. ~~[中优先] 图片压缩优化~~ [已完成]
- 已走 `canvas.toBlob()` 二进制发送路径

### 21. ~~[中优先] 响应时间计算修正~~ [已完成]
- 前端推理页现在区分 `End-to-end`、`Server total`、`Overhead`

### 22. ~~[中优先] WebSocket 重连机制~~ [已完成]
- 前端推理页增加了自动重连、心跳、inflight 超时释放

### 23. ~~[中优先] Lane 点击坐标在缩放/平移后错位~~ [已完成]
- 推理页 lane 绘制已按当前 translate / scale 逆变换回原坐标系

### 24. ~~[中优先] 长视频状态膨胀~~ [已完成]
- 已增加旧 vehicle track 清理，避免长视频无限增长

### 25. ~~[中优先] 轨迹断线恢复异常~~ [已完成]
- 已修复 gap 处理逻辑，使轨迹断开后能够正常恢复

### 26. [低优先] 标定页进一步工程化 [部分完成]
- 已迁移为 `frontend/calibration.html` + `assets/js/calibration/app.js`
- 目前仍保留单主模块，未继续细拆为更多独立子模块

---

## 第三部分：当前推荐路线

### 已完成的高价值项
- Batch 推理
- 后端默认配置接入
- 服务端生成规范 pipeline
- WebSocket 协议校验与结构化错误
- 有界线程池 / batch 队列 / session 限流
- Session 生命周期与 reset 健壮性
- 前端模块化迁移到 `frontend/`
- 删除 `test_v4.html`
- 前端自动重连 / 心跳 / inflight 超时
- 前端 lane 坐标修正
- 前端延迟指标修正

### 仍保留为后续工作的项
- 完整 pipeline parallelism
- 真正零拷贝 ORT 输出
- 二进制响应协议
- 前端双缓冲 / OffscreenCanvas / RAF 完整渲染架构
- 用成熟 YAML 库替换轻量解析器

---

## 当前状态小结

- 后端健壮性与配置模型：**已完成本轮目标**
- 前端目录重组与旧页清理：**已完成本轮目标**
- 前端渲染性能：**部分完成**
- 协议/部署/文档一致性：**已完成本轮目标**

**总计：本轮重点优化已落地，剩余项以更深层性能工程为主。**
