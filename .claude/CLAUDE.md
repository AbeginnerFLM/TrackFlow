# TrackFlow 协作规则

以下规则为本项目固定执行规范：

1. 每次对话开始前，先阅读 `HANDOVER.md` 熟悉当前约定与部署流程。
2. 每次修复 bug 后，必须将问题、原因、解决方案追加记录到 `debug.md`。
3. 每次代码更新后，必须执行完整发布链路：
   - 本地 `git push`
   - GPU 服务器 `git pull`
   - 杀死旧进程（`yolo_edge_server`）
   - 重新编译
   - 重新启动服务

