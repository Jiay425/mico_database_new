# Agent Evals Compatibility Pointer

本文原有的旧 harness/MCP 评测正文已移除。文件名暂作为旧 `ai-orchestrator` MCP 资源和兼容回归资产的入口保留，不再作为当前评测规范。

当前评测规范请以以下文档和代码为准：

- [P2-J4.1 Trace/Eval v2](../mico-agent-runtime/docs/agent-runtime/p2j4-trace-eval-v2.md)
- [P2-J4.1A architecture calibration](../mico-agent-runtime/docs/agent-runtime/p2j4-1-architecture-calibration-v1.md)
- [P2-J4.1 v2 task set](../mico-agent-runtime/evals/p2j4-task-set-v2.json)
- [P2-J4.1 expanded v3 task set](../mico-agent-runtime/evals/p2j4-task-set-v3.json)
- [P2-J4.1 runner](../mico-agent-runtime/evals/p2j4_runner.py)
- [P2-J4.1 Decision Trace migration](../mico-agent-runtime/evals/p2j4_decision_trace_migrate.py)
- [P2-J4.1 stability runner](../mico-agent-runtime/evals/p2j4_stability_runner.py)
- [P2-J4.1 decision dataset](../mico-agent-runtime/evals/p2j4_decision_dataset.py)

当前状态（2026-08-27）：P2-J4.1 的历史 Canary/真实 Eval 资产继续保留；Decision SFT Freeze v2（865）与 Qwen3-8B Decision SFT v5 已完成。DPO v4 Controlled Preference Freeze r2 为 408 pairs（train 326 / validation 82），已完成正式训练并保留 adapter；Test70 为 70/70，OOD30 为 29/30，generation smoke 为 10/10。DPO 冻结 manifest 中的 `trainingStarted=false` 仅表示冻结/参考日志阶段，不代表正式训练未执行。当前新增工作是 Dynamic Materialization Runtime 的真实 E2E，不得重新训练 DPO；应使用已有 SFT/DPO adapter 完成 DeepSeek Flash + Java/MySQL 联调。
