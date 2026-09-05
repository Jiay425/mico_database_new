# Agent Architecture Compatibility Pointer

本文原有的旧 harness 架构正文已移除。文件名暂作为 `ai-orchestrator` MCP 资源和历史回归资产的兼容入口保留，不再承载当前架构事实。

当前架构与一致性口径请以以下文档为准：

1. [MICO_LANGGRAPH_COLLABORATION_PLAN.md](../MICO_LANGGRAPH_COLLABORATION_PLAN.md)
2. [P2-J Scientific Exploration Agent plan](../mico-agent-runtime/docs/agent-runtime/p2j-scientific-exploration-agent-plan-v1.md)
3. [P2-J4.1A architecture calibration](../mico-agent-runtime/docs/agent-runtime/p2j4-1-architecture-calibration-v1.md)
4. [P2-J0 research contract](../mico-agent-runtime/docs/agent-runtime/p2j0-research-contract-v1.md)

当前摘要：Java 是业务事实唯一来源；Python/LangGraph 负责受控 Scientific Agent Loop；Python 不直连业务 MySQL；开放科研问题才进入 `State → Action → Observation → validate/continue/stop`；高层 Action 集合冻结，`execute_read_query` 仍是底层 Java 只读边界；Trace/Eval 先于 SFT/Preference，GRPO/RL 暂缓。
