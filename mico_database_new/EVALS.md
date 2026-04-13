# AGENT Evals Specification

## 1. 目标

本文件定义当前 harness 工程的评测目标、样例结构、评分规则与回归要求。

## 2. 为什么必须有 Evals

当前系统已经具备：
- 知识库
- RAG
- 外部证据
- MCP
- 混合式总控
- 多智能体

如果没有 eval，后续每次升级都可能出现：
- 回答变长但更空泛
- 加入知识库后反而更会复述
- 预测解释偏离模型边界
- 外部证据盖过内部事实
- 多智能体变复杂但回答质量下降

因此 eval 是 harness 的核心组成，不是附属功能。

## 3. 当前评测范围

第一阶段覆盖四类场景：
1. 患者页问答
2. 预测页问答
3. 首页问答
4. MCP Harness Connector

## 4. 样例结构

样例文件位于：
- `references/harness-evals/patient-analysis.jsonl`
- `references/harness-evals/prediction-analysis.jsonl`
- `references/harness-evals/dashboard-analysis.jsonl`
- `references/harness-evals/harness-connector.jsonl`

每条样例至少包含：
- `id`
- `intent`
- `question`
- `context`
- `expected_must`
- `expected_should`
- `forbidden`
- `notes`

可选扩展字段：
- `mcp_plan`
- `expected_mcp_usage`
- `expected_mcp_steps`

## 5. 评分维度

### 5.1 事实一致性
- 不编造 `patientId`、`sampleId`、疾病方向、概率
- 不把页面没有的数据说成存在
- 不把知识库规则说成当前业务事实

### 5.2 分层清晰
- 能区分数据观察、模型判断、内部知识、外部证据
- 不把不同层混成一个结论

### 5.3 边界控制
- 不把预测写成临床诊断
- 不把微生物偏离写成直接病因
- 不把外部网页证据写成内部统计

### 5.4 任务完成度
- 真正回答用户问题
- 有下一步建议
- 不只是复述页面字段

### 5.5 运行链路完整性
- 有 Task Packet
- 有 Execution Policy
- 有 Evidence Policy
- 有 trace 核心步骤
- 有 review 结果

### 5.6 MCP 连接层完整性
- 需要时能执行 `tool / resource / prompt` 计划
- trace 中能记录 `mcp_tool_call / mcp_resource_read / mcp_prompt_get`
- `mcpUsage` 与计划一致

## 6. 失败级别

### P0
- 编造核心事实
- 把预测写成确诊
- 混淆健康组与疾病组
- 请求失败或缺失 Task Packet / Policy / Trace 核心对象

### P1
- 未区分内部数据与外部证据
- 漏掉明显该提的模型边界
- 回答与问题不匹配
- trace 缺少关键步骤
- 需要命中的知识层、预测层或 MCP 层没有命中

### P2
- 语言重复
- 多智能体没有体现实际增益
- 下一步建议价值低
- `expected_should` 未命中

## 7. 通过标准

当前建议通过标准：
- `expected_must` 全部满足
- `forbidden` 全部不出现
- `expected_should` 至少命中 1 项
- trace 核心步骤完整
- review 与 evidence 层信息存在
- 若样例声明了 `expected_mcp_usage / expected_mcp_steps`，则必须满足

## 8. Bad Case 修复建议

bad case 报告现在会自动补：
- `primary_layer`
- `recommended_layers`
- `rationale`
- `next_action`

当前推荐修复层包括：
- `harness-runtime`
- `workflow-orchestration`
- `prediction-pipeline`
- `knowledge-rag`
- `mcp-connector`
- `agent-orchestration`
- `prompt-policy`
- `response-synthesis`
- `prompt-tuning`
- `runtime-ops`
- `connectivity`

## 9. 何时必须回归

以下改动必须重新跑 eval：
- `PatientWorkflow`
- `PredictionWorkflow`
- `DashboardWorkflow`
- `KnowledgeAugmentationService`
- `HybridResponseService`
- `MultiAgentOrchestratorService`
- MCP tools/resources/prompts
- 知识库关键文档

## 10. 运行方式

最小可跑脚本：
- `python scripts/run_harness_eval.py --suite all`

默认会：
- 读取 `references/harness-evals/*.jsonl`
- 调用 `ai-orchestrator` 的 `/api/ai/chat`
- 如样例声明 `mcp_plan`，继续调 MCP
- 拉取 `/api/ai/traces/{sessionId}/latest`
- 同时检查回答内容、trace 质量与 MCP 命中
- 输出完整报告：`references/harness-evals/latest-report.json`
- 输出 bad case 报告：`references/harness-evals/latest-bad-cases.json`