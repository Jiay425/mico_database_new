# P5 Governance Boundary v1

## 状态

本文件冻结 P5-A 的输入、工具和输出 Guardrail 边界。它不是审批页面、队列或临床系统的实现声明。

## 三层 Guardrail

1. 输入层只允许科研数据检索、证据汇总和统计编排。诊断、治疗、处方、因果结论、凭据索取、提示注入和变更性 SQL 请求使用固定 `RESEARCH_SCOPE_NOT_ALLOWED` 拒绝。
2. 工具层在模型规划后重新核验工作流是否属于 Java/Runtime 已批准候选，以及是否具备请求中已有的 scope。模型不能提升权限、增加工具、决定版本/定位器或绕过 Java。
3. 输出层只允许闭合的报告模型和数据版本元数据。内部 locator、样本 accession、内部记录 ID、Java 原始 payload、自由质量告警、SQL、Token、连接信息一旦进入公共投影即使用固定 `OUTPUT_SAFETY_BLOCKED` 拒绝。

## 动态只读 SQL 边界

`dynamic_read_query` 是通用受控动态只读查询工作流：模型可以提出 SQL 草案来表达用户的数据检索意图，但 SQL 不是事实，也不是权限。Java `DynamicReadQueryPolicy` 执行最终校验、固定业务 catalog、只读连接、超时、行数和语句限制；Python 不连接 MySQL，不能执行或改写 SQL。禁止 INSERT/UPDATE/DELETE/DDL/GRANT/文件导出、跨库引用、自由表名权限提升和无界结果。动态不等于自由 SQL。

`knowledge_retrieval` 是独立的全文证据工作流：它只读取显式配置的 65 篇全文索引，走向量与来源图谱检索，不访问业务 MySQL；结果必须带 `evidenceTier=fulltext`、来源分块和检索路径。两个工作流都由 LangGraph 依据自然语言问题路由，并经过同一安全策略边界。

## 审批与高风险操作

当前已支持的工具均为受控只读或统计编排，`ApprovalGate` 对这两类操作返回固定 `APPROVAL_NOT_REQUIRED`。批量导出、外部发布、敏感批量读取和业务写入一律返回 `APPROVAL_REQUIRED`，不会接受调用方布尔标志或自由审批文本绕过。Runtime 提供默认关闭的闭合审批票据 API；创建和决策均只使用固定枚举，决策还要求独立的部署注入 token 与 principal 绑定。Java BFF 仅代理创建和按 `runId + approvalId + 当前 principal` 绑定的查询，不代理审批决定。

## 脱敏审计

审计只保留 trace/run、节点、固定工具名、状态、固定错误码、瞬态 snapshot 标识和时间；不保存请求体、SQL、原始 locator、Java payload、Token 或连接信息。Java 自由 `qualitySummary.warnings` 不向 Runtime 公共结果转发。

## 进度与持久化边界

Java BFF 只转发浏览器请求；Runtime 现在提供受同一内部 Token 保护的进程内进度投影和有限事件流。事件只包含固定节点、工具名、状态、固定错误码、瞬态 snapshot 标识和时间，不包含 trace、请求、SQL 或 Java payload。启用 Runtime MySQL 后，进度查询在进程内记录缺失时可以从 `agent_run`/`agent_step` 的安全元数据重建有限投影；加密的 LangGraph checkpoint 由内部恢复路径读取，不会进入该投影。这不是 Redis 队列，也不提供时间旅行或完整 replay。

## 页面产品化状态

当前首页通过 Java BFF 展示通用科研问题的受控结果：页面只提交自然语言问题，结果仅展示
状态、汇总元数据、Python 生成分析的安全指标和固定局限性。页面不直连 Python，不展示审批人、
locator、样本 accession、内部记录 ID、SQL、Token 或原始 payload。

## 未完成项

P5 尚未接入审批页面、真正的暂停信号、Redis 队列、长任务 SSE、部署级恢复演练或时间旅行 replay。当前 Runtime 已有受控的加密 checkpoint 恢复路径，审批后的图释放只有在该路径和对应生命周期迁移均已部署后才可宣称。审批字段迁移 `0002_approval_ticket_lifecycle` 仍待远程 Schema 的独立迁移窗口执行。P2-B1 的 Runtime MySQL 是独立状态库，仍不改变 Python 不直连业务 MySQL 的边界。
