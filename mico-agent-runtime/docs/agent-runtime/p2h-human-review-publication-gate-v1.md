# P2-H 人工审核、可恢复中断与图谱发布门禁 v1

## 目标

P2-H 把 P2-G2 的低置信度、推测和冲突路径变成可审计的人工审核边界：

```text
staging graph
  -> GraphReviewQueue
  -> typed review decisions
  -> queue hash bound publication approval
  -> graph manifest approved
  -> explicit version publication
```

这条链路只处理独立文献知识图谱，不读取业务 MySQL，不包含受试者、样本或
`patient_data_manager` 数据。

## 审核队列

`build_graph_review_queue` 只从一个 `GraphBuildResult` 中提取保留下来的语义关系，
且要求 `qualityStatus=review_required`。每条工单使用不透明的：

- `reviewId=review-<32 位小写十六进制>`；
- `graphVersion` 和 `graphBuildRunId` 绑定构建版本；
- 固定 `issueCode`，例如 `CONFLICTING_ASSERTIONS`、`SPECULATIVE_RELATION`；
- 固定审核状态 `PENDING / APPROVED / REJECTED`；
- 受限的关系、证据 chunk 和最多 1200 字符的文献证据摘录。
- 安全投影的关系两端 `sourceEntity / targetEntity`、断言状态和置信度，供人工审阅；

原始 locator、样本 accession、内部记录 ID、Java payload、Token、URL、SQL 和自由审批
文本不属于审核模型。审核队列的 `queueHash` 对排序后的闭合工单计算 SHA-256，防止把
一个版本的审核结果套到另一个版本。

审核规则是 fail-closed：任何待审核或被拒绝的关系都不能直接发布。全部工单必须以
类型化 `GraphReviewDecision` 处理并全部批准，才能生成 `GraphPublicationApproval`。
拒绝关系后应重新构建 staging 版本，而不是在发布时静默删除或覆盖关系。

## 安全证据路径展示

`EvidencePathView` 只展示：

```text
pathId
status
hops[fromEntity, relation, toEntity, evidenceChunkId]
sourceDocumentIds
confidence
reviewRequired
```

路径最多 4 跳、最多 4 个来源文档，且对 URL、SQL、凭据、locator 片段和样本 accession
做结构化拒绝。`speculative`、`conflicted`、`partial` 或低于当前审阅阈值的路径只可
标记为待审阅，不能在生成层升级为 `supported`。

## LangGraph 暂停与恢复

现有 `IntentRuntime` 增加可选参数 `interrupt_on_review`。只有同时满足以下条件时，
知识证据图才会暂停：

1. 检索结果含冲突/推测/低置信度路径；
2. `interrupt_on_review=True`；
3. 注入了加密 LangGraph checkpoint saver。

暂停前先写入 `WAITING_APPROVAL` 和固定错误码 `GRAPH_REVIEW_REQUIRED`，然后由
LangGraph `interrupt()` 返回安全的 review/path 标识。首次返回不产生最终批准结论。
恢复只能接收闭合的：

```json
{
  "reviewId": "review-<32hex>",
  "decision": "APPROVED|REJECTED",
  "dataContractVersion": "v1"
}
```

Runtime HTTP 内部入口为：

```text
POST /internal/runtime/intent-runs/{runId}/review
```

它复用 Runtime Token，且只从加密恢复状态中取回原始任务请求。请求不能提交
checkpoint、工具参数、SQL、locator 或 Java 响应。恢复后的 `runId`、`taskId`、工具
响应关联和证据绑定仍由图节点重新验证。

环境开关 `MICO_RUNTIME_GRAPH_REVIEW_INTERRUPT_ENABLED` 默认关闭；没有持久化配置时，
不会退化成内存伪恢复。现阶段仍没有跨进程 replay/time-travel、Redis/SSE 长任务队列
或生产故障演练。

## 图谱发布状态

无待审核关系的 staging manifest 可以进入显式 publish。含待审核关系的 manifest 状态
为 `review_pending`，必须依次经过：

```text
review_pending
  -> approve_graph_manifest(GraphReviewQueue, GraphPublicationApproval)
  -> approved
  -> publish_graph_manifest
  -> published
```

`GraphPublicationApproval` 必须同时匹配 graph version、build run、review queue hash、
审核工单数量和受限审核人 ID。发布只切换独立知识图谱 registry，不触碰业务 MySQL，
也不自动删除旧版本；旧版本仍可作为显式回滚目标。

## 当前边界

本阶段已完成：

- 低置信度/冲突关系的闭合审核工单；
- 安全证据路径投影；
- LangGraph interrupt + 加密 checkpoint 的暂停/恢复边界；
- 审核队列哈希绑定的图谱发布前门。

本阶段未完成：

- 人工审核 Web 页面、组织身份目录和多审核人并行工作流；
- 生产 PostgreSQL/Redis/SSE 部署与故障恢复演练；
- 召回率、准确率、A/B 和成本评测；
- 统计显著性、临床决策、文献外部扩展或业务数据库写入。
