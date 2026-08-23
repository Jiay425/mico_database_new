# P2-C3 Java 科研队列 BFF 契约 v1

## 入口与信任边界

浏览器唯一入口是：

```text
POST /agent/cohort/feasibility
```

浏览器只能提交空 JSON 对象 `{}`（或无请求体）。它不能提交 requester、scope、allowed workflow、comparison ID、工具名、Python Token、数据库信息或任何疾病筛选条件。Java Controller 从当前已认证会话派生主体摘要，并在服务端生成 `runId`、`taskId`、`traceId`。

Java BFF 只在 `mico.agent.bff.enabled=true` 且环境注入 `MICO_AGENT_RUNTIME_BASE_URL`、`MICO_RUNTIME_INTERNAL_TOKEN` 时启用；缺少配置时返回固定 503。浏览器不直接访问 Python。Java 使用固定 JSON 请求调用 Python：

```text
POST {MICO_AGENT_RUNTIME_BASE_URL}/internal/runtime/cohort-runs
```

Python Runtime 继续使用相同的内部 Token fail-closed 认证，并只能执行 `resolve_disease → build_cohort` 的固定队列工作流。Python 仍不能连接 MySQL；业务事实只能从 Java 内部 Tool API 获取。

## 对外响应

成功响应使用 `p2d-cohort-bff-response-schema-v1.json` 的闭合结构，只包含：

- T2D/健康对照样本记录数和匹配标准丰度样本键数；
- age、gender、country、body site 缺失计数；
- 未映射、缺失、冲突、复合健康、T2D 共病/未映射成分排除计数；
- 固定映射版本、taxonomy/feature/source batch 分布和 transient 证据元数据；
- 固定限制项与 `not_clinical_diagnostic_or_treatment_advice`。

响应不得包含 sample accession、sourceSampleId、internalRecordId、原始疾病文本、患者姓名、locator、`cohortCondition`、SQL、业务库名、Token 或 Java 原始 payload。`subjectLinkStatus` 固定为 `unverified`，所有数字是样本记录/样本键统计，不是独立真实患者数。

## 差异任务 BFF（P3 提交/轮询入口）

在同一 Java 登录与运行时配置边界下，差异统计使用独立的异步任务入口：

```text
POST /agent/research/differential
GET  /agent/research/{runId}/differential/{jobId}
```

浏览器提交仍只能是空 JSON 对象 `{}`（或空请求体）。Java 服务端生成 `runId`、`taskId`、`traceId`，并固定提交 `t2d_vs_healthy_v1`、分析 scope 和已批准的 Runtime 端点；浏览器不能提交 comparison ID、工具名、版本、SQL、scope 或 Python Token。

Python 返回的 `jobId` 必须满足 `job-<32 位小写十六进制>`，Java 轮询时校验 `runId`、`jobId`、状态、transient snapshot、版本字段、结果上限和闭合响应字段。完成报告只允许包含汇总统计、物种结果和证据元数据，不返回 locator、样本标识、内部记录 ID、原始 Java payload 或自由告警。

该入口只提供进程内异步 job 的提交/轮询外观；不代表已经具备外部队列、持久化 artifact、崩溃恢复或跨进程回放能力。完整分析仍受 Java/Python 固定分页上限约束，超过上限时必须失败闭合，不得把截断结果冒充全量结果。

## 安全错误

| 条件 | HTTP/代码 |
|---|---|
| BFF 默认关闭 | 503 / `AGENT_BFF_DISABLED` |
| Runtime URL 或 Token 缺失/非法 | 503 / `AGENT_RUNTIME_NOT_CONFIGURED` |
| 浏览器未登录 | 401 / `UNAUTHORIZED` |
| 请求含未知或非空字段 | 400 / `INVALID_REQUEST` |
| Runtime Token 错误 | 502 / `AGENT_RUNTIME_UNAUTHORIZED` |
| Runtime 超时 | 504 / `AGENT_RUNTIME_TIMEOUT` |
| Runtime 响应不符合闭合安全 Schema | 502 / `AGENT_RUNTIME_RESPONSE_INVALID` |

错误响应不转发 Python 原文、请求头、连接地址、SQL、数据库信息或原始载荷。

## 真实远程基线

2026-08-21 通过临时 SSH loopback 与 Java 固定只读链路验证的严格队列汇总为：T2D 1,379 条样本记录、健康对照 5,016 条样本记录；匹配标准丰度样本键分别为 1,379 和 4,997。未映射/缺失/冲突/复合健康/T2D 共病排除分别为内部记录与标准样本键 `13,914/12,776`、`2,374/2,339`、`0/0`、`1,581/1,581`、`381/381`。这些数字不推断独立 Subject 或真实患者数。

2026-08-22 P3 固定差异数据集的远程只读复核选择
`taxonomyVersion=species-v1`、`featureVersion=v1`、`sourceBatch=merged_abundance`：初始 case/control
分别为 1,184/4,997 个样本记录；按年龄差不超过 5 年、性别/国家（地区）/body site 精确匹配的
1:1 无放回规则后，两组各保留 444 个样本记录，因缺失/非法协变量排除的 case/control 分别为
203/704。以上数字均为样本记录/样本键口径，不是独立 Subject 或真实患者计数。

## 非临床边界

该入口是科研数据队列可行性与受控差异统计展示，不是临床诊断、风险预测、因果推断、治疗或处方系统。差异统计使用固定、版本化的确定性方法和安全结果上限；文献检索、因果解释、临床建议和前端完整产品化仍未实现。
