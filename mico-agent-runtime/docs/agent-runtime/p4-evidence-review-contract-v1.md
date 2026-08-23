# P4 证据与科学审阅边界 v1

## 状态

P4-A 已建立闭合的证据查询、支持/反方分层和不足证据状态。默认端口
`UnconfiguredEvidenceSearchPort` 不访问外部网络、不伪造文献结果；PubMed/CrossRef 适配器
只有在显式环境配置后才启用。

已完成一次真实 PubMed 元数据只读核验：固定疾病 ID `type_2_diabetes`、标准物种名和上限
`limit=3`，返回 3 条有界元数据结果。该事实只证明适配器、响应闭合和上限有效，不证明全文阅读、
研究设计质量、因果关系或临床结论。对应测试为默认跳过的
`tests/test_evidence_remote_integration.py`，需显式设置
`MICO_EVIDENCE_REMOTE_INTEGRATION=true` 才联网运行。

真实 PubMed 适配器已接入一次 Evidence LangGraph 审阅：返回状态为 `COMPLETED`，1 个统计物种中
1 个得到有界证据投影。CrossRef 元数据适配器也已完成一次真实只读核验：固定
`type_2_diabetes`/`Bacteroides` 请求返回不超过 2 条、来源为 `crossref` 的有界结果。外部服务不可达时仍安全转换为
`EVIDENCE_SOURCE_FAILED`，不以 MockTransport 结果替代真实证据。

P4 现已接入 P3 异步差异任务的安全投影：差异统计完成后，运行时在同一进程内把已验证的
`DifferentialAnalysisReport` 交给 Evidence LangGraph。证据结果以可选的
`evidenceReport` 出现在差异 job 查询响应中；外部检索未配置时仍返回
`INSUFFICIENT_EVIDENCE`，不会阻断已经完成的 Java 统计，也不会伪造文献。证据失败只返回固定
`evidenceErrorCode`，不覆盖统计报告。完成的 `EvidenceReviewReport` 现在可由 Runtime 持久化协调器
写入独立 `agent_artifact` 的 `evidence_summary` 元数据：只保存版本、SHA-256 内容摘要、opaque
`artifact://artifact-<32hex>` 引用和已验证 transient snapshot 关联，不保存原始报告正文、文献摘要、
样本标识或 locator。真实对象存储仍未接入，引用不可下载也不代表报告已可跨进程恢复。

## Internal entry

Runtime app factory exposes `POST /internal/runtime/evidence-runs`, protected by
the same internal Runtime token as the other closed endpoints. It accepts only
the closed `EvidenceTaskRequest`; audit events and raw adapter payloads are not
returned. When external-source configuration is absent, the endpoint returns
`INSUFFICIENT_EVIDENCE` through `UnconfiguredEvidenceSearchPort` rather than
inventing literature.

## 信任边界

```text
Java 内部统计报告
  -> Python EvidenceTaskRequest（仅闭合统计结果与 transient snapshot 元数据）
  -> EvidenceSearchPort（未来外部文献/内部知识适配器）
  -> Python 确定性分层与科学审阅
  -> EvidenceReviewReport
```

Java/远程 MySQL 的内部事实优先。外部文献只能作为 supporting、contrary 或 context 证据，
不能覆盖 Java 的样本记录、队列、版本、丰度或统计事实；外部证据也不能把相关性改写成因果、
诊断或治疗建议。

## 闭合输入与输出

`EvidenceTaskRequest` 只允许一份已完成的 `mann_whitney_u_bh_fdr_v1` 报告，以及固定
`supporting`/`contrary` 方向。它不接受 locator、sample accession、内部记录 ID、SQL、数据库
地址、Token 或用户自由筛选条件。

每条 `LiteratureEvidenceItem` 必须绑定到统计报告中的一个标准物种名、固定疾病 ID
`type_2_diabetes`、来源、外部文献 ID、标题、年份、方向和有界摘要。未知物种或不符合闭合
模型的返回值直接丢弃/失败，不进入对外报告。

输出明确区分：

- `supporting`：与当前统计观察方向相符的外部证据；
- `contrary`：方向相反、限制或不支持的外部证据；
- `INSUFFICIENT_EVIDENCE`：没有可验证证据时的诚实状态，不补写结论。

报告保留内部统计 snapshot ID 作为证据关联，但不保存原始样本行或 locator。固定局限性包括
外部证据不覆盖内部事实、证据不等于因果证明，以及科研支持而非临床诊断/治疗建议。

## LangGraph 节点

```text
validate_evidence_task
  -> evidence_policy_gate
  -> prepare_evidence_queries
  -> retrieve_evidence
  -> scientific_review
  -> terminal
```

检索端口由依赖注入提供；默认空端口用于离线和 fail-closed 测试。未来真实适配器必须分别
记录来源、查询版本、检索时间和安全错误码，并通过同一闭合模型，不能把任意网页/RAG 文本
直接塞入报告。

## 当前未完成

- 未接入 Scholar；PubMed/CrossRef 的真实适配器仍未进入默认运行路径；
- 未实现内部知识索引、去重、重排、引用格式化和全文证据核验；
- 证据报告的 artifact 元数据写入已建立，但真实对象存储、artifact 下载和版本化回放尚未实现；
- 未实现真实人工审批页面、Redis/SSE 长任务队列、长期 Trace 和部署级恢复演练；P5/P6
  的当前离线 Guardrail 与评测资产见对应版本文档。
