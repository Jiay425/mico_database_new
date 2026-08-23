# P2-C2 Java 受控疾病与队列可行性工具契约 v1

状态：已实现 Java 固定只读 Read Model、受控 Executor 和远程只读集成验证。Java 仍是唯一业务事实源；Python LangGraph 只能通过 `/internal/agent/tools/execute` 使用这些工具，不能直连业务 MySQL。

## 固定工具

### `resolve_disease`

输入仅为闭合字段：

```json
{"rawLabel":"T2D"}
```

输出为 `DiseaseAssertion`：

- `canonicalDiseaseId`、`canonicalNameZh`、`mappingStatus`、`mappingConfidence`；
- `mappingVersion=disease-mapping-v1`；
- 受控 `sourceEvidence`；
- 未审核或不匹配的标签返回 `mappingStatus=unmapped`，不推断为不存在疾病。

`disease-mapping-v1` 只正式覆盖 `type_2_diabetes → 2型糖尿病` 和 `healthy_control → 健康对照` 两个最小映射族。T2D / Type 2 diabetes 及健康对照的大小写、空格、下划线、连字符变体按固定规范化匹配；不把原始疾病名称集合或 130 个名称称作正式临床分类。该工具只读 `patients.disease` 取得证据计数，不写入疾病表。

### `build_cohort`

输入严格为：

```json
{"comparisonId":"t2d_vs_healthy_v1"}
```

任何其他 comparison、任意疾病文本、自由条件、SQL、表名和字段均被拒绝。

输出为汇总级 `CohortFeasibilityReadModel`，包括：

- `t2d_case` 与 `healthy_control` 的 `sampleRecordCount`；
- 有标准丰度的不同 `(patient_id, sample_id)` 样本键匹配数；
- age、gender、country、body site 缺失计数；
- unmapped、missing、conflict、复合健康候选、T2D 共病/未映射成分排除计数；
- taxonomy/feature/source batch 分布；
- 固定局限性代码和 `subjectLinkStatus=unverified`。

所有数量均为内部业务记录或样本键，不是独立 Subject 或真实患者人数。健康对照只接受规范化后完整等于 `healthy`、`healthy control`、`healthy controls`、`healthy subject` 或 `control` 的标签；任何复合健康候选（包括与未映射成分组合）均排除。T2D 可保留复合标签，但 `t2dComorbid*` 单独计数，不能称为“纯 T2D”；T2D 与 healthy 同时命中时归为冲突并排除。

## 查询和证据边界

Mapper 使用固定 SQL、`#{}` 参数绑定、显式列和固定排序；没有 `${}`、动态表名、动态列名、自由 WHERE、自由 SQL 或 `SELECT *`。标准丰度按不同样本键和存储行口径聚合，不把丰度行称为唯一物种数。

两个工具的成功响应携带 `DataSnapshot`，但 P2-C2 仍只生成 transient 证据：`dataSnapshotId`、`queryHash`、`rowCount`、`generatedAt` 和版本字段来自 Java Read Receipt/固定结果上下文；不创建可回放快照，不写数据库。`snapshotPersistence=transient` 不代表 Runtime checkpoint 或跨进程恢复。

对外不返回 sample accession、sourceSampleId、internalRecordId、患者信息、原始疾病文本、`cohortCondition`、Java 自由 warning、原始 payload、SQL、连接信息或诊断/治疗结论。`patient_id` 只代表 `internalRecordId`。

## 远程实时只读基线

2026-08-21 通过临时 SSH loopback 隧道、Java 受控只读链路验证（`remote_live_verified`）：`patients` 24,264 条内部记录，标准丰度 20,852,714 条存储行，23,072 个不同标准丰度样本键；严格分类得到 T2D 1,379 条样本记录、healthy 5,016 条样本记录，其中分别有 1,379 和 4,997 个标准丰度样本键。排除汇总为：未映射 13,914/12,776（内部记录/标准样本键）、缺失疾病 2,374/2,339、冲突 0/0、复合健康 1,581/1,581、T2D 共病或含未映射成分 381/381。该统计不推断独立 Subject 数。

## 未实现

本契约不实现差异统计、显著性、效应量、因果推断、文献检索、任意队列、疾病人工审核扩展、持久化快照恢复或前端接口。
