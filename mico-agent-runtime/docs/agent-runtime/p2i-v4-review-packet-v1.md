# v4 图谱人工审核包 v1

## 审核对象

- 图谱版本：`fulltext-provenance-graphrag-v4`
- 审核状态：`OPEN`
- 待审核工单：280 条
- issue：全部为 `CONFLICTING_ASSERTIONS`
- severity：全部为 `high`
- queue hash：由生成的闭合队列文件保存，批准时必须原样绑定

审核包文件：

```text
../mico_database_new/references/knowledge/medical/rag/medical_knowledge_graph_v4_review_queue.json
```

每条工单包含安全投影的 reviewId、graphVersion、buildRunId、sourceEntity、relation、
targetEntity、assertionStatus、confidence、evidenceChunkId 和 evidenceExcerpt。

不包含样本 accession、locator、患者信息、Java payload、数据库连接信息或凭据。
`sourceEntity`、`targetEntity` 和证据摘录仍受闭合模型的敏感文本校验。

## 当前分布

- `INCREASED_IN`：108
- `ASSOCIATED_WITH`：55
- `CAUSES`：41
- `MEDIATES`：30
- `DECREASED_IN`：28
- `PROMOTES`：16
- `INHIBITS`：2

置信度分布：0.70–0.79 为 11 条，0.80–0.89 为 269 条。这里的置信度不是医学
真值；由于断言冲突，所有工单仍必须人工处理。

## 审核与发布规则

审核人只能提交闭合的 `GraphReviewDecision`。所有工单完成决定且没有拒绝项后，才可
生成绑定 queue hash 的 `GraphPublicationApproval`。任一关系被拒绝，应重新生成新的
staging 图谱版本，不在发布阶段静默删除关系。审核完成前继续使用 v3 published 基线。

可使用 `scripts/apply_graph_review_decisions.py` 提交完整决定文件。该脚本不会自动补
决定；缺少工单、存在拒绝项或审核人 ID 不符合闭合格式时，不会修改 manifest。
