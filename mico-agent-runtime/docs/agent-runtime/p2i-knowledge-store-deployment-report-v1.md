# P2-I 知识库首次导入报告

## 结论

独立的 pgvector 和 Neo4j 知识库已完成首次真实导入和只读混合检索验证。两个容器
仅使用本机回环端口；没有访问业务 MySQL，也没有把任何业务库凭据写入项目。

## 安全资产清单

| 存储 | 当前资产 | 数量/状态 |
|---|---|---:|
| PostgreSQL + pgvector | `fulltext-gemini-paper-embedding-v1` | 65 个文献向量 |
| PostgreSQL + pgvector | 全文 chunk | 3617 |
| Neo4j | `fulltext-provenance-graphrag-v3` | 5238 节点、24341 条边、`published` |
| Neo4j | `fulltext-provenance-graphrag-v4` | 5238 节点、24341 条边、`review_pending` |
| GraphReviewQueue | v4 待审核关系 | 280 |

PostgreSQL 的当前索引清单同时保留历史导入版本记录；当前运行检索使用
`fulltext-gemini-paper-embedding-v1`。文献范围是 65 篇全文，不把只有元数据的记录
计入全文证据资产。

## 实际验证

- `scripts/ingest_knowledge_stores.py`：向量/chunk 导入成功，v3 基线导入成功；
- `scripts/ingest_graph_versioned.py`：v4 导入成功，但保持 `review_pending`；
- `scripts/export_graph_review_queue.py`：生成 280 条待审核队列；
- 审核队列已补齐关系两端、断言状态和安全证据摘录，形成可人工审阅包；
- `tests/test_database_knowledge_integration.py`：真实 pgvector + Neo4j 混合检索
  `1 passed`；结果同时包含 vector 和 graph 来源及有界路径；
- `tests/test_database_knowledge_intent_integration.py`：主 LangGraph 意图路由真实接入
  database backend，`1 passed`；文学证据路线未调用 Java 业务工具；
- 全量 Python 测试：`161 passed, 2 skipped`；
- `compileall`：通过。

混合检索测试使用数据库中已经存在的 Gemini embedding 向量作为测试查询向量，未调用
外部 Gemini API。没有自动批准 v4，也没有执行发布切换。

## 发布边界

默认检索版本仍是 v3。Neo4j 查询要求对应 `KnowledgeGraphBuild.status='published'`，
因此 v4 的导入不会绕过人工审核进入检索。只有完整处理 `GraphReviewQueue`、生成绑定
queue hash 的批准凭证、再执行显式发布，v4 才能成为可检索版本。

当 pgvector 与 Neo4j 两个真实存储开关同时开启且没有显式指定 backend 时，Runtime
自动选择 `database`；旧 JSONL local backend 只在未启用真实存储或被显式指定时使用。

## 未完成

- v4 关系的人工审核和正式发布；
- Gemini 在线 query embedding 验证；
- 评测集、A/B、成本和 bad-case 回归；
- 生产环境的密钥管理、备份、监控与高可用部署。
