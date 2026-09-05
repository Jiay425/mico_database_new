# P2-G1 图谱生产管线与实体对齐 v1

本文件是图谱生产阶段记录。其“评测按计划最后实施”是当时的阶段状态；当前
P2-J4.1 已有 50 条任务和一轮真实 Trace 基线，其中开放任务实际观察到 vector
和 graph 路由。该运行仍不等于独立的 GraphRAG A/B 实验、图谱版本对照或知识库
数值检索质量结论；这些需要单独的 oracle 和实验设计。

## 目标

P2-G1 将现有全文 GraphRAG 图从一次性构建资产推进为可审计的版本化生产管线：

```text
medical_chunks.jsonl
  -> entity mention
  -> normalization / alignment
  -> relation + assertion extraction
  -> evidence binding
  -> quality gate
  -> staging graph
  -> manifest validation
  -> explicit publication
```

本阶段不读取业务 MySQL，不包含患者、样本或宏基因组原始行。知识库仍是独立的
PostgreSQL/pgvector + Neo4j；Java 业务工具仍是业务事实唯一入口。

## v4 资产和回滚边界

当前实现新增 `fulltext-provenance-graphrag-v4`。v3 不删除、不覆盖，仍可作为显式
回滚/对照版本。v4 的节点 ID 使用 `v4:` 版本前缀，避免不同图版本在 Neo4j 中
发生节点合并；关系和证据 chunk 仍通过原始 chunk 标识回连全文存储。

构建输出：

- `medical_knowledge_graph_v4.jsonl`：节点和关系记录；
- `medical_knowledge_graph_v4_manifest.json`：闭合构建清单；
- `medical_knowledge_graph_registry.json`：显式发布后的当前版本指针。

`build_graph_pipeline.py` 只生成 `staging` 资产。只有调用发布步骤后，manifest 才变为
`published`；只有将 `MICO_KNOWLEDGE_GRAPH_VERSION` 指向该版本，检索器才会读取它。
默认版本仍为 v3，防止未审核图自动上线。

## 实体标准化和对齐

实体节点至少保存：

```text
entityType
canonicalEntityId
canonicalLabel
ontologySource
ontologyId
normalizationStatus
aliases
```

状态含义：

- `resolved`：命中当前代码化别名契约；
- `candidate`：当前仅能识别为候选实体，例如未完成外部 taxonomy 对齐的 Taxon；
- `ambiguous`：实体类型或别名存在歧义；
- `unmapped`：未命中当前受控映射；
- `source_identifier` / `controlled_topic`：文献结构节点，不是生物医学标准实体。

当前 `mico-controlled-alias-v1` 不是外部临床本体的冒充。Taxon 的
`taxonomy-candidate-v1` 只表示候选标准化，未宣称已经完成 NCBI/GTDB 正式对齐。
正式外部 ID 接入前，候选 Taxon 不能作为确定事实输出。

## 关系、证据和质量门禁

关系只允许代码化集合：

```text
CAUSES / MEDIATES / PROMOTES / INHIBITS
INCREASED_IN / DECREASED_IN / ASSOCIATED_WITH
PART_OF / IN_SECTION / HAS_TOPIC / MENTIONS_ENTITY
```

每条关系必须同时具备：

- `relationClass`；
- `assertionStatus`：`asserted`、`negated`、`speculative` 或 `conflicted`；
- `confidence >= 0.65`；
- `evidenceChunkId`；
- 语义关系必须有 `evidenceText` 和起止位置；结构关系至少绑定 `evidenceChunkId`；
- `graphVersion` 与 `graphBuildRunId`。

质量门禁会拒绝缺少端点、非法关系、非法断言、缺少证据或低置信度关系。相同实体对
和关系同时出现 asserted/negated 时保留到 staging，但标记为 `review_required` 和
`conflicted`，不能在生成层升级为确定结论。

manifest 中记录：实体解析状态、关系/断言分布、拒绝记录数、需要审核的关系数和输入
内容指纹。未通过门禁的构建不能发布。

## Neo4j 发布方式

版本化入库使用固定参数化 Cypher：

1. 只删除正在重建的同一 `graphVersion`；
2. 不删除 v3 或其他版本；
3. 先写节点、关系和 `KnowledgeGraphBuild` 构建元数据；
4. `--publish` 才把该版本标为 published，并将旧 published 版本标为 retired；
5. 检索器通过受限的 `MICO_KNOWLEDGE_GRAPH_VERSION` 选择版本。

这不是业务数据库迁移，也不会访问 `patient_data_manager`。

## 当前未完成

- Taxon 的 NCBI/GTDB 外部 ID 正式对齐；
- 全量模型关系抽取和人工审核队列；
- Vector/Graph/Hybrid 的真实端到端评测和 A/B 测试。

P2-H 已补齐 checkpoint/interrupt 边界、低置信度/冲突关系审核队列和发布审批前门；
P2-J4.1 的离线能力评测已单独建立，真实端到端评测和 A/B 测试仍未验证。

图谱语义、版本管理、证据路径和生产发布边界已完成阶段性资产；真实端到端评测仍需在
P2-J4.2 的受控环境中单独执行。
