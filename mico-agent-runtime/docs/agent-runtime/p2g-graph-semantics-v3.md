# P2-G：GraphRAG 语义图谱 v3

## 目标

v3 是在现有全文向量索引和 Neo4j v2 来源图之上的语义强化版本。v2 保留用于回滚和对照，v3 使用独立版本字段和版本前缀节点，不覆盖 v2 数据。

## 实体契约

每个实体都包含：

```text
graphVersion
nodeId
entityType
label
canonicalLabel
aliases
normalizationStatus
extractionMethod
```

当前实体类型：

```text
Paper
Chunk
Section
Topic
Disease
Taxon
Metabolite
Pathway
Concept
HostProcess
```

业务实体使用 `v3:<entity-type>:<sha256-prefix>` 不透明 ID。Chunk 是来源连接例外：使用 `v3:chunk:<sourceChunkId>`，以便回连 PostgreSQL 的全文块；对外响应仍不暴露内部节点 ID。

## 关系契约

语义关系必须在同一全文句子中同时满足：

1. 至少两个已标准化实体；
2. 存在显式关系触发词；
3. 关系带有全文块和句子证据；
4. 置信度不低于 `0.65`。

关系类型包括：

```text
ASSOCIATED_WITH
CAUSES
MEDIATES
PROMOTES
INHIBITS
INCREASED_IN
DECREASED_IN
```

结构关系包括：

```text
PART_OF
IN_SECTION
HAS_TOPIC
MENTIONS_ENTITY
```

每条边包含：

```text
relationClass
confidence
assertionStatus
evidenceChunkId
evidenceText
evidenceStart
evidenceEnd
extractionMethod
```

`assertionStatus` 当前为：

```text
asserted
speculative
negated
```

`negated` 不会被 GraphRAG 当作支持性事实；进入路径后转换为 `conflicted` 状态。`speculative` 关系保持推测状态，不能被生成模型升级为确定结论。

## 当前 v3 入库事实

本次使用现有 65 篇全文、3,617 个全文块构建：

```text
节点：5,238
边：24,341
语义关系：2,135
语义关系覆盖块：1,369
asserted：1,763
speculative：297
negated：75
```

实体和关系导入 Neo4j，GraphRAG 查询只读取：

```text
graphVersion = fulltext-provenance-graphrag-v3
confidence >= 0.65
evidenceChunkId IS NOT NULL
```

旧 v2 图谱未删除，保留为回滚和对照版本。

## 多跳路径

检索使用固定的 1、2、3 跳查询形状，不使用无边界的变量长度展开：

```text
实体 → Chunk
实体 → 实体 → Chunk
实体 → 实体 → 实体 → Chunk
```

这样既保留跨文档共享标准化实体的多跳能力，又避免在 Neo4j 中对高连接度实体产生组合爆炸。

路径必须回连全文块。路径状态为：

```text
supported
speculative
conflicted
partial
unsupported
```

## 限制

- Taxon 目前是经过拉丁化双名词形过滤的候选实体，不等同于完成 NCBI/GTDB 等正式 taxonomy 对齐；
- 关系抽取当前使用可审计的句子模式，不宣称已完成医学知识人工审核；
- 生成模型只能引用检索返回的路径和全文块，不能把 GraphRAG 路径直接升级为因果或临床结论；
- 质量评测、人工审核队列、增量更新和版本回滚将在后续阶段完成。

