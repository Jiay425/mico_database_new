# Mico Java × Python LangGraph 协作实施方案

> 当前主链校准：本文保留 Java/Python 信任边界和 P0–P6 历史阶段背景；开放式科研探索的唯一当前设计依据是 `mico-agent-runtime/docs/agent-runtime/p2j-scientific-exploration-agent-plan-v1.md`。本文中早期的固定 T2D/健康对照、固定统计任务和固定首个工作流仅作历史示例，不得作为当前路由、Mapper、SQL 或 Python 脚本实现。

> **2026-08-27 状态校准**：本文保留早期 P0–P6 背景，但当前训练与运行事实以 P2-J4 冻结资产和 Dynamic Materialization 文档为准。Decision SFT Freeze v2（865）与 Qwen3-8B Decision SFT v5 已完成；DPO v4 Controlled Preference Freeze r2（408 pairs）已完成正式训练并保留 adapter，Test70 为 70/70，OOD30 为 29/30，generation smoke 为 10/10。Dynamic Materialization Runtime 是训练后的新运行实验：本地 QueryPlan/AnalysisPlan/Trace 改造已完成，尚未完成真实 DeepSeek Flash + Java/MySQL Dynamic E2E。冻结数据 manifest 中的 `trainingStarted=false` 只表示冻结/参考日志阶段，不代表 DPO 模型未训练。下一步是部署已有 SFT/DPO adapter 做 Dynamic E2E，不得重新训练 DPO。

## 1. 最终决策

Mico 采用**双服务、单一业务事实源**架构：

- **保留 Java 主业务系统**：已有页面、Controller、Service、Mapper、权限、远程 MySQL、数据导入和普通查询功能均不迁移。
- **新建 Python + LangGraph Agent Runtime**：只负责任务规划、图状态、工具调用、统计任务编排、证据综合、安全 Gate、审批、Trace 和 Eval。
- **Python 不直接连接业务 MySQL**：它没有业务数据库账号，也没有不受约束的自由 SQL 能力；若科研意图确实需要动态读取，模型只能提出闭合的单条只读 SQL 草案，由 Java 受控工具执行最终只读、跨库、危险语句、超时和行数校验。
- **不并行运行两个总控 Agent**：现有 `mico_ai_service/ai-orchestrator` 中可复用的工具定义、知识库、Trace 设计会被审计和迁移；当 Python Runtime 达到评测门槛后，它成为唯一的 Agent 总控。旧 Java AI 编排不做破坏性删除，先保留为兼容资产。

这不是“把 Java 改写成 Python”。Java 是稳定的业务服务，LangGraph 是独立的 Agent 引擎。

## 2. 服务边界

```text
浏览器
  |
  v
Java 主业务后端（认证、页面 API、事实源、权限、BFF）
  |                                  ^
  | POST /internal/agent/runs        | SSE / 查询运行结果
  v                                  |
Python Mico Agent Runtime（FastAPI + LangGraph）
  |                  |                    |
  | MCP/HTTPS 工具    | 统计任务             | 知识/文献检索
  v                  v                    v
Java Agent Tool API  Python Worker       内部知识库 / 外部证据
  |
  v
远程 MySQL patient_data_manager（唯一业务事实源）
```

### 2.1 Java 主业务系统负责什么

1. 用户认证、角色和数据权限。
2. 样本、项目、疾病、物种丰度、统计首页等业务事实。
3. 远程 MySQL 的唯一业务读写入口。
4. 给前端提供普通页面接口和 Agent 运行入口。
5. 对 Python 的所有请求做参数、权限、最大返回量、数据版本和审计校验。
6. 统一将 Agent 进度和结果转发给网页。

Java **不负责** LLM 循环、Agent 图状态、提示词、子 Agent 调度或外部证据检索。

### 2.2 Python LangGraph Runtime 负责什么

1. 把用户请求组装成 `TaskPacket`。
2. 执行显式 LangGraph 图：路由、规划、工具调用、条件重试、审阅和报告。
3. 持久化 checkpoint，支持暂停、审批、重启恢复、取消和回放。
4. 调用 Java 受控工具、统计 Worker、内部知识库和允许的外部证据工具。
5. 输出结构化结果、证据链、局限性、Trace 和风险状态。
6. 执行回归评测并生成 bad-case 报告。

Python **不负责** 自由查询 MySQL、维护用户权限、修改疾病映射或直接向页面暴露数据库内容。

### 2.3 状态与存储边界

| 数据类型 | 存储位置 | 说明 |
|---|---|---|
| 样本、项目、疾病、丰度等业务事实 | 现有远程 MySQL `patient_data_manager` | Java 唯一业务入口 |
| LangGraph checkpoint、任务状态、审批 | 独立 PostgreSQL 数据库 `mico_agent_runtime` | 不污染业务库；用于恢复和审计 |
| 长任务队列与实时进度 | Redis（优先复用现有实例，部署前核验） | 任务分发、取消、SSE 发布 |
| 差异分析结果、图表、可复现报告 | 独立分析产物目录/对象存储 | 必须带 `analysisJobId` 和数据快照 |
| 知识与文献索引 | 独立向量/检索存储 | 不写回业务事实表 |

## 3. 受控调用协议

### 3.1 Java 向 Python 创建任务

前端永远先调用 Java；Java 完成认证、权限与输入校验后，再向 Python 内部服务提交任务。

```json
{
  "runId": "run_...",
  "sessionId": "session_...",
  "user": {"id": "...", "roles": ["researcher"]},
  "question": "比较 2 型糖尿病与健康对照的物种差异",
  "pageContext": {"page": "analysis", "sampleAccession": null},
  "permissions": ["mico:cohort:read", "mico:analysis:submit"],
  "riskPolicy": "research_only",
  "allowedTools": ["resolve_disease", "build_cohort", "submit_differential_analysis", "search_evidence"],
  "dataContractVersion": "v1"
}
```

Python 返回 `runId`、运行状态、SSE 事件和最终结构化报告；Java 将其显示给网页。Python 不接受来自公网或浏览器的裸任务请求。

### 3.2 Python 调用 Java 工具

每一个工具都有确定的 JSON Schema、权限和只读/计算属性。初版工具如下：

| 工具 | 权限 | 行为 | 禁止事项 |
|---|---|---|---|
| `resolve_disease` | `mico:ontology:read` | 查询原始标签到正式映射 | 不创建或修改映射 |
| `get_sample` | `mico:sample:read` | 获取单个样本和项目上下文 | 不返回未授权字段 |
| `build_cohort` | `mico:cohort:read` | 依据固定筛选条件生成队列快照 | 不接受原生 SQL |
| `get_abundance_summary` | `mico:abundance:read` | 返回受限规模的丰度摘要 | 不返回全表扫描结果 |
| `submit_differential_analysis` | `mico:analysis:submit` | 创建异步统计任务 | 不在 Java HTTP 请求中同步跑计算 |
| `get_analysis_artifact` | `mico:analysis:read` | 获取已完成统计产物 | 不改写结果 |
| `search_internal_knowledge` | `mico:knowledge:read` | 检索内部知识 | 不当作实时样本事实 |

所有工具响应必须包含：`toolCallId`、`dataSnapshotId`、`source`、`rowCount`、`schemaVersion`、`generatedAt` 和 `data`。所有拒绝也必须结构化返回原因。

### 3.3 Agent 运行状态

`QUEUED -> RUNNING -> WAITING_APPROVAL | WAITING_JOB -> COMPLETED | FAILED | CANCELLED`

每个节点写入：输入摘要、输出摘要、工具调用、耗时、错误类型、重试次数、数据快照、风险判断和 Trace ID。敏感原文默认脱敏。

## 4. 首个 LangGraph 工作流

第一版只做一个主工作流，禁止一开始扩展成“万能医疗问答”。

**任务**：比较 2 型糖尿病队列与匹配健康对照的物种差异，并检索支持和反方文献证据。

```text
输入安全检查
  -> 术语/疾病标准化
  -> 队列可行性检查
  -> 队列物化（产生数据快照）
  -> 审批或确认队列条件（必要时）
  -> 提交统计任务
  -> 等待并读取统计产物
  -> 内部知识 + 外部文献：支持与反证并行检索
  -> 科学审阅：混杂因素、统计边界、医学安全
  -> 结构化报告与 Trace
```

报告固定分为：

1. 内部数据观察；
2. 队列条件和数据版本；
3. 统计结果；
4. 文献支持与反证；
5. 局限性和下一步；
6. 医学边界声明。

不得输出“该样本患有某疾病”“某菌导致某疾病”“建议患者用药”等结论。

## 5. 实施阶段、产物与验收

### P0：领域语义与数据契约

**范围**：只读审计，不改代码、不改库。

**产物**：五个领域实体定义、字段映射 CSV、质量基线、疾病映射契约、待核验项。

**通过条件**：可用 `SRR1518476`（样本）与 `2015_Castro-NallarE`（项目）证明显示语义；24,264 正确描述为样本记录而非已验证独立患者数。

### P1：Java 受控工具契约

**范围**：在 Java 主项目新增只读、强类型、鉴权的内部 Agent Tool API；不接 LangGraph。

**产物**：OpenAPI/MCP Schema、工具实现、权限策略、快照编号、契约测试。

**通过条件**：不提供自由 SQL；Python 无数据库账号仍可完成允许的事实查询；越权/超量请求被拒绝。

### P2：Python Runtime 骨架与安全桥接

**范围**：新建独立 FastAPI + LangGraph 服务，接入 Java 内部 API。

**产物**：`TaskPacket`、最小 StateGraph、PostgreSQL checkpoint、`agent_run` 状态、SSE、Java BFF 转发、Docker Compose 开发环境。

**通过条件**：一个只读样本查询可跨 Java → Python → Java 完整运行、看到 Trace；服务中断后可恢复；普通 Java 页面不受影响。

### P3：统计工作流与分析产物

**范围**：实现 T2D vs 健康对照队列、异步统计任务、产物版本化。

**产物**：队列构建、统计 Worker、方法和参数记录、结果表/图、报告草稿。

**通过条件**：每个物种结论能回溯至数据快照、统计方法、效应量和 FDR；可重复获得同一产物。

### P4：证据与审阅 Agent

**范围**：内部知识优先、支持/反证文献检索、科学审阅。

**产物**：Evidence Card、引用来源、缺证提示、局限性清单、最终报告 Schema。

**通过条件**：外部证据不会覆盖内部数据；证据不足时明确拒绝强结论。

### P5：安全、审批与治理

**范围**：输入/工具/输出三层 Guardrail，高风险审批，脱敏审计。

**产物**：风险矩阵、审批页面/API、策略测试、脱敏 Trace。

**通过条件**：诊断/治疗要求、提示注入、越权样本访问、批量导出均被安全处理；高风险操作 fail-closed。

### P6：评测、可观测性与 Demo

**范围**：回归 Eval、性能/成本观测、前端证据链、可演示发布。

**产物**：至少 100 条版本化 Eval、bad-case 报告、Trace 页面、三段固定 Demo、架构与运行文档。

**通过条件**：P0 错误（编造事实、越权、输出诊断）为零；数据快照和工具 Trace 完整率 100%；每次改动自动回归。

## 6. 安全与工程红线

1. 永不让 LLM 拼接或执行自由 SQL。
2. 永不让 Python 使用远程 MySQL 的业务写权限。
3. 永不让外部网页或 RAG 文本覆盖样本与统计事实。
4. 永不把样本 accession、项目名或内部记录 ID伪装成人名。
5. 永不把相关性、模型概率或文献关联描述为诊断或因果。
6. 永不删除旧 Java Agent 代码或业务数据，直到 Python Runtime 通过 P6 的回归门槛。
7. 每阶段结束后先提交审计/测试结果，再获得进入下一阶段的确认。

## 7. 一键运行目标

在 P2 之后提供一个项目级启动入口，一次启动：Java 主服务、Python Runtime、必要的 Worker 与前端代理。用户不需要手动分别打开 PowerShell、Java 或 Python 服务。
