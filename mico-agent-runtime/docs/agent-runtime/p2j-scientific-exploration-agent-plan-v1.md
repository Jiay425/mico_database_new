# P2-J：Scientific Exploration Agent 实施计划 v1

状态：当前开放式科研 Agent 的唯一主链设计文档。P2-J0 契约、P2-J1 语义目录桥接、P2-J2 Scientific Agent Loop、P2-J3 通用动态分析与 GraphRAG 证据闭环已完成代码与本地回归；P2-J4.1 的历史 Eval 资产已冻结。Decision SFT Freeze v2（865）与 Qwen3-8B Decision SFT v5 已完成；DPO v4 Controlled Preference Freeze r2（408 pairs）已完成正式训练并保留 adapter，Test70 为 70/70，OOD30 为 29/30，generation smoke 为 10/10。当前 P2-J4.2 工作是将已有模型接入新的 Dynamic Materialization Runtime，尚未完成真实 DeepSeek Flash + Java/MySQL Dynamic E2E。本文中的疾病、队列、物种和比较仅是运行时输入示例，不是系统内置分支。

## 0. 与开放式科研探索方案对齐

本计划以“自然语言问题 → 意图路由 → State/Action/Observation 循环 → 数据证据与文献证据结构化报告”为唯一主线。方案中的 T2D、CRC、肥胖、病例/对照等只属于可替换的任务示例，不是系统内置的业务分支。

允许固定的内容只有：

- Java 是业务事实唯一来源；
- Python 不直连业务 MySQL；
- 工具白名单、scope、只读策略、查询超时、返回上限、分析沙箱和脱敏规则；
- `State → Action → Observation → validate/continue/stop` 的状态机结构；
- transient evidence、审计字段和非诊断边界。

不得固定的内容包括：

- 疾病名称、健康标签、队列定义、比较 ID 或样本集合；
- SQL 的具体字段值、WHERE 条件、分组维度和查询顺序；
- Python 统计方法、特征筛选方法和结论模板；
- “先做什么、再做什么”的单一脚本顺序。

## 1. 目标

将 Mico 从一次性“查询 → 分析 → 回答”升级为面向开放科研问题的、受控的多步 Agent：

```text
Research State
  → Scientific Action
  → Java / Python / Knowledge Tool
  → Observation
  → Next Action 或 FINISH
```

目标不是让模型背诵微生物知识，也不是把业务 MySQL 暴露给 Python。上层 Scientific Agent 决定下一步需要哪种可验证的研究动作；下层受控工具执行数据库读取、统计计算、图谱检索和文献检索。

## 2. 已冻结的数据口径与边界

### 2.1 真实数据形态

逻辑上可视作“样本元数据 + 样本 × 标准化丰度长表”，但实现必须使用 Mico 的真实领域语义。不能把示例中的宽矩阵列（`species_1`、`species_2`）当成真实表结构，也不能把 `patient_id` 当作 Subject 或独立患者数。

| 事实 | 当前口径 |
|---|---|
| `patients.patient_id` | `internalRecordId`，内部业务记录 ID；不是 Subject ID，也不是独立患者人数 |
| 精确丰度谱定位 | `recordProfileLocator=(internalRecordId, sourceSampleId)` |
| 标准丰度 | `microbe_abundance_standard`；20,852,714 是存储行数，不是样本数或唯一物种数 |
| 标准丰度不同样本键 | 23,072 个 `(patient_id, sample_id)` 组合 |
| Meta2DB 元数据 | 13,897 个样本元数据记录；它是标准丰度全库的有富化字段子集 |
| Subject | `raw_metadata.host_subject_id` 仅为未核验的来源候选；默认 `subjectLinkStatus=unverified` |

所有 cohort、统计和报告中的数量必须标注为**内部业务记录数**或**样本键数**。不得把 `patient_id`、样本 accession 或 `host_subject_id` 候选直接解释为独立真实患者。

### 2.2 当前实现状态

当前 Java active Catalog 提供 metadata-only 的 `describe_read_schema` 和一个业务数据执行边界 `execute_read_query`。后者接受模型提出的 SQL，但 Java 侧通过 `DynamicReadQueryPolicy`、只读连接、catalog 检查、超时和行数上限执行；前者只返回版本化 Schema Catalog。Python 不获得 JDBC、Mapper、MySQL 凭据或业务库连接。

当前 Python Runtime 已有：

- `dynamic_read_query`：单次 Java 只读查询与受控 Python 分析；
- `knowledge_retrieval`：pgvector + Neo4j GraphRAG 文献检索与结构化生成；
- transient snapshot、响应关联校验、审计脱敏、GraphRAG 证据路径和人工图谱发布门。

当前已具备通用的 Observation 驱动循环、证据合并和结构化生成边界；P2-J4.1 v2 已建立 50 条任务、评分、Bad Case 和 Decision Dataset candidate 契约，并已采集一轮脱敏真实 Trace。该 Trace 不保存数据值或 SQL，故 PASS 仅表示轨迹、来源、证据绑定和停止契约合格；在补充独立结果 oracle 前，不能把它解释为统计数值已经金标准验证。已有动态意图图和 Scientific Agent Loop 都只能在闭合 Action、Java 只读策略、结果关联校验和预算约束下运行，不能被解释为开放数据库访问。

### 2.3 开放式科研探索的核心方法

本项目不需要为了 Agent 再创建一套疾病表、队列表或分析结果业务表。开放式探索围绕两类真实数据空间展开：

1. **样本元数据空间**：疾病原始标签、年龄、性别、地区、项目、采样部位以及实际可用的其他字段。它用于先判断队列覆盖、项目/地区分布、缺失情况和可比性。
2. **样本 × 微生物丰度空间**：当前真实库是标准化丰度长表；逻辑上等价于样本行与 feature 列组成的矩阵。它用于在元数据形成可解释的样本集合后执行特征级分析。任何大规模丰度读取都必须先在 Java 侧聚合、过滤并限制结果，不能把千万级原始行交给 LLM。

典型开放任务的动态路径不是固定脚本，而是由 Observation 驱动的可变探索，例如：

```text
用户自然语言问题
  → 意图识别与任务分类
  → 先检查元数据覆盖、项目/地区/年龄/性别和缺失
  → 根据观察选择可比的分层或验证动作
  → 在已确认样本集合上进行丰度分析
  → 按项目/地区/性别或其他可用维度检查稳定性
  → 评估年龄、性别、项目等混杂因素
  → 在其他疾病或队列上检查特异性/可迁移性
  → 将候选特征送入 GraphRAG + 文献向量检索补证
  → 统一重排、绑定数据与文献来源
  → 结构化科研报告
```

上面的顺序是**可解释的探索示例**，不是固定的节点链。每一步都遵循：

```text
State → Action → Tool → Observation → 验证 / 继续 / 停止
```

固定的是能力边界，例如 `execute_read_query`、`analyze_projection`、`retrieve_evidence`、`finish`；动态的是用户主题、查询字段与条件、分组维度、分析方法、验证顺序、是否并行调用 Vector/Graph/Java，以及何时停止。当前冻结的高层科研动作是：`inspect_cohort`、`compare_groups`、`stratified_analysis`、`adjust_confounders`、`cross_project_validate`、`cross_disease_validate`、`retrieve_evidence`、`analyze_projection`、`finish`。`execute_read_query` 是底层 Java 业务读取边界和快速路径适配器，不是继续扩张高层 Action Space。这些名称只表示通用能力，不能绑定 T2D、健康对照或任何固定 SQL/Python。

问题复杂度决定路由深度：

```text
简单事实问题       → Java 动态只读查询 → 直接回答
明确比较问题       → 动态数据查询 → 受限 Python 分析
文献/关系问题      → Vector + GraphRAG → 来源绑定报告
开放科研探索       → Scientific Agent Loop → 多轮验证 → GraphRAG/RAG → 报告
综合问题           → 按依赖关系并行或串行合并三类证据
```

训练和评测也不应让模型记忆某个疾病的固定流程。应先保存真实 `State → Action → Observation` 轨迹，再评估下一步动作是否完成了队列可比性检查、跨项目验证、混杂审查、证据补充和正确停止；稳定后才从高质量轨迹构建 SFT/偏好数据。

### 2.4 绝对边界

1. Java 是业务事实源；Python 不直连 `patient_data_manager`。
2. 不新建业务 MySQL 表，不把原始样本数据复制到 LLM 上下文。
3. 任何模型输出都必须先转换为闭合契约，并由 Runtime policy 与 Java 工具边界校验。
4. 动态 SQL/Python 是高级受控能力，不是 Scientific Agent 的默认动作语言。
5. 任何医学陈述只可作为来源绑定的科研证据描述；不得输出诊断、因果结论或治疗建议。

## 3. 目标路由

```text
自然语言问题
  ↓
LangGraph 意图路由
  ├─ data_fact
  │    └─ Java execute_read_query → 直接、受限的数据事实回答
  ├─ focused_comparison
  │    └─ LLM 生成当前问题所需的动态只读查询 → Python 受限分析
  ├─ literature_or_relationship
  │    └─ pgvector + Neo4j 并行检索 → hybrid rerank → grounded report
  └─ scientific_exploration
       └─ State → Action → Observation 循环
    ├─ 动态数据查询 / 验证
    ├─ Python 生成式受限分析
    ├─ GraphRAG / 文献补证
    └─ FINISH
```

对于综合问题，vector 与 graph 分支可并行；只有与其无依赖的数据事实查询可并行。Python 分析必须等待 Java 返回已校验、受限的数据投影，不能抢跑或直接读取业务数据库。

## 4. 动态科研动作契约

科研内容不在 Java 中预注册为疾病、队列或统计方法。Runtime 只固定执行能力，模型根据脱敏问题和当前 Observation 选择当前动作，并由闭合契约和策略校验：

| 动作能力 | 输入来源 | 执行边界 |
|---|---|---|
| `execute_read_query` | 模型生成的当前问题 SQL 草案 | 仅发送 Java；Java 最终校验 SELECT/WITH、列、跨库、行数和超时 |
| `inspect_cohort` | 模型生成的元数据探查 SQL 草案 | 作为元数据优先入口；仍只调用 Java 动态只读工具，不直接读取数据库 |
| `compare_groups` | 已验证 Observation + 动态分析目标 | Python 受限分析；分组定义必须来自当前 Observation，不注册疾病或队列 |
| `stratified_analysis` | 已验证 Observation + 动态维度 | Python 受限分层分析；维度必须来自语义目录和已返回投影 |
| `adjust_confounders` | 已验证 Observation + 动态混杂字段 | Python 受限校正/敏感性分析；不固化年龄、性别或项目字段 |
| `cross_project_validate` | 多个已验证 Observation + 当前目标 | Python 受限跨项目稳定性检查；项目只是运行时字段，不是固定分支 |
| `cross_disease_validate` | 多个已验证 Observation + 当前目标 | Python 受限跨疾病验证；未知标签不被自动标准化为正式疾病 |
| `retrieve_evidence` | 模型生成的脱敏主题摘要 | pgvector/Neo4j GraphRAG；固定 topK、跳数和来源绑定 |
| `analyze_projection` | 已验证 Observation + 当前分析目标 | Python 沙箱生成/执行受限代码，不访问数据库或网络 |
| `finish` | 证据充分、质量风险或预算耗尽 | 固定停止码和结构化局限性 |

SQL 是唯一允许携带查询文本的专用动作，不能由浏览器、普通文本字段或其他动作旁路注入。固定的是安全策略、权限、预算、超时、行数和脱敏规则，不是疾病、项目、队列或分析脚本。高层操作名称只表示能力类别，不能内含某个疾病或某种统计假设。

## 4.1 Schema Semantic Catalog

模型要动态生成查询，必须先获得“可查询语义”，而不是把业务问题写死在 Prompt 中。后续应由 Java 基于已核验 Mapper/DTO/数据库结构提供版本化的只读语义目录，至少描述：

- 可用实体、字段、字段类型、缺失语义和展示禁区；
- `internalRecordId`、`sourceSampleId`、`sampleAccession`、`studyKey` 的不同含义；
- 允许的关联键和关联方向；
- 丰度长表中的 feature、数值、版本和批次字段；
- 哪些字段可用于过滤、分组、聚合或排序；
- 行数、特征数和结果预览的边界。

目录是数据库事实的安全描述，不是疾病/队列知识库。它应带 `schemaVersion`、来源和生成时间；Python 只接收目录的脱敏投影，不能读取 Java 配置或数据库。

## 5. 闭合状态模型

P2-J0 需要定义以下版本化 Pydantic 契约：

```text
ResearchTask
ResearchState
ScientificAction
Observation
ResearchFinding
ResearchEvidenceBinding
StopDecision
ResearchTrace
ResearchExplorationReport
```

`Observation` 必须至少携带：

```text
source
queryHash
rowCount
generatedAt
schemaVersion
dataSnapshotId
snapshotPersistence
featureVersion
taxonomyVersion
sourceBatch
sampleRecordCount
sampleKeyCount
missingnessSummary
exclusionSummary
```

所有证据的 `snapshotPersistence=transient` 必须继续显式标记为不可跨进程、跨时间回放。没有可靠物理来源的导入批次、疾病映射版本或 taxonomy 版本必须保持 `null`，不得编造。

## 6. LangGraph 科研循环

目标图：

```text
validate_research_task
  → policy_gate
  → initialize_research_state
  → plan_next_action
  → authorize_action
  → execute_scientific_action
  → validate_observation
  → update_findings
  → decide_continue_or_stop
       ├─ continue → plan_next_action
       ├─ retrieve_evidence → execute_scientific_action
       └─ finish → synthesize_research_report
  → terminal
```

`plan_next_action` 允许当前配置的 Planner（当前真实基线使用 Gemini OpenAI-compatible 的
`gemini-3.5-flash`）选择动作和短的、受控的决策理由，但不能生成：SQL、表名、列名、数据库
URL、样本定位器、疾病原始标签、feature version、limit、权限或工具白名单。

每一轮都必须校验：

1. Action 属于 catalog，且参数满足闭合 schema；
2. Action 的 scope、预算和前置 Observation 满足要求；
3. Java 响应 `runId` / `toolCallId` 与本轮调用精确一致；
4. Observation 的版本、行数和质量状态可解释；
5. 模型不能把 `supported`、`speculative`、`conflicted` 关系升级为更强结论。

默认治理规则：总动作预算 6–8 次；相同 Action + 规范化参数签名不得重复；无新增信息、证据不足、映射未审核、质量风险过高或预算耗尽时必须 `finish`。

## 7. 通用统计执行器

P2-J 不是编写一个固定的“T2D 对 Healthy”脚本。其统计层应是面向任意已批准 cohort 的通用执行器：

```text
compare_groups
stratified_analysis
project_consistency
confounder_robustness
disease_specificity
continuous_variable_trend
microbial_module_summary
```

每个统计动作要绑定方法版本、过滤阈值、缺失策略、特征宇宙、FDR/效应量规则和结果快照。只有这样，数据发现才可复查、可比较、可用于 Eval。

动态 Python 可以继续作为隔离的派生分析能力，但只能读取 Java 已批准的有界投影，并经过 AST、资源、输出 schema 与脱敏检查。核心差异分析、FDR、置信区间和混杂处理不得依赖模型自由生成代码。

## 8. 数据规模与性能策略

1. 不将 20,852,714 条丰度记录传入 Python 或 LLM。
2. Java 对模型 SQL 做只读策略校验，并在数据库侧完成模型请求的过滤、聚合和限制。
3. Python 仅接收受限结果预览或明确上限的投影。
4. 多步探索由 Observation 驱动；下一次查询必须来自新问题状态和已验证观察，不能凭空拼接定位条件。
5. 宽范围请求需要先由模型提出分步计划，Runtime 依据预算、成本和结果质量决定继续或停止；不得在 Java 中新增某个疾病的专用 SQL 分支。

## 9. GraphRAG 与结构化证据

当数据侧出现候选特征后，才进入知识补证：

```text
ResearchFinding
  → vector retrieval + graph traversal
  → unified rerank
  → evidenceId / reasoningPathId binding
  → grounded generation
```

每个最终 Finding 必须保留：

```text
内部数据发现和统计版本
数据 snapshot / queryHash
文献 evidenceId
GraphRAG reasoningPathId
supported / speculative / conflicted 状态
固定科研局限性与非诊断声明
```

Graph 路径用于可审计的推理结构，不输出模型隐藏思维链。低置信度、候选 taxon 或冲突关系必须保留风险状态，不可作为确证结论。

## 10. Trace、Eval 与后训练

先建立 Trace 与 Eval，再开始任何 LoRA、DPO 或 RL：

```text
Research Task
  → State
  → Chosen Action
  → Observation
  → Next State
  → Final Report
  → Programmatic + human evaluation
```

Runtime 独立状态库中的 `agent_run`、`agent_step`、`tool_audit`、`agent_artifact` 用于保存加密状态、动作元数据、快照关联和脱敏结果。不得在普通列、审计日志或训练样本中保存原始 locator、Java payload、自由 warning、Token 或连接信息。

P2-J Eval 至少衡量：

```text
任务完成度
cohort 口径正确性
项目/地区/部位可比性检查
跨项目稳定性验证
混杂因素处理
疾病映射与排除规则
证据完整性
相关性与因果边界
正确停止
工具调用次数与重复调用
```

后训练顺序冻结为：

```text
高质量 Trace → LoRA SFT
同 State 的 chosen / rejected Action → Preference Optimization
Base vs SFT vs Preference Eval
```

GRPO/RL 暂缓，直到存在经过验证、难以被投机的程序化 reward。

当前实施顺序进一步明确为：

```text
50 条真实 Gemini 基线
  → Decision Trace v2 脱敏字段
  → 3 个代表性 Case × 3 次稳定性
  → 新增 50 条，累计 100 个任务
  → 新任务真实 Trace / 自动评分 / Bad Case 回流
  → Decision candidate 审核
  → Decision SFT
  → chosen / rejected Preference Optimization
```

用户方案中的 500–1000 个 step、2,000–5,000 个 Decision sample 是历史扩展目标，不应覆盖当前已冻结的 SFT/DPO 资产事实。DPO v4 冻结目录的 `trainingStarted=false` 只表示该 manifest 记录的是冻结/参考日志阶段；正式训练证据以 `sft-runs/qwen3-8b-decision-dpo-v4-controlled-20260826-r2/train_metrics.json`、adapter、generation smoke、Test70 和 OOD30 报告为准。当前不重新训练 DPO，下一步是 Dynamic Materialization E2E。

## 11. 实施阶段与验收

### P2-J0：契约与 pilot 范围

- 新增闭合 Research contract、Action catalog、Observation / Finding / Trace schema；
- 确定通用问题分类、动作预算、来源和安全边界；不在 Runtime 中预注册某个疾病或固定队列；
- 将 `patient_id`、样本记录、样本键和 Subject 候选的展示规则写入所有结果 schema；
- 定义动作预算、停止码与安全错误码。

验收：LangGraph 可编译；未知 Action、未审核疾病、无 version 的 Observation、Subject 误用均 fail-closed。

### P2-J1：动态数据动作桥接与语义目录

- 保留唯一 Java 业务数据工具 `execute_read_query`；
- 提供版本化、脱敏的 Schema Semantic Catalog；
- 将脱敏自然语言问题交给模型生成闭合 SQL 草案；
- Java 只执行通过 `DynamicReadQueryPolicy` 的参数化/受控只读请求；
- Python 根据结果预览生成受限分析代码，所有输出仍绑定 transient evidence。

验收：任意业务主题都走同一动态边界；没有 T2D、健康、疾病或固定队列专用代码；模型生成 SQL 前能够获得真实字段语义和关联边界。

### P2-J2：Scientific Agent Loop

- 实现多轮 LangGraph、Action 授权、Observation 校验、去重、预算与 FINISH；
- 接入可替换的 Planner 仅作闭合动作选择；当前真实基线使用 Gemini，模型异常或明确语义路径
  回退 deterministic 策略；
- 扩展 FastAPI 内部端点，但不向浏览器直接暴露 Python。

验收：不同自然语言研究问题能够在动态查询、动态分析、文献/图谱检索和停止之间作出可审计的分支选择；下一步由 Observation 驱动，不存在固定的 T2D 分析脚本。

### P2-J3：通用统计与 GraphRAG 证据闭环

- 实现通用 `compare_groups`、`cross_project_validate`、`adjust_confounders`、`cross_disease_validate`；
- 将候选特征转入现有 vector / graph hybrid retrieval；
- 输出带 data evidence 与 literature evidence 绑定的 `ResearchExplorationReport`。

验收：一个结果中能清楚区分数据发现、跨项目验证、混杂风险、图谱关系、文献证据与局限性。

### P2-J3.1：真实组合联调（历史下一步，尚未批量执行）

- 使用已发布的 v3 知识图谱和现有全文向量资产；不切换 review_pending 的 v4；
- 通过 FastAPI 内部入口运行真实 ScientificRuntime；
- 验证 Java Schema Catalog、动态只读查询、Python 沙箱、pgvector、Neo4j GraphRAG、统一重排和结构化生成的组合链路；
- 真实环境未配置时只保留 opt-in 集成测试并报告缺口，不把 fake/mock 结果宣称为真实联调；
- 原计划在完成后进入 P2-J4；当前 P2-J4.1 的离线资产已先建立，但不代表该真实组合已执行。

### P2-J4：Trace 与 Eval

- 保存脱敏科研轨迹；
- 建立任务集、自动评分、bad case 与人工复核流；
- 完成 Base Agent 多次运行稳定性基线。

验收：能够筛选高质量轨迹、定位错误停止/重复调用/错误 cohort，并产出 SFT 与偏好数据候选。

### 后续：SFT → Preference

在积累约 500–1000 个可评估科研任务、且 Eval 稳定后再开始 LoRA SFT；Preference Optimization 只在有可信 chosen/rejected 决策对后进行。

## 12. 端到端验收原则

验收问题不绑定某个疾病或队列。应使用多类自然语言任务验证同一动态链路：

```text
自然语言问题
  → 意图路由
  → 模型生成当前步骤的 SQL / 检索主题 / 分析目标
  → Java 或知识工具执行
  → Observation
  → 验证、继续或 FINISH
  → 数据证据 + 文献证据结构化报告
```

每个具体问题都必须重新生成计划；任何模型都不能改变工具白名单、权限、SQL 只读策略、上限、脱敏规则或非诊断边界。
