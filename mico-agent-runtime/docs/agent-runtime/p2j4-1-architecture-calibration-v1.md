# P2-J4.1A 全项目架构与评测资产一致性校准 v1（历史静态基线）

校准日期：2026-08-23
范围：根计划、Java Agent 契约、Python Runtime、知识库资产、前端/BFF 入口与 P2-J4.1 评测资产。
执行边界：只做静态审计、离线测试、任务集校验和 dry-run；本阶段不启动 Java、Python HTTP 服务、Docker、Neo4j、pgvector、SSH 或任何真实数据库，不执行 `--real`。

> 历史状态说明：本文件记录的是 P2-J4.1A 当时的静态校准边界；其中“真实
> Trace=0/未执行”的行只描述该静态阶段，已被 2026-08-23 的受控 Canary 和
> 50 条真实 Trace 基线取代。当前运行事实、逐条审计和剩余 Bad Case 以
> `p2j4-real-trace-audit-2026-08-23.md` 为准；本文件保留以解释为什么先做
> 校准、Canary，再做全量，而不是删除后丢失决策依据。

> **2026-08-27 当前发布状态补充**：上表中关于“当前没有 SFT/DPO/LoRA”的
> 行是静态校准快照，不能覆盖当前已完成的模型发布事实。Decision SFT Freeze
> v2（865）、Qwen3-8B Decision SFT v5 和 DPO v4 Controlled r2（408 pairs）
> 已完成；DPO adapter、Test70 70/70、OOD30 29/30 和 generation smoke 10/10
> 均有独立产物。当前新增工作是 Dynamic Materialization 推理联调，不是重新训练。

> **2026-08-24 当前事实补充**：静态校准完成后已按该边界完成 Gemini
> `gemini-3.5-flash` 的 50 条真实 Eval 基线。当前有效汇总为
> `evals/p2j4-gemini-full-baseline-final-20260824.json`，50/50 PASS，任务分布
> 10/15/20/5。本文关于“本阶段不执行 `--real`”和“当前不可宣称真实批量 Trace”的
> 文字均是 2026-08-23 静态阶段的历史结论；当前运行细节以
> `p2j4-real-trace-audit-2026-08-23.md` 的 Gemini 基线章节为准。

本次校准将用户给出的开放式科研探索方案作为当前工作基线：

- **数据最小充分性**：样本元数据与样本 × 微生物丰度投影已经足够支撑探索，不为 Agent 新造疾病表、队列表或分析结果业务表；宽矩阵和长表只是同一数据空间的两种表示，当前真实库仍以已核验的丰度长表语义为准。
- **路由分层**：简单事实问题和明确比较问题继续走快速、受控路径；只有“系统探索某疾病/现象、验证稳定性或寻找特异性”等开放问题进入 Scientific Agent Loop。
- **动作冻结**：高层科研能力固定为 `inspect_cohort`、`compare_groups`、`stratified_analysis`、`adjust_confounders`、`cross_project_validate`、`cross_disease_validate`、`retrieve_evidence`、`analyze_projection`、`finish`。`execute_read_query` 是底层 Java 业务读取边界和快速路径适配器，不视为继续扩张高层 Action Space。
- **改进顺序冻结**：先 Trace → Eval → Bad Case，再做 Decision SFT；偏好优化放在 SFT 后，GRPO/RL 暂缓，直到有经过验证且不易被投机的程序化 reward。

## 1. 当前唯一主链

```text
浏览器
  → Java BFF POST /agent/research
  → Python Runtime internal scientific-runs
  → Java describe_read_schema（metadata-only，Catalog prerequisite）
  → LangGraph StateGraph：State → Action → Observation → validate/continue/stop
  → Java execute_read_query（唯一业务数据执行边界）
     或 Python 受限分析 / 独立 vector+graph evidence
  → 脱敏结构化报告
```

Java 仍是 `patient_data_manager` 业务事实唯一来源。Python 的 MySQL 适配器只属于独立的 Runtime 状态库，并受 `MICO_AGENT_RUNTIME_MYSQL_ENABLED=true` 显式开关控制；它不是业务数据访问路径。

早期 T2D/Healthy、固定 `build_cohort`、固定统计脚本和旧 `ai-orchestrator` harness 方案保留为历史资产或兼容测试，不得作为当前开放式 Scientific Agent 路由。当前 Java active Catalog 的总数是 2：`describe_read_schema`、`execute_read_query`；其中只有后者执行业务数据读取。

## 2. 校准矩阵

状态值严格使用：`confirmed`、`partially_confirmed`、`historical`、`superseded`、`contradicted`、`unverified`。

| claim | source_plan_or_doc | actual_code_location | test_or_runtime_evidence | status | contradiction_or_risk | required_correction |
|---|---|---|---|---|---|---|
| 当前不是固定 T2D/Healthy/cohort 链；早期固定链只属于历史示例 | `MICO_LANGGRAPH_COLLABORATION_PLAN.md`；`p2j-scientific-exploration-agent-plan-v1.md` | `mico_agent_runtime/graph/scientific_workflow.py` 的通用 action scope；`knowledge/graph_v3.py` 的 `ENTITY_ALIASES` 仅为知识召回/规范化种子 | Scientific action 联合没有疾病专用 action；P2-J4 v2 任务不绑定固定 SQL | partially_confirmed | 业务主链没有固定分支，但图谱/检索别名仍包含 T2D、Healthy 等静态知识词 | 保留别名作为召回增强；禁止把别名升级为 cohort、Mapper、SQL 或统计分支，并继续在文档中标为 seed/historical |
| Java 是业务事实唯一来源 | 根计划；`p2j0-research-contract-v1.md`；`p2j4-trace-eval-v2.md` | Java `DynamicReadQueryService`；Python `ports/java_agent.py`、`graph/scientific_workflow.py` | Java contract tests；Python action 执行通过 `JavaAgentToolPort`，统一证据用 `java_controlled_read` | confirmed | 独立知识库和 Runtime Store 也有数据库连接，名称相似时容易被误读为业务事实源 | 在所有入口文档中区分 business MySQL、knowledge store、Runtime Store 三个边界 |
| 两类真实数据投影足以支撑开放式科研探索，不需要新增业务表 | `p2j-scientific-exploration-agent-plan-v1.md`；`domain-data-contract-v1.md` | Java Schema Catalog/动态只读查询提供元数据与丰度语义；Python 分析只消费受限投影 | Schema Catalog、动态查询和分析契约测试通过；仓库未新增疾病/队列专用业务表 | confirmed | Schema Catalog 的真实字段覆盖仍需在 P2-J4.2 逐项核验 | 保持元数据优先、先聚合/过滤再分析；宽/长表不改变 Java 事实边界 |
| Python 不直连业务 MySQL | `p2c-intent-query-demo-v1.md`；`p2j-scientific-exploration-agent-plan-v1.md` | Python 只有独立 `storage/mysql_store.py`；业务读取经 `ports/java_agent.py` 和 `ports/schema_catalog.py` | `tests/test_runtime_persistence.py` 检查不持久化 `patient_data_manager`；静态代码没有业务 DataSource/Mapper 路径 | confirmed | Python Runtime 的 MySQL 依赖名可能被误认为业务库依赖 | 保持独立库名/配置校验和开关；不在 Python 新增业务连接或 SSH 逻辑 |
| 简单/明确问题与开放式科研问题分层路由 | `p2c-intent-query-demo-v1.md`；`p2j-scientific-exploration-agent-plan-v1.md` | Intent compatibility path 与 `build_scientific_graph` Scientific path 并存 | P2-J4 v2 同时覆盖事实、比较、探索和安全拒绝任务；Scientific workflow 离线测试通过 | partially_confirmed | 旧 intent graph 仍存在，容易被误写成所有请求都启动 Agent Loop | 保持快速路径；仅开放探索进入多步 Loop，并在任务集与报告中分别统计两类路径 |
| Scientific 高层 Action 集合已冻结 | `p2j-scientific-exploration-agent-plan-v1.md`；`p2j0-research-contract-v1.md` | `contracts/research.py` 的闭合 Action 联合；`execute_read_query` 作为低层 Java 读取动作保留 | Scientific contract/workflow tests 通过；未新增疾病或固定队列 Action | confirmed | 代码联合同时包含 `execute_read_query`，若不区分层级会被误认为高层能力继续增加 | 后续只整理参数、前置条件、决策原因和停止规则，不继续增加高层 Action |
| 当前 active Java Catalog 只有 `describe_read_schema` 与 `execute_read_query` | `p2j0-research-contract-v1.md`；`p2j-scientific-exploration-agent-plan-v1.md` | `mico_database_new/src/main/java/com/database/mico_database/agent/contract/AgentToolCatalog.java`、`AgentToolName.java` | `AgentDynamicReadQueryContractTest` 断言 Catalog size=2、两工具注册、`build_cohort` 未注册 | confirmed | 旧 P1-A/P1-B/P1-C 文档仍列固定工具清单 | 当前引用统一指向本矩阵和 active Catalog；旧契约文档只作安全快照，不重新接入固定工具 |
| 动态 SQL 仍是受控只读，不是自由 SQL | `p2-controlled-dynamic-query-v1.md`；`p5-governance-boundary-v1.md` | `DynamicReadQueryPolicy`；`DynamicReadQueryService` | Java contract test 覆盖危险操作/多语句/跨库/无界 LIMIT；服务校验 catalog、只读连接、超时和最大行数 | confirmed | 关键字策略是安全边界的一部分，不能被描述为任意 SQL 权限 | 继续要求单条 SELECT/WITH、显式 LIMIT、业务 catalog、Java 最终校验和脱敏 receipt |
| Schema Catalog 是当前 Scientific 默认链的前置条件 | `p2j0-research-contract-v1.md`；`p2j-scientific-exploration-agent-plan-v1.md` | `runtime/scientific_service.py` 的 `_load_schema_catalog_if_needed`；`transport/app.py` 的 `ensure_scientific_runtime` | `JavaSchemaCatalogPort` 只调用 `describe_read_schema`；目录失败返回安全错误；Schema Catalog tests 覆盖模型校验 | partially_confirmed | `ScientificRuntime` 构造器仍允许预注入 catalog 或无 catalog/port 的低层实例，类级别不是绝对强制 | 生产 app 继续只通过 `JavaSchemaCatalogPort` 构造；若要形成硬契约，应后续禁止无 catalog 的生产构造并补专门测试 |
| 当前存在真实 LangGraph `StateGraph` | `p2j-scientific-exploration-agent-plan-v1.md`；`p2j0-research-contract-v1.md` | `graph/scientific_workflow.py` 的 `build_scientific_graph`：validate、policy、plan、authorize、execute、validate observation、update、decide、synthesize、terminal | `test_scientific_agent_loop.py`、Scientific workflow tests；`StateGraph` 编译路径可离线运行 | confirmed | 旧 P2-A intent graph 仍存在，容易被误认为当前 Scientific 主图 | 当前文档明确区分 intent compatibility graph 与 Scientific 主图；新增功能只进入 Scientific 主链 |
| Scientific Loop 是 State → Action → Observation → 继续/停止 | `p2j-scientific-exploration-agent-plan-v1.md` | `graph/scientific_state.py`、`graph/scientific_workflow.py` 的 action history、observations、stop reason | Scientific loop tests 覆盖预算、重复动作、观察校验和安全终止 | confirmed | 低层 intent 兼容流程与当前 Scientific loop 并存 | 对外只将 Scientific loop 作为开放科研主链；兼容流程标历史/内部 |
| Python 分析只能在受限沙箱执行 | `p2j-scientific-exploration-agent-plan-v1.md` | `graph/generated_analysis.py` 的 AST validator/sandbox；`_execute_analysis_action` | `test_dynamic_analysis_actions.py`、core dynamic analysis tests；禁止 import、文件、网络、SQL、eval/exec | confirmed | Planner 仍可返回模型代码，但执行前必须过 AST/输出 Schema | 不把生成代码当事实；持续保留行数、方法码、结果上限和 Observation 绑定 |
| vector、graph、Java evidence 可进入统一证据模型 | `p2j-scientific-exploration-agent-plan-v1.md`；`p2j0-research-contract-v1.md` | `contracts/unified_evidence.py`、`merge_unified_evidence`、Scientific report/persistence projection | `test_unified_evidence.py`、`test_runtime_persistence.py`；来源绑定限制为 vector/graph/java | confirmed | 统一模型存在不等于三路真实同时在线；默认配置可回退到 local/deterministic | 把“契约支持三路”与“真实组合已运行”分开报告；后者留给 P2-J4.2 |
| 多跳路径保留 supported/speculative/conflicted 等状态 | `p2g-graph-semantics-v3.md` | `knowledge/database_retriever.py` 的 `_hop_status`/`_path_status`；`knowledge/graph_v3.py` assertion 状态 | GraphRAG/graph v3 tests；合并与生成校验拒绝把 speculative/conflicted 升级为 supported | confirmed | 候选 Taxon 与 speculative 关系仍可能被误解为已证实实体/因果关系 | 对外报告保留 hop/path status、evidence chunk 和 limitation；不做自动实体本体升级 |
| DeepSeek/Gemini/pgvector/Neo4j 的真实与 mock 必须区分 | `p2j3` 联调门槛；`p2i-knowledge-store-deployment-report-v1.md` | `ports/research_planner.py` OpenAI-compatible/DeepSeek URL handling；`knowledge/embeddings.py` Gemini；`knowledge/database_retriever.py` pgvector+Neo4j | `test_scientific_real_integration.py` 是 opt-in skip；知识库报告证明过受控 DB 导入/集成查询，但未证明本次 P2-J4.1 真实批量组合 | partially_confirmed | 代码存在真实 port，离线回归使用 fake/MockTransport/确定性 fallback；外部模型与全部依赖当前没有本阶段运行证据 | P2-J4.2 前逐项记录 endpoint/model/version/DB graph version/权限；real report 必须标真实或 mock，不混用 |
| 65 篇全文、3,617 chunks 是全文证据资产，不是 65 条元数据 | `p2f-vector-graph-database-retrieval-v1.md`；`p2i-knowledge-store-deployment-report-v1.md` | `knowledge/local_retriever.py`；knowledge ingestion/index assets；database retriever | 知识库导入报告与测试记录 65 docs/3,617 chunks；全文证据带 PMCID/chunk/evidenceTier | confirmed | 旧 v2 图谱行已从当前引用移除，仍需防止 v2/v3 概念混淆 | 当前默认检索写 v3；将 65/3,617 解释为全文文档/分块，不把 metadata-only 记录计入全文 |
| Neo4j v3 已发布默认，v4 为 review_pending | `p2g-graph-production-pipeline-v1.md`；`p2i-v4-review-packet-v1.md` | `knowledge/graph_v3.py` 的 `GRAPH_VERSION`；graph registry/publish scripts | P2-I 报告：v3 published、v4 review_pending、280 review queue；v4 不自动切换 | confirmed | v4 staging/import 资产存在，容易被写成 production default | 默认保持 v3；v4 只有人工 review + explicit publish + env version switch 后才可用 |
| Runtime MySQL Schema 已部署，但不等于完整生产接入 | `p2b2b1a-mysql-schema-migration-report-v1.md`；`p2b1-runtime-persistence-contract-v1.md` | `alembic/versions/0001_agent_runtime_state.py`；`storage/mysql_store.py`；`alembic/README.md` | 部署报告记录 `mico_agent_runtime`/0001 已部署；0002 approval migration、app account/keys 仍非本次事实 | partially_confirmed | “Schema deployed”容易被误写为 Runtime app persistence 已启用或 approval lifecycle 已上线 | 继续分列 schema、账号/密钥、应用开关、0002 migration、生产连通性；缺一不可称 production enabled |
| Persistence/recovery/checkpoint 代码存在，但默认未启用且 Store 本身不可恢复 | `p2b1-runtime-persistence-contract-v1.md`；`p5-governance-boundary-v1.md` | `runtime/persistence.py` 的 `build_optional_runtime_persistence`；`runtime/checkpoint.py`；`MySqlRuntimeStore.recoverable=False` | `test_langgraph_checkpoint.py` 证明离线新 saver 可恢复；`test_runtime_persistence.py` 证明加密/安全投影；app 只有显式 env 开关才构造 MySQL coordinator | partially_confirmed | 离线两进程恢复不是生产部署证明；当前 frontend 也没有进度/SSE BFF 验收 | 只能写“可注入/opt-in/离线验证”；P2-J4.2 前确认 0002、账号、密钥、恢复演练和 BFF 生命周期 |
| P2-J4.1 v2 正好 50 条 Golden Case，按 10/15/20/5 分为四类，dry-run 与 real 边界分明 | `p2j4-trace-eval-v2.md`；`evals/p2j4-task-set-v2.json` | `evals/p2j4_runner.py` 的 `validate_task_set`、`_dry_run`、`_real_run` | `--validate` 检查数量/ID/四类分布/敏感字段/动作边界；`--dry-run` 只输出脱敏元数据；real 需要 `--real` + `MICO_P2J4_REAL_RUNS=true` | partially_confirmed | 50 条资产与 real 运行不是同一事实；real 分支本阶段刻意未执行 | 将 50 条、离线校验、dry-run、real requested、real executed 分成四个独立字段；不以 dry-run 代替真实 Trace |
| 历史静态阶段未执行真实 50 条 Trace；当前已有一轮可核验基线 | P2-J4.1A 任务要求；`p2j4-trace-eval-v2.md` | `p2j4_runner.py` 的显式 real 分支；`evals/p2j4-full-valid-20260823.json` | 本文件阶段未执行 `--real`；后续已完成 Canary 3/3 与 50/50 completed 的真实运行 | historical | 历史“真实 Trace=未验证”若被当作当前状态会造成误读 | 当前事实以 `p2j4-real-trace-audit-2026-08-23.md` 为准：原始 18/50 PASS、契约复评分 28/50，尚待修复后再运行 |
| 静态校准阶段尚未训练；当前 SFT/DPO 已完成 | `p2j4-trace-eval-v2.md`；`p2j-scientific-exploration-agent-plan-v1.md` | 历史候选生成器使用 `trainingStarted=False`；正式 DPO 产物位于 `sft-runs/qwen3-8b-decision-dpo-v4-controlled-20260826-r2` | DPO `train_metrics.json` 的 `trainingStarted=True`；adapter、generation smoke、Test70、OOD30 均存在 | historical/current split | 冻结 manifest 的 `trainingStarted=False` 容易被误读为训练未执行 | 不重新训练；将已有 SFT/DPO adapter 接入 Dynamic Materialization Runtime |
| Trace/Eval 已有脱敏闭合契约，但用户要求的决策字段尚未全部落地 | `p2j4-trace-eval-v2.md`；`p2j-scientific-exploration-agent-plan-v1.md` | `TraceDecision` 目前保存 `observationStateCode`、`allowedActions`、`chosenAction`、`decisionCode` 和证据元数据；`EvalScore` 保存 `score`/`criteria` | P2-J4 v2 离线测试通过；Decision Dataset 仅生成 metadata-only candidate，`trainingStarted=false` | partially_confirmed | 当前没有显式 `decision_reason`、`previous_observation_summary`、`next_action`、`agent_version`，也没有同名的 `sql_correct`、`analysis_correct`、`final_score` 字段 | P2-J4.2/训练前扩展脱敏 Trace/Eval schema，并把这些字段映射到固定枚举/标准化摘要；本阶段不把现有 `criteria` 误写成字段已完成 |
| 前端/BFF 已有同步入口，但 progress/events/审批页面不是完成态 | `p5-governance-boundary-v1.md` | Java `AgentResearchBffController` 只有 `POST /agent/research`；`templates/index.html` 提交并渲染同步安全结果；Runtime 内部有 progress/events | Java compile 和 contract tests；静态前端代码存在，未做浏览器端到端/进度 SSE 验收 | partially_confirmed | 代码中不存在 `/agent/research/{runId}/progress` 代理路由 | 文档只写同步 BFF 已接入；将进度 BFF、SSE、审批页面、长任务验收列为后续实现，不修改 Java 以掩盖文档矛盾 |

## 3. 已修正的文档冲突

- 根计划已把 P2-J4.1A 标为当前阶段，把 P2-J3.1、`>=100` 和固定 Demo 目标降为历史/未来语境。
- 旧 Java P1-A/P1-B/P1-C/P2 contract 文档保留，但已加历史快照说明；固定 `build_cohort`/疾病目录不再被误认为 active Catalog。
- P2-A、P2-C、P2-E、P2-G、P2-I、P2-J、P5 中已被替代的阶段快照将在本次校准后移除；当前有效内容统一回收到本计划、P2-J0、P2-F/P2-G v3、部署报告、P5 治理边界和 P2-J4 v2。
- Java 业务代码、旧 Java 安全契约、v1 任务集和知识库事实资产不因文档清理删除；它们仍分别承担回归边界、兼容性或部署证据职责。

## 4. P2-J4.1 v2 评测资产边界

### 4.1 任务集

唯一当前任务集为 `evals/p2j4-task-set-v2.json`，目标是 50 条 Golden Case，固定分布为 A 数据事实 10 条、B 确定分析 15 条、C 开放探索 20 条、D 安全 5 条。要求：case ID 唯一；C 类必须允许多个合法探索路径，并覆盖 metadata 优先、现象发现、稳定性/混杂/特异性验证、知识补证和正确停止；每条任务均有 `allowedActions`、`requiredActions`、`forbiddenActions`、`allowedActionPaths`、`maxActionCount`、`expectedStopReason` 和 `requiresNonDiagnostic`。任务文本、动作和边界不得携带 SQL、凭据、定位器、数据库配置或诊断/治疗结论。

### 4.2 六项核心评分

`score_trace()` 的核心 criteria 固定为：

1. `task_success`
2. `decision_accuracy`
3. `evidence_grounding`
4. `tool_efficiency`
5. `safety_pass`
6. `stop_correctness`

稳定性聚合还输出 `casePassRate`、`decisionAccuracy`、`evidenceGroundingRate`、`safetyPassRate`、`stopCorrectness`、`meanActionCount`、`duplicateActionRate` 和 `fallbackRate`。自由文本不作为评分依据。

### 4.3 Decision Dataset 边界

只有评分 `PASS` 或 Bad Case 已被人工 `ACCEPTED` 的 Trace 才能产生候选。候选包含脱敏
`state_summary`、`decision_reason`、`selected_action`、`alternative_actions`、
`stop_reason`，以及兼容保留的 State/Observation 元数据、允许动作、选择动作和固定
Decision Code；不包含 question、final answer、SQL、arguments、locator、Java payload、
样本定位或文献正文。该候选生成阶段输出显式 `trainingStarted=false`，表示候选
数据本身尚未启动训练；它不代表当前 SFT/DPO 模型发布状态。当前模型状态以
P2-J4 冻结目录对应的正式训练产物和独立评测报告为准。

### 4.4 用户方案对应的当前缺口

用户方案中的“至少 100 个任务、500–1000 个 step、2,000–5,000 个 Decision sample”是历史扩展目标。当前 DPO v4 已使用独立冻结集完成一次正式训练；剩余工作属于新 Dynamic Materialization Runtime 的真实联调和 SFT/DPO 推理对照，不再扩充或重新训练 DPO。

当前 Trace/Eval 的字段状态为：

- 已有：`observationStateCode`、`allowedActions`、`chosenAction`、`decisionCode`、证据路由/绑定计数、`score` 与闭合 `criteria`；
- 已补：脱敏的 `state_summary`、`decision_reason`、`selected_action`、`alternative_actions`、`stop_reason`；
- 待补：`agent_version`，以及将 `sql_correct`、`analysis_correct`、`evidence_grounded`、`tool_efficiency`、`final_score` 明确映射到固定评分契约；本次不把这些尚未实现项伪装成已完成；
- 训练顺序：先完成字段/任务/真实 Trace/Bad Case 闭环，再做 Decision SFT；SFT 稳定后再构造 chosen/rejected 做 Preference Optimization；GRPO/RL 暂不进入当前实施范围。

### 4.5 Safety / release gate

以下任一条件都不能进入真实批量运行或训练候选发布：

- Java Catalog 不可用、响应关联不匹配、快照不是 transient 或包含原始 payload；
- 绕过 Java、直连业务 MySQL、写操作/DDL、无限导出、配置/locator 泄露或诊断/治疗/因果结论；
- Trace 保存 question、SQL、arguments、Token、文献正文，或把 dry-run 标记为 real；
- v4 图谱未完成 review/publish，或外部模型/pgvector/Neo4j 的真实配置未经过独立核验；
- Runtime MySQL/加密 key/approval migration/恢复路径没有满足部署前置条件。

## 5. P2-J4.2 最小前置条件

1. P2-J4.1A 离线五条命令全部通过，并保留输出摘要与工作区 diff；
2. Java internal tool API、Catalog、BFF、Runtime auth 和请求/响应 correlation 由独立 smoke test 验证；
3. 明确并记录 DeepSeek Planner、Gemini embedding/generation、pgvector、Neo4j v3、Java 业务只读账号与 Runtime Store 的真实/模拟状态；
4. 真实 Trace 输出目录只保存脱敏 projection、score、bad case 和安全审计，不保存原始问题、SQL、arguments 或 payload；
5. 先运行 1–3 条小批量真实任务并人工复核，再决定是否扩展到 50 条；任何失败先停止，不自动重试为“成功”；
6. 生产持久化/恢复若要纳入 Trace，必须另外证明 Runtime Schema、0002 lifecycle、账号、密钥、checkpoint saver、resume handler 和 BFF 生命周期已部署。否则保持 in-memory/opt-in 语境。

## 6. 本阶段结论

本节为 2026-08-23 静态阶段结论。按 2026-08-24 补充，当前还可宣称：Gemini 配置下
50 条真实脱敏 Trace 已采集并完成自动 Eval 基线，Java、知识检索路由和 Planner 配置
已在该批次观察；仍不可宣称 Runtime 生产恢复已默认启用、进度/SSE 前端验收已完成、
独立数值 golden oracle 已覆盖全部统计结果，或训练已启动。

## 7. 当前补充：Decision Trace v2（2026-08-24）

表格中“用户要求的决策字段尚未全部落地”是 2026-08-23 的历史快照，现已被实现和
回归结果 supersede。当前 `TraceDecision` 已增加 `state_summary`、`decision_reason`、
`selected_action`、`alternative_actions`、`stop_reason`，同时保留旧的 camelCase 字段
以兼容历史读取器。50 条 Gemini 基线已离线迁移为
`evals/p2j4-gemini-full-baseline-decision-trace-v2-20260824.json`，生成 182 条
`trainingStarted=false` 的候选；该字段只描述候选迁移阶段，没有模型训练调用。当前
SFT/DPO 正式训练状态以冻结集对应的 run metrics、adapter 和评测报告为准。其余
`agent_version` 和明确的 SQL/分析字段映射仍属于后续扩展，不应与本次五字段完成状态混淆。

同日已完成 3 个代表性 Case 各 3 次的真实稳定性小批：9/9 PASS；并生成
`evals/p2j4-task-set-v3.json`，在冻结 v2 50 条的基础上新增 50 条，累计 100 条。
新增任务集仅完成离线契约校验，尚未作为 100 条真实基线执行；新增 Case 的结果 oracle
仍待后续接入。
