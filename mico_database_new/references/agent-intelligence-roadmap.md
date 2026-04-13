# 智能体能力升级路线图

## 1. 文档目的

本文档用于固定当前项目的智能体升级方案，避免后续迭代时遗忘设计原则、实施顺序和模块边界。

适用项目：

- 主业务项目：`E:\DeskTop\java_code\mico_database_new`
- AI 编排服务：`E:\DeskTop\java_code\mico_ai_service\ai-orchestrator`

核心目标：

- 让 AI 助手不再只复述页面内容，而是具备“数据分析 + 健康对照 + 预测解释 + 知识检索 + 外部证据”综合能力
- 保持业务真相、模型真相、外部知识三层边界清晰
- 逐步从“单工作流助手”升级到“混合式 agent”再到“多智能体系统”

---

## 2. 当前基础能力

当前系统已经具备以下基础：

### 2.1 主项目能力

- 患者检索、详情、样本切换、预测页工作台
- 标准丰度表 `microbe_abundance_standard`
- 健康组对照接口
- 预测接口与模型服务代理接口
- 首页 dashboard 统计接口

### 2.2 AI 编排能力

- 独立的 `ai-orchestrator` 服务
- LangChain4j 接入
- 患者页、首页、预测页已经有 AI 助手入口
- 当前工作流已支持：
  - 患者详情分析
  - 健康组对照
  - 与预测结果联动解释

### 2.3 当前主要短板

- AI 解释深度仍然有限
- 目前主要依赖接口数据，没有系统化知识库
- 没有正式的 RAG 检索链路
- 没有 MCP 标准化能力层
- 没有联网证据检索工具
- 没有真正的多智能体分工

---

## 3. 总体原则

### 3.1 三层真相分离

系统未来所有回答，应尽量区分三类信息来源：

1. 内部业务数据
   - 患者、样本、健康对照、dashboard 统计
2. 模型判断
   - 预测标签、概率、解释边界、模型限制
3. 外部知识证据
   - 医学知识库、文献、官方网页、联网搜索结果

禁止把这三层内容混在一起输出成“像事实但没有来源边界”的回答。

### 3.2 先深后广

优先把单条患者分析链路做深，再扩展成更多工具和更多 agent。

不要一开始就同时堆：

- 知识库
- MCP
- 联网搜索
- 多智能体

正确顺序应为：

1. 知识库
2. RAG
3. MCP
4. 混合式单总控 agent
5. 多智能体

### 3.3 联网搜索不等于主知识库

联网搜索得到的内容属于“临时外部证据”，默认不直接写入主知识库。

外部搜索内容必须带：

- 标题
- 来源链接
- 发布时间或更新时间
- 摘要

---

## 4. 分阶段实施路线

## 阶段 1：知识库

目标：先建立项目内部可控、可复用的知识资产。

### 4.1 业务库

内容范围：

- 项目字段说明
- `Group` 与疾病标签映射
- 页面分析模板
- 患者页、预测页、首页的标准解读框架
- 样本、样本对照、健康组对照的业务定义

建议文档：

- `references/knowledge/business/field-dictionary.md`
- `references/knowledge/business/group-mapping.md`
- `references/knowledge/business/page-analysis-templates.md`
- `references/knowledge/business/workflow-definitions.md`

### 4.2 医学库

内容范围：

- 疾病简介
- 健康组对照解释规则
- 常见微生物的简要释义
- 微生物异常时的通用解释边界

建议文档：

- `references/knowledge/medical/disease-overview.md`
- `references/knowledge/medical/healthy-baseline-rules.md`
- `references/knowledge/medical/common-microbes.md`
- `references/knowledge/medical/interpretation-boundaries.md`

### 4.3 模型库

内容范围：

- 训练特征空间说明
- 标签映射
- 模型输出解释边界
- 不能过度解释的规则
- 预测结果如何与健康对照联动解释

建议文档：

- `references/knowledge/model/label-definition.md`
- `references/knowledge/model/feature-space.md`
- `references/knowledge/model/prediction-boundaries.md`
- `references/knowledge/model/prediction-explainer-template.md`

### 4.4 阶段 1 交付标准

- 3 个库目录结构固定
- 文档均为中文专业表达
- 允许 AI 编排服务后续直接读取
- 内容优先保证“准确、可执行、可引用”

---

## 阶段 2：RAG 检索

目标：让 `ai-orchestrator` 在回答问题时，不只用接口数据，还能检索内部知识库。

### 5.1 检索目标

未来患者问题的回答应综合：

- 患者数据
- 样本特征
- 健康组基线
- 预测结果
- 知识库证据

### 5.2 检索优先级

优先顺序：

1. 业务库
2. 模型库
3. 医学库

原因：

- 业务规则和模型规则属于项目内部权威知识
- 医学库主要用于辅助解释，不应压过项目内部定义

### 5.3 RAG 在患者问答中的定位

患者页提问时，未来回答结构建议固定为：

1. 当前样本观察
2. 与健康组的偏离
3. 与预测结果是否同向
4. 知识库中可支持的解释
5. 下一步建议

### 5.4 向量库建议

第一版建议以“可快速落地”为优先：

- 先做本地可控、轻量方案
- 不优先引入过重的独立基础设施

建议路线：

1. 第一版：本地文档切分 + 嵌入 + 轻量向量存储
2. 第二版：如后续规模增大，再迁移到独立向量数据库

### 5.5 阶段 2 交付标准

- `ai-orchestrator` 可检索 3 个知识库
- 患者页问题可引用知识库内容
- 输出中能区分“业务规则”“模型边界”“医学解释”

---

## 阶段 3：联网搜索证据工具

目标：补充系统内部知识库之外的外部医学或行业证据。

### 6.1 角色定位

联网搜索不是主回答来源，而是外部证据补充层。

适合的问题：

- 某疾病的最新公共资料
- 某微生物与疾病的公开研究方向
- 指南、共识、科研机构说明

### 6.2 联网搜索的输出要求

必须返回：

- 标题
- 来源
- 链接
- 日期
- 简短摘要

必须避免：

- 无来源断言
- 把网页内容直接当作项目内部事实
- 用外部搜索结果覆盖模型或数据库真相

### 6.3 联网搜索的系统定位

建议未来作为独立能力存在：

- RAG 负责内部知识
- Web Search Tool 负责外部证据

不建议把联网抓到的结果直接自动写入内部知识库。

### 6.4 阶段 3 交付标准

- AI 可调用网页搜索
- 回答中能显式区分“内部结论”和“外部证据”
- 外部证据均带来源链接

---

## 阶段 4：MCP 服务

目标：把项目里的数据能力、知识能力、分析模板能力标准化暴露出来。

### 7.1 设计原则

MCP 不只是做接口搬运，而是把能力抽象成：

- `tools`
- `resources`
- `prompts`

### 7.2 计划中的 MCP Tools

- `get_patient_summary`
- `get_sample_top_features`
- `get_healthy_reference`
- `run_prediction`
- `get_dashboard_summary`
- `search_knowledge`
- `search_web_evidence`

### 7.3 计划中的 MCP Resources

- `business://field-dictionary`
- `business://group-mapping`
- `business://page-analysis-template`
- `model://label-definition`
- `model://prediction-boundaries`
- `medical://healthy-baseline-rules`

### 7.4 计划中的 MCP Prompts

- `patient_vs_healthy_analysis`
- `prediction_consistency_explainer`
- `dashboard_briefing`
- `sample_abnormality_review`

### 7.5 MCP 的业务意义

引入 MCP 之后，系统中的 AI 不需要只靠硬编码 workflow 才能做分析，还可以通过标准协议使用：

- 数据工具
- 知识资源
- 分析模板

这会让后续接入更多 agent 或更多模型时更稳。

### 7.6 阶段 4 交付标准

- MCP Server 可独立运行
- tools/resources/prompts 均能被调用
- 与当前 `ai-orchestrator` 解耦但可协作

---

## 阶段 5：混合式单总控 Agent

目标：先做“一个总控 agent + 多种工具能力”的深度编排，而不是马上拆多智能体。

### 8.1 混合式 agent 的工具构成

未来患者分析问题，建议总控 agent 同时编排：

- 业务数据工具
- 健康对照工具
- 预测工具
- RAG 检索工具
- 联网证据工具

### 8.2 推荐回答结构

对于患者分析类问题，统一按这 5 段生成：

1. 当前样本概况
2. 与健康组偏离
3. 模型预测方向
4. 知识库或外部证据支持
5. 下一步建议

### 8.3 阶段 5 交付标准

- AI 回答不再只复述接口字段
- 能综合多种工具做结论
- 回答具有来源层次感

---

## 阶段 6：多智能体

目标：在前面能力稳定后，再拆分角色，提升系统复杂任务处理能力。

### 9.1 推荐角色

- 数据分析 agent
  - 负责患者、样本、健康组、dashboard 数据理解
- 知识检索 agent
  - 负责内部知识库检索、联网证据检索
- 预测解释 agent
  - 负责模型结果、标签、概率、边界解释
- 总控 agent
  - 负责任务分发、结果整合、最终输出

### 9.2 不建议过早多智能体化的原因

- 工具边界未稳定前，多智能体只会放大混乱
- 知识库未建好前，多智能体依然会浅层复述
- 缺乏统一证据层时，多智能体容易给出相互矛盾的答案

### 9.3 阶段 6 交付标准

- 多 agent 有明确分工
- 工具与知识调用路径稳定
- 总控 agent 只做编排，不重复业务逻辑

---

## 10. 推荐实施顺序

实际执行时，按下面顺序推进：

1. 先建立 3 个知识库目录与首批内容
2. 在 `ai-orchestrator` 中接入第一版 RAG
3. 增加联网搜索证据工具
4. 设计并落地 MCP Server
5. 将当前患者分析工作流升级为混合式单总控 agent
6. 最后再拆多智能体

---

## 11. 第一批优先落地内容

最先落地的内容建议是：

### 11.1 业务库优先文档

- Group 与疾病映射
- 页面分析模板
- 字段说明

### 11.2 医学库优先文档

- 健康组对照解释规则
- 常见高频微生物释义
- 疾病简要介绍

### 11.3 模型库优先文档

- 预测标签定义
- 模型解释边界
- 特征空间与零值补齐规则

---

## 12. 后续执行约定

从本文件创建后，后续与智能体升级有关的设计和开发，应优先遵守本文档。

如果后续方案有变化，应直接修改本文档，而不是只在聊天中口头约定。

建议后续每完成一个阶段，就在本文档末尾追加：

- 已完成项
- 当前状态
- 下一步动作

---

## 13. 当前结论

本项目的 AI 升级路线，确定为：

`知识库 -> RAG -> 联网搜索证据工具 -> MCP -> 混合式单总控 agent -> 多智能体`

这条路线兼顾了：

- 实施稳定性
- 系统可维护性
- 医疗场景的解释严谨性
- 后续高级能力扩展空间

---

## 14. 当前实施进度

### 已完成

- 第 1 阶段知识库已落地
  - 已建立 `business / medical / model` 三个知识库目录
  - 已写入首批业务规则、医学解释、模型边界文档
- 第 2 阶段第一版 RAG 已接入 `ai-orchestrator`
  - 已支持读取 `references/knowledge` 本地知识库
  - 已支持在患者分析链路中补充内部知识库证据
  - 已在 `/api/ai/health` 中暴露知识库加载状态

### 当前状态

- 当前 RAG 为第一版内部知识检索
- 检索范围：业务库、医学库、模型库
- 当前优先接入场景：患者分析问答
- 当前目标：先把知识证据稳定接进现有工作流，再继续做联网搜索与 MCP

### 下一步建议

1. 扩展到预测页和首页的知识检索增强
2. 加入联网搜索证据工具
3. 设计 MCP Server 的 tools / resources / prompts
- 第 2 阶段第一版 RAG 已扩展到首页与预测页
  - 首页问答已支持结合业务库与医学库做数据重点解释
  - 预测问答已支持结合模型库做方向与边界解释
- 第 3 阶段联网搜索证据工具已接入框架
  - 已新增独立外部证据通道，不与内部知识库混用
  - 已支持在首页、患者页、预测页附加外部网页证据卡
  - 当前需配置搜索服务密钥后才能启用实时联网搜索
- 第 4 阶段 MCP Server 第一版已落地
  - 已新增 `/mcp` JSON-RPC 入口
  - 已暴露 tools/list、tools/call、resources/list、resources/read、prompts/list、prompts/get、initialize
  - 已接入患者摘要、样本特征、健康组对照、预测、首页摘要、知识检索、外部证据检索能力
- 第 5 阶段混合式单总控 agent 已落地第一版
  - 已将业务数据、模型判断、内部知识、外部证据统一整理为分层结构
  - 已在回答元数据中加入 `analysisLayers` 与 `evidenceChannels`
  - 已由单总控层统一组织首页、患者页、预测页的增强结果
- 第 6 阶段多智能体已落地第一版
  - 已拆分为数据分析 agent、知识检索 agent、预测解释 agent、总控 agent
  - 已在元数据中加入 `activeAgents` 与 `agentInsights`
  - 已由总控 agent 汇总多智能体输出作为最终回答

## 2026-03-31 Harness Runtime Progress

- 已补齐 system of record 文档：`ARCHITECTURE-AGENT.md`、`EVALS.md`、`AGENT-RUNBOOK.md`
- 已补齐可读的 harness eval 样例：患者页、预测页、首页三类套件
- `ai-orchestrator` 已接入 Task Packet、Execution Policy、Evidence Policy
- `ai-orchestrator` 已接入 Trace 与基础 Review Loop
- 已开放 trace 查询接口：`/api/ai/traces/{sessionId}`、`/api/ai/traces/{sessionId}/latest`
- 已补最小 eval runner：`scripts/run_harness_eval.py`
- 下一步优先项：
  - 让 MCP 不只暴露能力，还暴露 harness runtime 的 resources/prompts
  - 给 eval runner 增加更细的评分和 bad case 汇总
  - 在 trace 中补充 resources 命中与 MCP 调用记录

## 2026-03-31 MCP Harness Connector Progress

- MCP 已从单纯能力接口升级为 Harness Connector
- 已新增 harness 级 resources：
  - `harness://architecture-agent`
  - `harness://evals-spec`
  - `harness://agent-runbook`
  - `harness://runtime-status`
  - `harness://evidence-policy`
- 已新增 harness 级 tools：
  - `get_runtime_trace`
  - `get_harness_status`
- 已新增 harness 级 prompts：
  - `task_packet_review`
  - `runtime_trace_review`
  - `evidence_layer_audit`
- 下一步优先项：
  - 让 trace 记录 MCP resources/prompts 命中路径
  - 细化 eval runner 的评分与 bad case 汇总
  - 让外部评测器通过 MCP 直接拉取 runtime 状态与回归模板

## 2026-03-31 Evals Harness Progress

- `scripts/run_harness_eval.py` 已升级为回答内容 + runtime trace 双检查
- 已引入 P0 / P1 / P2 分级
- 已增加 bad case 自动汇总输出：`references/harness-evals/latest-bad-cases.json`
- 评测报告会同时记录：
  - must / should / forbidden 命中情况
  - trace 核心步骤是否完整
  - review 是否存在
  - 知识层 / 预测层是否实际命中
- 下一步优先项：
  - 在 trace 中补 MCP resources/prompts 命中记录
  - 让 bad case 报告增加修复建议字段
  - 支持按 session 或按页面维度做回归汇总

## 2026-03-31 MCP Trace Progress

- trace 已支持记录 MCP `tool/resource/prompt` 命中
- MCP 请求可通过 `sessionId` 关联到已有 runtime trace
- review 中已加入：
  - `usedMcpTools`
  - `usedMcpResources`
  - `usedMcpPrompts`
  - `mcpUsage`
- 下一步优先项：
  - 在 eval runner 中检查 MCP usage 是否符合预期
  - 给 bad case 自动补充“缺失的是哪类 MCP 命中”
  - 为关键 MCP prompts 增加更明确的使用场景约束

## 2026-03-31 MCP Eval Progress

- eval runner 已支持 `mcp_plan`
- eval runner 已支持检查：
  - `expected_mcp_usage`
  - `expected_mcp_steps`
- 新增 suite：`references/harness-evals/harness-connector.jsonl`
- 现在可以回归验证：
  - MCP tool 命中
  - MCP resource 命中
  - MCP prompt 命中
- 下一步优先项：
  - 为 bad case 自动补“建议修哪一层”
  - 输出按页面/agent/MCP层分类的失败统计

## 2026-03-31 Bad Case Recommendation Progress

- bad case 报告已支持自动推荐修复层
- 新增字段：
  - `primary_layer`
  - `recommended_layers`
  - `rationale`
  - `next_action`
- 汇总报告已支持按 `primaryFixLayers` 统计失败分布
- 下一步优先项：
  - 为每类修复层补更细的 owner/文件建议
  - 支持按页面和 intent 输出失败热点

## 2026-03-31 Medical Evidence MCP Progress

- 已新增第一版外部医学证据连接器
- 已新增统一证据对象：`MedicalEvidenceHit`
- 已新增白名单来源过滤，当前优先来源包括：
  - PubMed / PMC
  - NIH
  - WHO
  - CDC
  - Mayo Clinic
  - Cleveland Clinic
  - Johns Hopkins Medicine
- 已新增 MCP tools：
  - `search_medical_evidence`
  - `list_medical_sources`
- 已新增 MCP resource：
  - `medical://evidence-sources`
- 已新增 MCP prompt：
  - `medical_evidence_briefing`
- 下一步优先项：
  - 让评测集覆盖医学证据 MCP
  - 把医学证据真正接入患者页/预测页的知识增强链路
  - 配置搜索服务密钥后做真实联网验证

## 2026-03-31 Medical Evidence Integration Progress

- 医学证据已接入患者页和预测页知识增强链路
- 现在回答链路会在 metadata/cards 中输出：
  - `medicalEvidenceHitCount`
  - `medicalEvidenceHits`
  - `medical_evidence` card
- 评测样例已覆盖：
  - 患者页医学证据增强
  - 预测页医学证据增强
  - MCP 医学证据工具/资源/提示词联动
- 下一步优先项：
  - 配置搜索服务密钥后做真实联网验证
  - 再决定是否把医学证据引入首页解释或 bad case root-cause 分析
