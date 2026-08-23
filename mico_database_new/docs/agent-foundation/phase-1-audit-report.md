# P0 / Phase 1 审计报告：领域语义与数据契约

审计日期：2026-08-20  
远程只读核验记录时间：2026-08-20T18:45:43+08:00  
工作目录：`E:\DeskTop\java_project\mico_database_new`  
范围：P0.1 只读审计和文档修订；不实现 P1，不实现 LangGraph，不新增 Python 服务，不修改 Java 业务代码。

## 1. 执行结果

P0 文档已形成：

1. `docs/agent-foundation/domain-data-contract-v1.md`
2. `docs/agent-foundation/domain-field-mapping-v1.csv`
3. `docs/agent-foundation/data-quality-baseline-v1.md`
4. `docs/agent-foundation/phase-1-audit-report.md`
5. `docs/agent-foundation/p0-evidence-matrix-v1.md`

四份原有文件已根据远程实时事实修订，并新增 `p0-evidence-matrix-v1.md`。文档只定义领域实体、字段来源、质量基线、疾病映射契约和 P1 建议；没有创建数据库表，没有写入 DiseaseAssertion 映射，没有修改 Java/Python/前端/配置/Mapper/导入脚本。临时 SSH 隧道已停止，`13306` 无监听者。

## 2. 已确认的数据事实

### 2.1 身份和项目语义

| 结论 | 证据 |
|---|---|
| `SRR1518476` 是样本 accession/样本名，而不是项目名 | 远程 `meta2db_sample_metadata.raw_metadata.run_acc` 精确命中；同一行 `profile_sample` 以该 accession 开头，Java Mapper 优先从 `run_acc` 生成 `sample_name`（`PatientMapper.xml:8-13,39-43`） |
| `2015_Castro-NallarE` 是项目/研究名 | 同一远程行的 `project_name=2015_Castro-NallarE`；本地物种样本快照中该项目有 32 条记录 |
| 完整 `sample_id` 是来源复合键，不是短样本名 | 远程样例中 `patients.patient_name`、Meta2DB `sample_id` 和 `profile_sample` 均为复合样本标识，短 accession 由 `raw_metadata.run_acc` 提供 |
| `patients.patient_id` 只能称内部业务记录 ID | 远程 `patients.patient_id` 是物理 PK；Java 详情、样本、丰度和健康参考查询均以它连接（`PatientMapper.xml:38-43,50-119`）；不存在可据此证明 Subject 的证据 |
| 24,264 是远程 `patients` 内部记录行数，可按方案安全称宏基因组样本记录数 | 远程 `SELECT COUNT(*) FROM patients`=`24,264`，核验时间 `2026-08-20T18:45:43+08:00`；不是独立 Subject 数 |
| 20,852,714 是远程标准化丰度存储行数 | 远程 `SELECT COUNT(*) FROM microbe_abundance_standard`=`20,852,714`，同一核验时间；不是样本数或患者数 |

### 2.2 现有 Java 数据投影

- `Patient` 已有 `patientId`、`patientName`、`sampleName`、`projectName`、原始/展示疾病字段、人口学字段（`mico_database_new/src/main/java/com/database/mico_database/model/Patient.java:7-24`）。这是有价值的过渡结构，但 `patientName` 仍被现有页面作为姓名/样本标题使用。
- `MicrobeAbundance` 已有 `patientId`、`sampleId`、`sampleName`、物种名、丰度、单位、方法（`mico_database_new/src/main/java/com/database/mico_database/model/MicrobeAbundance.java:7-36`）。缺少 `featureVersion`、`sourceBatch`、`taxonomyVersion`、snapshot 字段。
- `PatientMapper.xml` 对标准丰度按 `patient_id` 过滤，并以 `(patient_id, sample_id)` 聚合样本摘要（`mico_database_new/src/main/resources/mapper/PatientMapper.xml:50-119`）。这证明当前查询语义是“内部记录下的 Sample”，不证明 `patient_id` 是 Subject。
- 远程 `meta2db_sample_metadata` 的 `patient_id` 是 PK、`sample_id` 是唯一键；13,897 个 Meta2DB 样本行与 13,897 个不同内部记录在该子集内一对一。标准丰度表实测 23,072 个不同 `sample_id`/`patient_id`/`(patient_id,sample_id)` 组合；全库 Sample↔InternalRecord 基数不外推。
- `microbe_abundance_standard.sample_id` 与 Meta2DB `sample_id` 的默认字符集不同，显式 collation 后 13,897/13,897 Meta2DB 样本命中且 `patient_id` 一致；表上无特征行唯一键，因此 `featureCount` 只能称精确条件下已存储丰度行数。
- `AgentApiController` 已暴露患者、样本列表、样本详情、Top features、预测 payload 和健康参考 API（`mico_database_new/src/main/java/com/database/mico_database/controller/AgentApiController.java:35-146`）。P1 应在保持业务兼容的前提下为 Agent 增加强类型版本和快照字段，而不是让 Python 直连数据库。

### 2.3 疾病标签

- `patients.disease` 可能含逗号/分号多标签；Java UI 规则拆分、去重和应用有限别名（`mico_database_new/src/main/java/com/database/mico_database/service/PatientService.java:356-402`）。
- Dashboard SQL 过滤控制/健康标签并折叠若干别名（`mico_database_new/src/main/resources/mapper/DashboardMapper.xml:8-66`）。这只是展示/统计规则，不是审核过的正式疾病本体。
- `patient_diseases -> diseases` 是另一条疾病检索路径，而疾病列表查询与 `patients.disease LIKE` 不是同一口径（`PatientMapper.xml:222-258`）。
- 远程 `diseases=178`、`patient_diseases=24,413`，关联使用的 `disease_id=163`，孤儿关联为 0；当前 Dashboard 规则复现的名称集合为 130，不等于 178 行字典。
- 远程复现的 130 名称由 `normalizedDiseaseCte` 生成，并包含粗粒度、症状/表型和病原体/感染相关标签；`patient_id=18423` 同时出现 `patients.disease=neurological` 与样本组 `control`。
- 本地 Meta2DB 原始字段包含多值、粗粒度类别、症状/表型、感染/病原体和控制标签。应使用 `DiseaseAssertion` 分层保存原值、映射状态、证据和审核信息。
- 当前“130 个疾病名称”必须写成“当前规则得到的名称集合”；不能写成 130 种正式临床疾病类别。

## 3. 字段和页面展示的高风险混用

### 3.1 页面风险

1. 患者结果页把 `patient.sampleName` 放在“姓名”列，并把它作为详情标题（`mico_database_new/src/main/resources/templates/patient_results.html:75-89,109-123`）。对于 Meta2DB，它可能显示为 `SRR1518476`，但标签会误导为患者姓名。
2. 页面把 `patient.patientId` 显示为“患者编号”（同文件 `:75,85,121`）；在当前数据语义下更安全的标签是“内部记录 ID”。
3. 页面把疾病标准化后的字符串直接放在“疾病”指标中（同文件 `:160-167`），没有原始值、映射状态、证据和“控制/症状/病原体”分类。
4. 页面 `当前样本` 选择器基于 `sampleId`/`sampleName`，这是正确的单样本交互方向，但缺少 Study、内部记录 ID、Subject link status 和 profile version 的同时展示（`patient_results.html:128-137`、`PatientService.java:77-108`）。
5. 首页 `total_patients` 来自 `COUNT(patient_id)`（`mico_database_new/src/main/resources/mapper/DashboardMapper.xml:78-80`），必须改为明确口径后才能用于 Agent 报告；P0 不修改它。

### 3.2 数据访问风险

1. `microbe_abundance` 用 `SELECT *`（`PatientMapper.xml:47`），对字段增删和原始/标准语义缺乏版本边界。
2. `microbe_abundance_standard` 的查询返回了 sample name，但 DTO 未返回 feature/version/batch，Agent 无法完整回放一个谱。
3. 健康参考以 `patients.disease='healthy'` 加内部记录连接筛选，既可能漏掉 `health/control/healthy subject study`，也可能把多标签记录处理成不同统计口径（`PatientMapper.xml:125-202`、`DashboardMapper.xml:109-117`）。
4. `findIdsByName` 允许用 `patients.patient_name` 或 `profile_sample` 查询（`PatientMapper.xml:215-219`），说明 `patient_name` 的语义不能稳定称为患者姓名。

## 4. Agent 研发风险

### P0 级风险

- 模型将 `patient_id` 误说成独立患者 ID。
- 模型将 `SRR1518476`、复合 `sample_id` 或 `profile_sample` 误说成患者姓名。
- 把 `TaxonomicProfile` 的物种丰度差异写成诊断、病因或患者确诊。
- 把 `control`、`healthy subject study`、`neurological`、`diarrhea`、`SARS-CoV-2`、`obese` 等不同语义当作同一正式疾病类。
- Python 直连 MySQL、自由 SQL 或绕开 Java 权限取数。

### P1/P2 风险

- 多 Sample 在 `patient_id` 下被错误合并。
- 元数据缺失时模型补写年龄、性别、地区、采样部位或疾病。
- 远程表与本地 Meta2DB 文件批次不一致但没有 `dataSnapshotId`/`importBatch`。
- `diseases` 的当前词典数量被误报为审核后的临床本体数量。
- 外部文献覆盖内部样本事实；未映射/低置信疾病标签被模型自行规范化。

## 5. 待核验项

以下事项在本次只读核验后仍必须标为“待核验”，不能从现有结果猜测：

1. `host_subject_id` 是否已经经过人工/来源规则证明为真实 Subject；9,603 个候选值、805 个重复候选和 142 个跨项目候选不能直接变成 Subject 数。
2. `patients` 的 24,264 行与标准丰度 23,072 个非空样本键之间的缺口语义，以及没有丰度行的内部记录是否仍有可用 Sample。
3. 标准丰度表全量按 `microbe_name_hash + feature_version + source_batch` 的重复规则、零值保留规则和 taxonomy 版本冻结；物理结构没有特征行唯一键。
4. `microbe_abundance` 与 `microbe_abundance_standard` 的原始/标准口径、数据来源批次和相互关系；本次只确认了两张表的结构和行数，未将其合并解释。
5. `patients.disease` 每个多值原始字符串与 `patient_diseases` 关联的逐患者完全一致性；当前只确认关联无孤儿，未将两条查询路径强行视为同一口径。
6. Java `Group`、`disease`、Meta2DB `health_disease_status`/`disease_category` 的同步与覆盖规则。
7. 正式数据快照的 `importBatch`、taxonomy 版本、疾病映射版本和生成时间的全量分布；本次样例 batch/version 已见，但 P1 必须从 Java 受控工具返回。

## 6. P1 入口判断与分阶段结论

### P1-A：Java 受控工具契约、Schema、权限模型、快照协议和离线契约测试

**可开始：是。** 六张表的远程 schema、主键/唯一键/外键摘要、精确总行数、样例回溯、快照最小字段和 Python 边界已经足够冻结 P1-A 的契约与 fixture。P1-A 不需要等待人工疾病映射；契约必须保留 `subjectLinkStatus=unverified`，并把 `patients.patient_id` 只命名为 `internalRecordId`。

### P1-B：真实远程数据工具实现

**具备受限实施条件：可以开始，但必须带剩余条件。** 远程 schema、访问路径和只读查询已真实验证，足以实现 Java 侧受控工具；实现前必须配置正式的非文档化凭据注入、连接超时/重试、权限与返回上限，并冻结本报告列出的 schema/版本 fixture。P1-B 不能把 Python 直接接到 MySQL，也不能提供自由 SQL；所有查询必须仍经 Java API/MCP。

### P3 前置条件：正式疾病映射审核与统计方法冻结

**尚未完成，且不阻塞 P1-A。** 正式 `DiseaseAssertion` 映射仍需人工审核、证据和版本冻结；P1 阶段的 `resolve_disease` 可以安全返回 `rawLabel + mapping_status=unmapped`，不创建或修改映射。P3 在映射审核、Subject 统计口径、featureCount/版本规则和队列统计方法冻结前不得宣称完成。

## 7. 对 P1 Java 受控工具的建议清单

P1 仅建议设计/实现以下 Java 受控只读工具；工具实现必须留在 Java 事实源一侧，Python 只通过 API/MCP 调用。

| 工具 | 最小输入 | 必须返回 | 风险控制 |
|---|---|---|---|
| `resolve_study` | `studyKey`/精确项目名 | Study、样本记录数、来源批次、快照 | 不从样本名猜项目；上限和权限 |
| `resolve_sample` | `sampleAccession` 或 `sourceSampleId`，可选 `studyKey` | accession、sampleName、sourceSampleId、study、internalRecordId、Subject link status | 同名/多命中时返回候选和冲突，不自动选第一条 |
| `get_subject_context` | `internalRecordId` 或已验证 `subjectKey` | Subject 候选、关联 Sample 数、证据状态、人口学覆盖 | 未验证时 fail-closed，不伪造独立患者 |
| `get_sample` | `internalRecordId+sourceSampleId` 或 `sampleKey` | 单一 Sample 元数据和原始值/规范值 | 必须精确到 Sample；禁止仅 patientId 取全量谱 |
| `list_sample_profiles` | `sampleKey`、可选版本 | Profile keys、taxonomy/feature/batch、featureCount | 不跨版本合并；限制返回数量 |
| `get_abundance_summary` | `sampleKey`、taxonomy/feature 版本、limit | 标准特征、丰度、单位、方法、快照 | 禁止自由 SQL/全表导出；上限和超时 |
| `resolve_disease` | `rawLabel`、来源上下文 | 完整 DiseaseAssertion 字段、证据、映射状态 | 只读；不得由 LLM 创建/修改映射 |
| `build_cohort` | 结构化 Study/Sample/Assertion 条件 | cohort 快照、行数、缺失摘要、冲突摘要 | 不接受 SQL；必须返回 `dataSnapshotId` |
| `get_data_snapshot` | `dataSnapshotId` | 数据源、批次、映射/taxonomy 版本、条件、queryHash、rowCount、generatedAt | 快照不可变；审计字段完整 |
| `get_quality_summary` | 快照或条件 | 覆盖率、缺失、重复、孤儿关联、Subject 证据等级 | Agent 报告必须携带质量边界 |

所有 P1 工具统一返回：`toolCallId`、`dataSnapshotId`、`source`、`rowCount`、`schemaVersion`、`generatedAt`、结构化 `data` 和结构化拒绝原因。权限建议至少拆分为 `mico:study:read`、`mico:sample:read`、`mico:subject:read`、`mico:cohort:read`、`mico:abundance:read`、`mico:ontology:read`、`mico:snapshot:read`。

## 8. 评测衔接

P1 设计完成后应新增/更新 Eval，至少覆盖：

- `SRR1518476` 必须返回样本语义和 `2015_Castro-NallarE` 项目语义。
- `patient_id` 必须被称为内部记录 ID，不能声称独立患者。
- 24,264 必须输出“样本记录数”。
- 多值、别名、粗粒度、症状、病原体和控制标签必须被分层。
- 未验证 Subject 关系时拒绝按患者人数去重。
- 所有 Agent 工具调用必须含快照、来源、行数和版本。

既有评测要求“事实一致性、分层清晰、边界控制、Task Packet/Policy/Trace 完整”（`mico_database_new/EVALS.md:20-66`）；本报告把 P0 领域断言补充为后续回归的确定性断言。

## 9. 阶段结论

P0.1 已把远程实时事实、源码事实、本地文件统计和历史方案基线分层记录：schema 为 `patient_data_manager`；六表计数、结构、索引、样例 `SRR1518476`/`2015_Castro-NallarE`、疾病关系和 Subject 候选均有脱敏证据。P1-A 可以开始，P1-B 具备受限真实数据工具实施条件但仍需连接治理和版本 fixture；疾病人工映射与统计方法冻结属于 P3 前置，不阻塞 P1-A。当前阶段到此停止，不自行进入 P1。
