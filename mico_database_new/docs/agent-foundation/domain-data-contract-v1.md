# Mico 领域语义与数据契约 v1

状态：P0 只读审计产物；本文件只定义契约，不创建表、不写入疾病映射、不改变 Java 业务逻辑。

审计日期：2026-08-20（Asia/Shanghai）  
远程只读核验记录时间：2026-08-20T18:45:43+08:00（`remote_live_verified` 结果均来自本次临时 SSH 隧道查询窗口）

## 1. 结论先行

Mico 当前业务库把一条可检索的内部记录称为 `patient`，但这条记录同时承载了样本、项目、疾病标签和丰度关联。v1 契约把它们拆成五类领域实体，并额外保留一个“内部业务记录 ID”概念：

```text
Study 1 ─── N Sample 1 ─── N TaxonomicProfile
                  │                  │
                  │                  └── 物理行：microbe_abundance_standard
                  └── 0..1 Subject（仅在 Subject 证据审核通过后建立）

InternalRecord  ──（仅通过来源关联事实连接）──  Sample
       │
       └── N DiseaseAssertion（当前 patients.disease / patient_diseases 的目标仍是内部记录）
```

`InternalRecord -> Sample` 不是 Subject 关系，也不是 v1 的全库基数承诺。远程实测确认：`meta2db_sample_metadata` 有 13,897 行，`patient_id` 为主键、`sample_id` 为唯一键，13,897 个不同 `patient_id` 全部回溯到 `patients` 且孤儿数为 0；因此在这个 Meta2DB 投影子集内是 13,897 行对应 13,897 个不同内部记录和 13,897 个不同来源样本键。该一对一结果不能外推到所有 `patients` 行或所有标准丰度样本。

`patients.patient_id` 是当前系统的内部业务关联键，不是受试者证明，也不是样本 accession。`SRR1518476` 是样本 accession/样本名；`2015_Castro-NallarE` 是项目/研究名。现有页面可以同时展示这三个值，但不得使用同一个“患者编号/患者姓名”标签解释它们。

证据：

- 远程 `meta2db_sample_metadata` 中，`patient_id=18423` 的记录同时给出：`patient_name` 为复合样本标识，`sample_id` 与 `profile_sample` 为同一复合值，`project_name=2015_Castro-NallarE`，`raw_metadata.run_acc=SRR1518476`；这条证据直接区分 Sample accession 与 Study。
- 远程 `COUNT(*)` 已实测：`patients=24,264`、`meta2db_sample_metadata=13,897`、`microbe_abundance=3,072,271`、`microbe_abundance_standard=20,852,714`、`patient_diseases=24,413`、`diseases=178`。其中 24,264 只能按当前方案表述为宏基因组样本/内部记录行数，不是已证明的独立 Subject 数。
- Meta2DB 的 13,897 个 `sample_id` 全部能在标准丰度表按显式 collation 精确匹配，且 `patient_id` 一致；标准丰度表本身实测 23,072 个不同 `sample_id`、23,072 个不同 `patient_id`，该反向覆盖关系不作为全库一对一证明。
- 本地样本快照中该项目有 32 条样本记录（`data/meta2db/metadata/meta2db_metadata.csv`、`mico_database_new/model/meta2db_species/meta2db_species_samples.csv`）。
- Java SQL 通过 `patients.patient_id` 连接 `meta2db_sample_metadata` 和 `microbe_abundance_standard`，并将 `run_acc` 优先作为 `sample_name` 展示（`mico_database_new/src/main/resources/mapper/PatientMapper.xml:8-43,50-119`）。
- 当前页面仍使用“患者详情/患者列表”以及 `patientId` 命名（`mico_database_new/src/main/resources/templates/patient_results.html:51-123`、`mico_database_new/src/main/java/com/database/mico_database/controller/AgentApiController.java:35-91`）。

## 2. 规范术语、键和不可推断规则

| 规范术语 | 规范键 | 当前实现中的对应物 | v1 解释 | 禁止推断 |
|---|---|---|---|---|
| Study / 项目或研究 | `studyKey` | `meta2db_sample_metadata.project_name` | 数据集/研究队列的名称；例：`2015_Castro-NallarE` | 不把 Study 名称当样本名或患者名 |
| Subject / 受试者 | `subjectKey` | 远程 `raw_metadata.host_subject_id`；当前仅为来源候选 | 现实世界受试者的来源标识，必须有来源证据和关系审核才建立 | 不从 `patient_id`、样本 accession 或项目名推断一名真实患者 |
| Sample / 宏基因组样本 | `sampleKey` = `(sourceSystem, sampleAccession)`；缺 accession 时退化到带来源的 `sourceSampleId` | 远程 `run_acc`、`profile_sample`、`sample_id`；Java 仍按 Mapper 回退链投影 | 一次样本/测序运行的可追溯记录；例：`SRR1518476` | 不把样本记录数当受试者数 |
| TaxonomicProfile / 物种丰度谱 | `profileKey` = `(sampleKey, taxonomyVersion, featureVersion, sourceBatch)` | `microbe_abundance_standard` 的多行物种记录 | 一个样本在特定 taxonomy、特征和导入批次下的谱；不是患者级平均值 | 不把任意 `patient_id` 下所有样本丰度拼成一个样本谱 |
| DiseaseAssertion / 疾病原始标签及标准化断言 | `diseaseAssertionId`；实现前可用来源元组稳定生成：`(targetKey, rawLabel, source, mappingVersion)` | `patients.disease`、`meta2db_sample_metadata.disease_category`、`raw_metadata` 详细字段、`diseases`/`patient_diseases` | 带来源、状态和证据的标签断言；不是临床诊断 | 不把粗粒度类别、症状、病原体、控制组标签自动提升为正式疾病 |
| Internal business record / 内部业务记录 | `internalRecordId` = `patients.patient_id` | Java DTO 的 `patientId`，Mapper 的主要连接键 | 当前系统一行业务记录的内部 ID | 不把它命名成“独立患者 ID”或样本 accession |

### 2.1 键的优先级

1. `internalRecordId` 只用于 Java 业务 API 内部关联和审计回溯。
2. Sample API 对外优先返回 `sampleAccession`、`sampleName`、`sourceSampleId` 和 `studyKey`，同时返回 `internalRecordId` 作为受控回溯字段。
3. `subjectKey` 只有在 `host_subject_id` 的来源、唯一性、跨样本复用关系通过远程查询核验后才能进入队列去重逻辑。
4. 若同一 `subjectKey` 关联多个样本，必须保留多个 Sample；不得把样本折叠成一条 Subject 记录而丢失采样/测序维度。
5. `patient_name` 在当前系统可能是源样本标识或复合样本键。它不是规范 Subject 名称；任何使用它的工具必须同时返回 `sourceSampleId` 和 `studyKey`。

## 3. 五类实体契约

### 3.1 Study

**主键**：`studyKey`，来源系统内的 `project_name`。当前没有被核验的独立 Study 表，因此 v1 将其作为由样本元数据投影出的领域实体。

**展示名**：`studyName`，默认原样展示 `project_name`；页面标签必须是“项目/Study”。

| 规范字段 | 原始来源 | Java/页面来源 | 缺失规则 | 展示规则 |
|---|---|---|---|---|
| `studyKey` | `meta2db_sample_metadata.project_name`；原始元数据 `project_name` | `Patient.projectName`，`PatientMapper.xml:19,39-43,234-235`；患者页“项目”字段 `patient_results.html:88,123` | 缺失即 `unknown`，不得从 `sample_id` 截取猜测 | 以“项目：{studyKey}”展示；不拼入样本名 |
| `studyName` | 同上 | 同上 | 不做别名或翻译 | 可与 `studyKey` 相同；不显示为患者姓名 |
| `studySource` | Meta2DB 元数据来源 | 当前 DTO 无字段 | 待核验 | API 返回来源名称和批次 |
| `studySampleCount` | 由 Sample 快照聚合 | `PatientService.listPatientSamples` 只在内部记录下聚合，`PatientMapper.xml:107-119` | 未物化快照时不返回 | 必须注明“样本数”，不能标“患者数” |

### 3.2 Subject

**主键**：`subjectKey`，仅使用有明确来源的 `host_subject_id` 或未来经审核的跨表 Subject ID。当前 `patients.patient_id` 不能作为 Subject 主键。

**展示名**：不提供真实姓名；默认展示脱敏/来源受控的 `subjectKey` 或“受试者标识不可用”。

| 规范字段 | 原始来源 | Java/页面来源 | 缺失规则 | 展示规则 |
|---|---|---|---|---|
| `subjectKey` | `meta2db_sample_metadata.raw_metadata.host_subject_id` | 当前 Java DTO 无独立字段；`Patient` 仅有 `patientId`/`patientName`（`model/Patient.java:7-16`） | 远程实时核验：11,811/13,897 行为有效候选；9,603 个不同值；805 个候选值对应多行；最大单值关联 42 行；仍不得回退到 `patient_id` | 若展示，标签为“来源受试者标识”；在 `subjectLinkStatus=unverified` 时不得显示“患者编号” |
| `subjectLinkStatus` | 由 Subject-Sample 关联核验生成 | 当前无字段 | `unverified` 为默认 | Agent 必须把 `unverified` 作为队列去重风险 |

Subject 证据台账（远程实时只读核验时间：`2026-08-20T20:55:15+08:00`）统一采用以下缺失规则：先对 `raw_metadata` 使用 `IS NOT NULL` 且 `TRIM(raw_metadata) <> ''` 判定非空；再对 `raw_metadata.host_subject_id` 做 `TRIM`，将 `NULL`、空字符串以及 `null`、`unknown`、`n/a`、`na`、`not available`、`not applicable`、`missing`、`none` 视为缺失。按该规则，`raw_metadata` 非空 11,811 行，有效 `host_subject_id` 11,811 行，2,086 行缺失；有效候选值 9,603 个，其中 805 个候选值关联多行，最大单候选值关联 42 行。该字段仍只是未核验的来源 Subject 候选；不得实现 Subject 去重、真实患者计数或把 `patient_id` 当作 Subject ID。
| `age` | `patients.age`；源候选 `host_age` | `Patient.age`、患者页年龄（`patient_results.html:151-155`） | `NULL` 或非法值保持缺失；不得用项目均值填补 | 缺失显示“未录入” |
| `gender` | `patients.gender`；源候选 `host_sex` | `Patient.gender`、页面 Male/Female/M/F 映射（`patient_results.html:157-159`） | 空值保持缺失 | 只做展示层男/女/原值映射，不推断生物学性别 |
| `country` | `patients.country`；源候选 `geo_loc_name` | `Patient.country`、页面地区（`patient_results.html:121-123,95-96`） | 空值保持缺失 | 标签为“地区/国家来源” |

### 3.3 Sample

**主键**：`sampleKey=(sourceSystem,sampleAccession)`。当 accession 缺失时，使用 `sourceSampleId` 作为带来源的替代键，并把 `sampleKeyStatus=accession_missing`。当前业务记录 `internalRecordId` 不是 Sample 主键；它只作为受控回溯字段。

**样本名规则**：当前 Java Mapper 的显示表达式按以下优先级取值：

```text
raw_metadata.run_acc
  -> raw_metadata.filename_match
  -> SUBSTRING_INDEX(profile_sample, '_hum_removed', 1)
  -> patients.patient_name / microbe_abundance_standard.sample_id
```

因此目标事实为：`SRR1518476` 是 Sample accession/样本名，`2015_Castro-NallarE` 是 Study 名。远程样例的 `patients.patient_name`、`meta2db_sample_metadata.sample_id`、`profile_sample` 均为复合样本标识；完整值不应直接替代短样本名，也不应称患者姓名。

| 规范字段 | 原始来源 | Java/页面来源 | 缺失规则 | 展示规则 |
|---|---|---|---|---|
| `sampleAccession` | `meta2db_sample_metadata.raw_metadata.run_acc` | `PatientMapper.xml:8-13,39,55,73,92,111` | 缺失时记录 `accession_missing`，允许 `filename_match` 作为来源回退但必须标注 | “样本 accession/样本名”；例：`SRR1518476` |
| `sampleName` | 同上，按 Mapper 优先级解析 | `Patient.sampleName`，`MicrobeAbundance.sampleName`（`model/Patient.java:9-12`、`model/MicrobeAbundance.java:8-15`） | 所有来源为空时为 `unknown` | 页面标题显示样本名，不显示“患者姓名” |
| `sourceSampleId` | `meta2db_sample_metadata.sample_id`；标准丰度 `microbe_abundance_standard.sample_id` | `PatientService` 的 `sampleId` 参数/返回（`PatientService.java:77-108`），API 路径（`AgentApiController.java:95-146`） | 缺失则 Sample 不可寻址 | “来源样本记录键”；可折叠查看，禁止当姓名 |
| `profileSample` | `meta2db_sample_metadata.profile_sample` | 当前只作为 SQL 回退和 name 查询条件（`PatientMapper.xml:215-219`） | 缺失保持缺失 | 仅在审计详情中显示 |
| `studyKey` | `meta2db_sample_metadata.project_name` | `Patient.projectName`、API `projectName` | 缺失不回填 | 与样本名分栏显示 |
| `sampleGroup` | `patients.Group` / `meta2db_sample_metadata.health_disease_status` | `Patient.group`；当前 resultMap 以 `Group` 映射（`PatientMapper.xml:15-27`） | 来源冲突时返回冲突状态，不覆盖原值 | 只能称“样本组/来源标签”，不能称诊断 |
| `bodySite` | `patients.body_site`；Meta2DB `body_product`/`body_site_detail` | `Patient.bodySite`、健康参考查询（`PatientMapper.xml:129-165`） | 缺失不按 feces 等默认 | “采样部位/样本来源部位” |

### 3.4 TaxonomicProfile

**主键**：`profileKey=(sampleKey,taxonomyVersion,featureVersion,sourceBatch)`。远程物理行主键是 `standard_abundance_id`，且没有 `(patient_id,sample_id,feature)` 唯一键；`(patient_id,sample_id,microbeNameStandard,featureVersion,sourceBatch)` 只能作为查询/质量检查组合，不能宣称业务唯一。样例 `patient_id=18423` 的 hash+feature_version 组无重复，但不能外推全表。

| 规范字段 | 原始来源 | Java/页面来源 | 缺失规则 | 展示规则 |
|---|---|---|---|---|
| `sampleKey` | Sample 契约 | `MicrobeAbundance.patientId/sampleId/sampleName` | 无 Sample 键则禁止返回谱 | 每个谱必须绑定一个 Sample |
| `taxonomyVersion` | 当前导入批次/物种 taxonomy 版本；本地导入意图为 `feature_version` 与 taxonomy rank | 当前 Java DTO 无版本字段；导入脚本/摘要为 `meta2db-species-v1`、`species`（`model/import_meta2db_species_direct.py`、`model/meta2db_species/meta2db_species_summary.json`） | 缺失即 `unknown`，不能跨版本合并 | 作为数据版本显示 |
| `featureVersion` | `microbe_abundance_standard.feature_version` | 当前 `MicrobeAbundance` 无字段 | 远程列 `NOT NULL DEFAULT 'v1'`；缺失时不可复现分析 | 必须随分析快照返回 |
| `sourceBatch` | `microbe_abundance_standard.source_batch` | 当前 `MicrobeAbundance` 无字段 | 远程列允许 `NULL`；缺失时标记来源批次未知 | 不隐藏批次差异 |
| `microbeNameStandard` | `microbe_abundance_standard.microbe_name_standard` | `MicrobeAbundance.microbeName`（`PatientMapper.xml:56-60`） | 空特征行拒绝 | 标准物种名；不能解释为病原体或诊断 |
| `microbeNameHash` | `microbe_abundance_standard.microbe_name_hash` | Dashboard 用于去重计数（`DashboardMapper.xml:118-121`） | 远程列 `NOT NULL`；全表 `COUNT(DISTINCT)=40,266` | 仅内部键，不展示 |
| `abundanceValue` | `microbe_abundance_standard.abundance_value` | `MicrobeAbundance.abundanceValue`、患者页（`patient_results.html:208-212`） | `NULL`/非数值拒绝；零值保留“未观测/零”语义 | 与 `abundanceUnit` 一起展示 |
| `abundanceUnit` | `microbe_abundance_standard.abundance_unit` | `MicrobeAbundance.unit` | 缺失不默认填 relative_abundance | 明示“相对丰度”或实际单位 |
| `normalizationMethod` | `microbe_abundance_standard.normalization_method` | `MicrobeAbundance.method` | 缺失即不可复现说明 | 作为方法信息，不写成生物学含义 |
| `featureCount` | 按 `(patient_id,sample_id)` 条件聚合 `COUNT(*)` | `PatientService.java:77-89,100-106` | 只能解释为精确条件下的已存储丰度行数；样例 `patient_id=18423` 为 394 行，394 个 hash/名称且无重复 hash+feature_version 组 | 展示为“已存储丰度行数”；未验证全表唯一性/版本规则前不得称完整特征数 |

远程实时标准丰度表为 20,852,714 行、23,072 个不同 `sample_id`、40,266 个不同 `microbe_name_hash`；按 `(patient_id,sample_id)` 分组的已存储行数最小 7、平均 903.8104、最大 8,501。样例的 `featureVersion=v1`、`sourceBatch=meta2db-species-v1`、`normalizationMethod=per_sample_species_count` 均为远程查询结果。另有本地 Meta2DB 物种批次 16,241,028 条物种行、13,897 个 Sample、33,313 个不同标准特征；本地数字不替代远程实时结果。

### 3.5 DiseaseAssertion

**主键**：`diseaseAssertionId`。在没有单独断言表前，API 可用稳定哈希/不可变来源元组作为临时标识；不得在本阶段写入任何映射表。

**目标对象**：如果标签来自样本级 Meta2DB 字段，默认目标为 Sample；如果来自 `patients.disease`/`patient_diseases`，目标是当前内部业务记录，只有 Subject 关系核验后才可投影到 Subject。

| 规范字段 | 原始来源 | Java/页面来源 | 缺失规则 | 展示规则 |
|---|---|---|---|---|
| `rawLabel` | `patients.disease`、`meta2db_sample_metadata.disease_category`、`raw_metadata` 的详细疾病字段 | `Patient.rawDisease`/`Patient.disease`；`PatientService.java:385-402` | 原始值不可丢；空值为 `NULL` | 审计视图原样展示并标来源 |
| `canonicalDiseaseId` | 未来人工审核映射 | 当前不存在正式映射表/API | 未映射为 `NULL` | 未映射不显示为正式疾病 |
| `canonicalNameZh` | 未来人工审核词表 | 当前无字段 | 未映射为 `NULL` | 只显示审核通过名称 |
| `mappingStatus` | 契约状态机 | 当前 Java 仅做 UI 规范化，无正式状态 | 默认 `unmapped` | 状态必须可见于 Agent 证据 |
| `mappingConfidence` | 审核/规则证据 | 当前无字段 | 未映射为 `NULL` | 不把规则命中当医学置信度 |
| `mappingVersion` | 版本化映射资产 | 当前无字段 | 未映射为 `NULL` | 随快照返回 |
| `sourceEvidence` | 字段名、原始记录、导入批次、来源 URL/文件 | 当前无统一字段 | 无证据不得标准化 | Agent 只能引用有证据断言 |
| `reviewedBy` / `reviewedAt` | 人工审核 | 当前无字段 | 未审核为 `NULL` | 明示未审核 |

疾病映射契约必须精确包含以下字段：

```text
raw_label
canonical_disease_id
canonical_name_zh
mapping_status
mapping_confidence
mapping_version
source_evidence
reviewed_by
reviewed_at
```

当前 Java 的 UI 规则仅把逗号/分号拆开、去重、转小写、把下划线替换为空格，并应用有限别名（`PatientService.java:356-373,385-402`）；Dashboard SQL 还会排除 `healthy`、`health`、`control`、`NC` 等控制标签并折叠一组别名（`DashboardMapper.xml:8-66`）。这些规则是展示/统计规则，不是经过审核的 DiseaseAssertion 映射。

## 4. 数据来源与缺失/冲突规则

### 4.1 来源优先级

1. 远程 MySQL 通过 Java 业务 API/MCP 暴露的版本化事实。
2. 业务表原始字段和 `meta2db_sample_metadata.raw_metadata`。
3. Java 页面投影字段（仅作现状证据，不反向生成事实）。
4. 本地文件快照（必须携带文件路径、批次和生成时间；不自动替代远程）。

字段冲突时同时返回 `rawValue`、`normalizedValue`、`source` 和 `conflictStatus`。不能用 UI 规范化结果覆盖原始值。

### 4.2 缺失值

空字符串、`NULL`、`not available`、`not applicable`、`unknown` 等均按字段级规则标记为缺失；不得把“控制组”“未提供”“未匹配元数据”当作疾病或健康事实。缺失值不以全局均值、项目名或患者 ID 填补。

## 5. Python LangGraph Runtime 边界

未来 Python LangGraph Runtime **只能通过 Java 业务 API / MCP 获取事实**，不得直连业务 MySQL，不持有业务数据库账号，不接受自由 SQL，不直接修改疾病映射。Java 负责认证、权限、返回上限、快照和审计；Python 负责规划、状态、工具调用、统计编排和证据综合。

所有工具响应至少包含：`toolCallId`、`dataSnapshotId`、`source`、`rowCount`、`schemaVersion`、`generatedAt`、`data`。拒绝也必须结构化返回原因。

## 6. 数据快照最小字段

每次队列或分析读取必须生成不可变快照元数据，最小字段如下：

| 字段 | 含义 | 规则 |
|---|---|---|
| `dataSnapshotId` | 快照唯一 ID | 不可复用；工具响应和报告必须回传 |
| `dataSource` | 事实来源 | 例如 Java 业务 API + 远程业务库；不写密码/连接串 |
| `importBatch` | 导入批次 | 来自来源批次，不用当前时间猜测 |
| `diseaseMappingVersion` | 疾病映射版本 | 未使用映射时填 `none`，不写假版本 |
| `taxonomyVersion` | taxonomy/特征版本 | 必须与 TaxonomicProfile 绑定 |
| `cohortCondition` | 队列筛选条件 | 结构化、可重放，不能只留自然语言 |
| `queryHash` | 规范化查询/条件哈希 | 对参数排序、脱敏后计算 |
| `rowCount` | 返回或物化行数 | 明确是 Sample、Subject、Assertion 还是丰度行 |
| `generatedAt` | 生成时间 | ISO-8601 含时区 |

建议同时保留 `schemaVersion`、`toolCallId`、权限范围、最大返回量、脱敏策略和质量摘要。

### 6.1 本次远程规模证据

以下数字的 `evidence_level` 均为 `remote_live_verified`，核验记录时间为 `2026-08-20T18:45:43+08:00`；它们不是由页面标题或历史文件推算：

| 事实 | 数值 | 语义边界 |
|---|---:|---|
| `patients` 总行数 | 24,264 | 当前内部业务记录行数；按方案的安全产品表述可称“宏基因组样本记录数”，不能称 24,264 位独立患者 |
| `meta2db_sample_metadata` 总行数 | 13,897 | Meta2DB 样本元数据记录数；每行 `patient_id` 主键、`sample_id` 唯一 |
| `microbe_abundance` 总行数 | 3,072,271 | 原始/非标准丰度表存储行数 |
| `microbe_abundance_standard` 总行数 | 20,852,714 | 标准化丰度存储行数；不能称样本数或患者数 |
| `patient_diseases` 总行数 | 24,413 | 内部记录—疾病字典关联行数 |
| `diseases` 总行数 | 178 | 当前数据库疾病字典行数，不等于审核后的临床疾病类别数 |

远程结构还显示：`patients.patient_id`、`meta2db_sample_metadata.patient_id`、`microbe_abundance.abundance_id`、`microbe_abundance_standard.standard_abundance_id`、`patient_diseases.id`、`diseases.disease_id` 分别是物理主键；`patient_id,disease_id` 在 `patient_diseases` 上有唯一键。`microbe_abundance_standard` 没有 `(patient_id,sample_id,feature)` 唯一键，因此 `featureCount` 必须保守称已存储行数。

## 7. 页面展示契约（目标规则，不在 P0 实施）

- 样本页标题：`样本名：SRR1518476`；副标题：`项目：2015_Castro-NallarE`；审计信息单独显示 `内部记录 ID`。
- 不使用“患者姓名”承载 `sampleName`；不使用“患者编号”承载 `sampleAccession`。
- 受试者信息只有在 `subjectLinkStatus=verified` 时才显示“受试者”；否则显示“样本记录/受试者关系待核验”。
- 丰度图表必须标明“当前 Sample 的 TaxonomicProfile”；不能只写“患者微生物组”。
- 疾病区域分为“原始标签”“标准化断言”“映射状态”，不把一个字符串直接渲染成正式临床疾病。
- 首页“24,264”若继续使用，标题必须是“宏基因组样本记录数”，不能是“患者总数”。

## 8. P0 不做的事情

- 不新建 Study/Subject/Sample/DiseaseAssertion 表。
- 不写入疾病映射，不修改 `patients.disease`、`diseases` 或 `patient_diseases`。
- 不改 Java DTO、Mapper、Controller、页面、配置或 Python 服务。
- 不把无法通过远程只读查询验证的主键、外键、行数和唯一性写成已确认事实。
