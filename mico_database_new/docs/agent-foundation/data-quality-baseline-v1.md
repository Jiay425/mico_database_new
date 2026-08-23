# Mico P0 数据质量基线 v1

审计日期：2026-08-20（Asia/Shanghai）  
范围：源码、Mapper、DTO、页面字段、本地 Meta2DB 文件快照，以及通过新远程地址建立的临时 SSH 隧道只读核验。  
限制：未启动 Java、Python 或长驻服务；只执行 `SHOW`、`EXPLAIN`、`COUNT`、聚合和样例 `SELECT`；未对任何数据库执行写操作。临时隧道已关闭，核验结束后 `13306` 无监听者。

## 1. 证据分层与重要限制

本文件把证据分为三层：

- **源码事实**：Java DTO、Mapper、Service、Controller、页面模板中已经存在的字段与查询语义。
- **本地文件快照**：`data/meta2db/**`、`mico_database_new/model/meta2db_species/**` 的只读统计；这些文件不是远程数据库实时快照。
- **远程实时事实**：本次通过 `10.31.2.52` 的临时 SSH 隧道访问 `patient_data_manager`，只记录脱敏后的结构、计数和样例关系；证据等级为 `remote_live_verified`，核验记录时间为 `2026-08-20T18:45:43+08:00`。
- **历史/方案基线**：方案文件的 24,264 和 20,852,714 先前是历史基线；本次远程 `COUNT(*)` 恰好复核到相同数值，但二者仍按表行语义分别记录，不改变为患者数。

远程读取边界证据：`RemoteDatabaseTunnel.java` 定义默认目标为 `10.31.2.52` 的 SSH 转发到远端 `127.0.0.1:3306`；本次只使用临时转发并在结束后停止。连接配置路径仅用于运行时读取，文档不记录密码、私钥或完整连接串。架构边界证据：`mico_database_new/src/main/resources/application.properties:8-13`、`mico_database_new/src/main/java/com/database/mico_database/config/RemoteDatabaseTunnel.java:15-51`、`mico_database_new/ARCHITECTURE-AGENT.md:10-28`。

## 2. 相关表总行数与主键核验状态

| 表 | 远程总行数 | 当前可确认语义 | 远程主键/唯一键/外键（脱敏摘要） | 核验时间 |
|---|---:|---|---|---|
| `patients` | 24,264 | Java `Patient` 和核心 Mapper 以 `patient_id` 连接；是内部业务记录表，不自动等于 Subject | PK `patient_id`；普通索引 `patient_name`；无唯一 `patient_name` | 2026-08-20T18:45:43+08:00 |
| `meta2db_sample_metadata` | 13,897 | Meta2DB 样本元数据投影，含 `sample_id`、`project_name`、`profile_sample`、`raw_metadata` | PK `patient_id`；唯一键 `sample_id`；普通索引 `project_name`、`disease_category`；未声明 FK | 2026-08-20T18:45:43+08:00 |
| `patient_diseases` | 24,413 | 内部记录—疾病字典关联；不能据此证明 Subject | PK `id`；唯一键 `(patient_id,disease_id)`；FK 到 `patients`、`diseases` | 2026-08-20T18:45:43+08:00 |
| `diseases` | 178 | 当前数据库疾病字典；不等于审核后的 DiseaseAssertion 词表 | PK `disease_id`；唯一键 `disease_name` | 2026-08-20T18:45:43+08:00 |
| `microbe_abundance` | 3,072,271 | 原始/非标准丰度记录；Mapper 当前使用 `SELECT *` | PK `abundance_id`；普通索引/FK `patient_id`→`patients.patient_id`；`sample_id` 无唯一键/独立索引 | 2026-08-20T18:45:43+08:00 |
| `microbe_abundance_standard` | 20,852,714 | 标准化丰度存储行；TaxonomicProfile 的物理行源 | PK `standard_abundance_id`；普通索引 `patient_id`、`sample_id`、`feature_version`、`(patient_id,sample_id)`、`microbe_name_hash`、物种名前缀；FK `patient_id`→`patients.patient_id`；无特征行唯一键 | 2026-08-20T18:45:43+08:00 |

方案文件声明的 **24,264 条宏基因组样本记录** 和 **20,852,714 条标准化物种丰度记录**（`MICO_AGENT_UPGRADE_PLAN.md:9`）已由远程 `COUNT(*)` 分别在 `patients` 与 `microbe_abundance_standard` 实测相同。它们现在是 `remote_live_verified`，但仍只能表述为**内部记录/样本记录数与丰度存储行数，不是独立患者数**。

## 3. 本地 Meta2DB 文件快照统计

### 3.1 样本/项目文件

| 文件 | 行数/规模 | 结果 |
|---|---:|---|
| `data/meta2db/metadata/meta2db_metadata.csv` | 13,897 行，171 列 | 84 个 `project_name`；`project_name` 100% 有值；`filename_match` 100% 有值 |
| `mico_database_new/model/meta2db_species/meta2db_species_samples.csv` | 13,897 行，7 列 | 85 个项目；`sample_id` 13,897 个唯一；`profile_sample` 13,897 个唯一 |
| `mico_database_new/model/meta2db_species/meta2db_species_summary.json` | 摘要 | 85 个 profile、13,897 个 sample、2,086 个未匹配元数据 sample、16,241,028 条非零物种行 |

文件之间的 84/85 个项目差异是批次/项目命名一致性风险，不能自行判定谁是事实源。

### 3.2 关键字段覆盖

以下覆盖率来自 `data/meta2db/metadata/meta2db_metadata.csv` 的 13,897 行；“可用”排除了空字符串、`not available`、`not applicable`、`unknown` 等明显缺失标记。

| 字段 | 可用行 | 缺失行 | 覆盖率约 | 质量解释 |
|---|---:|---:|---:|---|
| `project_name` | 13,897 | 0 | 100.0% | 可作为 Study 候选键 |
| `run_acc` | 12,639 | 1,258 | 90.95% | 可作为 Sample accession，但不能假定全覆盖 |
| `filename_match` | 13,897 | 0 | 100.0% | 可作为来源回退，但不等同 run accession |
| `host_age` | 9,386 | 4,511 | 67.54% | 年龄缺失明显 |
| `host_sex` | 9,657 | 4,240 | 69.49% | 性别缺失明显 |
| `geo_loc_name` | 10,744 | 3,153 | 77.31% | 国家/地区覆盖不完整 |
| `host_body_product` | 13,149 | 748 | 94.62% | 多数有样本来源部位 |
| `host_body_site` | 1,304 | 12,593 | 9.38% | 细粒度采样部位严重缺失 |
| `host_subject_id` | 11,811 | 2,086 | 84.99% | 按 NULL/空字符串及 null、unknown、n/a、na、not available、not applicable、missing、none 缺失规则；有效候选不等于已证明 Subject 关系 |
| `disease_category` | 13,897 | 0 | 100.0% | 原始类别有值，但含粗粒度、多值和非疾病语义 |

证据：上述文件只读流式统计；字段原始表头位于 `data/meta2db/metadata/meta2db_metadata.csv:1`。

### 3.2a 远程实时字段覆盖（`remote_live_verified`）

追加核验时间：`2026-08-20T19:02:45+08:00`。以下为远程业务表当前覆盖，不替代上一节本地 CSV 覆盖；表计数/结构结果的主核验时间仍记录为 `2026-08-20T18:45:43+08:00`：

| 表/字段 | 可用或非空行 | 总行数 | 约覆盖率 | 备注 |
|---|---:|---:|---:|---|
| `patients.patient_name` | 24,264 | 24,264 | 100.00% | 远程样例证明其可能是复合样本标识，不是姓名语义 |
| `patients.disease` | 21,890 | 24,264 | 90.23% | 非空原始字符串；含控制/多值/非疾病语义 |
| `patients.age` | 15,662 | 24,264 | 64.55% | `NULL` 仍按缺失处理 |
| `patients.gender` | 17,048 | 24,264 | 70.26% | 空字符串按缺失处理 |
| `patients.country` | 18,845 | 24,264 | 77.64% | 空字符串按缺失处理 |
| `patients.Group` | 22,177 | 24,264 | 91.42% | 样本组/来源标签，不是疾病断言 |
| `patients.body_site` | 19,510 | 24,264 | 80.41% | 采样部位候选，不证明 Subject |
| `meta2db_sample_metadata.project_name` | 13,897 | 13,897 | 100.00% | Study 候选字段 |
| `meta2db_sample_metadata.sample_id` | 13,897 | 13,897 | 100.00% | 唯一来源样本键 |
| `meta2db_sample_metadata.profile_sample` | 13,897 | 13,897 | 100.00% | 完整 Profile 样本标识 |
| `meta2db_sample_metadata.disease_category` | 11,811 | 13,897 | 84.99% | 与 raw JSON 非空覆盖一致 |
| `meta2db_sample_metadata.health_disease_status` | 11,811 | 13,897 | 84.99% | 组别/状态候选，不是正式疾病 |
| `meta2db_sample_metadata.country` | 9,923 | 13,897 | 71.41% | 来源地区候选 |
| `meta2db_sample_metadata.raw_metadata` | 11,811 | 13,897 | 84.99% | `host_subject_id`、`run_acc` 等候选字段的载体 |

远程 `meta2db_sample_metadata` 的 13,897 个 distinct `patient_id` 全部在 `patients` 命中，孤儿数为 0；这使 Meta2DB 子集的 Sample—InternalRecord 回溯成立，但不把 `patient_id` 提升为 Subject。

### 3.3 `SRR1518476` 与 `2015_Castro-NallarE`

本地源记录明确显示：

```text
run_acc       = SRR1518476
project_name  = 2015_Castro-NallarE
host_age      = 29
host_sex      = male
geo_loc_name  = United States of America
health_disease_stat = control
disease_category = neurological
host_body_site = oral cavity
```

物种样本快照中对应记录为：

```text
sample_id     = 2015_Castro-NallarE|SRR1518476_Study_of_microbial_diversity_from_samples_derived_from_throat_swabs_of_schizophrenia_patients_hum_removed_nt_mhl15
project_name  = 2015_Castro-NallarE
profile_sample= SRR1518476_Study_of_microbial_diversity_from_samples_derived_from_throat_swabs_of_schizophrenia_patients_hum_removed_nt_mhl15
group         = control
disease       = neurological
```

这组证据足以证明：`SRR1518476` 是样本 accession/样本名，`2015_Castro-NallarE` 是项目/研究名；复合 `sample_id` 是来源样本记录键，不是项目名本身。

### 3.4 项目名、样本名、内部 ID 的混用风险

- 当前文件中的 `sample_id` 由 `project_name|profile_sample` 组成，天然包含项目名；它是导入/关联键，不是短样本名。
- `profile_sample` 又把 accession 与研究描述后缀拼在一起；不能直接把完整文件名当展示名。
- Java Mapper 将 `raw_metadata.run_acc` 优先投影为 `sampleName`，并用 `patients.patient_id` 连接 `meta2db_sample_metadata` 和丰度表（`PatientMapper.xml:8-13,39-64`）。
- 现有 `Patient.patientName` 的注释已说明它可能是不可变源键，而 `sampleName` 是展示名（`model/Patient.java:8-12`）；但页面仍把 `sampleName` 放在“姓名”列/患者详情标题（`patient_results.html:75-89,109-123`）。
- 当前 Java Service 的样本列表输出还同时返回 `patientId`、`sampleId`、`sampleName`（`PatientService.java:77-89`），说明契约层仍需要显式区分三种 ID。

## 4. 标准丰度记录与每样本特征分布

### 4.1 远程实时标准丰度结果（`remote_live_verified`）

核验时间：`2026-08-20T18:45:43+08:00`。远程 `microbe_abundance_standard` 的精确 `COUNT(*)` 为 20,852,714；`COUNT(DISTINCT sample_id)` 为 23,072；`COUNT(DISTINCT patient_id)` 为 23,072；`COUNT(DISTINCT (patient_id,sample_id))` 为 23,072。Meta2DB 的 13,897 个 `sample_id` 全部通过显式 collation 的索引探测匹配到标准丰度表，且每个匹配均有相同 `patient_id`。

| 指标 | 值 |
|---|---:|
| 远程丰度行数 | 20,852,714 |
| 远程不同 sample_id | 23,072 |
| 远程不同 patient_id | 23,072 |
| 远程不同 microbe_name_hash | 40,266 |
| 远程按 `(patient_id,sample_id)` 分组的最小已存储行数 | 7 |
| 远程按 `(patient_id,sample_id)` 分组的平均已存储行数 | 903.8104 |
| 远程按 `(patient_id,sample_id)` 分组的最大已存储行数 | 8,501 |
| 样例 `patient_id=18423` 的已存储行数 | 394 |
| 样例 `patient_id=18423` 的 distinct hash/名称 | 394 / 394 |
| 样例 `feature_version` | `v1` |
| 样例 `source_batch` | `meta2db-species-v1` |
| 样例 `normalization_method` | `per_sample_species_count` |

`microbe_abundance_standard.sample_id` 与 Meta2DB `sample_id` 的默认字符集不同（远程 JOIN 若不显式处理 collation 会报错），但按显式 collation 的逐值索引探测，Meta2DB 13,897/13,897 命中且 `patient_id` 一致。表上没有特征行唯一键；因此 `featureCount` 只能称“精确条件下的已存储丰度行数”，不能无条件称唯一物种特征数。

### 4.2 本地文件快照结果（`local_source_verified`）

以下统计的是本地物种长表 `mico_database_new/model/meta2db_species/meta2db_species_relative_abundance_long.csv`，不是远程实时结果。

| 指标 | 值 |
|---|---:|
| 丰度行数 | 16,241,028 |
| 不同 sample key | 13,897 |
| 不同标准特征名 | 33,313 |
| taxonomy rank | 全部为 `species` |
| 每样本特征数最小值 | 81 |
| P25 | 692 |
| 中位数 | 1,002 |
| P75 | 1,454 |
| P90 | 2,077 |
| P95 | 2,505 |
| 平均值 | 1,168.67 |
| 最大值 | 6,355 |

解释边界：本地导入/摘要称这些是非零物种行；本地 `featureCount` 只能描述“已存/非零特征行数”，不能描述完整测序矩阵的零值特征数。远程 `featureCount` 同理还受数据库重复行和版本规则约束。Java 页面按单一样本排序后取 Top 特征，Service 对 `patientId+sampleId` 过滤（`PatientMapper.xml:68-119`、`PatientService.java:92-113`）。

## 5. 疾病标签质量

### 5.1 现有字符串处理规则

- `patients.disease` 被 Java UI 规则按逗号或分号拆分、去重、转小写、下划线改空格，再应用有限别名（`PatientService.java:356-373,385-402`）。
- Dashboard SQL 把 `healthy`、`health`、`healthy subject study`、`control`、`NC`、`normal control` 视为控制/健康排除项，并将 `IBD`、`T2D`、`CRC`、`PD`、`MS`、`AD`、`MCI`、`CAD`、`HF` 等别名折叠（`DashboardMapper.xml:35-65`）。
- `findIdsByDisease` 走 `patient_diseases -> diseases` 精确词典关联，而列表检索直接对 `patients.disease` 做 `LIKE`（`PatientMapper.xml:222-258`）；两条路径的结果可能不一致。

### 5.2 远程实时疾病事实（`remote_live_verified`）

核验时间：`2026-08-20T18:45:43+08:00`。

- `patients.disease` 非空内部记录数为 21,890，原始非空 distinct 字符串值为 315；其中精确控制/健康标签（`healthy`、`health`、`healthy subject study`、`control`、`nc`、`normal control`）覆盖 7,134 行；逗号或分号多值原始值覆盖 1,582 行。
- 远程 `patient_diseases` 有 24,413 行、22,177 个不同 `patient_id`、163 个不同 `disease_id`；与 `patients`、`diseases` 的孤儿关联均为 0；`diseases` 中 15 个字典名当前没有关联行。
- `diseases` 有 178 个唯一 `disease_name`。这 178 个字典名、163 个已关联字典 ID、以及 Dashboard 规则得到的 130 个名称集合是三个不同统计口径。
- 当前“130 个疾病名称”可由 `DashboardMapper.xml` 的 `normalizedDiseaseCte` 复现：对 `patients.disease` 按逗号/分号拆分，trim、转小写、下划线换空格，应用有限别名，排除六个控制/健康精确标签，再按 `(patient_id,disease)` 去重，最后 `COUNT(DISTINCT disease)`；本次远程复现结果为 130。
- 该 130 集合仍包含非正式临床类别/标签，例如远程结果中的 `neurological`、`cardiovascular`、`pulmonary`、`dermatologic`、`autoimmune`、`cancer`，以及 `diarrhea`、`obese`、`lactose intolerant`、`poor growth`、`SARS-CoV-2`、`HIV-1`、`cholera`、`plasmodium falciparum`、`ascaris lumbricoides`。所以它只能称“当前规则得到的名称集合”，不能称 130 种正式临床疾病类别。
- 实时样例 `patient_id=18423` 的 `patients.disease=neurological`，同时 `meta2db_sample_metadata.health_disease_status=control`、`disease_category=neurological`，并关联 `diseases.disease_name=neurological`；这证明组别、原始标签和疾病字典关联需分栏表达。

### 5.3 本地标签分布和多值情况

在 `mico_database_new/model/meta2db_species/meta2db_species_samples.csv` 中：

- 13,897 行中有 11,811 行可用 `group`/`disease`，2,086 行缺失，说明元数据匹配或回退不完整。
- `disease` 有 39 个不同原始值；`group` 有 `control`、`diseased`、`pre-diseased` 三类。
- `body_site` 有 2,683 行缺失；可用值主要包括 `feces`、`saliva`、`sputum`、`dental plaque`、`nasal lavage`。

在 `data/meta2db/metadata/meta2db_metadata.csv` 中：

- `disease_category` 有 39 个不同值，395 行包含逗号多值分隔。
- 详细字段中的多值分隔包括：`gastrointest_disord` 483 行、`metabolic_disord` 18 行、`neuro_disord` 50 行、`host_disease_stat` 1 行、`disease_notes` 62 行含逗号。
- 粗粒度类别示例：`neurological`、`cardiovascular`、`pulmonary`、`dermatologic`、`cancer`、`autoimmune`。
- 症状/表型示例：`diarrhea`、`obese`、`poor growth`、`arthralgia`、`lactose intolerant`。
- 病原体或感染相关标签示例：`SARS-CoV-2`、`HIV-1`、`cholera`、`plasmodium_falciparum`、`ascaris_lumbricoides`。
- 控制/非疾病语义可出现在疾病字段或详细字段中，例如 `control`、`healthy subject study`、`normal control`；`neuro_disord` 中还出现 `control`。
- 别名/拼写/命名不一致示例来自源码和原始字段：`T2D/type 2 diabetes`、`IBD/inflammatory bowel disease`、`CRC/colorectal cancer`、`PD/parkinson disease/parkinsons disease`、`AD/alzheimers`、`fatty_liver/fatty liver`、`premature born/prematurity`、`dental carries` 等。

因此，当前“130 个疾病名称”只能表述为**Dashboard 当前规则得到的名称集合**；它不是 `diseases` 表的 178 行，也不是人工审核后的 DiseaseAssertion 本体。

## 6. 受试者关系风险

- 远程实时只读核验（`2026-08-20T20:55:15+08:00`）按统一规则确认：`raw_metadata` 非空 11,811/13,897；有效 `raw_metadata.host_subject_id` 为 11,811 行，缺失 2,086 行；有效候选值 9,603 个；805 个候选值对应多行，最大对应 42 行；另有 142 个候选值跨多个 `project_name`。有效规则为 `TRIM` 后排除 NULL、空字符串及 `null`、`unknown`、`n/a`、`na`、`not available`、`not applicable`、`missing`、`none`。这些结果支持“可用于 Subject 候选核验”，不支持直接声明真实 Subject 数。
- 本地文件与远程 JSON 的 2,086 行缺失数字相同，但本地 `host_subject_id` 原始列的可用/缺失判定仍需按导入清洗规则复核；两者不能自动视为同一版本。
- `patient_id` 在 Java 端被当作 `patients` 的主关联键：患者详情、标准丰度、样本列表、健康参考和 API 路径全部围绕它构造（`PatientMapper.xml:38-43,50-119`、`AgentApiController.java:35-146`）。这适合当前业务记录查询，但不能作为独立 Subject 数量。
- 远程 `patients` 的 24,264 行只是内部业务记录表行数；Subject 去重仍需明确 `host_subject_id` 的跨 Study 规则、缺失处理和人工审核状态。

## 7. Agent 数据工具的质量风险清单

1. **身份混淆**：工具若只返回 `patientId`/`sampleId`，模型会把内部记录、样本和 Subject 混成“患者”。
2. **一对多丢失**：按 `patient_id` 聚合丰度可能把多个 Sample 混成一个 profile；必须强制指定 `sampleId` 或返回多个 profile。
3. **样本展示错误**：短 accession、完整 profile_sample、复合 sample_id 和 project_name 长度不同、语义不同，页面当前仍有“姓名”标签风险。
4. **受试者去重错误**：没有通过 `host_subject_id` 的重复关系核验前，队列不能按 `patient_id` 推断独立人数，也不能按 source field 无条件去重。
5. **疾病标签过度标准化**：别名、粗粒度、症状、病原体、控制组和多值标签不能由 LLM 自动写入正式映射。
6. **缺失和未匹配**：2,086 个本地样本标签缺失、年龄/性别/地区覆盖有限，统计工具必须返回缺失率和纳入/排除数。
7. **版本不可复现**：当前 Java `MicrobeAbundance` DTO 未返回 `featureVersion`、`sourceBatch`、`taxonomyVersion`；P1 工具必须补齐快照字段。
8. **源表语义需分层**：远程已确认 `microbe_abundance` 与 `microbe_abundance_standard` 是两张不同结构和索引的表；前者仍由 Java `SELECT *` 使用，不能把两者当作同义 TaxonomicProfile 来源。
9. **疾病查询不一致**：精确 `patient_diseases` 关联和 `patients.disease LIKE` 可能返回不同队列。
10. **统计口径误标**：Dashboard 的 “total_patients” 实际是 `COUNT(patient_id)`（`DashboardMapper.xml:78-80`）；未来必须改为“内部记录数”或明确口径。

## 8. 进入 P1 前的剩余数据治理事项

- 冻结远程 schema/索引摘要和本次核验时间，作为 P1 契约测试 fixture；不把密码、私钥或完整连接串放入 fixture。
- 继续核验标准丰度全表的特征唯一性、跨 `feature_version`/`source_batch` 的合并规则；当前表无特征行唯一键，`featureCount` 保持已存储行语义。
- 固化 `host_subject_id` 的 Subject 候选审核规则；P1-A 不需要等待人工疾病映射，但任何按 Subject 去重的 P1-B 工具必须返回 `unverified`。
- 输出 DiseaseAssertion 原始标签全集和多值拆分统计；由人工审核 `mapping_status`、映射版本和非疾病标签。
- 在 P1 API Schema 中强制要求 `dataSnapshotId`、`source`、`rowCount`、版本字段、`schemaVersion`、`queryHash` 和质量摘要。
