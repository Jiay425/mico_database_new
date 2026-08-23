# Mico Agent Runtime P2-B1/P2-B2A 持久化契约 v1

状态：P2-B2B.1A 已实际部署 `mico_agent_runtime` 与 `0001_agent_runtime_state`；审批字段扩展由 `0002_approval_ticket_lifecycle` 管理，Runtime 应用账号、密钥注入和应用持久化连通性仍需单独启用与验收。

## 1. 绝对数据边界

Runtime 状态库是现有 MySQL 实例中的独立数据库 `mico_agent_runtime`，只服务于 Agent Runtime 的运行编排状态、步骤状态、审批状态、产物元数据和安全工具审计。它不是业务事实库，也不承载患者、样本、疾病或丰度数据。

只允许包含五张 Runtime 表：

```text
agent_run
agent_step
agent_artifact
approval_ticket
tool_audit
```

Java Read Model / `patient_data_manager` 仍是业务事实唯一来源；Python 不直连业务 MySQL。Runtime MySQL 不使用、不迁移、不引用业务表，也不会退化到 SQLite、业务 MySQL 或内存存储作为生产后备。

配置必须同时满足：

```text
MICO_AGENT_RUNTIME_MYSQL_ENABLED=true
MICO_AGENT_RUNTIME_DATABASE_URL=mysql+asyncmy://.../mico_agent_runtime
MICO_RUNTIME_STATE_ENCRYPTION_KEY=<deployment-injected-value>
MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID=<deployment-injected-value>
```

URL、密钥、账号和密码不写入仓库、文档示例、日志或错误响应。配置错误在建立连接前以固定错误码 fail-closed。`patient_data_manager` 永远不是 Runtime 数据库。

## 2. 五类实体与允许保存的信息

所有实体都是 Pydantic v2 闭合模型，`dataContractVersion` 固定为 `v1`；没有自由 JSON 作为审计输出。实体 ID 使用专属前缀加 32 位小写十六进制不透明 ID：

| ID 类型 | 格式 |
|---|---|
| `runId` | `run-<32hex>` |
| `taskId` | `task-<32hex>` |
| `traceId` | `trace-<32hex>` |
| `stepId` | `step-<32hex>` |
| `approvalId` | `approval-<32hex>` |
| `auditId` | `audit-<32hex>` |
| `artifactId` | `artifact-<32hex>` |
| `toolCallId` | `call-<32hex>` |
| `requestedBy` | `principal-<32hex>` |
| `dataSnapshotId` | `transient-<小写 UUID>`（Java P1 evidence ID） |

前缀互不兼容。样本 accession、项目名、locator、URL、路径和业务数据库名不能伪装成 Runtime ID。`dataSnapshotId` 是 Java P1 的瞬态事实证据关联键，不是 Runtime checkpoint ID；必须原样保存 `transient-<UUID>`，保持 `snapshotPersistence=transient`、`replayable=false`，不能用于跨进程恢复或运行重放。

| 实体 | 职责 | 允许保存 | 明确禁止 |
|---|---|---|---|
| `agent_run` | 运行生命周期 | run/task/trace、固定状态、UTC 时间、成功步骤、控制失败码、加密恢复状态、显式 key ID | 明文任务状态、locator、Java data、Token、请求头 |
| `agent_step` | 节点尝试与安全摘要 | step/run、固定节点、尝试、状态、UTC 时间、控制码、有限 Runtime step code、规范化 snapshot 元数据 | 自由摘要、原始 arguments、Java data、Java warning、完整 locator |
| `agent_artifact` | 未来产物元数据索引 | 固定类型、schema 版本、严格内容摘要、绑定后的 opaque 引用、snapshot ID、UTC 时间 | 真实统计内容、S3/HTTP/文件路径、连接串、原始 payload |
| `approval_ticket` | 审批状态模型 | approval/run、固定 operation、请求方、固定审批状态、UTC 时间、绑定后的 reviewer principal、控制决策码 | 审批页面内容、自由审批文本、执行凭证 |
| `tool_audit` | 工具调用审计 | 工具名、状态、耗时、run/toolCall 关联、控制错误码、规范化 snapshot 元数据 | 原始 arguments、Java payload、Token、请求头、数据库地址 |

`AgentStepRecord` 的 `safeInputCode`、`safeOutputCode` 只接受固定代码：

```text
TASK_CONTRACT_VALIDATED
POLICY_ALLOWED
TOOL_PLAN_BUILT
JAVA_TOOL_COMPLETED
JAVA_TOOL_REJECTED
JAVA_TOOL_FAILED
EVIDENCE_METADATA_CREATED
RUN_QUEUED
RUN_COMPLETED
RUN_FAILED
APPROVAL_REQUESTED
```

`failureCode`、`errorCode`、`decisionCode` 是 Runtime 控制码，只接受大写下划线代码格式，不承载自由错误文本、上游 warning 或 payload。工具名固定为当前五个 Java 工具；产物类型当前只有 `evidence_summary`。

## 3. 产物、版本与哈希

`artifactStorageRef` 必须严格等于 `artifact://<artifactId>`，其中 `<artifactId>` 必须符合 `artifact-<32hex>`。本阶段没有对象存储，不生成真实 bucket、S3、文件系统路径、下载链接或外部位置。

Snapshot 版本字段使用字段专属受限 token 类型，允许真实值，例如：

```text
meta2db-species-20260726:2015_Castro-NallarE
meta2db-species-v1
p1b2-java-read-only-tool-executor-v1
```

字段类型拒绝 URL、路径、SQL、Bearer Token、locator 语法、JSON/payload 和自由句子；它不硬编码具体样本或项目名称。`contentHash`、`queryHash` 只接受 `sha256:<64位小写十六进制>`。

## 4. 加密状态与时间边界

`RuntimeStateCipher` 使用成熟 `cryptography` 库的 AES-256-GCM。密钥只从 `MICO_RUNTIME_STATE_ENCRYPTION_KEY` 读取，必须是 32 字节随机密钥的 Base64URL 编码；显式 key ID 只从 `MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID` 读取，不从密钥哈希推导。任一缺失、格式错误或长度错误时 fail-closed。

AAD 固定绑定：

```text
runId
taskId
traceId
dataContractVersion=v1
```

解密失败只返回 `RUNTIME_STATE_DECRYPT_FAILED`。密钥、明文、nonce、密文、完整 locator 和 Java payload 不进入日志、异常文本、审计模型或 FastAPI 输出。

MySQL `DATETIME` 不保存时区。Runtime 只接受 aware datetime，在写入前转换为 UTC 的无时区值，读取后按 UTC 恢复 aware datetime；不接受 naive datetime。

## 5. 状态机与 MySQL Store 能力

状态固定为：

```text
QUEUED -> RUNNING | CANCELLED
RUNNING -> WAITING_APPROVAL | WAITING_JOB | COMPLETED | FAILED | CANCELLED
WAITING_APPROVAL -> RUNNING | FAILED | CANCELLED
WAITING_JOB -> RUNNING | FAILED | CANCELLED
COMPLETED / FAILED / CANCELLED -> terminal
```

`MySqlRuntimeStore` 使用 SQLAlchemy 2.x async ORM + `asyncmy`，固定参数化操作、事务和行锁；没有自由 SQL、动态表名、业务库回退或连接外部数据库的接口。唯一键/完整性冲突映射为 `RUNTIME_RUN_ALREADY_EXISTS` 等稳定领域码，不暴露 SQL、表名、DSN、主机、账号或原始异常。

```text
persistent=True
recoverable=False
```

`InMemoryRuntimeStore` 保持 `persistent=False`、`recoverable=False`，仅供测试依赖注入。Runtime Store 本身仍不宣称可恢复；显式注入 `EncryptedLangGraphCheckpointSaver` 后，Runtime 才获得单一最新 checkpoint 的受控恢复能力。它不提供时间旅行、分支复制或任意 checkpoint 查询。

## 6. MySQL 8 迁移资产

`alembic/versions/0001_agent_runtime_state.py` 已在 P2-B2B.1A 部署到专用 Runtime Schema：五张表使用 InnoDB、utf8mb4；元数据使用 MySQL JSON；正则校验使用 `REGEXP`；artifact 绑定使用 `CONCAT('artifact://', artifact_id)`；时间列按 UTC `DATETIME` 处理。迁移没有创建、引用或迁移 `patient_data_manager` 的任何表。审批票据字段扩展位于未执行的 `0002_approval_ticket_lifecycle`。

P2-B2B.1A 已执行 `alembic upgrade head` 并通过目标 Schema 只读验收；后续应用账号、审批字段迁移和持久化启用仍必须完成部署审批、权限核验、TLS/隧道核验、备份和回滚演练。

## 7. 完成状态与离线验证

已完成：独立五实体闭合模型；专属 opaque Runtime ID；Java transient snapshot ID；字段专属版本元数据；固定 step code、工具名、产物类型、控制码；opaque artifact reference；严格 SHA-256；AES-GCM codec、AAD、显式 key ID；InMemory 状态机 Store；MySQL 8 ORM 元数据、固定 ORM 端口和版本化迁移文件；加密 LangGraph checkpoint 适配器；EvidenceReviewReport 的安全 artifact 元数据投影；离线安全测试。

当前 `RuntimePersistenceCoordinator` 已可通过依赖注入把一次运行的加密请求状态、加密 LangGraph 检查点、固定步骤、工具审计、Evidence artifact 摘要和安全快照元数据提交给 `RuntimeStore`；检查点只更新 `agent_run.encrypted_state_payload`，不会把明文投影到普通列。应用默认不创建 Store。只有显式启用 Runtime MySQL 且 URL、加密 key、key ID 均通过环境/密钥管理校验后，入口才会在图执行前建立运行记录，持久化失败则 fail-closed。进度接口还可在进程内投影缺失时从 `agent_run`/`agent_step` 重建安全的有限进度视图；内部 resume 只能从加密 checkpoint 恢复，不能接受客户端状态。该协调器不读取业务 MySQL，也不保存原始 locator、Java data、SQL 或自由 warning。

不能宣称：Runtime 应用账号已启用或应用持久化连通性已验收；远程生产崩溃恢复演练；运行重放/时间旅行；Redis/SSE 长任务队列；审批页面或完成 `0002_approval_ticket_lifecycle` 部署后的真实审批执行流；Docker Compose；真实统计产物对象存储或下载；业务 MySQL 访问。`0002_approval_ticket_lifecycle` 尚未在远程 Schema 执行，当前远程 manifest 仍是 `0001_agent_runtime_state` 的五表部署。

测试只使用项目 `.venv`，不连接任何数据库，也不执行迁移：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
```
