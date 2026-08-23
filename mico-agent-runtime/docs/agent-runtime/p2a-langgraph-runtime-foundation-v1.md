# Mico P2-A Python LangGraph Runtime Foundation v1

状态：P2-A 骨架与契约实现；动态查询入口已加入离线闭环，仍不调用真实 Java 服务、不连接 MySQL。

## 1. 信任边界

```text
Python Runtime
  └─ closed TaskPacket + deterministic LangGraph state
      └─ JavaAgentToolPort
          └─ POST {MICO_JAVA_AGENT_TOOL_BASE_URL}/internal/agent/tools/execute
              └─ Java Agent Tool API -> Java Read Model -> remote MySQL
```

Java 仍是唯一业务事实源。Python 不读取 Java `application.properties`、Mapper、SSH 配置、数据库配置、本地数据文件或数据库账号。Python 不包含 MySQL 驱动、SSH 客户端或数据库执行能力。动态 SQL 草案只作为模型输出的闭合字段，经 LangGraph 交给 Java，不在 Python 执行。

`MICO_JAVA_AGENT_TOOL_BASE_URL` 与 `MICO_AGENT_INTERNAL_TOKEN` 只由 `HttpJavaAgentToolPort.from_environment()` 读取。任一缺失时，客户端在建立 HTTP 请求前 fail-closed。P2-A 测试使用 fake port 或 `httpx.MockTransport`，不请求真实 Java 服务。

## 2. 闭合 TaskPacket

Runtime 的 Pydantic v2 模型使用 `extra="forbid"`。`TaskPacket` 字段为：

```text
runId
taskId
requesterId
requestedScopes
intent
input
allowedTools
createdAt
traceId
```

当前业务入口的 `IntentTaskRequest.question` 是自然语言；Planner 只能返回闭合的动态只读路由与 SQL 草案。Java Tool Call 的 `arguments` 仍是闭合结构，不能由浏览器直接注入。精确定位器若出现在未来专用工具中，仍只能是：

```text
(internalRecordId, sourceSampleId)
```

其中 `internalRecordId` 是 Mico 内部业务记录 ID，不是 Subject ID；Runtime 不把它转换、展示或解释为真实受试者。`sampleAccession` 只能用于候选检索，不能替代精确 locator。

当前 Java 可执行工具严格为：

```text
execute_read_query
```

该工具接受模型生成的 SQL 草案，但不接受 `patientId`、`subjectId`、`sampleKey`、数据库地址、Java Token、工具结果或内部审计字段。动态 SQL 不在 Python 执行，也不由 Python 直接连接 MySQL；它只能随闭合 ToolCall 发送给 Java，由 Java 最终校验和执行。请求 scope 不是认证，也不能提升 Java 权限。

## 3. Java 工具端口

HTTP port 只序列化以下四个顶层字段：

```text
toolName
runId
toolCallId
arguments
```

Python 不提交 `requesterId` 或 `requestedScopes`，由 Java 内部 API 绑定 `internal-agent-runtime` 及固定 effective scopes。HTTP 实现不记录 Authorization、Token、请求体或响应原始内容；非 200 响应和网络异常转换为不含内部细节的安全错误。

`arguments` 不是开放 `dict`：每个工具各有独立的 Java 调用模型，并由 `toolName` discriminator 选择。缺少必填参数、类型错误、未知参数、危险 locator 字段或跨工具参数都会在发起 HTTP 请求前拒绝。`execute_read_query` 是唯一允许携带 `sql` 的专用动态只读工具；SQL 只能来自模型闭合计划，不能由浏览器或普通固定工具注入。Java 响应还必须满足 `response.runId == request.runId` 且 `response.toolCallId == request.toolCallId`；缺失或不匹配时返回安全的 `JAVA_TOOL_RESPONSE_MISMATCH` 失败。

## 4. 实际 LangGraph 图

代码使用 LangGraph `StateGraph`，不是手写 if/else 替代品：

```text
START
  -> validate_task
  -> policy_gate
  -> build_tool_plan
  -> execute_tool
  -> summarize_evidence
  -> terminal
  -> END
```

拒绝/失败分支从 `validate_task`、`policy_gate` 或 `execute_tool` 直接进入 `terminal`。`execute_tool` 只调用注入的 `JavaAgentToolPort`；它不反射、不拼接 SQL、不推断工具参数。

- `validate_task`：重新验证闭合 Pydantic TaskPacket。
- `policy_gate`：仅允许研究数据检索和证据汇总；诊断、治疗、处方、风险结论、用户要求的无边界自由 SQL、工具未授权均拒绝。动态只读查询是受批准的研究工作流，不等同于开放数据库权限。
- `build_tool_plan`：固定输入从结构化 `input` 构造 Java 调用；动态意图图从闭合 `sqlDraft` 构造 `execute_read_query` 调用，并生成不含原始 locator 的稳定 `call-<32 位小写十六进制>` toolCallId。
- `execute_tool`：真实 Java `REJECTED`、`FAILED`、`NOT_IMPLEMENTED` 原样转为停止状态。
- `summarize_evidence`：只复制 Java 证据元数据和质量边界，不读取/推断医学结论。
- `terminal`：写入最终审计事件。

## 5. 证据边界

Java `COMPLETED` 响应必须携带 transient `dataSnapshot`。Runtime 摘要保留：

```text
source
dataSnapshotId
snapshotPersistence
queryHash
generatedAt
schemaVersion
rowCount
importBatch
diseaseMappingVersion
taxonomyVersion
featureVersion
sourceBatch
evidenceScope
```

`snapshotPersistence` 固定为 `transient`，`replayable` 固定为 `false`。Runtime 不创建 checkpoint、数据库快照或持久化审计存储，因此不能跨进程、跨时间回放。Java 返回 completed 但缺少 snapshot 时，Runtime 返回 `FAILED/EVIDENCE_MISSING`，不编造证据。

Java 原始 `dataSnapshot.cohortCondition` 只允许留在本次进程内的原始 `toolResult` 状态，不能进入 `EvidenceSummary`、FastAPI 响应或审计事件。对外仅允许固定无值枚举 `evidenceScope`，例如 `exact_record_profile_locator`；不输出 locator、样本 accession、内部记录 ID、可逆摘要或原始 payload。

Java `qualitySummary.warnings` 同样只保留在本次进程内的原始 `toolResult`，P2-A 不向外转发任何自由文本。P2-A 尚无版本化 `QualityWarningCode` 契约；以后必须先冻结代码型告警契约，才能安全暴露更多质量信息。

`HttpJavaAgentToolPort` 与 LangGraph `execute_tool` 均独立校验 `runId` 和 `toolCallId`。第二层校验失败时不写入 `toolResult`，不生成 `EvidenceSummary`，审计事件也不携带上游响应的 snapshot 或 payload。

Java P1 返回的 `dataSnapshotId` 采用 `transient-<小写 UUID>` 形状。Runtime 只原样保留该瞬态证据关联键，不把它改写为 Runtime `snapshot-...` ID，也不将其解释为可恢复 checkpoint；`snapshotPersistence=transient` 且 `replayable=false`。

`internalRecordId` 不是 Subject ID，结果不代表独立真实患者数；`subjectLinkStatus` 保持 `unverified`。疾病映射未审核时只保留 Java 的原始状态，不生成标准化疾病结论。

## 6. 审计与隐私

每个节点产生结构化事件，至少包括：

```text
traceId
runId
node
toolName
status
errorCode
dataSnapshotId
snapshotPersistence
occurredAt
```

事件不包含 Bearer Token、Java Token、请求头、数据库地址、SSH 信息、SQL、完整 locator 或丰度 payload。Runtime 对外只返回结构化证据摘要和审计事件；原始 Java `data` 只存在本次进程内的图状态。动态查询的 SQL 也不会进入审计、快照、BFF 响应或页面。

## 7. FastAPI 边界

`create_app()` 只构造 ASGI 应用，不启动 uvicorn、不监听端口。内部入口为：

```text
POST /internal/runtime/runs
```

Runtime Token 只从 `MICO_RUNTIME_INTERNAL_TOKEN` 读取：

- 未配置：503 `RUNTIME_DISABLED`；
- 缺失/错误 Bearer Token：401；
- JSON 闭合校验失败：400；
- 通过认证后才解析 TaskPacket、构造图并执行 fake/注入的 Runtime。

P2-A 不接前端，不提供公网路由，不调用真实 Java。没有真实 Token，也没有服务启动命令。动态 SQL 只允许经内部 Java Tool API 流转，不能由浏览器直接提交。

## 8. 动态意图与 SQL 草案边界

动态意图路线由 LangGraph 主导：

```text
validate_user_task -> policy_gate -> recognize_intent
  -> execute_selected_workflow -> terminal
```

当模型选择 `dynamic_read_query` 时，闭合 `IntentRoutePlan` 可以携带
`sqlDraft`。LangGraph 不直接执行这段文本，而是构造固定的
`execute_read_query` ToolCall 调用 Java。Java 返回的原始行只保留在本次
进程的工具结果中；对外报告只返回列名、行数、快照元数据和固定局限性。
没有模型或没有 SQL 草案时，Runtime 返回 `QUERY_PLAN_REQUIRED`，不会猜测
或自行拼接 SQL。

Java 的 `DynamicReadQueryPolicy` 仍强制单条 SELECT/CTE、允许动态表/列/JOIN/WHERE/GROUP BY/ORDER BY 形状、禁止
变更/危险操作/跨库引用、要求有界 `LIMIT`（含 `OFFSET` 但偏移也有上限）、设置最大行数和查询超时。MySQL 只读
账号是第二层防线，不能替代 Java 侧策略。

## 9. P2-A 未实现项

- LLM、自然语言规划、医学诊断/治疗/处方和风险结论；
- 真实 Java HTTP 调用与生产 Token 配置；
- 持久化 checkpoint、断点恢复、跨进程回放；
- 前端、MCP、真实 Java/MySQL 联调、完整差异统计、疾病映射和真实 Subject 去重；
- 真实用户身份传播和多租户权限。

依赖和离线测试已在项目专属 `.venv` 中完成；真实 Java endpoint 联调仍不属于 P2-A。P2-A 本身不修改 Java 项目。

## 9. 本次验证状态

依赖版本声明与参考项目 `ops-autoagent-diagnosis-python/pyproject.toml` 对齐。项目专属 `.venv` 使用 Python 3.12.13，依赖仅安装在该环境中：FastAPI 0.116.1、LangGraph 1.2.9、Pydantic 2.11.7、pytest 8.4.1。

已完成 Python 语法编译、Pydantic 合约 smoke check、固定 Java payload/响应关联检查、静态无数据访问扫描，以及真实 pytest 测试。依赖安装后运行：

```text
`.venv\Scripts\python.exe -m pip install -e ".[test]"`
`.venv\Scripts\python.exe -m pytest`

此前 P2-A 基线为 `46 passed`；动态查询改动后的最新测试结果以交付报告为准。测试使用 Fake Port 或 `httpx.MockTransport`，没有启动服务、调用真实 Java 或连接数据库。
```
