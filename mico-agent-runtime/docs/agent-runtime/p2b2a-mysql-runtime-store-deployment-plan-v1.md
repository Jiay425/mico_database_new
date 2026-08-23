# P2-B2A MySQL Runtime Store 部署方案 v1

状态：目标 Schema 已按 P2-B2B.1A 部署；Runtime 应用账号、密钥注入和应用持久化连通性仍未启用。

## 1. 目标与绝对边界

Runtime 状态库使用现有 MySQL 实例中的独立数据库：

```text
mico_agent_runtime
```

业务数据库 `patient_data_manager` 仍由 Java 业务系统独占。Runtime 账号不得拥有该库的任何权限，迁移脚本也不创建、引用或检查业务表。Python Runtime 不通过任何路径访问业务 MySQL 表；业务事实仍只能经 Java 受控工具获得。

生产预检要求 MySQL `8.0.16+`。该版本门槛用于确认 CHECK 约束按预期执行；还必须确认 InnoDB、utf8mb4、JSON 类型、事务隔离级别、时钟与连接池策略均符合部署标准。

## 2. 账号与最小权限

### migration 账号

由 DBA 临时创建，仅在迁移窗口使用。它需要在 `mico_agent_runtime` 库内创建/变更五张 Runtime 表、索引、外键和 CHECK 约束的权限。迁移完成并通过 schema 验收后，撤销或禁用该账号；它不获得 `patient_data_manager` 权限。

### runtime 账号

应用运行账号只对 `mico_agent_runtime.*` 授予最小的：

```text
SELECT, INSERT, UPDATE, DELETE
```

不得授予 `CREATE`、`ALTER`、`DROP`、`INDEX`、`REFERENCES`、`GRANT OPTION` 或全局权限；不得授予 `patient_data_manager.*` 的任何权限。部署验收必须以 `SHOW GRANTS` 和一条受控权限审计结果证明边界，审计输出不得包含密码。

## 3. 配置与秘密

部署环境注入以下配置，仓库不提供默认值：

```text
MICO_AGENT_RUNTIME_MYSQL_ENABLED=true
MICO_AGENT_RUNTIME_DATABASE_URL=<secret-manager-injected MySQL URL>
MICO_RUNTIME_STATE_ENCRYPTION_KEY=<secret-manager-injected Base64URL key>
MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID=<secret-manager-injected key identifier>
```

URL、账号、密码、加密密钥和完整连接串不得写入 `.env`、镜像层、Git、日志、错误响应或文档示例。key ID 必须与密钥轮换记录绑定；不得从密钥内容推导或打印密钥材料。

## 4. 部署顺序与验证点

以下步骤只在 P2-B2B 获得人工批准后执行：

1. 预检实例版本为 MySQL 8.0.16+，确认 TLS、InnoDB、utf8mb4、JSON、CHECK 约束、UTC 时钟和备份可用性。
2. 目标 `mico_agent_runtime` 已创建并通过只读 manifest 验收；任何后续环境必须确认名称不是 `patient_data_manager`。
3. 在迁移窗口创建临时 migration 账号，仅授权 `mico_agent_runtime` 的必要 DDL/索引/外键权限。
4. 审核本项目 `alembic/versions/0001_agent_runtime_state.py` 的 revision、五表清单和预期约束；Schema 部署使用 migration-only URL 执行过一次 `alembic upgrade head`，不将 DSN 写入日志。
5. 用 `SHOW TABLES`、`SHOW CREATE TABLE`、`SHOW INDEX` 和约束核验确认仅有五张 Runtime 表，全部为 InnoDB/utf8mb4，JSON 列为 MySQL JSON，ID/状态/代码/哈希/opaque 引用约束存在，且无业务表引用。
6. 撤销或禁用 migration 账号，创建 runtime 账号并仅授予 `mico_agent_runtime.*` 的四项 DML 权限。
7. 用 runtime 账号执行不写入数据的健康检查：连接、事务启动、五表可读、权限审计；明确验证访问 `patient_data_manager` 被拒绝，且 runtime 账号不能执行 DDL。
8. 配置密钥管理、连接池、TLS 和告警；再由应用部署审批决定是否显式开启 Runtime Store。当前 P2-B2A 不接入 FastAPI 生命周期。
9. 建立迁移后备份与回滚记录。回滚只能按经审批的迁移/备份流程执行，不在应用启动时自动清理数据。

## 5. TLS 与 SSH 隧道

直连 MySQL 时必须启用并验证 MySQL TLS、证书校验、允许的 TLS 版本和服务器身份。若因网络边界使用 SSH 隧道，隧道监听只允许 loopback（例如 `127.0.0.1`），不得绑定 `0.0.0.0` 或暴露公网；隧道密钥由部署系统管理，不写仓库，不进入应用日志。隧道生命周期、端口冲突、关闭确认和断开告警由部署平台负责。

## 6. UTC 与备份

Runtime 只接受 aware datetime，写入 MySQL `DATETIME` 前转换为 UTC，读取后恢复 UTC aware datetime。部署健康检查应验证主机时钟、连接会话时区策略和备份恢复后的时间解释一致。备份必须覆盖五张 Runtime 表、Alembic revision 和密钥 ID 关联记录；密钥材料由独立密钥系统管理，不与数据库备份混存。

## 7. 当前部署状态与 P2-B2B 后续人工确认项

- DBA/部署核验已确认 MySQL 版本 `8.0.46` 和 `mico_agent_runtime` 已部署。
- 已确认不会触碰 `patient_data_manager`，并完成网络 ACL/TLS 评审。
- 已批准 migration 账号的临时最小权限、窗口、审计与撤销步骤。
- 已批准 runtime 账号仅有 `mico_agent_runtime.*` 的 `SELECT/INSERT/UPDATE/DELETE`，无 DDL、无业务库权限。
- 仍需准备 secret manager 注入的数据库 URL、32 字节 Base64URL 加密密钥和显式 key ID。
- 已完成备份、回滚、连接池、健康检查、告警和隧道 loopback 方案。
- `mico_agent_runtime` 与 `0001_agent_runtime_state` 已完成实际部署和只读 manifest 验收；`0002_approval_ticket_lifecycle` 尚未执行。
- 真实 Runtime 应用账号、密钥注入、应用生命周期持久化连通性和审批字段迁移仍必须单独审批并验收。
