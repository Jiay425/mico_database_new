# P2-B2B.0 MySQL Runtime Schema 远程只读预检报告 v1

核验时间：2026-08-21 18:28:31 +08:00  
核验方式：临时 loopback SSH 隧道，MySQL 只读查询  
目标 Schema：`mico_agent_runtime`  
状态：预检完成；未进入 P2-B2B.1

## 1. 安全边界与执行范围

本次仅通过临时 SSH 隧道访问远程 MySQL 的 `127.0.0.1:3306`，本地监听只绑定 loopback。查询限定为服务变量、当前身份、`SHOW GRANTS` 和 `information_schema` 中目标 Schema 的元数据。

未读取 `patient_data_manager` 的表、样本、疾病或丰度数据；未执行任何 DDL、DML、迁移、账号创建、授权、配置写入或清理操作。

临时隧道已关闭，核验后 `13306` 无监听，且无遗留 SSH 进程。

## 2. 核验结论总表

| 核验项 | 状态 | 安全事实证据 | 结论/限制 |
|---|---|---|---|
| 服务端产品 | PASS | `@@version_comment = MySQL Community Server - GPL`；产品判定为 MySQL | 不是 MariaDB |
| 服务端版本 | PASS | `@@version = 8.0.46` | 满足 `>= 8.0.16` |
| 服务端字符集 | PASS | `character_set_server = utf8mb4` | 满足 Runtime 迁移要求 |
| 服务端排序规则 | PASS | `collation_server = utf8mb4_unicode_ci` | 支持 utf8mb4；Runtime 表将由迁移显式使用 `utf8mb4_bin` |
| MySQL TLS 能力 | PASS | `have_ssl = YES`；`tls_version = TLSv1.2,TLSv1.3`；当前会话为 TLSv1.3 | 原生 TLS 可用 |
| 原生 TLS 强制策略 | BLOCKED | `require_secure_transport = OFF` | 直连部署前必须补充强制 TLS 策略，或经批准使用 loopback SSH 隧道 |
| 当前会话加密 | PASS（本次连接） | `Ssl_version = TLSv1.3`；`Ssl_cipher = TLS_AES_256_GCM_SHA384` | 仅证明本次会话加密，不证明服务端强制 TLS |
| `mico_agent_runtime` 冲突 | PASS | `information_schema.SCHEMATA` 未返回该 Schema | 当前可创建；本次未创建 |
| 当前管理身份能力 | PASS（仅能力评估） | `CURRENT_USER()` 为管理身份；`SHOW GRANTS` 显示全局 `*.*` 权限、创建用户能力及 `WITH GRANT OPTION` | 具备后续一次性部署能力，但不是 Runtime 应用账号 |
| 最小 Runtime 应用账号 | BLOCKED | 当前未创建/未授权专用 Runtime 账号 | P2-B2B.1 必须单独创建并核验 |
| 预期迁移资产 | BLOCKED（未执行） | 远程 Schema 不存在，无法做部署后表/版本核对 | 只能在批准的迁移窗口执行后验证 |

## 3. MySQL 兼容性与 TLS 解释

远程服务满足 MySQL 8.0.46、utf8mb4 和 InnoDB/JSON/Check 约束所需的版本前提。服务端声明支持 TLS 1.2/1.3，本次只读连接实际协商为 TLS 1.3。

但是 `require_secure_transport=OFF`，表示服务端没有强制所有连接使用 TLS。因此不能把“当前连接使用 TLS”写成“服务端已强制 TLS”。若 P2-B2B.1 采用直连，必须先完成 MySQL TLS 强制策略、证书校验和客户端配置验收。

本次使用的 SSH 转发仅绑定本机 loopback，并将远程 MySQL 的 loopback 端口转发到本地。该边界在网络上不暴露 MySQL 端口，SSH 通道本身提供传输加密；它不等同于 MySQL 原生 TLS，也不替代原生 TLS 能力核验。若采用该方案，部署平台必须保证隧道端口只监听 `127.0.0.1`，不得绑定公网地址。

## 4. 目标 Schema 冲突检查

使用以下只读元数据范围核验：

```text
information_schema.SCHEMATA
information_schema.TABLES
information_schema.COLUMNS
information_schema.TABLE_CONSTRAINTS
information_schema.STATISTICS
```

查询条件限定为 `SCHEMA_NAME/TABLE_SCHEMA = 'mico_agent_runtime'`。结果为：

- `mico_agent_runtime` 当前不存在；
- 因 Schema 不存在，没有现有表可列出；
- 没有现有 `alembic_version` 可读取，版本状态为“未部署”；
- 结论为“可创建”，但本次没有创建。

P2-B2B.1 创建后，预期只允许存在：

```text
agent_run
agent_step
agent_artifact
approval_ticket
tool_audit
alembic_version
```

其中五张业务无关的 Runtime 表由已审核初始迁移创建，`alembic_version` 只记录迁移版本。不得出现业务表或业务外键。

## 5. 当前身份与权限评估

只读执行了 `CURRENT_USER()`、`USER()` 和 `SHOW GRANTS`。当前身份为 `root@%`，授权范围为全局 `*.*`，包含创建用户、建表/变更等管理能力，并带有 `WITH GRANT OPTION`。

这证明当前管理身份具备后续创建独立 Schema、创建受限账号和授予该 Schema 权限的能力；同时也证明它不符合 Runtime 应用账号的最小权限要求。当前身份不得写入 Runtime 配置，也不得作为 Python Runtime 的应用账号。

最终 Runtime 应用账号必须满足：

```text
SELECT, INSERT, UPDATE, DELETE ON mico_agent_runtime.*
```

并且同时满足：

- 无 `patient_data_manager` 或其他业务库权限；
- 无 `CREATE`、`ALTER`、`DROP`、`INDEX`、`REFERENCES` 等 DDL/结构权限；
- 无 `GRANT OPTION`；
- Host 不得使用 `%`。

## 6. 待执行但本次不执行的部署清单

### 6.1 Schema 与迁移

1. 由 DBA 在批准窗口创建 `mico_agent_runtime`，默认字符集使用 `utf8mb4`。
2. 仅使用已审核的 `alembic upgrade head` 执行迁移；不得手工复制表结构或执行自由 SQL。
3. 验证五张 Runtime 表和 `alembic_version`，所有表为 InnoDB、utf8mb4；JSON 列为 MySQL JSON；ID、状态、代码、哈希、opaque 引用约束存在。
4. 验证没有 `patient_data_manager` 引用、外键或表访问授权。

### 6.2 账号与 Host 假设

本部署方案假设 Python Runtime 与 SSH loopback 隧道在同一受控主机上运行；在该假设下，建议 Runtime 账号 Host 使用 `localhost`。这是部署假设，不是本次已创建或已验证的账号事实；P2-B2B.1 必须根据实际进程位置和 MySQL 连接来源重新确认。

不得使用 Host `%`。若实际 Runtime 位于其他受控主机，必须使用明确的固定来源地址或经批准的受限网络身份，并重新验证权限范围。

### 6.3 配置与密钥

以下配置必须由部署环境或密钥管理系统注入：

```text
MICO_AGENT_RUNTIME_MYSQL_ENABLED=true
MICO_AGENT_RUNTIME_DATABASE_URL=<secret-manager-injected-value>
MICO_RUNTIME_STATE_ENCRYPTION_KEY=<secret-manager-injected-value>
MICO_RUNTIME_STATE_ENCRYPTION_KEY_ID=<secret-manager-injected-value>
MICO_AGENT_INTERNAL_TOKEN=<secret-manager-injected-value>
```

不得把 URL、密码、状态加密密钥、key ID、Java Token 或 SSH 私钥路径写入仓库、报告、日志、镜像或 `.env`。

### 6.4 验证、备份与回滚

- 迁移前：确认实例版本、TLS/SSH 边界、Schema 不存在、备份可用、migration 账号临时权限已审批。
- 迁移后：使用 `SHOW CREATE TABLE`、`SHOW INDEX`、`information_schema` 和 `alembic_version` 验证五表及约束；用 Runtime 账号验证只能 DML 访问 Runtime Schema。
- 权限后：验证 Runtime 账号访问 `patient_data_manager` 被拒绝，且不能执行 DDL 或授权操作。
- 备份：备份五张 Runtime 表、`alembic_version` 和版本元数据；密钥材料由独立密钥系统管理，不与数据库备份混存。
- 回滚：只允许执行已审批的 Alembic 回滚或恢复备份；不得删除、更新或清理 `patient_data_manager`，不得自动回滚业务库。

## 7. 进入 P2-B2B.1 的精确前置条件

当前不能直接进入实际部署，必须先满足：

1. DBA 确认 MySQL 8.0.46 实例、网络 ACL 和备份策略。
2. 明确选择并批准直连 TLS 或 loopback SSH 隧道方案；若直连，先解决 `require_secure_transport=OFF`；若隧道，确认只监听 loopback。
3. 人工批准 `mico_agent_runtime` 建库和已审核 Alembic 初始迁移窗口。
4. 人工批准一次性 migration 账号的最小 DDL 权限，并在迁移后撤销/禁用。
5. 创建 Runtime 应用账号，Host 不使用 `%`，只授予 `mico_agent_runtime.*` 的四项 DML 权限，无业务库权限、无 DDL、无 `GRANT OPTION`。
6. 由部署环境注入 Runtime MySQL URL、状态加密 key、key ID 和 Java Token；仓库和报告中不出现秘密。
7. 完成迁移后 Schema、索引、约束、权限、备份和回滚演练的验收标准。

在上述条件全部确认前，本预检结论为：目标 Schema 可创建，但安全部署条件尚未全部闭合；不执行 P2-B2B.1。
