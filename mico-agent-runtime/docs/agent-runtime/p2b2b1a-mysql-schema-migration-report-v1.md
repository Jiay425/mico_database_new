# P2-B2B.1A MySQL Runtime Schema 与初始迁移部署报告 v1

执行时间：2026-08-21（Asia/Shanghai）  
目标：`mico_agent_runtime`  
结论：`PASS`（仅目标 Runtime Schema 已部署；应用持久化尚未启用）

## 1. 执行边界

本次只使用临时 SSH loopback 隧道和一次性管理身份完成目标 Schema 创建及已审核
Alembic 初始迁移。隧道本地监听仅为 `127.0.0.1:13306`，迁移完成后已关闭；收尾
核验确认无 `13306` 监听、无遗留 SSH 进程。

本次未读取、修改、授权或迁移 `patient_data_manager`，未创建 Runtime 应用账号，
未启动 Java、Python、uvicorn、Docker 或长驻服务。未插入测试数据。

## 2. 写入前复核

| 项目 | 状态 | 安全事实 |
|---|---|---|
| 产品 | PASS | MySQL Community Server |
| 版本 | PASS | `8.0.46`，满足 `>= 8.0.16` |
| 服务端字符集 | PASS | `utf8mb4` |
| 目标 Schema | PASS | 写入前 `information_schema.SCHEMATA` 计数为 `0` |
| 隧道边界 | PASS | 本地端口仅绑定 loopback |
| migration-only 配置 | PASS | Alembic 仅读取 `MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL`；不要求 Runtime state key/key ID |

## 3. 实际执行

1. 创建唯一目标 Schema `mico_agent_runtime`，默认字符集为 `utf8mb4`，二进制排序规则为 `utf8mb4_bin`。
2. 仅执行仓库已审核的：

   ```text
   alembic upgrade head
   ```

3. 未创建 Runtime 应用账号或 `%` Host 账号，未将管理身份写入应用配置。
4. 未执行测试 DML、业务表扫描或清理操作。

## 4. 只读验收 manifest

| 项目 | 实际值 |
|---|---|
| Schema | `mico_agent_runtime` |
| Alembic revision | `0001_agent_runtime_state` |
| 表 | `agent_artifact`, `agent_run`, `agent_step`, `alembic_version`, `approval_ticket`, `tool_audit` |
| 存储引擎 | 六张表均为 `InnoDB` |
| 排序规则 | 六张表均为 `utf8mb4_bin` |
| JSON 列 | `agent_step.snapshot_metadata`, `tool_audit.snapshot_metadata`，类型为 MySQL `JSON` |
| 业务库外键计数 | `0`（按目标 Schema 的外键元数据核验） |
| 目标 Schema 视图 | 无 |

外键只指向 Runtime 自身的 `agent_run`；目标 Schema 中没有指向
`patient_data_manager` 的外键、视图或表。未读取业务表数据。

## 5. 当前未启用能力

Schema 已部署不等于 Runtime 已可持久化运行。以下事项仍未完成：

- Runtime 应用账号及最小权限授权；
- 环境/密钥管理系统注入数据库 URL、状态加密 key 和显式 key ID；
- `MySqlRuntimeStore` 的真实应用连通性和事务健康检查；
- LangGraph checkpoint 恢复、运行重放和崩溃后继续。

因此不能宣称 Agent Runtime 已启用持久化或可恢复运行。

## 6. 回归测试

在项目专属 `.venv` 中执行：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

本报告对应的部署验收后回归结果记录在最终交付报告中；测试使用本地 fake/mock，
不连接远程 MySQL。
