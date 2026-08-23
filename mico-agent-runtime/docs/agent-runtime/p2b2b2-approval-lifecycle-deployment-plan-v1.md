# P2-B2B.2 Runtime 审批生命周期迁移计划 v1

状态：代码与迁移资产已就绪；远程 `mico_agent_runtime` 的 `0002_approval_ticket_lifecycle` 尚待独立批准窗口执行。

## 目标

只为已经部署的独立 Runtime Schema 增加：

- `approval_ticket.operation`
- `approval_ticket.decided_by`
- 对 operation、reviewer principal 的 MySQL CHECK 约束

迁移只允许由 `MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL` 提供目标为
`mico_agent_runtime` 的 MySQL/asyncmy URL，并运行仓库脚本中的：

```text
alembic upgrade head
```

迁移-only 配置不读取 Runtime 状态加密 key；Runtime 应用启用仍必须同时
提供数据库 URL、AES-256-GCM 状态 key 和显式 key ID。

## 不可触碰范围

迁移不读取、不修改、不授权、不引用 `patient_data_manager`，不创建业务
账号，不写入测试数据，不接受自由 SQL。失败时停止，不自动 DROP 或回滚。

## 部署后只读验收

1. `alembic_version` 为 `0002_approval_ticket_lifecycle`。
2. `approval_ticket` 仍为 InnoDB、utf8mb4，且只增加上述两列和命名约束。
3. 五张 Runtime 表清单不变；没有业务外键、视图或授权。
4. `MySqlRuntimeStore` 应用账号仍只拥有 `mico_agent_runtime.*` 的
   `SELECT/INSERT/UPDATE/DELETE`，没有 DDL、`GRANT OPTION` 或业务库权限。
5. 只读验收结束关闭 loopback SSH 隧道并确认 `13306` 无监听。

## 启用顺序

完成迁移验收后，才允许注入 Runtime 应用账号、数据库 URL、状态加密 key、
key ID 和 Runtime Token；在此之前审批 API 保持默认关闭，不能宣称已完成
生产审批恢复演练。
