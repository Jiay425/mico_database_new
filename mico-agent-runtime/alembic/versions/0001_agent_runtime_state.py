"""Create independent Agent Runtime state tables in MySQL 8.

Revision ID: 0001_agent_runtime_state
Revises:
Create Date: 2026-08-21

This migration is an offline asset only. It must be reviewed and run against
the dedicated ``mico_agent_runtime`` database in a separately approved
deployment window.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision = "0001_agent_runtime_state"
down_revision = None
branch_labels = None
depends_on = None

_RUN_STATUS = "'QUEUED','RUNNING','WAITING_APPROVAL','WAITING_JOB','COMPLETED','FAILED','CANCELLED'"
_TOOL_STATUS = "'COMPLETED','REJECTED','FAILED','NOT_IMPLEMENTED'"
_APPROVAL_STATUS = "'PENDING','APPROVED','REJECTED','EXPIRED','CANCELLED'"
_STEP_CODES = "'TASK_CONTRACT_VALIDATED','POLICY_ALLOWED','TOOL_PLAN_BUILT','JAVA_TOOL_COMPLETED','JAVA_TOOL_REJECTED','JAVA_TOOL_FAILED','EVIDENCE_METADATA_CREATED','RUN_QUEUED','RUN_COMPLETED','RUN_FAILED','APPROVAL_REQUESTED'"
_CONTROL_CODE_PATTERN = "'^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$'"
_KEY_IDENTIFIER_PATTERN = "'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'"
_RUN_ID_PATTERN = "'^run-[0-9a-f]{32}$'"
_TASK_ID_PATTERN = "'^task-[0-9a-f]{32}$'"
_TRACE_ID_PATTERN = "'^trace-[0-9a-f]{32}$'"
_STEP_ID_PATTERN = "'^step-[0-9a-f]{32}$'"
_APPROVAL_ID_PATTERN = "'^approval-[0-9a-f]{32}$'"
_AUDIT_ID_PATTERN = "'^audit-[0-9a-f]{32}$'"
_ARTIFACT_ID_PATTERN = "'^artifact-[0-9a-f]{32}$'"
_CALL_ID_PATTERN = "'^call-[0-9a-f]{32}$'"
_PRINCIPAL_ID_PATTERN = "'^principal-[0-9a-f]{32}$'"
_SNAPSHOT_ID_PATTERN = "'^transient-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'"
_TOOL_NAMES = "'execute_read_query','literature_evidence'"
_ARTIFACT_TYPES = "'evidence_summary'"
_NODE_NAMES = "'validate_task','policy_gate','build_tool_plan','execute_tool','summarize_evidence','terminal'"

_TABLE_OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
}


def upgrade() -> None:
    op.create_table(
        "agent_run",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("data_contract_version", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("current_successful_step_id", sa.String(length=128)),
        sa.Column("failure_code", sa.String(length=128)),
        sa.Column("encrypted_state_payload", sa.Text(), nullable=False),
        sa.Column("encryption_key_id", sa.String(length=128), nullable=False),
        sa.CheckConstraint(f"status IN ({_RUN_STATUS})", name="ck_agent_run_status"),
        sa.CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_agent_run_id"),
        sa.CheckConstraint(f"task_id REGEXP {_TASK_ID_PATTERN}", name="ck_agent_run_task_id"),
        sa.CheckConstraint(f"trace_id REGEXP {_TRACE_ID_PATTERN}", name="ck_agent_run_trace_id"),
        sa.CheckConstraint(f"current_successful_step_id IS NULL OR current_successful_step_id REGEXP {_STEP_ID_PATTERN}", name="ck_agent_run_success_step_id"),
        sa.CheckConstraint(f"encryption_key_id REGEXP {_KEY_IDENTIFIER_PATTERN}", name="ck_agent_run_encryption_key_id"),
        sa.CheckConstraint("data_contract_version = 'v1'", name="ck_agent_run_contract_version"),
        sa.CheckConstraint(f"failure_code IS NULL OR failure_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_agent_run_failure_code"),
        **_TABLE_OPTIONS,
    )
    op.create_index("ix_agent_run_task_id", "agent_run", ["task_id"])
    op.create_index("ix_agent_run_trace_id", "agent_run", ["trace_id"])
    op.create_index("ix_agent_run_status_updated_at", "agent_run", ["status", "updated_at"])

    op.create_table(
        "agent_step",
        sa.Column("step_id", sa.String(length=128), primary_key=True),
        sa.Column("run_id", sa.String(length=128), sa.ForeignKey("agent_run.run_id"), nullable=False),
        sa.Column("data_contract_version", sa.String(length=16), nullable=False),
        sa.Column("node_name", sa.String(length=128), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=False)),
        sa.Column("error_code", sa.String(length=128)),
        sa.Column("safe_input_code", sa.String(length=64)),
        sa.Column("safe_output_code", sa.String(length=64)),
        sa.Column("snapshot_metadata", mysql.JSON()),
        sa.CheckConstraint(f"status IN ({_RUN_STATUS})", name="ck_agent_step_status"),
        sa.CheckConstraint(f"step_id REGEXP {_STEP_ID_PATTERN}", name="ck_agent_step_id"),
        sa.CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_agent_step_run_id"),
        sa.CheckConstraint("attempt_number >= 1 AND attempt_number <= 100", name="ck_agent_step_attempt"),
        sa.CheckConstraint(f"node_name IN ({_NODE_NAMES})", name="ck_agent_step_node_name"),
        sa.CheckConstraint("data_contract_version = 'v1'", name="ck_agent_step_contract_version"),
        sa.CheckConstraint(f"error_code IS NULL OR error_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_agent_step_error_code"),
        sa.CheckConstraint(f"safe_input_code IS NULL OR safe_input_code IN ({_STEP_CODES})", name="ck_agent_step_input_code"),
        sa.CheckConstraint(f"safe_output_code IS NULL OR safe_output_code IN ({_STEP_CODES})", name="ck_agent_step_output_code"),
        **_TABLE_OPTIONS,
    )
    op.create_index("ix_agent_step_run_started_at", "agent_step", ["run_id", "started_at"])

    op.create_table(
        "agent_artifact",
        sa.Column("artifact_id", sa.String(length=128), primary_key=True),
        sa.Column("run_id", sa.String(length=128), sa.ForeignKey("agent_run.run_id"), nullable=False),
        sa.Column("data_contract_version", sa.String(length=16), nullable=False),
        sa.Column("artifact_type", sa.String(length=128), nullable=False),
        sa.Column("schema_version", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=256), nullable=False),
        sa.Column("artifact_storage_ref", sa.String(length=256), nullable=False),
        sa.Column("data_snapshot_id", sa.String(length=128)),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.CheckConstraint(f"artifact_id REGEXP {_ARTIFACT_ID_PATTERN}", name="ck_agent_artifact_id"),
        sa.CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_agent_artifact_run_id"),
        sa.CheckConstraint(f"artifact_type IN ({_ARTIFACT_TYPES})", name="ck_agent_artifact_type"),
        sa.CheckConstraint(f"data_snapshot_id IS NULL OR data_snapshot_id REGEXP {_SNAPSHOT_ID_PATTERN}", name="ck_agent_artifact_snapshot_id"),
        sa.CheckConstraint("data_contract_version = 'v1'", name="ck_agent_artifact_contract_version"),
        sa.CheckConstraint("content_hash REGEXP '^sha256:[0-9a-f]{64}$'", name="ck_agent_artifact_content_hash"),
        sa.CheckConstraint("artifact_storage_ref REGEXP '^artifact://artifact-[0-9a-f]{32}$'", name="ck_agent_artifact_storage_ref"),
        sa.CheckConstraint("artifact_storage_ref = CONCAT('artifact://', artifact_id)", name="ck_agent_artifact_storage_ref_binding"),
        **_TABLE_OPTIONS,
    )
    op.create_index("ix_agent_artifact_run_id", "agent_artifact", ["run_id"])
    op.create_index("ix_agent_artifact_created_at", "agent_artifact", ["created_at"])

    op.create_table(
        "approval_ticket",
        sa.Column("approval_id", sa.String(length=128), primary_key=True),
        sa.Column("run_id", sa.String(length=128), sa.ForeignKey("agent_run.run_id"), nullable=False),
        sa.Column("data_contract_version", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by", sa.String(length=128), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=False)),
        sa.Column("decision_code", sa.String(length=128)),
        sa.CheckConstraint(f"status IN ({_APPROVAL_STATUS})", name="ck_approval_ticket_status"),
        sa.CheckConstraint(f"approval_id REGEXP {_APPROVAL_ID_PATTERN}", name="ck_approval_ticket_id"),
        sa.CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_approval_ticket_run_id"),
        sa.CheckConstraint(f"requested_by REGEXP {_PRINCIPAL_ID_PATTERN}", name="ck_approval_ticket_requested_by"),
        sa.CheckConstraint("data_contract_version = 'v1'", name="ck_approval_ticket_contract_version"),
        sa.CheckConstraint(f"decision_code IS NULL OR decision_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_approval_ticket_decision_code"),
        **_TABLE_OPTIONS,
    )
    op.create_index("ix_approval_ticket_run_id", "approval_ticket", ["run_id"])

    op.create_table(
        "tool_audit",
        sa.Column("audit_id", sa.String(length=128), primary_key=True),
        sa.Column("run_id", sa.String(length=128), sa.ForeignKey("agent_run.run_id"), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("data_contract_version", sa.String(length=16), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("tool_call_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=128)),
        sa.Column("snapshot_metadata", mysql.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.CheckConstraint(f"status IN ({_TOOL_STATUS})", name="ck_tool_audit_status"),
        sa.CheckConstraint(f"audit_id REGEXP {_AUDIT_ID_PATTERN}", name="ck_tool_audit_id"),
        sa.CheckConstraint(f"run_id REGEXP {_RUN_ID_PATTERN}", name="ck_tool_audit_run_id"),
        sa.CheckConstraint(f"trace_id REGEXP {_TRACE_ID_PATTERN}", name="ck_tool_audit_trace_id"),
        sa.CheckConstraint(f"tool_call_id REGEXP {_CALL_ID_PATTERN}", name="ck_tool_audit_call_id"),
        sa.CheckConstraint(f"tool_name IN ({_TOOL_NAMES})", name="ck_tool_audit_tool_name"),
        sa.CheckConstraint("duration_ms >= 0", name="ck_tool_audit_duration"),
        sa.CheckConstraint("data_contract_version = 'v1'", name="ck_tool_audit_contract_version"),
        sa.CheckConstraint(f"error_code IS NULL OR error_code REGEXP {_CONTROL_CODE_PATTERN}", name="ck_tool_audit_error_code"),
        **_TABLE_OPTIONS,
    )
    op.create_index("ix_tool_audit_trace_id", "tool_audit", ["trace_id"])
    op.create_index("ix_tool_audit_run_created_at", "tool_audit", ["run_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_tool_audit_run_created_at", table_name="tool_audit")
    op.drop_index("ix_tool_audit_trace_id", table_name="tool_audit")
    op.drop_table("tool_audit")
    op.drop_index("ix_approval_ticket_run_id", table_name="approval_ticket")
    op.drop_table("approval_ticket")
    op.drop_index("ix_agent_artifact_created_at", table_name="agent_artifact")
    op.drop_index("ix_agent_artifact_run_id", table_name="agent_artifact")
    op.drop_table("agent_artifact")
    op.drop_index("ix_agent_step_run_started_at", table_name="agent_step")
    op.drop_table("agent_step")
    op.drop_index("ix_agent_run_status_updated_at", table_name="agent_run")
    op.drop_index("ix_agent_run_trace_id", table_name="agent_run")
    op.drop_index("ix_agent_run_task_id", table_name="agent_run")
    op.drop_table("agent_run")
