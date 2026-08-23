"""Align the deployed audit catalog with the dynamic read boundary.

The initial runtime schema may already exist. This migration changes only the
closed tool-name CHECK on ``tool_audit``; it does not add business tables or
touch the business database.
"""

from alembic import op


revision = "0003_dynamic_read_tool_catalog"
down_revision = "0002_approval_ticket_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_tool_audit_tool_name", "tool_audit", type_="check")
    op.create_check_constraint(
        "ck_tool_audit_tool_name",
        "tool_audit",
        "tool_name IN ('execute_read_query','literature_evidence')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_tool_audit_tool_name", "tool_audit", type_="check")
    op.create_check_constraint(
        "ck_tool_audit_tool_name",
        "tool_audit",
        "tool_name IN ('execute_read_query','literature_evidence')",
    )
