"""Add closed operation and reviewer binding fields to approval tickets.

The dedicated Runtime Schema may already have revision 0001 deployed.  This
revision changes only ``mico_agent_runtime.approval_ticket`` and never refers
to the business database.
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_approval_ticket_lifecycle"
down_revision = "0001_agent_runtime_state"
branch_labels = None
depends_on = None

_OPERATIONS = (
    "'read_only_research','statistical_analysis','sensitive_batch_read',"
    "'bulk_export','external_publish','business_write'"
)
_PRINCIPAL_ID_PATTERN = "'^principal-[0-9a-f]{32}$'"


def upgrade() -> None:
    # The 0001 Runtime deployment is an empty operational store in the
    # current rollout.  No default operation is invented for old tickets.
    op.add_column(
        "approval_ticket",
        sa.Column("operation", sa.String(length=64), nullable=False),
    )
    op.add_column(
        "approval_ticket",
        sa.Column("decided_by", sa.String(length=128), nullable=True),
    )
    op.create_check_constraint(
        "ck_approval_ticket_operation",
        "approval_ticket",
        f"operation IN ({_OPERATIONS})",
    )
    op.create_check_constraint(
        "ck_approval_ticket_decided_by",
        "approval_ticket",
        f"decided_by IS NULL OR decided_by REGEXP {_PRINCIPAL_ID_PATTERN}",
    )


def downgrade() -> None:
    op.drop_constraint("ck_approval_ticket_decided_by", "approval_ticket", type_="check")
    op.drop_constraint("ck_approval_ticket_operation", "approval_ticket", type_="check")
    op.drop_column("approval_ticket", "decided_by")
    op.drop_column("approval_ticket", "operation")
