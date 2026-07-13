"""add conversations, messages, campaign_approvals

Revision ID: 4ae1cf2d0afd
Revises:
Create Date: 2026-07-13 14:29:30.919712

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4ae1cf2d0afd"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the Copilot business tables (conversations, messages, campaign_approvals).

    This migration does not touch customers/rules/products/monthly/country —
    those are bulk-replaced from parquet by ``db/loader.py`` and are outside
    Alembic's scope.
    """
    op.create_table(
        "conversations",
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_table(
        "campaign_approvals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("draft", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reviewer", sa.String(), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending','approved','rejected','revised')",
            name="ck_campaign_approvals_status",
        ),
        sa.ForeignKeyConstraint(["thread_id"], ["conversations.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_campaign_approvals_thread_id"), "campaign_approvals", ["thread_id"], unique=False
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("role IN ('human','ai','tool')", name="ck_messages_role"),
        sa.ForeignKeyConstraint(["thread_id"], ["conversations.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_thread_id"), "messages", ["thread_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_messages_thread_id"), table_name="messages")
    op.drop_table("messages")
    op.drop_index(op.f("ix_campaign_approvals_thread_id"), table_name="campaign_approvals")
    op.drop_table("campaign_approvals")
    op.drop_table("conversations")
