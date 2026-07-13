"""SQLAlchemy ORM models for Copilot business data (conversations, campaign review).

These tables are distinct from the analytics tables in ``repository.py``
(customers/rules/products/monthly/country), which are bulk-replaced from
parquet by ``loader.py`` on every pipeline run. The tables here are written to
incrementally at request time — conversation turns and campaign approvals —
so they are modelled and migrated with Alembic (``alembic/``) instead.

Responsibility split (also documented in CLAUDE.md): the LangGraph
checkpointer persists graph *execution* state (which node ran, how to resume
after an interrupt); these ORM tables persist *business* records (chat
history, audit trail of who approved what). The two are never the same store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import JSON, CheckConstraint, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

MessageRole = Literal["human", "ai", "tool"]
ApprovalStatus = Literal["pending", "approved", "rejected", "revised"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    """One Copilot chat thread."""

    __tablename__ = "conversations"

    thread_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    campaign_approvals: Mapped[list[CampaignApproval]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """One turn in a conversation (human / ai / tool)."""

    __tablename__ = "messages"
    __table_args__ = (CheckConstraint("role IN ('human','ai','tool')", name="ck_messages_role"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.thread_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class CampaignApproval(Base):
    """One win-back campaign draft awaiting (or having received) human review."""

    __tablename__ = "campaign_approvals"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','approved','rejected','revised')",
            name="ck_campaign_approvals_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.thread_id", ondelete="CASCADE"), index=True
    )
    draft: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    reviewer: Mapped[str | None] = mapped_column(String, nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="campaign_approvals")
