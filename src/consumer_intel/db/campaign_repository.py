"""Read queries against the Copilot business tables (campaign_approvals).

Unlike ``repository.py`` (parameterized ``text()`` SQL against the pandas-
loaded analytics tables), these use the SQLAlchemy ORM directly against the
Alembic-managed tables in ``db/models.py`` — a different data domain, kept
in its own module rather than blended into ``repository.py``.

State *transitions* (generating a new draft, resuming after review) go
through the campaign graph, not this module — these are read-only lookups
for listing/displaying what's already been persisted.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from consumer_intel.db.models import CampaignApproval


def _to_dict(row: CampaignApproval) -> dict:
    return {
        "thread_id": row.thread_id,
        "status": row.status,
        "draft": row.draft,
        "reviewer": row.reviewer,
        "review_note": row.review_note,
        "decided_at": row.decided_at,
        "created_at": row.created_at,
    }


def list_campaigns(session: Session, status: str | None = None, limit: int = 50) -> list[dict]:
    """Campaign drafts, most recent first — optionally filtered by status."""
    query = session.query(CampaignApproval).order_by(CampaignApproval.created_at.desc())
    if status:
        query = query.filter_by(status=status)
    return [_to_dict(r) for r in query.limit(limit).all()]


def get_campaign(session: Session, thread_id: str) -> dict | None:
    """The latest campaign_approvals row for a thread_id, or None if unknown."""
    row = (
        session.query(CampaignApproval)
        .filter_by(thread_id=thread_id)
        .order_by(CampaignApproval.created_at.desc())
        .first()
    )
    return _to_dict(row) if row else None
