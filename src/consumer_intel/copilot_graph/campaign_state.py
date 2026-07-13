"""State for the win-back campaign human-in-the-loop graph.

Distinct from ``CopilotState`` (the customer-insight graph): a campaign run
isn't tied to one customer_id, and it carries the interrupt/resume fields
the insight graph doesn't need.
"""

from __future__ import annotations

import uuid
from typing import Literal, TypedDict

CampaignStatus = Literal["new", "drafting", "pending_review", "approved", "revised", "rejected"]


class CampaignState(TypedDict):
    thread_id: str
    candidates: list[dict]
    brief: dict | None
    excluded_customer_ids: list[str]
    discount_overrides: dict[str, float]
    review_note: str | None
    reviewer: str | None
    decision: str | None
    status: CampaignStatus


def initial_campaign_state(thread_id: str | None = None) -> CampaignState:
    """A fully-populated starting state for one campaign-drafting run.

    ``thread_id`` is generated if not supplied — it's both the LangGraph
    checkpoint key and the ``campaign_approvals.thread_id`` this run's
    records are written under.
    """
    return CampaignState(
        thread_id=thread_id or str(uuid.uuid4()),
        candidates=[],
        brief=None,
        excluded_customer_ids=[],
        discount_overrides={},
        review_note=None,
        reviewer=None,
        decision=None,
        status="new",
    )
