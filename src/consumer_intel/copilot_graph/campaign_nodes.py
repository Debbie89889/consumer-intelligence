"""Node functions for the win-back campaign HITL graph.

    campaign_intent -> build_candidates -> match_offers -> draft_campaign
      -> persist_pending -> await_approval [interrupt]
           |-- approved -> commit_campaign
           |-- revised  -> draft_campaign (loops back, edits applied)
           `-- rejected -> reject_campaign

``persist_pending`` is deliberately a separate node from ``await_approval``.
LangGraph re-executes any code that runs *before* an ``interrupt()`` call
within the same node on every resume — verified empirically before writing
this — so putting the ``campaign_approvals`` write there would duplicate or
stomp it on every resume. Keeping the write in its own (non-interrupting)
node means it only runs once per draft, exactly when a new draft is produced.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from langgraph.types import interrupt
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot.narrator import get_chat_model
from consumer_intel.copilot_graph.campaign_schema import CampaignBrief
from consumer_intel.copilot_graph.campaign_state import CampaignState
from consumer_intel.db import repository
from consumer_intel.db.models import CampaignApproval, Conversation

DISCOUNT_TIERS = (0.20, 0.15, 0.10)  # top / mid / bottom third of predicted_clv

_SYSTEM = (
    "你是一位零售行銷文案撰寫人。你會收到一份 win-back 行銷活動的『已計算好的彙總事實』"
    "(候選客戶數、平均終身價值、平均折扣、客群分布)。請用繁體中文寫出簡潔的活動標題"
    "(headline)、訴求文案(message)與 2 至 4 個賣點(selling_points)。"
    "不要捏造或重新計算任何數字,只能根據提供的事實撰寫語氣與角度。"
)


def assign_win_back_discounts(candidates: list[dict]) -> list[dict]:
    """Tier the win-back discount by predicted_clv percentile within this batch.

    Top third of predicted_clv gets the deepest discount (20%) — retaining a
    higher-value sleeping customer justifies more margin given up; bottom
    third gets 10%. Ties broken by customer_id for determinism.
    """
    ranked = sorted(candidates, key=lambda c: (-c["predicted_clv"], c["customer_id"]))
    n = len(ranked)
    return [{**c, "discount": DISCOUNT_TIERS[i * 3 // n]} for i, c in enumerate(ranked)]


def apply_campaign_revision(
    candidates: list[dict], excluded_ids: list[str], discount_overrides: dict[str, float]
) -> list[dict]:
    """Drop excluded customers and apply reviewer-edited discounts."""
    kept = [c for c in candidates if c["customer_id"] not in excluded_ids]
    return [
        {**c, "discount": discount_overrides.get(c["customer_id"], c["discount"])} for c in kept
    ]


def campaign_summary(candidates: list[dict]) -> dict:
    """Aggregate, Python-computed facts the LLM narrates from — no per-customer PII."""
    n = len(candidates)
    by_segment: dict[str, int] = {}
    for c in candidates:
        by_segment[c["segment"]] = by_segment.get(c["segment"], 0) + 1
    return {
        "count": n,
        "avg_predicted_clv": round(sum(c["predicted_clv"] for c in candidates) / n, 2)
        if n
        else 0.0,
        "avg_discount": round(sum(c["discount"] for c in candidates) / n, 4) if n else 0.0,
        "by_segment": by_segment,
    }


def draft_template(summary: dict) -> dict:
    """Deterministic campaign copy (no LLM),繁體中文."""
    return {
        "headline": f"喚醒 {summary['count']} 位沉睡的高價值客戶",
        "message": (
            f"這群客戶平均終身價值 £{summary['avg_predicted_clv']:,.0f},"
            f"平均折扣 {summary['avg_discount']:.0%},值得專屬的挽回活動。"
        ),
        "selling_points": [
            "依個人終身價值分層的折扣",
            "搭配客戶最常購買商品的推薦",
            "限時喚醒沉睡客戶",
        ],
    }


def draft_langchain(summary: dict, review_note: str | None) -> dict:
    """LLM-narrated campaign copy via LCEL + structured output."""
    from langchain_core.prompts import ChatPromptTemplate

    human = "活動事實(JSON):\n{facts}\n\n請撰寫活動文案。"
    if review_note:
        human += f"\n\n審核意見(請納入考量):{review_note}"

    model = get_chat_model()
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", human)])
    chain = prompt | model.with_structured_output(CampaignBrief)
    brief: CampaignBrief = chain.invoke({"facts": json.dumps(summary, ensure_ascii=False)})
    return brief.model_dump()


def campaign_intent(state: CampaignState) -> dict:
    """Entry node. No I/O — just marks the run as started."""
    return {"status": "drafting"}


def make_build_candidates(session_factory: sessionmaker[Session]):
    """Build the build_candidates node: win-back segment query + discount tiering."""

    def build_candidates(state: CampaignState) -> dict:
        with session_factory() as session:
            rows = repository.win_back_candidates(session)
        return {"candidates": assign_win_back_discounts(rows)}

    return build_candidates


def make_match_offers(session_factory: sessionmaker[Session]):
    """Build the match_offers node: per-customer Next Best Offer lookup."""

    def match_offers(state: CampaignState) -> dict:
        with session_factory() as session:
            updated = []
            for c in state["candidates"]:
                offers = repository.next_best_offers_for_customer(
                    session, c["customer_id"], limit=1
                )
                updated.append({**c, "offer": offers[0]["consequents"] if offers else None})
        return {"candidates": updated}

    return match_offers


def draft_campaign(state: CampaignState) -> dict:
    """Narrate the campaign copy. Falls back to a deterministic template
    exactly like the customer-insight narrator does — numbers are always
    Python's, the LLM only phrases them.

    Also applies any pending revision (excluded customers / discount
    overrides) from a previous "revised" resume before re-drafting, and
    clears those one-shot instructions once applied.
    """
    candidates = apply_campaign_revision(
        state["candidates"], state["excluded_customer_ids"], state["discount_overrides"]
    )
    summary = campaign_summary(candidates)

    has_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    try:
        brief = (
            draft_langchain(summary, state["review_note"]) if has_key else draft_template(summary)
        )
    except Exception:
        brief = draft_template(summary)

    return {
        "candidates": candidates,
        "brief": brief,
        "status": "drafting",
        "excluded_customer_ids": [],
        "discount_overrides": {},
    }


def make_persist_pending(session_factory: sessionmaker[Session]):
    """Build the persist_pending node: write/refresh the pending campaign_approvals row."""

    def persist_pending(state: CampaignState) -> dict:
        with session_factory() as session:
            if session.get(Conversation, state["thread_id"]) is None:
                session.add(Conversation(thread_id=state["thread_id"]))
                session.flush()
            approval = (
                session.query(CampaignApproval)
                .filter_by(thread_id=state["thread_id"], status="pending")
                .one_or_none()
            )
            draft = {"brief": state["brief"], "candidates": state["candidates"]}
            if approval is None:
                session.add(
                    CampaignApproval(thread_id=state["thread_id"], draft=draft, status="pending")
                )
            else:
                approval.draft = draft
            session.commit()
        return {"status": "pending_review"}

    return persist_pending


def await_approval(state: CampaignState) -> dict:
    """Pause for human review. No DB write here (see module docstring) —
    this node's only job is the interrupt/resume handshake."""
    decision = interrupt(
        {
            "thread_id": state["thread_id"],
            "brief": state["brief"],
            "candidate_count": len(state["candidates"]),
        }
    )
    return {
        "decision": decision.get("action"),
        "review_note": decision.get("review_note"),
        "reviewer": decision.get("reviewer"),
        "excluded_customer_ids": decision.get("excluded_customer_ids", []),
        "discount_overrides": decision.get("discount_overrides", {}),
    }


def route_after_approval(state: CampaignState) -> str:
    if state["decision"] == "approved":
        return "commit_campaign"
    if state["decision"] == "revised":
        return "draft_campaign"
    return "reject_campaign"


def _finalize(session_factory: sessionmaker[Session], state: CampaignState, status: str) -> None:
    with session_factory() as session:
        approval = (
            session.query(CampaignApproval)
            .filter_by(thread_id=state["thread_id"], status="pending")
            .one()
        )
        approval.status = status
        approval.reviewer = state["reviewer"]
        approval.review_note = state["review_note"]
        approval.decided_at = datetime.now(UTC)
        session.commit()


def make_commit_campaign(session_factory: sessionmaker[Session]):
    """Build the commit_campaign node: mark the pending row approved."""

    def commit_campaign(state: CampaignState) -> dict:
        _finalize(session_factory, state, "approved")
        return {"status": "approved"}

    return commit_campaign


def make_reject_campaign(session_factory: sessionmaker[Session]):
    """Build the reject_campaign node: mark the pending row rejected."""

    def reject_campaign(state: CampaignState) -> dict:
        _finalize(session_factory, state, "rejected")
        return {"status": "rejected"}

    return reject_campaign
