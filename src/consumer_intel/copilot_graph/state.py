"""Shared LangGraph state for the Copilot workflow.

``tool_results`` needs an explicit reducer because the customer-insight graph
fans out to four parallel fetch nodes that all write to it within the same
superstep — without one, LangGraph raises ``InvalidUpdateError`` on
concurrent writes to the same key. Each ``fetch_*`` node returns a
single-key partial dict (e.g. ``{"rfm": ...}``); the reducer shallow-merges
them instead of one clobbering another.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["segment", "clv", "nbo", "propensity", "campaign", "unclear"]


def merge_tool_results(left: dict, right: dict) -> dict:
    """Reducer: shallow-merge partial ``tool_results`` updates from parallel nodes."""
    return {**left, **right}


class CopilotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str | None
    intent: Intent
    customer_exists: bool
    tool_results: Annotated[dict, merge_tool_results]
    clarification: str | None
    narration_backend: str | None
    insight: dict | None
    draft: dict | None
    approved: bool | None
    error: str | None


def initial_state(customer_id: str, intent: Intent = "unclear") -> CopilotState:
    """A fully-populated starting state for one customer-insight run.

    Explicit rather than relying on partial-dict defaults, so every run
    (tests, benchmark, future API wiring) starts from the same shape.
    ``intent`` defaults to "unclear" (routes to the clarify branch); callers
    that already know what the user wants pass it explicitly.
    """
    return CopilotState(
        messages=[],
        customer_id=customer_id,
        intent=intent,
        customer_exists=True,
        tool_results={},
        clarification=None,
        narration_backend=None,
        insight=None,
        draft=None,
        approved=None,
        error=None,
    )
