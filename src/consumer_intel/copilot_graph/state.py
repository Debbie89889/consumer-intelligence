"""Shared LangGraph state for the customer-insight (chat) graph.

``tool_results`` needs an explicit reducer because the graph fans out to
four parallel fetch nodes that all write to it within the same superstep —
without one, LangGraph raises ``InvalidUpdateError`` on concurrent writes to
the same key. Each ``fetch_*`` node returns a single-key partial dict (e.g.
``{"rfm": ...}``); the reducer shallow-merges them instead of one clobbering
another.

No ``intent`` field (Phase 1/2 had one, unused beyond a single unclear-vs-not
check) and no ``draft``/``approved`` fields (speculative placeholders for the
campaign flow, which ended up with its own separate ``CampaignState`` in
``campaign_state.py``) — removed once Phase 4 made clear they were dead
weight rather than "used later."
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_tool_results(left: dict, right: dict) -> dict:
    """Reducer: shallow-merge partial ``tool_results`` updates from parallel nodes."""
    return {**left, **right}


class CopilotState(TypedDict):
    thread_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str | None
    customer_exists: bool
    tool_results: Annotated[dict, merge_tool_results]
    clarification: str | None
    narration_backend: str | None
    answer: str | None
    error: str | None


def initial_state(
    thread_id: str,
    customer_id: str | None = None,
    messages: list[AnyMessage] | None = None,
) -> CopilotState:
    """A fully-populated starting state for one conversation turn.

    ``customer_id=None`` means "let extract_context resolve it from
    ``messages``" (the chat path); passing it directly skips that LLM call
    (direct/non-chat invocations — tests, the benchmark script).
    """
    return CopilotState(
        thread_id=thread_id,
        messages=messages or [],
        customer_id=customer_id,
        customer_exists=True,
        tool_results={},
        clarification=None,
        narration_backend=None,
        answer=None,
        error=None,
    )
