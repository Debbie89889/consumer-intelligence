"""Node functions for the customer-insight LangGraph graph.

Every fetch_* node calls an *existing* ``db/repository.py`` query function —
none of them re-implement SQL. ``fetch_rfm``/``fetch_clv``/``fetch_propensity``
all call the same :func:`repository.get_customer`: the analytics DB already
denormalises RFM, CLV and propensity into one row per customer (a Phase 5
design choice), so there is nothing to split them into. They stay three
separate graph nodes anyway — matching the four-way fan-out this phase is
meant to demonstrate, and giving future conditional routing something to
selectively skip once it can tell which pieces an intent actually needs.
``fetch_nbo`` is the one node doing genuinely different work, via the
``next_best_offers_for_customer`` join query added alongside this graph.
"""

from __future__ import annotations

from langgraph.graph import END
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot.context import build_context
from consumer_intel.copilot.narrator import generate_insight_with_backend
from consumer_intel.copilot_graph.state import CopilotState
from consumer_intel.db import repository

CLARIFYING_QUESTION = (
    "您想了解的是哪一項?例如:客群輪廓(segment)、顧客終身價值(clv)、"
    "推薦商品(nbo)、回購傾向(propensity),或行銷活動(campaign)。"
)


def make_router(session_factory: sessionmaker[Session]):
    """Build the router node, bound to a session factory.

    Only decides ``customer_exists`` here (a cheap existence check) — the
    actual conditional routing logic lives in :func:`route_from_router`,
    which LangGraph runs against the state this node returns.
    """

    def router(state: CopilotState) -> dict:
        with session_factory() as session:
            exists = repository.customer_exists(session, str(state["customer_id"]))
        return {"customer_exists": exists}

    return router


def route_from_router(state: CopilotState) -> str | list[str]:
    """Decide where router sends execution: not_found / clarify / the fetch fan-out."""
    if not state["customer_exists"]:
        return "not_found"
    if state["intent"] == "unclear":
        return "clarify"
    return ["fetch_rfm", "fetch_clv", "fetch_nbo", "fetch_propensity"]


def not_found(state: CopilotState) -> dict:
    """Deterministic not-found response. Never reaches the LLM."""
    return {"error": f"Customer {state['customer_id']} not found"}


def clarify(state: CopilotState) -> dict:
    """Deterministic clarifying question when intent is ambiguous. No LLM call."""
    return {"clarification": CLARIFYING_QUESTION}


def make_fetch_rfm(session_factory: sessionmaker[Session]):
    """Build the fetch_rfm node, bound to a session factory."""

    def fetch_rfm(state: CopilotState) -> dict:
        with session_factory() as session:
            row = repository.get_customer(session, str(state["customer_id"]))
        return {"tool_results": {"rfm": row}}

    return fetch_rfm


def make_fetch_clv(session_factory: sessionmaker[Session]):
    """Build the fetch_clv node, bound to a session factory."""

    def fetch_clv(state: CopilotState) -> dict:
        with session_factory() as session:
            row = repository.get_customer(session, str(state["customer_id"]))
        return {"tool_results": {"clv": row}}

    return fetch_clv


def make_fetch_propensity(session_factory: sessionmaker[Session]):
    """Build the fetch_propensity node, bound to a session factory."""

    def fetch_propensity(state: CopilotState) -> dict:
        with session_factory() as session:
            row = repository.get_customer(session, str(state["customer_id"]))
        return {"tool_results": {"propensity": row}}

    return fetch_propensity


def make_fetch_nbo(session_factory: sessionmaker[Session]):
    """Build the fetch_nbo node, bound to a session factory."""

    def fetch_nbo(state: CopilotState) -> dict:
        with session_factory() as session:
            offers = repository.next_best_offers_for_customer(session, str(state["customer_id"]))
        return {"tool_results": {"nbo": offers}}

    return fetch_nbo


def join(state: CopilotState) -> dict:
    """Fan-in synchronisation point. router already ruled out a missing
    customer before the fan-out started, so there is nothing left to check
    here — this is a pass-through the graph needs so all four fetch edges
    converge before response_generator runs.
    """
    return {}


def response_generator(state: CopilotState) -> dict:
    """Narrate the grounded facts. Reuses the existing LCEL chain unchanged.

    Also records which backend actually served the narration
    (template / langchain / template_fallback) so :func:`route_from_response`
    can send a failed LLM call through the observable ``fallback`` node.
    """
    results = state["tool_results"]
    offer_names = [r["consequents"] for r in results.get("nbo", [])]
    ctx = build_context(results["rfm"], next_best_offers=offer_names)
    insight, used = generate_insight_with_backend(ctx)
    return {"insight": insight.model_dump(), "narration_backend": used}


def route_from_response_generator(state: CopilotState) -> str:
    """Send a failed-LLM-fell-back-to-template result through the fallback node."""
    if state["narration_backend"] == "template_fallback":
        return "fallback"
    return END


def fallback(state: CopilotState) -> dict:
    """Observability marker for the (already-completed) fallback narration.

    generate_insight_with_backend already produced a valid template-based
    insight before this node runs — there is nothing left to compute. This
    node exists purely so the fallback path shows up as its own step in the
    graph and in execution traces, instead of being silently swallowed
    inside response_generator like it always has been for the plain LCEL
    Copilot.
    """
    return {}
