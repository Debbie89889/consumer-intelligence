"""Node functions for the customer-insight LangGraph graph.

Every fetch_* node calls an *existing* ``db/repository.py`` query function —
none of them re-implement SQL. ``fetch_rfm``/``fetch_clv``/``fetch_propensity``
all call the same :func:`repository.get_customer`: the analytics DB already
denormalises RFM, CLV and propensity into one row per customer (a Phase 5
design choice), so there is nothing to split them into. They stay three
separate graph nodes anyway — matching the four-way fan-out this phase is
meant to demonstrate, and giving Phase 2's conditional routing something to
selectively skip once it can tell which pieces an intent actually needs.
``fetch_nbo`` is the one node doing genuinely different work, via the
``next_best_offers_for_customer`` join query added alongside this graph.
"""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot.context import build_context
from consumer_intel.copilot.narrator import generate_insight
from consumer_intel.copilot_graph.state import CopilotState
from consumer_intel.db import repository


def router(state: CopilotState) -> dict:
    """Entry node. Phase 2 adds conditional edges here (not-found / unclear /
    fallback); for now this is a pass-through into the parallel fetch fan-out.
    """
    return {}


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
    """Fan-in point: confirm the customer exists before the LLM narration step."""
    if state["tool_results"].get("rfm") is None:
        return {"error": f"Customer {state['customer_id']} not found"}
    return {}


def response_generator(state: CopilotState) -> dict:
    """Narrate the grounded facts. Reuses the existing LCEL chain unchanged."""
    if state.get("error"):
        return {"insight": None}
    results = state["tool_results"]
    offer_names = [r["consequents"] for r in results.get("nbo", [])]
    ctx = build_context(results["rfm"], next_best_offers=offer_names)
    insight = generate_insight(ctx)
    return {"insight": insight.model_dump()}
