"""Builds the customer-insight graph: fan-out to parallel fetch nodes, then fan-in.

router
  ├─(fan-out, parallel)──► fetch_rfm
                        ├──► fetch_clv
                        ├──► fetch_nbo
                        ├──► fetch_propensity
                              │ (fan-in)
                              ▼
                            join
                              ▼
                     response_generator
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot_graph.nodes import (
    join,
    make_fetch_clv,
    make_fetch_nbo,
    make_fetch_propensity,
    make_fetch_rfm,
    response_generator,
    router,
)
from consumer_intel.copilot_graph.state import CopilotState

FETCH_NODES = ("fetch_rfm", "fetch_clv", "fetch_nbo", "fetch_propensity")


def build_customer_insight_graph(session_factory: sessionmaker[Session]):
    """Compile the fan-out/fan-in customer-insight graph bound to a session factory.

    No checkpointer yet: this graph runs start-to-finish in one invocation.
    Phase 3 adds a checkpointer once the (separate) campaign graph needs to
    interrupt and resume for human review.
    """
    graph = StateGraph(CopilotState)
    graph.add_node("router", router)
    graph.add_node("fetch_rfm", make_fetch_rfm(session_factory))
    graph.add_node("fetch_clv", make_fetch_clv(session_factory))
    graph.add_node("fetch_nbo", make_fetch_nbo(session_factory))
    graph.add_node("fetch_propensity", make_fetch_propensity(session_factory))
    graph.add_node("join", join)
    graph.add_node("response_generator", response_generator)

    graph.add_edge(START, "router")
    for fetch_node in FETCH_NODES:
        graph.add_edge("router", fetch_node)
        graph.add_edge(fetch_node, "join")
    graph.add_edge("join", "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()
