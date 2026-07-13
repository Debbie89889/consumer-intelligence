"""Builds the customer-insight (chat) graph: extract who, route, fan-out, answer.

extract_context
  |-- customer_id still unresolved -> clarify -> END
  `-- resolved                     -> router
                                         |-- not found -> not_found -> END
                                         `-- exists    -> fetch_rfm
                                                        -> fetch_clv
                                                        -> fetch_nbo
                                                        -> fetch_propensity
                                                                | (fan-in)
                                                                v
                                                              join
                                                                v
                                                       response_generator
                                                                |-- fell back -> fallback -> END
                                                                `-- normal    -----------> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from consumer_intel.copilot_graph.nodes import (
    clarify,
    extract_context,
    fallback,
    join,
    make_fetch_clv,
    make_fetch_nbo,
    make_fetch_propensity,
    make_fetch_rfm,
    make_router,
    not_found,
    response_generator,
    route_from_extract_context,
    route_from_response_generator,
    route_from_router,
)
from consumer_intel.copilot_graph.state import CopilotState

FETCH_NODES = ("fetch_rfm", "fetch_clv", "fetch_nbo", "fetch_propensity")


def build_customer_insight_graph(session_factory: sessionmaker[Session]):
    """Compile the customer-insight graph bound to a session factory.

    No checkpointer: conversation history lives in the SQLAlchemy
    ``messages`` table, reconstructed fresh on every turn by
    ``chat.run_turn`` — this graph runs start-to-finish in a single
    invocation per turn. Phase 3's separate campaign graph is the one that
    needs a checkpointer, for its interrupt()/resume() HITL flow.
    """
    graph = StateGraph(CopilotState)
    graph.add_node("extract_context", extract_context)
    graph.add_node("router", make_router(session_factory))
    graph.add_node("not_found", not_found)
    graph.add_node("clarify", clarify)
    graph.add_node("fetch_rfm", make_fetch_rfm(session_factory))
    graph.add_node("fetch_clv", make_fetch_clv(session_factory))
    graph.add_node("fetch_nbo", make_fetch_nbo(session_factory))
    graph.add_node("fetch_propensity", make_fetch_propensity(session_factory))
    graph.add_node("join", join)
    graph.add_node("response_generator", response_generator)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "extract_context")
    graph.add_conditional_edges(
        "extract_context", route_from_extract_context, ["clarify", "router"]
    )
    graph.add_edge("clarify", END)
    graph.add_conditional_edges("router", route_from_router, ["not_found", *FETCH_NODES])
    graph.add_edge("not_found", END)
    for fetch_node in FETCH_NODES:
        graph.add_edge(fetch_node, "join")
    graph.add_edge("join", "response_generator")
    graph.add_conditional_edges(
        "response_generator", route_from_response_generator, ["fallback", END]
    )
    graph.add_edge("fallback", END)

    return graph.compile()
