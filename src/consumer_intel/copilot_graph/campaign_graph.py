"""Builds the win-back campaign human-in-the-loop graph.

    campaign_intent -> build_candidates -> match_offers -> draft_campaign
      -> persist_pending -> await_approval [interrupt]
           |-- approved -> commit_campaign -> END
           |-- revised  -> draft_campaign (loop, edits applied)
           `-- rejected -> reject_campaign -> END

Requires a checkpointer (unlike the customer-insight graph): interrupt() /
Command(resume=...) only work across separate invocations if graph state is
actually persisted between them. Use ``db.checkpointer.make_checkpointer()``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from consumer_intel.copilot_graph.campaign_nodes import (
    await_approval,
    campaign_intent,
    draft_campaign,
    make_build_candidates,
    make_commit_campaign,
    make_match_offers,
    make_persist_pending,
    make_reject_campaign,
    route_after_approval,
)
from consumer_intel.copilot_graph.campaign_state import CampaignState


def build_campaign_graph(session_factory, checkpointer):
    """Compile the campaign graph bound to a session factory and checkpointer."""
    graph = StateGraph(CampaignState)
    graph.add_node("campaign_intent", campaign_intent)
    graph.add_node("build_candidates", make_build_candidates(session_factory))
    graph.add_node("match_offers", make_match_offers(session_factory))
    graph.add_node("draft_campaign", draft_campaign)
    graph.add_node("persist_pending", make_persist_pending(session_factory))
    graph.add_node("await_approval", await_approval)
    graph.add_node("commit_campaign", make_commit_campaign(session_factory))
    graph.add_node("reject_campaign", make_reject_campaign(session_factory))

    graph.add_edge(START, "campaign_intent")
    graph.add_edge("campaign_intent", "build_candidates")
    graph.add_edge("build_candidates", "match_offers")
    graph.add_edge("match_offers", "draft_campaign")
    graph.add_edge("draft_campaign", "persist_pending")
    graph.add_edge("persist_pending", "await_approval")
    graph.add_conditional_edges(
        "await_approval",
        route_after_approval,
        ["commit_campaign", "draft_campaign", "reject_campaign"],
    )
    graph.add_edge("commit_campaign", END)
    graph.add_edge("reject_campaign", END)

    return graph.compile(checkpointer=checkpointer)
