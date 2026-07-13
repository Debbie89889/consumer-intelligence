"""Tests for the customer-insight LangGraph graph.

Uses the same ``populated_engine`` SQLite fixture as ``tests/test_db.py`` —
customer C1 (Champions, segment/CLV/propensity all populated) with a top
product ("20725") that has real association rules, and C3 (Lost, no
propensity) with no matching rules — so both the happy path and the
no-recommendations path are covered without a real LLM.

Phase 1 built the fan-out/fan-in fetch skeleton; Phase 2 adds the router's
conditional edges (not_found / clarify / proceed) and the fallback
observability node. Fan-out tests now pass an explicit ``intent`` — a bare
customer_id with no intent is deliberately ambiguous and routes to clarify.
"""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from consumer_intel.copilot_graph.graph import build_customer_insight_graph
from consumer_intel.copilot_graph.state import initial_state, merge_tool_results


def test_merge_tool_results_shallow_merges_partial_updates():
    left = {"rfm": {"a": 1}}
    right = {"clv": {"b": 2}}
    assert merge_tool_results(left, right) == {"rfm": {"a": 1}, "clv": {"b": 2}}


# --- happy path: fan-out/fan-in (Phase 1) ----------------------------------


def test_graph_fans_out_to_all_four_fetch_results(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1", intent="segment"))
    assert result["error"] is None
    assert sorted(result["tool_results"].keys()) == ["clv", "nbo", "propensity", "rfm"]
    # rfm/clv/propensity all come from the same get_customer row (see nodes.py docstring)
    assert result["tool_results"]["rfm"] == result["tool_results"]["clv"]
    assert result["tool_results"]["rfm"] == result["tool_results"]["propensity"]


def test_graph_fetch_nbo_uses_customer_top_product(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1", intent="nbo"))
    offers = result["tool_results"]["nbo"]
    assert len(offers) == 2
    assert offers[0]["lift"] >= offers[1]["lift"]
    assert offers[0]["consequents_codes"] == "22384"


def test_graph_produces_grounded_insight_for_known_customer(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1", intent="clv"))
    insight = result["insight"]
    assert insight is not None
    assert insight["customer_id"] == "C1"
    assert insight["segment"] == "Champions"
    assert insight["grounding"]["predicted_clv"] == 1200.0
    assert result["narration_backend"] == "template"


def test_graph_customer_with_no_matching_rules_gets_empty_nbo(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C3", intent="segment"))
    assert result["error"] is None
    assert result["tool_results"]["nbo"] == []
    assert result["insight"] is not None


# --- not_found branch --------------------------------------------------


def test_graph_unknown_customer_routes_to_not_found(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("NOPE", intent="segment"))
    assert result["error"] == "Customer NOPE not found"
    assert result["insight"] is None
    assert result["tool_results"] == {}  # fan-out never ran


def test_graph_not_found_takes_priority_over_unclear_intent(populated_engine):
    """A missing customer short-circuits to not_found even with an ambiguous intent."""
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("NOPE"))  # default intent="unclear"
    assert result["error"] == "Customer NOPE not found"
    assert result["clarification"] is None


# --- clarify branch ----------------------------------------------------


def test_graph_unclear_intent_routes_to_clarify(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1"))  # default intent="unclear"
    assert result["clarification"] is not None
    assert result["insight"] is None
    assert result["tool_results"] == {}  # fan-out never ran, no wasted queries
    assert result["error"] is None


# --- fallback branch -----------------------------------------------------


def test_graph_llm_failure_routes_through_fallback_node(populated_engine, monkeypatch):
    """When narrate_langchain raises, the graph should still produce a valid
    template-based insight *and* visibly pass through the fallback node —
    asserted here via the node execution trace, not just the final state.
    """
    import consumer_intel.copilot.narrator as narrator

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    def _boom(ctx):
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(narrator, "narrate_langchain", _boom)

    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    events = list(graph.stream(initial_state("C1", intent="clv"), stream_mode="updates"))
    node_names = [name for event in events for name in event]

    assert "fallback" in node_names
    final = graph.invoke(initial_state("C1", intent="clv"))
    assert final["narration_backend"] == "template_fallback"
    assert final["insight"] is not None


def test_graph_normal_narration_does_not_visit_fallback_node(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    events = list(graph.stream(initial_state("C1", intent="clv"), stream_mode="updates"))
    node_names = [name for event in events for name in event]
    assert "fallback" not in node_names
