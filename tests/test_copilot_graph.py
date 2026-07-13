"""Tests for the customer-insight LangGraph graph (Phase 1: fan-out/fan-in skeleton).

Uses the same ``populated_engine`` SQLite fixture as ``tests/test_db.py`` —
customer C1 (Champions, segment/CLV/propensity all populated) with a top
product ("20725") that has real association rules, and C3 (Lost, no
propensity) with no matching rules — so both the happy path and the
no-recommendations path are covered without a real LLM.
"""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from consumer_intel.copilot_graph.graph import build_customer_insight_graph
from consumer_intel.copilot_graph.state import initial_state, merge_tool_results


def test_merge_tool_results_shallow_merges_partial_updates():
    left = {"rfm": {"a": 1}}
    right = {"clv": {"b": 2}}
    assert merge_tool_results(left, right) == {"rfm": {"a": 1}, "clv": {"b": 2}}


def test_graph_fans_out_to_all_four_fetch_results(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1"))
    assert result["error"] is None
    assert sorted(result["tool_results"].keys()) == ["clv", "nbo", "propensity", "rfm"]
    # rfm/clv/propensity all come from the same get_customer row (see nodes.py docstring)
    assert result["tool_results"]["rfm"] == result["tool_results"]["clv"]
    assert result["tool_results"]["rfm"] == result["tool_results"]["propensity"]


def test_graph_fetch_nbo_uses_customer_top_product(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1"))
    offers = result["tool_results"]["nbo"]
    assert len(offers) == 2
    assert offers[0]["lift"] >= offers[1]["lift"]
    assert offers[0]["consequents_codes"] == "22384"


def test_graph_produces_grounded_insight_for_known_customer(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C1"))
    insight = result["insight"]
    assert insight is not None
    assert insight["customer_id"] == "C1"
    assert insight["segment"] == "Champions"
    assert insight["grounding"]["predicted_clv"] == 1200.0


def test_graph_customer_with_no_matching_rules_gets_empty_nbo(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("C3"))
    assert result["error"] is None
    assert result["tool_results"]["nbo"] == []
    assert result["insight"] is not None


def test_graph_unknown_customer_short_circuits_without_crashing(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("NOPE"))
    assert result["error"] == "Customer NOPE not found"
    assert result["insight"] is None
