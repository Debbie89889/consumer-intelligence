"""Tests for the customer-insight (chat) LangGraph graph.

Uses the same ``populated_engine`` SQLite fixture as ``tests/test_db.py`` —
customer C1 (Champions, segment/CLV/propensity all populated) with a top
product ("20725") that has real association rules, and C3 (Lost, no
propensity) with no matching rules — so both the happy path and the
no-recommendations path are covered without a real LLM.

Phase 1 built the fan-out/fan-in fetch skeleton; Phase 2 added not_found /
clarify / fallback conditional routing; Phase 4 adds conversational
answering (a plain grounded text reply instead of a fixed CustomerInsight
structure) and an extract_context node that resolves customer_id from
conversation history — see test_chat.py for the multi-turn / pronoun tests.
"""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from consumer_intel.copilot_graph.graph import build_customer_insight_graph
from consumer_intel.copilot_graph.state import initial_state, merge_tool_results


def test_merge_tool_results_shallow_merges_partial_updates():
    left = {"rfm": {"a": 1}}
    right = {"clv": {"b": 2}}
    assert merge_tool_results(left, right) == {"rfm": {"a": 1}, "clv": {"b": 2}}


# --- happy path: fan-out/fan-in ---------------------------------------------


def test_graph_fans_out_to_all_four_fetch_results(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("t1", customer_id="C1"))
    assert result["error"] is None
    assert sorted(result["tool_results"].keys()) == ["clv", "nbo", "propensity", "rfm"]
    # rfm/clv/propensity all come from the same get_customer row (see nodes.py docstring)
    assert result["tool_results"]["rfm"] == result["tool_results"]["clv"]
    assert result["tool_results"]["rfm"] == result["tool_results"]["propensity"]


def test_graph_fetch_nbo_uses_customer_top_product(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("t1", customer_id="C1"))
    offers = result["tool_results"]["nbo"]
    assert len(offers) == 2
    assert offers[0]["lift"] >= offers[1]["lift"]
    assert offers[0]["consequents_codes"] == "22384"


def test_graph_produces_grounded_answer_for_known_customer(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("t1", customer_id="C1"))
    assert result["answer"] is not None
    assert "Champions" not in result["answer"]  # segment is shown in zh, not the raw English
    assert "核心客戶" in result["answer"]  # segment_zh("Champions")
    assert result["narration_backend"] == "template"


def test_graph_customer_with_no_matching_rules_still_answers(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("t1", customer_id="C3"))
    assert result["error"] is None
    assert result["tool_results"]["nbo"] == []
    assert result["answer"] is not None


# --- not_found branch --------------------------------------------------


def test_graph_unknown_customer_routes_to_not_found(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("t1", customer_id="NOPE"))
    assert result["error"] == "查無客戶 NOPE。"
    assert result["answer"] is None
    assert result["tool_results"] == {}  # fan-out never ran
    assert result["messages"][-1].content == "查無客戶 NOPE。"


# --- clarify branch (customer_id can't be resolved) ---------------------


def test_graph_no_customer_id_and_no_llm_key_routes_to_clarify(populated_engine, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    result = graph.invoke(initial_state("t1", customer_id=None))
    assert result["clarification"] is not None
    assert result["answer"] is None
    assert result["tool_results"] == {}  # fan-out never ran, no wasted queries
    assert result["error"] is None
    assert result["messages"][-1].content == result["clarification"]


# --- fallback branch -----------------------------------------------------


def test_graph_llm_failure_routes_through_fallback_node(populated_engine, monkeypatch):
    """When answer_langchain raises, the graph should still produce a valid
    template-based answer *and* visibly pass through the fallback node —
    asserted here via the node execution trace, not just the final state.
    """
    import consumer_intel.copilot_graph.nodes as nodes

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    def _boom(facts, messages):
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(nodes, "answer_langchain", _boom)

    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    events = list(graph.stream(initial_state("t1", customer_id="C1"), stream_mode="updates"))
    node_names = [name for event in events for name in event]

    assert "fallback" in node_names
    final = graph.invoke(initial_state("t1", customer_id="C1"))
    assert final["narration_backend"] == "template_fallback"
    assert final["answer"] is not None


def test_graph_normal_narration_does_not_visit_fallback_node(populated_engine):
    graph = build_customer_insight_graph(sessionmaker(bind=populated_engine))
    events = list(graph.stream(initial_state("t1", customer_id="C1"), stream_mode="updates"))
    node_names = [name for event in events for name in event]
    assert "fallback" not in node_names
