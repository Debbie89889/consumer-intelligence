"""Unit tests for the pure/narration functions in campaign_nodes.py.

The end-to-end interrupt/resume graph flow is covered separately in
tests/test_campaign_graph.py; this file isolates the business logic
(discount tiering, revision application, summary aggregation, narration)
so failures point at the right place.
"""

from __future__ import annotations

import pytest

from consumer_intel.copilot_graph import campaign_nodes as cn


@pytest.fixture
def six_candidates() -> list[dict]:
    # sorted by predicted_clv desc already: C1..C6 at 900,700,500,300,200,100
    return [
        {"customer_id": "C1", "segment": "At Risk", "predicted_clv": 900.0},
        {"customer_id": "C2", "segment": "At Risk", "predicted_clv": 700.0},
        {"customer_id": "C3", "segment": "Can't Lose Them", "predicted_clv": 500.0},
        {"customer_id": "C4", "segment": "At Risk", "predicted_clv": 300.0},
        {"customer_id": "C5", "segment": "Can't Lose Them", "predicted_clv": 200.0},
        {"customer_id": "C6", "segment": "At Risk", "predicted_clv": 100.0},
    ]


def test_assign_win_back_discounts_tiers_by_clv(six_candidates):
    out = cn.assign_win_back_discounts(six_candidates)
    by_id = {c["customer_id"]: c["discount"] for c in out}
    assert by_id["C1"] == 0.20
    assert by_id["C2"] == 0.20
    assert by_id["C3"] == 0.15
    assert by_id["C4"] == 0.15
    assert by_id["C5"] == 0.10
    assert by_id["C6"] == 0.10


def test_assign_win_back_discounts_ties_broken_by_customer_id():
    tied = [
        {"customer_id": "B", "segment": "At Risk", "predicted_clv": 100.0},
        {"customer_id": "A", "segment": "At Risk", "predicted_clv": 100.0},
    ]
    out = cn.assign_win_back_discounts(tied)
    assert [c["customer_id"] for c in out] == ["A", "B"]


def test_assign_win_back_discounts_empty_list():
    assert cn.assign_win_back_discounts([]) == []


def test_apply_campaign_revision_excludes_customers(six_candidates):
    tiered = cn.assign_win_back_discounts(six_candidates)
    out = cn.apply_campaign_revision(tiered, excluded_ids=["C1", "C3"], discount_overrides={})
    assert {c["customer_id"] for c in out} == {"C2", "C4", "C5", "C6"}


def test_apply_campaign_revision_overrides_discount(six_candidates):
    tiered = cn.assign_win_back_discounts(six_candidates)
    out = cn.apply_campaign_revision(tiered, excluded_ids=[], discount_overrides={"C6": 0.05})
    by_id = {c["customer_id"]: c["discount"] for c in out}
    assert by_id["C6"] == 0.05
    assert by_id["C1"] == 0.20  # untouched


def test_campaign_summary_aggregates(six_candidates):
    tiered = cn.assign_win_back_discounts(six_candidates)
    summary = cn.campaign_summary(tiered)
    assert summary["count"] == 6
    assert summary["avg_predicted_clv"] == pytest.approx((900 + 700 + 500 + 300 + 200 + 100) / 6)
    assert summary["by_segment"] == {"At Risk": 4, "Can't Lose Them": 2}


def test_campaign_summary_empty():
    summary = cn.campaign_summary([])
    assert summary["count"] == 0
    assert summary["avg_predicted_clv"] == 0.0


def test_draft_template_uses_only_given_facts():
    summary = {"count": 6, "avg_predicted_clv": 450.0, "avg_discount": 0.15, "by_segment": {}}
    brief = cn.draft_template(summary)
    assert "6" in brief["headline"]
    assert "450" in brief["message"]
    assert "15%" in brief["message"]
    assert len(brief["selling_points"]) >= 1


def test_draft_langchain_uses_structured_output(monkeypatch):
    """Exercises the real ChatPromptTemplate | model.with_structured_output(...)
    composition, with a fake model swapped in via get_chat_model — no network.
    """
    from langchain_core.runnables import RunnableLambda

    import consumer_intel.copilot_graph.campaign_nodes as cn_module

    class FakeModel:
        def with_structured_output(self, schema):
            return RunnableLambda(
                lambda _inputs: schema(headline="h", message="m", selling_points=["p1"])
            )

    monkeypatch.setattr(cn_module, "get_chat_model", lambda: FakeModel())
    brief = cn.draft_langchain({"count": 1}, review_note=None)
    assert brief["headline"] == "h"
    assert brief["selling_points"] == ["p1"]


def test_draft_langchain_includes_review_note_in_prompt(monkeypatch):
    """The review_note must actually reach the prompt, not just be accepted
    as a parameter — captured via a fake model that records its input."""
    from langchain_core.runnables import RunnableLambda

    import consumer_intel.copilot_graph.campaign_nodes as cn_module

    captured = {}

    class FakeModel:
        def with_structured_output(self, schema):
            def _invoke(inputs):
                captured["prompt_text"] = str(inputs)
                return schema(headline="h", message="m", selling_points=["p1"])

            return RunnableLambda(_invoke)

    monkeypatch.setattr(cn_module, "get_chat_model", lambda: FakeModel())
    cn.draft_langchain({"count": 1}, review_note="折扣太深,請調低")
    assert "折扣太深" in captured["prompt_text"]
