"""Tests for the grounded Insight Copilot."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from consumer_intel.copilot import narrator
from consumer_intel.copilot.context import build_context, risk_from_prob_alive
from consumer_intel.copilot.narrator import generate_insight, narrate_template
from consumer_intel.copilot.schema import CustomerInsight, InsightContext, NarratedInsight


@pytest.fixture
def profile() -> dict:
    return {
        "customer_id": "12345",
        "segment": "Champions",
        "recency": 10,
        "frequency": 12,
        "monetary": 5400.0,
        "action": "Reward loyalty.",
        "predicted_clv": 1320.0,
        "prob_alive": 0.95,
        "propensity": 0.81,
    }


def test_risk_bands():
    assert risk_from_prob_alive(0.95) == "low"
    assert risk_from_prob_alive(0.55) == "medium"
    assert risk_from_prob_alive(0.2) == "high"


def test_build_context_is_grounded(profile):
    ctx = build_context(profile)
    assert isinstance(ctx, InsightContext)
    assert ctx.customer_id == "12345"
    assert ctx.risk_level == "low"  # derived from prob_alive in Python
    assert ctx.propensity == 0.81


def test_template_narration_only_restates_facts(profile):
    ctx = build_context(profile)
    parts = narrate_template(ctx)
    # numbers in the text must come from the context (no invented figures)
    assert "5,400" in parts["observations"][0]
    assert "95%" in parts["observations"][1]


def test_generate_insight_template_backend(profile):
    ctx = build_context(profile)
    insight = generate_insight(ctx, backend="template")
    assert isinstance(insight, CustomerInsight)
    assert insight.customer_id == "12345"
    assert insight.segment == "Champions"
    assert insight.risk_level == "low"
    assert len(insight.observations) >= 1
    assert len(insight.recommended_actions) >= 1
    # grounding trace is preserved
    assert insight.grounding["predicted_clv"] == 1320.0


def test_auto_backend_without_keys_uses_template(profile, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = build_context(profile)
    insight = generate_insight(ctx, backend="auto")  # should not raise / hit network
    assert isinstance(insight, CustomerInsight)


def test_schema_rejects_bad_risk_level():
    with pytest.raises(ValidationError):
        CustomerInsight(
            customer_id="1",
            segment="X",
            risk_level="catastrophic",  # not in Literal["low","medium","high"]
            headline="hi",
            observations=["a"],
            recommended_actions=["b"],
        )


def test_schema_rejects_empty_observations():
    with pytest.raises(ValidationError):
        CustomerInsight(
            customer_id="1",
            segment="X",
            risk_level="low",
            headline="hi",
            observations=[],  # min_length=1
            recommended_actions=["b"],
        )


# --- LangChain backend (offline, via monkeypatch) --------------------------
def test_narrated_insight_schema_validates():
    n = NarratedInsight(headline="hi", observations=["a", "b"], recommended_actions=["do x"])
    assert n.headline == "hi"
    with pytest.raises(ValidationError):
        NarratedInsight(headline="", observations=["a"], recommended_actions=["b"])


def test_langchain_backend_assembly(profile, monkeypatch):
    """generate_insight(backend='langchain') uses the narrated text but keeps
    the grounded fields (id/segment/risk/grounding) from Python."""
    monkeypatch.setattr(
        narrator,
        "narrate_langchain",
        lambda ctx: {
            "headline": "LLM headline",
            "observations": ["narrated obs"],
            "recommended_actions": ["narrated action"],
        },
    )
    ctx = build_context(profile)
    insight = generate_insight(ctx, backend="langchain")
    assert insight.headline == "LLM headline"
    assert insight.observations == ["narrated obs"]
    # grounded fields still come from Python, not the LLM
    assert insight.customer_id == "12345"
    assert insight.segment == "Champions"
    assert insight.risk_level == "low"
    assert insight.grounding["monetary"] == 5400.0


def test_langchain_failure_falls_back_to_template(profile, monkeypatch):
    def _boom(ctx):
        raise RuntimeError("no api key / network down")

    monkeypatch.setattr(narrator, "narrate_langchain", _boom)
    ctx = build_context(profile)
    insight = generate_insight(ctx, backend="langchain")  # must not raise
    assert isinstance(insight, CustomerInsight)
    # fell back to the deterministic template headline
    assert insight.headline.startswith("Champions customer")
