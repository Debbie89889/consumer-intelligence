"""Turn grounded facts into a business narrative.

Two backends: a deterministic **template** narrator that needs no network, and
a **LangChain** narrator that orchestrates the LLM call. The LLM is given only
the pre-computed facts and is asked to *phrase* them; it returns a structured
``NarratedInsight`` (validated by LangChain's ``with_structured_output``), so it
can only fill the free-text fields — never the grounded numbers, segment or
risk level, which Python owns. If no provider key is configured (or the call
fails) we fall back to the template, so the endpoint always returns a valid,
grounded insight.

Provider-agnostic: ``init_chat_model`` selects OpenAI or Anthropic from
``LLM_PROVIDER`` / ``LLM_MODEL`` (or inferred from whichever API key is set).
"""

from __future__ import annotations

import os

from consumer_intel.copilot.schema import CustomerInsight, InsightContext

Backend = str  # "auto" | "template" | "langchain"

_SYSTEM = (
    "You are a retail analytics assistant. You are given PRE-COMPUTED facts about "
    "one customer. Restate them as a short business insight. Do NOT invent, "
    "recompute or alter any numbers — only phrase the facts provided."
)


def _money(x: float) -> str:
    return f"£{x:,.0f}"


def narrate_template(ctx: InsightContext) -> dict:
    """Deterministic narration straight from the facts (no LLM)."""
    headline = (
        f"{ctx.segment} customer — {ctx.risk_level} churn risk, "
        f"predicted value {_money(ctx.predicted_clv)}."
    )
    observations = [
        f"Last purchase {ctx.recency_days} days ago across {ctx.frequency} orders "
        f"totalling {_money(ctx.monetary)}.",
        f"Probability still active: {ctx.prob_alive:.0%}.",
    ]
    if ctx.propensity is not None:
        observations.append(f"Modelled 90-day repurchase propensity: {ctx.propensity:.0%}.")

    actions = [ctx.recommended_action] if ctx.recommended_action else []
    if ctx.next_best_offers:
        actions.append("Cross-sell: " + ", ".join(ctx.next_best_offers[:3]) + ".")
    if not actions:
        actions = ["No specific action; monitor."]

    return {"headline": headline, "observations": observations, "recommended_actions": actions}


def _chat_model():
    """Build a provider-agnostic LangChain chat model from the environment."""
    provider = os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL")
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise RuntimeError("No LLM provider configured (set OPENAI/ANTHROPIC key).")
    if not model:
        model = "claude-3-5-haiku-latest" if provider == "anthropic" else "gpt-4o-mini"

    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider)


def narrate_langchain(ctx: InsightContext) -> dict:
    """LLM narration orchestrated by LangChain, returning validated free text.

    Builds ``prompt | model.with_structured_output(NarratedInsight)`` so the
    model's output is parsed and schema-validated by LangChain. Raises if no
    provider/key is available (caller falls back to the template).
    """
    from langchain_core.prompts import ChatPromptTemplate

    from consumer_intel.copilot.schema import NarratedInsight

    model = _chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", "FACTS (JSON):\n{facts}\n\nWrite the insight.")]
    )
    chain = prompt | model.with_structured_output(NarratedInsight)
    narrated: NarratedInsight = chain.invoke({"facts": ctx.model_dump_json()})
    return {
        "headline": narrated.headline,
        "observations": narrated.observations,
        "recommended_actions": narrated.recommended_actions,
    }


def _resolve_backend(backend: Backend) -> str:
    """Pick a concrete backend; 'auto' uses LangChain only if a key is present."""
    if backend != "auto":
        return backend
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "langchain"
    return "template"


def generate_insight(ctx: InsightContext, backend: Backend = "auto") -> CustomerInsight:
    """Produce a validated :class:`CustomerInsight` from grounded facts.

    The id, segment, risk level and grounding come from Python; only the free
    text is narrated. Any LLM/LangChain failure degrades to the template.
    """
    chosen = _resolve_backend(backend)
    try:
        parts = narrate_template(ctx) if chosen == "template" else narrate_langchain(ctx)
    except Exception:
        parts = narrate_template(ctx)

    return CustomerInsight(
        customer_id=ctx.customer_id,
        segment=ctx.segment,
        risk_level=ctx.risk_level,
        headline=parts["headline"],
        observations=parts["observations"],
        recommended_actions=parts["recommended_actions"],
        grounding=ctx.model_dump(),
    )
