"""Pydantic models for the Insight Copilot.

Two models enforce the grounding contract:

* ``InsightContext`` — the facts, all computed in Python/SQL. Risk level is a
  number-derived classification, *not* something the LLM decides.
* ``CustomerInsight`` — the validated output. The LLM may only fill the free-
  text fields (headline / observations / actions); the customer id, segment,
  risk level and the grounding numbers come from Python, so the narrative can
  always be traced back to the facts it was built from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]


class InsightContext(BaseModel):
    """Pre-computed facts handed to the narrator. No LLM math allowed."""

    customer_id: str
    segment: str
    recency_days: int
    frequency: int
    monetary: float
    predicted_clv: float
    prob_alive: float
    propensity: float | None
    risk_level: RiskLevel
    recommended_action: str
    next_best_offers: list[str] = Field(default_factory=list)


class CustomerInsight(BaseModel):
    """Validated, narrated insight returned by the Copilot."""

    customer_id: str
    segment: str
    risk_level: RiskLevel
    headline: str = Field(min_length=1, max_length=240)
    observations: list[str] = Field(min_length=1, max_length=6)
    recommended_actions: list[str] = Field(min_length=1, max_length=6)
    # the exact facts this narrative was built from (traceability)
    grounding: dict = Field(default_factory=dict)


class NarratedInsight(BaseModel):
    """Only the free-text fields the LLM is allowed to fill.

    Used as the ``with_structured_output`` schema so the model returns exactly
    these — and nothing about the grounded numbers, segment or risk level,
    which Python owns. Keeps the LLM strictly in the role of phrasing facts.
    """

    headline: str = Field(min_length=1, max_length=240, description="One-line summary")
    observations: list[str] = Field(
        min_length=1, max_length=6, description="Short factual observations, restating given facts"
    )
    recommended_actions: list[str] = Field(
        min_length=1, max_length=6, description="Concrete next actions for this customer"
    )
