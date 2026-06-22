"""Build the grounded :class:`InsightContext` from database facts.

Everything numeric is decided here, in Python — including the risk level, which
is a deterministic function of the churn (alive) probability. The narrator
never computes anything; it only phrases what this module produces.
"""

from __future__ import annotations

from consumer_intel.copilot.schema import InsightContext, RiskLevel

HIGH_RISK_BELOW = 0.4
MEDIUM_RISK_BELOW = 0.7


def risk_from_prob_alive(prob_alive: float) -> RiskLevel:
    """Map the alive probability to a churn-risk band (Python decides, not the LLM)."""
    if prob_alive < HIGH_RISK_BELOW:
        return "high"
    if prob_alive < MEDIUM_RISK_BELOW:
        return "medium"
    return "low"


def build_context(profile: dict, next_best_offers: list[str] | None = None) -> InsightContext:
    """Assemble the facts for one customer into an :class:`InsightContext`.

    Parameters
    ----------
    profile:
        A customer row from :func:`consumer_intel.db.repository.get_customer`.
    next_best_offers:
        Human-readable product names to cross-sell (already computed elsewhere).
    """
    prob_alive = float(profile.get("prob_alive") or 0.0)
    return InsightContext(
        customer_id=str(profile["customer_id"]),
        segment=str(profile.get("segment") or "Unknown"),
        recency_days=int(profile.get("recency") or 0),
        frequency=int(profile.get("frequency") or 0),
        monetary=float(profile.get("monetary") or 0.0),
        predicted_clv=float(profile.get("predicted_clv") or 0.0),
        prob_alive=prob_alive,
        propensity=(
            float(profile["propensity"]) if profile.get("propensity") is not None else None
        ),
        risk_level=risk_from_prob_alive(prob_alive),
        recommended_action=str(profile.get("action") or ""),
        next_best_offers=next_best_offers or [],
    )
