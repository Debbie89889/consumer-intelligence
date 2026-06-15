"""Rule-based RFM segmentation.

Two steps:

1. Turn raw Recency / Frequency / Monetary into 1-5 quantile **scores**.
   Recency is reversed (more recent -> higher score). Frequency has heavy
   ties (many customers buy once), so we rank before binning to guarantee
   five populated bins instead of letting ``qcut`` collapse duplicate edges.
2. Map each customer to a named marketing segment using a (Recency,
   FrequencyMonetary) grid, and attach a recommended action per segment.

The grid is a *business* choice, not a statistical one; it is laid out
explicitly below so the boundaries are auditable and easy to tune. K-means
(see :mod:`consumer_intel.segmentation.kmeans`) offers the complementary
data-driven view of the same customers.
"""

from __future__ import annotations

import pandas as pd

N_BINS = 5

# Segment grid keyed by (R_score, FM_score), each 1-5.
# R: 5 = bought recently, 1 = long lapsed.
# FM: 5 = frequent & high spend, 1 = infrequent & low spend.
#
#        FM=5            FM=4            FM=3                FM=2              FM=1
# R=5  Champions       Champions       Potential Loyalist  Potential Loyalist New Customers
# R=4  Champions       Loyal Customers Loyal Customers     Potential Loyalist New Customers
# R=3  Loyal Customers Loyal Customers Need Attention      Promising          Promising
# R=2  At Risk         At Risk         Need Attention      About to Sleep     About to Sleep
# R=1  Can't Lose Them Can't Lose Them At Risk             Hibernating        Lost
SEGMENT_GRID: dict[tuple[int, int], str] = {
    (5, 5): "Champions", (5, 4): "Champions", (5, 3): "Potential Loyalist", (5, 2): "Potential Loyalist", (5, 1): "New Customers",  # noqa: E501
    (4, 5): "Champions", (4, 4): "Loyal Customers", (4, 3): "Loyal Customers", (4, 2): "Potential Loyalist", (4, 1): "New Customers",  # noqa: E501
    (3, 5): "Loyal Customers", (3, 4): "Loyal Customers", (3, 3): "Need Attention", (3, 2): "Promising", (3, 1): "Promising",  # noqa: E501
    (2, 5): "At Risk", (2, 4): "At Risk", (2, 3): "Need Attention", (2, 2): "About to Sleep", (2, 1): "About to Sleep",  # noqa: E501
    (1, 5): "Can't Lose Them", (1, 4): "Can't Lose Them", (1, 3): "At Risk", (1, 2): "Hibernating", (1, 1): "Lost",  # noqa: E501
}  # fmt: skip

# What to do with each segment — the "so what" the JD asks for.
SEGMENT_ACTIONS: dict[str, str] = {
    "Champions": "Reward loyalty; early access, referrals, VIP perks.",
    "Loyal Customers": "Upsell higher-value lines; ask for reviews.",
    "Potential Loyalist": "Membership / loyalty programme to deepen the habit.",
    "New Customers": "Strong onboarding; make the second purchase easy.",
    "Promising": "Nurture with targeted offers to build frequency.",
    "Need Attention": "Time-limited offers on recently browsed / bought lines.",
    "About to Sleep": "Reactivation nudges before they churn.",
    "At Risk": "Win-back: personalised offers, remind them of value.",
    "Can't Lose Them": "High-touch win-back; they were valuable, don't lose them.",
    "Hibernating": "Low-cost reactivation; otherwise deprioritise spend.",
    "Lost": "Minimal spend; only broad, cheap campaigns.",
}


def _score_ascending(series: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    """1..n score where higher value -> higher score (rank-based, tie-safe)."""
    ranks = series.rank(method="first")
    return pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1)).astype(int)


def _score_descending(series: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    """1..n score where lower value -> higher score (used for Recency)."""
    return _score_ascending(-series, n_bins)


def assign_segment(r_score: int, fm_score: int) -> str:
    """Look up the named segment for an (R, FM) score pair."""
    return SEGMENT_GRID[(int(r_score), int(fm_score))]


def score_rfm(rfm: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Add R/F/M scores, a combined RFM score, segment name and action.

    Parameters
    ----------
    rfm:
        Output of :func:`consumer_intel.features.rfm.compute_rfm`.

    Returns
    -------
    The input frame with added columns: ``R``, ``F``, ``M`` (1-5),
    ``FM`` (rounded mean of F and M), ``RFM_Score`` (e.g. ``"545"``),
    ``Segment`` and ``Action``.
    """
    out = rfm.copy()
    out["R"] = _score_descending(out["Recency"], n_bins)
    out["F"] = _score_ascending(out["Frequency"], n_bins)
    out["M"] = _score_ascending(out["Monetary"], n_bins)
    out["FM"] = ((out["F"] + out["M"]) / 2).round().clip(1, n_bins).astype(int)

    out["RFM_Score"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    out["Segment"] = [assign_segment(r, fm) for r, fm in zip(out["R"], out["FM"], strict=True)]
    out["Action"] = out["Segment"].map(SEGMENT_ACTIONS)
    return out


def summarize_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: size, share, mean RFM, revenue and revenue share."""
    summary = (
        scored.groupby("Segment")
        .agg(
            customers=("Recency", "size"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            total_revenue=("Monetary", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    summary["revenue_share"] = summary["total_revenue"] / summary["total_revenue"].sum()
    return summary
