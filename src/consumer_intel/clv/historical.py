"""Simple historical (observed) CLV.

This is the naive baseline: what each customer has *actually* spent so far,
plus the components that make it up (orders, average order value, tenure).
It deliberately makes no forward-looking claim — that is what the BG/NBD +
Gamma-Gamma model in :mod:`consumer_intel.clv.predictive` adds. Contrasting
the two is the point: averages describe the past, the probabilistic model
estimates the future.
"""

from __future__ import annotations

import pandas as pd


def historical_clv(transactions: pd.DataFrame) -> pd.DataFrame:
    """Observed value per customer with its driving components.

    Parameters
    ----------
    transactions:
        Cleaned transactions with ``CustomerID``, ``InvoiceNo``,
        ``InvoiceDate`` and ``TotalPrice``.

    Returns
    -------
    DataFrame indexed by ``CustomerID`` with:
    ``total_revenue`` (observed CLV), ``n_orders``, ``avg_order_value``,
    ``first_purchase``, ``last_purchase``, ``tenure_days`` and
    ``annual_value`` (revenue annualised over the customer's tenure).
    """
    g = transactions.groupby("CustomerID")
    out = g.agg(
        total_revenue=("TotalPrice", "sum"),
        n_orders=("InvoiceNo", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )
    out["avg_order_value"] = out["total_revenue"] / out["n_orders"]
    out["tenure_days"] = (out["last_purchase"] - out["first_purchase"]).dt.days
    # Annualised spend rate; tenure 0 (single-day customers) -> treat as 1 day.
    out["annual_value"] = out["total_revenue"] / out["tenure_days"].clip(lower=1) * 365
    return out
