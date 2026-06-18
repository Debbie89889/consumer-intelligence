"""Snapshot feature table for purchase-propensity modelling.

The task: as of a **cutoff** date, predict whether a customer will purchase
again within the next ``horizon_days``.

Avoiding leakage is the whole game:
* **Features** are computed only from transactions on or before the cutoff.
* **Label** is 1 iff the customer purchases in ``(cutoff, cutoff + horizon]``.
* Customers whose first purchase is after the cutoff have no features and are
  excluded — they are not part of the prediction population.

This single-cutoff design yields one row per eligible customer, which is the
clean, standard setup for a propensity model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "recency_days",
    "frequency",
    "monetary",
    "tenure_days",
    "avg_order_value",
    "avg_interpurchase_days",
    "distinct_products",
    "avg_basket_size",
    "recency_over_tenure",
]
LABEL_COLUMN = "label"


def default_cutoff(transactions: pd.DataFrame, horizon_days: int = 90) -> pd.Timestamp:
    """Cutoff that leaves exactly ``horizon_days`` of data for the label window."""
    return transactions["InvoiceDate"].max() - pd.Timedelta(days=horizon_days)


def build_training_table(
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """Build the per-customer feature + label table for a given cutoff.

    Parameters
    ----------
    transactions:
        Cleaned transactions (``CustomerID``, ``InvoiceNo``, ``InvoiceDate``,
        ``StockCode``, ``TotalPrice``).
    cutoff:
        Snapshot date. Features use data ``<= cutoff``; the label window is
        ``(cutoff, cutoff + horizon_days]``.
    horizon_days:
        Length of the prediction window for the label.

    Returns
    -------
    DataFrame indexed by ``CustomerID`` with :data:`FEATURE_COLUMNS` plus the
    binary :data:`LABEL_COLUMN`.
    """
    tx = transactions.copy()
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])

    past = tx[tx["InvoiceDate"] <= cutoff]
    future = tx[
        (tx["InvoiceDate"] > cutoff)
        & (tx["InvoiceDate"] <= cutoff + pd.Timedelta(days=horizon_days))
    ]

    g = past.groupby("CustomerID")
    feat = g.agg(
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        line_items=("InvoiceNo", "size"),
        distinct_products=("StockCode", "nunique"),
    )

    feat["recency_days"] = (cutoff - feat["last_purchase"]).dt.days
    feat["tenure_days"] = (cutoff - feat["first_purchase"]).dt.days
    active_span = (feat["last_purchase"] - feat["first_purchase"]).dt.days
    feat["avg_order_value"] = feat["monetary"] / feat["frequency"]
    # typical gap between orders; for one-time buyers, how long since their
    # (only) purchase — a meaningful "no repeat yet" signal rather than 0.
    feat["avg_interpurchase_days"] = np.where(
        feat["frequency"] > 1,
        active_span / (feat["frequency"] - 1),
        feat["tenure_days"],
    )
    feat["avg_basket_size"] = feat["line_items"] / feat["frequency"]
    feat["recency_over_tenure"] = feat["recency_days"] / feat["tenure_days"].clip(lower=1)

    buyers_next = set(future["CustomerID"].unique())
    feat[LABEL_COLUMN] = feat.index.to_series().isin(buyers_next).astype(int)

    return feat[FEATURE_COLUMNS + [LABEL_COLUMN]]
