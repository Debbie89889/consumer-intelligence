"""Per-customer RFM features from the cleaned transactions table.

RFM is the backbone of both the rule-based segmentation and the K-means
clustering in :mod:`consumer_intel.segmentation`.

Definitions used here
---------------------
* **Recency**   days between a customer's last purchase and the *snapshot
  date*. Lower = more recently active.
* **Frequency** number of distinct invoices (purchase occasions), not line
  items. Buying 10 different SKUs in one order counts as one purchase.
* **Monetary**  total spend (sum of ``TotalPrice``) over the period.

The snapshot date defaults to one day after the last transaction, so the most
recent buyers get ``Recency == 1`` rather than 0 (avoids a log(0) downstream).
"""

from __future__ import annotations

import pandas as pd


def snapshot_date(transactions: pd.DataFrame) -> pd.Timestamp:
    """Reference 'today' for recency: one day after the last transaction."""
    return transactions["InvoiceDate"].max() + pd.Timedelta(days=1)


def compute_rfm(
    transactions: pd.DataFrame,
    snapshot: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate transactions into one RFM row per customer.

    Parameters
    ----------
    transactions:
        Cleaned transactions with ``CustomerID``, ``InvoiceNo``,
        ``InvoiceDate`` and ``TotalPrice`` columns.
    snapshot:
        Reference date for recency. Defaults to :func:`snapshot_date`.

    Returns
    -------
    DataFrame indexed by ``CustomerID`` with integer ``Recency`` (days),
    ``Frequency`` (distinct invoices) and float ``Monetary`` (total spend).
    """
    if snapshot is None:
        snapshot = snapshot_date(transactions)

    grouped = transactions.groupby("CustomerID")
    rfm = grouped.agg(
        last_purchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    rfm["Recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm = rfm.drop(columns="last_purchase")
    return rfm[["Recency", "Frequency", "Monetary"]]
