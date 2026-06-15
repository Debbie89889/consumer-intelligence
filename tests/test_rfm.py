"""Unit tests for RFM feature computation."""

from __future__ import annotations

import pandas as pd
import pytest

from consumer_intel.features.rfm import compute_rfm, snapshot_date


@pytest.fixture
def tx() -> pd.DataFrame:
    """Three customers with known, hand-checkable RFM values.

    Last transaction overall is 2024-01-31, so snapshot = 2024-02-01.
      A: 2 invoices (1001, 1002), last 2024-01-30 -> Recency 2,
         spend 10*2 + 5*4 = 40
      B: 1 invoice over two line items, last 2024-01-20 -> Recency 12,
         spend 3*7 + 2*7 = 35  (Frequency 1, not 2)
      C: 1 invoice, last 2024-01-31 -> Recency 1, spend 100
    """
    rows = [
        ("A", "1001", "2024-01-10", 10.0),
        ("A", "1002", "2024-01-30", 20.0),
        ("B", "2001", "2024-01-20", 21.0),
        ("B", "2001", "2024-01-20", 14.0),
        ("C", "3001", "2024-01-31", 100.0),
    ]
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_snapshot_is_day_after_last(tx):
    assert snapshot_date(tx) == pd.Timestamp("2024-02-01")


def test_recency_frequency_monetary(tx):
    rfm = compute_rfm(tx)
    assert rfm.loc["A", "Recency"] == 2
    assert rfm.loc["A", "Frequency"] == 2
    assert rfm.loc["A", "Monetary"] == pytest.approx(30.0)

    # B's two line items share one invoice -> Frequency 1
    assert rfm.loc["B", "Frequency"] == 1
    assert rfm.loc["B", "Recency"] == 12
    assert rfm.loc["B", "Monetary"] == pytest.approx(35.0)

    assert rfm.loc["C", "Recency"] == 1
    assert rfm.loc["C", "Monetary"] == pytest.approx(100.0)


def test_columns_and_one_row_per_customer(tx):
    rfm = compute_rfm(tx)
    assert list(rfm.columns) == ["Recency", "Frequency", "Monetary"]
    assert rfm.index.is_unique
    assert len(rfm) == 3
