"""Tests for scripts/build_summaries.py — build_customer_top_product.

Only the newly added summary is covered here; the other three summaries
(product/monthly/country) predate this test file and have no direct tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_summaries import build_customer_top_product  # noqa: E402


@pytest.fixture
def tx() -> pd.DataFrame:
    """Two customers: C1 has a clear top product; C2 has a tie broken by StockCode."""
    return pd.DataFrame(
        [
            {"CustomerID": "C1", "StockCode": "AAA", "TotalPrice": 10.0},
            {"CustomerID": "C1", "StockCode": "BBB", "TotalPrice": 50.0},
            {"CustomerID": "C1", "StockCode": "BBB", "TotalPrice": 5.0},
            {"CustomerID": "C2", "StockCode": "ZZZ", "TotalPrice": 20.0},
            {"CustomerID": "C2", "StockCode": "YYY", "TotalPrice": 20.0},
            {"CustomerID": None, "StockCode": "AAA", "TotalPrice": 999.0},
        ]
    )


def test_picks_highest_revenue_product_per_customer(tx):
    out = build_customer_top_product(tx)
    row = out[out["customer_id"] == "C1"].iloc[0]
    assert row["stock_code"] == "BBB"
    assert row["revenue"] == 55.0  # 50.0 + 5.0, summed across repeat purchases


def test_ties_broken_deterministically_by_stock_code(tx):
    out = build_customer_top_product(tx)
    row = out[out["customer_id"] == "C2"].iloc[0]
    assert row["stock_code"] == "YYY"  # "YYY" < "ZZZ"


def test_drops_rows_with_missing_customer_id(tx):
    out = build_customer_top_product(tx)
    assert out["customer_id"].isna().sum() == 0
    assert set(out["customer_id"]) == {"C1", "C2"}


def test_one_row_per_customer(tx):
    out = build_customer_top_product(tx)
    assert out["customer_id"].is_unique
