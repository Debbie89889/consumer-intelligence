"""Unit tests for the EDA summary functions."""

from __future__ import annotations

import pandas as pd
import pytest

from consumer_intel.data.clean import clean_transactions
from consumer_intel.eda import profile


@pytest.fixture
def clean_sample(raw_sample) -> pd.DataFrame:
    return clean_transactions(raw_sample).transactions


def test_dataset_overview_counts(clean_sample):
    ov = profile.dataset_overview(clean_sample)
    assert ov["n_line_items"] == len(clean_sample)
    assert ov["n_customers"] == clean_sample["CustomerID"].nunique()
    assert ov["total_revenue"] == pytest.approx(clean_sample["TotalPrice"].sum())


def test_monthly_summary_avg_order_value(clean_sample):
    monthly = profile.monthly_summary(clean_sample)
    # AOV must equal revenue / orders for every month.
    recomputed = monthly["revenue"] / monthly["orders"]
    assert (monthly["avg_order_value"] - recomputed).abs().max() < 1e-9


def test_country_summary_shares_sum_to_one(clean_sample):
    ctry = profile.country_summary(clean_sample)
    assert ctry["revenue_share"].sum() == pytest.approx(1.0)
    # sorted descending by revenue
    assert ctry["revenue"].is_monotonic_decreasing


def test_top_products_respects_n(clean_sample):
    top = profile.top_products(clean_sample, n=2)
    assert len(top) <= 2
    assert top["revenue"].is_monotonic_decreasing
