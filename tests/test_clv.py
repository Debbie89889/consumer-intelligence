"""Unit tests for the CLV module (historical, predictive, validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from consumer_intel.clv import historical, predictive, validate


@pytest.fixture
def small_tx() -> pd.DataFrame:
    """Hand-checkable transactions for the historical CLV test.

    A: invoices on 2024-01-01 (£100) and 2024-01-31 (£300) -> 2 orders,
       revenue 400, AOV 200, tenure 30 days.
    B: single invoice 2024-01-10 (£50) -> 1 order, tenure 0.
    """
    rows = [
        ("A", "1", "2024-01-01", 100.0),
        ("A", "2", "2024-01-31", 300.0),
        ("B", "3", "2024-01-10", 50.0),
    ]
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture
def synthetic_tx() -> pd.DataFrame:
    """Larger synthetic transaction log so lifetimes models can fit.

    200 customers, each with 1-12 invoices over ~1 year, positive amounts.
    """
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2023-01-01")
    rows = []
    inv = 0
    for c in range(200):
        n_orders = int(rng.integers(1, 13))
        days = np.sort(rng.integers(0, 360, n_orders))
        for d in days:
            inv += 1
            rows.append(
                (
                    f"C{c:04d}",
                    str(inv),
                    start + pd.Timedelta(days=int(d)),
                    float(rng.uniform(10, 500)),
                )
            )
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"])
    return df


# --- historical ------------------------------------------------------------
def test_historical_clv_values(small_tx):
    h = historical.historical_clv(small_tx)
    assert h.loc["A", "total_revenue"] == pytest.approx(400.0)
    assert h.loc["A", "n_orders"] == 2
    assert h.loc["A", "avg_order_value"] == pytest.approx(200.0)
    assert h.loc["A", "tenure_days"] == 30
    # B has tenure 0 -> annualised value uses a 1-day floor (no div-by-zero)
    assert h.loc["B", "tenure_days"] == 0
    assert np.isfinite(h.loc["B", "annual_value"])


# --- predictive ------------------------------------------------------------
def test_build_summary_columns(synthetic_tx):
    s = predictive.build_summary(synthetic_tx)
    assert set(["frequency", "recency", "T", "monetary_value"]).issubset(s.columns)
    # frequency counts repeat purchases, so it's strictly less than total invoices
    assert (s["frequency"] >= 0).all()


def test_eligible_mask_excludes_one_timers(synthetic_tx):
    s = predictive.build_summary(synthetic_tx)
    mask = predictive.eligible_mask(s)
    assert (s.loc[mask, "frequency"] > 0).all()
    assert (s.loc[mask, "monetary_value"] > 0).all()


def test_predict_outputs_are_sane(synthetic_tx):
    result = predictive.compute_predictive_clv(synthetic_tx, horizon_months=3, penalizer=0.1)
    preds = result.predictions
    assert len(preds) == result.summary.shape[0]
    # predictions non-negative
    assert (preds["predicted_purchases"] >= 0).all()
    assert preds["prob_alive"].between(0, 1).all()
    assert (preds["predicted_clv"] >= 0).all()
    # methods are tagged and limited to the two expected values
    assert set(preds["clv_method"].unique()).issubset({"bg_nbd_gamma_gamma", "fallback_pop_mean"})


def test_one_time_buyers_get_fallback(synthetic_tx):
    result = predictive.compute_predictive_clv(synthetic_tx, horizon_months=3, penalizer=0.1)
    s = result.summary
    preds = result.predictions
    one_timers = s.index[s["frequency"] == 0]
    if len(one_timers):
        assert (preds.loc[one_timers, "clv_method"] == "fallback_pop_mean").all()


def test_correlation_is_finite(synthetic_tx):
    s = predictive.build_summary(synthetic_tx)
    corr = predictive.frequency_monetary_correlation(s)
    assert np.isfinite(corr)


# --- validation ------------------------------------------------------------
def test_calibration_holdout_metrics(synthetic_tx):
    obs_end = synthetic_tx["InvoiceDate"].max()
    cal_end = obs_end - pd.Timedelta(days=90)
    val = validate.calibration_holdout(
        synthetic_tx, calibration_end=cal_end, observation_end=obs_end, penalizer=0.1
    )
    assert "predicted_purchases" in val.data.columns
    assert val.metrics["mae"] >= 0
    assert val.metrics["holdout_days"] == 90
    assert np.isfinite(val.metrics["correlation"])
