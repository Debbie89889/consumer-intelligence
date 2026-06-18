"""Unit tests for the purchase-propensity module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from consumer_intel.propensity import explain, model
from consumer_intel.propensity.features import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_training_table,
    default_cutoff,
)


# --- feature / label construction -----------------------------------------
@pytest.fixture
def tx() -> pd.DataFrame:
    """Transactions around a cutoff of 2024-04-01 (horizon 90d -> label window
    2024-04-01..2024-06-30).

    A: buys before cutoff AND after  -> in table, label 1
    B: buys before cutoff, NOT after -> in table, label 0
    C: buys ONLY after cutoff        -> excluded (no pre-cutoff features)
    """
    rows = [
        ("A", "i1", "2024-01-01", "P1", 50.0),
        ("A", "i2", "2024-02-01", "P2", 70.0),
        ("A", "i3", "2024-05-01", "P1", 30.0),  # after cutoff -> label
        ("B", "i4", "2024-01-15", "P1", 40.0),
        ("B", "i5", "2024-03-01", "P3", 60.0),
        ("C", "i6", "2024-05-10", "P2", 80.0),  # only after cutoff
    ]
    df = pd.DataFrame(
        rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "StockCode", "TotalPrice"]
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_default_cutoff(tx):
    # max date is 2024-05-10; horizon 90 -> cutoff 90 days earlier
    assert default_cutoff(tx, 90) == pd.Timestamp("2024-05-10") - pd.Timedelta(days=90)


def test_population_excludes_after_only_customers(tx):
    cutoff = pd.Timestamp("2024-04-01")
    table = build_training_table(tx, cutoff, horizon_days=90)
    assert set(table.index) == {"A", "B"}  # C excluded
    assert table.columns.tolist() == FEATURE_COLUMNS + [LABEL_COLUMN]


def test_labels_reflect_future_purchases(tx):
    cutoff = pd.Timestamp("2024-04-01")
    table = build_training_table(tx, cutoff, horizon_days=90)
    assert table.loc["A", LABEL_COLUMN] == 1  # bought again after cutoff
    assert table.loc["B", LABEL_COLUMN] == 0  # did not


def test_features_use_only_pre_cutoff_data(tx):
    cutoff = pd.Timestamp("2024-04-01")
    table = build_training_table(tx, cutoff, horizon_days=90)
    # A made 2 purchases before cutoff (i3 is after, must not count)
    assert table.loc["A", "frequency"] == 2
    assert table.loc["A", "monetary"] == pytest.approx(120.0)  # 50 + 70, not + 30
    # recency: last pre-cutoff purchase 2024-02-01 -> cutoff 2024-04-01 = 60 days
    assert table.loc["A", "recency_days"] == 60


def test_no_nan_features(tx):
    cutoff = pd.Timestamp("2024-04-01")
    table = build_training_table(tx, cutoff, horizon_days=90)
    assert not table[FEATURE_COLUMNS].isna().any().any()


# --- models ----------------------------------------------------------------
@pytest.fixture
def learnable_table() -> pd.DataFrame:
    """Synthetic feature table where recency strongly predicts the label, so
    any working model should clear AUC 0.5 comfortably."""
    rng = np.random.default_rng(0)
    n = 600
    recency = rng.integers(1, 400, n)
    # lower recency -> higher purchase probability
    prob = 1 / (1 + np.exp((recency - 150) / 50))
    label = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame(
        {
            "recency_days": recency,
            "frequency": rng.integers(1, 30, n),
            "monetary": rng.uniform(10, 5000, n),
            "tenure_days": rng.integers(30, 700, n),
            "avg_order_value": rng.uniform(10, 500, n),
            "avg_interpurchase_days": rng.uniform(5, 300, n),
            "distinct_products": rng.integers(1, 50, n),
            "avg_basket_size": rng.uniform(1, 20, n),
            "recency_over_tenure": rng.uniform(0, 1, n),
            LABEL_COLUMN: label,
        }
    )
    return df


def test_split_xy(learnable_table):
    X, y = model.split_xy(learnable_table)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == LABEL_COLUMN


def test_train_test_is_stratified(learnable_table):
    X, y = model.split_xy(learnable_table)
    X_tr, X_te, y_tr, y_te = model.train_test(X, y, test_size=0.25)
    assert len(X_te) == pytest.approx(len(X) * 0.25, abs=2)
    # stratified -> similar positive rate in both splits
    assert abs(y_tr.mean() - y_te.mean()) < 0.1


def test_models_beat_random(learnable_table):
    X, y = model.split_xy(learnable_table)
    X_tr, X_te, y_tr, y_te = model.train_test(X, y)
    for fit in (model.fit_logistic, model.fit_lightgbm):
        m = fit(X_tr, y_tr)
        metrics = model.evaluate(m, X_te, y_te)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["roc_auc"] > 0.6  # real signal present
        assert 0.0 <= metrics["brier"] <= 1.0


def test_calibration_table_shape(learnable_table):
    X, y = model.split_xy(learnable_table)
    X_tr, X_te, y_tr, y_te = model.train_test(X, y)
    m = model.fit_lightgbm(X_tr, y_tr)
    cal = model.calibration_table(m, X_te, y_te, n_bins=5)
    assert {"mean_predicted", "fraction_positive"}.issubset(cal.columns)
    assert len(cal) <= 5


# --- SHAP ------------------------------------------------------------------
def test_shap_importance(learnable_table):
    X, y = model.split_xy(learnable_table)
    X_tr, X_te, y_tr, y_te = model.train_test(X, y)
    m = model.fit_lightgbm(X_tr, y_tr)
    imp = explain.shap_importance(m, X_te)
    assert set(imp["feature"]) == set(FEATURE_COLUMNS)
    assert (imp["mean_abs_shap"] >= 0).all()
    # recency is the planted driver -> should rank highly
    assert imp.iloc[0]["feature"] in {"recency_days", "recency_over_tenure"}
