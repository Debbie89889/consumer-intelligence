"""Shared test fixtures.

A tiny synthetic frame that exercises every cleaning rule, in the *raw* schema
(original column names) so tests cover ``standardize_columns`` too.
"""
# ruff: noqa: E501  -- tabular fixture rows are intentionally kept on one line each

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_keys(monkeypatch):
    """Never let a locally-exported OpenAI/Anthropic key leak into tests.

    ``generate_insight(backend="auto")`` calls a real, billed LLM whenever
    one of these is set — CLAUDE.md requires tests to stay offline and
    deterministic (template narration only). Without this, any dev/CI shell
    that happens to export a real key would silently start making network
    calls from the test suite.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """12 rows covering: valid sales, a duplicate, a cancellation, a return,
    an admin stock code, a zero price, and a missing customer id.

    Row guide:
      0,1 : identical valid UK sale (row 1 is an exact duplicate)
      2   : valid UK sale (5-digit + letter stock code)
      3   : valid EIRE sale
      4   : cancellation (invoice C..., negative qty)
      5   : return (negative qty, not a C invoice)
      6   : admin stock code POST
      7   : zero unit price
      8   : missing customer id
      9   : valid France sale
      10  : admin stock code M
      11  : valid France sale
    """
    rows = [
        (
            "489434",
            "85048",
            "GLASS BALL LIGHTS",
            12,
            "2009-12-01 07:45:00",
            6.95,
            "13085.0",
            "United Kingdom",
        ),
        (
            "489434",
            "85048",
            "GLASS BALL LIGHTS",
            12,
            "2009-12-01 07:45:00",
            6.95,
            "13085.0",
            "United Kingdom",
        ),
        (
            "489435",
            "79323P",
            "PINK CHERRY LIGHTS",
            5,
            "2009-12-01 08:00:00",
            6.75,
            "13085.0",
            "United Kingdom",
        ),
        ("489436", "22423", "REGENCY TEACUP", 3, "2009-12-02 10:00:00", 12.50, "14911.0", "EIRE"),
        ("C489437", "22423", "REGENCY TEACUP", -3, "2009-12-03 11:00:00", 12.50, "14911.0", "EIRE"),
        ("489438", "22423", "REGENCY TEACUP", -2, "2009-12-04 09:00:00", 12.50, "14911.0", "EIRE"),
        ("489439", "POST", "POSTAGE", 1, "2009-12-04 09:30:00", 18.00, "14911.0", "EIRE"),
        (
            "489440",
            "85048",
            "GLASS BALL LIGHTS",
            6,
            "2009-12-05 12:00:00",
            0.00,
            "13085.0",
            "United Kingdom",
        ),
        ("489441", "84997B", "CHILDRENS CUTLERY", 4, "2009-12-06 14:00:00", 4.15, None, "France"),
        (
            "489442",
            "84997B",
            "CHILDRENS CUTLERY",
            4,
            "2009-12-06 14:30:00",
            4.15,
            "12680.0",
            "France",
        ),
        ("489443", "M", "Manual", 1, "2009-12-07 10:00:00", 25.00, "12680.0", "France"),
        ("489444", "22423", "REGENCY TEACUP", 2, "2009-12-08 16:00:00", 12.50, "12680.0", "France"),
    ]
    cols = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["Customer ID"] = df["Customer ID"].astype("string")
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")
    return df


@pytest.fixture
def populated_engine(tmp_path):
    """A throwaway SQLite database with a few customers and rules loaded.

    Lets the db/api tests run without PostgreSQL or the real parquet outputs.
    """
    from consumer_intel.db.engine import make_engine

    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    customers = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "recency": 5,
                "frequency": 10,
                "monetary": 5000.0,
                "rfm_score": "555",
                "segment": "Champions",
                "action": "Reward loyalty.",
                "cluster_name": "High-Value Active",
                "predicted_clv": 1200.0,
                "prob_alive": 0.95,
                "predicted_purchases": 4.0,
                "historical_clv": 5000.0,
                "clv_method": "bg_nbd_gamma_gamma",
                "propensity": 0.8,
            },
            {
                "customer_id": "C2",
                "recency": 300,
                "frequency": 2,
                "monetary": 8000.0,
                "rfm_score": "155",
                "segment": "Can't Lose Them",
                "action": "Win them back.",
                "cluster_name": "High-Value Lapsing",
                "predicted_clv": 300.0,
                "prob_alive": 0.30,
                "predicted_purchases": 0.2,
                "historical_clv": 8000.0,
                "clv_method": "bg_nbd_gamma_gamma",
                "propensity": 0.1,
            },
            {
                "customer_id": "C3",
                "recency": 120,
                "frequency": 1,
                "monetary": 40.0,
                "rfm_score": "111",
                "segment": "Lost",
                "action": "Minimal spend.",
                "cluster_name": "Dormant Low-Value",
                "predicted_clv": 5.0,
                "prob_alive": 1.0,
                "predicted_purchases": 0.1,
                "historical_clv": 40.0,
                "clv_method": "fallback_pop_mean",
                "propensity": None,
            },
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "antecedents": "LUNCH BAG RED",
                "consequents": "LUNCH BAG PINK",
                "antecedents_codes": "20725",
                "consequents_codes": "22384",
                "support": 0.02,
                "confidence": 0.18,
                "lift": 9.9,
            },
            {
                "antecedents": "LUNCH BAG RED",
                "consequents": "LUNCH BAG BLACK",
                "antecedents_codes": "20725",
                "consequents_codes": "20727",
                "support": 0.018,
                "confidence": 0.17,
                "lift": 8.9,
            },
        ]
    )
    customers.to_sql("customers", engine, index=False, if_exists="replace")
    rules.to_sql("rules", engine, index=False, if_exists="replace")
    products = pd.DataFrame(
        [
            {
                "stock_code": "20725",
                "description": "LUNCH BAG RED",
                "revenue": 12000.0,
                "quantity": 5000,
                "orders": 800,
                "customers": 600,
            },
            {
                "stock_code": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "revenue": 24000.0,
                "quantity": 9000,
                "orders": 1200,
                "customers": 900,
            },
            {
                "stock_code": "22384",
                "description": "LUNCH BAG PINK",
                "revenue": 6000.0,
                "quantity": 3000,
                "orders": 500,
                "customers": 400,
            },
        ]
    )
    products.to_sql("products", engine, index=False, if_exists="replace")
    monthly = pd.DataFrame(
        [
            {"month": "2010-01", "revenue": 1000.0, "orders": 100, "customers": 80},
            {"month": "2010-02", "revenue": 1500.0, "orders": 120, "customers": 95},
            {"month": "2010-03", "revenue": 1200.0, "orders": 110, "customers": 90},
        ]
    )
    monthly.to_sql("monthly", engine, index=False, if_exists="replace")
    country = pd.DataFrame(
        [
            {"country": "United Kingdom", "revenue": 9000.0, "orders": 800, "customers": 700},
            {"country": "Germany", "revenue": 2000.0, "orders": 150, "customers": 120},
            {"country": "France", "revenue": 1500.0, "orders": 130, "customers": 100},
        ]
    )
    country.to_sql("country", engine, index=False, if_exists="replace")
    customer_top_product = pd.DataFrame(
        [
            {"customer_id": "C1", "stock_code": "20725", "revenue": 3000.0},
            {"customer_id": "C2", "stock_code": "99999", "revenue": 500.0},
        ]
    )
    customer_top_product.to_sql("customer_top_product", engine, index=False, if_exists="replace")

    from consumer_intel.db.models import Base

    Base.metadata.create_all(engine)  # conversations / messages / campaign_approvals
    return engine
