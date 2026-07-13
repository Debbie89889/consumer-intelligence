"""Load the phase 1-4 outputs into the database.

Consolidates the per-customer parquet outputs (RFM segments, CLV, propensity)
into a single ``customers`` table, and loads the association rules into a
``rules`` table. This is the "data engineering" step: many analytical outputs
land in a queryable store the API serves from.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from consumer_intel import config

CUSTOMERS_TABLE = "customers"
RULES_TABLE = "rules"
PRODUCTS_TABLE = "products"
MONTHLY_TABLE = "monthly"
COUNTRY_TABLE = "country"
CUSTOMER_TOP_PRODUCT_TABLE = "customer_top_product"


def build_customers_frame() -> pd.DataFrame:
    """Merge segment / CLV / propensity parquet outputs into one customer table."""
    seg = pd.read_parquet(config.PROCESSED_DIR / "customer_segments.parquet")
    clv = pd.read_parquet(config.PROCESSED_DIR / "customer_clv.parquet")
    prop = pd.read_parquet(config.PROCESSED_DIR / "propensity_scores.parquet")

    seg = seg[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "RFM_Score",
            "Segment",
            "Action",
            "ClusterName",
        ]
    ]
    clv = clv[
        [
            "CustomerID",
            "predicted_clv",
            "prob_alive",
            "predicted_purchases",
            "historical_clv",
            "clv_method",
        ]
    ]
    prop = prop[["CustomerID", "propensity"]]

    df = seg.merge(clv, on="CustomerID", how="left").merge(prop, on="CustomerID", how="left")
    df = df.rename(
        columns={
            "CustomerID": "customer_id",
            "Recency": "recency",
            "Frequency": "frequency",
            "Monetary": "monetary",
            "RFM_Score": "rfm_score",
            "Segment": "segment",
            "Action": "action",
            "ClusterName": "cluster_name",
        }
    )
    return df


def build_rules_frame() -> pd.DataFrame:
    """Association rules as a flat, query-friendly table (codes joined to text)."""
    rules = pd.read_parquet(config.PROCESSED_DIR / "association_rules.parquet")
    out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    out["antecedents_codes"] = rules["antecedents_codes"].apply(lambda x: ",".join(x))
    out["consequents_codes"] = rules["consequents_codes"].apply(lambda x: ",".join(x))
    return out


def build_products_frame() -> pd.DataFrame:
    """Per-product summary (revenue / units / orders / customers)."""
    return pd.read_parquet(config.PROCESSED_DIR / "product_summary.parquet")


def build_monthly_frame() -> pd.DataFrame:
    """Per-month summary (revenue / orders / customers) for the trend line."""
    return pd.read_parquet(config.PROCESSED_DIR / "monthly_summary.parquet")


def build_country_frame() -> pd.DataFrame:
    """Per-country summary (revenue / orders / customers)."""
    return pd.read_parquet(config.PROCESSED_DIR / "country_summary.parquet")


def build_customer_top_product_frame() -> pd.DataFrame:
    """Each customer's single highest-revenue product (drives per-customer NBO)."""
    return pd.read_parquet(config.PROCESSED_DIR / "customer_top_product_summary.parquet")


def load_all(engine: Engine) -> dict[str, int]:
    """Load all tables into the database (replacing existing). Returns row counts."""
    frames = {
        CUSTOMERS_TABLE: build_customers_frame(),
        RULES_TABLE: build_rules_frame(),
        PRODUCTS_TABLE: build_products_frame(),
        MONTHLY_TABLE: build_monthly_frame(),
        COUNTRY_TABLE: build_country_frame(),
        CUSTOMER_TOP_PRODUCT_TABLE: build_customer_top_product_frame(),
    }
    for table, frame in frames.items():
        frame.to_sql(table, engine, if_exists="replace", index=False)
    return {table: len(frame) for table, frame in frames.items()}
