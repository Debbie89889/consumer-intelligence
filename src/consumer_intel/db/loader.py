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


def load_all(engine: Engine) -> dict[str, int]:
    """Load all tables into the database (replacing existing). Returns row counts."""
    customers = build_customers_frame()
    rules = build_rules_frame()
    products = build_products_frame()
    customers.to_sql(CUSTOMERS_TABLE, engine, if_exists="replace", index=False)
    rules.to_sql(RULES_TABLE, engine, if_exists="replace", index=False)
    products.to_sql(PRODUCTS_TABLE, engine, if_exists="replace", index=False)
    return {
        CUSTOMERS_TABLE: len(customers),
        RULES_TABLE: len(rules),
        PRODUCTS_TABLE: len(products),
    }
