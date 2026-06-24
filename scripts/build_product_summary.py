"""Build a small product-level summary from the cleaned transactions.

Produces data/processed/product_summary.parquet — one row per product with
revenue, units sold, distinct orders and distinct customers. This is the
product-analysis input the API serves from (the full transaction file is too
large to ship to the cloud, so we precompute this compact summary).

    python scripts/build_product_summary.py
"""

from __future__ import annotations

import pandas as pd

from consumer_intel import config


def build() -> pd.DataFrame:
    tx = pd.read_parquet(config.PROCESSED_DIR / "transactions_clean.parquet")
    # Most frequent (non-null) description per stock code.
    desc = (
        tx.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    g = tx.groupby("StockCode").agg(
        revenue=("TotalPrice", "sum"),
        quantity=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    )
    out = g.join(desc).reset_index()
    out = out.rename(columns={"StockCode": "stock_code", "Description": "description"})
    out = out[["stock_code", "description", "revenue", "quantity", "orders", "customers"]]
    out = out.sort_values("revenue", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    out = build()
    path = config.PROCESSED_DIR / "product_summary.parquet"
    out.to_parquet(path, index=False)
    print(f"Wrote {len(out):,} products -> {path}")


if __name__ == "__main__":
    main()
