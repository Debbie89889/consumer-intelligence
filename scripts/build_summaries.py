"""Build the compact serving summaries from the cleaned transactions.

Produces small parquet files the API serves from (the full transaction file
is too large to ship to the cloud, so we precompute these):

* product_summary.parquet             — per product: revenue / units / orders / customers
* monthly_summary.parquet             — per month:   revenue / orders / customers
* country_summary.parquet             — per country: revenue / orders / customers
* customer_top_product_summary.parquet — per customer: their single highest-revenue
  product, used by the Copilot graph's per-customer Next Best Offer lookup

    python scripts/build_summaries.py
"""

from __future__ import annotations

import pandas as pd

from consumer_intel import config


def build_products(tx: pd.DataFrame) -> pd.DataFrame:
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
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)


def build_monthly(tx: pd.DataFrame) -> pd.DataFrame:
    m = tx.assign(month=tx["InvoiceDate"].dt.to_period("M").astype(str))
    out = (
        m.groupby("month")
        .agg(
            revenue=("TotalPrice", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .reset_index()
    )
    return out.sort_values("month").reset_index(drop=True)


def build_customer_top_product(tx: pd.DataFrame) -> pd.DataFrame:
    """Each customer's single highest-revenue product.

    Drives the Copilot graph's per-customer Next Best Offer lookup: the
    ``rules`` table is keyed by a single antecedent product, so a customer-
    level recommendation needs one representative product per customer.
    Ties broken deterministically by ``StockCode`` (via ``idxmax`` on a
    groupby-sorted frame).
    """
    g = (
        tx.dropna(subset=["CustomerID"])
        .groupby(["CustomerID", "StockCode"])["TotalPrice"]
        .sum()
        .reset_index()
    )
    top = g.loc[g.groupby("CustomerID")["TotalPrice"].idxmax()].reset_index(drop=True)
    top = top.rename(
        columns={"CustomerID": "customer_id", "StockCode": "stock_code", "TotalPrice": "revenue"}
    )
    return top.sort_values("customer_id").reset_index(drop=True)


def build_country(tx: pd.DataFrame) -> pd.DataFrame:
    out = (
        tx.groupby("Country")
        .agg(
            revenue=("TotalPrice", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .reset_index()
    )
    out = out.rename(columns={"Country": "country"})
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)


def main() -> None:
    tx = pd.read_parquet(config.PROCESSED_DIR / "transactions_clean.parquet")
    for name, frame in {
        "product_summary": build_products(tx),
        "monthly_summary": build_monthly(tx),
        "country_summary": build_country(tx),
        "customer_top_product_summary": build_customer_top_product(tx),
    }.items():
        path = config.PROCESSED_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        print(f"  {name}: {len(frame):,} rows -> {path.name}")


if __name__ == "__main__":
    main()
